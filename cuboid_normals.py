"""Normals + recursive cuboids on a single photo.

The two cues fail in complementary ways (NOTES sections 11-12):
  * integrating normals gives each smooth part's shape very accurately (0.5 %), but normals
    carry no information across a depth discontinuity, so separate parts cannot be placed;
  * the recursive cuboids place every segment metrically (seeds on the main box, contacts,
    continuity), but each segment is a solid box, so its surface shape is wrong.

Combination: every superpixel gets its SHAPE from the normals and its PLACE from the cuboids.
  1. Normals are integrated inside each segment only (no equation crosses a segment boundary),
     giving a log-depth field x(p) per segment, known up to one offset c_s: the same single
     unknown per segment as a cuboid's scale (cuboids_global.py).
  2. The offsets are solved jointly by robust least squares from
       - the cuboid prior: c_s puts the segment's surface at its cuboid's front depth
         (seed segments, which sit on the main box, weigh more);
       - continuity: across a shared boundary, the two surfaces meet, unless the pair is an
         occlusion boundary, which the robust (Cauchy, IRLS) weights detect per pair.

Normal sources tested (no learned estimator is reachable from this environment):
  * exact normals of the true surface (upper bound);
  * exact + 10 deg noise (about a learned estimator's accuracy);
  * image only: box.JPG - the axis of each segment's plane from the planar-patch method
    (vanishing-point orientation); pig - silhouette normals (exact on the outline, smooth inside).

    python cuboid_normals.py --eval-dir out/segeval --out docs/segments
"""
import argparse
import json
import time

import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

import cuboids_global as cg
import lift3d
import normal_integration as ni
import segment3d as s3
import synthetic3d


# ---------------------------------------------------------------------------
# The combination
# ---------------------------------------------------------------------------
def gradients(cam, uv, n, step):
    """Predicted log-depth differences to the next sample in u and v (normal_integration.py)."""
    f, cx, cy = cam.K[0, 0], cam.K[0, 2], cam.K[1, 2]
    den = n[:, 0] * (uv[:, 0] - cx) + n[:, 1] * (uv[:, 1] - cy) + n[:, 2] * f
    with np.errstate(divide="ignore", invalid="ignore"):
        return -n[:, 0] / den * step, -n[:, 1] / den * step


def neighbour_pairs(uv):
    """(k, m, axis) for 4-neighbour sample pairs on the regular grid."""
    lut, ij, step = ni.grid_of(uv)
    out = []
    for k, (i, j) in enumerate(ij):
        for axis, (di, dj) in enumerate(((1, 0), (0, 1))):
            m = lut.get((i + di, j + dj))
            if m is not None:
                out.append((k, m, axis))
    return np.array(out, int).reshape(-1, 3), step


def integrate_within_segments(cam, uv, n, seg, max_grad=0.5):
    """Log-depth per sample from the normals, integrated inside each segment only; each
    segment's field has median 0 (its offset is unknown)."""
    pairs, step = neighbour_pairs(uv)
    gu, gv = gradients(cam, uv, n, step)
    g = np.where(pairs[:, 2] == 0, 0.5 * (gu[pairs[:, 0]] + gu[pairs[:, 1]]), 0.5 * (gv[pairs[:, 0]] + gv[pairs[:, 1]]))
    same = seg[pairs[:, 0]] == seg[pairs[:, 1]]
    ok = same & np.isfinite(g) & (np.abs(g) < max_grad)
    p, g = pairs[ok], g[ok]
    r = np.arange(len(p))
    A = coo_matrix((np.r_[np.ones(len(p)), -np.ones(len(p))], (np.r_[r, r], np.r_[p[:, 1], p[:, 0]])), shape=(len(p), len(uv))).tocsr()
    x = spsolve((A.T @ A + diags(np.full(len(uv), 1e-6))).tocsc(), A.T @ g)
    for s in np.unique(seg):
        sel = seg == s
        x[sel] -= np.median(x[sel])
    return x


def place_segments(cam, uv, n, seg, x, box_depth, seeds=(), w_box=1.0, w_seed=10.0, w_cont=1.0,
                   continuity=True, robust=True, sigma=0.01, iters=15, parts=False, keep=0.5):
    """Offsets c_s (log depth) from cuboid priors and robust boundary continuity.

    parts=True: two stages. The robust solve only decides which boundaries are continuous
    (link weight >= keep); the segments joined by those links form parts. Inside a part the
    offsets come from continuity alone (shape purely from normals); each part gets one offset
    from the cuboids: the median over its seed segments (on the main box) when it has any,
    else over all its segments."""
    ids = np.unique(seg)
    idx = {s: i for i, s in enumerate(ids)}
    rows, rhs, w0 = [], [], []
    for s in ids:
        sel = (seg == s) & np.isfinite(box_depth)
        if sel.sum() >= 3:
            rows.append({idx[s]: 1.0})
            rhs.append(float(np.median(np.log(box_depth[sel]) - x[sel])))
            w0.append(w_seed if s in seeds else w_box)
    n_prior = len(rows)
    if continuity:
        pairs, step = neighbour_pairs(uv)
        gu, gv = gradients(cam, uv, n, step)
        g = np.where(pairs[:, 2] == 0, 0.5 * (gu[pairs[:, 0]] + gu[pairs[:, 1]]), 0.5 * (gv[pairs[:, 0]] + gv[pairs[:, 1]]))
        a, b = seg[pairs[:, 0]], seg[pairs[:, 1]]
        cross = (a != b) & np.isfinite(g) & (np.abs(g) < 0.5)
        # c_b - c_a = x_a(k) - x_b(m) + g   (log depth continues smoothly across the boundary)
        delta = x[pairs[:, 0]] - x[pairs[:, 1]] + g
        keys = np.c_[np.minimum(a, b), np.maximum(a, b)][cross]
        d = np.where(a < b, delta, -delta)[cross]                 # always c_hi - c_lo
        for key in np.unique(keys, axis=0):
            sel = np.all(keys == key, axis=1)
            rows.append({idx[key[1]]: 1.0, idx[key[0]]: -1.0})
            rhs.append(float(np.median(d[sel])))
            w0.append(w_cont * min(1.0, sel.sum() / 5))
    A = np.zeros((len(rows), len(ids)))
    for k, r in enumerate(rows):
        for c, v in r.items():
            A[k, c] = v
    b, w0 = np.asarray(rhs), np.asarray(w0)
    w = w0.copy()
    for _ in range(iters if robust else 1):
        sw = np.sqrt(w)
        c = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
        res = A @ c - b
        w = w0.copy()
        w[n_prior:] = w0[n_prior:] / (1 + (res[n_prior:] / sigma) ** 2)    # only continuity rows are robust
    if parts and continuity:
        from scipy.sparse.csgraph import connected_components
        link = [k for k in range(n_prior, len(rows)) if w[k] >= keep * w0[k]]
        ij = np.array([[c for c in rows[k]] for k in link]).reshape(-1, 2)
        G = coo_matrix((np.ones(len(ij)), (ij[:, 0], ij[:, 1])), shape=(len(ids), len(ids)))
        _, part = connected_components(G, directed=False)
        # continuity-only offsets inside each part (gauge: 0 at one segment per part)
        Al = np.zeros((len(link) + part.max() + 1, len(ids)))
        bl = np.zeros(len(Al))
        for r, k in enumerate(link):
            for col, v in rows[k].items():
                Al[r, col] = v
            bl[r] = b[k]
        for q in range(part.max() + 1):
            Al[len(link) + q, np.nonzero(part == q)[0][0]] = 1.0
        c = np.linalg.lstsq(Al, bl, rcond=None)[0]
        # one offset per part from the cuboid priors (seeds first)
        prior_of = {next(iter(rows[k])): (b[k], w0[k] >= w_seed) for k in range(n_prior)}
        for q in range(part.max() + 1):
            members = np.nonzero(part == q)[0]
            vals = [(prior_of[m][0] - c[m], prior_of[m][1]) for m in members if m in prior_of]
            if not vals:
                continue
            seed_vals = [v for v, is_seed in vals if is_seed]
            c[members] += np.median(seed_vals if seed_vals else [v for v, _ in vals])
    return np.exp(x + c[np.array([idx[s] for s in seg])])


def combine(cam, image, mask, vps, uv, n, labels=None, cuboids=None, **kw):
    """Depth at samples uv: shape from normals n, placement from the global recursive cuboids."""
    s3.SIGNS = s3.axis_signs(cam, vps)
    if labels is None:
        labels, _ = s3.superpixels(image, mask)
    if cuboids is None:
        cuboids = cg.global_boxes(cam, image, mask, vps, labels=labels, constraints=("contact", "continuity"), log=lambda *a: None)
    seg = labels[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    ok = seg > 0
    z = np.full(len(uv), np.nan)
    box_depth = s3.depth_from_boxes(cam, uv[ok], cg.boxes_of(cuboids))
    x = integrate_within_segments(cam, uv[ok], n[ok], seg[ok])
    z[ok] = place_segments(cam, uv[ok], n[ok], seg[ok], x, box_depth, seeds=set(cuboids["seeds"]), **kw)
    return z, cuboids


# ---------------------------------------------------------------------------
# Normal sources
# ---------------------------------------------------------------------------
def box_exact_normals(cam, uv, corners):
    """Outward face normals (camera frame) of an oriented box at each pixel's ray hit."""
    d = cam.ray(uv)
    o = corners[0]
    axes = [corners[3] - o, corners[1] - o, corners[4] - o]
    lens = np.array([np.linalg.norm(a) for a in axes])
    U = np.array([a / L for a, L in zip(axes, lens)])
    z = s3.depth_from_boxes(cam, uv, [corners])
    P = cam.C + d * (z / (d @ cam.R[2]))[:, None]
    q = (P - o) @ U.T
    dist = np.minimum(np.abs(q), np.abs(q - lens))
    axis = np.argmin(dist, axis=1)
    sign = np.where(np.abs(q[np.arange(len(q)), axis]) < np.abs(q[np.arange(len(q)), axis] - lens[axis]), -1.0, 1.0)
    nw = U[axis] * sign[:, None]
    return nw @ cam.R.T


def plane_normals(cam, uv, labels, planes):
    """Image-only normals for flat-faced objects: each segment's plane axis (planar patches,
    i.e. vanishing-point orientation), facing the camera."""
    seg = labels[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    nw = np.zeros((len(uv), 3))
    for s, (axis, _) in planes.items():
        nw[seg == s, axis] = 1.0
    nc = nw @ cam.R.T
    d = cam.ray(uv) @ cam.R.T
    nc *= -np.sign(np.sum(nc * d, axis=1, keepdims=True) + 1e-12)
    nc[~nw.any(1)] = np.nan
    return nc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(name, cam, img, mask, vps, uv, z_gt, sources, out_rows):
    labels, contour = s3.superpixels(img, mask)
    cub = cg.global_boxes(cam, img, mask, vps, labels=labels, constraints=("contact", "continuity"), log=lambda *a: None)
    rows = {}

    def report(label, z):
        e = s3.depth_errors(z, z_gt)
        rows[label] = e
        print(f"{name:4s} {label:66s} median {100*e['median_rel']:.2f}%  <=3% {100*e['within_3pct']:.0f}%  "
              f"<=1% {100*e['within_1pct']:.0f}%", flush=True)

    report("recursive cuboids, global solve (no normals)", s3.depth_from_boxes(cam, uv, cg.boxes_of(cub)))
    a_idx, a_z = ni.ground_anchor(cam, uv)
    for src, n in sources(labels, contour).items():
        ok = np.isfinite(n).all(1)
        n = np.where(ok[:, None], n, np.array([0.0, 0.0, -1.0]) @ np.eye(3))
        report(f"{src}: normals alone (whole image, ground-anchored)", ni.integrate_normals(cam, uv, n, a_idx, a_z, robust=False))
        report(f"{src}: normals in segments + cuboid placement", combine(cam, img, mask, vps, uv, n, labels, cub, continuity=False)[0])
        report(f"{src}: normals in segments + cuboid placement + robust continuity", combine(cam, img, mask, vps, uv, n, labels, cub)[0])
    out_rows[name] = rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default="docs/segments")
    a = ap.parse_args()
    E = a.eval_dir.rstrip("/") + "/"
    t0 = time.time()
    cam = lift3d.default_camera()
    vps = lift3d.vanishing_points()
    s3.SIGNS = s3.axis_signs(cam, vps)
    res = {}

    # box.JPG: ground truth = the hand-traced box
    img = cv2.imread("box.JPG")
    mask = lift3d.object_mask("reference.JPG", "box.JPG")
    lab0, _ = s3.superpixels(img, mask)
    ys, xs = np.nonzero(lab0[::4, ::4] > 0)
    uv = np.c_[xs * 4, ys * 4].astype(float)
    corners = lift3d.box_ground_truth(cam)["corners"]
    z_gt = s3.depth_from_boxes(cam, uv, [corners])
    n_true = box_exact_normals(cam, uv, corners)

    def box_sources(labels, contour):
        p = s3.patch_reconstruct(cam, img, mask, vps, labels, contour)
        return {"exact normals": n_true, "exact + 10 deg noise": ni.perturb(n_true, 10),
                "image only (planar-patch orientation)": plane_normals(cam, uv, labels, p["planes"])}
    evaluate("box", cam, img, mask, vps, uv, z_gt, box_sources, res)

    # synthetic pig: ground truth = the rendered shape (surface hits refined by bisection)
    img = cv2.imread(E + "synth_pig.png")
    mask = np.load(E + "synth_pig_mask.npy")
    g = np.load(E + "synth_pig_gt.npz")
    ok = np.isfinite(g["z"])
    uv = g["uv"][ok]
    inside, _ = synthetic3d.make_pig()
    z_gt = ni.refine_hits(cam, uv, g["z"][ok], inside)
    n_true = ni.exact_normals(cam, uv, z_gt, inside)

    def pig_sources(labels, contour):
        return {"exact normals": n_true, "exact + 10 deg noise": ni.perturb(n_true, 10),
                "image only (silhouette normals)": ni.silhouette_normals(cam, mask, uv)}
    evaluate("pig", cam, img, mask, vps, uv, z_gt, pig_sources, res)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(f"{a.out.rstrip('/')}/cuboid_normals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
