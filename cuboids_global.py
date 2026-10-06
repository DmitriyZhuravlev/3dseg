"""Improved recursive cuboids: one global solve instead of greedy propagation.

The observation behind it: with a calibrated camera, the VP-tangent 2D box of a segment
fixes its 3D box up to ONE number. All lifts of the same 2D box are homothetic about the
camera centre C: box(s) = C + s * (B - C) for a canonical lift B. Projection and axis
alignment are both unchanged by that scaling. So the recursive method's per-segment depth
is a single scalar s_i, and every constraint used by the recursion is linear in it:

  * contact:    s_i's face on plane x_a = p touches s_j's opposite face
                ->  C_a + s_i (f_i - C_a) = C_a + s_j (g_j - C_a)
  * continuity: at a shared boundary pixel the two front surfaces have the same depth
                ->  s_i d_i(p) = s_j d_j(p)   (camera depth scales with s as well)
  * seed:       a seed segment's box lies on a face of the main box  ->  s_i = s_seed
  * prior:      (optional) a whole-object model's depth over the segment ->  s_i d_i = z_model

The original propagates these one neighbour at a time (breadth-first), so an error made
early is copied down every chain ("drift"). Here all of them are solved jointly by robust
(IRLS, Huber) linear least squares, with each scale bounded so its box stays inside the
main box. Wrong contacts and true occlusion boundaries become down-weighted outliers
instead of seeds of new errors.

    python cuboids_global.py --eval-dir out/segeval      # variants vs ground truth (box, pig)
"""
import argparse
import json
import time

import cv2
import numpy as np
from scipy.optimize import lsq_linear

import lift3d
import segment3d as s3


# ---------------------------------------------------------------------------
# Canonical lifts and the scale family
# ---------------------------------------------------------------------------
def canonical_box(cam, corners2d, ref_value):
    """One metric lift of a 2D tangent box: (origin, size, rms), bottom face on z = ref_value.
    Falls back to the other faces' planes when the bottom ray misses the plane."""
    best = None
    for axis, side, value in ((2, 0, ref_value), (2, 1, ref_value + 0.1), (0, 0, cam.C[0] + 1.0), (1, 0, cam.C[1] + 1.0)):
        try:
            o, sz, rms = s3.lift_tangent_box(cam, corners2d, axis, side, value)
        except Exception:
            continue
        _, depth = cam.project(s3.box_corners(o, sz))
        if (depth > 0).all() and (best is None or rms < best[2]):
            best = (o, sz, rms)
    return best


def scaled(cam, o, sz, s):
    """The member of the homothety family with scale s."""
    return cam.C + s * (o - cam.C), s * sz


def scale_of(cam, o_ref, o):
    """Scale that maps the canonical origin o_ref onto origin o (same ray)."""
    return float(np.linalg.norm(o - cam.C) / max(np.linalg.norm(o_ref - cam.C), 1e-12))


def scale_bounds(cam, o, sz, lo, hi, tol):
    """Interval of s for which the scaled box stays inside [lo - tol, hi + tol]."""
    blo, bhi = s3.box_extent(o, sz)
    smin, smax = 1e-3, 1e3
    for a in range(3):
        for b, bound, upper in ((blo[a], lo[a] - tol[a], False), (bhi[a], hi[a] + tol[a], True)):
            # coordinate(s) = C_a + s * (b - C_a); need >= bound (lower) or <= bound (upper)
            k = b - cam.C[a]
            if abs(k) < 1e-12:
                continue
            lim = (bound - cam.C[a]) / k
            if (k > 0) == upper:
                smax = min(smax, lim)
            else:
                smin = max(smin, lim)
    return smin, smax


def front_depth(cam, uv, o, sz):
    """Camera depth of the nearest hit of each pixel ray with the box (nan where it misses)."""
    return s3.depth_from_boxes(cam, uv, [s3.box_corners(o, sz)])


# ---------------------------------------------------------------------------
# The global solve
# ---------------------------------------------------------------------------
def main_box_of(cam, image, mask, refine=True):
    main, _ = lift3d.fit_bounding_box(cam, mask, fit_yaw=False)
    if refine:
        main = lift3d.refine_box_with_edges(cam, image, mask, main)
        main[5] = 0.0
    cx, cy, ln, wd, ht, _ = main
    return np.array([cx - ln / 2, cy - wd / 2, 0.0]), np.array([cx + ln / 2, cy + wd / 2, ht])


def global_boxes(cam, image, mask, vps, labels=None, region_size=160, ruler=120, main_box=None, refine_main=True,
                 constraints=("continuity",), seed_weight=10.0, bounds=True, robust=True, huber=0.02, iters=6,
                 prior_depth=None, prior_weight=0.3, samples_per_pair=40, max_rms=6.0, log=print):
    """Recursive-cuboid constraints solved jointly. Returns dict like segment3d.reconstruct.

    constraints: any of "continuity" (depth equality on shared boundaries) and "contact" (the
    original rule: the face pair with the largest 2D overlap is coplanar).
    prior_depth: optional callable uv -> camera depth of a whole-object model (soft prior).
    """
    s3.SIGNS = s3.axis_signs(cam, vps)
    if labels is None:
        labels, _ = s3.superpixels(image, mask, region_size, ruler)
    lo, hi = (np.asarray(v, float) for v in main_box) if main_box is not None else main_box_of(cam, image, mask, refine_main)
    tol = 0.05 * (hi - lo)
    ids = [int(v) for v in np.unique(labels) if v > 0]
    canon = {}
    for i in ids:
        c2 = s3.tangent_box_2d(labels == i, vps)
        if c2 is None:
            continue
        cb = canonical_box(cam, c2, lo[2])
        if cb is not None and cb[2] <= max_rms:
            canon[i] = (cb[0], cb[1], c2)
    idx = {i: k for k, i in enumerate(canon)}
    n = len(idx)
    if n == 0:
        return dict(cuboids=[], labels=labels, n_segments=len(ids), seeds=[], main=(lo, hi))
    rows, rhs, wts, kinds = [], [], [], []

    def add(coef, b, w, kind):
        rows.append(coef)
        rhs.append(b)
        wts.append(w)
        kinds.append(kind)

    # seeds: as in the original, the segments at the extreme points on their best main-box face
    seeds = sorted({s3.segment_at(labels, p) for p in s3.extreme_points(mask, vps)} - {0})
    for s in seeds:
        if s not in canon:
            continue
        o, sz, c2 = canon[s]
        best = None
        for axis in range(3):
            for side in (0, 1):
                for value in (lo[axis], hi[axis]):
                    try:
                        so, ssz, rms = s3.lift_tangent_box(cam, c2, axis, side, value)
                    except Exception:
                        continue
                    blo, bhi = s3.box_extent(so, ssz)
                    score = rms + (0 if np.all(blo >= lo - tol) and np.all(bhi <= hi + tol) else 1e3)
                    if best is None or score < best[0]:
                        best = (score, so)
        if best is not None and best[0] < 1e3:
            add({idx[s]: 1.0}, scale_of(cam, o, best[1]), seed_weight, "seed")

    bnd = s3.pair_boundaries(labels)
    rng = np.random.default_rng(0)
    for (i, j), pts in bnd.items():
        if i not in canon or j not in canon:
            continue
        if "continuity" in constraints:
            if len(pts) > samples_per_pair:
                pts = pts[rng.choice(len(pts), samples_per_pair, replace=False)]
            di = front_depth(cam, pts, canon[i][0], canon[i][1])
            dj = front_depth(cam, pts, canon[j][0], canon[j][1])
            ok = np.isfinite(di) & np.isfinite(dj)
            if ok.sum() >= 3:
                # one row per pair: median depths (robust to boundary pixels that just miss a box)
                a, b = float(np.median(di[ok])), float(np.median(dj[ok]))
                m = (a + b) / 2
                add({idx[i]: a / m, idx[j]: -b / m}, 0.0, min(1.0, ok.sum() / 10), "continuity")
        if "contact" in constraints:
            (oi, szi, c2i), (oj, szj, _) = canon[i], canon[j]
            best = None
            for axis in range(3):
                for side in (0, 1):
                    nq = s3.face_quad(cam, oj, szj, axis, 1 - side)
                    sq = c2i[s3.FACES[(axis, side)]]
                    ov = s3.quad_iou(sq, nq, mask.shape)
                    if ov > 0 and (best is None or ov > best[0]):
                        best = (ov, axis, side)
            if best is not None:
                ov, axis, side = best
                fi = oi[axis] + side * szi[axis] - cam.C[axis]           # s_i's face, relative to C
                gj = oj[axis] + (1 - side) * szj[axis] - cam.C[axis]     # s_j's touching face
                m = (abs(fi) + abs(gj)) / 2
                add({idx[i]: fi / m, idx[j]: -gj / m}, 0.0, ov, "contact")

    if prior_depth is not None:
        pix = s3.segment_pixels(labels, list(canon))
        for i, (o, sz, _) in canon.items():
            uv = pix[i]
            d, z = front_depth(cam, uv, o, sz), prior_depth(uv)
            ok = np.isfinite(d) & np.isfinite(z)
            if ok.sum() >= 5:
                r = float(np.median(z[ok] / d[ok]))
                add({idx[i]: 1.0}, r, prior_weight, "prior")

    A = np.zeros((len(rows), n))
    for k, coef in enumerate(rows):
        for c, v in coef.items():
            A[k, c] = v
    b = np.asarray(rhs, float)
    w0 = np.asarray(wts, float)
    lb = np.full(n, 1e-3)
    ub = np.full(n, 1e3)
    if bounds:
        for i, k in idx.items():
            smin, smax = scale_bounds(cam, canon[i][0], canon[i][1], lo, hi, tol)
            if smin < smax:
                lb[k], ub[k] = smin, smax
    # unconstrained segments (no rows) default to the median seed scale
    w = w0.copy()
    s = None
    for _ in range(iters if robust else 1):
        sw = np.sqrt(w)[:, None]
        res = lsq_linear(A * sw, b * sw[:, 0], bounds=(lb, ub), lsmr_tol="auto")
        s = res.x
        if not robust:
            break
        r = np.abs(A @ s - b)
        w = w0 * np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-12))
        w[np.array(kinds) == "seed"] = w0[np.array(kinds) == "seed"]
    cuboids = []
    for i, k in idx.items():
        o, sz = scaled(cam, canon[i][0], canon[i][1], s[k])
        box = s3.box_corners(o, sz)
        cuboids.append({"bottom": box[:4].tolist(), "height": float(abs(sz[2])), "segment": int(i), "scale": float(s[k])})
    kinds = np.array(kinds)
    log(f"{len(ids)} segments, {n} lifted; rows: " + ", ".join(f"{k} {int((kinds == k).sum())}" for k in np.unique(kinds)))
    return dict(cuboids=cuboids, labels=labels, n_segments=len(ids), seeds=seeds, main=(lo, hi))


def boxes_of(r):
    return [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in r["cuboids"]]


# ---------------------------------------------------------------------------
# Evaluation of the variants (same ground truth as segment3d_eval.py)
# ---------------------------------------------------------------------------
def model_depth_fn(cam, cuboids):
    boxes = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in cuboids]
    return lambda uv: s3.depth_from_boxes(cam, uv, boxes)


VARIANTS = [
    ("original recursive boxes (greedy propagation)", None),
    ("global: contact constraints only", dict(constraints=("contact",))),
    ("global: continuity constraints only", dict(constraints=("continuity",))),
    ("global: contact + continuity", dict(constraints=("contact", "continuity"))),
    ("global: contact + continuity, no robust loss", dict(constraints=("contact", "continuity"), robust=False)),
    ("global: contact + continuity, no main-box bounds", dict(constraints=("contact", "continuity"), bounds=False)),
    ("global: contact + continuity, silhouette-only main box", dict(constraints=("contact", "continuity"), refine_main=False)),
    ("global: contact + continuity, smaller segments (100 px)", dict(constraints=("contact", "continuity"), region_size=100, ruler=80)),
    ("global: contact + continuity + whole-object prior", dict(constraints=("contact", "continuity"), prior=True)),
]


def evaluate(name, cam, img, mask, vps, uv_gt, z_gt, shape, out):
    res = {}
    labels, _ = s3.superpixels(img, mask)
    L = lift3d.lift_mask(cam, mask, img, shape=shape)
    for label, kw in VARIANTS:
        t0 = time.time()
        if kw is None:
            r = s3.reconstruct(cam, img, mask, vps, log=lambda *a: None)
        else:
            kw = dict(kw)
            if kw.pop("prior", False):
                kw["prior_depth"] = model_depth_fn(cam, L["cuboids"])
            lab = None if "region_size" in kw else labels
            r = global_boxes(cam, img, mask, vps, labels=lab, log=lambda *a: None, **kw)
        e = s3.depth_errors(s3.depth_from_boxes(cam, uv_gt, boxes_of(r)), z_gt)
        e["solved"] = f"{len(r['cuboids'])}/{r['n_segments']}"
        e["seconds"] = round(time.time() - t0, 1)
        res[label] = e
        print(f"{name:4s} {label:58s} median {100*e['median_rel']:.2f}%  mean {100*e['mean_rel']:.2f}%  "
              f"<=3% {100*e['within_3pct']:.0f}%  solved {e['solved']}  ({e['seconds']}s)", flush=True)
    e = s3.depth_errors(s3.depth_from_boxes(cam, uv_gt, boxes_of(L)), z_gt)
    res[f"lift3d whole-object {shape} model (reference)"] = e
    print(f"{name:4s} {'lift3d whole-object model (reference)':58s} median {100*e['median_rel']:.2f}%  <=3% {100*e['within_3pct']:.0f}%")
    json.dump(res, open(out, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="directory with segment3d_eval.py outputs (synthetic pig)")
    ap.add_argument("--out", default="docs/segments")
    a = ap.parse_args()
    E = a.eval_dir.rstrip("/") + "/"
    cam = lift3d.default_camera()
    vps = lift3d.vanishing_points()
    img = cv2.imread(s3.__file__.rsplit("/", 1)[0] + "/box.JPG")
    mask = lift3d.object_mask(s3.__file__.rsplit("/", 1)[0] + "/reference.JPG", s3.__file__.rsplit("/", 1)[0] + "/box.JPG")
    labels, _ = s3.superpixels(img, mask)
    ys, xs = np.nonzero(labels[::4, ::4] > 0)
    uv = np.c_[xs * 4, ys * 4].astype(float)
    s3.SIGNS = s3.axis_signs(cam, vps)
    z_gt = s3.depth_from_boxes(cam, uv, [lift3d.box_ground_truth(cam)["corners"]])
    evaluate("box", cam, img, mask, vps, uv, z_gt, "box", f"{a.out}/variants_box.json")
    g = np.load(E + "synth_pig_gt.npz")
    evaluate("pig", cam, cv2.imread(E + "synth_pig.png"), np.load(E + "synth_pig_mask.npy"), vps, g["uv"], g["z"], "round",
             f"{a.out}/variants_pig.json")


if __name__ == "__main__":
    main()
