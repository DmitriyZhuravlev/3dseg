"""3D shape from 3D normals in a single calibrated image (perspective normal integration).

For a pixel (u, v) with camera-frame normal n, the surface point is X = z * q with
q = ((u - cx)/f, (v - cy)/f, 1). Tangency (n . dX/du = 0) gives the gradient of log depth:

    d log z / du = -n_x / (n_x (u - cx) + n_y (v - cy) + n_z f),   same for v with n_y.

So a normal field fixes log z up to one constant: the shape up to scale. Least squares over
the object's pixels (Frankot-Chellappa / Poisson integration, perspective form) recovers it.
The scale comes from one metric anchor; here the ground contact of the calibrated camera
(the lowest object pixel lies on the table plane), so no ground truth is used.

Normals tested on the synthetic pig (exact 3D known, segment3d_eval.py):
  * exact normals of the true surface: an upper bound that shows integration itself works;
  * silhouette normals, from the image alone: on the outline the normal is exactly the
    outline's 2D normal (perpendicular to the viewing ray); inside, the two in-image
    components are interpolated harmonically and n_z follows from |n| = 1 (as in Teddy /
    shape-from-silhouette inflation);
  * perturbed exact normals (noise of a few degrees), standing in for a learned normal
    estimator's accuracy.

    python normal_integration.py --eval-dir out/segeval
"""
import argparse
import json

import cv2
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

import lift3d
import segment3d as s3
import synthetic3d


def grid_of(uv):
    """Index the sample pixels (on a regular step grid) by their grid position."""
    step = int(np.min(np.diff(np.unique(uv[:, 0])))) if len(np.unique(uv[:, 0])) > 1 else 1
    ij = np.round(uv / step).astype(int)
    return {tuple(p): k for k, p in enumerate(ij)}, ij, step


def integrate_normals(cam, uv, n_cam, anchor_idx=None, anchor_depth=None, robust=True, sigma=0.002, iters=12, cut=None):
    """Least-squares log-depth from camera-frame normals at pixels uv (regular grid).
    Returns camera depth per pixel (scale fixed by the anchors, or 1 at the first pixel).
    cut: optional set of (k, m) sample pairs known to straddle a depth discontinuity; their
    equations keep only a tiny weight (the surface stays one connected piece)."""
    lut, ij, step = grid_of(uv)
    f = cam.K[0, 0]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    den = n_cam[:, 0] * (uv[:, 0] - cx) + n_cam[:, 1] * (uv[:, 1] - cy) + n_cam[:, 2] * f
    with np.errstate(divide="ignore", invalid="ignore"):
        gu = -n_cam[:, 0] / den * step
        gv = -n_cam[:, 1] / den * step
    rows, cols, vals, b, prior = [], [], [], [], []
    r = 0
    for k, (i, j) in enumerate(ij):
        for di, dj, g in ((1, 0, gu), (0, 1, gv)):
            m = lut.get((i + di, j + dj))
            if m is None:
                continue
            gm = 0.5 * (g[k] + g[m])
            if not np.isfinite(gm) or abs(gm) > 0.5:          # grazing normals: skip the equation
                continue
            rows += [r, r]
            cols += [m, k]
            vals += [1.0, -1.0]
            b.append(gm)
            prior.append(1e-3 if cut is not None and (k, m) in cut else 1.0)
            r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, len(uv))).tocsr()
    b = np.asarray(b)
    # gauge: anchors (log depth) with a strong weight
    if anchor_idx is None:
        anchor_idx, anchor_depth = [0], [1.0]
    Aa = csr_matrix((np.ones(len(anchor_idx)), (np.arange(len(anchor_idx)), anchor_idx)), shape=(len(anchor_idx), len(uv)))
    ba = np.log(np.asarray(anchor_depth, float))
    # Robust integration (IRLS, Cauchy weights): equations across a depth discontinuity (one part
    # occluding another) disagree with their neighbours and are down-weighted, instead of
    # smearing the jump over the whole surface. A few weak links keep everything connected.
    prior = np.asarray(prior)
    w = prior.copy()
    for _ in range(iters if robust else 1):
        W = diags(w)
        N = (A.T @ W @ A + 1e4 * (Aa.T @ Aa) + diags(np.full(len(uv), 1e-9))).tocsc()
        x = spsolve(N, A.T @ (w * b) + 1e4 * (Aa.T @ ba))
        if not robust:
            break
        res = A @ x - b
        w = prior * np.maximum(1.0 / (1.0 + (res / sigma) ** 2), 1e-3)
    return np.exp(x)


def refine_hits(cam, uv, z, inside, back=0.004, iters=24):
    """Exact surface depth by bisection along each ray (the stored ground truth was found by
    ray marching in ~3 mm steps, coarser than the depth change between neighbouring samples)."""
    d = cam.ray(uv)
    t1 = z / (d @ cam.R[2])                    # inside (or on) the surface
    t0 = np.maximum(t1 - back - 0.003, 1e-3)   # outside
    ok = ~inside(cam.C + d * t0[:, None])
    for _ in range(iters):
        tm = 0.5 * (t0 + t1)
        ins = inside(cam.C + d * tm[:, None])
        t1 = np.where(ins, tm, t1)
        t0 = np.where(ins, t0, tm)
    zr = s3.camera_depth(cam, cam.C + d * t1[:, None])
    return np.where(ok, zr, z)


def exact_normals(cam, uv, z, inside, r=0.0015, n=300, seed=0):
    """Outward normals of the true surface at the hit points (mean direction to outside samples)."""
    rng = np.random.default_rng(seed)
    d = cam.ray(uv)
    P = cam.C + d * (z / (d @ cam.R[2]))[:, None]
    off = rng.normal(size=(n, 3))
    off = r * off / np.linalg.norm(off, axis=1, keepdims=True)
    out = np.zeros_like(P)
    for k in range(0, len(P), 500):
        Q = P[k:k + 500, None, :] + off[None]
        o = ~inside(Q.reshape(-1, 3)).reshape(Q.shape[:2])
        out[k:k + 500] = (o[..., None] * off[None]).sum(1)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out @ cam.R.T                                     # world -> camera frame


def silhouette_normals(cam, mask, uv, iters=3000):
    """Normals from the silhouette alone: exact 2D outline normals, harmonic interior."""
    H, W = mask.shape
    step = int(np.min(np.diff(np.unique(uv[:, 0]))))
    m = mask[::step, ::step].astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nx = np.zeros(m.shape)
    ny = np.zeros(m.shape)
    fixed = np.zeros(m.shape, bool)
    for c in cs:
        c = c[:, 0, :].astype(float)
        t = np.roll(c, -2, 0) - np.roll(c, 2, 0)                        # tangent along the outline
        nrm = np.c_[t[:, 1], -t[:, 0]]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        # outward: away from the mask interior
        probe = np.round(c + 2 * nrm).astype(int)
        inb = (probe[:, 0] >= 0) & (probe[:, 0] < m.shape[1]) & (probe[:, 1] >= 0) & (probe[:, 1] < m.shape[0])
        flip = np.zeros(len(c), bool)
        flip[inb] = m[probe[inb, 1], probe[inb, 0]] > 0
        nrm[flip] *= -1
        x, y = c[:, 0].astype(int), c[:, 1].astype(int)
        nx[y, x], ny[y, x] = nrm[:, 0], nrm[:, 1]
        fixed[y, x] = True
    # harmonic interpolation (Jacobi) inside the mask
    inside = m > 0
    for _ in range(iters):
        for a in (nx, ny):
            avg = 0.25 * (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
            a[inside & ~fixed] = avg[inside & ~fixed]
    ij = np.round(uv / step).astype(int)
    n2 = np.c_[nx[ij[:, 1], ij[:, 0]], ny[ij[:, 1], ij[:, 0]]]
    # image-plane normal -> camera frame: on the outline it is perpendicular to the viewing ray
    q = np.c_[(uv[:, 0] - cam.K[0, 2]) / cam.K[0, 0], (uv[:, 1] - cam.K[1, 2]) / cam.K[1, 1], np.ones(len(uv))]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    e1 = np.c_[np.ones(len(uv)), np.zeros(len(uv)), np.zeros(len(uv))]
    e1 -= (e1 * q).sum(1, keepdims=True) * q
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(q, e1)
    s = np.clip(np.linalg.norm(n2, axis=1), 0, 0.999)
    n = n2[:, :1] * e1 + n2[:, 1:] * e2                                   # in the plane perpendicular to the ray
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9) * s[:, None]
    return n - np.sqrt(1 - s ** 2)[:, None] * q                           # towards the camera


def perturb(n, deg, seed=1):
    rng = np.random.default_rng(seed)
    v = n + rng.normal(scale=np.radians(deg), size=n.shape)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def ground_anchor(cam, uv, mask_pts=12):
    """Metric anchors without ground truth: the lowest object pixels touch the table (z = 0)."""
    order = np.argsort(-uv[:, 1])[:mask_pts]
    P = cam.backproject_to_plane(uv[order], 0.0)
    return order, s3.camera_depth(cam, P)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    E = a.eval_dir.rstrip("/") + "/"
    cam = lift3d.default_camera()
    g = np.load(E + "synth_pig_gt.npz")
    mask = np.load(E + "synth_pig_mask.npy")
    ok = np.isfinite(g["z"])
    uv, z = g["uv"][ok], g["z"][ok]
    inside, _ = synthetic3d.make_pig()
    z = refine_hits(cam, uv, z, inside)
    res = {}
    n_true = exact_normals(cam, uv, z, inside)
    variants = {"exact normals": n_true, "exact normals + 5 deg noise": perturb(n_true, 5),
                "exact normals + 10 deg noise": perturb(n_true, 10), "silhouette normals (image only)": silhouette_normals(cam, mask, uv)}
    a_idx, a_z = ground_anchor(cam, uv)
    # oracle: the true depth discontinuities (from the ground truth) - what a perfect
    # discontinuity detector would give; isolates integration from discontinuity finding
    lut, ij, _ = grid_of(uv)
    cut = {(k, m) for k, (i, j) in enumerate(ij) for m in (lut.get((i + 1, j)), lut.get((i, j + 1)))
           if m is not None and abs(np.log(z[m] / z[k])) > 0.01}
    runs = [(name, n, True, None) for name, n in variants.items()] + [
        ("exact normals, plain least squares", n_true, False, None),
        ("exact normals, true discontinuities, parts placed at true depth (oracle)", n_true, False, cut),
        ("exact normals + 10 deg noise, true discontinuities, parts placed (oracle)", variants["exact normals + 10 deg noise"], False, cut)]
    depth = {}
    for name, n, robust, c in runs:
        zr = integrate_normals(cam, uv, n, a_idx, a_z, robust=robust, cut=c)
        if c is not None:
            # normals carry no information across a discontinuity: each separated part is only
            # known up to its own depth offset. Oracle: place every part at its true depth.
            from scipy.sparse.csgraph import connected_components
            e_k = [(k, m) for k, (i, j) in enumerate(ij) for m in (lut.get((i + 1, j)), lut.get((i, j + 1)))
                   if m is not None and (k, m) not in c]
            G = coo_matrix((np.ones(len(e_k)), tuple(np.array(e_k).T)), shape=(len(uv), len(uv)))
            _, part = connected_components(G, directed=False)
            for q in np.unique(part):
                sel = part == q
                zr[sel] *= np.median(z[sel] / zr[sel])
        depth[name] = zr
        e = s3.depth_errors(zr, z)
        # shape only (best single scale): separates shape error from the anchor's scale error
        k = np.median(z / zr)
        es = s3.depth_errors(zr * k, z)
        res[name] = dict(ground_anchored=e, best_scale=es, scale_error=float(1 / k - 1))
        print(f"{name:62s} ground-anchored median {100*e['median_rel']:.2f}%  <=3% {100*e['within_3pct']:.0f}%   "
              f"shape-only median {100*es['median_rel']:.2f}%  <=3% {100*es['within_3pct']:.0f}%  (scale error {100*(1/k-1):+.1f}%)", flush=True)
    np.savez_compressed(E + "pig_normals.npz", uv=uv, z=z, **{k.replace(" ", "_"): v for k, v in variants.items()})
    np.savez_compressed(E + "pig_normal_depths.npz", uv=uv, z=z, names=np.array(list(depth)), depths=np.array(list(depth.values())))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
