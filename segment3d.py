"""The original recursive per-segment 3D box method, made metric.

This is the method of surf.py / cube.py:
    1. a 2D box from vanishing-point tangents is fitted to the whole object;
    2. the superpixels at the object's extreme points become seeds placed on the
       faces of that main box;
    3. every other superpixel gets its own tangent box, and its 3D box is derived
       recursively from a neighbouring superpixel whose 3D box is already known.

The original turned 2D boxes into 3D with per-face homographies onto hard-coded
object dimensions (21 x 7 x 7) and scaled depths by 2D length ratios, so the
cuboids had no metric meaning; its recursion also marked a segment visited
before solving it, so cycles returned nothing (~350 failures on box.JPG).

Here the same steps use the calibrated camera from lift3d.py (built from the
original vanishing-point lines). A tangent box is the exact image of an
axis-aligned 3D box (its edges run to the three vanishing points), so it is
determined in 3D as soon as one of its faces lies on a known plane:
    - seeds: a face on a plane of the metric main box (lift3d.fit_bounding_box),
    - other segments: a face shared with an already-solved neighbour's box.
Propagation is breadth-first; every segment is then re-estimated from all its
solved neighbours and the results averaged (the original's averaging step).

Status: the box version (`reconstruct`) solves every segment but its boxes drift in
3D (a superpixel is a surface patch, while its box depth comes from its 2D extent).
`patch_reconstruct` fixes that: each segment is a planar patch, placed recursively
(coplanar / fold through the shared edge, colour + vanishing-point edge evidence),
then refined for connectivity, anchored on the edge-refined metric main box.
`make3d_planes` is a Make3D-style global baseline. Results: NOTES.md section 9.

Corner order of cube.compute_3d_box_from_plain_mask_new (verified on box.JPG):
    lower = [a front, b (a + Y), h (a + X + Y), c (a + X)],  upper = same + Z
with X towards the left vanishing point, Y towards the right one, Z up.

    python segment3d.py box.JPG pig.JPG --out out/
"""
import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
from scipy.optimize import least_squares

import lift3d
from cube import compute_3d_box_from_plain_mask_new
from graph import find_neighbors

REPO = os.path.dirname(os.path.abspath(__file__))

# corner offsets (in units of the box sizes along X, Y, Z) for lower a, b, h, c and upper a1, b1, h1, c1
OFFSETS = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                    [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]], float)
# faces as (axis, side) -> corner indices
FACES = {(0, 0): [0, 1, 5, 4], (0, 1): [3, 2, 6, 7],
         (1, 0): [0, 3, 7, 4], (1, 1): [1, 2, 6, 5],
         (2, 0): [0, 1, 2, 3], (2, 1): [4, 5, 6, 7]}


# ---------------------------------------------------------------------------
# 2D tangent boxes (the original method, sped up)
# ---------------------------------------------------------------------------
def tangent_box_2d(mask, vps):
    """The original VP-tangent 2D box of a region, as 8 image corners (or None).

    The tangent points always lie on the convex hull, so only the hull outline is
    passed to the original routine (same result, much faster).
    """
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    hull = cv2.convexHull(max(cs, key=len))
    if len(hull) < 3:
        return None
    outline = np.zeros(mask.shape, np.uint8)
    cv2.polylines(outline, [hull], True, 1, 1)
    try:
        lo, up, _ = compute_3d_box_from_plain_mask_new(outline > 0, vps["vertical"], vps["left"], vps["right"])
        pts = np.vstack([np.asarray(lo, float), np.asarray(up, float)])
    except Exception:
        return None
    return pts if pts.shape == (8, 2) and np.isfinite(pts).all() else None


# ---------------------------------------------------------------------------
# Metric boxes
# ---------------------------------------------------------------------------
def axis_signs(cam, vps):
    """+1/-1 per world axis: the sign of the step that moves a corner towards that axis's
    vanishing point in the image (a VP is the image of +axis or of -axis infinity)."""
    signs = []
    for axis, key in ((0, "left"), (1, "right")):
        e = np.zeros(3)
        e[axis] = 1.0
        near = cam.backproject_to_plane(np.array([[cam.K[0, 2], cam.K[1, 2] + 300]]), 0.0)[0]
        uv0, _ = cam.project(near[None])
        uv1, _ = cam.project((near + 0.01 * e)[None])
        towards = np.dot(uv1[0] - uv0[0], np.asarray(vps[key]) - uv0[0]) > 0
        signs.append(1.0 if towards else -1.0)
    return np.array(signs + [1.0])           # upper face is above the lower one: +Z


def box_corners(origin, size):
    """Corners a, b, h, c, a1, b1, h1, c1 of a box with corner a at `origin` and signed edge
    lengths `size` (sign = direction of the edge from a along each world axis)."""
    return np.asarray(origin, float) + OFFSETS * np.asarray(size, float)


def box_extent(origin, size):
    c = box_corners(origin, size)
    return c.min(0), c.max(0)


SIGNS = None


def lift_tangent_box(cam, corners2d, axis, side, value, signs=None):
    """3D axis-aligned box whose projection matches `corners2d`, with face (axis, side)
    (side 0: the face through corner a, side 1: the opposite one) on plane x[axis] = value."""
    signs = SIGNS if signs is None else signs
    face = FACES[(axis, side)]
    d = cam.ray(corners2d[face[0]])[0]
    t = (value - cam.C[axis]) / d[axis] if abs(d[axis]) > 1e-9 else 1.0
    anchor = cam.C + d * t
    off = OFFSETS[face[0]]

    def unpack(p):
        size = signs * (np.abs(p[3:]) + 1e-6)
        origin = p[:3] - off * size
        origin[axis] = value - side * size[axis]
        return origin, size

    def resid(p):
        origin, size = unpack(p)
        uv, depth = cam.project(box_corners(origin, size))
        r = (uv - corners2d).ravel()
        return np.where(np.repeat(depth, 2) > 0, r, 1e4)

    scale = max(np.linalg.norm(anchor - cam.C) * 0.05, 1e-3)
    p0 = np.r_[anchor, scale, scale, scale]
    res = least_squares(resid, p0, x_scale="jac", max_nfev=200)
    origin, size = unpack(res.x)
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    return origin, size, rms


def box_iou_2d(cam, origin, size, mask):
    sil = np.zeros(mask.shape, np.uint8)
    uv, depth = cam.project(box_corners(origin, size))
    if (depth <= 0).any():
        return 0.0
    cv2.fillConvexPoly(sil, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    inter = np.logical_and(sil, mask).sum()
    return inter / max(np.logical_or(sil, mask).sum(), 1)


def face_quad(cam, origin, size, axis, side):
    uv, _ = cam.project(box_corners(origin, size)[FACES[(axis, side)]])
    return uv


def quad_iou(q1, q2, shape):
    a = np.zeros(shape, np.uint8)
    b = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(a, cv2.convexHull(q1.astype(np.float32)).astype(np.int32), 1)
    cv2.fillConvexPoly(b, cv2.convexHull(q2.astype(np.float32)).astype(np.int32), 1)
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


# ---------------------------------------------------------------------------
# The recursive segment method
# ---------------------------------------------------------------------------
def superpixels(image, mask, region_size=160, ruler=120, iters=10):
    """SLIC superpixels kept where >= 50 % of a segment lies on the object (as in surf.py)."""
    slic = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.MSLIC, region_size=region_size, ruler=ruler)
    slic.iterate(iters)
    labels = slic.getLabels().copy()
    contour = slic.getLabelContourMask(False)
    out = np.zeros_like(labels)
    k = 1
    for lab in np.unique(labels):
        seg = labels == lab
        if mask[seg].mean() >= 0.5:
            out[seg & mask] = k
            k += 1
    return out, contour


def extreme_points(mask, vps):
    """The six tangent points of the whole silhouette (seeds of the original method)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cs, key=len))
    outline = np.zeros(mask.shape, np.uint8)
    cv2.polylines(outline, [hull], True, 1, 1)
    try:
        _, _, ext = compute_3d_box_from_plain_mask_new(outline > 0, vps["vertical"], vps["left"], vps["right"])
        ext = np.asarray([np.asarray(e, float).ravel()[:2] for e in ext], float)
        if ext.shape == (6, 2) and np.isfinite(ext).all():
            return ext
    except Exception:
        pass
    # degenerate tangent geometry: fall back to the lowest, highest, leftmost and rightmost points
    ys, xs = np.nonzero(mask)
    return np.array([[xs[np.argmax(ys)], ys.max()], [xs[np.argmin(ys)], ys.min()],
                     [xs.min(), ys[np.argmin(xs)]], [xs.max(), ys[np.argmax(xs)]]], float)


def segment_at(labels, p, r=25):
    """Label of the object segment at (or nearest within r px of) image point p."""
    x, y = int(round(p[0])), int(round(p[1]))
    H, W = labels.shape
    win = labels[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
    vals = win[win > 0]
    return int(np.bincount(vals).argmax()) if len(vals) else 0


def reconstruct(cam, image, mask, vps, region_size=160, ruler=120, passes=2, max_rms=6.0, log=print,
                attach_main=True, main_box=None):
    """Recursive per-segment metric boxes. Returns dict with boxes {label: (origin, size)} and diagnostics."""
    global SIGNS
    SIGNS = axis_signs(cam, vps)
    if main_box is not None:                       # a known extent (e.g. from multi-view carving), (lo, hi)
        main_lo, main_hi = (np.asarray(v, float) for v in main_box)
        main, main_iou = None, float("nan")
    else:
        main, main_iou = lift3d.fit_bounding_box(cam, mask, fit_yaw=False)     # axis-aligned with the VPs
        cx, cy, ln, wd, ht, _ = main
        main_lo = np.array([cx - ln / 2, cy - wd / 2, 0.0])
        main_hi = np.array([cx + ln / 2, cy + wd / 2, ht])
    labels, contour = superpixels(image, mask, region_size, ruler)
    ids = [int(v) for v in np.unique(labels) if v > 0]
    boxes2d = {i: tangent_box_2d(labels == i, vps) for i in ids}
    neighbours, _ = find_neighbors(labels, contour)
    log(f"{len(ids)} segments, {sum(b is not None for b in boxes2d.values())} with a tangent box; main box IoU {main_iou:.3f}")

    tol = 0.05 * (main_hi - main_lo)
    solved, how = {}, {}
    # seeds: segments at the extreme points, each placed on the best-fitting plane of the main box
    seeds = sorted({segment_at(labels, p) for p in extreme_points(mask, vps)} - {0})
    for s in seeds:
        if boxes2d.get(s) is None:
            continue
        best = None
        for axis in range(3):
            for side in (0, 1):
                for value in (main_lo[axis], main_hi[axis]):
                    o, sz, rms = lift_tangent_box(cam, boxes2d[s], axis, side, value)
                    blo, bhi = box_extent(o, sz)
                    inside = np.all(blo >= main_lo - tol) and np.all(bhi <= main_hi + tol)
                    score = rms + (0 if inside else 1e3)
                    if best is None or score < best[0]:
                        best = (score, o, sz, (axis, side))
        if best is not None and best[0] < 1e3:
            solved[s] = (best[1], best[2])
            how[s] = f"seed on main-box face {best[3]}"

    def estimate(s, known):
        """Box of s from every solved neighbour: shared face chosen by 2D overlap of projected faces."""
        cands = []
        for n in neighbours.get(s, {}):
            if n not in known or boxes2d.get(s) is None:
                continue
            no, nsz = known[n]
            best = None
            for axis in range(3):
                for side in (0, 1):
                    # s's face (axis, side) touches n's opposite face (axis, 1 - side)
                    plane = no[axis] + (1 - side) * nsz[axis]
                    nq = face_quad(cam, no, nsz, axis, 1 - side)
                    sq = boxes2d[s][FACES[(axis, side)]]
                    ov = quad_iou(sq, nq, mask.shape)
                    if ov <= 0:
                        continue
                    o, sz, rms = lift_tangent_box(cam, boxes2d[s], axis, side, plane)
                    if rms <= max_rms and (best is None or ov > best[0]):
                        best = (ov, o, sz)
            if best is not None:
                cands.append(best)
        if attach_main and boxes2d.get(s) is not None:
            # the segment may also lie on the object's outer surface: a plane of the metric main box
            for axis in range(3):
                for side in (0, 1):
                    for value in (main_lo[axis], main_hi[axis]):
                        o, sz, rms = lift_tangent_box(cam, boxes2d[s], axis, side, value)
                        blo, bhi = box_extent(o, sz)
                        if rms <= max_rms and np.all(blo >= main_lo - tol) and np.all(bhi <= main_hi + tol):
                            cands.append((0.5, o, sz))
        if not cands:
            return None
        # keep only candidates that stay inside the main box when any do (they are consistent with the object)
        inside = [c for c in cands if np.all(box_extent(c[1], c[2])[0] >= main_lo - tol) and np.all(box_extent(c[1], c[2])[1] <= main_hi + tol)]
        cands = inside or cands
        # robust average: candidates closest to the median box
        med = np.median([c[1] for c in cands], axis=0)
        cands = sorted(cands, key=lambda c: np.linalg.norm(c[1] - med))[: max(1, (len(cands) + 1) // 2)]
        w = np.array([c[0] for c in cands])
        o = np.average([c[1] for c in cands], axis=0, weights=w)
        sz = np.average([c[2] for c in cands], axis=0, weights=w)
        return o, sz

    # breadth-first propagation from the seeds (each segment solved once, from its solved neighbours)
    queue = deque(n for s in solved for n in neighbours.get(s, {}))
    while queue:
        s = queue.popleft()
        if s in solved:
            continue
        e = estimate(s, solved)
        if e is not None:
            solved[s] = e
            how[s] = "propagated"
            queue.extend(n for n in neighbours.get(s, {}) if n not in solved)
    # consistency passes: re-estimate every non-seed segment from all its solved neighbours
    for _ in range(passes):
        new = {}
        for s in solved:
            if how[s].startswith("seed"):
                new[s] = solved[s]
                continue
            e = estimate(s, solved)
            new[s] = e if e is not None else solved[s]
        solved = new

    cuboids = []
    for s, (o, sz) in solved.items():
        b = box_corners(o, sz)
        cuboids.append({"bottom": b[:4].tolist(), "height": float(abs(sz[2])), "segment": int(s), "how": how[s]})
    return dict(cuboids=cuboids, labels=labels, n_segments=len(ids), seeds=seeds, main=main, main_iou=main_iou,
                failed=[s for s in ids if s not in solved])


def draw_overlay(image, cam, labels, cuboids):
    img = image.copy()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    rng = np.random.default_rng(1)
    for c in cuboids:
        col = tuple(int(v) for v in rng.integers(60, 255, 3))
        img[labels == c["segment"]] = (0.55 * img[labels == c["segment"]] + 0.45 * np.array(col)).astype(np.uint8)
        b = np.asarray(c["bottom"])
        uv, _ = cam.project(np.vstack([b, b + [0, 0, c["height"]]]))
        for a, k in edges:
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[k]).astype(int)), col, 3, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--reference", default=os.path.join(REPO, "reference.JPG"))
    ap.add_argument("--out", default=".")
    ap.add_argument("--region-size", type=int, default=160)
    ap.add_argument("--ruler", type=int, default=120)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cam = lift3d.default_camera()
    vps = lift3d.vanishing_points()
    for path in a.images:
        image = cv2.imread(path)
        mask = lift3d.object_mask(a.reference, path)
        r = reconstruct(cam, image, mask, vps, a.region_size, a.ruler)
        sil = lift3d.render_silhouette(cam, r["cuboids"], mask.shape)
        scores = lift3d.silhouette_scores(sil, mask)
        stem = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
        json.dump({"z_up": True, "units": "camera heights", "cuboids": r["cuboids"], "scores": scores}, open(stem + "_segments.json", "w"))
        cv2.imwrite(stem + "_segments.png", draw_overlay(image, cam, r["labels"], r["cuboids"]))
        print(f"{path}: {len(r['cuboids'])}/{r['n_segments']} segments solved ({len(r['seeds'])} seeds); "
              f"reprojection IoU {scores['iou']:.3f}")


if __name__ == "__main__":
    main()


# ===========================================================================
# Drift-free version: superpixels as planar surface patches
# ===========================================================================
# A superpixel is a piece of the object's surface, not a solid. Each one is given a
# plane (axis-aligned with the vanishing-point directions, as the original boxes are),
# and its 3D points are its pixels back-projected onto that plane, so no depth is
# guessed from 2D extents. The recursion of the original method is kept: seeds at the
# extreme points sit on faces of the metric main box; a neighbour of a solved segment is
# either on the same plane (coplanar) or on a perpendicular plane through their shared
# boundary (a fold); colour similarity across the boundary chooses between the two, and
# the patch must stay inside the main box.

def pair_boundaries(labels):
    """{(i, j): (N, 2) array of (x, y) boundary pixel positions} for adjacent object segments i < j."""
    out = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dy, : labels.shape[1] - dx]
        b = labels[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        ys, xs = np.nonzero(m)
        i, j = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        pts = np.c_[xs + dx / 2.0, ys + dy / 2.0]
        for key in set(zip(i.tolist(), j.tolist())):
            sel = (i == key[0]) & (j == key[1])
            out.setdefault(key, []).append(pts[sel])
    return {k: np.vstack(v) for k, v in out.items()}


def backproject_plane(cam, uv, axis, value):
    """Points where the rays through pixels `uv` meet the plane x[axis] = value (and the ray parameter)."""
    d = cam.ray(uv)
    t = (value - cam.C[axis]) / np.where(np.abs(d[:, axis]) < 1e-9, 1e-9, d[:, axis])
    return cam.C + d * t[:, None], t


def segment_pixels(labels, ids, step=4):
    """Sub-sampled pixel coordinates of every segment (enough to place a planar patch)."""
    ys, xs = np.nonzero(labels[::step, ::step])
    lab = labels[::step, ::step][ys, xs]
    pts = np.c_[xs * step, ys * step].astype(float)
    return {i: pts[lab == i] for i in ids}


def mean_colours(image, labels, ids):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    flat = labels.ravel()
    return {i: lab[flat == i].mean(0) for i in ids}


def _inside_frac(P, lo, hi, tol):
    return float(np.mean(np.all((P >= lo - tol) & (P <= hi + tol), axis=1))) if len(P) else 0.0


def orientation_votes(image, labels, vps, min_len=12, max_angle_deg=2.5):
    """Per segment, total length of short image edges running towards each vanishing point.

    Returns {segment: array([X, Y, Z])} (X: left VP, Y: right VP, Z: vertical VP). A planar
    patch with normal along axis k can only contain edges along the other two axes, which
    is the orientation-map cue of Lee, Hebert & Kanade (2009).
    """
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(g)[0]
    out = {int(i): np.zeros(3) for i in np.unique(labels) if i > 0}
    if segs is None:
        return out
    segs = segs.reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs, L = segs[L >= min_len], L[L >= min_len]
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    dirn = (segs[:, 2:] - segs[:, :2]) / L[:, None]
    H, W = labels.shape
    xi = np.clip(mid[:, 0].astype(int), 0, W - 1)
    yi = np.clip(mid[:, 1].astype(int), 0, H - 1)
    lab = labels[yi, xi]
    cosmax = np.cos(np.radians(max_angle_deg))
    for k, key in enumerate(("left", "right", "vertical")):
        to_vp = np.asarray(vps[key])[None] - mid
        to_vp /= np.linalg.norm(to_vp, axis=1, keepdims=True)
        hit = np.abs(np.sum(to_vp * dirn, axis=1)) > cosmax
        for i, l in zip(lab[hit], L[hit]):
            if i > 0:
                out[int(i)][k] += l
    return out


def orientation_score(votes, axis):
    """In [-1, 1]: +1 when all edges lie in the plane with normal `axis`, -1 when all run along that normal."""
    tot = votes.sum()
    if tot <= 0:
        return 0.0
    return float((tot - 2 * votes[axis]) / tot)


def patch_reconstruct(cam, image, mask, vps, labels, contour, region_seeds=None, colour_tau=12.0, passes=2,
                      use_orientation=True, orient_weight=2.0, icm_iters=6, w_gap=1.0, w_orient=0.3, w_in=1.0,
                      refine_main=True):
    """Recursive planar-patch reconstruction (the original method without drift).

    Returns {segment: (axis, value)} planes, the main box and diagnostics.
    """
    main, main_iou = lift3d.fit_bounding_box(cam, mask, fit_yaw=False)
    if refine_main:
        # the main box anchors every seed, so use lift3d's edge-refined fit (visible edges on image edges)
        main = lift3d.refine_box_with_edges(cam, image, mask, main)
        main[5] = 0.0                                       # keep it aligned with the vanishing-point axes
    cx, cy, ln, wd, ht, _ = main
    lo = np.array([cx - ln / 2, cy - wd / 2, 0.0])
    hi = np.array([cx + ln / 2, cy + wd / 2, ht])
    tol = 0.04 * (hi - lo)
    ids = [int(v) for v in np.unique(labels) if v > 0]
    pix = segment_pixels(labels, ids)
    col = mean_colours(image, labels, ids)
    votes = orientation_votes(image, labels, vps) if use_orientation else {i: np.zeros(3) for i in ids}
    bnd = pair_boundaries(labels)
    nbrs = {i: set() for i in ids}
    for (i, j) in bnd:
        nbrs[i].add(j)
        nbrs[j].add(i)
    # visible faces of the main box: those whose outward normal points towards the camera
    visible = []
    for axis in range(3):
        for value, normal_sign in ((lo[axis], -1.0), (hi[axis], 1.0)):
            centre = (lo + hi) / 2
            centre[axis] = value
            if normal_sign * (cam.C[axis] - value) > 0:
                visible.append((axis, float(value)))

    def score_plane(s, axis, value):
        P, t = backproject_plane(cam, pix[s], axis, value)
        if (t <= 0).any():
            return -1.0
        return _inside_frac(P, lo, hi, tol)

    planes, how = {}, {}
    seeds = region_seeds if region_seeds is not None else sorted({segment_at(labels, p) for p in extreme_points(mask, vps)} - {0})
    def oriented(s, axis):
        return max(0.05, 1.0 + orient_weight * orientation_score(votes[s], axis)) if use_orientation else 1.0

    for s in seeds:
        best = max(visible, key=lambda pl: score_plane(s, *pl) * oriented(s, pl[0]))
        if score_plane(s, *best) > 0.5:
            planes[s], how[s] = best, "seed"

    def candidates(s, known):
        out = []
        for n in nbrs[s]:
            if n not in known:
                continue
            axis, value = known[n]
            key = (min(s, n), max(s, n))
            bp, _ = backproject_plane(cam, bnd[key], axis, value)
            similar = np.linalg.norm(col[s] - col[n]) < colour_tau
            w = len(bnd[key])
            out.append(((axis, value), w * (2.0 if similar else 1.0)))           # coplanar
            for k in range(3):                                                   # folds through the shared edge
                if k != axis:
                    out.append(((k, float(np.median(bp[:, k]))), w * (1.0 if similar else 2.0)))
        return out

    def choose(s, known):
        best, best_score = None, -1.0
        votes = {}
        for pl, w in candidates(s, known):
            key = (pl[0], round(pl[1], 3))
            votes.setdefault(key, [0.0, []])
            votes[key][0] += w
            votes[key][1].append(pl[1])
        for (axis, _), (w, vals) in votes.items():
            value = float(np.median(vals))
            sc = score_plane(s, axis, value)
            if sc <= 0.5:
                continue
            total = sc * w * oriented(s, axis)
            if total > best_score:
                best, best_score = (axis, value), total
        return best

    queue = deque(n for s in planes for n in nbrs[s])
    while queue:
        s = queue.popleft()
        if s in planes:
            continue
        pl = choose(s, planes)
        if pl is not None:
            planes[s], how[s] = pl, "propagated"
            queue.extend(n for n in nbrs[s] if n not in planes)
    for _ in range(passes):
        planes = {s: (planes[s] if how[s] == "seed" else (choose(s, planes) or planes[s])) for s in planes}

    # Connectivity refinement (iterated conditional modes): every non-seed patch re-picks, among its
    # candidate planes, the one whose boundaries meet the neighbours' patches with the smallest 3D gap
    # (relative to depth), plus orientation evidence and staying inside the main box.
    def gap(s, pl, n, pn):
        key = (min(s, n), max(s, n))
        pts = bnd[key][:: max(1, len(bnd[key]) // 15)]
        A, ta = backproject_plane(cam, pts, *pl)
        B, tb = backproject_plane(cam, pts, *pn)
        if (ta <= 0).any() or (tb <= 0).any():
            return 1.0
        return float(np.median(np.linalg.norm(A - B, axis=1) / np.maximum(np.linalg.norm(A - cam.C, axis=1), 1e-9)))

    def cost(s, pl):
        inside = score_plane(s, *pl)
        if inside <= 0.3:
            return np.inf
        g, wsum = 0.0, 0.0
        for n in nbrs[s]:
            if n in planes:
                w = np.sqrt(len(bnd[(min(s, n), max(s, n))]))
                g += w * min(gap(s, pl, n, planes[n]), 0.1)        # truncated: tolerate true depth edges
                wsum += w
        g = g / wsum if wsum else 0.0
        o = (1 - orientation_score(votes[s], pl[0])) / 2 if use_orientation else 0.0
        return w_gap * g / 0.01 + w_orient * o + w_in * (1 - inside)

    for _ in range(icm_iters):
        changed = 0
        for s in sorted(planes):
            if how[s] == "seed":
                continue
            cands = {planes[s]}
            cands.update(pl for pl, _ in candidates(s, planes))
            cands.update(visible)
            best = min(cands, key=lambda pl: cost(s, pl))
            if best != planes[s] and cost(s, best) < cost(s, planes[s]):
                planes[s] = best
                changed += 1
        if not changed:
            break
    return dict(planes=planes, how=how, main=(lo, hi), main_iou=main_iou, seeds=seeds, ids=ids,
                failed=[s for s in ids if s not in planes])


def make3d_planes(cam, image, labels, anchors, colour_tau=12.0, w_conn=1.0, w_copl=0.5, w_anchor=10.0, reg=1e-4):
    """Make3D-style global solve (Saxena et al. 2009) without the learned depth term.

    Each superpixel i has plane parameters a_i (camera frame) with inverse depth
    1/z = r . a_i along ray r = K^-1 [u, v, 1]. Least squares over all superpixels:
        connectivity  sum over boundary pixels p of w_conn (r_p . a_i - r_p . a_j)^2
        coplanarity   for colour-similar neighbours, at each other's centres
        anchors       w_anchor (r_p . a_i - 1/z_p)^2 at given pixels (replacing the
                      learned per-pixel depth: the same seeds as the recursive method)
    Returns {segment: a (3,)}.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr
    ids = [int(v) for v in np.unique(labels) if v > 0]
    idx = {s: k for k, s in enumerate(ids)}
    Kinv = np.linalg.inv(cam.K)
    rays = lambda uv: np.c_[uv, np.ones(len(uv))] @ Kinv.T
    bnd = pair_boundaries(labels)
    col = mean_colours(image, labels, ids)
    cen = {s: np.c_[np.nonzero(labels == s)[1].mean(), np.nonzero(labels == s)[0].mean()] for s in ids}
    rows, rhs, entries = 0, [], []

    def add(i, ri, j=None, rj=None, b=0.0, w=1.0):
        nonlocal rows
        entries.append((rows, i, ri * w))
        if j is not None:
            entries.append((rows, j, -rj * w))
        rhs.append(b * w)
        rows += 1

    for (i, j), pts in bnd.items():
        sub = pts[:: max(1, len(pts) // 20)]
        for r in rays(sub):
            add(idx[i], r, idx[j], r, w=w_conn / np.sqrt(len(sub)))
        if np.linalg.norm(col[i] - col[j]) < colour_tau:
            for s, other in ((i, j), (j, i)):
                r = rays(cen[other])[0]
                add(idx[s], r, idx[other], r, w=w_copl)
    for s, (uv, z) in anchors.items():
        sub = np.arange(len(uv))[:: max(1, len(uv) // 30)]
        for r, zz in zip(rays(uv[sub]), z[sub]):
            add(idx[s], r, b=1.0 / zz, w=w_anchor / np.sqrt(len(sub)))
    for s in ids:                                                     # tiny ridge so every system is solvable
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            add(idx[s], e, w=reg)
    A = lil_matrix((rows, 3 * len(ids)))
    for r, i, v in entries:
        A[r, 3 * i: 3 * i + 3] = v
    sol = lsqr(A.tocsr(), np.array(rhs), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return {s: sol[3 * idx[s]: 3 * idx[s] + 3] for s in ids}


# ---------------------------------------------------------------------------
# Depth maps for evaluation
# ---------------------------------------------------------------------------
def camera_depth(cam, P):
    """Depth (camera z) of world points."""
    return ((P - cam.C) @ cam.R.T)[:, 2]


def depth_from_planes(cam, labels, planes, step=4):
    """Per-pixel camera depth (sub-sampled grid) from world-axis planes {s: (axis, value)}."""
    ys, xs = np.nonzero(labels[::step, ::step] > 0)
    uv = np.c_[xs * step, ys * step].astype(float)
    lab = labels[ys * step, xs * step]
    z = np.full(len(uv), np.nan)
    for s, (axis, value) in planes.items():
        sel = lab == s
        if sel.any():
            P, t = backproject_plane(cam, uv[sel], axis, value)
            z[sel] = np.where(t > 0, camera_depth(cam, P), np.nan)
    return uv, z


def depth_from_make3d(cam, labels, alphas, step=4):
    ys, xs = np.nonzero(labels[::step, ::step] > 0)
    uv = np.c_[xs * step, ys * step].astype(float)
    lab = labels[ys * step, xs * step]
    r = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(cam.K).T
    inv = np.array([r[k] @ alphas[l] for k, l in enumerate(lab)])
    return uv, np.where(inv > 1e-9, 1.0 / np.maximum(inv, 1e-9), np.nan)


def depth_from_boxes(cam, uv, boxes):
    """Nearest ray hit with a set of oriented boxes (each: 8 corners as from box_corners/lift3d)."""
    d = cam.ray(uv)
    best = np.full(len(uv), np.inf)
    for corners in boxes:
        o = corners[0]
        axes = [corners[3] - o, corners[1] - o, corners[4] - o]         # a->c, a->b, a->a1
        lens = [np.linalg.norm(a) for a in axes]
        U = np.array([a / max(L, 1e-12) for a, L in zip(axes, lens)])
        q = (cam.C - o) @ U.T
        dd = d @ U.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (0 - q) / dd
            t2 = (np.array(lens) - q) / dd
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmax > 0)
        best = np.where(hit & (tmin < best), tmin, best)
    P = cam.C + d * best[:, None]
    z = camera_depth(cam, P)
    return np.where(np.isfinite(best), z, np.nan)


def depth_errors(z, z_gt):
    ok = np.isfinite(z) & np.isfinite(z_gt) & (z_gt > 0)
    rel = np.abs(z[ok] - z_gt[ok]) / z_gt[ok]
    cover = float(np.mean(np.isfinite(z[np.isfinite(z_gt)]))) if np.isfinite(z_gt).any() else 0.0
    return dict(median_rel=float(np.median(rel)) if len(rel) else None, mean_rel=float(np.mean(rel)) if len(rel) else None,
                within_1pct=float(np.mean(rel < 0.01)) if len(rel) else None,
                within_3pct=float(np.mean(rel < 0.03)) if len(rel) else None, coverage=cover)
