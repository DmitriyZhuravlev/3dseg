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

Status (work in progress): with the calibrated camera the whole-object tangent box
is metric (box.JPG: 0.320 x 0.461 x 0.304 m vs traced 0.348 x 0.446 x 0.282 m), and
every segment gets a box (104/104 on box.JPG, 27/27 on pig.JPG; reprojection IoU
0.965 / 0.873). But the per-segment boxes drift from the true surfaces in 3D: a
superpixel is a surface patch, while its box depth comes from its 2D extent, and
the error accumulates along the propagation.

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
    except Exception:
        return None
    pts = np.vstack([lo, up]).astype(float)
    return pts if np.isfinite(pts).all() else None


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
    _, _, ext = compute_3d_box_from_plain_mask_new(outline > 0, vps["vertical"], vps["left"], vps["right"])
    return np.asarray(ext, float)


def segment_at(labels, p, r=25):
    """Label of the object segment at (or nearest within r px of) image point p."""
    x, y = int(round(p[0])), int(round(p[1]))
    H, W = labels.shape
    win = labels[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
    vals = win[win > 0]
    return int(np.bincount(vals).argmax()) if len(vals) else 0


def reconstruct(cam, image, mask, vps, region_size=160, ruler=120, passes=2, max_rms=6.0, log=print,
                attach_main=True):
    """Recursive per-segment metric boxes. Returns dict with boxes {label: (origin, size)} and diagnostics."""
    global SIGNS
    SIGNS = axis_signs(cam, vps)
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
