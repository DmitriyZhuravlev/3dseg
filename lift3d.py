"""Metric 2D -> 3D lifting of an object photo into cuboids.

The original pipeline (surf.py + cube.py) fits a box to the silhouette in 2D
from vanishing-point tangents and then maps it with a homography onto
hard-coded object dimensions (21 x 7 x 7), so its 3D cuboids never reproduce
the object. This module does the lifting with a calibrated camera instead:

1. Camera from the three orthogonal vanishing points (the same hard-coded VP
   lines as surf.py): principal point = orthocentre of the VP triangle,
   focal length from VP orthogonality, rotation from the back-projected VP
   directions. World frame: Z up, X/Y along the two horizontal VPs, ground
   plane Z = 0, camera height = 1 (the reconstruction is metric up to scale).
2. Object mask by background subtraction against reference.JPG (same steps
   as surf.py.process_images).
3. A 3D bounding box (centre, size, yaw) on the ground, fitted so its
   projection matches the silhouette (max IoU).
4. Carving: the box footprint is split into a grid of columns; each column
   is raised while its centre line still projects inside the silhouette
   (single-view visual hull, bounded by the fitted box). Columns become
   cuboids in the pipeline's format (bottom quad, height, colour).
5. Verification: the cuboids are projected back into the photo and compared
   to the silhouette (IoU, precision, recall).

    python lift3d.py box.JPG pig.JPG --out out/
    python -m viewer3d --scene out/box_cuboids.json --texture "" --no-showcase --no-surface
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
from scipy.optimize import minimize

REPO = os.path.dirname(os.path.abspath(__file__))

# Vanishing-point lines measured in the photos (identical to surf.py / img_rec_ipm.py)
VP_LINES = {
    "vertical": ([[710, 952], [646, 234]], [[2326, 1162], [2466, 478]]),
    "left": ([[1072, 1572], [720, 982]], [[1082, 810], [668, 240]]),
    "right": ([[1128, 1574], [2320, 1156]], [[1098, 874], [2458, 488]]),
}


def line_intersection(l1, l2):
    a = np.cross([*l1[0], 1.0], [*l1[1], 1.0])
    b = np.cross([*l2[0], 1.0], [*l2[1], 1.0])
    p = np.cross(a, b)
    return p[:2] / p[2]


def vanishing_points(lines=VP_LINES):
    return {k: line_intersection(*v) for k, v in lines.items()}


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera:
    """Pinhole camera: x ~ K R (X - C)."""

    def __init__(self, K, R, C):
        self.K, self.R, self.C = np.asarray(K, float), np.asarray(R, float), np.asarray(C, float)
        self.K_inv = np.linalg.inv(self.K)

    @classmethod
    def from_vanishing_points(cls, v_left, v_right, v_vert, height=1.0):
        a, b, c = (np.asarray(v, float) for v in (v_vert, v_left, v_right))
        # principal point = orthocentre of the VP triangle
        A = np.array([b - c, a - c])
        pp = np.linalg.solve(A, [a @ (b - c), b @ (a - c)])
        f = math.sqrt(-(b - pp) @ (c - pp))
        K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
        K_inv = np.linalg.inv(K)

        def direction(v):
            d = K_inv @ np.r_[v, 1.0]
            return d / np.linalg.norm(d)

        z = -direction(a)          # the vertical VP is below the camera: it is "down"
        if z[1] > 0:               # image y grows downward; world-up must point to -y
            z = -z
        x = direction(b)
        y = direction(c)
        if np.cross(x, y) @ z < 0:
            y = -y
        U, _, Vt = np.linalg.svd(np.column_stack([x, y, z]))  # nearest rotation
        R = U @ Vt                                            # columns: world axes in camera frame
        return cls(K, R, [0.0, 0.0, height])

    @property
    def focal(self):
        return self.K[0, 0]

    def project(self, P):
        P = np.atleast_2d(P)
        pc = (P - self.C) @ self.R.T
        uv = pc @ self.K.T
        return uv[:, :2] / uv[:, 2:3], pc[:, 2]

    def ray(self, uv):
        uv = np.atleast_2d(uv)
        d = np.c_[uv, np.ones(len(uv))] @ self.K_inv.T @ self.R  # world directions
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def backproject_to_plane(self, uv, z=0.0):
        d = self.ray(uv)
        t = (z - self.C[2]) / d[:, 2]
        return self.C + d * t[:, None]


# ---------------------------------------------------------------------------
# Segmentation (same steps as surf.py.process_images)
# ---------------------------------------------------------------------------
def object_mask(reference_path, image_path):
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or img is None:
        raise FileNotFoundError(reference_path if ref is None else image_path)
    diff = cv2.GaussianBlur(cv2.absdiff(ref, img), (15, 15), 0)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(mask, contours, i, 255, thickness=cv2.FILLED)
    # keep the largest connected component (the object)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if n > 1:
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    return mask > 0


# ---------------------------------------------------------------------------
# Cuboids
# ---------------------------------------------------------------------------
def box_corners(cx, cy, length, width, height, yaw=0.0, z0=0.0):
    """8 corners (bottom 4 CCW from above, then top 4) of a yawed box on the ground."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = length / 2, width / 2
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    xy = local @ np.array([[c, s], [-s, c]]) + [cx, cy]
    bottom = np.c_[xy, np.full(4, z0)]
    return np.vstack([bottom, bottom + [0, 0, height]])


def render_silhouette(cam, cuboids, shape, scale=1.0):
    """Rasterise the union of the projected cuboids (convex hull of 8 corners each)."""
    sil = np.zeros(shape, np.uint8)
    for cub in cuboids:
        b = np.asarray(cub["bottom"], float)
        corners = np.vstack([b, b + [0, 0, cub["height"]]])
        uv, depth = cam.project(corners)
        if (depth <= 0).any():
            continue
        hull = cv2.convexHull((uv * scale).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(sil, hull, 1)
    return sil > 0


def silhouette_scores(pred, mask):
    inter = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return {"iou": inter / max(union, 1), "precision": inter / max(pred.sum(), 1),
            "recall": inter / max(mask.sum(), 1)}


def fit_bounding_box(cam, mask, scale=0.25, fit_yaw=True):
    """(cx, cy, length, width, height, yaw) whose projection best matches the silhouette."""
    small = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) > 0
    shape = small.shape

    # initial guess: ground points of the lowest silhouette pixels (where it touches the ground)
    ys, xs = np.nonzero(mask)
    order = np.argsort(ys)[::-1]
    low = np.c_[xs[order[: max(50, len(order) // 20)]], ys[order[: max(50, len(order) // 20)]]]
    g_all = cam.backproject_to_plane(np.c_[xs[::97], ys[::97]], 0.0)
    g_low = cam.backproject_to_plane(low, 0.0)
    ext = np.ptp(g_all[:, :2], axis=0)
    size0 = max(ext.min(), 1e-3)
    x0 = np.r_[g_low[:, :2].mean(0) + 0.0, size0, size0, size0, 0.0]

    def project_box(p):
        cx, cy, ln, wd, ht, yaw = p
        if min(ln, wd, ht) <= 0:
            return None
        uv, depth = cam.project(box_corners(cx, cy, ln, wd, ht, yaw))
        if (depth <= 0).any():
            return None
        sil = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(sil, cv2.convexHull((uv * scale).astype(np.float32)).astype(np.int32), 1)
        return sil > 0

    def loss(p):
        if not fit_yaw:
            p = np.r_[p[:5], 0.0]
        sil = project_box(p)
        if sil is None:
            return 2.0
        inter = np.logical_and(sil, small).sum()
        return 1.0 - inter / np.logical_or(sil, small).sum()

    best = None
    yaws = np.radians([-60, -30, 0, 30, 60]) if fit_yaw else [0.0]
    for yaw in yaws:
        for k in (0.6, 1.0, 1.6):
            start = np.r_[x0[:2], x0[2:5] * k, yaw]
            res = minimize(loss, start, method="Nelder-Mead",
                           options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True})
            if best is None or res.fun < best.fun:
                best = res
    # polish
    best = minimize(loss, best.x, method="Powell", options={"xtol": 1e-6, "ftol": 1e-7})
    p = best.x if fit_yaw else np.r_[best.x[:5], 0.0]
    return p, 1.0 - best.fun


def visible_box_edges(cam, corners):
    """Edges of a ground box visible from a camera above it: 4 top edges, the 3 vertical
    edges not at the far corner, and the 2 bottom edges meeting at the near corner."""
    _, depth = cam.project(corners[:4])
    near, far = int(np.argmin(depth)), int(np.argmax(depth))
    edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    edges += [(k, k + 4) for k in range(4) if k != far]
    edges += [(near, (near + 1) % 4), (near, (near + 3) % 4)]
    return edges


def edge_distance_map(image_bgr, mask, truncate=25.0):
    """Distance (px) to the nearest image edge, near the object only, truncated."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    near = cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    edges[~near] = 0
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return np.minimum(dist, truncate)


def refine_box_with_edges(cam, image_bgr, mask, box, weight_sil=40.0, samples=60,
                          truncations=(120.0, 40.0, 12.0)):
    """Refine a box-like fit so that all its visible 3D edges lie on image edges.

    The silhouette constrains only the outer hexagon; interior edges (the near
    vertical edge, the top edges in front) fix how the size splits between length
    and width and where the near corner is. Loss = mean truncated edge distance
    along the projected visible edges + weight_sil * (1 - silhouette IoU), minimised
    coarse-to-fine (large truncation first so far-off edges still pull, then small
    truncation so texture edges far from the box do not).
    """
    dist_full = edge_distance_map(image_bgr, mask, truncate=1e9)
    H, W = mask.shape
    scale = 0.25
    small = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) > 0
    t = np.linspace(0, 1, samples)[:, None]

    def loss(p):
        cx, cy, ln, wd, ht, yaw = p
        if min(ln, wd, ht) <= 0:
            return 1e6
        corners = box_corners(*p)
        uv, depth = cam.project(corners)
        if (depth <= 0).any():
            return 1e6
        pts = np.vstack([uv[a] * (1 - t) + uv[b] * t for a, b in visible_box_edges(cam, corners)])
        px = np.clip(np.round(pts).astype(int), [0, 0], [W - 1, H - 1])
        d = np.minimum(dist_full[px[:, 1], px[:, 0]], trunc).mean() / trunc * 25.0
        sil = np.zeros(small.shape, np.uint8)
        cv2.fillConvexPoly(sil, cv2.convexHull((uv * scale).astype(np.float32)).astype(np.int32), 1)
        iou = np.logical_and(sil, small).sum() / np.logical_or(sil, small).sum()
        return d + weight_sil * (1 - iou)

    x = np.asarray(box, float)
    for trunc in truncations:
        res = minimize(loss, x, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-5, "adaptive": True})
        x = minimize(loss, res.x, method="Powell", options={"xtol": 1e-6, "ftol": 1e-7}).x
    return x


def carve(cam, mask, box, grid=40, nz=96, margin=0.08, height_factor=2.5, samples=3,
          symmetry="bilateral"):
    """Split the (slightly enlarged) box into columns and carve each one to the silhouette.

    The footprint gets ~square cells, `grid` along the longer side. Each column is
    sampled on a `samples` x `samples` sub-grid; a level is kept while at least half
    of its samples project inside the silhouette (majority rule, so columns neither
    overhang the outline nor leave gaps), contiguous from the ground up. Columns
    may grow up to `height_factor` times the box height, so parts that stick out
    above the fitted box (ears, handles) are still recovered.

    One view cannot see the back: carving alone leaves the visual-hull "wedge"
    (columns in front grow until their projection leaves the silhouette above the
    object, and hidden columns behind it survive). `symmetry` resolves this with
    the reflection prior surf.py's `reflect` option was meant for: each column is
    limited by its mirror image across the box's short axis ("bilateral", e.g. an
    animal), across both axes ("both", e.g. a box), or not at all ("none").
    """
    cx, cy, ln, wd, ht, yaw = box
    ln, wd = ln * (1 + margin), wd * (1 + margin)
    zmax = ht * height_factor
    nx = grid if ln >= wd else max(4, round(grid * ln / wd))
    ny = grid if wd > ln else max(4, round(grid * wd / ln))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    H, W = mask.shape

    sub = (np.arange(samples) + 0.5) / samples - 0.5
    du, dv = np.meshgrid(sub, sub)
    zs = (np.arange(nz) + 0.5) / nz * zmax
    levels = np.zeros((nx, ny), int)
    for i in range(nx):
        for j in range(ny):
            u = ((i + 0.5 + du.ravel()) / nx - 0.5) * ln
            v = ((j + 0.5 + dv.ravel()) / ny - 0.5) * wd
            xy = np.c_[u, v] @ rot + [cx, cy]
            pts = np.c_[np.repeat(xy, nz, 0), np.tile(zs, len(xy))]
            uv, depth = cam.project(pts)
            px = np.round(uv).astype(int)
            ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            ok[ok] = mask[px[ok, 1], px[ok, 0]]
            frac = ok.reshape(len(xy), nz).mean(0)
            inside = frac >= 0.5
            levels[i, j] = nz if inside.all() else int(np.argmin(inside))

    short_axis = 1 if wd <= ln else 0
    if symmetry in ("bilateral", "both"):
        levels = np.minimum(levels, np.flip(levels, axis=short_axis))
    if symmetry == "both":
        levels = np.minimum(levels, np.flip(levels, axis=1 - short_axis))

    cuboids = []
    for i0, j0, i1, j1, k in merge_levels(levels):
        u0, u1 = i0 / nx - 0.5, i1 / nx - 0.5
        v0, v1 = j0 / ny - 0.5, j1 / ny - 0.5
        quad = np.array([[u0 * ln, v0 * wd], [u1 * ln, v0 * wd], [u1 * ln, v1 * wd], [u0 * ln, v1 * wd]])
        xy = quad @ rot + [cx, cy]
        cuboids.append({"bottom": np.c_[xy, np.zeros(4)].tolist(), "height": float(k / nz * zmax)})
    return cuboids


def hull_heights(cam, mask, frame, nu, nv, nz=128, zmax=None, samples=3):
    """Visual-hull column heights on a grid in an object frame.

    frame = (cx, cy, length, width, yaw) defines a rectangle on the ground; it is
    split into nu x nv columns (u along `length`, v along `width`). Returns heights
    (nu, nv) in world units: how far each column rises, contiguous from the ground,
    while (by majority of sub-samples) it still projects inside the silhouette.
    """
    cx, cy, ln, wd, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    H, W = mask.shape
    sub = (np.arange(samples) + 0.5) / samples - 0.5
    du, dv = np.meshgrid(sub, sub)
    zs = (np.arange(nz) + 0.5) / nz * zmax
    heights = np.zeros((nu, nv))
    for i in range(nu):
        u = ((i + 0.5 + du.ravel()) / nu - 0.5) * ln
        for j in range(nv):
            v = ((j + 0.5 + dv.ravel()) / nv - 0.5) * wd
            xy = np.c_[u, v] @ rot + [cx, cy]
            pts = np.c_[np.repeat(xy, nz, 0), np.tile(zs, len(xy))]
            uv, depth = cam.project(pts)
            px = np.round(uv).astype(int)
            ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            ok[ok] = mask[px[ok, 1], px[ok, 0]]
            inside = ok.reshape(len(xy), nz).mean(0) >= 0.5
            k = nz if inside.all() else int(np.argmin(inside))
            heights[i, j] = k / nz * zmax
    return heights


def hull_occupancy(cam, mask, frame, nu, nv, nz, zmax, samples=3):
    """Visual hull sampled on a (nu, nv, nz) grid in an object frame (no ground contiguity).

    A cell is occupied when the majority of its sub-samples project inside the silhouette.
    """
    cx, cy, ln, wd, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    H, W = mask.shape
    sub = (np.arange(samples) + 0.5) / samples - 0.5
    du, dv = np.meshgrid(sub, sub)
    zs = (np.arange(nz) + 0.5) / nz * zmax
    occ = np.zeros((nu, nv, nz), bool)
    for i in range(nu):
        u = ((i + 0.5 + du.ravel()) / nu - 0.5) * ln
        for j in range(nv):
            v = ((j + 0.5 + dv.ravel()) / nv - 0.5) * wd
            xy = np.c_[u, v] @ rot + [cx, cy]
            pts = np.c_[np.repeat(xy, nz, 0), np.tile(zs, len(xy))]
            uv, depth = cam.project(pts)
            px = np.round(uv).astype(int)
            ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            ok[ok] = mask[px[ok, 1], px[ok, 0]]
            occ[i, j] = ok.reshape(len(xy), nz).mean(0) >= 0.5
    return occ


def incircle(occ, v, z, aspect=1.0):
    """Largest disc resting on the ground (centre (vc, r)) inside a 2D occupancy grid occ[v, z].

    With aspect != 1 the "disc" is an ellipse of half-height r and half-width aspect * r.
    """
    V, Z = np.meshgrid(v, z, indexing="ij")
    free = ~occ
    best = (0.0, 0.0)
    for jc in np.nonzero(occ[:, 0])[0]:
        vc = v[jc]
        lo, hi = 0.0, z[-1]
        for _ in range(25):
            r = (lo + hi) / 2
            disc = ((V - vc) / aspect) ** 2 + (Z - r) ** 2 <= r * r
            if not (disc & free).any():
                lo = r
            else:
                hi = r
        if lo > best[1]:
            best = (vc, lo)
    return best


def round_model(cam, mask, box, slices=40, nv=64, nz=64, depth_factor=4.0, height_factor=3.0, aspect=1.0):
    """Generalised cylinder with round cross-sections (for rounded, elongated objects).

    The object's long axis comes from the box fit. In every slice across it, the
    visual hull is a wedge bounded by the ground and the near and far lines of
    sight; a round cross-section resting on the ground and touching the wedge is
    unique: its incircle. Each slice becomes a disc of radius r(u) centred at
    v_c(u), represented by columns spanning the disc (bottom z may be above the
    ground where the disc curves underneath).
    Returns (bottom, top) heights (nu, nv) in world units, the frame and the discs.
    """
    cx, cy, ln, wd, ht, yaw = box
    if wd > ln:  # make u the long axis
        ln, wd, yaw = wd, ln, yaw + math.pi / 2
    span_u, span_v = ln * 1.4, max(wd, ht) * depth_factor
    zmax = max(ht, wd) * height_factor
    occ = hull_occupancy(cam, mask, (cx, cy, span_u, span_v, yaw), slices, nv, nz, zmax)
    v = ((np.arange(nv) + 0.5) / nv - 0.5) * span_v
    z = (np.arange(nz) + 0.5) / nz * zmax
    bottom = np.zeros((slices, nv))
    top = np.zeros((slices, nv))
    fits = np.array([incircle(occ[i], v, z, aspect) for i in range(slices)])  # (vc, r) per slice
    # drop slivers (end slices that only graze the hull) and smooth along the body
    rmed = np.median(fits[fits[:, 1] > 0, 1]) if (fits[:, 1] > 0).any() else 0.0
    valid = fits[:, 1] >= 0.3 * rmed
    for k in (0, 1):
        col = fits[:, k].copy()
        for i in np.nonzero(valid)[0]:
            win = [w for w in range(i - 1, i + 2) if 0 <= w < slices and valid[w]]
            col[i] = np.median(fits[win, k])
        fits[:, k] = col
    discs = []
    for i in np.nonzero(valid)[0]:
        vc, r = fits[i]
        discs.append((i, vc, r))
        d = np.abs(v - vc) / aspect
        inside = d < r
        half = np.sqrt(r * r - d[inside] ** 2)
        bottom[i, inside] = r - half
        top[i, inside] = r + half
    return (bottom, top), (cx, cy, span_u, span_v, yaw), discs


def columns_to_cuboids(heights, frame, nz=128):
    """Merge equal grid columns into cuboids.

    heights: (nu, nv) tops, or a (bottoms, tops) pair, in world units.
    """
    bottoms, tops = heights if isinstance(heights, tuple) else (np.zeros_like(heights), heights)
    cx, cy, ln, wd, yaw = frame
    nu, nv = tops.shape
    zmax = max(tops.max(), 1e-9)
    q = lambda a: np.round(a / zmax * nz).astype(int)
    kb, kt = q(bottoms), q(tops)
    levels = np.where(kt > kb, kb * (nz + 1) + kt, 0)  # encode (bottom, top) as one key
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    cuboids = []
    for i0, j0, i1, j1, key in merge_levels(levels):
        b, t = divmod(int(key), nz + 1)
        u0, u1 = i0 / nu - 0.5, i1 / nu - 0.5
        v0, v1 = j0 / nv - 0.5, j1 / nv - 0.5
        quad = np.array([[u0 * ln, v0 * wd], [u1 * ln, v0 * wd], [u1 * ln, v1 * wd], [u0 * ln, v1 * wd]])
        xy = quad @ rot + [cx, cy]
        cuboids.append({"bottom": np.c_[xy, np.full(4, b / nz * zmax)].tolist(),
                        "height": float((t - b) / nz * zmax)})
    return cuboids


def merge_levels(levels):
    """Greedily merge grid cells of equal height into maximal rectangles.

    Yields (i0, j0, i1, j1, level) with half-open cell ranges; empty cells are skipped.
    """
    nx, ny = levels.shape
    used = levels == 0
    for i in range(nx):
        for j in range(ny):
            if used[i, j]:
                continue
            k = levels[i, j]
            j1 = j
            while j1 < ny and not used[i, j1] and levels[i, j1] == k:
                j1 += 1
            i1 = i + 1
            while i1 < nx and all(not used[i1, jj] and levels[i1, jj] == k for jj in range(j, j1)):
                i1 += 1
            used[i:i1, j:j1] = True
            yield i, j, i1, j1, k


def colorize(cam, cuboids, image_bgr, mask=None):
    """Colour each cuboid by the photo pixels its top face projects to (object pixels only)."""
    H, W = image_bgr.shape[:2]
    if mask is not None and len(cuboids) == 1:  # one solid: its colour is the object's overall colour
        bgr = np.median(image_bgr[mask], axis=0)
        cuboids[0]["color"] = [float(bgr[2]) / 255, float(bgr[1]) / 255, float(bgr[0]) / 255]
        return cuboids
    if mask is not None:
        # for every pixel, the coordinates of the nearest object pixel
        _, labels = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(mask)
        lut = np.zeros((labels.max() + 1, 2), int)
        lut[labels[ys, xs]] = np.c_[xs, ys]
    for cub in cuboids:
        b = np.asarray(cub["bottom"])
        top = b.mean(0) + [0, 0, cub["height"]]
        uv, _ = cam.project(top)
        x, y = np.clip(np.round(uv[0]).astype(int), [0, 0], [W - 1, H - 1])
        if mask is not None and not mask[y, x]:
            x, y = lut[labels[y, x]]
        patch = image_bgr[max(0, y - 6): y + 7, max(0, x - 6): x + 7].reshape(-1, 3)
        if mask is not None:
            pm = mask[max(0, y - 6): y + 7, max(0, x - 6): x + 7].reshape(-1)
            patch = patch[pm] if pm.any() else patch
        bgr = np.median(patch, axis=0)
        cub["color"] = [float(bgr[2]) / 255, float(bgr[1]) / 255, float(bgr[0]) / 255]
    return cuboids


# A box explains its own silhouette almost perfectly (real box 0.983, synthetic box 0.983);
# rounded objects leave the box's corners empty (lying cylinder 0.967, pig 0.87-0.88).
BOX_LIKE_IOU = 0.975


def lift(image_path, reference_path=None, grid=40, fit_yaw=True, symmetry="bilateral", shape="auto", aspect=1.0):
    """Lift one photo. shape: "box" (one solid box, edge-refined), "free" (carved
    columns with a symmetry prior) or "auto" (box if the box fit explains the silhouette)."""
    reference_path = reference_path or os.path.join(REPO, "reference.JPG")
    mask = object_mask(reference_path, image_path)
    image = cv2.imread(image_path)
    return lift_mask(default_camera(), mask, image, grid=grid, fit_yaw=fit_yaw, symmetry=symmetry, shape=shape,
                     aspect=aspect)


def default_camera():
    vps = vanishing_points()
    return Camera.from_vanishing_points(vps["left"], vps["right"], vps["vertical"])


def lift_mask(cam, mask, image=None, grid=40, fit_yaw=True, symmetry="bilateral", shape="auto", aspect=1.0):
    """Lift a silhouette (bool HxW) seen by `cam` into cuboids; `image` (BGR) is optional."""
    box, box_iou = fit_bounding_box(cam, mask, fit_yaw=fit_yaw)
    if shape == "auto":
        shape = "box" if box_iou >= BOX_LIKE_IOU else "round"
    if shape == "box":
        if image is not None:
            box = refine_box_with_edges(cam, image, mask, box)
        bc = box_corners(*box)
        cuboids = [{"bottom": bc[:4].tolist(), "height": float(box[4])}]
    elif shape == "round":
        heights, frame, _ = round_model(cam, mask, box, aspect=aspect)
        cuboids = columns_to_cuboids(heights, frame)
    else:
        cuboids = carve(cam, mask, box, grid=grid, symmetry=symmetry)
    if image is not None:
        colorize(cam, cuboids, image, mask)
    sil = render_silhouette(cam, cuboids, mask.shape)
    scores = silhouette_scores(sil, mask)
    return {"camera": cam, "mask": mask, "box": box, "box_iou": box_iou, "shape": shape, "cuboids": cuboids,
            "silhouette": sil, "scores": scores, "image": image}


def model_extent(cuboids, yaw):
    """Length, width, height of the union of cuboids, measured in the object's own frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.vstack([np.vstack([np.asarray(q["bottom"]), np.asarray(q["bottom"]) + [0, 0, q["height"]]])
                     for q in cuboids])
    u = c * pts[:, 0] + s * pts[:, 1]
    v = -s * pts[:, 0] + c * pts[:, 1]
    return float(np.ptp(u)), float(np.ptp(v)), float(pts[:, 2].max())


def overlay(result):
    """Photo with the cuboids' wireframes reprojected onto it (green) and the silhouette outline."""
    img = result["image"].copy()
    cam = result["camera"]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for cub in result["cuboids"]:
        b = np.asarray(cub["bottom"])
        uv, _ = cam.project(np.vstack([b, b + [0, 0, cub["height"]]]))
        for a, c in edges:
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[c]).astype(int)),
                     (60, 220, 60), 2, cv2.LINE_AA)
    cs, _ = cv2.findContours(result["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (0, 0, 255), 4)
    return img


def export(result, path):
    data = {"z_up": True, "units": "camera heights",
            "box": dict(zip(["cx", "cy", "length", "width", "height", "yaw"], map(float, result["box"]))),
            "scores": {k: float(v) for k, v in result["scores"].items()},
            "cuboids": result["cuboids"]}
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--reference", default=os.path.join(REPO, "reference.JPG"))
    ap.add_argument("--out", default=".")
    ap.add_argument("--grid", type=int, default=40, help="columns along the longer side of the box")
    ap.add_argument("--symmetry", choices=["none", "bilateral", "both"], default="bilateral",
                    help="reflection prior for the hidden side (default: bilateral)")
    ap.add_argument("--shape", choices=["auto", "box", "round", "free"], default="auto",
                    help="box: one edge-refined solid box; round: generalised cylinder with round "
                         "cross-sections; free: carved columns with a symmetry prior (default: auto = "
                         "box if one box explains the silhouette, else round)")
    ap.add_argument("--aspect", type=float, default=1.0,
                    help="round: cross-section half-width / half-height (not observable from one view)")
    ap.add_argument("--no-yaw", action="store_true", help="keep the box aligned with the VP axes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in args.images:
        r = lift(path, args.reference, grid=args.grid, fit_yaw=not args.no_yaw, symmetry=args.symmetry,
                 shape=args.shape, aspect=args.aspect)
        stem = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0])
        export(r, stem + "_cuboids.json")
        cv2.imwrite(stem + "_reprojection.png", overlay(r))
        ln, wd, ht = model_extent(r["cuboids"], r["box"][5])
        s = r["scores"]
        print(f"{path}: [{r['shape']}] model {ln:.3f} x {wd:.3f} x {ht:.3f} camera heights "
              f"(box-fit IoU {r['box_iou']:.3f}); {len(r['cuboids'])} cuboids; reprojection IoU {s['iou']:.3f} "
              f"(precision {s['precision']:.3f}, recall {s['recall']:.3f})")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 3D verification
# ---------------------------------------------------------------------------
def box_ground_truth(cam, lines=VP_LINES):
    """3D box measured from the hand-traced box edges in surf.py (independent of the mask).

    The six traced edges meet at six visible corners. Bottom corners are back-projected
    onto the ground; the vertical-edge heights come from lifting each top corner above
    its bottom corner. The traced top edges give a second estimate of length and width
    (top corners back-projected onto the plane z = H). The physical box is slightly
    dented/bulged and the tracing is manual, so the two disagree by a few percent; the
    returned box uses their mean and `uncertainty` holds half their difference.
    """
    v1, v2 = lines["vertical"]
    l_bot, l_top = lines["left"]
    r_bot, r_top = lines["right"]
    img = {
        "front_bottom": line_intersection(l_bot, r_bot), "left_bottom": line_intersection(l_bot, v1),
        "right_bottom": line_intersection(r_bot, v2), "front_top": line_intersection(l_top, r_top),
        "left_top": line_intersection(l_top, v1), "right_top": line_intersection(r_top, v2),
    }
    bottom = {k: cam.backproject_to_plane(img[k + "_bottom"], 0.0)[0] for k in ("front", "left", "right")}

    def lift_height(base, uv):
        hs = np.linspace(0, 2, 20001)
        proj, _ = cam.project(np.c_[np.tile(base[:2], (len(hs), 1)), hs])
        return hs[np.argmin(np.linalg.norm(proj - uv, axis=1))]

    heights = {k: lift_height(bottom[k], img[k + "_top"]) for k in bottom}
    h = float(np.mean(list(heights.values())))
    top = {k: cam.backproject_to_plane(img[k + "_top"], h)[0] for k in ("front", "left", "right")}
    L = [np.linalg.norm(bottom["left"] - bottom["front"]), np.linalg.norm((top["left"] - top["front"])[:2])]
    W = [np.linalg.norm(bottom["right"] - bottom["front"]), np.linalg.norm((top["right"] - top["front"])[:2])]
    front = (bottom["front"][:2] + top["front"][:2]) / 2
    ex = (bottom["left"] - bottom["front"])[:2]
    ex /= np.linalg.norm(ex)
    ey = (bottom["right"] - bottom["front"])[:2]
    ey /= np.linalg.norm(ey)
    length, width = float(np.mean(L)), float(np.mean(W))
    b = np.array([front, front + ex * length, front + ex * length + ey * width, front + ey * width])
    b = np.c_[b, np.zeros(4)]
    return {"corners": np.vstack([b, b + [0, 0, h]]), "heights": heights, "dims": (length, width, h),
            "uncertainty": (abs(L[0] - L[1]) / 2, abs(W[0] - W[1]) / 2,
                            (max(heights.values()) - min(heights.values())) / 2), "image": img}


def _inside_prism(points, bottom_xy, z0, z1):
    """Points inside a convex vertical prism (bottom polygon, any winding, z range)."""
    p = points[:, :2]
    q = np.asarray(bottom_xy)[:, :2]
    signs = []
    for k in range(len(q)):
        a, b = q[k], q[(k + 1) % len(q)]
        signs.append((b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0]))
    signs = np.array(signs)
    inside2d = (signs >= 0).all(0) | (signs <= 0).all(0)
    return inside2d & (points[:, 2] >= z0) & (points[:, 2] <= z1)


def volume_iou(cuboids, gt_corners, n=400_000, seed=0):
    """Monte-Carlo volumetric IoU between the union of cuboids and a ground-truth box."""
    rng = np.random.default_rng(seed)
    allpts = np.vstack([gt_corners] + [np.vstack([np.asarray(c["bottom"]),
                                                  np.asarray(c["bottom"]) + [0, 0, c["height"]]])
                                       for c in cuboids])
    lo, hi = allpts.min(0), allpts.max(0)
    pts = rng.uniform(lo, hi, size=(n, 3))
    in_model = np.zeros(n, bool)
    for c in cuboids:
        z0 = float(np.asarray(c["bottom"])[:, 2].min())
        in_model |= _inside_prism(pts, c["bottom"], z0, z0 + c["height"])
    in_gt = _inside_prism(pts, gt_corners[:4], gt_corners[:4, 2].min(), gt_corners[4:, 2].max())
    inter, union = (in_model & in_gt).sum(), (in_model | in_gt).sum()
    return {"volume_iou": inter / max(union, 1), "volume_precision": inter / max(in_model.sum(), 1),
            "volume_recall": inter / max(in_gt.sum(), 1)}
