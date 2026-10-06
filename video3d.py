"""3D reconstruction of a rigid object moving on the ground, filmed by a fixed camera.

Extends the single-photo method (lift3d.py) to video. Every frame is another view
of the same object, so the hidden side no longer has to be guessed:

1. Background: per-pixel median over the video (the "reference.JPG" of surf.py,
   computed instead of photographed). Object mask = difference to it, largest blob.
2. Camera: ground-plane geometry from the vanishing point of the painted lines
   (horizon + tilt), metric scale from a known ground distance (e.g. parking-stall
   width). The focal length is the one quantity the lines cannot fix here, so it is
   self-calibrated: the value at which the object carves most consistently.
3. Poses: ground position from the silhouette's ground contact, heading from the
   direction of travel (vehicles move the way they point), lean from the turn
   (tan(lean) = v^2 / (g r)); then each frame's pose (x, y, heading, lean) is refined
   against its silhouette.
4. Shape: one voxel grid in the object's own frame is carved by all silhouettes
   (kept where it projects inside the mask in at least `consensus` of the frames,
   which tolerates segmentation errors and a rider moving slightly).
5. Verification: silhouettes of frames that were never used for carving.

    python video3d.py bike.mp4 --out out/bike
"""
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from lift3d import Camera

G = 9.81


# ---------------------------------------------------------------------------
# Video, background and masks
# ---------------------------------------------------------------------------
def read_frames(path, scale=0.5, step=1, limit=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, idx, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok or (limit and len(frames) >= limit):
            break
        if i % step == 0:
            frames.append(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else f)
            idx.append(i)
        i += 1
    return np.stack(frames), np.array(idx), fps


def median_background(frames, samples=120):
    sel = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    return np.median(frames[sel], axis=0).astype(np.uint8)


def strip_ground_shadow(mask, low_frac=0.25, min_run_frac=0.12):
    """Remove a cast shadow from the bottom of a silhouette.

    In the lowest `low_frac` of the silhouette only pixels belonging to long vertical
    runs survive: legs and wheels reach down from the body, a shadow on the ground is a
    thin horizontal band.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    cut = y1 - int(low_frac * h)
    min_run = max(6, int(min_run_frac * h))
    out = mask.copy()
    for x in np.unique(xs):
        d = np.diff(np.r_[0, mask[:, x].astype(np.int8), 0])
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            if b - 1 > cut and (b - a) < min_run:
                out[max(a, cut):b, x] = False
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(out)
    return lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]) if n > 1 else out > 0


def object_masks(frames, background, threshold=30, min_area=150, fill_holes=True, close=7, keep_near=0):
    """Largest changed blob per frame. Returns bool (T, H, W).

    fill_holes=False keeps see-through gaps (wheel spokes, frame triangles), which is
    what lets carving remove them; keep_near > 0 also keeps smaller blobs within that
    many pixels of the main one (e.g. a wheel separated from the body by a thin gap).
    """
    bg = background.astype(np.int16)
    out = np.zeros(frames.shape[:3], bool)
    for t, f in enumerate(frames):
        m = (np.abs(f.astype(np.int16) - bg).max(2) > threshold).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if close:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        blob = (lab == k).astype(np.uint8)
        if keep_near:
            near = cv2.dilate(blob, np.ones((2 * keep_near + 1,) * 2, np.uint8)) > 0
            for j in range(1, n):
                if j != k and st[j, cv2.CC_STAT_AREA] >= 20 and near[lab == j].any():
                    blob[lab == j] = 1
        if fill_holes:
            cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blob, cs, -1, 1, -1)
        out[t] = blob > 0
    return out


# ---------------------------------------------------------------------------
# Camera from the ground's vanishing point
# ---------------------------------------------------------------------------
def ground_camera(f, image_size, vp_forward, roll=0.0, height=1.0):
    """Camera above the ground plane z=0 looking towards the 'forward' vanishing point.

    World: X right, Y forward (towards vp_forward), Z up; camera at (0, 0, height).
    Principal point at the image centre. Returns lift3d.Camera.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    d = np.linalg.inv(K) @ np.array([vp_forward[0], vp_forward[1], 1.0])
    yfwd = d / np.linalg.norm(d)                        # world +Y in camera coords
    # world up is perpendicular to the forward direction and (with zero roll) lies in
    # the camera's vertical plane: project camera -y onto the plane orthogonal to yfwd
    up = np.array([-math.sin(roll), -math.cos(roll), 0.0])
    zup = up - yfwd * (up @ yfwd)
    zup /= np.linalg.norm(zup)
    xr = np.cross(yfwd, zup)
    R = np.column_stack([xr, yfwd, zup])                # columns: world axes in camera frame
    U, _, Vt = np.linalg.svd(R)
    return Camera(K, U @ Vt, [0.0, 0.0, height])


def scale_camera_to(cam, p1_img, p2_img, metres):
    """Re-scale the camera height so that two ground pixels are `metres` apart."""
    a, b = cam.backproject_to_plane(np.array([p1_img, p2_img], float), 0.0)
    s = metres / np.linalg.norm(a[:2] - b[:2])
    return Camera(cam.K, cam.R, cam.C * s)


# ---------------------------------------------------------------------------
# Object poses on the ground
# ---------------------------------------------------------------------------
def pose_matrix(x, y, yaw, lean=0.0):
    """Object -> world: rotate by lean about the object's forward axis, then yaw, then move."""
    cl, sl = math.cos(lean), math.sin(lean)
    Rl = np.array([[1, 0, 0], [0, cl, -sl], [0, sl, cl]])        # roll about +x (forward)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Ry @ Rl
    T[:3, 3] = [x, y, 0.0]
    return T


def initial_poses(cam, masks, fps, step=1, smooth_s=0.25):
    """Ground track from the silhouettes' bottom-centre, heading from motion, lean from the turn."""
    T = len(masks)
    pts = np.full((T, 2), np.nan)
    for t, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            continue
        ybot = ys.max()
        band = xs[ys >= ybot - 3]
        pts[t] = [band.mean(), ybot]
    ok = ~np.isnan(pts[:, 0])
    ground = np.full((T, 3), np.nan)
    ground[ok] = cam.backproject_to_plane(pts[ok], 0.0)
    # interpolate gaps, smooth
    tt = np.arange(T)
    for k in (0, 1):
        ground[:, k] = np.interp(tt, tt[ok], ground[ok, k])
    dt = step / fps
    sig = max(1.0, smooth_s / dt)
    xy = np.stack([gaussian_filter1d(ground[:, k], sig) for k in (0, 1)], 1)
    v = np.gradient(xy, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    speed = np.linalg.norm(v, axis=1)
    yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    curv = (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.maximum(speed, 0.3) ** 3   # signed 1/r
    lean = -np.arctan(speed ** 2 * curv / G)          # lean into the turn
    lean = np.clip(gaussian_filter1d(lean, sig), -0.7, 0.7)
    return np.column_stack([xy, yaw, lean]), speed


# ---------------------------------------------------------------------------
# Voxel carving in the object frame
# ---------------------------------------------------------------------------
class VoxelGrid:
    def __init__(self, lo, hi, size):
        self.lo, self.hi, self.size = np.asarray(lo, float), np.asarray(hi, float), float(size)
        self.shape = tuple(np.ceil((self.hi - self.lo) / size).astype(int))
        g = [self.lo[k] + (np.arange(self.shape[k]) + 0.5) * size for k in range(3)]
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        self.centres = np.stack([X, Y, Z], -1).reshape(-1, 3)


def project_object(cam, pose, pts):
    Tm = pose_matrix(*pose)
    w = pts @ Tm[:3, :3].T + Tm[:3, 3]
    return cam.project(w)


def carve(cam, masks, poses, grid, frames_idx, consensus=0.9, margin_px=1):
    """Fraction of frames (where the voxel projects into the image) whose mask contains it."""
    H, W = masks.shape[1:]
    inside = np.zeros(len(grid.centres))
    seen = np.zeros(len(grid.centres))
    for t in frames_idx:
        m = masks[t]
        if margin_px:
            m = cv2.dilate(m.astype(np.uint8), np.ones((2 * margin_px + 1,) * 2, np.uint8)) > 0
        uv, depth = project_object(cam, poses[t], grid.centres)
        px = np.round(uv).astype(int)
        ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        seen += ok
        inside[ok] += m[px[ok, 1], px[ok, 0]]
    frac = np.where(seen > 0, inside / np.maximum(seen, 1), 0.0)
    return frac.reshape(grid.shape), seen.reshape(grid.shape)


def silhouette_of(cam, pose, occ_pts, shape, voxel, dilate=True):
    """Render the silhouette of occupied voxel centres (splatted at their projected size)."""
    H, W = shape
    uv, depth = project_object(cam, pose, occ_pts)
    ok = depth > 0
    sil = np.zeros((H, W), np.uint8)
    if not ok.any():
        return sil > 0
    px = np.round(uv[ok]).astype(int)
    keep = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    sil[px[keep, 1], px[keep, 0]] = 1
    if dilate:
        r = max(1, int(round(cam.focal * voxel / np.median(depth[ok]) / 2 + 0.5)))
        sil = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return sil > 0


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def surface_points(occ, grid):
    """Centres of occupied voxels with at least one empty 6-neighbour (enough for silhouettes)."""
    o = np.pad(occ, 1)
    interior = o[1:-1, 1:-1, 1:-1].copy()
    for ax in range(3):
        for sh in (-1, 1):
            interior &= np.roll(o, sh, axis=ax)[1:-1, 1:-1, 1:-1]
    return grid.centres[(occ & ~interior).ravel()]


def refine_poses(cam, masks, poses, occ_pts, voxel, frames_idx, bounds=(0.4, 0.4, 0.5, 0.35), pad=60):
    """Per-frame (x, y, yaw, lean) maximising silhouette IoU, starting from `poses`.

    IoU is evaluated in a window around the observed silhouette, which keeps each
    evaluation cheap and still penalises a model that spills outside it.
    """
    out = poses.copy()
    H, W = masks.shape[1:]
    lim = np.array(bounds)
    for t in frames_idx:
        p0 = poses[t].copy()
        ys, xs = np.nonzero(masks[t])
        if len(xs) < 50:
            continue
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        mt = masks[t][y0:y1, x0:x1]
        shift = np.array([x0, y0])

        def loss(d):
            if np.any(np.abs(d) > lim):
                return 1.0
            uv, depth = project_object(cam, p0 + d, occ_pts)
            ok = depth > 0
            sil = np.zeros(mt.shape, np.uint8)
            px = np.round(uv[ok] - shift).astype(int)
            k = (px[:, 0] >= 0) & (px[:, 0] < mt.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < mt.shape[0])
            sil[px[k, 1], px[k, 0]] = 1
            r = max(1, int(round(cam.focal * voxel / np.median(depth[ok]) / 2 + 0.5)))
            sil = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            return 1.0 - iou(sil, mt)

        res = minimize(loss, np.zeros(4), method="Powell",
                       options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 250})
        out[t] = p0 + res.x
    return out


def refine_poses_chamfer(cam, masks, poses, occ_pts, voxel, frames_idx, bounds=(0.3, 0.3, 0.35, 0.3), pad=60, trunc=12.0):
    """Like refine_poses, but minimises a symmetric truncated chamfer distance between the
    model outline and the mask outline (smoother than IoU, so poses converge more precisely)."""
    out = poses.copy()
    H, W = masks.shape[1:]
    lim = np.array(bounds)
    for t in frames_idx:
        p0 = poses[t].copy()
        ys, xs = np.nonzero(masks[t])
        if len(xs) < 50:
            continue
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        mt = masks[t][y0:y1, x0:x1].astype(np.uint8)
        dt_out = np.minimum(cv2.distanceTransform(1 - mt, cv2.DIST_L2, 3), trunc)   # distance to the mask
        dt_in = np.minimum(cv2.distanceTransform(mt, cv2.DIST_L2, 3), trunc)        # depth inside the mask
        shift = np.array([x0, y0])

        def loss(d):
            if np.any(np.abs(d) > lim):
                return 1e3
            uv, depth = project_object(cam, p0 + d, occ_pts)
            ok = depth > 0
            px = np.round(uv[ok] - shift).astype(int)
            k = (px[:, 0] >= 0) & (px[:, 0] < mt.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < mt.shape[0])
            if k.sum() < 10:
                return 1e3
            sil = np.zeros(mt.shape, np.uint8)
            sil[px[k, 1], px[k, 0]] = 1
            r = max(1, int(round(cam.focal * voxel / np.median(depth[ok]) / 2 + 0.5)))
            sil = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            spill = dt_out[sil > 0].mean()                       # model outside the mask
            miss = dt_in[(mt > 0) & (sil == 0)].sum() / max(mt.sum(), 1) * 4   # mask not covered
            return spill + miss + 2.0 * (1 - (k.sum() / len(k)))

        res = minimize(loss, np.zeros(4), method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 300})
        out[t] = p0 + res.x
    return out


def smooth_poses(poses, frames_idx, sigma=1.5):
    p = poses.copy()
    idx = np.array(sorted(frames_idx))
    for k in range(4):
        p[idx, k] = gaussian_filter1d(poses[idx, k], sigma)
    return p


# ---------------------------------------------------------------------------
# Surface, colour, export
# ---------------------------------------------------------------------------
def occupancy_mesh(occ, grid, smooth=0.8):
    """Smooth triangle mesh (object frame, Z up) of the carved occupancy."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    vol = gaussian_filter(np.pad(occ.astype(float), 2), smooth)
    verts, faces, normals, _ = marching_cubes(vol, 0.5)
    verts = (verts - 2 + 0.5) * grid.size + grid.lo
    return verts, faces[:, ::-1].copy(), -normals   # outward winding/normals for viewer3d


def vertex_colours(cam, frames, masks, poses, verts, normals, frame_ids, depth_tol=0.04):
    """Median photo colour of each vertex over the frames where it faces the camera,
    is not hidden behind the model's own surface and lands on the object mask."""
    H, W = masks.shape[1:]
    samples = []
    for t in frame_ids:
        Tm = pose_matrix(*poses[t])
        vw = verts @ Tm[:3, :3].T + Tm[:3, 3]
        nw = normals @ Tm[:3, :3].T
        uv, depth = cam.project(vw)
        px = np.round(uv).astype(int)
        ok = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        view = cam.C - vw
        ok &= np.einsum("ij,ij->i", nw, view) > 0
        zbuf = np.full((H, W), np.inf)
        np.minimum.at(zbuf, (px[ok, 1], px[ok, 0]), depth[ok])
        zb = cv2.erode(zbuf.astype(np.float32), np.ones((3, 3), np.uint8))  # fill splat gaps (min filter)
        vis = ok.copy()
        vis[ok] = (depth[ok] <= zb[px[ok, 1], px[ok, 0]] + depth_tol) & masks[t][px[ok, 1], px[ok, 0]]
        c = frames[t][px[vis, 1], px[vis, 0]][:, ::-1] / 255.0  # BGR -> RGB
        samples.append((np.nonzero(vis)[0], c))
    # median per vertex (memory-light: up to 25 samples each)
    per = [[] for _ in range(len(verts))]
    for ids, c in samples:
        for i, col in zip(ids, c):
            if len(per[i]) < 25:
                per[i].append(col)
    out = np.full((len(verts), 3), 0.5)
    for i, lst in enumerate(per):
        if lst:
            out[i] = np.median(np.array(lst), axis=0)
    have = np.array([len(lst) > 0 for lst in per])
    if have.any() and not have.all():   # unseen vertices take their nearest seen neighbour's colour
        from scipy.spatial import cKDTree
        _, nn = cKDTree(verts[have]).query(verts[~have])
        out[~have] = out[have][nn]
    return out


def mesh_geometry(verts, faces, normals, colours):
    """viewer3d Geometry, converting the Z-up object frame to the viewer's Y-up frame."""
    from viewer3d.geometry import Geometry
    to_y = lambda a: np.column_stack([a[:, 0], a[:, 2], -a[:, 1]])
    return Geometry(to_y(verts), to_y(normals), np.zeros((len(verts), 2)), faces.reshape(-1), colors=colours)


def save_ply(path, verts, faces, colours):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        c8 = np.clip(np.round(colours * 255), 0, 255).astype(int)
        for v, c in zip(verts, c8):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


class CVCamera:
    """Adapter so viewer3d can render through a calibrated OpenCV camera (K, R, C)."""

    def __init__(self, cam, image_size, near=0.05, far=200.0):
        self.cam, self.size, self.near, self.far = cam, image_size, near, far
        self.position = np.asarray(cam.C, np.float32)

    def view_matrix(self):
        V = np.eye(4)
        V[:3, :3] = self.cam.R
        V[:3, 3] = -self.cam.R @ self.cam.C
        return (np.diag([1, -1, -1, 1]) @ V).astype(np.float32)   # OpenCV -> OpenGL axes

    def projection_matrix(self):
        (fx, _, cx), (_, fy, cy), _ = self.cam.K
        w, h = self.size
        n, f = self.near, self.far
        P = np.zeros((4, 4), np.float32)
        P[0, 0], P[0, 2] = 2 * fx / w, 1 - 2 * cx / w
        P[1, 1], P[1, 2] = 2 * fy / h, 2 * cy / h - 1
        P[2, 2], P[2, 3] = (f + n) / (n - f), 2 * f * n / (n - f)
        P[3, 2] = -1
        return P


def render_overlay(ctx, renderer, cam, geom_zup, pose, background_bgr, sun=(-0.4, 0.5, -1.0)):
    """Render the model at `pose` through `cam` and composite it over a video frame."""
    from PIL import Image
    from viewer3d.scene import Material, MeshNode, Scene
    from viewer3d import math3d as m3
    h, w = background_bgr.shape[:2]
    scene = Scene()
    scene.background = (1.0, 0.0, 1.0)
    node = MeshNode(geom_zup, Material(color=(1, 1, 1), shininess=24, specular=0.15))
    scene.add(node)
    scene.sun.direction = m3.normalize(sun).astype(np.float32)
    scene.ambient.intensity = 0.9
    scene.update(0, 0)
    node.world_matrix = pose_matrix(*pose).astype(np.float32)
    fbo = ctx.simple_framebuffer((w, h), components=4)
    renderer.render(scene, CVCamera(cam, (w, h)), fbo)
    img = np.asarray(Image.frombytes("RGB", (w, h), fbo.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    fbo.release()
    key = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)
    out = background_bgr.copy()
    out[~key] = img[~key][:, ::-1]
    return out, ~key


# ---------------------------------------------------------------------------
# Non-rigid subjects: silhouette inflation, thickness calibrated from other views
# ---------------------------------------------------------------------------
def medial_radius(mask, levels=40):
    """For every mask pixel, the radius of the largest inscribed disc that covers it."""
    D = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    R = np.zeros_like(D)
    for r in np.linspace(1, D.max(), levels):
        centres = (D >= r).astype(np.uint8)
        k = int(2 * r + 1)
        cover = cv2.dilate(centres, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        R[cover & mask] = r
    return D, np.maximum(R, D)


def inflate(mask, thickness=1.0):
    """Half-thickness map of a symmetric 'inflated' solid behind a side silhouette.

    Every part is treated as a tube with a round cross-section whose radius is the
    local inscribed-disc radius R; at distance D from the outline the half-depth is
    sqrt(D (2R - D)). `thickness` scales the depth (1 = round cross-sections) and is
    the one number a single silhouette cannot determine.
    """
    D, R = medial_radius(mask)
    return thickness * np.sqrt(np.clip(D * (2 * R - D), 0, None))


class InflatedModel:
    """Solid from a side-view silhouette: x along the image, z up, y = depth (symmetric)."""

    def __init__(self, mask, image_bgr, metres_per_px, thickness=1.0, max_px=360):
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = mask[y0:y1, x0:x1]
        img = image_bgr[y0:y1, x0:x1]
        s = min(1.0, max_px / max(m.shape))
        if s < 1:
            m = cv2.resize(m.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0
            img = cv2.resize(img, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA)
        self.mask, self.image, self.px = m, img, metres_per_px / s
        self.half = inflate(m, 1.0)                     # in pixels, thickness applied later
        self.thickness = thickness
        self.height_px = m.shape[0]

    def voxels(self, thickness=None, depth_res=None):
        """Occupancy grid (X, Y, Z) in metres-sized cells of `self.px`."""
        th = self.thickness if thickness is None else thickness
        half = self.half * th
        nz, nx = self.mask.shape
        ny = int(np.ceil(half.max())) * 2 + 3
        yc = (np.arange(ny) - ny / 2 + 0.5)
        occ = np.abs(yc)[None, :, None] <= half.T[:, None, ::-1]      # (x, y, z) with z up
        return occ, ny

    def surface_points(self, thickness=None, step=2):
        """Dense points on the inflated surface (object frame, metres), for fast projection."""
        th = self.thickness if thickness is None else thickness
        nz, nx = self.mask.shape
        zz, xx = np.nonzero(self.mask[::step, ::step])
        zz, xx = zz * step, xx * step
        h = self.half[zz, xx] * th
        pts = []
        for sgn in (-1.0, 0.0, 1.0):
            pts.append(np.c_[xx, sgn * h, (nz - 1 - zz)])
        P = np.vstack(pts).astype(float) * self.px
        P[:, 0] -= nx / 2 * self.px
        return P

    def mesh(self, thickness=None, smooth=1.0):
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        occ, ny = self.voxels(thickness)
        vol = gaussian_filter(np.pad(occ.astype(float), 2), smooth)
        verts, faces, normals, _ = marching_cubes(vol, 0.5)
        verts = verts - 2
        nz, nx = self.mask.shape
        # colour: sample the photo at (x, z); both sides take the visible side's colour
        xi = np.clip(np.round(verts[:, 0]).astype(int), 0, nx - 1)
        zi = np.clip(nz - 1 - np.round(verts[:, 2]).astype(int), 0, nz - 1)
        cols = self.image[zi, xi][:, ::-1] / 255.0
        P = np.c_[(verts[:, 0] - nx / 2) * self.px, (verts[:, 1] - ny / 2) * self.px, verts[:, 2] * self.px]
        return P, faces[:, ::-1].copy(), -normals, cols


def project_weak(P, yaw, pitch, scale, u, v):
    """Weak-perspective view: rotate about Z by yaw, tilt by pitch, scale to pixels, shift."""
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x = cy_ * P[:, 0] - sy_ * P[:, 1]
    y = sy_ * P[:, 0] + cy_ * P[:, 1]
    cp, sp = math.cos(pitch), math.sin(pitch)
    zz = cp * P[:, 2] - sp * y
    return np.c_[u + scale * x, v - scale * zz]


def weak_silhouette(P, params, shape, dot=2):
    H, W = shape
    uv = np.round(project_weak(P, *params)).astype(int)
    k = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    sil = np.zeros(shape, np.uint8)
    sil[uv[k, 1], uv[k, 0]] = 1
    sil = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dot + 1,) * 2))
    return cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def upper_part(mask, frac):
    """The top `frac` of a silhouette's height (e.g. an animal's torso, neck and head)."""
    ys = np.nonzero(mask.any(1))[0]
    if not len(ys):
        return mask
    out = mask.copy()
    out[ys.min() + int(frac * (ys.max() - ys.min() + 1)):] = False
    return out


def fit_weak_view(P, mask, init_yaw=0.0, pitch=0.1, upper=None):
    """Best (yaw, pitch, scale, u, v) of a model for one observed silhouette (max IoU).

    upper: if set (e.g. 0.6), only the top part of both silhouettes is compared, so
    moving legs do not decide the fit of a non-rigid animal.
    """
    target = upper_part(mask, upper) if upper else mask
    ys, xs = np.nonzero(mask)
    span = np.ptp(P[:, 0]) or 1.0
    s0 = np.ptp(xs) / span
    best = None
    for yaw0 in (init_yaw, init_yaw + 0.6, init_yaw - 0.6, init_yaw + math.pi):
        x0 = np.array([yaw0, pitch, s0, xs.mean(), ys.max() - 0.0])

        def loss(p):
            if p[2] <= 0:
                return 1.0
            sil = weak_silhouette(P, p, mask.shape)
            if upper:
                sil = sil & (np.arange(mask.shape[0])[:, None] < (np.nonzero(target.any(1))[0].max() + 1))
            return 1.0 - iou(sil, target)

        r = minimize(loss, x0, method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 600})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, 1.0 - best.fun


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def painted_line_segments(background_bgr, min_len=60, horizon_frac=0.4):
    """Long bright (painted) segments on the ground part of the image (LSD)."""
    g = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    white = ((g > 170) & (background_bgr[..., 0] > 150)).astype(np.uint8) * 255
    white[: int(len(white) * horizon_frac)] = 0
    segs = cv2.createLineSegmentDetector().detect(white)[0]
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4)
    return segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) >= min_len]


def vanishing_point(segs, iters=500, tol=4.0, seed=0):
    """RANSAC + least-squares vanishing point of a family of image segments."""
    rng = np.random.default_rng(seed)
    L = np.array([np.cross([x1, y1, 1.0], [x2, y2, 1.0]) for x1, y1, x2, y2 in segs])
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(L), 2, replace=False)
        p = np.cross(L[i], L[j])
        if abs(p[2]) < 1e-9:
            continue
        p = p / p[2]
        inl = np.abs(L @ p) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    _, _, Vt = np.linalg.svd(L[best])
    p = Vt[-1]
    return p[:2] / p[2], best


def forward_vanishing_point(background_bgr):
    """VP of the painted lines that run away from the camera (not the near-horizontal ones)."""
    segs = painted_line_segments(background_bgr)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    fwd = segs[(ang > 12) & (ang < 168)]
    return vanishing_point(fwd)
