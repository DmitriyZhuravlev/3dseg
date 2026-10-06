"""Reproduce the video -> 3D results for the seven test videos (see NOTES.md section 7).

    python video_demos.py bike  --video bike.mp4  --out out/bike     # rigid object, fixed camera (~40 min)
    python video_demos.py horse --video Horse.mp4 --out out/horse    # non-rigid animal, panning camera
    python video_demos.py yeop  --video Yeop.mp4  --out out/yeop     # person, static camera, black studio
    python video_demos.py air|monument|bender|por --video X.mp4 --out out/x   # single reference frame

Every command writes <name>_model.ply (coloured mesh), <name>_metrics.json and
<name>_sheet.png. The per-video settings below (reference frames, seed rectangles,
the aircraft outline) are the ones used for the published results.
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import video3d as v3

# ---------------------------------------------------------------------------
# Per-video settings
# ---------------------------------------------------------------------------
BIKE = dict(
    scale=0.5,                     # work at 960x540
    focal_half=740.0,              # px at half resolution: makes the parking stalls 5.49 m (18 ft) deep
    stall_points_full=((682.0, 605.0), (1171.9, 605.0)),   # two stall lines on the cross line, full-res px
    stall_width=2.74,              # m (9 ft standard stall)
)
HORSE = dict(ref_frame=1140, every=5, back_height=1.55, turning_until_s=25.0, upper=0.6,
             thickness_grid=(0.35, 0.6, 0.8, 1.0, 1.3, 1.7))
YEOP = dict(ref_frame=40, body_height=1.65, upper=0.45, thickness_grid=(0.7, 1.0, 1.4, 1.8, 2.3))
SINGLE = {
    "air": dict(frame=30),
    "monument": dict(frame=0, rect=(300, 540, 190, 225)),
    "bender": dict(frame=20, rect=(225, 3, 300, 347)),
    "por": dict(frame=10, rect=(80, 5, 222, 234)),
}
# hand-traced outline of the aircraft in Air.mp4 frame 30 (848x464); GrabCut snaps it to the edges.
# Automatic segmentation fails there: the hazy sky has the paint's colour and the crowd hides the gear.
AIR_OUTLINE = dict(
    polygons=[[(55, 78), (180, 57), (205, 57), (280, 88), (205, 86), (200, 112), (180, 112), (170, 86)],
              [(150, 115), (260, 103), (330, 98), (395, 86), (450, 85), (480, 95), (500, 120), (505, 160),
               (495, 205), (440, 212), (300, 215), (190, 215), (150, 200), (140, 150)],
              [(260, 125), (140, 130), (0, 148), (0, 190), (120, 175), (150, 195), (250, 170)]],
    circles=[(175, 162, 42), (10, 185, 30), (538, 172, 38)],
)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def largest_blob(m, close=9, fill=True):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros(m.shape, bool)
    out = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    if fill:
        cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def grabcut(img, rect=None, init=None, fg=None, bg=None, iters=8):
    """GrabCut from a rectangle or an initial probable-foreground mask, with optional seeds."""
    m = np.full(img.shape[:2], cv2.GC_BGD, np.uint8)
    if rect is not None:
        x, y, w, h = rect
        m[y:y + h, x:x + w] = cv2.GC_PR_FGD
    if init is not None:
        m[cv2.dilate(init.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = cv2.GC_PR_BGD
        m[init] = cv2.GC_PR_FGD
        m[cv2.erode(init.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0] = cv2.GC_FGD
    if bg is not None:
        m[bg] = cv2.GC_BGD
    if fg is not None:
        m[fg] = cv2.GC_FGD
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, m, None, bgm, fgm, iters, cv2.GC_INIT_WITH_MASK)
    return largest_blob((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), close=5)


def horse_mask(f):
    """Black horse on grass: dark, non-green seed -> GrabCut -> drop green shadow -> strip ground shadow."""
    V = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2].astype(int)
    b, g, r = [f[..., k].astype(int) for k in range(3)]
    dark = (V < 75) & (g - np.maximum(r, b) < 18)
    seed = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    seed[: int(0.06 * len(seed))] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(seed)
    if n < 2:
        return np.zeros(seed.shape, bool)
    seed = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    gc = np.where(seed > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    gc[cv2.erode(seed, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    gc[(cv2.dilate(seed, np.ones((15, 15), np.uint8)) > 0) & (seed == 0)] = cv2.GC_PR_BGD
    gc[(seed > 0) & (V < 40) & (g - np.maximum(r, b) < 4)] = cv2.GC_FGD
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(f, gc, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
    out = largest_blob((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), close=3, fill=False)
    ys = np.nonzero(out.any(1))[0]
    if len(ys):
        band = np.zeros_like(out)
        band[ys.max() - int(0.35 * (ys.max() - ys.min())):] = True
        out &= ~(band & ((g - np.maximum(r, b)) > 9))
        out = largest_blob(out, close=1, fill=False)
    return v3.strip_ground_shadow(out)


def studio_masks(frames):
    """Static camera, black backdrop, yellow mat (Yeop.mp4)."""
    bg = np.median(frames[::3], 0).astype(np.int16)
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        d = np.abs(f.astype(np.int16) - bg)
        m = (d.max(2) > 30) | ((d.sum(2) > 30) & (f[..., 0] > f[..., 2] + 6))   # dark-blue trousers on black
        m &= ~((hsv[..., 0] > 18) & (hsv[..., 0] < 40) & (hsv[..., 1] > 120))  # yellow mat
        m[int(0.86 * len(m)):] = False
        m[:60] = False
        out.append(largest_blob(m))
    return np.stack(out)


def read_all(path, every=1, size=None):
    cap = cv2.VideoCapture(path)
    frames, ids, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size else f)
            ids.append(i)
        i += 1
    return np.stack(frames), np.array(ids), cap.get(cv2.CAP_PROP_FPS) or 25.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _to_y(a):
    return np.column_stack([a[:, 0], a[:, 2], -a[:, 1]])


def model_views(verts, faces, normals, cols, yaws=(0.0, 0.8, 1.57, 2.6), size=(360, 300), zoom=0.55, pitch=0.2):
    from viewer3d import geometry as geo
    from viewer3d.app import run_headless
    from viewer3d.geometry import Geometry
    from viewer3d.scene import Material, MeshNode, Scene
    g = Geometry(_to_y(verts), _to_y(normals), np.zeros((len(verts), 2)), faces.reshape(-1), colors=cols)
    ext = float(np.ptp(verts, 0).max())
    tiles = []
    for yaw in yaws:
        sc = Scene()
        sc.background = (0.83, 0.85, 0.89)
        sc.add(MeshNode(g, Material(color=(1, 1, 1), shininess=24, specular=0.2)))
        sc.add(MeshNode(geo.plane(2.5 * ext), Material(color=(0.5, 0.5, 0.52)), cast_shadow=False))
        sc.ambient.intensity = 0.8
        img, _ = run_headless(sc, None, size, frames=1, camera={"yaw": yaw, "pitch": pitch, "zoom": zoom, "auto_rotate": 0})
        tiles.append(img)
    return tiles


def sheet(path, title, frame_bgr, mask, mesh, extra_tiles=(), footer=""):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    ys, xs = np.nonzero(mask)
    crop = frame_bgr[max(0, ys.min() - 20):ys.max() + 20, max(0, xs.min() - 20):xs.max() + 20]
    ph = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ph.thumbnail((360, 300))
    t0 = Image.new("RGB", (360, 300), (30, 32, 38))
    t0.paste(ph, ((360 - ph.width) // 2, (300 - ph.height) // 2))
    tiles = [t0] + model_views(*mesh)
    rows = [tiles] + ([list(extra_tiles)] if extra_tiles else [])
    W = 360 * max(len(r) for r in rows)
    out = Image.new("RGB", (W, 34 + 300 * len(rows) + (30 if footer else 0)), (40, 42, 50))
    d = ImageDraw.Draw(out)
    d.text((10, 7), title, fill="white", font=font)
    for r, row in enumerate(rows):
        for k, t in enumerate(row):
            out.paste(t, (360 * k, 34 + 300 * r))
    if footer:
        d.text((10, 34 + 300 * len(rows) + 6), footer, fill=(230, 230, 230), font=small)
    out.save(path)


def contour_tile(frame_bgr, mask, model_sil, text):
    from PIL import Image, ImageDraw, ImageFont
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    c = frame_bgr.copy()
    for msk, col in ((mask, (0, 0, 255)), (model_sil, (0, 255, 0))):
        cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c, cs, -1, col, 2)
    ys, xs = np.nonzero(mask | model_sil)
    c = c[max(0, ys.min() - 20):ys.max() + 20, max(0, xs.min() - 20):xs.max() + 20]
    im = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    im.thumbnail((360, 300))
    t = Image.new("RGB", (360, 300), (30, 32, 38))
    t.paste(im, ((360 - im.width) // 2, (300 - im.height) // 2))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, d.textlength(text, font=small) + 10, 20], fill=(0, 0, 0))
    d.text((5, 2), text, fill="white", font=small)
    return t


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def run_bike(video, out, log=print):
    t0 = time.time()
    s = BIKE["scale"]
    frames, idx, fps = v3.read_frames(video, scale=s, step=1)
    full_bg = None
    cap = cv2.VideoCapture(video)
    sel = np.linspace(0, len(frames) - 2, 120).astype(int)
    stack = []
    for i in sel:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            stack.append(f)
    full_bg = np.median(np.stack(stack), 0).astype(np.uint8)
    bg = cv2.resize(full_bg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    vp_full, _ = v3.forward_vanishing_point(full_bg)
    H, W = bg.shape[:2]
    cam = v3.ground_camera(BIKE["focal_half"], (W, H), vp_full * s)
    p1, p2 = (np.array(p) * s for p in BIKE["stall_points_full"])
    cam = v3.scale_camera_to(cam, p1, p2, BIKE["stall_width"])
    log(f"camera: VP {np.round(vp_full, 1)} (full res), height {cam.C[2]:.2f} m  [{time.time() - t0:.0f}s]")

    # stage 1: filled silhouettes, every 2nd frame -> robust poses
    step = 2
    Mf = v3.object_masks(frames[::step], bg)
    T = len(Mf)
    train = list(range(0, T, 2))
    poses, _ = v3.initial_poses(cam, Mf, fps, step=step)
    grid = v3.VoxelGrid((-1.3, -0.7, 0.0), (1.3, 0.7, 1.9), 0.03)
    for it in range(3):
        frac, seen = v3.carve(cam, Mf, poses, grid, train, margin_px=1)
        pts = v3.surface_points((frac >= 0.8) & (seen >= 20), grid)
        poses = v3.smooth_poses(v3.refine_poses(cam, Mf, poses, pts, grid.size, range(T)), range(T), sigma=1.0)
        log(f"pose round {it + 1}/3  [{time.time() - t0:.0f}s]")
    for cons in (0.85, 0.9, 0.93, 0.95):
        frac, seen = v3.carve(cam, Mf, poses, grid, train, margin_px=1)
        pts = v3.surface_points((frac >= cons) & (seen >= 20), grid)
        poses = v3.smooth_poses(v3.refine_poses_chamfer(cam, Mf, poses, pts, grid.size, range(T)), range(T), sigma=1.0)
        log(f"chamfer round consensus {cons}  [{time.time() - t0:.0f}s]")

    # stage 2: shadow-free silhouettes that keep see-through gaps (spokes, frame) -> fine poses
    def fine_masks(fr):
        M = v3.object_masks(fr, bg, threshold=28, fill_holes=False, close=3, keep_near=6)
        return np.stack([v3.strip_ground_shadow(m, low_frac=0.10, min_run_frac=0.05) if m.any() else m for m in M])

    M2 = fine_masks(frames[::step])
    grid = v3.VoxelGrid((-1.3, -0.7, 0.0), (1.3, 0.7, 1.9), 0.025)
    for cons in (0.9, 0.93, 0.95):
        frac, seen = v3.carve(cam, M2, poses, grid, train, margin_px=1)
        pts = v3.surface_points((frac >= cons) & (seen >= 20), grid)
        poses = v3.smooth_poses(v3.refine_poses_chamfer(cam, M2, poses, pts, grid.size, range(T),
                                                        bounds=(0.15, 0.15, 0.2, 0.2)), range(T), sigma=1.0)
        log(f"fine round consensus {cons}  [{time.time() - t0:.0f}s]")

    # stage 3: every frame, 2 cm voxels; even frames carve, odd frames are held out
    M = fine_masks(frames)
    P = np.stack([np.interp(np.arange(len(M)), np.arange(T) * step, poses[:, k]) for k in range(4)], 1)
    grid = v3.VoxelGrid((-1.25, -0.6, 0.0), (1.25, 0.6, 1.75), 0.02)
    train, test = list(range(0, len(M), 2)), list(range(1, len(M), 2))
    frac, seen = v3.carve(cam, M, P, grid, train, margin_px=1)
    occ = (frac >= 0.93) & (seen >= 40)
    pts = v3.surface_points(occ, grid)
    ious = [v3.iou(v3.silhouette_of(cam, P[t], pts, M.shape[1:], grid.size), M[t]) for t in test[::4]]
    verts, faces, normals = v3.occupancy_mesh(occ, grid)
    cols = v3.vertex_colours(cam, frames, M, P, verts, normals, range(0, len(M), 3))
    v3.save_ply(os.path.join(out, "bike_model.ply"), verts, faces, cols)
    np.savez_compressed(os.path.join(out, "bike_solution.npz"), poses=P, occ=occ, K=cam.K, R=cam.R, C=cam.C,
                        lo=grid.lo, hi=grid.hi, size=grid.size)
    metrics = dict(held_out_iou_mean=float(np.mean(ious)), held_out_iou_median=float(np.median(ious)),
                   extent_m=np.ptp(verts, 0).round(3).tolist(), camera_height_m=float(cam.C[2]),
                   frames=len(M), carved_with=len(train), held_out=len(test[::4]), seconds=round(time.time() - t0))
    json.dump(metrics, open(os.path.join(out, "bike_metrics.json"), "w"), indent=1)
    sheet(os.path.join(out, "bike_sheet.png"), "bike.mp4: rigid motorcycle, fixed camera, carved from %d frames" % len(train),
          frames[470], M[470], (verts, faces, normals, cols),
          footer=f"held-out IoU {metrics['held_out_iou_mean']:.3f}; size {metrics['extent_m']} m (with rider)")
    log(json.dumps(metrics))
    return metrics


def calibrate_inflation(model_mask, model_img, mpp, masks, calib, held, grid_th, upper, tol=0.005):
    res = {}
    for th in grid_th:
        P = v3.InflatedModel(model_mask, model_img, mpp, thickness=th).surface_points()
        res[th] = float(np.mean([v3.fit_weak_view(P, masks[t], upper=upper)[1] for t in calib]))
    # The score is often flat across thicknesses (the views barely constrain depth): among
    # the values within `tol` of the best, prefer the one closest to round (1.0).
    top = max(res.values())
    best = min((th for th in res if res[th] >= top - tol), key=lambda th: abs(math.log(th)))
    P = v3.InflatedModel(model_mask, model_img, mpp, thickness=best).surface_points()
    Pf = v3.InflatedModel(model_mask, model_img, mpp, thickness=0.05).surface_points()
    fits = [v3.fit_weak_view(P, masks[t], upper=upper) for t in held]
    return best, res, fits, float(np.mean([v3.fit_weak_view(Pf, masks[t], upper=upper)[1] for t in held]))


def run_inflated(name, frames, masks, ref, mpp, calib, held, grid_th, upper, out, title, unit="m"):
    t0 = time.time()
    if calib:
        best, res, fits, flat = calibrate_inflation(masks[ref], frames[ref], mpp, masks, calib, held, grid_th, upper)
    else:
        best, res, fits, flat = 1.0, {}, [], None
    model = v3.InflatedModel(masks[ref], frames[ref], mpp, thickness=best)
    verts, faces, normals, cols = model.mesh()
    v3.save_ply(os.path.join(out, f"{name}_model.ply"), verts, faces, cols)
    Ps = model.surface_points()
    tiles = []
    for t, (p, s) in list(zip(held, fits))[:: max(1, len(held) // 4)][:4]:
        tiles.append(contour_tile(frames[t], masks[t], v3.weak_silhouette(Ps, p, masks[t].shape), f"held-out frame {t}: IoU {s:.2f}"))
    metrics = dict(reference_frame=int(ref), thickness=best, calibration=res, extent=np.ptp(verts, 0).round(3).tolist(), unit=unit,
                   held_out_iou=float(np.mean([s for _, s in fits])) if fits else None, held_out_iou_flat_cutout=flat,
                   n_calib=len(calib), n_held=len(held), seconds=round(time.time() - t0))
    json.dump(metrics, open(os.path.join(out, f"{name}_metrics.json"), "w"), indent=1)
    foot = (f"held-out IoU {metrics['held_out_iou']:.3f} vs flat cut-out {flat:.3f}; " if fits else "single view: thickness assumed round; ") + \
           f"size {metrics['extent']} {unit}"
    sheet(os.path.join(out, f"{name}_sheet.png"), title, frames[ref], masks[ref], (verts, faces, normals, cols), tiles, foot)
    return metrics


def run_horse(video, out, log=print):
    frames, ids, fps = read_all(video, every=HORSE["every"], size=(960, 540))
    masks = np.stack([horse_mask(f) for f in frames])
    ref = int(np.argmin(np.abs(ids - HORSE["ref_frame"])))
    ys, xs = np.nonzero(masks[ref])
    x0, x1 = xs.min(), xs.max()
    mid = (xs > x0 + 0.35 * (x1 - x0)) & (xs < x0 + 0.65 * (x1 - x0))
    back_px = ys.max() - np.median([ys[xs == x].min() for x in np.unique(xs[mid])])
    mpp = HORSE["back_height"] / back_px
    area = masks.reshape(len(masks), -1).sum(1)
    inside = np.array([m.any() and np.nonzero(m.any(0))[0].min() > 5 and np.nonzero(m.any(0))[0].max() < 955 for m in masks])
    turn = list(np.nonzero(inside & (ids / fps < HORSE["turning_until_s"]) & (area > 2000))[0])
    turn = turn[:: max(1, len(turn) // 24)]
    m = run_inflated("horse", frames, masks, ref, mpp, turn[0::2], turn[1::2], HORSE["thickness_grid"], HORSE["upper"], out,
                     "Horse.mp4: side silhouette inflated; thickness fitted on frames where the horse turns")
    log(json.dumps(m))
    return m


def run_yeop(video, out, log=print):
    frames, ids, fps = read_all(video)
    masks = studio_masks(frames)
    ref = YEOP["ref_frame"]
    ys, xs = np.nonzero(masks[ref])
    mpp = YEOP["body_height"] / (ys.max() - ys.min())
    area = masks.reshape(len(masks), -1).sum(1)
    cand = [t for t in range(10, len(masks) - 5) if area[t] > 0.5 * area[ref] and abs(t - ref) > 8]
    m = run_inflated("yeop", frames, masks, ref, mpp, cand[0::6], cand[3::6], YEOP["thickness_grid"], YEOP["upper"], out,
                     "Yeop.mp4: performer inflated; thickness fitted on frames where she turns")
    log(json.dumps(m))
    return m


def run_single(name, video, out, log=print):
    cfg = SINGLE[name]
    frames, ids, fps = read_all(video)
    f = frames[cfg["frame"]]
    if name == "air":
        init = np.zeros(f.shape[:2], np.uint8)
        for p in AIR_OUTLINE["polygons"]:
            cv2.fillPoly(init, [np.array(p, np.int32)], 1)
        for x, y, r in AIR_OUTLINE["circles"]:
            cv2.circle(init, (x, y), r, 1, -1)
        mask = grabcut(f, init=init > 0, iters=6)
    elif name == "monument":
        b, g, r = [f[..., k].astype(int) for k in range(3)]
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        x, y, w, h = cfg["rect"]
        core = np.zeros(f.shape[:2], bool)
        core[y:y + h, x:x + w] = (hsv[y:y + h, x:x + w, 2] < 80) & (hsv[y:y + h, x:x + w, 1] < 90)
        sky = (b > r + 25) & (b > 120)
        foliage = (g > r + 8) & (g > b + 8)
        mask = grabcut(f, rect=cfg["rect"], fg=core, bg=sky | foliage)
    elif name == "bender":
        fg = np.zeros(f.shape[:2], bool)
        fg[60:110, 300:420] = fg[150:220, 320:390] = fg[250:330, 300:430] = True
        bg = np.zeros(f.shape[:2], bool)
        bg[:, :215] = True
        mask = grabcut(f, rect=cfg["rect"], fg=fg, bg=bg)
    else:
        mask = grabcut(f, rect=cfg["rect"])
    ys, xs = np.nonzero(mask)
    masks = np.zeros((len(frames),) + mask.shape, bool)
    masks[cfg["frame"]] = mask
    m = run_inflated(name, frames, masks, cfg["frame"], 1.0 / (ys.max() - ys.min()), [], [], (1.0,), None, out,
                     f"{os.path.basename(video)}: single reference frame inflated (round cross-sections assumed)",
                     unit="subject heights")
    log(json.dumps(m))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["bike", "horse", "yeop", "air", "monument", "bender", "por"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.which == "bike":
        run_bike(a.video, a.out)
    elif a.which == "horse":
        run_horse(a.video, a.out)
    elif a.which == "yeop":
        run_yeop(a.video, a.out)
    else:
        run_single(a.which, a.video, a.out)


if __name__ == "__main__":
    main()
