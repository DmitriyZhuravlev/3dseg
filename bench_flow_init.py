"""Benchmark: optical-flow heading + footprint centre (SpringerLifting article) vs the
track-based initial poses of video3d.initial_poses, on rendered video with known poses.

A textured box (1.6 x 0.6 x 0.9 m) drives on a textured ground in front of the fixed
camera of tests/test_video3d.py. Three trajectories:
  arc   - constant speed half circle
  start - pulls away from standstill (slow first second), then turns
  s     - S-curve with a speed dip
For each: heading and position error of the initial poses, then the box carved from
those poses alone (no refinement): volume IoU with the true box and silhouette IoU on
held-out frames.

    python bench_flow_init.py            # writes docs/video3d/flow_init_benchmark.{json,png}
"""
import json
import math
import os

import cv2
import numpy as np

import video3d as v3

SHAPE = (540, 960)
DIMS = (1.6, 0.6, 0.9)
FPS = 30.0


def camera():
    cam = v3.ground_camera(800.0, (960, 540), (480.0, 180.0))
    return v3.scale_camera_to(cam, (380.0, 400.0), (580.0, 400.0), 2.0)


def texture(p, seed):
    """Smooth colour pattern fixed to the surface (so it moves with the object)."""
    r = np.random.default_rng(seed)
    out = np.zeros(len(p))
    for _ in range(6):
        k = r.normal(0, 9, 3)
        out += np.sin(p @ k + r.uniform(0, 6.3))
    return np.clip(128 + 30 * out, 0, 255)


def box_surface(n=260000, seed=1):
    rng = np.random.default_rng(seed)
    L, W, H = DIMS
    lo, hi = np.array([-L / 2, -W / 2, 0]), np.array([L / 2, W / 2, H])
    p = rng.uniform(lo, hi, (n, 3))
    ax = rng.integers(0, 3, n)
    side = rng.integers(0, 2, n)
    p[np.arange(n), ax] = np.where(side, hi[ax], lo[ax])
    return p, texture(p, 7)


def render(cam, poses, ground_img):
    pts, col = box_surface()
    frames = np.repeat(ground_img[None], len(poses), 0)
    masks = np.zeros((len(poses),) + SHAPE, bool)
    for t, pose in enumerate(poses):
        uv, depth = v3.project_object(cam, pose, pts)
        order = np.argsort(-depth)                       # far first, near overwrite
        px = np.round(uv[order]).astype(int)
        c = col[order]
        k = (depth[order] > 0) & (px[:, 0] >= 1) & (px[:, 0] < SHAPE[1] - 1) & (px[:, 1] >= 1) & (px[:, 1] < SHAPE[0] - 1)
        f, m = frames[t], np.zeros(SHAPE, np.uint8)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            f[px[k, 1] + dy, px[k, 0] + dx] = c[k]
            m[px[k, 1] + dy, px[k, 0] + dx] = 1
        masks[t] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return frames, masks


def ground_image(cam):
    v, u = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(float)
    uv = np.column_stack([u.ravel(), v.ravel()])
    img = np.full(len(uv), 90.0)
    ok = cam.ray(uv)[:, 2] < 0
    g = cam.backproject_to_plane(uv[ok], 0.0)
    img[ok] = texture(np.column_stack([g[:, :2] * 0.5, np.zeros(ok.sum())]), 3)
    return img.reshape(SHAPE).astype(np.uint8)


def trajectory(name, T=90):
    s = np.linspace(0, 1, T)
    if name == "arc":
        ang = math.pi * s
        x, y = 2.5 * np.cos(ang), 9.0 + 2.5 * np.sin(ang)
    elif name == "start":
        d = np.where(s < 0.35, 0.6 * (s / 0.35) ** 2, 0.6 + 2 * 0.6 / 0.35 * (s - 0.35))   # rest, then cruise
        ang = np.clip(d - 0.6, 0, None) * 0.9
        x = -3.0 + np.cumsum(np.gradient(d) * np.cos(ang))
        y = 8.0 + np.cumsum(np.gradient(d) * np.sin(ang))
    else:  # "s"
        u = s - 0.5 * 0.25 * np.sin(2 * math.pi * s) / math.pi     # speed dip in the middle
        x, y = -3.0 + 6.0 * u, 9.0 + 1.2 * np.sin(2 * math.pi * u)
    yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    return np.column_stack([x, y, yaw, np.zeros(T)])


def box_inside(pts):
    L, W, H = DIMS
    return (np.abs(pts[:, 0]) <= L / 2) & (np.abs(pts[:, 1]) <= W / 2) & (pts[:, 2] >= 0) & (pts[:, 2] <= H)


def evaluate(cam, masks, truth, est):
    T = len(masks)
    dyaw = np.abs(np.angle(np.exp(1j * (est[:, 2] - truth[:, 2]))))
    dpos = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
    grid = v3.VoxelGrid((-1.3, -0.7, 0.0), (1.3, 0.7, 1.3), 0.04)
    train = range(0, T, 2)
    # carve in the estimated object frame; compare to the truth after aligning the
    # carved model's ground footprint (the object frame origin is arbitrary)
    frac, seen = v3.carve(cam, masks, est, grid, train, margin_px=1)
    occ = ((frac >= 0.95) & (seen >= 10)).ravel()
    shift = np.r_[grid.centres[occ, :2].mean(0), 0.0] if occ.any() else np.zeros(3)
    box = box_inside(grid.centres - shift)
    vol = (occ & box).sum() / max((occ | box).sum(), 1)
    pts = v3.surface_points(occ.reshape(grid.shape), grid)
    held = list(range(1, T, 4))
    ious = [v3.iou(v3.silhouette_of(cam, est[t], pts, SHAPE, grid.size), masks[t]) for t in held] if len(pts) else [0.0]
    return dict(yaw_err_deg_median=float(np.degrees(np.median(dyaw))),
                yaw_err_deg_p90=float(np.degrees(np.percentile(dyaw, 90))),
                yaw_err_deg_first_15=float(np.degrees(np.median(dyaw[:15]))),
                pos_err_m_median=float(np.median(dpos)),
                volume_iou=float(vol), voxels=int(occ.sum()),
                held_out_iou=float(np.mean(ious))), dyaw


def main(out="docs/video3d"):
    cam = camera()
    gimg = ground_image(cam)
    results, curves = {}, {}
    for name in ("arc", "start", "s"):
        truth = trajectory(name)
        frames, masks = render(cam, truth, gimg)
        track, _ = v3.initial_poses(cam, masks, FPS)
        variants = dict(track=track,
                        flow_bottom=v3.initial_poses(cam, masks, FPS, frames=frames)[0],
                        flow_contacts=v3.initial_poses(cam, masks, FPS, frames=frames, footprint="contacts")[0],
                        flow_closed=v3.initial_poses(cam, masks, FPS, frames=frames, footprint="closed")[0],
                        flow_closed_true_dims=v3.initial_poses(cam, masks, FPS, frames=frames, footprint="closed",
                                                               dims=DIMS[:2])[0])
        results[name], errs = {}, {}
        for k, est in variants.items():
            results[name][k], errs[k] = evaluate(cam, masks, truth, est)
        yaw = variants["flow_closed"][:, 2]
        fits = [v3.footprint_closed_form(cam, m, yaw[t], DIMS[:2]) for t, m in enumerate(masks)]
        results[name]["closed_form_fit_rate"] = float(np.mean([f is not None for f in fits]))
        results[name]["closed_form_err_m_median"] = float(np.median([f[1] for f in fits if f is not None]))
        # footprint accuracy alone, with the exact heading: closed form vs ground contacts
        cf = [v3.footprint_closed_form(cam, m, truth[t, 2], DIMS[:2]) for t, m in enumerate(masks)]
        ct = [v3.footprint_centre(cam, m, truth[t, 2]) for t, m in enumerate(masks)]
        results[name]["true_heading_pos_err_m_median"] = dict(
            closed_form=float(np.median([np.linalg.norm(f[0] - truth[t, :2]) for t, f in enumerate(cf) if f is not None])),
            contacts=float(np.median([np.linalg.norm(f[0] - truth[t, :2]) for t, f in enumerate(ct) if f is not None])))
        curves[name] = (errs, frames[len(frames) // 2], masks[len(frames) // 2])
        print(name, json.dumps(results[name], indent=1))
    os.makedirs(out, exist_ok=True)
    json.dump(results, open(os.path.join(out, "flow_init_benchmark.json"), "w"), indent=1)
    plot(curves, results, os.path.join(out, "flow_init_benchmark.png"))


def plot(curves, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    for j, (name, (errs, frame, mask)) in enumerate(curves.items()):
        a = ax[0, j]
        a.imshow(frame, cmap="gray")
        a.contour(mask, [0.5], colors="r", linewidths=0.8)
        a.set_title(f"'{name}' (middle frame)")
        a.axis("off")
        a = ax[1, j]
        r = results[name]
        for k in ("track", "flow_bottom", "flow_contacts", "flow_closed", "flow_closed_true_dims"):
            a.bar(k.replace("flow_", "").replace("_", "\n"), r[k]["held_out_iou"])
            a.text(k.replace("flow_", "").replace("_", "\n"), r[k]["held_out_iou"] + 0.005,
                   f"{r[k]['held_out_iou']:.3f}\n{r[k]['pos_err_m_median'] * 100:.0f} cm", ha="center", fontsize=8)
        a.set_title("held-out silhouette IoU (label: IoU, median position error)", fontsize=9)
        a.set_ylim(0.6, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


if __name__ == "__main__":
    main()
