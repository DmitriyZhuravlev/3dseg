"""Benchmark for parts3d: an articulated 'motorcycle' with known parts, rendered on
textured ground in front of the fixed camera of tests/test_video3d.py.

Parts: body (box), front and rear wheel (cylinders that roll with the motion), rider
(box swaying about the hip: lean and pitch). Every part carries its own surface
texture, so optical flow sees how it moves. The object is carved from the masks with
the true body poses, then segmented; every carved voxel has a ground-truth part.

Compared:
  rigid        - one part (what video3d.py assumes)
  traj-kmeans  - classic motion segmentation: k-means on each voxel's flow-residual
                 trajectory, with the true number of parts given (oracle K)
  ours-nosmooth- parts3d without the spatial smoothness term
  ours         - parts3d (number of parts found automatically)
Metrics on surface voxels (and the whole volume): mIoU after optimal matching of
labels to parts, adjusted Rand index, number of parts, flow end-point error.

    python bench_parts3d.py      # docs/parts3d/bench_parts3d.{json,png}
"""
import json
import math
import os
import time

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import bench_flow_init as b
import parts3d as p3
import video3d as v3

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "parts3d")
FPS = 30.0
R_WHEEL = 0.33
PARTS = ["body", "front wheel", "rear wheel", "rider"]
BODY = ((-0.40, -0.15, 0.45), (0.40, 0.15, 0.85))
RIDER = ((-0.25, -0.18, 0.85), (0.15, 0.18, 1.55))
HIP = np.array([-0.05, 0.0, 0.85])
WHEELS = (np.array([0.75, 0.0, R_WHEEL]), np.array([-0.75, 0.0, R_WHEEL]))
HALF_W = 0.07


def box_points(lo, hi, n, rng):
    lo, hi = np.array(lo), np.array(hi)
    p = rng.uniform(lo, hi, (n, 3))
    ax = rng.integers(0, 3, n)
    side = rng.integers(0, 2, n)
    p[np.arange(n), ax] = np.where(side, hi[ax], lo[ax])
    return p


def wheel_points(n, rng):
    """Cylinder about the y axis, centred at the origin: tyre surface + both side discs."""
    k = rng.random(n)
    a = rng.uniform(0, 2 * math.pi, n)
    r = np.where(k < 0.3, R_WHEEL, R_WHEEL * np.sqrt(rng.random(n)))
    y = np.where(k < 0.3, rng.uniform(-HALF_W, HALF_W, n), np.where(rng.random(n) < 0.5, -HALF_W, HALF_W))
    return np.column_stack([r * np.cos(a), y, r * np.sin(a)])


def model(seed=1):
    rng = np.random.default_rng(seed)
    parts = [box_points(*BODY, 70000, rng), wheel_points(60000, rng), wheel_points(60000, rng),
             box_points(*RIDER, 70000, rng)]
    cols = [b.texture(p * 4.0, 11 + i) for i, p in enumerate(parts)]   # ~15 cm features (spokes, tread, clothing folds)
    return parts, cols


def articulation(t, dist):
    """Per part (R, centre, translation) applied in the object frame at frame t."""
    theta = dist / R_WHEEL
    Rw = Rotation.from_rotvec([0, theta, 0]).as_matrix()
    lean = 0.15 * math.sin(2 * math.pi * t / 40)
    pitch = 0.10 * math.sin(2 * math.pi * t / 27)
    Rr = Rotation.from_rotvec([lean, 0, 0]).as_matrix() @ Rotation.from_rotvec([0, pitch, 0]).as_matrix()
    return [(np.eye(3), np.zeros(3), np.zeros(3)), (Rw, np.zeros(3), WHEELS[0]), (Rw, np.zeros(3), WHEELS[1]),
            (Rr, HIP, np.zeros(3))]


def posed_parts(parts, t, dist):
    out = []
    for p, (R, c, tr) in zip(parts, articulation(t, dist)):
        out.append((p - c) @ R.T + c + tr)
    return out


def render(cam, poses, ground):
    parts, cols = model()
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
    frames = np.repeat(ground[None], len(poses), 0)
    masks = np.zeros((len(poses),) + b.SHAPE, bool)
    part_imgs = np.full((len(poses),) + b.SHAPE, -1, np.int8)
    col = np.concatenate(cols)
    pid = np.concatenate([np.full(len(p), k, np.int8) for k, p in enumerate(parts)])
    for t, pose in enumerate(poses):
        pts = np.vstack(posed_parts(parts, t, dist[t]))
        uv, depth = v3.project_object(cam, pose, pts)
        order = np.argsort(-depth)
        px = np.round(uv[order]).astype(int)
        c = col[order]
        pk = pid[order]
        k = (depth[order] > 0) & (px[:, 0] >= 1) & (px[:, 0] < b.SHAPE[1] - 1) & (px[:, 1] >= 1) & (px[:, 1] < b.SHAPE[0] - 1)
        f, m = frames[t], np.zeros(b.SHAPE, np.uint8)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            f[px[k, 1] + dy, px[k, 0] + dx] = c[k]
            m[px[k, 1] + dy, px[k, 0] + dx] = 1
            part_imgs[t][px[k, 1] + dy, px[k, 0] + dx] = pk[k]
        masks[t] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    render.part_imgs = part_imgs
    return frames, masks


def truth_labels(X, return_dist=False):
    """Ground-truth part of points in the object frame (rest pose); outside -> nearest."""
    d = np.full((len(X), 4), np.inf)
    for k, (lo, hi) in ((0, BODY), (3, RIDER)):
        q = np.maximum(np.maximum(np.array(lo) - X, X - np.array(hi)), 0)
        d[:, k] = np.linalg.norm(q, axis=1)
    for k, c in ((1, WHEELS[0]), (2, WHEELS[1])):
        q = X - c
        rad = np.maximum(np.hypot(q[:, 0], q[:, 2]) - R_WHEEL, 0)
        wid = np.maximum(np.abs(q[:, 1]) - HALF_W, 0)
        d[:, k] = np.hypot(rad, wid)
    return (d.argmin(1), d.min(1)) if return_dist else d.argmin(1)


def scores(pred, gt, K=4):
    P = pred.max() + 1
    C = np.zeros((K, P))
    np.add.at(C, (gt, pred), 1)
    r, c = linear_sum_assignment(-C)
    ious = np.zeros(K)
    for i, j in zip(r, c):
        ious[i] = C[i, j] / (C[i].sum() + C[:, j].sum() - C[i, j])
    n = len(gt)
    comb = lambda x: x * (x - 1) / 2
    sij = comb(C).sum()
    si, sj = comb(C.sum(1)).sum(), comb(C.sum(0)).sum()
    exp = si * sj / comb(n)
    ari = (sij - exp) / (0.5 * (si + sj) - exp)
    return dict(miou=float(ious.mean()), iou_per_part=dict(zip(PARTS, ious.round(3).tolist())), ari=float(ari), parts=int(P))


def project_labels(cam, pose, X, labels, mask, voxel):
    """Label image inside `mask`: every pixel takes the label of the nearest visible voxel."""
    from scipy.spatial import cKDTree
    uv, _, vis = p3.visible(cam, pose, X, mask.shape, voxel)
    ys, xs = np.nonzero(mask)
    out = np.full(mask.shape, -1, int)
    if vis.sum() == 0:
        return out
    _, nn = cKDTree(uv[vis]).query(np.column_stack([xs, ys]))
    out[ys, xs] = labels[vis][nn]
    return out


def image_scores(pred_imgs, gt_imgs, per_frame_matching=False):
    """mIoU / ARI over object pixels of all frames; labels matched to parts globally
    (one 3D labelling) or, for per-frame 2D methods, per frame (favours them)."""
    if per_frame_matching:
        remapped = []
        for p, g in zip(pred_imgs, gt_imgs):
            m = g >= 0
            if not m.any():
                continue
            P = p[m].max() + 1
            C = np.zeros((4, P))
            np.add.at(C, (g[m], p[m]), 1)
            r, c = linear_sum_assignment(-C)
            mp = np.full(P, 4)
            mp[c] = r
            q = p.copy()
            q[m] = mp[p[m]]
            remapped.append(q[m])
        pred = np.concatenate(remapped)
        gt = np.concatenate([g[g >= 0] for g in gt_imgs if (g >= 0).any()])
    else:
        m = gt_imgs >= 0
        pred, gt = pred_imgs[m], gt_imgs[m]
    pred = np.maximum(pred, 0)
    return scores(pred, gt.astype(int))


def flow2d_kmeans(cam, frames, masks, poses, K=4, seed=0):
    """Per-frame 2D baseline in the spirit of the article: k-means on the flow (minus the
    rigid prediction of the ground under the object's motion is not available in 2D, so
    on the raw flow) of the object's pixels, every frame on its own."""
    out = np.full(masks.shape, -1, int)
    for t in range(len(frames) - 1):
        ys, xs = np.nonzero(masks[t])
        if len(xs) < K * 10:
            continue
        fl = p3.dense_flow(frames[t], frames[t + 1])
        F = np.column_stack([fl[ys, xs], (xs - xs.mean()) / 200.0, (ys - ys.mean()) / 200.0])
        np.random.seed(seed)
        _, lab = kmeans2(F, K, minit="++", seed=seed)
        out[t, ys, xs] = lab
    return out


def traj_kmeans(cam, poses, X, obs, K, seed=0):
    keys = sorted(obs)
    F = np.zeros((len(X), 2 * len(keys)))
    for j, t in enumerate(keys):
        idx, uv0, f = obs[t]
        F[idx, 2 * j:2 * j + 2] = f - p3.predicted_flow(cam, poses[t + 1], X[idx], uv0, np.zeros(6), np.zeros(3))
    np.random.seed(seed)
    _, lab = kmeans2(F, K, minit="++", seed=seed)
    return lab


def run(name, cam, ground, log=print, **kw):
    t0 = time.time()
    poses = b.trajectory(name)
    frames, masks = render(cam, poses, ground)
    gt_imgs = render.part_imgs
    grid = v3.VoxelGrid((-1.2, -0.5, 0.0), (1.2, 0.5, 1.7), 0.03)
    frac, seen = v3.carve(cam, masks, poses, grid, range(len(poses)), margin_px=1)
    occ = (frac >= 0.9) & (seen >= 10)
    pairs = list(range(0, len(poses) - 1, 2))            # flow from even frames only
    test = list(range(1, len(poses) - 1, 2))             # evaluated on the odd frames
    seg = p3.segment_parts(cam, frames, poses, occ, grid, pairs, log=log, **kw)
    seg0 = p3.segment_parts(cam, frames, poses, occ, grid, pairs, **dict(kw, lam=0.0))
    X = grid.centres[seg.surf_idx]
    obs = p3.observe(cam, frames, poses, X, grid.size, pairs)
    tk = traj_kmeans(cam, poses, X, obs, 4)
    gt_t = gt_imgs[test]

    def proj(labels):
        return np.stack([project_labels(cam, poses[t], X, labels, masks[t] & (gt_imgs[t] >= 0), grid.size) for t in test])

    res = {}
    res["rigid (one part)"] = image_scores(np.zeros_like(gt_t, dtype=int), gt_t)
    res["2D flow k-means per frame (oracle K=4, per-frame matching)"] = image_scores(
        flow2d_kmeans(cam, frames, masks, poses)[test], gt_t, per_frame_matching=True)
    res["3D trajectory k-means (oracle K=4)"] = image_scores(proj(tk), gt_t)
    res["ours, no smoothness"] = image_scores(proj(seg0.labels_surface), gt_t)
    res["ours"] = image_scores(proj(seg.labels_surface), gt_t)
    gt_v, dist = truth_labels(X, True)
    on = dist <= grid.size
    res["ours, on-object surface voxels"] = scores(seg.labels_surface[on], gt_v[on])
    res["on_object_fraction_of_carved_surface"] = float(on.mean())
    res["seconds"] = round(time.time() - t0)
    for k, v in res.items():
        log(name, k, v)
    return res, dict(frames=frames, masks=masks, poses=poses, occ=occ, grid=grid, seg=seg, X=X, gt_imgs=gt_imgs,
                     traj=tk, test=test, proj=proj)


COLS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                 [0.58, 0.4, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5]])


def match_to_parts(pred_imgs, gt_imgs, labels):
    """Recolour predicted labels with the ground-truth part they overlap most (for display)."""
    m = gt_imgs >= 0
    P = max(labels.max(), pred_imgs.max()) + 1
    C = np.zeros((4, P))
    np.add.at(C, (gt_imgs[m].astype(int), np.maximum(pred_imgs[m], 0)), 1)
    r, c = linear_sum_assignment(-C)
    mp = np.arange(P) + 4
    mp[c] = r
    return mp


def figure(name, d, res, path, cam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, seg, test = d["X"], d["seg"], d["test"]
    gt_t = d["gt_imgs"][test]
    gt_v = truth_labels(X)
    views = [("ground truth (nearest part)", gt_v, None),
             ("3D trajectory k-means (oracle K=4)", d["traj"], "3D trajectory k-means (oracle K=4)"),
             ("ours", seg.labels_surface, "ours")]
    fig = plt.figure(figsize=(17, 10))
    for j, (title, lab, key) in enumerate(views):
        if key:
            mp = match_to_parts(d["proj"](lab), gt_t, lab)
            lab = mp[lab]
        a = fig.add_subplot(2, 3, j + 1, projection="3d")
        a.scatter(X[:, 0], X[:, 1], X[:, 2], c=COLS[lab % len(COLS)], s=2)
        a.set_box_aspect(np.ptp(X, 0))
        a.view_init(18, -60)
        a.set_axis_off()
        sub = f"\nheld-out frames: mIoU {res[key]['miou']:.2f}, ARI {res[key]['ari']:.2f}, {res[key]['parts']} parts" if key else ""
        a.set_title(title + sub, fontsize=11)
    mp = match_to_parts(d["proj"](seg.labels_surface), gt_t, seg.labels_surface)
    for j, t in enumerate((test[len(test) // 5], test[len(test) // 2], test[4 * len(test) // 5])):
        a = fig.add_subplot(2, 3, 4 + j)
        img = cv2.cvtColor(d["frames"][t], cv2.COLOR_GRAY2RGB).astype(float) / 255
        g = d["gt_imgs"][t]
        p = p3_project = project_labels(cam, d["poses"][t], X, mp[seg.labels_surface], g >= 0, d["grid"].size)
        ys, xs = np.nonzero(g >= 0)
        y0, y1, x0, x1 = max(ys.min() - 25, 0), ys.max() + 25, max(xs.min() - 25, 0), xs.max() + 25
        left = img.copy()
        left[g >= 0] = 0.4 * left[g >= 0] + 0.6 * COLS[g[g >= 0]]
        right = img.copy()
        right[p >= 0] = 0.4 * right[p >= 0] + 0.6 * COLS[p[p >= 0] % len(COLS)]
        a.imshow(np.hstack([left[y0:y1, x0:x1], np.ones((y1 - y0, 6, 3)), right[y0:y1, x0:x1]]))
        a.set_title(f"held-out frame {t}: true parts | ours (3D labels projected)", fontsize=10)
        a.axis("off")
    fig.suptitle(f"3D part segmentation from motion, synthetic '{name}' video (one fixed camera, no training)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=75)


def main():
    os.makedirs(OUT, exist_ok=True)
    cam = b.camera()
    ground = b.ground_image(cam)
    allres = {}
    for name in ("arc", "s"):
        res, d = run(name, cam, ground)
        allres[name] = res
        figure(name, d, res, os.path.join(OUT, f"bench_parts3d_{name}.png"), cam)
    json.dump(allres, open(os.path.join(OUT, "bench_parts3d.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
