"""3D part segmentation (parts3d.py) of the motorcycle carved from data/bike.mp4.

    python docs/bike_parts3d.py --solution out/bike/bike_solution.npz

Uses the carved model and poses of `python video_demos.py bike` (bike_solution.npz),
re-sampled to 4 cm voxels, and the optical flow of every 4th frame pair. Writes
docs/parts3d/bike_parts3d.{png,json}.

No ground truth exists for the parts, so the check is predictive: the surface voxels
are split into two halves by 12 cm blocks; part motions are fitted on one half and
the flow of the other half is predicted (end-point error, px), on frame pairs not
used for the segmentation. A labelling that only over-fits does not lower this error.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import parts3d as p3  # noqa: E402
import video3d as v3  # noqa: E402
import video_demos as vd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(ROOT, "docs", "parts3d")
COLS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                 [0.58, 0.4, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5]])


def coarsen(occ, grid, f=2):
    s = [n // f * f for n in occ.shape]
    o = occ[:s[0], :s[1], :s[2]].reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean((1, 3, 5)) >= 0.5
    g = v3.VoxelGrid(grid.lo, grid.lo + np.array(o.shape) * grid.size * f, grid.size * f)
    return o, g


def heldout_epe(cam, poses, X, obs, labels, block=0.12):
    """Fit part motions on one half of the voxels (12 cm blocks), predict the other half."""
    key = np.floor(X / block).astype(int)
    half = (key.sum(1) % 2) == 0
    out = []
    for fit_half in (half, ~half):
        lab_fit = np.where(fit_half, labels, -1)
        K = labels.max() + 1
        centres = np.array([X[labels == k].mean(0) for k in range(K)])
        mot = [dict() for _ in range(K)]
        for t, (idx, uv0, f) in obs.items():
            for k in range(K):
                m = lab_fit[idx] == k
                if m.sum() >= 12:
                    mot[k][t], _ = p3.fit_motion(cam, poses[t + 1], X[idx[m]], uv0[m], f[m], centres[k])
        test = {t: (idx[~fit_half[idx]], uv0[~fit_half[idx]], f[~fit_half[idx]]) for t, (idx, uv0, f) in obs.items()}
        out.append(p3.flow_epe(cam, poses, X, test, labels, centres, mot))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data", "bike.mp4"))
    ap.add_argument("--solution", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--gap", type=int, default=1, help="flow between frames t and t + gap")
    ap.add_argument("--beta", type=float, default=0.05, help="label cost, px per surface voxel")
    ap.add_argument("--lam", type=float, default=0.1, help="Potts smoothness, px per neighbouring voxel pair")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sol = np.load(a.solution)
    frames, fps, bg, cam, _ = vd.bike_setup(a.video)
    poses = sol["poses"][::a.gap]
    frames = frames[::a.gap]
    grid0 = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    occ, grid = coarsen(sol["occ"], grid0)
    T = len(poses)
    present = [t for t in range(T - 1) if (cv2.absdiff(frames[t], bg).max(2) > 30).sum() > 2000]
    train = [t for t in present if t % a.every == 0]
    test = [t for t in present if t % a.every == a.every // 2]
    print(f"{occ.sum()} voxels at {grid.size * 100:.0f} cm, {len(train)} train / {len(test)} test frame pairs")

    seg = p3.segment_parts(cam, frames, poses, occ, grid, train, beta=a.beta, lam=a.lam, log=print)
    X = grid.centres[seg.surf_idx]
    obs_test = p3.observe(cam, frames, poses, X, grid.size, test)
    zeros = np.zeros(len(X), int)
    res = dict(parts=int(seg.labels_surface.max() + 1),
               part_sizes=np.bincount(seg.labels_surface).tolist(),
               part_extents_m=[np.ptp(X[seg.labels_surface == k], 0).round(2).tolist()
                               for k in range(seg.labels_surface.max() + 1)],
               heldout_flow_epe_px=dict(rigid=heldout_epe(cam, poses, X, obs_test, zeros),
                                        parts=heldout_epe(cam, poses, X, obs_test, seg.labels_surface)),
               history=seg.history, seconds=round(time.time() - t0))
    print(json.dumps(res, indent=1))
    res["gap"], res["beta"], res["lam"] = a.gap, a.beta, a.lam
    tag = f"gap{a.gap}_beta{a.beta:g}_lam{a.lam:g}"
    json.dump(res, open(os.path.join(OUT, f"bike_parts3d_{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "bike_parts3d_labels.npz"), labels=seg.labels_volume, lo=grid.lo,
                        hi=grid.hi, size=grid.size)

    fig = plt.figure(figsize=(17, 10))
    lab = seg.labels_surface
    for j, (el, az) in enumerate(((15, -90), (15, 0), (60, -60))):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=COLS[lab % len(COLS)], s=4)
        ax.set_box_aspect(np.ptp(X, 0))
        ax.view_init(el, az)
        ax.set_axis_off()
        ax.set_title(["side", "front", "above"][j] + " view of the carved bike, coloured by part", fontsize=10)
    picks = [test[int(len(test) * q)] for q in (0.15, 0.5, 0.85)]
    for j, t in enumerate(picks):
        ax = fig.add_subplot(2, 3, 4 + j)
        img = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB).astype(float) / 255
        uv, _, vis = p3.visible(cam, poses[t], X, img.shape[:2], grid.size)
        over = img.copy()
        px = np.round(uv[vis]).astype(int)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                over[np.clip(px[:, 1] + dy, 0, img.shape[0] - 1), np.clip(px[:, 0] + dx, 0, img.shape[1] - 1)] = \
                    COLS[lab[vis] % len(COLS)]
        y0, x0 = np.maximum(px.min(0)[::-1] - 40, 0)
        y1, x1 = px.max(0)[::-1] + 40
        ax.imshow(np.hstack([img[y0:y1, x0:x1], np.ones((y1 - y0, 6, 3)), (0.45 * img + 0.55 * over)[y0:y1, x0:x1]]))
        ax.set_title(f"frame {t * a.gap} (not used for segmentation): video | parts", fontsize=10)
        ax.axis("off")
    e = res["heldout_flow_epe_px"]
    fig.suptitle(f"bike.mp4: {res['parts']} rigid parts found from motion; held-out flow error "
                 f"{e['rigid']:.2f} px (one rigid body) -> {e['parts']:.2f} px (parts)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"bike_parts3d_{tag}.png"), dpi=75)


if __name__ == "__main__":
    main()
