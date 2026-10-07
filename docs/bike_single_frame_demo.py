"""Demo picture for video_single_frame.py: the motorcycle from ONE frame of bike.mp4.

    python docs/bike_single_frame_demo.py --solution out/bike/bike_solution.npz --background bg.png

Rows: the best and the median frame (by held-out IoU of the class-prior model). Columns:
the input frame, the single-frame models in 3D (global cuboids with the main box fitted
to the one silhouette; with the class-size prior, cut by the frame's cone; the
multi-frame carving for reference), and the class-prior model seen from another,
held-out viewpoint.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import video3d as v3  # noqa: E402
import video_normals as vn  # noqa: E402
from lift3d import Camera  # noqa: E402
from video_single_frame import frame_and_mask  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT = os.path.join(ROOT, "docs", "single_frame")


def mesh(ax, occ, grid, col, view=(14, -75)):
    if occ.sum() > 10:
        v, f, n = v3.occupancy_mesh(occ, grid, smooth=0.8)
        nn = n[f].mean(1)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
        k = 0.35 + 0.65 * np.clip(nn @ np.array([0.3, -0.6, 0.75]) / 1.02, 0, 1)
        ax.add_collection3d(Poly3DCollection(v[f], facecolors=np.clip(np.array(col) * k[:, None], 0, 1), linewidths=0))
    ax.set_xlim(grid.lo[0], grid.hi[0])
    ax.set_ylim(grid.lo[1], grid.hi[1])
    ax.set_zlim(grid.lo[2], grid.hi[2])
    ax.set_box_aspect(grid.hi - grid.lo)
    ax.view_init(*view)
    ax.set_axis_off()


def overlay(img, cam, pose, occ, grid, mask):
    sil = v3.silhouette_of(cam, pose, v3.surface_points(occ, grid), mask.shape, grid.size)
    over = img.copy()
    over[sil & mask] = (0, 200, 0)
    over[sil & ~mask] = (0, 0, 230)
    over[~sil & mask] = (230, 120, 0)
    out = cv2.addWeighted(img, 0.4, over, 0.6, 0)
    ys, xs = np.nonzero(mask | sil)
    return out[max(ys.min() - 20, 0):ys.max() + 20, max(xs.min() - 20, 0):xs.max() + 20], v3.iou(sil, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data", "bike.mp4"))
    ap.add_argument("--solution", required=True)
    ap.add_argument("--background", required=True)
    a = ap.parse_args()
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    M = np.load(os.path.join(OUT, "bike_single_frame_models.npz"))
    met = json.load(open(os.path.join(OUT, "bike_single_frame.json")))
    grid = v3.VoxelGrid(M["lo"], M["hi"], float(M["size"]))
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)
    carved = sol["occ"]
    other = {int(M["best"]): 701, int(M["median"]): 301}
    rows = [("best frame", int(M["best"])), ("median frame", int(M["median"]))]

    fig = plt.figure(figsize=(22, 10))
    for r, (tag, i) in enumerate(rows):
        f, m = frame_and_mask(a.video, bg, i)
        pf = met["per_frame"][str(i)]
        ax = fig.add_subplot(2, 5, 5 * r + 1)
        img = f.copy()
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, (0, 255, 255), 2)
        ys, xs = np.nonzero(m)
        ax.imshow(cv2.cvtColor(img[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{tag}: input = frame {i} only", fontsize=11)
        ax.axis("off")
        for c, (key, title, col) in enumerate((
                ("global", "cuboids, main box from the silhouette", (0.6, 0.6, 0.6)),
                ("global, class-size prior ∩ cone", "cuboids, class-size prior, ∩ cone", (0.35, 0.55, 0.8)),
                (None, "carving, all 589 frames (reference)", (0.4, 0.7, 0.4)))):
            ax = fig.add_subplot(2, 5, 5 * r + 2 + c, projection="3d")
            occ = carved if key is None else M[f"{i}|{key}"]
            mesh(ax, occ, grid, col)
            s = met["summary"]["carving, all frames (reference)"] if key is None else pf[key]
            ax.set_title(f"{title}\nheld-out IoU {s['held_out_iou']:.2f}, points {s['held_out_point_distance_median_cm']:.1f} cm",
                         fontsize=10)
        j = other[i]
        f2, m2 = frame_and_mask(a.video, bg, j)
        crop, iou = overlay(f2, cam, poses[j], M[f"{i}|global, class-size prior ∩ cone"], grid, m2)
        ax = fig.add_subplot(2, 5, 5 * r + 5)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"class-prior model seen from held-out frame {j}: IoU {iou:.2f}", fontsize=10)
        ax.axis("off")
    s = met["summary"]
    fig.suptitle("bike.mp4: single-frame reconstruction with recursive cuboids. Median over "
                 f"{met['frames_with_cuboids']} frames, held-out IoU: silhouette cone {s['cone']['held_out_iou_median']:.2f}, "
                 f"cuboids {s['global']['held_out_iou_median']:.2f}, cuboids + class-size prior ∩ cone "
                 f"{s['global, class-size prior ∩ cone']['held_out_iou_median']:.2f}  "
                 "(overlay: green agree, red model only, blue video only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bike_single_frame.png"), dpi=70)


if __name__ == "__main__":
    main()
