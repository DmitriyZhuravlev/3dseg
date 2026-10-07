"""Pictures for video_cuboids_parts.py: the fused cuboid models of bike.mp4.

    python docs/bike_cuboids_parts_demo.py --solution out/bike/bike_solution.npz --background bg.png

Reads docs/parts3d/bike_cuboids_parts_models.npz and .json; writes
docs/parts3d/bike_cuboids_parts.png: each model in 3D (front assembly in orange) and its
silhouette against held-out frames.
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT = os.path.join(ROOT, "docs", "parts3d")
SHOW = [("cuboids_(global_+_points),_one_tau", "cuboids, one threshold"),
        ("+_per-part_tau_+_articulated_fusion", "cuboids, per-part + articulated"),
        ("+_per-part_tau_+_articulated_fusion,_intersected_with_carving", "... intersected with carving"),
        ("carving_(reference)", "silhouette carving")]
KEYS = {"cuboids_(global_+_points),_one_tau": "cuboids (global + points), one tau",
        "+_per-part_tau_+_articulated_fusion": "+ per-part tau + articulated fusion",
        "+_per-part_tau_+_articulated_fusion,_intersected_with_carving":
            "+ per-part tau + articulated fusion, intersected with carving",
        "carving_(reference)": "carving (reference)"}


def mesh_draw(ax, occ, front, grid):
    for part, col in ((occ & ~front, np.array([0.35, 0.55, 0.8])), (occ & front, np.array([1.0, 0.55, 0.1]))):
        if part.sum() < 10:
            continue
        v, f, n = v3.occupancy_mesh(part, grid, smooth=0.8)
        nn = n[f].mean(1)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
        k = 0.35 + 0.65 * np.clip(nn @ np.array([0.3, -0.6, 0.75]) / 1.02, 0, 1)
        ax.add_collection3d(Poly3DCollection(v[f], facecolors=np.clip(col * k[:, None], 0, 1), linewidths=0))
    lo, hi = grid.lo, grid.hi
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(14, -75)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data", "bike.mp4"))
    ap.add_argument("--solution", required=True)
    ap.add_argument("--background", required=True)
    a = ap.parse_args()
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    M = np.load(os.path.join(OUT, "bike_cuboids_parts_models.npz"))
    met = json.load(open(os.path.join(OUT, "bike_cuboids_parts.json")))
    grid = v3.VoxelGrid(M["lo"], M["hi"], float(M["size"]))
    front = M["front"]
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)
    test = [301, 701]
    frames = {i: (f, m) for i, f, m in vn.frames_and_masks(a.video, bg, test)}
    cap = cv2.VideoCapture(a.video)
    colour = {}
    for i in test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        colour[i] = cv2.resize(cap.read()[1], vn.SIZE, interpolation=cv2.INTER_AREA)

    fig = plt.figure(figsize=(20, 12))
    for j, (key, title) in enumerate(SHOW):
        occ = M[key]
        r = met[KEYS[key]]
        ax = fig.add_subplot(3, 4, j + 1, projection="3d")
        mesh_draw(ax, occ, front, grid)
        ax.set_title(f"{title}\nheld-out IoU {r['held_out_iou_rigid']:.3f}, points {r['held_out_point_distance_median_cm']:.1f} cm",
                     fontsize=10)
        sp = v3.surface_points(occ, grid)
        for row, i in enumerate(test):
            f, m = frames[i]
            sil = v3.silhouette_of(cam, poses[i], sp, m.shape, grid.size)
            img = colour[i].copy()
            over = img.copy()
            over[sil & m] = (0, 200, 0)
            over[sil & ~m] = (0, 0, 230)
            over[~sil & m] = (230, 120, 0)
            blend = cv2.addWeighted(img, 0.4, over, 0.6, 0)
            ys, xs = np.nonzero(m | sil)
            crop = blend[max(ys.min() - 20, 0):ys.max() + 20, max(xs.min() - 20, 0):xs.max() + 20]
            ax2 = fig.add_subplot(3, 4, 4 * (row + 1) + j + 1)
            ax2.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"held-out frame {i}: IoU {v3.iou(sil, m):.3f}", fontsize=10)
            ax2.axis("off")
    fig.suptitle("bike.mp4: recursive cuboids fused per motion part (front assembly orange). "
                 "Silhouettes: green = agree, red = model only, blue = video only", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bike_cuboids_parts.png"), dpi=70)


if __name__ == "__main__":
    main()
