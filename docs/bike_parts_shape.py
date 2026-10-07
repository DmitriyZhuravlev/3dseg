"""Shape of the bike from data/bike.mp4 split into the parts found by parts3d.py.

    python docs/bike_parts_shape.py --solution out/bike/bike_solution.npz

Takes the carved model (2 cm voxels) of `python video_demos.py bike` and the part labels
of docs/bike_parts3d.py (docs/parts3d/bike_parts3d_labels.npz), meshes every part on its
own and writes docs/parts3d/bike_parts_shape.png (views, exploded view, overlay on video
frames) and bike_parts_shape.json (part sizes) and one PLY per part.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import video3d as v3  # noqa: E402
import video_demos as vd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT = os.path.join(ROOT, "docs", "parts3d")
COLS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                 [0.58, 0.4, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5], [0.74, 0.74, 0.13]])
NAMES = {0: "frame + rider"}


def shade(verts, faces, normals, colour, light=(0.3, -0.6, 0.75)):
    n = normals[faces].mean(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    k = 0.35 + 0.65 * np.clip(n @ (np.array(light) / np.linalg.norm(light)), 0, 1)
    return np.clip(colour[None] * k[:, None], 0, 1)


def draw(ax, meshes, offsets=None, view=(15, -90)):
    allv = []
    for k, (v, f, n) in meshes.items():
        o = np.zeros(3) if offsets is None else offsets[k]
        vv = v + o
        pc = Poly3DCollection(vv[f], facecolors=shade(vv, f, n, COLS[k % len(COLS)]), linewidths=0)
        ax.add_collection3d(pc)
        allv.append(vv)
    allv = np.vstack(allv)
    lo, hi = allv.min(0), allv.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(*view)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data", "bike.mp4"))
    ap.add_argument("--solution", required=True)
    ap.add_argument("--labels", default=os.path.join(OUT, "bike_parts3d_labels.npz"))
    a = ap.parse_args()
    sol = np.load(a.solution)
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    occ = sol["occ"]
    L = np.load(a.labels)
    lgrid = v3.VoxelGrid(L["lo"], L["hi"], float(L["size"]))
    lab4 = L["labels"].ravel()
    have = lab4 >= 0
    _, nn = cKDTree(lgrid.centres[have]).query(grid.centres[occ.ravel()])
    lab = -np.ones(occ.size, int)
    lab[np.nonzero(occ.ravel())[0]] = lab4[have][nn]
    lab = lab.reshape(occ.shape)
    K = lab.max() + 1

    meshes, info = {}, []
    for k in range(K):
        part = lab == k
        if part.sum() < 30:
            continue
        v, f, n = v3.occupancy_mesh(part, grid, smooth=0.8)
        if len(f) == 0:
            continue
        meshes[k] = (v, f, n)
        ext = np.ptp(v, 0)
        info.append(dict(part=k, voxels=int(part.sum()), volume_l=round(part.sum() * grid.size ** 3 * 1000, 1),
                         extent_m=ext.round(2).tolist(), centre_m=v.mean(0).round(2).tolist()))
        v3.save_ply(os.path.join(OUT, f"bike_part{k}.ply"), v, f, np.tile(COLS[k % len(COLS)], (len(v), 1)))
    json.dump(dict(voxel_m=grid.size, parts=info), open(os.path.join(OUT, "bike_parts_shape.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))

    centre = np.vstack([m[0] for m in meshes.values()]).mean(0)
    offsets = {}
    for k, (v, _, _) in meshes.items():
        d = v.mean(0) - centre
        offsets[k] = 0.0 if k == 0 else d / (np.linalg.norm(d) + 1e-9) * 0.35

    fig = plt.figure(figsize=(18, 11))
    for j, (view, title) in enumerate((((12, -90), "side"), ((12, 0), "front"), ((35, -55), "three-quarter"))):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        draw(ax, meshes, view=view)
        ax.set_title(f"reconstructed shape, {title} view", fontsize=11)
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    draw(ax, meshes, offsets={k: (np.zeros(3) if k == 0 else offsets[k]) for k in meshes}, view=(20, -70))
    ax.set_title("exploded view: parts moved apart", fontsize=11)

    frames, fps, bg, cam, _ = vd.bike_setup(a.video)
    poses = sol["poses"]
    for j, t in enumerate((300, 990)):
        ax = fig.add_subplot(2, 3, 5 + j)
        img = frames[t].copy()
        over = img.copy()
        # painter's algorithm over all parts' triangles, far to near
        tris, cols, dep = [], [], []
        for k, (v, f, n) in meshes.items():
            uv, depth = v3.project_object(cam, poses[t], v)
            sh = shade(v, f, n, COLS[k % len(COLS)])
            tris.append(uv[f])
            cols.append(sh)
            dep.append(depth[f].mean(1))
        tris, cols, dep = np.concatenate(tris), np.concatenate(cols), np.concatenate(dep)
        for i in np.argsort(-dep):
            cv2.fillConvexPoly(over, np.round(tris[i] * 4).astype(np.int32), (cols[i][::-1] * 255).tolist(), cv2.LINE_8, 2)
        blend = cv2.addWeighted(img, 0.35, over, 0.65, 0)
        ys, xs = np.nonzero(np.any(over != img, 2))
        y0, y1, x0, x1 = max(ys.min() - 40, 0), ys.max() + 40, max(xs.min() - 40, 0), xs.max() + 40
        pair = np.hstack([img[y0:y1, x0:x1], np.full((y1 - y0, 6, 3), 255, np.uint8), blend[y0:y1, x0:x1]])
        ax.imshow(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {t}: video | reconstructed parts projected with the estimated pose", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"bike.mp4: shape carved from 589 frames (2 cm voxels), split into {len(meshes)} parts by their motion",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bike_parts_shape.png"), dpi=72)


if __name__ == "__main__":
    main()
