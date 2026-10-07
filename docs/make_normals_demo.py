"""Demo images for triangulation / 3D normals (docs/segments/). Run from the repo root after
video_normals.py and normal_integration.py:

    PYTHONPATH=. python docs/make_normals_demo.py --eval-dir out/segeval --normals-dir out/normals \
        --video bike.mp4 --background background.png
"""
import argparse
import json

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import video3d as v3  # noqa: E402
import video_normals as vn  # noqa: E402
from make_segment_demo import caption, grid, OUT  # noqa: E402
from video_demos import model_views  # noqa: E402


def fig_to_pil(fig):
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return im


def points_plot(pts, normals, title):
    fig = plt.figure(figsize=(5.6, 3.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    col = np.clip(0.5 + 0.5 * normals, 0, 1)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=6, depthshade=False)
    sub = np.arange(0, len(pts), 6)
    ax.quiver(pts[sub, 0], pts[sub, 1], pts[sub, 2], normals[sub, 0], normals[sub, 1], normals[sub, 2], length=0.08, color="k", lw=0.5)
    ax.set_box_aspect(np.ptp(pts, 0))
    ax.view_init(18, -70)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return fig_to_pil(fig)


def error_image(mask_shape, uv, z, z_gt, crop, vmax=0.08):
    v = np.full(mask_shape + (3,), 40, np.uint8)
    err = np.abs(z - z_gt) / z_gt
    ok = np.isfinite(err)
    col = cv2.applyColorMap(np.uint8(np.clip(err[ok] / vmax, 0, 1) * 255)[:, None], cv2.COLORMAP_JET)[:, 0]
    for (x, y), c in zip(uv[ok].astype(int), col):
        cv2.rectangle(v, (x, y), (x + 3, y + 3), tuple(int(t) for t in c), -1)
    y0, y1, x0, x1 = crop
    return v[y0:y1, x0:x1]


def normal_image(mask_shape, uv, n, crop):
    v = np.full(mask_shape + (3,), 40, np.uint8)
    col = np.uint8(np.clip(0.5 + 0.5 * n[:, ::-1] * [1, -1, 1], 0, 1) * 255)    # BGR, camera frame
    for (x, y), c in zip(uv.astype(int), col):
        cv2.rectangle(v, (x, y), (x + 3, y + 3), tuple(int(t) for t in c), -1)
    y0, y1, x0, x1 = crop
    return v[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--normals-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--background", required=True)
    a = ap.parse_args()
    E, N = a.eval_dir.rstrip("/") + "/", a.normals_dir.rstrip("/") + "/"

    # --- bike: tracks, triangulated points with normals, carving vs free-space carving
    sol = np.load("docs/video3d/bike_solution.npz")
    nsol = np.load(N + "bike_normals_solution.npz")
    met = json.load(open(N + "bike_normals_metrics.json"))
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)
    idx = list(range(160, 200, 2))
    tracks = vn.klt_tracks(vn.frames_and_masks(a.video, bg, idx), min_len=6)
    frame = next(vn.frames_and_masks(a.video, bg, [idx[-1]]))
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx[-1])
    f = cv2.resize(cap.read()[1], vn.SIZE, interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(0)
    for t in tracks:
        c = tuple(int(v) for v in rng.integers(80, 255, 3))
        p = np.int32([uv for _, uv in t])
        cv2.polylines(f, [p], False, c, 1, cv2.LINE_AA)
        cv2.circle(f, tuple(p[-1]), 2, c, -1)
    m = frame[2]
    ys, xs = np.nonzero(m)
    tiles = [caption(f[max(0, ys.min() - 40):ys.max() + 40, max(0, xs.min() - 80):xs.max() + 80],
                     f"KLT tracks inside the mask ({len(tracks)} tracks, frames {idx[0]}-{idx[-1]})")]
    pts, nrm = nsol["points"], nsol["normals"]
    tiles.append(caption(points_plot(pts, nrm, f"{len(pts)} triangulated points (bike frame), PCA normals"),
                         f"triangulated: reprojection median {met['fit_reproj_rms_median_px']:.2f} px"))
    grd = v3.VoxelGrid(nsol["lo"], nsol["hi"], float(nsol["size"]))
    removed = sol["occ"] & ~nsol["free_space"]
    rp = grd.centres[removed.ravel()]
    fig = plt.figure(figsize=(5.6, 3.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    sp = v3.surface_points(sol["occ"], grd)[::3]
    ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=1, c="0.75")
    ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=2, c="r")
    ax.set_box_aspect(np.ptp(sp, 0)); ax.view_init(18, -70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    tiles.append(caption(fig_to_pil(fig), f"red: {removed.sum()} carved voxels proven empty by the points"))
    for key, label in (("carving", "carving (silhouettes only)"), ("freespace", "carving + free-space from triangulated points"),
                       ("poisson_fused", "Poisson from triangulated + carved-surface oriented points")):
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(N + f"bike_normals_{key}_model.ply")
        mesh.compute_vertex_normals()
        v, fc, nn = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_normals)
        img = model_views(v, fc, nn, np.full((len(v), 3), 0.75), yaws=(0.8,), size=(560, 380), zoom=0.5)[0]
        r = met[label]
        tiles.append(caption(img, f"{key}: held-out pts {r['held_out_point_distance_median_cm']:.1f} cm, "
                                  f"<5cm {100*r['held_out_points_within_5cm']:.0f}%, IoU {r['held_out_silhouette_iou']:.3f}"))
    grid(tiles, 3, "bike.mp4: triangulation in the object frame, 3D normals, surfaces from oriented points",
         f"{OUT}/bike_triangulation_normals.png",
         "held-out = points triangulated only from the odd frames (never used) and silhouette IoU on odd frames")

    # --- pig: normal fields and depth from normal integration
    d = np.load(E + "pig_normal_depths.npz")
    nn = np.load(E + "pig_normals.npz")
    res = json.load(open(f"{OUT}/normals_pig.json"))
    mask = np.load(E + "synth_pig_mask.npy")
    uv, z = d["uv"], d["z"]
    crop = (500, 1700, 700, 2500)
    tiles = [caption(normal_image(mask.shape, uv, nn["exact_normals"], crop), "exact normals (camera frame, RGB = xyz)"),
             caption(normal_image(mask.shape, uv, nn["silhouette_normals_(image_only)"], crop), "silhouette normals (from the image only)")]
    for name in ("exact normals, plain least squares", "exact normals + 10 deg noise",
                 "silhouette normals (image only)", "exact normals + 10 deg noise, true discontinuities, parts placed (oracle)"):
        k = list(d["names"]).index(name)
        e = res[name]["ground_anchored"]
        tiles.append(caption(error_image(mask.shape, uv, d["depths"][k], z, crop),
                             f"{name[:58]}: {100*e['median_rel']:.1f}%"))
    grid(tiles, 3, "Synthetic pig: depth from integrated normals (error: blue 0% .. red 8%)", f"{OUT}/pig_normal_integration.png",
         "ground-anchored scale (lowest pixels on the table), no ground truth used except in the oracle panel")


if __name__ == "__main__":
    main()
