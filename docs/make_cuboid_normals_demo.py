"""Demo image for normals + recursive cuboids on a single photo (docs/segments/cuboid_normals.png).
Run from the repo root after cuboid_normals.py:

    PYTHONPATH=.:docs python docs/make_cuboid_normals_demo.py --eval-dir out/segeval
"""
import argparse
import json

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cuboid_normals as cn  # noqa: E402
import cuboids_global as cg  # noqa: E402
import lift3d  # noqa: E402
import normal_integration as ni  # noqa: E402
import segment3d as s3  # noqa: E402
from make_normals_demo import error_image, fig_to_pil, normal_image  # noqa: E402
from make_segment_demo import OUT, caption, grid  # noqa: E402


def cloud(cam, uv, z, img, title):
    d = cam.ray(uv)
    P = cam.C + d * (z / (d @ cam.R[2]))[:, None]
    ok = np.isfinite(P).all(1)
    P = P[ok]
    col = img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)][:, ::-1] / 255.0
    sub = np.arange(0, len(P), max(1, len(P) // 15000))
    fig = plt.figure(figsize=(5.6, 3.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[sub, 0], P[sub, 1], P[sub, 2], c=col[sub], s=1.5, depthshade=False)
    ax.set_box_aspect(np.ptp(P[sub], 0))
    ax.view_init(25, 30)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=9)
    return fig_to_pil(fig)


def row(name, cam, img, mask, vps, uv, z_gt, n, src, res, crop):
    labels, contour = s3.superpixels(img, mask)
    cub = cg.global_boxes(cam, img, mask, vps, labels=labels, constraints=("contact", "continuity"), log=lambda *a: None)
    z_cub = s3.depth_from_boxes(cam, uv, cg.boxes_of(cub))
    a_idx, a_z = ni.ground_anchor(cam, uv)
    z_n = ni.integrate_normals(cam, uv, n, a_idx, a_z, robust=False)
    z_c, _ = cn.combine(cam, img, mask, vps, uv, n, labels, cub)
    r = res[name]
    k_c = "recursive cuboids, global solve (no normals)"
    k_n = f"{src}: normals alone (whole image, ground-anchored)"
    k_b = f"{src}: normals in segments + cuboid placement + robust continuity"
    return [caption(normal_image(mask.shape, uv, n, crop), f"{name}: {src}"),
            caption(error_image(mask.shape, uv, z_cub, z_gt, crop), f"cuboids only: {100*r[k_c]['median_rel']:.1f}%"),
            caption(error_image(mask.shape, uv, z_n, z_gt, crop), f"normals only: {100*r[k_n]['median_rel']:.1f}%"),
            caption(error_image(mask.shape, uv, z_c, z_gt, crop), f"normals + cuboids: {100*r[k_b]['median_rel']:.1f}%"),
            caption(cloud(cam, uv, z_c, img, f"{name}: normals + cuboids, 3D"), "combined surface (3D)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    a = ap.parse_args()
    E = a.eval_dir.rstrip("/") + "/"
    res = json.load(open(f"{OUT}/cuboid_normals.json"))
    cam = lift3d.default_camera()
    vps = lift3d.vanishing_points()
    s3.SIGNS = s3.axis_signs(cam, vps)
    import synthetic3d
    # box.JPG with image-only normals (planar-patch orientation)
    img = cv2.imread("box.JPG")
    mask = lift3d.object_mask("reference.JPG", "box.JPG")
    labels, contour = s3.superpixels(img, mask)
    ys, xs = np.nonzero(labels[::4, ::4] > 0)
    uv = np.c_[xs * 4, ys * 4].astype(float)
    z_gt = s3.depth_from_boxes(cam, uv, [lift3d.box_ground_truth(cam)["corners"]])
    p = s3.patch_reconstruct(cam, img, mask, vps, labels, contour)
    n = cn.plane_normals(cam, uv, labels, p["planes"])
    n = np.where(np.isfinite(n).all(1)[:, None], n, np.array([0.0, 0.0, -1.0]))
    tiles = row("box", cam, img, mask, vps, uv, z_gt, n, "image only (planar-patch orientation)", res, (0, 1760, 250, 2850))
    # synthetic pig with image-only normals (silhouette)
    img = cv2.imread(E + "synth_pig.png")
    mask = np.load(E + "synth_pig_mask.npy")
    g = np.load(E + "synth_pig_gt.npz")
    ok = np.isfinite(g["z"])
    uv = g["uv"][ok]
    inside, _ = synthetic3d.make_pig()
    z_gt = ni.refine_hits(cam, uv, g["z"][ok], inside)
    n = ni.silhouette_normals(cam, mask, uv)
    tiles += row("pig", cam, img, mask, vps, uv, z_gt, n, "image only (silhouette normals)", res, (500, 1700, 700, 2500))
    grid(tiles, 5, "Single photo: normals give each segment's shape, recursive cuboids place the segments (error: blue 0% .. red 8%)",
         f"{OUT}/cuboid_normals.png", "image-only normals in both rows (no ground truth used); exact / noisy normals: see NOTES section 14")


if __name__ == "__main__":
    main()
