"""Demo images for the recursive segment methods (docs/segments/). Run from the repo root:
    PYTHONPATH=. python docs/make_segment_demo.py --eval-dir out/segeval --video bike.mp4 --background bg.png
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import lift3d
import segment3d as s3
import video3d as v3
from video_demos import model_views
from video_segments import object_vps, rotate_extent, working_frame

B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
OUT = "docs/segments"


def caption(img_bgr_or_pil, text, size=(560, 380)):
    im = img_bgr_or_pil if isinstance(img_bgr_or_pil, Image.Image) else Image.fromarray(cv2.cvtColor(img_bgr_or_pil, cv2.COLOR_BGR2RGB))
    im = im.convert("RGB")
    im.thumbnail(size)
    t = Image.new("RGB", size, (28, 30, 36))
    t.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, d.textlength(text, font=S) + 12, 22], fill=(0, 0, 0))
    d.text((6, 3), text, fill="white", font=S)
    return t


def grid(tiles, cols, title, path, footer=""):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, 44 + rows * h + (32 if footer else 0)), (40, 42, 50))
    d = ImageDraw.Draw(out)
    d.text((10, 10), title, fill="white", font=B)
    for k, t in enumerate(tiles):
        out.paste(t, ((k % cols) * w, 44 + (k // cols) * h))
    if footer:
        d.text((10, 44 + rows * h + 7), footer, fill=(230, 230, 230), font=S)
    out.save(path)


def error_map(image, uv, z, z_gt, crop, vmax=0.08):
    err = np.abs(z - z_gt) / z_gt
    v = (image * 0.35).astype(np.uint8)
    ok = np.isfinite(err)
    col = cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(err[ok]) / vmax, 0, 1) * 255)[:, None], cv2.COLORMAP_JET)[:, 0]
    for (x, y), c in zip(uv[ok].astype(int), col):
        cv2.rectangle(v, (x, y), (x + 3, y + 3), tuple(int(t) for t in c), -1)
    y0, y1, x0, x1 = crop
    return v[y0:y1, x0:x1]


def plane_map(image, labels, planes, crop):
    cols = {0: (60, 60, 255), 1: (60, 220, 60), 2: (255, 140, 60)}
    v = image.copy()
    for s, (axis, _) in planes.items():
        m = labels == s
        v[m] = (0.45 * image[m] + 0.55 * np.array(cols[axis])).astype(np.uint8)
    y0, y1, x0, x1 = crop
    return v[y0:y1, x0:x1]


def methods(cam, img, mask, vps, labels, contour, uv_gt, z_gt, grid_align=False):
    def on_grid(uv, z):
        m = {(int(a), int(b)): v for (a, b), v in zip(uv, z)}
        return np.array([m.get((int(a), int(b)), np.nan) for a, b in uv_gt])
    r = s3.reconstruct(cam, img, mask, vps, log=lambda *a: None)
    boxes = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in r["cuboids"]]
    z_box = s3.depth_from_boxes(cam, uv_gt, boxes)
    p = s3.patch_reconstruct(cam, img, mask, vps, labels, contour)
    z_patch = on_grid(*s3.depth_from_planes(cam, labels, p["planes"]))
    pix = s3.segment_pixels(labels, p["ids"])
    anchors = {}
    for s in p["seeds"]:
        if s in p["planes"]:
            P, _ = s3.backproject_plane(cam, pix[s], *p["planes"][s])
            anchors[s] = (pix[s], s3.camera_depth(cam, P))
    z_m3d = on_grid(*s3.depth_from_make3d(cam, labels, s3.make3d_planes(cam, img, labels, anchors)))
    return r, p, {"recursive boxes": z_box, "recursive planar patches": z_patch, "Make3D-style MRF": z_m3d}


def object_demo(name, img, mask, uv_gt, z_gt, crop, res_json):
    cam = lift3d.default_camera()
    vps = lift3d.vanishing_points()
    s3.SIGNS = s3.axis_signs(cam, vps)
    labels, contour = s3.superpixels(img, mask)
    r, p, depths = methods(cam, img, mask, vps, labels, contour, uv_gt, z_gt)
    res = json.load(open(res_json))
    key = {"recursive boxes": "recursive boxes (original method, calibrated)", "recursive planar patches": "recursive planar patches (fixed)",
           "Make3D-style MRF": "Make3D-style MRF (same anchors, no learned term)"}
    y0, y1, x0, x1 = crop
    tiles = [caption(img[y0:y1, x0:x1], f"{name}: input"),
             caption(s3.draw_overlay(img, cam, r["labels"], r["cuboids"])[y0:y1, x0:x1], "recursive boxes (calibrated original)"),
             caption(plane_map(img, labels, p["planes"], crop), "planar patches: red X, green Y, blue Z")]
    for k, z in depths.items():
        e = res[key[k]]
        tiles.append(caption(error_map(img, uv_gt, z, z_gt, crop), f"{k}: median {100*e['median_rel']:.1f}%, <=3% {100*e['within_3pct']:.0f}%"))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--skip-video", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    E = a.eval_dir.rstrip("/") + "/"
    cam = lift3d.default_camera()
    # 1. box photo
    img = cv2.imread("box.JPG")
    mask = lift3d.object_mask("reference.JPG", "box.JPG")
    labels, _ = s3.superpixels(img, mask)
    ys, xs = np.nonzero(labels[::4, ::4] > 0)
    uv = np.c_[xs * 4, ys * 4].astype(float)
    z_gt = s3.depth_from_boxes(cam, uv, [lift3d.box_ground_truth(cam)["corners"]])
    tiles = object_demo("box.JPG", img, mask, uv, z_gt, (0, 1760, 250, 2850), E + "compare_box.json")
    grid(tiles, 3, "box.JPG: recursive segment methods vs traced 3D box (depth error: blue 0% .. red 8%)", f"{OUT}/box_methods.png")
    # 2. synthetic pig
    img = cv2.imread(E + "synth_pig.png")
    mask = np.load(E + "synth_pig_mask.npy")
    g = np.load(E + "synth_pig_gt.npz")
    tiles = object_demo("synthetic pig", img, mask, g["uv"], g["z"], (500, 1700, 700, 2500), E + "compare_pig.json")
    grid(tiles, 3, "Synthetic pig (exact 3D known): recursive segment methods (depth error: blue 0% .. red 8%)", f"{OUT}/pig_methods.png")
    if a.skip_video:
        return
    # 3. bike video: per-frame recursive boxes and the fused model
    sol = np.load("docs/video3d/bike_solution.npz")
    seg = np.load("docs/video3d/bike_segments_solution.npz")
    met = json.load(open("docs/video3d/bike_segments_metrics.json"))
    bcam = lift3d.Camera(sol["K"], sol["R"], sol["C"])
    bg = cv2.resize(cv2.imread(a.background), (960, 540), interpolation=cv2.INTER_AREA)
    cap = cv2.VideoCapture(a.video)
    tiles = []
    for i in (64, 176, 400):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        f = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA)
        m = v3.strip_ground_shadow(v3.object_masks(f[None], bg, threshold=28, fill_holes=True, close=5)[0], 0.10, 0.05)
        oc, Q = working_frame(bcam, sol["poses"][i])
        grd = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
        pts = grd.centres[sol["occ"].ravel()]
        r = s3.reconstruct(oc, f, m, object_vps(oc), region_size=18, ruler=30, log=lambda *a: None,
                           main_box=rotate_extent(Q, pts.min(0) - grd.size, pts.max(0) + grd.size))
        ys, xs = np.nonzero(m)
        crop = (max(0, ys.min() - 30), ys.max() + 30, max(0, xs.min() - 60), xs.max() + 60)
        tiles.append(caption(s3.draw_overlay(f, oc, r["labels"], r["cuboids"])[crop[0]:crop[1], crop[2]:crop[3]],
                             f"frame {i}: {len(r['cuboids'])}/{r['n_segments']} segment boxes"))
    grd = v3.VoxelGrid(seg["lo"], seg["hi"], float(seg["size"]))
    for occ, label in ((seg["occ"], f"fused recursive boxes ({met['frames_fused']} frames)"), (sol["occ"], "carved silhouettes (reference)")):
        verts, faces, normals = v3.occupancy_mesh(occ, grd)
        for yaw, view in ((0.0, "side"), (0.8, "3/4"), (1.57, "front")):
            img_v = model_views(verts, faces, normals, np.full((len(verts), 3), 0.75), yaws=(yaw,), size=(560, 380), zoom=0.5)[0]
            tiles.append(caption(img_v, f"{label}: {view}"))
    foot = (f"held-out silhouette IoU: fused recursive boxes {met['held_out_iou']:.3f}, one frame {met['held_out_iou_single_frame']:.3f}, "
            f"carving {met['held_out_iou_carving']:.3f}  ({met['n_held_out']} held-out frames)")
    grid(tiles, 3, "bike.mp4: original recursive box method per frame, fused in the motorcycle's frame",
         f"{OUT}/bike_recursive_boxes.png", foot)


if __name__ == "__main__":
    main()
