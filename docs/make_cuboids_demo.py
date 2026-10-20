"""Demo images for the improved (global-solve) cuboids and the cuboids + triangulation
combination (docs/segments/). Run from the repo root:

    PYTHONPATH=.:docs python docs/make_cuboids_demo.py --eval-dir out/segeval \
        [--video bike.mp4 --background background.png --combo-dir out/vcub]
"""
import argparse
import json
import os

import cv2
import numpy as np

import cuboids_global as cg
import lift3d
import segment3d as s3
from make_segment_demo import OUT, caption, error_map, grid


def photo_demo(name, img, mask, uv, z_gt, crop, shape, res, path):
    cam = lift3d.default_camera()
    vps = lift3d.vanishing_points()
    s3.SIGNS = s3.axis_signs(cam, vps)
    labels, _ = s3.superpixels(img, mask)
    L = lift3d.lift_mask(cam, mask, img, shape=shape)
    runs = [("original recursive boxes (greedy propagation)", "greedy recursion",
             lambda: s3.reconstruct(cam, img, mask, vps, log=lambda *a: None)),
            ("global: contact + continuity", "global solve",
             lambda: cg.global_boxes(cam, img, mask, vps, labels=labels, constraints=("contact", "continuity"), log=lambda *a: None)),
            ("global: contact + continuity + whole-object prior", "global solve + whole-object prior",
             lambda: cg.global_boxes(cam, img, mask, vps, labels=labels, constraints=("contact", "continuity"),
                                     prior_depth=cg.model_depth_fn(cam, L["cuboids"]), log=lambda *a: None))]
    y0, y1, x0, x1 = crop
    over, errs = [], []
    for key, label, fn in runs:
        r = fn()
        e = res[key]
        z = s3.depth_from_boxes(cam, uv, cg.boxes_of(r))
        over.append(caption(s3.draw_overlay(img, cam, r["labels"], r["cuboids"])[y0:y1, x0:x1], f"{label}: boxes"))
        errs.append(caption(error_map(img, uv, z, z_gt, crop), f"{label}: median {100*e['median_rel']:.1f}%, <=3% {100*e['within_3pct']:.0f}%"))
    grid(over + errs, 3, f"{name}: recursive cuboids, greedy vs global solve (depth error: blue 0% .. red 8%)", path,
         "global solve: one scale per segment (lifts are homothetic about the camera); contact, continuity and seeds solved jointly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--video")
    ap.add_argument("--background")
    ap.add_argument("--combo-dir")
    a = ap.parse_args()
    E = a.eval_dir.rstrip("/") + "/"
    cam = lift3d.default_camera()
    if not os.path.exists(f"{OUT}/cuboids_global_box.png"):
        img = cv2.imread("box.JPG")
        mask = lift3d.object_mask("reference.JPG", "box.JPG")
        labels, _ = s3.superpixels(img, mask)
        ys, xs = np.nonzero(labels[::4, ::4] > 0)
        uv = np.c_[xs * 4, ys * 4].astype(float)
        z_gt = s3.depth_from_boxes(cam, uv, [lift3d.box_ground_truth(cam)["corners"]])
        photo_demo("box.JPG", img, mask, uv, z_gt, (0, 1760, 250, 2850), "box", json.load(open(f"{OUT}/variants_box.json")),
                   f"{OUT}/cuboids_global_box.png")
        g = np.load(E + "synth_pig_gt.npz")
        photo_demo("synthetic pig", cv2.imread(E + "synth_pig.png"), np.load(E + "synth_pig_mask.npy"), g["uv"], g["z"],
                   (500, 1700, 700, 2500), "round", json.load(open(f"{OUT}/variants_pig.json")), f"{OUT}/cuboids_global_pig.png")
    if a.combo_dir:
        combo_demo(a)


def combo_demo(a):
    import video3d as v3
    import video_normals as vn
    from lift3d import Camera
    from video_cuboids import _init, _solve
    from video_demos import model_views
    from video_segments import SOL, working_frame
    C = a.combo_dir.rstrip("/") + "/"
    sol = np.load(SOL)
    res = np.load(C + "bike_cuboids_points_solution.npz")
    met = json.load(open(C + "bike_cuboids_points_metrics.json"))
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    grd = v3.VoxelGrid(res["lo"], res["hi"], float(res["size"]))
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)
    bg_path = C + "_bg.png"
    cv2.imwrite(bg_path, bg)
    # one frame: its triangulated points (from tracks through this frame) anchor the segments
    i0 = 176
    idx = list(range(i0 - 30, i0 + 32, 2))
    cams = {i: __import__("video_segments").object_camera(cam, poses[i]) for i in idx}
    tr = vn.klt_tracks(vn.frames_and_masks(a.video, bg, idx))
    pts, view, err, used = vn.triangulate(tr, cams)
    ok = vn.clean_points(pts, view, grd, sol["occ"])
    X = np.array([p for p, k, o in zip(pts, used, ok) if o and any(i == i0 for i, _ in tr[k])]).reshape(-1, 3)
    _init(SOL, a.video, bg_path, {i0: X}, 18, 30)
    os.remove(bg_path)
    _, out = _solve(i0)
    f, m = __import__("video_cuboids")._frame(i0)
    ocam, Q = working_frame(cam, poses[i0])
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    ys, xs = np.nonzero(cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))
    crop = (max(0, ys.min() - 30), ys.max() + 30, max(0, xs.min() - 60), xs.max() + 60)
    labels, _ = s3.superpixels(f, m, 18, 30)
    tiles = []
    for key in ("greedy", "global + points"):
        cub = [{"bottom": (b @ Q).tolist(), "height": h, "segment": 0} for b, h in out[key]]
        ov = s3.draw_overlay(f, ocam, np.full_like(labels, -1), cub)
        if key != "greedy":
            uv, _ = ocam.project(X @ Q)
            for p in uv.astype(int):
                cv2.circle(ov, tuple(p), 3, (0, 0, 255), -1)
        tiles.append(caption(ov[crop[0]:crop[1], crop[2]:crop[3]],
                             f"frame {i0}: {key}" + (f" ({len(X)} triangulated points, red)" if key != "greedy" else "")))
    names = [("greedy", "recursive cuboids, greedy"), ("global_and_points", "recursive cuboids, global + points"),
             ("global_and_points_and_free_space", "recursive cuboids, global + points + free space"),
             ("carving", "carving (reference)")]
    for key, label in names:
        occ = sol["occ"] if key == "carving" else res[key]
        verts, faces, normals = v3.occupancy_mesh(occ, grd)
        img = model_views(verts, faces, normals, np.full((len(verts), 3), 0.75), yaws=(0.8,), size=(560, 380), zoom=0.5)[0]
        r = met[label]
        tiles.append(caption(img, f"{label.replace('recursive cuboids, ', '')}: IoU {r['held_out_iou']:.3f}, "
                                  f"pts {r['held_out_point_distance_median_cm']:.1f} cm"))
    m_ = {k: met[f"recursive cuboids, {k}"] for k in ("greedy", "global", "global + points", "global + points + free space")}
    foot = "held-out IoU / point distance: " + ", ".join(f"{k} {v['held_out_iou']:.3f} / {v['held_out_point_distance_median_cm']:.1f} cm"
                                                          for k, v in m_.items())
    grid([tiles[0], tiles[1], tiles[5], tiles[2], tiles[3], tiles[4]], 3,
         "bike.mp4: recursive cuboids anchored by triangulated points, fused over frames",
         f"{OUT}/bike_cuboids_points.png", foot)


if __name__ == "__main__":
    main()
