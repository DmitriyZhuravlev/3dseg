"""Recursive cuboids + triangulation on bike.mp4: the combination.

Everything is reused from earlier steps:
  * bike_solution.npz (video3d.py): camera, per-frame object poses, carved extent;
  * video_segments.py: per-frame object camera / working frame, voxel voting of boxes;
  * video_normals.py: KLT tracks triangulated in the object frame;
  * cuboids_global.py: the recursive-cuboid constraints solved jointly.

Per sampled (even) frame, three cuboid reconstructions on the same superpixels:
  1. greedy   - the original recursive propagation (segment3d.reconstruct);
  2. global   - all contact/continuity/seed constraints solved jointly (cuboids_global);
  3. global + points - as 2, plus every triangulated 3D point tracked in this frame anchors
     the segment it falls in: the scale that puts the segment's box front at the point's
     measured depth. Triangulation measures depth where there is texture; the cuboid
     constraints carry it to the segments without points.
Each method's boxes are fused across frames by voxel voting (tau chosen on fusion frames).
Finally the fused model of 3 is cut by free space (no voxel between a camera and a point it
saw), combining both cues.

Evaluation on held-out odd frames only: silhouette IoU and the distance of points
triangulated from odd frames (never used) to each model's surface.

    python video_cuboids.py --video bike.mp4 --background background.png --out docs/video3d
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np

import cuboids_global as cg
import segment3d as s3
import video3d as v3
import video_normals as vn
from lift3d import Camera
from video_segments import SOL, object_camera, object_vps, rotate_extent, votes_from_boxes, working_frame

_G = {}


def _init(sol_path, video, bg_path, frame_points, region_size, ruler):
    sol = np.load(sol_path)
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    pts = grid.centres[sol["occ"].ravel()]
    _G.update(cam=Camera(sol["K"], sol["R"], sol["C"]), poses=sol["poses"], cap=cv2.VideoCapture(video),
              bg=cv2.imread(bg_path), fp=frame_points, rs=region_size, ruler=ruler,
              main=(pts.min(0) - grid.size, pts.max(0) + grid.size))


def _frame(i):
    cap = _G["cap"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    f = cv2.resize(cap.read()[1], vn.SIZE, interpolation=cv2.INTER_AREA)
    m = v3.object_masks(f[None], _G["bg"], threshold=28, fill_holes=True, close=5)[0]
    return f, v3.strip_ground_shadow(m, low_frac=0.10, min_run_frac=0.05)


def _solve(i):
    t0 = time.time()
    f, m = _frame(i)
    if m.sum() < 2500:
        return i, None
    ocam, Q = working_frame(_G["cam"], _G["poses"][i])
    vps = object_vps(ocam)
    main = rotate_extent(Q, *_G["main"])
    out = {}
    try:
        r = s3.reconstruct(ocam, f, m, vps, region_size=_G["rs"], ruler=_G["ruler"], log=lambda *a: None, main_box=main)
        out["greedy"] = r["cuboids"]
        labels = r["labels"]
        kw = dict(labels=labels, main_box=main, constraints=("contact", "continuity"), log=lambda *a: None)
        out["global"] = cg.global_boxes(ocam, f, m, vps, **kw)["cuboids"]
        X = _G["fp"].get(int(i), np.zeros((0, 3)))
        anchors = None
        if len(X):
            uv, depth = ocam.project(X @ Q)              # object -> working frame
            anchors = (uv, depth)
        out["global + points"] = cg.global_boxes(ocam, f, m, vps, point_anchors=anchors, **kw)["cuboids"]
        out["n_points"] = len(X)
    except Exception as e:                               # a degenerate frame must not stop the run
        print("frame", i, "failed:", e, flush=True)
        return i, None
    for k in ("greedy", "global", "global + points"):    # back to object coordinates
        out[k] = [(np.asarray(c["bottom"]) @ Q.T, c["height"]) for c in out[k]]
    print(f"frame {i}: {len(out['greedy'])} boxes, {out['n_points']} points ({time.time() - t0:.0f}s)", flush=True)
    return i, out


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--solution", default=SOL)
    ap.add_argument("--out", default="docs/video3d")
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    T = len(poses)
    cams = {i: object_camera(cam, poses[i]) for i in range(T)}
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)

    # triangulation (even frames: used; odd frames: held out)
    sets = {}
    for name, idx in (("fit", range(0, T, 2)), ("held", range(1, T, 2))):
        tracks = vn.klt_tracks(vn.frames_and_masks(a.video, bg, idx))
        pts, view, err, used = vn.triangulate(tracks, cams)
        ok = vn.clean_points(pts, view, grid, sol["occ"])
        sets[name] = (pts[ok], [[i for i, _ in tracks[k]] for k, o in zip(used, ok) if o])
        print(name, int(ok.sum()), "points", f"{time.time() - t0:.0f}s", flush=True)
    fit_pts, fit_obs = sets["fit"]
    held_pts = sets["held"][0]
    frame_points = {}
    for X, frames in zip(fit_pts, fit_obs):
        for i in frames:
            frame_points.setdefault(i, []).append(X)
    frame_points = {i: np.array(v) for i, v in frame_points.items()}

    fuse = list(range(0, T, 2))[:: a.every // 2]
    tmp = tempfile.mkdtemp()
    bg_tmp = os.path.join(tmp, "background_half.png")
    cv2.imwrite(bg_tmp, bg)
    with Pool(a.workers, initializer=_init, initargs=(a.solution, a.video, bg_tmp, frame_points, 18, 30)) as pool:
        results = [r for r in pool.map(_solve, fuse, chunksize=1) if r[1]]
    shutil.rmtree(tmp, ignore_errors=True)
    n = len(results)
    print(n, "frames solved", f"{time.time() - t0:.0f}s", flush=True)

    test = list(range(1, T, 2))[::8]
    masks = {i: m for i, _, m in vn.frames_and_masks(a.video, bg, test)}
    fit_masks = {i: m for i, _, m in vn.frames_and_masks(a.video, bg, [r[0] for r in results][::3])}

    def iou_on(occ, ms):
        sp = v3.surface_points(occ, grid)
        if not len(sp):
            return 0.0
        return float(np.mean([v3.iou(v3.silhouette_of(cam, poses[i], sp, m.shape, grid.size), m) for i, m in ms.items() if m.sum() >= 2500]))

    def score(occ):
        d = vn.surface_distance(o3d, vn.occ_to_mesh(o3d, occ, grid), held_pts)
        return dict(held_out_iou=iou_on(occ, masks), held_out_point_distance_median_cm=float(100 * np.median(d)),
                    held_out_points_within_2cm=float(np.mean(d < 0.02)), held_out_points_within_5cm=float(np.mean(d < 0.05)),
                    voxels=int(occ.sum()))

    metrics = dict(frames_fused=n, fit_points=len(fit_pts), held_points=len(held_pts),
                   mean_points_per_frame=float(np.mean([r[1]["n_points"] for r in results])))
    occs = {}
    for method in ("greedy", "global", "global + points"):
        votes = votes_from_boxes(grid, [r[1][method] for r in results])
        fit = {tau: iou_on(votes >= max(1, round(tau * n)), fit_masks) for tau in (0.2, 0.3, 0.4, 0.5)}
        tau = max(fit, key=fit.get)
        occs[method] = votes >= max(1, round(tau * n))
        metrics[f"recursive cuboids, {method}"] = dict(tau=tau, **score(occs[method]))
    fs, removed = vn.free_space_carve(occs["global + points"], grid, fit_pts, fit_obs, cams)
    occs["global + points + free space"] = fs
    metrics["recursive cuboids, global + points + free space"] = dict(removed=removed, **score(fs))
    metrics["carving (reference)"] = score(sol["occ"])
    fsc, _ = vn.free_space_carve(sol["occ"], grid, fit_pts, fit_obs, cams)
    metrics["carving + free space (reference)"] = score(fsc)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k:52s} IoU {v['held_out_iou']:.3f}  points median {v['held_out_point_distance_median_cm']:.1f} cm  "
                  f"<5cm {100*v['held_out_points_within_5cm']:.0f}%", flush=True)
    metrics["seconds"] = round(time.time() - t0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(metrics, open(os.path.join(a.out, "bike_cuboids_points_metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "bike_cuboids_points_solution.npz"), lo=grid.lo, hi=grid.hi, size=grid.size,
                        points=fit_pts, held=held_pts, frames=np.array([r[0] for r in results]),
                        **{k.replace(" ", "_").replace("+", "and"): v for k, v in occs.items()})


if __name__ == "__main__":
    main()
