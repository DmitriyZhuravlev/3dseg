"""Single-frame 3D reconstruction of the motorcycle in bike.mp4 with recursive cuboids.

One video frame, the camera (calibrated once from the parking lines) and that frame's
ground pose; no other frame is used for the shape. Per frame:

  cone            - the frame's own silhouette cone (all a single silhouette gives:
                    unbounded in depth, cut to the grid)
  greedy          - the original recursive cuboids (segment3d.reconstruct), main box
                    fitted from this frame only
  global          - all cuboid constraints solved jointly (cuboids_global), main box from
                    this frame only
  global ∩ cone   - the global cuboids cut by the frame's own silhouette cone
  global, class-size prior (∩ cone) - main box = a typical motorcycle-with-rider size
                    (2.2 x 0.9 x 1.5 m) at the frame's ground pose, instead of a box fitted
                    to the one silhouette (the SpringerLifting chapter's average size per
                    object type); a prior about the class, not about this video
  global, carved extent (reference, not single-frame) - main box from the multi-frame carving

Each single-frame model is scored on the held-out frames (odd frames, never used):
silhouette IoU from the other viewpoints and median distance of held-out triangulated
points to its surface. Also: the multi-frame carving for reference.

    python video_single_frame.py --video data/bike.mp4 --background bg.png \\
        --solution out/bike/bike_solution.npz --cache out/cuboids_cache.pkl
"""
import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import cv2
import numpy as np

import cuboids_global as cg
import segment3d as s3
import video3d as v3
import video_normals as vn
from lift3d import Camera
from video_segments import object_vps, rotate_extent, votes_from_boxes, working_frame

_G = {}
CLASS_PRIOR = (2.2, 0.9, 1.5)   # typical motorcycle with rider, m (article: average size per object type)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "single_frame")


def _init(sol_path, video, bg):
    sol = np.load(sol_path)
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    pts = grid.centres[sol["occ"].ravel()]
    _G.update(cam=Camera(sol["K"], sol["R"], sol["C"]), poses=sol["poses"], video=video, bg=bg,
              carved=(pts.min(0) - grid.size, pts.max(0) + grid.size))


def frame_and_mask(video, bg, i):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    f = cv2.resize(cap.read()[1], vn.SIZE, interpolation=cv2.INTER_AREA)
    m = v3.object_masks(f[None], bg, threshold=28, fill_holes=True, close=5)[0]
    return f, v3.strip_ground_shadow(m, low_frac=0.10, min_run_frac=0.05)


def _solve(i):
    t0 = time.time()
    f, m = frame_and_mask(_G["video"], _G["bg"], i)
    if m.sum() < 2500:
        return i, None
    ocam, Q = working_frame(_G["cam"], _G["poses"][i])
    vps = object_vps(ocam)
    out = {}
    quiet = lambda *a: None  # noqa: E731
    try:
        r = s3.reconstruct(ocam, f, m, vps, region_size=18, ruler=30, log=quiet)
        out["greedy"] = r["cuboids"]
        kw = dict(labels=r["labels"], constraints=("contact", "continuity"), log=quiet)
        out["global"] = cg.global_boxes(ocam, f, m, vps, **kw)["cuboids"]
        L, W, H = CLASS_PRIOR
        prior = (np.array([-L / 2, -W / 2, 0.0]), np.array([L / 2, W / 2, H]))
        out["global, class-size prior"] = cg.global_boxes(ocam, f, m, vps, main_box=rotate_extent(Q, *prior), **kw)["cuboids"]
        out["global, carved extent"] = cg.global_boxes(ocam, f, m, vps, main_box=rotate_extent(Q, *_G["carved"]),
                                                      **kw)["cuboids"]
    except Exception as e:                               # degenerate view (head-on): no boxes
        print("frame", i, "failed:", e, flush=True)
        return i, None
    for k in list(out):
        out[k] = [(np.asarray(c["bottom"]) @ Q.T, c["height"]) for c in out[k]]
    print(f"frame {i}: {len(out['global'])} boxes ({time.time() - t0:.0f}s)", flush=True)
    return i, out


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--cache", required=True, help="video_cuboids_parts.py cache (held-out triangulated points)")
    ap.add_argument("--frames", type=int, default=24, help="number of single frames to try")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    T = len(poses)
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)
    held_pts = pickle.load(open(a.cache, "rb"))["held_pts"]
    cand = list(range(0, T, 2))[:: max(1, (T // 2) // a.frames)]
    with Pool(a.workers, initializer=_init, initargs=(a.solution, a.video, bg)) as pool:
        results = [r for r in pool.map(_solve, cand, chunksize=1) if r[1] and len(r[1]["global"])]
    print(len(results), "of", len(cand), "frames give cuboids", f"{time.time() - t0:.0f}s", flush=True)

    test = list(range(1, T, 2))[::8]
    masks = {i: m for i, _, m in vn.frames_and_masks(a.video, bg, test) if m.sum() >= 2500}

    def score(occ):
        sp = v3.surface_points(occ, grid)
        if occ.sum() < 10 or not len(sp):
            return dict(held_out_iou=0.0, held_out_point_distance_median_cm=float("nan"), voxels=0)
        ious = [v3.iou(v3.silhouette_of(cam, poses[i], sp, m.shape, grid.size), m) for i, m in masks.items()]
        try:
            d = 100 * np.median(vn.surface_distance(o3d, vn.occ_to_mesh(o3d, occ, grid), held_pts))
        except ValueError:                               # no surface inside the grid (fills it)
            d = float("nan")
        return dict(held_out_iou=float(np.mean(ious)), held_out_point_distance_median_cm=float(d),
                    voxels=int(occ.sum()))

    per_frame, models = {}, {}
    for i, out in results:
        _, m = frame_and_mask(a.video, bg, i)
        frac, seen = v3.carve(cam, m[None], poses[i:i + 1], grid, [0], margin_px=1)
        cone = (frac >= 1.0) & (seen >= 1)
        occ = {"cone": cone}
        for k in ("greedy", "global", "global, class-size prior", "global, carved extent"):
            occ[k] = votes_from_boxes(grid, [out[k]]) > 0
        occ["global ∩ cone"] = occ["global"] & cone
        occ["global, class-size prior ∩ cone"] = occ["global, class-size prior"] & cone
        per_frame[i] = {k: score(v) for k, v in occ.items()}
        models[i] = occ
        print(i, {k: round(v["held_out_iou"], 3) for k, v in per_frame[i].items()}, flush=True)
    names = list(next(iter(per_frame.values())))
    summary = {k: dict(held_out_iou_median=float(np.median([p[k]["held_out_iou"] for p in per_frame.values()])),
                       held_out_iou_best=float(np.max([p[k]["held_out_iou"] for p in per_frame.values()])),
                       point_distance_cm_median=float(np.nanmedian([p[k]["held_out_point_distance_median_cm"]
                                                                    for p in per_frame.values()])))
               for k in names}
    summary["carving, all frames (reference)"] = score(sol["occ"])
    print(json.dumps(summary, indent=1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(frames_tried=len(cand), frames_with_cuboids=len(results), summary=summary,
                   per_frame={int(k): v for k, v in per_frame.items()}), open(os.path.join(OUT, "bike_single_frame.json"), "w"),
              indent=1)
    key = "global, class-size prior ∩ cone"
    best = max(per_frame, key=lambda i: per_frame[i][key]["held_out_iou"])
    med = sorted(per_frame, key=lambda i: per_frame[i][key]["held_out_iou"])[len(per_frame) // 2]
    np.savez_compressed(os.path.join(OUT, "bike_single_frame_models.npz"), best=best, median=med,
                        **{f"{i}|{k}": v for i in (best, med) for k, v in models[i].items()},
                        lo=grid.lo, hi=grid.hi, size=grid.size)


if __name__ == "__main__":
    main()
