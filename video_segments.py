"""3D from video with the original recursive segment-box method (segment3d.py).

Reuses the precomputed bike.mp4 solution of video3d.py (docs/video3d/bike_solution.npz):
the calibrated camera (from the parking lines' vanishing point), every frame's object
pose and the carved model's extent. For each sampled frame:

  1. the camera is expressed in the motorcycle's own frame (pose incl. lean), which
     gives the three vanishing points of the object's axes for that frame;
  2. segment3d.reconstruct runs unchanged: superpixels, VP-tangent box per superpixel,
     seeds at the extreme points on the faces of the main box (here: the carved
     model's extent), recursive propagation between neighbours;
  3. the boxes are already in the object frame, so frames are fused by voting in a
     voxel grid: a voxel is kept when at least `tau` of the frames' boxes contain it
     (tau chosen on the frames used for fusion).

Held-out frames (never used) are scored by silhouette IoU, as for the carving.

    python video_segments.py --video bike.mp4 --out docs/video3d
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

import segment3d as s3
import video3d as v3
from lift3d import Camera

SOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "video3d", "bike_solution.npz")


QZ = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # 90 deg about the object's up axis


def object_camera(cam, pose, Q=np.eye(3)):
    """The camera expressed in the object's frame (X forward, Y lateral, Z the object's up),
    optionally in a frame rotated by Q (object point = Q @ working point)."""
    T = v3.pose_matrix(*pose)
    Rp, tp = T[:3, :3], T[:3, 3]
    return Camera(cam.K, cam.R @ Rp @ Q, Q.T @ Rp.T @ (cam.C - tp))


def working_frame(cam, pose):
    """The original 2D tangent-box construction expects the X vanishing point left of the Y one.
    When the object faces the other way, work in the frame rotated by 90 deg about the up axis
    (this swaps the two VPs); axis-aligned boxes stay axis-aligned under that rotation."""
    oc = object_camera(cam, pose)
    v = object_vps(oc)
    if v["left"][0] <= v["right"][0]:
        return oc, np.eye(3)
    return object_camera(cam, pose, QZ), QZ


def rotate_extent(Q, lo, hi):
    """Axis-aligned extent (lo, hi) in object coordinates -> the working frame (p_work = Q.T @ p_obj)."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ Q
    return c.min(0), c.max(0)


def object_vps(ocam):
    """Vanishing points of the object's X, Y and Z axes in this frame (keys as in lift3d)."""
    out = {}
    for key, e in (("left", [1, 0, 0]), ("right", [0, 1, 0]), ("vertical", [0, 0, -1])):
        p = ocam.K @ ocam.R @ np.array(e, float)
        out[key] = p[:2] / (p[2] if abs(p[2]) > 1e-12 else 1e-12)
    return out


_G = {}


def _init(sol_path, video, bg_path, region_size, ruler):
    sol = np.load(sol_path)
    _G.update(cam=Camera(sol["K"], sol["R"], sol["C"]), poses=sol["poses"], region_size=region_size, ruler=ruler,
              cap=cv2.VideoCapture(video), bg=cv2.imread(bg_path))
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    pts = grid.centres[sol["occ"].ravel()]
    _G["main"] = (pts.min(0) - grid.size, pts.max(0) + grid.size)


def _frame(i):
    cap = _G["cap"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    f = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA)
    m = v3.object_masks(f[None], _G["bg"], threshold=28, fill_holes=True, close=5)[0]
    return f, v3.strip_ground_shadow(m, low_frac=0.10, min_run_frac=0.05)


def _solve(i):
    t0 = time.time()
    f, m = _frame(i)
    if m.sum() < 2500:
        return i, None, 0, 0
    ocam, Q = working_frame(_G["cam"], _G["poses"][i])
    try:
        r = s3.reconstruct(ocam, f, m, object_vps(ocam), region_size=_G["region_size"], ruler=_G["ruler"],
                           log=lambda *a: None, main_box=rotate_extent(Q, *_G["main"]))
    except Exception as e:                                   # a degenerate frame must not stop the run
        print("frame", i, "failed:", e, flush=True)
        return i, None, 0, 0
    boxes = [(np.asarray(c["bottom"]) @ Q.T, c["height"]) for c in r["cuboids"]]    # back to object coordinates
    print(f"frame {i}: {len(boxes)}/{r['n_segments']} segments ({time.time() - t0:.0f}s)", flush=True)
    return i, boxes, len(boxes), r["n_segments"]


def votes_from_boxes(grid, frame_boxes):
    """Per frame, the union of its (object-frame, axis-aligned) boxes; summed over frames."""
    votes = np.zeros(grid.shape, np.int32)
    for boxes in frame_boxes:
        occ = np.zeros(grid.shape, bool)
        for bottom, h in boxes:
            lo = np.r_[bottom[:, :2].min(0), bottom[:, 2].min()]
            hi = np.r_[bottom[:, :2].max(0), bottom[:, 2].min() + h]
            a = np.clip(np.floor((lo - grid.lo) / grid.size).astype(int), 0, np.array(grid.shape) - 1)
            b = np.clip(np.ceil((hi - grid.lo) / grid.size).astype(int), 0, np.array(grid.shape))
            occ[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
        votes += occ
    return votes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--solution", default=SOL)
    ap.add_argument("--background", required=True, help="median background frame (full resolution)")
    ap.add_argument("--out", default="docs/video3d")
    ap.add_argument("--every", type=int, default=16, help="use every N-th even frame")
    ap.add_argument("--region-size", type=int, default=18)
    ap.add_argument("--ruler", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    T = len(poses)
    fuse = list(range(0, T, 2))[:: a.every // 2]              # even frames only: odd frames stay held out
    test = list(range(1, T, 2))[::8]
    bg_half = cv2.resize(cv2.imread(a.background), (960, 540), interpolation=cv2.INTER_AREA)
    tmp = tempfile.mkdtemp()
    bg_tmp = os.path.join(tmp, "background_half.png")
    cv2.imwrite(bg_tmp, bg_half)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.solution, a.video, bg_tmp, a.region_size, a.ruler)) as pool:
        results = pool.map(_solve, fuse, chunksize=1)
    shutil.rmtree(tmp, ignore_errors=True)
    results = [r for r in results if r[1]]
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    votes = votes_from_boxes(grid, [r[1] for r in results])
    n = len(results)

    # silhouettes of every frame (fusion frames and held-out ones)
    _init(a.solution, a.video, a.background, a.region_size, a.ruler)
    _G["bg"] = bg_half

    def score(occ, frames):
        pts = v3.surface_points(occ, grid)
        if not len(pts):
            return 0.0
        ious = []
        for i in frames:
            f, m = _frame(i)
            if m.sum() >= 2500:
                ious.append(v3.iou(v3.silhouette_of(cam, poses[i], pts, m.shape, grid.size), m))
        return float(np.mean(ious))

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    fit = {tau: score(votes >= max(1, round(tau * n)), [r[0] for r in results][::3]) for tau in taus}
    tau = max(fit, key=fit.get)
    occ = votes >= max(1, round(tau * n))
    held = score(occ, test)
    single = votes_from_boxes(grid, [results[len(results) // 2][1]]) >= 1
    held_single = score(single, test)
    carved = sol["occ"]
    held_carved = score(carved, test)
    metrics = dict(frames_fused=n, segments_solved=int(sum(r[2] for r in results)), segments_total=int(sum(r[3] for r in results)),
                   tau=tau, tau_fit=fit, held_out_iou=held, held_out_iou_single_frame=held_single,
                   held_out_iou_carving=held_carved, n_held_out=len(test), seconds=round(time.time() - t0))
    verts, faces, normals = v3.occupancy_mesh(occ, grid)
    np.savez_compressed(os.path.join(a.out, "bike_segments_solution.npz"), occ=occ, votes=votes, lo=grid.lo, hi=grid.hi, size=grid.size)
    v3.save_ply(os.path.join(a.out, "bike_segments_model.ply"), verts, faces, np.full((len(verts), 3), 0.7))
    json.dump(metrics, open(os.path.join(a.out, "bike_segments_metrics.json"), "w"), indent=1)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
