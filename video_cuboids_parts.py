"""Recursive cuboids + triangulation, fused per motion part (bike.mp4).

video_cuboids.py fuses every frame's recursive cuboids by voxel voting in the
motorcycle's frame, which assumes the motorcycle is one rigid body. parts3d.py shows it
is not: the front wheel and fork (steering) move relative to the frame. Two
improvements use the parts:

  per-part tau  - the voting threshold is chosen for each part on the fusion frames,
                  not once for the whole object (a part that moves relative to the
                  body collects fewer consistent votes in the body frame);
  articulated   - each fusion frame's steering angle (rotation of the front assembly
                  about a vertical axis through its centre) is estimated from that
                  frame's silhouette, and the frame's votes in the front assembly are
                  rotated back before voting, so all frames vote in the part's own
                  frame.

Finally the part-aware cuboid model is intersected with the silhouette carving (the
visual hull bounds the object; the cuboids remove what no silhouette can: concavities
seen face-on).

Evaluation as in video_cuboids.py, on held-out odd frames: silhouette IoU (every model
gets the same per-frame steering-angle fit, so the comparison isolates the fusion) and
the distance of points triangulated from held-out frames to each model's surface.

    python video_cuboids_parts.py --video data/bike.mp4 --background bg.png \\
        --solution out/bike/bike_solution.npz --labels docs/parts3d/bike_parts3d_labels.npz
"""
import argparse
import json
import os
import pickle
import shutil
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial import cKDTree

import video3d as v3
import video_cuboids as vc
import video_normals as vn
from lift3d import Camera
from video_segments import object_camera, votes_from_boxes

FRONT_X = 0.4            # parts whose centre is ahead of this (m, object frame) form the front assembly
ANGLES = np.radians(np.arange(-30, 31, 3))


def part_labels(grid, occ, labels_npz):
    """Part id for every voxel of `grid` (nearest labelled voxel of the parts3d result)."""
    L = np.load(labels_npz)
    lg = v3.VoxelGrid(L["lo"], L["hi"], float(L["size"]))
    lab = L["labels"].ravel()
    have = lab >= 0
    _, nn = cKDTree(lg.centres[have]).query(grid.centres)
    return lab[have][nn].reshape(grid.shape)


def front_mask(grid, labels):
    ids = [k for k in np.unique(labels)
           if grid.centres[labels.ravel() == k, 0].mean() > FRONT_X and k != 0]
    return np.isin(labels, ids), ids


def rotate_grid(vol, grid, mask, angle, centre):
    """vol with the voxels in `mask` replaced by vol sampled at the rotated position
    (rotation about the vertical axis through `centre`)."""
    if angle == 0:
        return vol
    out = vol.copy()
    idx = np.nonzero(mask.ravel())[0]
    p = grid.centres[idx] - centre
    c, s = np.cos(angle), np.sin(angle)
    q = np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]]) + centre
    ijk = np.floor((q - grid.lo) / grid.size).astype(int)
    ok = np.all((ijk >= 0) & (ijk < grid.shape), 1)
    flat = out.ravel()
    flat[idx] = 0
    flat[idx[ok]] = vol.ravel()[np.ravel_multi_index(ijk[ok].T, grid.shape)]
    return flat.reshape(grid.shape)


def steer(occ, grid, front, centre, angle):
    """The model with its front assembly rotated by `angle` (object frame, as a pose)."""
    if angle == 0:
        return occ
    out = occ & ~front
    idx = np.nonzero((occ & front).ravel())[0]
    p = grid.centres[idx] - centre
    c, s = np.cos(angle), np.sin(angle)
    q = np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]]) + centre
    ijk = np.floor((q - grid.lo) / grid.size).astype(int)
    ok = np.all((ijk >= 0) & (ijk < grid.shape), 1)
    flat = out.ravel()
    flat[np.ravel_multi_index(ijk[ok].T, grid.shape)] = True
    return flat.reshape(grid.shape)


def best_angle(cam, pose, occ, grid, front, centre, mask):
    best = (-1, 0.0)
    for a in ANGLES:
        sp = v3.surface_points(steer(occ, grid, front, centre, a), grid)
        i = v3.iou(v3.silhouette_of(cam, pose, sp, mask.shape, grid.size), mask)
        if i > best[0]:
            best = (i, a)
    return best


def frame_votes(grid, boxes):
    return votes_from_boxes(grid, [boxes]) > 0


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="docs/parts3d")
    ap.add_argument("--cache", default="out/cuboids_cache.pkl")
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    T = len(poses)
    bg = cv2.resize(cv2.imread(a.background), vn.SIZE, interpolation=cv2.INTER_AREA)

    if os.path.exists(a.cache):
        C = pickle.load(open(a.cache, "rb"))
    else:                                                # the expensive part, as in video_cuboids.py
        cams = {i: object_camera(cam, poses[i]) for i in range(T)}
        sets = {}
        for name, idx in (("fit", range(0, T, 2)), ("held", range(1, T, 2))):
            tracks = vn.klt_tracks(vn.frames_and_masks(a.video, bg, idx))
            pts, view, err, used = vn.triangulate(tracks, cams)
            ok = vn.clean_points(pts, view, grid, sol["occ"])
            sets[name] = (pts[ok], [[i for i, _ in tracks[k]] for k, o in zip(used, ok) if o])
            print(name, int(ok.sum()), "points", f"{time.time() - t0:.0f}s", flush=True)
        frame_points = {}
        for X, frames in zip(*sets["fit"]):
            for i in frames:
                frame_points.setdefault(i, []).append(X)
        frame_points = {i: np.array(v) for i, v in frame_points.items()}
        fuse = list(range(0, T, 2))[:: a.every // 2]
        tmp = tempfile.mkdtemp()
        bg_tmp = os.path.join(tmp, "bg.png")
        cv2.imwrite(bg_tmp, bg)
        with Pool(a.workers, initializer=vc._init, initargs=(a.solution, a.video, bg_tmp, frame_points, 18, 30)) as pool:
            results = [r for r in pool.map(vc._solve, fuse, chunksize=1) if r[1]]
        shutil.rmtree(tmp, ignore_errors=True)
        C = dict(results=results, held_pts=sets["held"][0])
        os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
        pickle.dump(C, open(a.cache, "wb"))
    results, held_pts = C["results"], C["held_pts"]
    n = len(results)
    print(n, "frames with cuboids", f"{time.time() - t0:.0f}s", flush=True)

    labels = part_labels(grid, sol["occ"], a.labels)
    front, front_ids = front_mask(grid, labels)
    centre = grid.centres[(front & sol["occ"]).ravel()].mean(0)
    print("front assembly parts", front_ids, "centre", centre.round(2))

    fuse_ids = [r[0] for r in results]
    test = list(range(1, T, 2))[::8]
    masks = {i: m for i, _, m in vn.frames_and_masks(a.video, bg, test) if m.sum() >= 2500}
    fmasks = {i: m for i, _, m in vn.frames_and_masks(a.video, bg, fuse_ids) if m.sum() >= 2500}

    # steering angle of every fusion frame, from its silhouette and the carved model
    occ_c = sol["occ"]
    angles = {i: best_angle(cam, poses[i], occ_c, grid, front, centre, m)[1] for i, m in fmasks.items()}
    print("fusion-frame steering angles (deg): median |a| %.1f, range %.0f..%.0f" % (
        np.degrees(np.median(np.abs(list(angles.values())))), np.degrees(min(angles.values())),
        np.degrees(max(angles.values()))), flush=True)

    def iou_on(occ, ms, articulated=True):
        vals = []
        for i, m in ms.items():
            if articulated:
                vals.append(best_angle(cam, poses[i], occ, grid, front, centre, m)[0])
            else:
                sp = v3.surface_points(occ, grid)
                vals.append(v3.iou(v3.silhouette_of(cam, poses[i], sp, m.shape, grid.size), m) if len(sp) else 0)
        return float(np.mean(vals))

    def score(occ):
        d = vn.surface_distance(o3d, vn.occ_to_mesh(o3d, occ, grid), held_pts)
        return dict(held_out_iou_rigid=iou_on(occ, masks, False), held_out_iou_steered=iou_on(occ, masks, True),
                    held_out_point_distance_median_cm=float(100 * np.median(d)),
                    held_out_points_within_2cm=float(np.mean(d < 0.02)), voxels=int(occ.sum()))

    fit_masks = dict(list(fmasks.items())[::3])
    taus = (0.2, 0.3, 0.4, 0.5)
    metrics = dict(frames_fused=n, held_points=len(held_pts), front_parts=[int(k) for k in front_ids],
                   steering_angles_deg={int(i): float(np.degrees(v)) for i, v in angles.items()})
    method = "global + points"
    per_frame = {i: frame_votes(grid, out[method]) for i, out in results}
    votes_rigid = sum(per_frame.values())
    votes_art = sum(rotate_grid(v.astype(np.int32), grid, front, -angles.get(i, 0.0), centre) for i, v in per_frame.items())

    def fused(votes, tau_body, tau_front):
        return np.where(front, votes >= max(1, round(tau_front * n)), votes >= max(1, round(tau_body * n)))

    variants = {}
    fit = {t: iou_on(votes_rigid >= max(1, round(t * n)), fit_masks, False) for t in taus}
    t1 = max(fit, key=fit.get)
    variants["cuboids (global + points), one tau"] = (fused(votes_rigid, t1, t1), dict(tau=t1))
    fit = {(tb, tf): iou_on(fused(votes_rigid, tb, tf), fit_masks) for tb in taus for tf in (0.1,) + taus}
    tb, tf = max(fit, key=fit.get)
    variants["+ per-part tau"] = (fused(votes_rigid, tb, tf), dict(tau_body=tb, tau_front=tf))
    fit = {(tb, tf): iou_on(fused(votes_art, tb, tf), fit_masks) for tb in taus for tf in (0.1,) + taus}
    tb, tf = max(fit, key=fit.get)
    variants["+ per-part tau + articulated fusion"] = (fused(votes_art, tb, tf), dict(tau_body=tb, tau_front=tf))
    best_parts = variants["+ per-part tau + articulated fusion"][0]
    variants["+ per-part tau + articulated fusion, intersected with carving"] = (best_parts & occ_c, {})
    variants["carving (reference)"] = (occ_c, {})
    occs = {}
    for name, (occ, extra) in variants.items():
        metrics[name] = dict(extra, **score(occ))
        occs[name] = occ
        print(name, json.dumps(metrics[name]), f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(metrics, open(os.path.join(a.out, "bike_cuboids_parts.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "bike_cuboids_parts_models.npz"),
                        **{k.replace(" ", "_"): v for k, v in occs.items()}, lo=grid.lo, hi=grid.hi, size=grid.size,
                        front=front, centre=centre)


if __name__ == "__main__":
    main()
