"""Triangulation, 3D normals and surface-from-normals on bike.mp4.

Reuses the bike.mp4 solution of video3d.py (docs/video3d/bike_solution.npz): the calibrated
camera and every frame's object pose. In the motorcycle's own frame, a fixed camera watching a
moving object is a moving camera around a static object, so feature tracks can be triangulated
directly in object coordinates (metres), with no further unknowns.

  1. KLT tracks inside the object mask, run separately on the even frames (reconstruction)
     and on the odd frames (held out, for evaluation only).
  2. Each track is triangulated as the least-squares intersection of its rays in the object
     frame; tracks with a small baseline or a large reprojection error are rejected.
  3. Normals: local plane fit (PCA) on the triangulated points, oriented towards the cameras
     that saw each point. The carving's outer surface provides oriented points as well
     (normal = gradient of the smoothed occupancy).
  4. Surface from oriented points: screened Poisson reconstruction (Kazhdan & Hoppe 2013).
  5. Free-space carving: a voxel between a camera and a point it saw cannot be occupied. This
     uses the triangulated points to remove volume that silhouettes cannot (concavities).

Evaluation, all on the held-out odd frames:
  * distance from independently triangulated held-out points to each model's surface;
  * silhouette IoU, as for the carving.

    python video_normals.py --video bike.mp4 --background background.png --out docs/video3d
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import video3d as v3
from lift3d import Camera
from video_segments import SOL, object_camera

SIZE = (960, 540)


def frames_and_masks(video, bg, idx):
    cap = cv2.VideoCapture(video)
    want = set(idx)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
            m = v3.object_masks(f[None], bg, threshold=28, fill_holes=True, close=5)[0]
            m = v3.strip_ground_shadow(m, low_frac=0.10, min_run_frac=0.05)
            yield i, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), m
        i += 1


def klt_tracks(seq, redetect=4, max_pts=600, fb_tol=1.0, min_len=4):
    """Tracks as lists of (frame index, (x, y)), with a forward-backward consistency check."""
    tracks, live, prev = [], [], None
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for k, (i, g, m) in enumerate(seq):
        if prev is not None and live:
            p0 = np.float32([t[-1][1] for t in live]).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev[1], g, p0, None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev[1], p1, None, **lk)
            fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
            keep = []
            for t, p, s1, s2, e in zip(live, p1.reshape(-1, 2), st.ravel(), st2.ravel(), fb):
                x, y = int(round(p[0])), int(round(p[1]))
                ok = s1 and s2 and e < fb_tol and 0 <= x < g.shape[1] and 0 <= y < g.shape[0] and m[y, x]
                if ok:
                    t.append((i, (float(p[0]), float(p[1]))))
                    keep.append(t)
                elif len(t) >= min_len:
                    tracks.append(t)
            live = keep
        if k % redetect == 0:
            occupied = np.zeros_like(m, np.uint8)
            for t in live:
                cv2.circle(occupied, tuple(int(v) for v in t[-1][1]), 6, 1, -1)
            det_mask = (m & (occupied == 0)).astype(np.uint8)
            det_mask = cv2.erode(det_mask, np.ones((5, 5), np.uint8))
            n_new = max_pts - len(live)
            if n_new > 0:
                pts = cv2.goodFeaturesToTrack(g, n_new, 0.01, 6, mask=det_mask)
                if pts is not None:
                    live += [[(i, (float(x), float(y)))] for x, y in pts.reshape(-1, 2)]
        prev = (i, g)
    tracks += [t for t in live if len(t) >= min_len]
    return tracks


def triangulate(tracks, cams, max_reproj=2.5, min_angle_deg=3.0):
    """Least-squares ray intersection per track in the object frame. Returns points, the mean
    direction towards the observing cameras, the reprojection RMS and the used track indices."""
    pts, view, err, used = [], [], [], []
    for k, t in enumerate(tracks):
        Cs = np.array([cams[i].C for i, _ in t])
        D = np.vstack([cams[i].ray(np.array([uv]))[0] for i, uv in t])
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        cosmin = np.min(D @ D[np.argmax(np.abs(D @ D.mean(0)))])
        if np.degrees(np.arccos(np.clip(cosmin, -1, 1))) < min_angle_deg:
            continue
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for c, d in zip(Cs, D):
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ c
        X = np.linalg.solve(A, b)
        r = [np.linalg.norm(cams[i].project(X[None])[0][0] - np.array(uv)) for i, uv in t]
        if not all(cams[i].project(X[None])[1][0] > 0 for i, _ in t):
            continue
        rms = float(np.sqrt(np.mean(np.square(r))))
        if rms <= max_reproj:
            pts.append(X)
            v = (Cs - X)
            view.append((v / np.linalg.norm(v, axis=1, keepdims=True)).mean(0))
            err.append(rms)
            used.append(k)
    return np.array(pts).reshape(-1, 3), np.array(view).reshape(-1, 3), np.array(err), used


def clean_points(pts, view, grid, occ, margin=3):
    """Keep points near the carved hull (removes background tracks that slipped into the mask)."""
    from scipy.ndimage import binary_dilation
    near = binary_dilation(occ, iterations=margin)
    ijk = np.floor((pts - grid.lo) / grid.size).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(grid.shape)), axis=1)
    ok[ok] = near[tuple(ijk[ok].T)]
    return ok


def pca_normals(o3d, pts, view, k=16):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(k))
    n = np.asarray(pc.normals)
    n *= np.sign(np.sum(n * view, axis=1, keepdims=True) + 1e-12)
    return n


def occupancy_normals(occ, grid, sigma=1.5):
    """Oriented points on the carved surface: normal = -gradient of the smoothed occupancy."""
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(occ.astype(float), sigma)
    g = np.stack(np.gradient(f), -1)
    surf = v3.surface_points(occ, grid)
    ijk = np.clip(np.round((surf - grid.lo) / grid.size - 0.5).astype(int), 0, np.array(grid.shape) - 1)
    n = -g[tuple(ijk.T)]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return surf, n


def poisson(o3d, pts, normals, depth=8, trim=0.04):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.normals = o3d.utility.Vector3dVector(normals)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    mesh.compute_vertex_normals()
    return mesh


def mesh_occupancy(o3d, mesh, grid):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    occ = scene.compute_occupancy(o3d.core.Tensor(grid.centres.astype(np.float32))).numpy()
    return occ.reshape(grid.shape).astype(bool)


def surface_distance(o3d, mesh, pts):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()


def occ_to_mesh(o3d, occ, grid):
    verts, faces, normals = v3.occupancy_mesh(occ, grid)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    m.compute_vertex_normals()
    return m


def free_space_carve(occ, grid, pts, obs, cams, margin=2.0):
    """Remove voxels on the segment camera -> point (stopping `margin` voxels before the point)."""
    occ = occ.copy()
    step = grid.size / 2
    removed = 0
    for X, frames in zip(pts, obs):
        for i in frames:
            c = cams[i].C
            d = X - c
            L = np.linalg.norm(d)
            ts = np.arange(0, L - margin * grid.size, step)
            if not len(ts):
                continue
            P = c + d[None] / L * ts[:, None]
            ijk = np.floor((P - grid.lo) / grid.size).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(grid.shape)), axis=1)
            if ok.any():
                sel = tuple(ijk[ok].T)
                removed += int(occ[sel].sum())
                occ[sel] = False
    return occ, removed


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--solution", default=SOL)
    ap.add_argument("--out", default="docs/video3d")
    a = ap.parse_args()
    t0 = time.time()
    sol = np.load(a.solution)
    cam, poses = Camera(sol["K"], sol["R"], sol["C"]), sol["poses"]
    grid = v3.VoxelGrid(sol["lo"], sol["hi"], float(sol["size"]))
    carved = sol["occ"]
    T = len(poses)
    cams = {i: object_camera(cam, poses[i]) for i in range(T)}
    bg = cv2.resize(cv2.imread(a.background), SIZE, interpolation=cv2.INTER_AREA)
    res = {}
    sets = {}
    for name, idx in (("fit", range(0, T, 2)), ("held", range(1, T, 2))):
        tracks = klt_tracks(frames_and_masks(a.video, bg, idx))
        pts, view, err, used = triangulate(tracks, cams)
        ok = clean_points(pts, view, grid, carved)
        sets[name] = (pts[ok], view[ok], [[i for i, _ in tracks[k]] for k, o in zip(used, ok) if o])
        res[f"{name}_tracks"] = len(tracks)
        res[f"{name}_triangulated"] = len(pts)
        res[f"{name}_kept_near_hull"] = int(ok.sum())
        res[f"{name}_reproj_rms_median_px"] = float(np.median(err)) if len(err) else None
        print(name, len(tracks), "tracks", len(pts), "triangulated", int(ok.sum()), "kept",
              f"reproj median {res[f'{name}_reproj_rms_median_px']:.2f} px", f"{time.time()-t0:.0f}s", flush=True)
    pts, view, obs = sets["fit"]
    held = sets["held"][0]
    normals = pca_normals(o3d, pts, view)
    carved_mesh = occ_to_mesh(o3d, carved, grid)
    models = {"carving (silhouettes only)": (carved, carved_mesh)}
    m_pts = poisson(o3d, pts, normals)
    models["Poisson from triangulated points + PCA normals"] = (mesh_occupancy(o3d, m_pts, grid), m_pts)
    s_pts, s_n = occupancy_normals(carved, grid)
    m_all = poisson(o3d, np.vstack([pts, s_pts]), np.vstack([normals, s_n]))
    models["Poisson from triangulated + carved-surface oriented points"] = (mesh_occupancy(o3d, m_all, grid), m_all)
    fs, removed = free_space_carve(carved, grid, pts, obs, cams)
    models["carving + free-space from triangulated points"] = (fs, occ_to_mesh(o3d, fs, grid))
    res["free_space_voxels_removed"] = removed

    # evaluation on held-out frames
    test = list(range(1, T, 2))[::8]
    masks = {i: m for i, _, m in frames_and_masks(a.video, bg, test)}
    for name, (occ, mesh) in models.items():
        d = surface_distance(o3d, mesh, held)
        sp = v3.surface_points(occ, grid)
        ious = [v3.iou(v3.silhouette_of(cam, poses[i], sp, m.shape, grid.size), m) for i, m in masks.items() if m.sum() >= 2500]
        res[name] = dict(held_out_point_distance_median_cm=float(100 * np.median(d)),
                         held_out_points_within_2cm=float(np.mean(d < 0.02)),
                         held_out_points_within_5cm=float(np.mean(d < 0.05)),
                         held_out_silhouette_iou=float(np.mean(ious)), voxels=int(occ.sum()))
        print(f"{name:62s} point dist median {100*np.median(d):.1f} cm  <2cm {100*np.mean(d<0.02):.0f}%  "
              f"<5cm {100*np.mean(d<0.05):.0f}%  held-out IoU {np.mean(ious):.3f}", flush=True)
    res["seconds"] = round(time.time() - t0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "bike_normals_metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "bike_normals_solution.npz"), points=pts, normals=normals, held=held,
                        free_space=fs, lo=grid.lo, hi=grid.hi, size=grid.size)
    for key, (occ, mesh) in models.items():
        slug = {"carving (silhouettes only)": "carving", "Poisson from triangulated points + PCA normals": "poisson_points",
                "Poisson from triangulated + carved-surface oriented points": "poisson_fused",
                "carving + free-space from triangulated points": "freespace"}[key]
        o3d.io.write_triangle_mesh(os.path.join(a.out, f"bike_normals_{slug}_model.ply"), mesh)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
