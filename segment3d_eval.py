"""Evaluate the recursive segment methods against 3D ground truth (NOTES.md section 9).

    python segment3d_eval.py --out out/segeval

box: box.JPG against the hand-traced 3D box (lift3d.box_ground_truth).
pig: a synthetic pig of known shape rendered through the same camera onto reference.JPG.
Metric: per-pixel camera-depth error relative to the true depth.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import lift3d
import segment3d as s3

R = os.path.dirname(os.path.abspath(__file__)) + "/"


def run_box(S):
    cam = lift3d.default_camera(); vps = lift3d.vanishing_points(); s3.SIGNS = s3.axis_signs(cam, vps)
    img = cv2.imread(R + "box.JPG"); mask = lift3d.object_mask(R + "reference.JPG", R + "box.JPG")
    gt = lift3d.box_ground_truth(cam)
    labels, contour = s3.superpixels(img, mask)
    uv_all = np.c_[np.nonzero(labels[::4, ::4] > 0)[1] * 4, np.nonzero(labels[::4, ::4] > 0)[0] * 4].astype(float)
    z_gt = s3.depth_from_boxes(cam, uv_all, [gt["corners"]])
    res = {}
    t0 = time.time()
    # 1. original recursive boxes (calibrated)
    r = s3.reconstruct(cam, img, mask, vps, log=lambda *a: None)
    boxes = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in r["cuboids"]]
    res["recursive boxes (original method, calibrated)"] = s3.depth_errors(s3.depth_from_boxes(cam, uv_all, boxes), z_gt)
    print("1 done", time.time() - t0, flush=True)
    # 2. recursive planar patches (drift fix)
    p = s3.patch_reconstruct(cam, img, mask, vps, labels, contour)
    uv2, z2 = s3.depth_from_planes(cam, labels, p["planes"])
    res["recursive planar patches (fixed)"] = s3.depth_errors(z2, z_gt)
    p0 = s3.patch_reconstruct(cam, img, mask, vps, labels, contour, use_orientation=False)
    pg = s3.patch_reconstruct(cam, img, mask, vps, labels, contour, icm_iters=0)
    pu = s3.patch_reconstruct(cam, img, mask, vps, labels, contour, refine_main=False)
    res["recursive patches, silhouette-only main box"] = s3.depth_errors(s3.depth_from_planes(cam, labels, pu["planes"])[1], z_gt)
    res["recursive patches, no connectivity refinement"] = s3.depth_errors(s3.depth_from_planes(cam, labels, pg["planes"])[1], z_gt)
    res["recursive planar patches, no orientation cue"] = s3.depth_errors(s3.depth_from_planes(cam, labels, p0["planes"])[1], z_gt)
    print("2 done", len(p["planes"]), "/", len(p["ids"]), "failed", len(p["failed"]), time.time() - t0, flush=True)
    # 3. Make3D-style global MRF with the same seed anchors
    pix = s3.segment_pixels(labels, p["ids"])
    anchors = {}
    for s in p["seeds"]:
        if s in p["planes"]:
            axis, value = p["planes"][s]
            P, _ = s3.backproject_plane(cam, pix[s], axis, value)
            anchors[s] = (pix[s], s3.camera_depth(cam, P))
    alphas = s3.make3d_planes(cam, img, labels, anchors)
    uv3, z3 = s3.depth_from_make3d(cam, labels, alphas)
    res["Make3D-style MRF (same anchors, no learned term)"] = s3.depth_errors(z3, z_gt)
    print("3 done", time.time() - t0, flush=True)
    # 4. reference: lift3d single solid box
    L = lift3d.lift(R + "box.JPG", shape="box")
    bc = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in L["cuboids"]]
    res["lift3d solid box (reference)"] = s3.depth_errors(s3.depth_from_boxes(cam, uv_all, bc), z_gt)
    for k, v in res.items():
        print(f"{k:52s} median rel {100*v['median_rel']:.2f}%  mean {100*v['mean_rel']:.2f}%  <1% {100*v['within_1pct']:.0f}%  <3% {100*v['within_3pct']:.0f}%  coverage {100*v['coverage']:.0f}%")
    json.dump(res, open(S + "compare_box.json", "w"), indent=1)
    np.savez(S + "box_depths.npz", uv=uv_all, gt=z_gt, rec=s3.depth_from_boxes(cam, uv_all, boxes), patch=z2, m3d=z3, planes=np.array([[k, *v] for k, v in p["planes"].items()]))


def render_synthetic_pig(S):
    import moderngl
    import synthetic3d
    import video3d as v3
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    from viewer3d import math3d as m3
    from viewer3d.geometry import Geometry
    from viewer3d.renderer import Renderer
    from viewer3d.scene import Material, MeshNode, Scene
    cam = lift3d.default_camera()
    inside, (lo, hi) = synthetic3d.make_pig()
    n = 160
    g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], n) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    vol = gaussian_filter(inside(g).reshape(n, n, n).astype(float), 1.0)
    verts, faces, normals, _ = marching_cubes(vol, 0.5, spacing=tuple((hi - lo) / (n - 1)))
    verts += lo
    geom = Geometry(verts, -normals, np.zeros((len(verts), 2)), faces[:, ::-1].reshape(-1), colors=np.tile([0.95, 0.55, 0.40], (len(verts), 1)))
    ref = cv2.imread(R + "reference.JPG"); H, W = ref.shape[:2]
    ctx = moderngl.create_standalone_context(backend="egl"); rend = Renderer(ctx)
    sc = Scene(); sc.background = (1.0, 0.0, 1.0)
    node = MeshNode(geom, Material(color=(1, 1, 1), shininess=20, specular=0.15)); sc.add(node)
    sc.sun.direction = m3.normalize((-0.3, 0.4, -1.0)).astype(np.float32); sc.ambient.intensity = 0.7
    sc.update(0, 0); node.world_matrix = np.eye(4, dtype=np.float32)
    fbo = ctx.simple_framebuffer((W, H), components=4); rend.render(sc, v3.CVCamera(cam, (W, H)), fbo)
    img = np.asarray(Image.frombytes("RGB", (W, H), fbo.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    key = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)
    out = ref.copy(); out[~key] = img[~key][:, ::-1]
    cv2.imwrite(S + "synth_pig.png", out); np.save(S + "synth_pig_mask.npy", ~key)
    # ground-truth depth: march along each pixel's ray to the first point inside the shape
    ys, xs = np.nonzero((~key)[::4, ::4]); uv = np.c_[xs * 4, ys * 4].astype(float)
    d = cam.ray(uv); far = np.linalg.norm(((lo + hi) / 2) - cam.C) + 1.0
    ts = np.linspace(0.3, far, 1500)
    zgt = np.full(len(uv), np.nan)
    for k in range(0, len(uv), 4000):
        P = cam.C + d[k:k + 4000, None, :] * ts[None, :, None]
        ins = inside(P.reshape(-1, 3)).reshape(P.shape[:2])
        first = np.argmax(ins, axis=1); ok = ins.any(1)
        hit = cam.C + d[k:k + 4000] * ts[first][:, None]
        zgt[k:k + 4000] = np.where(ok, ((hit - cam.C) @ cam.R.T)[:, 2], np.nan)
    np.savez(S + "synth_pig_gt.npz", uv=uv, z=zgt)
    print("rendered", (~key).sum(), "object px; gt depth valid", np.isfinite(zgt).mean())


def run_pig(S):
    cam = lift3d.default_camera(); vps = lift3d.vanishing_points(); s3.SIGNS = s3.axis_signs(cam, vps)
    img = cv2.imread(S + "synth_pig.png"); mask = np.load(S + "synth_pig_mask.npy")
    gt = np.load(S + "synth_pig_gt.npz"); uv_gt, z_gt_all = gt["uv"], gt["z"]
    labels, contour = s3.superpixels(img, mask)
    def on_grid(uv, z):
        """Align a (uv, z) depth sample set to the ground-truth grid."""
        m = {(int(a), int(b)): v for (a, b), v in zip(uv, z)}
        return np.array([m.get((int(a), int(b)), np.nan) for a, b in uv_gt])
    res = {}
    r = s3.reconstruct(cam, img, mask, vps, log=lambda *a: None)
    boxes = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in r["cuboids"]]
    res["recursive boxes (original method, calibrated)"] = s3.depth_errors(s3.depth_from_boxes(cam, uv_gt, boxes), z_gt_all)
    for name, kw in [("recursive planar patches (fixed)", {}), ("recursive patches, no orientation cue", {"use_orientation": False}),
                     ("recursive patches, no connectivity refinement", {"icm_iters": 0}), ("recursive patches, silhouette-only main box", {"refine_main": False})]:
        p = s3.patch_reconstruct(cam, img, mask, vps, labels, contour, **kw)
        uv, z = s3.depth_from_planes(cam, labels, p["planes"])
        res[name] = s3.depth_errors(on_grid(uv, z), z_gt_all)
        if name.startswith("recursive planar"):
            pbest = p
    pix = s3.segment_pixels(labels, pbest["ids"]); anchors = {}
    for s in pbest["seeds"]:
        if s in pbest["planes"]:
            P, _ = s3.backproject_plane(cam, pix[s], *pbest["planes"][s]); anchors[s] = (pix[s], s3.camera_depth(cam, P))
    alphas = s3.make3d_planes(cam, img, labels, anchors)
    uv3, z3 = s3.depth_from_make3d(cam, labels, alphas)
    res["Make3D-style MRF (same anchors, no learned term)"] = s3.depth_errors(on_grid(uv3, z3), z_gt_all)
    # reference: lift3d round model (cuboid columns with bottoms above ground)
    L = lift3d.lift_mask(cam, mask, img, shape="round")
    cb = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in L["cuboids"]]
    res["lift3d round model (reference)"] = s3.depth_errors(s3.depth_from_boxes(cam, uv_gt, cb), z_gt_all)
    for k, v in res.items():
        print(f"{k:52s} median rel {100*v['median_rel']:.2f}%  mean {100*v['mean_rel']:.2f}%  <1% {100*v['within_1pct']:.0f}%  <3% {100*v['within_3pct']:.0f}%  coverage {100*v['coverage']:.0f}%")
    json.dump(res, open(S + "compare_pig.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/segeval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = a.out.rstrip("/") + "/"
    run_box(S)
    render_synthetic_pig(S)
    run_pig(S)
