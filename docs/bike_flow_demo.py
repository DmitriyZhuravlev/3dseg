"""Optical-flow initial poses on data/bike.mp4: demo pictures and numbers.

    python docs/bike_flow_demo.py [--solution out/bike/bike_solution.npz]

Writes docs/video3d/bike_flow_*.png and bike_flow_init.json:
  bike_flow_frames.png  - frames with the box from the track init (before) and the
                          flow init (now); with --solution also the refined pose.
  bike_flow_vectors.png - the optical flow on the bike and the heading it gives.
  bike_flow_tracks.png  - ground tracks and headings on the bird's-eye view.
  bike_flow_construction.png - the article's closed-form footprint on one frame.
Numbers: the bike carved from each set of initial poses alone (no refinement),
held-out silhouette IoU; with --solution, heading difference to the refined poses.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import video3d as v3  # noqa: E402
import video_demos as vd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(ROOT, "docs", "video3d")
STEP = 2
BOX = (2.1, 0.8, 1.5)          # motorcycle with rider, m (only for drawing)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
STYLE = {"track (before)": ((40, 40, 230), "tab:red"),
         "flow (now)": ((230, 120, 0), "tab:blue"),
         "flow + contacts": ((200, 0, 200), "tab:purple"),
         "flow + closed form": ((0, 150, 255), "tab:orange"),
         "refined (reference)": ((0, 200, 0), "tab:green")}


def corners(pose):
    L, W, H = BOX
    c = np.array([[sx * L / 2, sy * W / 2, z] for z in (0, H) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    M = v3.pose_matrix(pose[0], pose[1], pose[2], 0.0)
    return (M[:3, :3] @ c.T).T + M[:3, 3]


def draw_box(img, cam, pose, colour, thick=2):
    uv, _ = cam.project(corners(pose))
    for i, j in EDGES:
        cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int)), colour, thick, cv2.LINE_AA)


def wrap(a):
    return np.angle(np.exp(1j * a))


def carve_score(cam, masks, poses):
    T = len(masks)
    grid = v3.VoxelGrid((-1.6, -0.9, 0.0), (1.6, 0.9, 1.9), 0.04)
    train = range(0, T, 2)
    frac, seen = v3.carve(cam, masks, poses, grid, train, margin_px=1)
    occ = (frac >= 0.9) & (seen >= 20)
    pts = v3.surface_points(occ, grid)
    held = list(range(1, T, 4))
    if not len(pts):
        return 0.0, 0
    ious = [v3.iou(v3.silhouette_of(cam, poses[t], pts, masks.shape[1:], grid.size), masks[t]) for t in held]
    return float(np.mean(ious)), int(occ.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data", "bike.mp4"))
    ap.add_argument("--solution", default=None, help="bike_solution.npz from video_demos.py bike")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    frames, fps, bg, cam, _ = vd.bike_setup(a.video)
    F = frames[::STEP]
    M = v3.object_masks(F, bg)
    T = len(M)
    present = np.array([m.sum() > 400 for m in M])
    init = {"track (before)": v3.initial_poses(cam, M, fps, step=STEP)[0],
            "flow (now)": v3.initial_poses(cam, M, fps, step=STEP, frames=F)[0],
            "flow + contacts": v3.initial_poses(cam, M, fps, step=STEP, frames=F, footprint="contacts")[0],
            "flow + closed form": v3.initial_poses(cam, M, fps, step=STEP, frames=F, footprint="closed")[0]}
    raw_yaw, conf = v3.flow_headings(cam, F, M)
    ref = None
    if a.solution:
        P = np.load(a.solution)["poses"]
        ref = P[::STEP][:T]
        init["refined (reference)"] = ref

    res = {"frames_used": int(T), "step": STEP}
    for k in ("track (before)", "flow (now)", "flow + contacts", "flow + closed form"):
        iou, vox = carve_score(cam, M, init[k])
        res[k] = dict(held_out_iou_init_only=iou, voxels=vox)
        if ref is not None:
            d = np.abs(wrap(init[k][present, 2] - ref[present, 2]))
            res[k]["heading_diff_to_refined_deg_median"] = float(np.degrees(np.median(d)))
            res[k]["heading_diff_to_refined_deg_p90"] = float(np.degrees(np.percentile(d, 90)))
        print(k, res[k])
    json.dump(res, open(os.path.join(OUT, "bike_flow_init.json"), "w"), indent=1)

    # 1. frames with boxes
    idx = np.nonzero(present)[0]
    picks = idx[np.linspace(len(idx) * 0.08, len(idx) * 0.92, 6).astype(int)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for a_, t in zip(ax.ravel(), picks):
        img = F[t].copy()
        for k, est in init.items():
            draw_box(img, cam, est[t], STYLE[k][0], 3 if k.startswith("refined") else 2)
        ys, xs = np.nonzero(M[t])
        pad = 110
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        a_.imshow(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        txt = f"frame {t * STEP}"
        if ref is not None:
            txt += ": heading off by " + ", ".join(
                f"{k.split(' ')[0]} {abs(math.degrees(wrap(init[k][t, 2] - ref[t, 2]))):.0f}°"
                for k in ("track (before)", "flow (now)"))
        a_.set_title(txt, fontsize=10)
        a_.axis("off")
    hs = [plt.Line2D([], [], color=STYLE[k][1], lw=3) for k in init]
    fig.legend(hs, list(init), loc="lower center", ncol=len(init), fontsize=11)
    fig.suptitle("bike.mp4: box placed by each initial pose (2.1 x 0.8 x 1.5 m box, drawn only to show pose)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT, "bike_flow_frames.png"), dpi=75)

    # 2. flow vectors and the heading they give
    picks2 = picks[[1, 3, 5]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a_, t in zip(ax, picks2):
        g0 = cv2.cvtColor(F[t], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(F[t + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ys, xs = np.nonzero(M[t])
        pad = 80
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, g0.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, g0.shape[1])
        a_.imshow(cv2.cvtColor(F[t][y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        a_.contour(M[t][y0:y1, x0:x1], [0.5], colors="y", linewidths=1)
        gy, gx = np.mgrid[y0:y1:12, x0:x1:12]
        sel = M[t][gy, gx]
        u = flow[gy[sel], gx[sel]]
        a_.quiver(gx[sel] - x0, gy[sel] - y0, u[:, 0], u[:, 1], color="cyan", angles="xy", scale_units="xy", scale=0.25)
        # heading arrow on the ground from the footprint centre
        c = init["flow (now)"][t]
        p0 = np.array([[c[0], c[1], 0.0], [c[0] + 1.2 * math.cos(raw_yaw[t]), c[1] + 1.2 * math.sin(raw_yaw[t]), 0.0]])
        uv, _ = cam.project(p0)
        a_.annotate("", uv[1] - [x0, y0], uv[0] - [x0, y0], arrowprops=dict(color="magenta", lw=3, arrowstyle="->"))
        a_.set_title(f"frame {t * STEP}: flow (cyan) -> heading on the ground (magenta), agreement {conf[t]:.2f}", fontsize=9)
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bike_flow_vectors.png"), dpi=75)

    # 3. tracks
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    g = ax[0]
    for k, est in init.items():
        e = est[present]
        g.plot(e[:, 0], e[:, 1], color=STYLE[k][1], lw=2.5 if k.startswith("refined") else 1.5, label=k)
        s = slice(0, len(e), 25)
        g.quiver(e[s, 0], e[s, 1], np.cos(e[s, 2]), np.sin(e[s, 2]), color=STYLE[k][1], scale=25, width=0.003)
    g.set_aspect("equal")
    g.set_xlabel("x, m")
    g.set_ylabel("y, m (forward)")
    g.set_title("ground track and heading (arrows every 25 used frames)")
    g.legend(fontsize=9)
    g = ax[1]
    tt = np.arange(T) * STEP
    for k, est in init.items():
        y = np.degrees(np.unwrap(est[:, 2]))
        y[~present] = np.nan
        g.plot(tt, y, color=STYLE[k][1], lw=2.5 if k.startswith("refined") else 1.5, label=k)
    raw = np.degrees(np.unwrap(np.where(np.isnan(raw_yaw), 0, raw_yaw)))
    raw[np.isnan(raw_yaw) | ~present] = np.nan
    g.plot(tt, raw, ".", color="tab:blue", ms=2, alpha=0.4, label="flow heading per frame (before smoothing)")
    g.set_xlabel("frame")
    g.set_ylabel("heading, deg")
    g.set_title("heading over time")
    g.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bike_flow_tracks.png"), dpi=75)

    # 4. closed-form construction on a real frame
    dims = np.median([c[1] for c in (v3.footprint_centre(cam, M[t], init["flow (now)"][t, 2]) for t in idx) if c], 0)
    for t in picks[[2, 3, 1, 4]]:
        fit = v3.footprint_closed_form(cam, M[t], init["flow (now)"][t, 2], dims)
        if fit is not None:
            break
    if fit is not None:
        plot_construction(cam, F[t], M[t], init["flow (now)"][t, 2], dims, fit, t)
    res["contact_dims_m"] = [float(x) for x in dims]
    json.dump(res, open(os.path.join(OUT, "bike_flow_init.json"), "w"), indent=1)


def plot_construction(cam, frame, mask, yaw, dims, fit, t):
    ys, xs = np.nonzero(mask)
    x0, x1, yb = xs.min(), xs.max() + 1, ys.max() + 1
    img = np.array([[x0, yb], [x1, yb], [x0, yb - 20], [x1, yb - 20]], float)
    A, B, Ru, Tu = cam.backproject_to_plane(img, 0.0)[:, :2]
    dL, dR = Ru - A, Tu - B
    a, q = fit[2]
    h = np.array([math.cos(yaw), math.sin(yaw)])
    axis = h if abs(a - dims[0]) < 1e-9 else np.array([-h[1], h[0]])
    for sgn in (1, -1):
        u = sgn * axis
        hit = v3._line_hit(B, u, A, dL)
        if hit is not None and hit[0] > a and hit[1] >= 0:
            break
    l = hit[0]
    E = B + l * u
    C = ((l - a) * A + a * B) / l
    K = A + (a / l) * (E - A)
    p = np.array([-u[1], u[0]])
    p = p if p @ (dL + dR) > 0 else -p
    D = C + q * p
    rect = np.array([C, K, K + D - C, D, C])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    im = frame.copy()
    cv2.rectangle(im, (x0, ys.min()), (x1, yb), (0, 220, 255), 2)
    rr = cam.project(np.column_stack([rect, np.zeros(len(rect))]))[0]
    cv2.polylines(im, [np.round(rr).astype(np.int32)], True, (0, 150, 255), 2, cv2.LINE_AA)
    for P, nm in ((A, "A"), (B, "B"), (C, "C"), (K, "K"), (D, "D")):
        uv = cam.project(np.array([[*P, 0.0]]))[0][0]
        cv2.circle(im, tuple(np.round(uv).astype(int)), 4, (0, 0, 255), -1)
        cv2.putText(im, nm, tuple(np.round(uv + [5, 18]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    pad = 90
    cy0, cy1 = max(ys.min() - pad, 0), min(yb + pad, im.shape[0])
    cx0, cx1 = max(x0 - pad, 0), min(x1 + pad, im.shape[1])
    ax[0].imshow(cv2.cvtColor(im[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"frame {t * STEP}: 2D box (yellow), closed-form footprint (orange)", fontsize=10)
    ax[0].axis("off")
    g = ax[1]
    for P, d in ((A, dL), (B, dR)):
        d = d / np.linalg.norm(d)
        g.plot([P[0], P[0] + 4 * d[0]], [P[1], P[1] + 4 * d[1]], color="0.4", ls="--")
    g.plot([A[0], B[0]], [A[1], B[1]], "k", lw=2, label="AB = bottom edge of the 2D box")
    g.plot([B[0], E[0]], [B[1], E[1]], color="tab:purple", label="ray from B along the flow heading -> E")
    g.plot(rect[:, 0], rect[:, 1], color="tab:orange", lw=2.5, label="footprint CKD")
    for P, nm in ((A, "A"), (B, "B"), (E, "E"), (C, "C"), (K, "K"), (D, "D")):
        g.plot(*P, "o", color="tab:red", ms=5)
        g.annotate(nm, P, xytext=(5, 5), textcoords="offset points", fontsize=11, color="tab:red")
    allp = np.vstack([A, B, E, C, K, D])
    g.set_xlim(allp[:, 0].min() - 0.8, allp[:, 0].max() + 0.8)
    g.set_ylim(allp[:, 1].min() - 0.8, allp[:, 1].max() + 1.2)
    g.set_aspect("equal")
    g.set_title(f"bird's-eye view: a = {a:.2f} m, l = {l:.2f} m, |CD| = {q:.2f} m vs b = {fit[3]:.2f} m "
                f"(fit error {fit[1] * 100:.0f} cm)", fontsize=10)
    g.set_xlabel("x, m")
    g.set_ylabel("y, m")
    g.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bike_flow_construction.png"), dpi=80)


if __name__ == "__main__":
    main()
