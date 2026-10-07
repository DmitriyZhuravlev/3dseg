"""Demo pictures for the optical-flow initial poses and the closed-form footprint.

    python docs/flow_init_demo.py      # writes docs/video3d/flow_init_*.png

1. flow_init_frames.png    - frames of the 'start' and 's' videos with the box placed
                             by each initialisation (true pose in green).
2. flow_init_construction.png - the article's construction on the bird's-eye view
                             for one frame: A, B, side rays, E, C, K, D.
3. flow_init_tracks.png    - ground tracks and headings, true vs each method.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_flow_init as b  # noqa: E402
import video3d as v3  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video3d")
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
METHODS = [("true pose", (0, 200, 0), "tab:green"),
           ("track (before)", (40, 40, 230), "tab:red"),
           ("flow + contacts (default)", (230, 120, 0), "tab:blue"),
           ("flow + closed form", (0, 150, 255), "tab:orange")]


def corners(pose):
    L, W, H = b.DIMS
    c = np.array([[sx * L / 2, sy * W / 2, z] for z in (0, H) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return (v3.pose_matrix(*pose)[:3, :3] @ c.T).T + v3.pose_matrix(*pose)[:3, 3]


def draw_box(img, cam, pose, colour, thick=2):
    uv, _ = cam.project(corners(pose))
    for i, j in EDGES:
        cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int)), colour, thick, cv2.LINE_AA)


def run(name, cam, gimg):
    truth = b.trajectory(name)
    frames, masks = b.render(cam, truth, gimg)
    est = [truth,
           v3.initial_poses(cam, masks, b.FPS)[0],
           v3.initial_poses(cam, masks, b.FPS, frames=frames)[0],
           v3.initial_poses(cam, masks, b.FPS, frames=frames, footprint="closed")[0]]
    return truth, frames, masks, est


def crop_box(cam, poses, pad=60):
    uv = np.vstack([cam.project(corners(p))[0] for p in poses])
    x0, y0 = np.maximum(uv.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(uv.max(0) + pad, [b.SHAPE[1], b.SHAPE[0]]).astype(int)
    return x0, y0, x1, y1


def figure_frames(cam, runs):
    picks = {"start": (10, 45, 80), "s": (15, 45, 75)}
    fig, ax = plt.subplots(len(runs), 3, figsize=(15, 4.3 * len(runs)))
    for r, (name, (truth, frames, masks, est)) in enumerate(runs.items()):
        for c, t in enumerate(picks[name]):
            img = cv2.cvtColor(frames[t], cv2.COLOR_GRAY2BGR)
            for k in range(len(METHODS) - 1, -1, -1):
                draw_box(img, cam, est[k][t], METHODS[k][1], 3 if k == 0 else 2)
            x0, y0, x1, y1 = crop_box(cam, [e[t] for e in est])
            ax[r, c].imshow(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            errs = ", ".join(f"{METHODS[k][0].split(' ')[0] if k == 1 else METHODS[k][0].split('+ ')[-1].split(' ')[0]} "
                             f"{np.linalg.norm(est[k][t, :2] - truth[t, :2]) * 100:.0f} cm" for k in (1, 2, 3))
            ax[r, c].set_title(f"'{name}' frame {t}: {errs}", fontsize=9)
            ax[r, c].axis("off")
    handles = [plt.Line2D([], [], color=m[2], lw=3) for m in METHODS]
    fig.legend(handles, [m[0] for m in METHODS], loc="lower center", ncol=4, fontsize=10)
    fig.suptitle("Box placed by each initial pose (no refinement); error = ground distance to the true centre", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT, "flow_init_frames.png"), dpi=80)


def figure_construction(cam, truth, frames, masks, t=30):
    """Re-trace footprint_closed_form step by step for one frame, with the true heading."""
    m = masks[t]
    ys, xs = np.nonzero(m)
    x0, x1, yb = xs.min(), xs.max() + 1, ys.max() + 1
    side = 20
    img = np.array([[x0, yb], [x1, yb], [x0, yb - side], [x1, yb - side]], float)
    A, B, Ru, Tu = cam.backproject_to_plane(img, 0.0)[:, :2]
    dL, dR = Ru - A, Tu - B
    yaw = truth[t, 2]
    best = v3.footprint_closed_form(cam, m, yaw, b.DIMS[:2])
    a, q = best[2]
    L, W = b.DIMS[:2]
    h = np.array([math.cos(yaw), math.sin(yaw)])
    axis = h if abs(a - L) < 1e-9 else np.array([-h[1], h[0]])
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

    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = cv2.cvtColor(frames[t], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(im, (x0, ys.min()), (x1, yb), (0, 220, 255), 2)
    draw_box(im, cam, truth[t], (0, 200, 0), 2)
    for P, nm in ((A, "A"), (B, "B"), (C, "C"), (K, "K"), (D, "D")):
        uv = cam.project(np.array([[*P, 0.0]]))[0][0]
        cv2.circle(im, tuple(np.round(uv).astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(im, nm, tuple(np.round(uv + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cx0, cy0, cx1, cy1 = crop_box(cam, [truth[t]], pad=90)
    ax[0].imshow(cv2.cvtColor(im[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
    ax[0].set_title("image: 2D box of the silhouette (yellow), true box (green)", fontsize=10)
    ax[0].axis("off")

    g = ax[1]
    far = 3.0
    for P, d in ((A, dL), (B, dR)):
        d = d / np.linalg.norm(d)
        g.plot([P[0], P[0] + far * d[0]], [P[1], P[1] + far * d[1]], color="0.4", ls="--")
    g.plot([A[0], B[0]], [A[1], B[1]], color="k", lw=2, label="AB = bottom edge of the 2D box")
    g.plot([B[0], E[0]], [B[1], E[1]], color="tab:purple", lw=1.5, label="ray from B along the heading, meets AR in E")
    g.plot(rect[:, 0], rect[:, 1], color="tab:orange", lw=2.5, label="closed-form footprint CKD")
    tc = corners(truth[t])[:4, :2]
    g.plot(*np.vstack([tc, tc[:1]]).T, color="tab:green", lw=1.5, label="true footprint")
    for P, nm in ((A, "A"), (B, "B"), (E, "E"), (C, "C"), (K, "K"), (D, "D")):
        g.plot(*P, "o", color="tab:red", ms=5)
        g.annotate(nm, P, xytext=(5, 5), textcoords="offset points", fontsize=11, color="tab:red")
    g.plot(*best[0], "x", color="tab:orange", ms=10, mew=2)
    g.plot(*truth[t, :2], "+", color="tab:green", ms=12, mew=2)
    g.set_aspect("equal")
    g.set_title(f"bird's-eye view: C = ((l-a)A + aB)/l, a = {a:.2f} m, l = {l:.2f} m;\n"
                f"|CD| = {q:.2f} m vs b = {best[3]:.2f} m (fit error {best[1] * 100:.0f} cm), "
                f"centre off by {np.linalg.norm(best[0] - truth[t, :2]) * 100:.0f} cm", fontsize=10)
    allp = np.vstack([A, B, E, C, K, D, tc])
    g.set_xlim(allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5)
    g.set_ylim(allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.8)
    g.set_xlabel("x, m")
    g.set_ylabel("y (away from camera), m")
    g.legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "flow_init_construction.png"), dpi=85)


def figure_tracks(runs):
    fig, ax = plt.subplots(1, len(runs), figsize=(7 * len(runs), 5.5))
    for a, (name, (truth, _, _, est)) in zip(ax, runs.items()):
        for k, (lab, _, col) in enumerate(METHODS):
            e = est[k]
            a.plot(e[:, 0], e[:, 1], color=col, lw=2.5 if k == 0 else 1.5, label=lab)
            s = slice(0, len(e), 10)
            a.quiver(e[s, 0], e[s, 1], np.cos(e[s, 2]), np.sin(e[s, 2]), color=col, scale=18, width=0.004)
        a.set_aspect("equal")
        a.set_title(f"'{name}': ground track and heading (arrows every 10 frames)", fontsize=10)
        a.set_xlabel("x, m")
        a.set_ylabel("y, m")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flow_init_tracks.png"), dpi=80)


def main():
    os.makedirs(OUT, exist_ok=True)
    cam = b.camera()
    gimg = b.ground_image(cam)
    runs = {n: run(n, cam, gimg) for n in ("start", "s")}
    figure_frames(cam, runs)
    truth, frames, masks, _ = runs["s"]
    figure_construction(cam, truth, frames, masks)
    figure_tracks(runs)


if __name__ == "__main__":
    main()
