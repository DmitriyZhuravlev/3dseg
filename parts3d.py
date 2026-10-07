"""Motion-based 3D part segmentation of an object carved from video (fixed camera).

video3d.py carves one rigid voxel model of a moving object and estimates its pose in
every frame. Real objects are not one rigid body: a motorcycle's wheels spin, its
rider sways, a car's wheels steer. This module splits the carved model into parts
that move rigidly relative to each other, using only the video:

1. Prediction. For every frame pair (t, t+1) the visible surface voxels (z-buffer)
   are projected with the object's poses; under the single-rigid-body hypothesis
   their image motion is uv(t+1) - uv(t). Dense optical flow measures the actual
   motion; the difference is what the rigid model does not explain.
2. Supervoxel motions. The surface is cut into spatially compact supervoxels; for
   each supervoxel and frame pair a small rigid motion in the object frame (rotation
   about the supervoxel's centre + translation, 6 parameters) is fitted to the
   observed flow by robust Gauss-Newton.
3. Multi-model labelling. Each supervoxel's motion sequence, plus "no relative
   motion" (the whole object's motion refitted to the flow, which absorbs the pose
   error of the silhouette-based poses), is a candidate part motion. Every surface voxel is scored by its flow
   residual under every candidate, and the labelling minimises
       sum_v cost(v, l_v) + lam * #{neighbouring voxels with different labels}
                          + beta * N * #parts
   (iterated conditional modes, then greedy removal of parts while the energy falls).
   The label cost makes the number of parts a result, not an input. Two wheels
   spinning at the same rate stay separate: a rotation about one axle does not
   explain the other wheel's flow.
4. Refinement. Each part's motions are refitted from all its voxels and the labelling
   is repeated until it stops changing. Interior voxels take the label of the nearest
   surface voxel, so the whole volume is segmented.

This is the 3D counterpart of the flow clustering of the SpringerLifting chapter: there
the 2D flow of an image is clustered into objects; here flow residuals against a 3D
model are clustered into rigid parts of the object, and every label lives on a voxel.
"""
import math

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import video3d as v3

HUBER_PX = 1.5
RIDGE = 0.05
CAP_PX = 4.0


# ---------------------------------------------------------------------------
# Observations: visible surface voxels and their optical flow
# ---------------------------------------------------------------------------
def _gray(f):
    return f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def visible(cam, pose, pts, shape, voxel):
    """uv, depth and a visibility mask (coarse z-buffer, cell ~ one projected voxel)."""
    uv, depth = v3.project_object(cam, pose, pts)
    H, W = shape
    ok = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    vis = np.zeros(len(pts), bool)
    if ok.sum() == 0:
        return uv, depth, vis
    cell = max(1, int(math.ceil(cam.K[0, 0] * voxel / np.median(depth[ok]))))
    cx = (uv[ok, 0] // cell).astype(int)
    cy = (uv[ok, 1] // cell).astype(int)
    wc = W // cell + 1
    lin = cy * wc + cx
    zb = np.full((H // cell + 1) * wc, np.inf)
    np.minimum.at(zb, lin, depth[ok])
    vis[np.nonzero(ok)[0]] = depth[ok] <= zb[lin] + 1.5 * voxel
    return uv, depth, vis


FLOW = "farneback7"


def dense_flow(a, b, method=None):
    """Dense optical flow a -> b (H, W, 2). Default "farneback7": Farneback with a 7 px
    window; the usual 15 px window averages a small wheel's rotation away (measured
    against the true part motions in bench_parts3d.py)."""
    method = method or FLOW
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if method == "farneback7":
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 7, 5, 5, 1.1, 0)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST if method == "dis-fast" else
                                    cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST if method == "dis-ultra" else
                                    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(a, b, None)


def observe(cam, frames, poses, pts, voxel, pairs, pad=30, method=None):
    """For each frame pair t -> t+1: indices of visible points, their uv(t), observed flow."""
    H, W = frames.shape[1:3]
    obs = {}
    for t in pairs:
        uv, _, vis = visible(cam, poses[t], pts, (H, W), voxel)
        idx = np.nonzero(vis)[0]
        if len(idx) < 30:
            continue
        p = uv[idx]
        x0, y0 = np.maximum(np.floor(p.min(0)) - pad, 0).astype(int)
        x1, y1 = np.minimum(np.ceil(p.max(0)) + pad, [W, H]).astype(int)
        fl = dense_flow(np.ascontiguousarray(_gray(frames[t])[y0:y1, x0:x1]),
                        np.ascontiguousarray(_gray(frames[t + 1])[y0:y1, x0:x1]), method)
        mx = (p[:, 0] - x0).astype(np.float32)
        my = (p[:, 1] - y0).astype(np.float32)
        f = np.stack([cv2.remap(fl[..., k], mx[:, None], my[:, None], cv2.INTER_LINEAR)[:, 0] for k in (0, 1)], 1)
        obs[t] = (idx, p, f.astype(float))
    return obs


# ---------------------------------------------------------------------------
# Rigid motion of a set of points, fitted to their flow
# ---------------------------------------------------------------------------
def move(X, xi, c):
    """Rotate X by rotation vector xi[:3] about c, then translate by xi[3:] (object frame)."""
    R = Rotation.from_rotvec(xi[:3]).as_matrix()
    return (X - c) @ R.T + c + xi[3:]


def predicted_flow(cam, pose_next, X, uv0, xi, c):
    uv1, _ = v3.project_object(cam, pose_next, move(X, xi, c))
    return uv1 - uv0


def fit_motion(cam, pose_next, X, uv0, f, c, xi0=None, iters=5, prior=None):
    """Robust Gauss-Newton for the 6 motion parameters. Returns (xi, residual norms).

    The motion is relative to the object's pose, so zero is the natural prior; a ridge
    term (prior x the mean curvature of the data term) keeps the fit of a small,
    nearly flat patch from trading rotation against translation, which would fit the
    patch but extrapolate wrongly to its neighbours.
    """
    xi = np.zeros(6) if xi0 is None else xi0.copy()
    eps = 1e-4
    for _ in range(iters):
        pred = predicted_flow(cam, pose_next, X, uv0, xi, c)
        r = (f - pred).ravel()
        J = np.empty((len(r), 6))                        # d(predicted flow) / d(xi)
        for k in range(6):
            e = np.zeros(6)
            e[k] = eps
            J[:, k] = (predicted_flow(cam, pose_next, X, uv0, xi + e, c) - pred).ravel() / eps
        n = np.linalg.norm(r.reshape(-1, 2), axis=1)
        w = np.repeat(np.where(n < HUBER_PX, 1.0, HUBER_PX / np.maximum(n, 1e-9)), 2)
        A = J.T @ (J * w[:, None])
        lam = (RIDGE if prior is None else prior) * np.trace(A) / 6 + 1e-9
        d = np.linalg.solve(A + lam * np.eye(6), J.T @ (w * r) - lam * xi)   # r(xi + d) ~ r - J d
        xi = xi + d
        if np.linalg.norm(d) < 1e-6:
            break
    res = np.linalg.norm(f - predicted_flow(cam, pose_next, X, uv0, xi, c), axis=1)
    return xi, res


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def _neighbours(grid, surf_idx):
    """26-neighbour pairs among surface voxels (indices into surf_idx)."""
    shape = grid.shape
    ijk = np.stack(np.unravel_index(surf_idx, shape), 1)
    lookup = -np.ones(int(np.prod(shape)), int)
    lookup[surf_idx] = np.arange(len(surf_idx))
    pairs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if (di, dj, dk) <= (0, 0, 0):
                    continue
                q = ijk + [di, dj, dk]
                ok = np.all((q >= 0) & (q < shape), 1)
                j = lookup[np.ravel_multi_index(q[ok].T, shape)]
                i = np.nonzero(ok)[0]
                keep = j >= 0
                pairs.append(np.stack([i[keep], j[keep]], 1))
    return np.vstack(pairs)


def _robust(r):
    return np.minimum(r, CAP_PX)


class PartSegmentation:
    def __init__(self, labels_surface, surf_idx, labels_volume, motions, centres, history):
        self.labels_surface = labels_surface      # part id per surface voxel
        self.surf_idx = surf_idx                  # flat grid indices of surface voxels
        self.labels_volume = labels_volume        # part id per grid voxel, -1 = empty
        self.motions = motions                    # motions[k][t] = xi (6,) or None
        self.centres = centres
        self.history = history


def supervoxel_motions(cam, poses, X, obs, sv, n_sv, min_pts=12):
    centres = np.array([X[sv == s].mean(0) if np.any(sv == s) else np.zeros(3) for s in range(n_sv)])
    mot = [dict() for _ in range(n_sv)]
    own = [dict() for _ in range(n_sv)]
    for t, (idx, uv0, f) in obs.items():
        lab = sv[idx]
        for s in np.unique(lab):
            m = lab == s
            if m.sum() < min_pts:
                continue
            xi, res = fit_motion(cam, poses[t + 1], X[idx[m]], uv0[m], f[m], centres[s])
            mot[s][t] = xi
            own[s][t] = float(np.median(_robust(res)))
    return centres, mot, own


def part_motions(cam, poses, X, obs, labels, k_parts, min_pts=12):
    centres = np.array([X[labels == k].mean(0) if np.any(labels == k) else np.zeros(3) for k in range(k_parts)])
    mot = [dict() for _ in range(k_parts)]
    for t, (idx, uv0, f) in obs.items():
        lab = labels[idx]
        for k in range(k_parts):
            m = lab == k
            if m.sum() < min_pts:
                continue
            mot[k][t], _ = fit_motion(cam, poses[t + 1], X[idx[m]], uv0[m], f[m], centres[k])
    return centres, mot


def voxel_costs(cam, poses, X, obs, centres, mot):
    """Mean robust flow residual of every surface voxel under every part's motions."""
    K = len(centres)
    S = np.zeros((len(X), K))
    N = np.zeros(len(X))
    for t, (idx, uv0, f) in obs.items():
        N[idx] += 1
        for k in range(K):
            xi = mot[k].get(t, np.zeros(6))             # no estimate for this frame: no relative motion
            r = np.linalg.norm(f - predicted_flow(cam, poses[t + 1], X[idx], uv0, xi, centres[k]), axis=1)
            S[idx, k] += _robust(r)
    return S / np.maximum(N, 1)[:, None], N


def icm(costs, nbr, labels, lam, iters=8):
    """Iterated conditional modes for unary costs + lam * Potts on the voxel graph."""
    n, K = costs.shape
    lab = labels.copy()
    for _ in range(iters):
        votes = np.zeros((n, K))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(votes, (nbr[:, a], lab[nbr[:, b]]), 1.0)
        deg = votes.sum(1, keepdims=True)
        E = costs + lam * (deg - votes)
        new = E.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _compact(lab, min_size):
    ids, counts = np.unique(lab, return_counts=True)
    keep = ids[counts >= min_size]
    remap = -np.ones(lab.max() + 1, int)
    remap[keep] = np.arange(len(keep))
    return remap[lab], len(keep)


def energy(C, nbr, lab, lam, beta):
    return C[np.arange(len(lab)), lab].sum() + lam * np.count_nonzero(lab[nbr[:, 0]] != lab[nbr[:, 1]]) \
        + beta * len(np.unique(lab))


def label_with_costs(C, nbr, lam, beta, lab=None):
    """Data + Potts + label cost: ICM, then greedily drop labels while the energy falls."""
    lab = icm(C, nbr, C.argmin(1) if lab is None else lab, lam)
    E = energy(C, nbr, lab, lam, beta)
    improved = True
    while improved:
        improved = False
        used, counts = np.unique(lab, return_counts=True)
        if len(used) == 1:
            break
        for k in used[np.argsort(counts)]:
            Ck = C.copy()
            Ck[:, k] = np.inf
            keep = np.setdiff1d(used, [k])
            mask = np.full(C.shape[1], np.inf)
            mask[keep] = 0
            Cr = C + mask                              # only the remaining labels
            start = np.where(lab == k, Cr.argmin(1), lab)
            lab2 = icm(Cr, nbr, start, lam)
            E2 = energy(C, nbr, lab2, lam, beta)
            if E2 < E:
                lab, E, improved = lab2, E2, True
                break
    return lab


def segment_parts(cam, frames, poses, occ, grid, pairs, n_super=40, lam=0.1, beta=0.05, em_iters=4,
                  min_part=0.01, seed=0, log=None):
    """Split the carved model `occ` (grid.shape) into rigid parts. Returns PartSegmentation.

    Multi-model fitting with label costs: candidate motions are those of compact
    supervoxels plus "no relative motion"; every surface voxel is scored by the flow
    residual under every candidate; the labelling minimises data + lam x Potts (per
    neighbouring voxel pair, px) + beta x N x (number of parts) (N = surface voxels, so
    beta is the mean residual, px, a part must save per voxel to be worth keeping).
    Then each part's motions are refitted from its voxels and the labelling repeated.
    """
    say = log or (lambda *a: None)
    o = np.pad(occ, 1)
    interior = o[1:-1, 1:-1, 1:-1].copy()
    for ax in range(3):
        for sh in (-1, 1):
            interior &= np.roll(o, sh, axis=ax)[1:-1, 1:-1, 1:-1]
    surf_idx = np.nonzero((occ & ~interior).ravel())[0]
    X = grid.centres[surf_idx]
    obs = observe(cam, frames, poses, X, grid.size, pairs)
    say(f"{len(X)} surface voxels, {len(obs)} frame pairs with flow")
    nbr = _neighbours(grid, surf_idx)
    B = beta * len(X)

    np.random.seed(seed)
    _, sv = kmeans2(X, n_super, minit="++", seed=seed)
    sv_centres, sv_mot, own = supervoxel_motions(cam, poses, X, obs, sv, n_super)
    noise = float(np.median([v for o_ in own for v in o_.values()]))
    # hypothesis 0: the whole object as one rigid body, its motion refitted to the flow
    # (silhouette-based poses are less precise than flow; without this correction the
    # pose error shows up everywhere as apparent part motion)
    c0, m0 = part_motions(cam, poses, X, obs, np.zeros(len(X), int), 1)
    centres = np.vstack([c0, sv_centres])
    mot = m0 + sv_mot
    C, seen = voxel_costs(cam, poses, X, obs, centres, mot)
    labels = label_with_costs(C, nbr, lam, B)
    labels, K = _compact(labels, 1)
    say(f"flow noise {noise:.2f} px; {n_super + 1} candidate motions -> {K} parts")
    history = [dict(stage="hypotheses", parts=K)]
    for it in range(em_iters):
        centres, mot = part_motions(cam, poses, X, obs, labels, K)
        C, seen = voxel_costs(cam, poses, X, obs, centres, mot)
        new = label_with_costs(C, nbr, lam, B, labels)
        new, K = _compact(new, max(1, int(min_part * len(X))))
        if (new < 0).any():                               # parts too small to keep: best remaining part
            new[new < 0] = C[np.ix_(np.nonzero(new < 0)[0], np.arange(K))].argmin(1) if K else 0
        changed = int(np.count_nonzero(new != labels)) if len(new) == len(labels) else -1
        labels = new
        history.append(dict(stage=f"em{it + 1}", parts=K, changed=changed,
                            mean_cost=float(C[np.arange(len(X)), np.minimum(labels, C.shape[1] - 1)].mean())))
        say(f"EM {it + 1}: {K} parts, {changed} voxels changed")
        if changed == 0:
            break
    centres, mot = part_motions(cam, poses, X, obs, labels, K)

    occ_idx = np.nonzero(occ.ravel())[0]
    _, nn = cKDTree(X).query(grid.centres[occ_idx])
    vol = -np.ones(occ.size, int)
    vol[occ_idx] = labels[nn]
    return PartSegmentation(labels, surf_idx, vol.reshape(occ.shape), mot, centres, history)


def flow_epe(cam, poses, X, obs, labels, centres, mot):
    """Mean end-point error of the flow predicted by a labelling and its part motions."""
    err, n = 0.0, 0
    for t, (idx, uv0, f) in obs.items():
        for k in np.unique(labels[idx]):
            m = labels[idx] == k
            xi = mot[k].get(t, np.zeros(6))
            err += np.linalg.norm(f[m] - predicted_flow(cam, poses[t + 1], X[idx[m]], uv0[m], xi, centres[k]), axis=1).sum()
            n += m.sum()
    return err / max(n, 1)
