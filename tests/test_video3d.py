"""video3d: carving a rigid object that moves on the ground, and silhouette inflation."""
import math

import cv2
import numpy as np
import pytest

import video3d as v3


def _camera():
    # 960x540 view of the ground, looking towards a vanishing point above the image centre
    cam = v3.ground_camera(800.0, (960, 540), (480.0, 180.0))
    return v3.scale_camera_to(cam, (380.0, 400.0), (580.0, 400.0), 2.0)


def _box_inside(pts, dims):
    L, W, H = dims
    return (np.abs(pts[:, 0]) <= L / 2) & (np.abs(pts[:, 1]) <= W / 2) & (pts[:, 2] >= 0) & (pts[:, 2] <= H)


def _render_box_masks(cam, poses, dims, shape=(540, 960), n=60000, seed=0):
    rng = np.random.default_rng(seed)
    L, W, H = dims
    pts = rng.uniform([-L / 2, -W / 2, 0], [L / 2, W / 2, H], size=(n, 3))
    masks = np.zeros((len(poses),) + shape, bool)
    for t, p in enumerate(poses):
        uv, depth = v3.project_object(cam, p, pts)
        px = np.round(uv[depth > 0]).astype(int)
        k = (px[:, 0] >= 0) & (px[:, 0] < shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < shape[0])
        m = np.zeros(shape, np.uint8)
        m[px[k, 1], px[k, 0]] = 1
        masks[t] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return masks


def test_ground_camera_maps_forward_vp_and_ground():
    cam = _camera()
    # a far point straight ahead on the ground projects onto the vanishing point
    uv, _ = cam.project(np.array([[0.0, 1e6, 0.0]]))
    assert uv[0] == pytest.approx([480.0, 180.0], abs=0.5)
    # ground back-projection is consistent with projection
    g = cam.backproject_to_plane(np.array([[300.0, 450.0]]), 0.0)
    uv2, _ = cam.project(g)
    assert uv2[0] == pytest.approx([300.0, 450.0], abs=1e-6)
    a, b = cam.backproject_to_plane(np.array([[380.0, 400.0], [580.0, 400.0]]), 0.0)
    assert np.linalg.norm(a - b) == pytest.approx(2.0, rel=1e-6)


def test_carving_a_box_driving_in_a_circle_recovers_its_volume():
    cam = _camera()
    dims = (1.6, 0.6, 0.9)
    T = 48
    ang = np.linspace(0, 2 * math.pi, T, endpoint=False)
    poses = np.column_stack([2.5 * np.cos(ang), 9.0 + 2.5 * np.sin(ang), ang + math.pi / 2, np.zeros(T)])
    masks = _render_box_masks(cam, poses, dims)
    grid = v3.VoxelGrid((-1.2, -0.6, 0.0), (1.2, 0.6, 1.2), 0.04)
    frac, seen = v3.carve(cam, masks, poses, grid, range(0, T, 2), margin_px=1)
    occ = ((frac >= 0.95) & (seen >= 10)).ravel()
    truth = _box_inside(grid.centres, dims)
    vol_iou = (occ & truth).sum() / (occ | truth).sum()
    assert vol_iou > 0.8, vol_iou
    # held-out views (odd frames) are explained by the carved model
    pts = v3.surface_points(occ.reshape(grid.shape), grid)
    ious = [v3.iou(v3.silhouette_of(cam, poses[t], pts, masks.shape[1:], grid.size), masks[t]) for t in range(1, T, 6)]
    assert np.mean(ious) > 0.8


def test_one_view_cannot_do_this():
    """With a single frame the visual hull is a cone: much worse than many frames."""
    cam = _camera()
    dims = (1.6, 0.6, 0.9)
    poses = np.array([[0.0, 9.0, 0.0, 0.0]])
    masks = _render_box_masks(cam, poses, dims)
    grid = v3.VoxelGrid((-1.2, -0.6, 0.0), (1.2, 0.6, 1.2), 0.04)
    frac, seen = v3.carve(cam, masks, poses, grid, [0], margin_px=1)
    occ = ((frac >= 0.999) & (seen >= 1)).ravel()
    truth = _box_inside(grid.centres, dims)
    assert (occ & truth).sum() / (occ | truth).sum() < 0.8


def test_initial_poses_follow_heading_and_lean_into_turns():
    cam = _camera()
    T, fps = 90, 30.0
    ang = np.linspace(0, math.pi, T)
    poses = np.column_stack([2.5 * np.cos(ang), 9.0 + 2.5 * np.sin(ang), ang + math.pi / 2, np.zeros(T)])
    masks = _render_box_masks(cam, poses, (1.6, 0.6, 0.9))
    est, speed = v3.initial_poses(cam, masks, fps)
    mid = slice(15, 75)
    dyaw = np.angle(np.exp(1j * (est[mid, 2] - poses[mid, 2])))
    assert np.median(np.abs(dyaw)) < 0.15
    assert np.all(est[mid, 3] < 0)          # counter-clockwise turn: leans towards the centre (-x of heading)


def test_inflating_a_disc_gives_a_sphere():
    r = 60
    m = np.zeros((2 * r + 11, 2 * r + 11), bool)
    cv2.circle(m.view(np.uint8), (r + 5, r + 5), r, 1, -1)
    half = v3.inflate(m)
    assert half.max() == pytest.approx(r, rel=0.08)               # depth at the centre = radius
    D = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    ring = (D > r * 0.45) & (D < r * 0.55)                         # halfway to the rim
    expected = math.sqrt(r * r - (r / 2) ** 2)
    assert np.median(half[ring]) == pytest.approx(expected, rel=0.08)


def test_weak_view_fit_recovers_rotation():
    m = np.zeros((200, 300), bool)
    cv2.ellipse(m.view(np.uint8), (150, 100), (110, 40), 0, 0, 360, 1, -1)
    model = v3.InflatedModel(m, np.zeros(m.shape + (3,), np.uint8), 0.01, thickness=1.0)
    P = model.surface_points(step=1)
    truth = (0.7, 0.0, 250.0, 300.0, 300.0)
    target = v3.weak_silhouette(P, truth, (420, 600))
    params, score = v3.fit_weak_view(P, target, init_yaw=0.4, pitch=0.0)
    assert score > 0.9
    assert abs(abs(params[0]) - 0.7) < 0.15


def test_flow_heading_and_footprint_beat_the_track_on_an_arc():
    import bench_flow_init as b
    cam = b.camera()
    truth = b.trajectory("arc")
    frames, masks = b.render(cam, truth, b.ground_image(cam))
    track, _ = v3.initial_poses(cam, masks, b.FPS)
    flow, _ = v3.initial_poses(cam, masks, b.FPS, frames=frames, footprint="contacts")
    err = lambda p: np.median(np.abs(np.angle(np.exp(1j * (p[:, 2] - truth[:, 2])))))
    assert err(flow) < math.radians(4) and err(flow) < err(track)
    # the footprint centre is the box centre, not its nearest point
    assert np.median(np.linalg.norm(flow[:, :2] - truth[:, :2], axis=1)) < 0.3


def test_closed_form_footprint_finds_the_box_centre_given_heading_and_size():
    import bench_flow_init as b
    cam = b.camera()
    truth = b.trajectory("arc")
    masks = b.render(cam, truth, b.ground_image(cam))[1]
    fits = [v3.footprint_closed_form(cam, masks[t], truth[t, 2], b.DIMS[:2]) for t in range(0, len(truth), 5)]
    errs = [np.linalg.norm(f[0] - truth[5 * i, :2]) for i, f in enumerate(fits) if f is not None]
    assert len(errs) >= 0.8 * len(fits)
    assert np.median(errs) < 0.15
