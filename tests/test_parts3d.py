"""parts3d: rigid part motions from flow, and labelling with label costs."""
import numpy as np

import parts3d as p3
import video3d as v3


def _camera():
    cam = v3.ground_camera(800.0, (960, 540), (480.0, 180.0))
    return v3.scale_camera_to(cam, (380.0, 400.0), (580.0, 400.0), 2.0)


def test_fit_motion_recovers_a_wheel_rotation_from_its_flow():
    cam = _camera()
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 400)
    r = 0.33 * np.sqrt(rng.random(400))
    c = np.array([0.7, -0.07, 0.33])
    X = c + np.column_stack([r * np.cos(a), np.zeros(400), r * np.sin(a)])     # wheel face
    pose0, pose1 = np.array([0.0, 9.0, 0.3, 0.0]), np.array([0.05, 9.02, 0.31, 0.0])
    true = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    uv0, _ = v3.project_object(cam, pose0, X)
    f = p3.predicted_flow(cam, pose1, X, uv0, true, c)
    xi, res = p3.fit_motion(cam, pose1, X, uv0, f, c, prior=0.0, iters=10)
    assert abs(xi[1] - 0.2) < 0.01
    assert np.median(res) < 0.05


def test_label_costs_remove_a_part_that_explains_nothing_new():
    # two labels with identical costs on a chain of voxels: one part is enough
    n = 50
    C = np.column_stack([np.full(n, 1.0), np.full(n, 1.0)])
    C[::2, 1] -= 0.01                                    # a tiny, noisy preference
    nbr = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    lab = p3.label_with_costs(C, nbr, lam=0.0, beta=5.0)
    assert len(np.unique(lab)) == 1
    # a real difference survives the label cost
    C = np.column_stack([np.full(n, 1.0), np.full(n, 1.0)])
    C[:25, 1] += 1.0
    C[25:, 0] += 1.0
    lab = p3.label_with_costs(C, nbr, lam=0.1, beta=5.0)
    assert set(lab[25:]) == {1} and set(lab[:25]) == {0}
