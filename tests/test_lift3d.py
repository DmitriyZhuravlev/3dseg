"""3D accuracy of lift3d: against the traced box edges and synthetic shapes with known 3D."""
import os

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

import lift3d  # noqa: E402
import synthetic3d  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
have_photos = all(os.path.exists(os.path.join(REPO, f)) for f in ("reference.JPG", "box.JPG"))


def test_camera_reproduces_vanishing_points():
    vps = lift3d.vanishing_points()
    cam = lift3d.default_camera()
    for name, d in (("left", [1, 0, 0]), ("right", [0, 1, 0]), ("vertical", [0, 0, -1])):
        p = cam.K @ cam.R @ np.array(d, float)
        assert p[:2] / p[2] == pytest.approx(vps[name], rel=1e-6)
    assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-9)


@pytest.mark.skipif(not have_photos, reason="photos not available")
def test_box_photo_matches_traced_3d_box():
    cam = lift3d.default_camera()
    gt = lift3d.box_ground_truth(cam)
    r = lift3d.lift(os.path.join(REPO, "box.JPG"))
    assert r["shape"] == "box"
    dims = lift3d.model_extent(r["cuboids"], r["box"][5])  # yaw ~ 0: (length along X, width along Y, height)
    for got, want, tol in zip(dims, gt["dims"], gt["uncertainty"]):
        assert abs(got - want) <= max(tol, 0.03 * want), (dims, gt["dims"], gt["uncertainty"])
    assert lift3d.volume_iou(r["cuboids"], gt["corners"])["volume_iou"] > 0.85


@pytest.mark.parametrize("name,min_iou", [("box", 0.9), ("cylinder", 0.8), ("pig", 0.6)])
def test_synthetic_shapes_volume_iou(name, min_iou):
    scores = synthetic3d.evaluate([name])[name]
    assert scores["volume_iou"] >= min_iou, scores
