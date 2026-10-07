"""segment3d: lifting VP-tangent boxes and planar patches with the calibrated camera."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

import lift3d  # noqa: E402
import segment3d as s3  # noqa: E402


@pytest.fixture(scope="module")
def cam():
    c = lift3d.default_camera()
    s3.SIGNS = s3.axis_signs(c, lift3d.vanishing_points())
    return c


def test_tangent_box_lifts_exactly_with_one_known_plane(cam):
    origin = np.array([0.9, -0.5, 0.0])
    size = s3.SIGNS * np.array([0.3, 0.4, 0.25])
    uv, _ = cam.project(s3.box_corners(origin, size))
    o, sz, rms = s3.lift_tangent_box(cam, uv, axis=2, side=0, value=0.0)     # bottom face on the ground
    assert rms < 0.05
    assert o == pytest.approx(origin, abs=1e-4)
    assert sz == pytest.approx(size, abs=1e-4)
    # same box from its top plane
    o2, sz2, _ = s3.lift_tangent_box(cam, uv, axis=2, side=1, value=0.25)
    assert o2 == pytest.approx(origin, abs=1e-4)


def test_backprojected_patch_lies_on_its_plane(cam):
    uv = np.array([[1500.0, 900.0], [1600.0, 1000.0], [1400.0, 1100.0]])
    P, t = s3.backproject_plane(cam, uv, axis=2, value=0.1)
    assert (t > 0).all()
    assert P[:, 2] == pytest.approx(0.1)
    back, _ = cam.project(P)
    assert back == pytest.approx(uv, abs=1e-6)


def test_depth_errors_and_box_depth(cam):
    origin = np.array([0.9, -0.5, 0.0])
    size = s3.SIGNS * np.array([0.3, 0.4, 0.25])
    corners = s3.box_corners(origin, size)
    uv, _ = cam.project(corners.mean(0)[None])
    z = s3.depth_from_boxes(cam, uv, [corners])
    assert np.isfinite(z).all()
    e = s3.depth_errors(z, z)
    assert e["median_rel"] == 0.0 and e["within_1pct"] == 1.0


def test_video_working_frame_orders_vps_and_maps_boxes_back():
    import video_segments as vs
    K = np.array([[800.0, 0, 480], [0, 800, 270], [0, 0, 1]])
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])          # looking along +Y, image y down
    cam = lift3d.Camera(K, R, np.array([0.0, -8.0, 1.5]))
    for yaw in (0.4, 0.4 + np.pi / 2, 0.4 + np.pi, -1.2):
        pose = (0.5, 1.0, yaw, 0.0)
        oc, Q = vs.working_frame(cam, pose)
        v = vs.object_vps(oc)
        assert v["left"][0] <= v["right"][0]
        # the same world point projects identically through both object-frame cameras
        p_obj = np.array([0.3, -0.2, 0.7])
        assert np.allclose(oc.project(Q.T @ p_obj)[0], vs.object_camera(cam, pose).project(p_obj)[0], atol=1e-6)
        lo, hi = vs.rotate_extent(Q, np.array([-1.0, -0.3, 0.0]), np.array([1.0, 0.3, 1.2]))
        assert np.allclose(hi - lo, np.abs(Q.T) @ np.array([2.0, 0.6, 1.2]))
