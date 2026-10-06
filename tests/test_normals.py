"""Global cuboid scale family, triangulation and normal integration on exact synthetic data."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

import lift3d  # noqa: E402
import segment3d as s3  # noqa: E402


def test_lifts_of_one_tangent_box_are_homothetic_about_the_camera():
    import cuboids_global as cg
    cam = lift3d.default_camera()
    s3.SIGNS = s3.axis_signs(cam, lift3d.vanishing_points())
    o, sz = np.array([0.9, -0.5, 0.0]), s3.SIGNS * np.array([0.12, 0.08, 0.10])
    c2, _ = cam.project(s3.box_corners(o, sz))
    oc, szc, rms = cg.canonical_box(cam, c2, 0.05)
    assert rms < 1e-3
    s = cg.scale_of(cam, oc, o)
    o2, sz2 = cg.scaled(cam, oc, szc, s)
    assert np.allclose(o2, o, atol=1e-5) and np.allclose(sz2, sz, atol=1e-5)
    blo, bhi = s3.box_extent(o, sz)
    lo, hi = cg.scale_bounds(cam, o, sz, blo - 0.05, bhi + 0.05, np.zeros(3))
    assert lo < 1.0 < hi


def test_sphere_depth_from_exact_normals():
    import normal_integration as ni
    K = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    cam = lift3d.Camera(K, np.eye(3), np.zeros(3))
    centre, r = np.array([0.0, 0.0, 2.0]), 0.4
    ys, xs = np.mgrid[100:380:4, 180:460:4]
    uv = np.c_[xs.ravel(), ys.ravel()].astype(float)
    d = cam.ray(uv)
    b = d @ centre
    disc = b ** 2 - (centre @ centre - r ** 2)
    keep = disc > 0.05 * r ** 2                                  # away from the grazing rim
    uv, d, b, disc = uv[keep], d[keep], b[keep], disc[keep]
    P = d * (b - np.sqrt(disc))[:, None]
    z = P[:, 2]
    n = (P - centre) / r
    zr = ni.integrate_normals(cam, uv, n, [0], [z[0]], robust=False)
    assert np.median(np.abs(zr - z) / z) < 0.003


def test_triangulation_recovers_a_point():
    import video_normals as vn
    K = np.array([[700.0, 0, 480], [0, 700, 270], [0, 0, 1]])
    X = np.array([0.2, 0.1, 0.5])
    cams, track = {}, []
    for i, x in enumerate(np.linspace(-1, 1, 6)):
        C = np.array([x, -4.0, 0.5])
        R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        cams[i] = lift3d.Camera(K, R, C)
        uv, _ = cams[i].project(X[None])
        track.append((i, tuple(uv[0])))
    pts, view, err, used = vn.triangulate([track], cams)
    assert used == [0] and np.allclose(pts[0], X, atol=1e-6) and err[0] < 1e-6
