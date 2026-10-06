import math

import numpy as np
import pytest

from viewer3d import geometry as geo
from viewer3d import math3d as m3
from viewer3d.scene import Node, OrbitControls, PerspectiveCamera


def project(m, p):
    v = m @ np.r_[p, 1.0]
    return v[:3] / v[3]


def test_perspective_maps_near_and_far_to_ndc():
    proj = m3.perspective(60, 1.5, 0.1, 100)
    assert project(proj, [0, 0, -0.1])[2] == pytest.approx(-1, abs=1e-5)
    assert project(proj, [0, 0, -100])[2] == pytest.approx(1, abs=1e-4)


def test_look_at_puts_target_on_negative_z():
    v = m3.look_at([3, 4, 5], [0, 0, 0])
    p = v @ np.array([0, 0, 0, 1.0])
    assert p[:2] == pytest.approx([0, 0], abs=1e-5)
    assert p[2] == pytest.approx(-math.sqrt(50), abs=1e-5)


def test_quaternion_rotation_matches_axis_angle():
    q = m3.Quaternion.from_axis_angle((0, 1, 0), math.pi / 2)
    assert q.rotate([1, 0, 0]) == pytest.approx([0, 0, -1], abs=1e-6)
    q2 = q * q
    assert q2.rotate([1, 0, 0]) == pytest.approx([-1, 0, 0], abs=1e-6)


def test_hierarchical_transforms_compose():
    root = Node("root", position=(10, 0, 0))
    child = root.add(Node("child", position=(0, 1, 0),
                          rotation=m3.Quaternion.from_axis_angle((0, 0, 1), math.pi / 2)))
    grandchild = child.add(Node("gc", position=(1, 0, 0)))
    root.update(0, 0)
    # (1,0,0) rotated 90deg about Z -> (0,1,0); plus child (0,1,0) and root (10,0,0)
    assert grandchild.world_matrix[:3, 3] == pytest.approx([10, 2, 0], abs=1e-5)


def test_orbit_controls_clamp_and_pan():
    cam = PerspectiveCamera()
    c = OrbitControls(cam, distance=5, min_distance=1, max_distance=10)
    c.zoom(100)
    assert c.distance == 10
    c.rotate(0, 10)
    assert c.pitch <= 1.5
    before = c.target.copy()
    c.pan(0.1, 0)
    assert not np.allclose(before, c.target)
    c.apply()
    assert np.linalg.norm(cam.position - cam.target) == pytest.approx(10, rel=1e-5)


CLOSED = {
    "box": lambda: geo.box(1, 2, 3),
    "sphere": lambda: geo.sphere(1, 16, 8),
    "torus": lambda: geo.torus(1, 0.3, 12, 24),
    "cuboid_cw": lambda: geo.cuboid_from_bottom([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]], 2),
    "cuboid_ccw": lambda: geo.cuboid_from_bottom([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]], 2),
}


@pytest.mark.parametrize("name", CLOSED)
def test_closed_meshes_are_outward_ccw_with_unit_normals(name):
    g = CLOSED[name]()
    assert np.linalg.norm(g.normals, axis=1) == pytest.approx(1.0, abs=1e-4)
    assert g.uvs.min() >= 0 and g.uvs.max() <= 1
    tri = g.indices.reshape(-1, 3)
    p = g.positions
    fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    ok = area > 1e-9  # degenerate pole triangles carry no orientation
    vn = g.normals[tri[ok]].mean(1)
    assert (np.einsum("ij,ij->i", fn[ok], vn) > 0).all()
    # outward: centroid -> face points the same way as the face normal
    centroid = p.mean(0)
    if name != "torus":  # torus is not star-shaped about its centre
        fc = p[tri[ok]].mean(1)
        assert (np.einsum("ij,ij->i", fn[ok], fc - centroid) > 0).all()


def test_heightfield_and_obj_roundtrip(tmp_path):
    pts = np.array([[0, 0, 0], [1, 1, 0], [0, 2, 1], [1, 0.5, 1], [0.5, 3, 0.5]], float)
    g = geo.heightfield(pts, resolution=8)
    assert g.triangle_count == 2 * 7 * 7
    assert not np.isnan(g.positions).any()
    assert (g.normals[:, 1] != 0).any()

    obj = tmp_path / "quad.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 0 -1\nv 0 0 -1\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                   "f 1/1 2/2 3/3 4/4\n")
    q = geo.load_obj(obj)
    assert q.triangle_count == 2
    assert q.normals[:, 1] == pytest.approx(1.0)
