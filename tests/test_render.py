"""GPU tests: render offscreen through EGL and inspect the pixels."""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from viewer3d import geometry as geo  # noqa: E402
from viewer3d.app import run_headless  # noqa: E402
from viewer3d.scene import Material, MeshNode, Scene  # noqa: E402
from viewer3d.scenes import build_cuboid_scene, demo_cuboids  # noqa: E402


def _egl_ok():
    try:
        moderngl.create_standalone_context(backend="egl").release()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _egl_ok(), reason="no headless OpenGL (EGL) available")
SIZE = (320, 240)


def _two_box_scene(order):
    scene = Scene()
    near = MeshNode(geo.box(), Material(color=(1, 0, 0)), name="near", position=(0, 0, 1.5))
    far = MeshNode(geo.box(2, 2, 0.2), Material(color=(0, 0, 1)), name="far", position=(0, 0, -1))
    for n in (near, far)[::order]:
        scene.add(n)
    return scene


@pytest.mark.parametrize("order", [1, -1])
def test_depth_test_independent_of_draw_order(order):
    cam = {"yaw": 0.0, "pitch": 0.0, "auto_rotate": 0.0}
    img, info = run_headless(_two_box_scene(order), None, SIZE, frames=1, camera=cam)
    px = np.asarray(img)[SIZE[1] // 2, SIZE[0] // 2].astype(int)
    assert px[0] > 2 * px[2], f"near red box must win the depth test, got {px}"
    assert info["samples"] >= 1


def test_demo_scene_renders_lit_textured_geometry():
    scene = build_cuboid_scene(demo_cuboids())
    img, info = run_headless(scene, None, SIZE, frames=2)
    a = np.asarray(img).astype(int)
    bg = np.array([round(c * 255) for c in scene.background])
    covered = (np.abs(a - bg).sum(-1) > 12).mean()
    assert covered > 0.25, "scene should cover a good part of the frame (no black screen)"
    assert a.std() > 20, "lighting/texture should produce varied shading"
    assert info["triangles"] > 1000


def test_camera_reacts_to_orbit_input():
    scene = build_cuboid_scene(demo_cuboids())
    a, _ = run_headless(scene, None, SIZE, frames=1, camera={"yaw": 0.2, "auto_rotate": 0})
    scene = build_cuboid_scene(demo_cuboids())
    b, _ = run_headless(scene, None, SIZE, frames=1, camera={"yaw": 1.4, "auto_rotate": 0})
    diff = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).mean()
    assert diff > 5


def test_shadows_darken_ground_next_to_a_box():
    from viewer3d import math3d as m3
    scene = Scene()
    scene.add(MeshNode(geo.plane(8), Material(color=(0.8, 0.8, 0.8)), cast_shadow=False))
    scene.add(MeshNode(geo.box(1, 3, 1), Material(color=(0.8, 0.8, 0.8)), position=(0, 1.5, 0)))
    scene.sun.direction = m3.normalize((-1, -1, 0)).astype(np.float32)  # light travels to -X
    cam = {"yaw": 0.0, "pitch": 1.45, "auto_rotate": 0.0, "zoom": 0.8}  # look straight down
    img, _ = run_headless(scene, None, SIZE, frames=1, camera=cam)
    a = np.asarray(img).astype(float).mean(-1)
    # Camera looks straight down with screen-right = +X. The shadow of the box
    # must stretch toward -X (left); the ground on the +X side stays lit.
    row = a[a.shape[0] // 2 - 15: a.shape[0] // 2 + 15]
    ground = row[:, (row > 60).all(0)][:, 3:-3]  # ground columns, minus anti-aliased edges
    left, right = ground[:, : ground.shape[1] // 3], ground[:, -ground.shape[1] // 3:]
    assert left.min() < right.mean() * 0.7, (left.min(), right.mean())
    assert right.min() > right.mean() * 0.85, "no shadow expected on the lit side"
