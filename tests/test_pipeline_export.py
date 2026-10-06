import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cv2")
pytest.importorskip("scipy")

import opengl_drawer  # noqa: E402
from viewer3d.scenes import build_cuboid_scene, load_cuboids_json  # noqa: E402


def test_export_roundtrip_builds_scene(tmp_path):
    bottoms = [np.array([[0, 0], [40, 0], [40, 30], [0, 30]], float),
               np.array([[60, 10, 0], [90, 10, 0], [90, 50, 0], [60, 50, 0]], float)]
    path = opengl_drawer.export_cuboids_json(tmp_path / "c.json", bottoms, [50, 80],
                                             ["orange", (0.2, 0.4, 0.9)])
    data = load_cuboids_json(path)
    assert len(data["cuboids"]) == 2 and len(data["cuboids"][0]["bottom"][0]) == 3
    json.dumps(data)
    scene = build_cuboid_scene(data, ground_texture=None, with_surface=False, with_showcase=False)
    scene.update(0, 0)
    lo, hi = scene.bounds()
    assert hi[1] - lo[1] > 0, "cuboids must extend upward (Y-up) in the viewer"
    assert (hi - lo)[[0, 2]].max() == pytest.approx(10 * 1.3, rel=0.05)  # scaled + ground margin
