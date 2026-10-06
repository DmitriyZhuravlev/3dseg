"""Scene construction: the cuboid scene produced by the segmentation pipeline.

The pipeline (img_rec_ipm.py, segment_rec.py, surf.py, ...) lifts image
segments to cuboids described by a bottom quad (4 points, Z-up, in BEV/pixel
units), a height and a colour — see opengl_drawer.draw_cubes_with_bounding_image.
`opengl_drawer.export_cuboids_json` writes exactly that to JSON, and
`build_cuboid_scene` turns it into a lit, shadowed, textured 3D scene.
"""
import json
import math
import os

import numpy as np

from . import geometry as geo
from . import math3d as m3
from .scene import Material, MeshNode, Node, Scene

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GROUND_TEXTURE = os.path.join(REPO_ROOT, "Background.bmp")


def load_cuboids_json(path):
    with open(path) as f:
        data = json.load(f)
    data.setdefault("z_up", True)
    return data


def demo_cuboids(seed=7, grid=9, spacing=60.0):
    """Procedural stand-in for pipeline output: one cuboid per 'superpixel'.

    Jittered, rotated footprints on a grid with heights from a smooth bump
    field, in the pipeline's units (Z-up, ~pixels).
    """
    rng = np.random.default_rng(seed)
    cuboids = []
    palette = [(0.91, 0.36, 0.30), (0.25, 0.58, 0.86), (0.98, 0.75, 0.25),
               (0.42, 0.75, 0.45), (0.70, 0.45, 0.85), (0.30, 0.80, 0.80)]
    for i in range(grid):
        for j in range(grid):
            if rng.random() < 0.25:  # gaps, like unsegmented background
                continue
            cx = (i - (grid - 1) / 2) * spacing + rng.normal(0, 6)
            cy = (j - (grid - 1) / 2) * spacing + rng.normal(0, 6)
            w, l = rng.uniform(25, 45), rng.uniform(25, 45)
            a = rng.uniform(0, math.pi / 2)
            ca, sa = math.cos(a), math.sin(a)
            corners = [(-w / 2, -l / 2), (w / 2, -l / 2), (w / 2, l / 2), (-w / 2, l / 2)]
            bottom = [[cx + x * ca - y * sa, cy + x * sa + y * ca, 0.0] for x, y in corners]
            r = math.hypot(cx, cy) / (spacing * grid / 2)
            h = 20 + 140 * math.exp(-3 * r * r) * (0.7 + 0.3 * rng.random())
            cuboids.append({"bottom": bottom, "height": h,
                            "color": palette[(i * grid + j) % len(palette)]})
    return {"cuboids": cuboids, "z_up": True}


def build_cuboid_scene(data, ground_texture=DEFAULT_GROUND_TEXTURE, with_surface=True,
                       with_showcase=True):
    """Build the 3D scene.

    Hierarchy:
        root
        └── world (recentre + uniform scale to ~10 units)
            ├── cuboids (group) ── cuboid_0 … cuboid_N
            ├── ground (textured plane)
            └── surface (height field interpolated over cuboid tops)
        └── showcase (spinning pivot) ── sphere ── torus (tilted ring) ── moon (cube on the ring)
    """
    scene = Scene()
    cubs = data["cuboids"]
    if not cubs:
        raise ValueError("scene has no cuboids")

    bottoms = np.array([c["bottom"] for c in cubs], dtype=np.float64)
    if bottoms.shape[2] == 2:
        bottoms = np.concatenate([bottoms, np.zeros(bottoms.shape[:2] + (1,))], axis=2)
    heights = np.array([float(c["height"]) for c in cubs])

    def to_local(p):
        """Pipeline (x, y, z-up) -> GL Y-up (x, z, -y); a proper rotation, not a mirror."""
        p = np.asarray(p, dtype=np.float64)
        if not data.get("z_up", True):
            return p.copy()
        return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])

    # `world` recentres the data and scales it to ~10 units; children keep
    # pipeline units, so lighting/camera parameters are independent of input scale.
    xy = bottoms[:, :, :2].reshape(-1, 2)
    extent = float(np.ptp(xy, axis=0).max()) or 1.0
    s = 10.0 / extent
    center = to_local(np.r_[xy.mean(0), bottoms[:, :, 2].min()][None])[0]
    world = scene.add(Node("world", position=-center * s, scale=s))
    group = world.add(Node("cuboids"))

    for k, (c, b, h) in enumerate(zip(cubs, bottoms, heights)):
        color = c.get("color") or (0.6, 0.6, 0.65)
        if max(color) > 1.0:  # accept 0..255 colours from cv2-style code
            color = [v / 255.0 for v in color]
        g = geo.cuboid_from_bottom(to_local(b), h)
        group.add(MeshNode(g, Material(color=color, shininess=48, specular=0.35), name=f"cuboid_{k}"))

    # Ground plane under everything, textured with a frame from the project.
    margin = extent * 0.15
    lo, hi = xy.min(0) - margin, xy.max(0) + margin
    zmin = float(bottoms[:, :, 2].min())
    gquad = [[lo[0], lo[1], zmin], [hi[0], lo[1], zmin], [hi[0], hi[1], zmin], [lo[0], hi[1], zmin]]
    ground_geo = geo._quads_to_geometry([to_local(gquad).tolist()])
    # _quads_to_geometry orients by vertex order; ensure the normal points up (+Y).
    if ground_geo.normals[0, 1] < 0:
        ground_geo = geo._quads_to_geometry([to_local(gquad)[::-1].tolist()],
                                            [[(0, 0), (0, 1), (1, 1), (1, 0)]])
    tex = ground_texture if ground_texture and os.path.exists(ground_texture) else None
    world.add(MeshNode(ground_geo, Material(color=(1, 1, 1) if tex else (0.45, 0.47, 0.5),
                                            texture=tex, shininess=8, specular=0.05,
                                            double_sided=True),
                       name="ground", cast_shadow=False))

    if with_surface and len(cubs) >= 4:
        # Custom mesh: smooth surface through the cuboid tops, floating above them.
        tops = bottoms.mean(1)
        tops[:, 2] += heights
        pts = to_local(tops)
        pts[:, 1] += heights.max() * 0.6
        surf = geo.heightfield(pts, resolution=28)
        # Drawn as a wireframe so the cuboids below stay visible.
        world.add(MeshNode(surf, Material(color=(0.55, 0.85, 1.0), specular=0.0, double_sided=True,
                                          wireframe=True), name="surface", cast_shadow=False))

    if with_showcase:
        # Hierarchical transform demo: pivot spins; the tilted torus and its cube "moon" inherit it.
        pivot = scene.add(Node("showcase", position=(0.0, 5.2, 0.0)))
        sphere = pivot.add(MeshNode(geo.sphere(0.55), Material(color=(0.95, 0.95, 0.97),
                                                                shininess=96, specular=0.8),
                                    name="sphere"))
        torus = sphere.add(MeshNode(geo.torus(0.9, 0.12), Material(color=(1.0, 0.55, 0.15),
                                                                     shininess=64, specular=0.6),
                                    name="torus"))
        torus.add(MeshNode(geo.box(0.22, 0.22, 0.22), Material(color=(0.3, 0.75, 1.0)),
                           name="moon", position=(0.9, 0.0, 0.0)))

        def spin(node, dt, t):
            node.rotation = m3.Quaternion.from_axis_angle((0, 1, 0), t * 0.8)

        def wobble(node, dt, t):
            node.rotation = m3.Quaternion.from_euler(0.6 + 0.3 * math.sin(t * 1.3), t * 1.7, 0.0)

        pivot.on_update = spin
        torus.on_update = wobble

    scene.sun.direction = m3.normalize((-0.6, -0.75, -0.4)).astype(np.float32)
    return scene


def frame_scene(scene, controls):
    """Point the orbit controls at the scene's bounding sphere."""
    scene.update(0.0, 0.0)
    lo, hi = scene.bounds()
    controls.target = ((lo + hi) / 2).astype(np.float32)
    radius = float(np.linalg.norm(hi - lo) / 2)
    controls.distance = radius / math.sin(math.radians(controls.camera.fov) / 2) * 0.7
    controls.min_distance, controls.max_distance = radius * 0.1, radius * 8
    controls.apply()
