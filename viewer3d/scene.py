"""Scene graph: nodes with hierarchical transforms, meshes, lights, camera."""
import math

import numpy as np

from . import math3d as m3


class Node:
    """A transform in the scene graph. World = parent.world @ local."""

    def __init__(self, name="node", position=(0, 0, 0), rotation=None, scale=1.0):
        self.name = name
        self.position = np.asarray(position, dtype=np.float32)
        self.rotation = rotation or m3.Quaternion()
        self.scale = np.broadcast_to(np.asarray(scale, dtype=np.float32), (3,)).copy()
        self.children = []
        self.parent = None
        self.world_matrix = m3.identity()
        self.on_update = None  # optional callable(node, dt, t)

    def add(self, child):
        if child.parent is not None:
            child.parent.children.remove(child)
        child.parent = self
        self.children.append(child)
        return child

    def local_matrix(self):
        return m3.translate(self.position) @ self.rotation.to_matrix() @ m3.scale(self.scale)

    def update(self, dt, t, parent_world=None):
        if self.on_update is not None:
            self.on_update(self, dt, t)
        local = self.local_matrix()
        self.world_matrix = local if parent_world is None else parent_world @ local
        for c in self.children:
            c.update(dt, t, self.world_matrix)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


class Material:
    def __init__(self, color=(0.8, 0.8, 0.8), texture=None, shininess=32.0,
                 specular=0.3, double_sided=False, wireframe=False):
        self.color = tuple(float(c) for c in color)
        self.texture = texture          # path to an image file or None
        self.shininess = float(shininess)
        self.specular = float(specular)
        self.double_sided = double_sided
        self.wireframe = wireframe


class MeshNode(Node):
    def __init__(self, geometry, material=None, name="mesh", cast_shadow=True, **kw):
        super().__init__(name=name, **kw)
        self.geometry = geometry
        self.material = material or Material()
        self.cast_shadow = cast_shadow


class DirectionalLight:
    def __init__(self, direction=(-0.5, -1.0, -0.3), color=(1.0, 0.96, 0.88), intensity=1.1):
        self.direction = m3.normalize(direction).astype(np.float32)  # direction light travels
        self.color = np.asarray(color, dtype=np.float32)
        self.intensity = float(intensity)


class AmbientLight:
    def __init__(self, color=(0.6, 0.65, 0.8), intensity=0.45):
        self.color = np.asarray(color, dtype=np.float32)
        self.intensity = float(intensity)


class PerspectiveCamera:
    def __init__(self, fov=50.0, aspect=16 / 9, near=0.05, far=200.0):
        self.fov, self.aspect, self.near, self.far = fov, aspect, near, far
        self.position = np.array([0.0, 2.0, 6.0], dtype=np.float32)
        self.target = np.zeros(3, dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def view_matrix(self):
        return m3.look_at(self.position, self.target, self.up)

    def projection_matrix(self):
        return m3.perspective(self.fov, self.aspect, self.near, self.far)


class OrbitControls:
    """Spherical orbit around a target: rotate, dolly (zoom) and pan."""

    def __init__(self, camera, target=(0, 0, 0), distance=6.0, yaw=0.6, pitch=0.55,
                 min_distance=0.5, max_distance=100.0):
        self.camera = camera
        self.target = np.asarray(target, dtype=np.float32)
        self.distance = float(distance)
        self.yaw, self.pitch = float(yaw), float(pitch)
        self.min_distance, self.max_distance = min_distance, max_distance
        self.auto_rotate = 0.0  # rad/s, used when idle
        self.apply()

    def rotate(self, d_yaw, d_pitch):
        self.yaw += d_yaw
        self.pitch = float(np.clip(self.pitch + d_pitch, -1.5, 1.5))

    def zoom(self, factor):
        self.distance = float(np.clip(self.distance * factor, self.min_distance, self.max_distance))

    def pan(self, dx, dy):
        """Pan in screen space; dx, dy are fractions of the viewport height."""
        view = self.camera.view_matrix()
        right, up = view[0, :3], view[1, :3]
        h = 2.0 * self.distance * math.tan(math.radians(self.camera.fov) / 2.0)
        self.target = (self.target - right * dx * h + up * dy * h).astype(np.float32)

    def update(self, dt):
        self.yaw += self.auto_rotate * dt
        self.apply()

    def apply(self):
        cp = math.cos(self.pitch)
        offset = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        self.camera.target = self.target.copy()
        self.camera.position = (self.target + offset * self.distance).astype(np.float32)
        # keep the depth range tight around the scene for good depth precision
        self.camera.near = max(0.01, self.distance * 0.01)
        self.camera.far = self.distance * 10.0 + 50.0


class Scene:
    def __init__(self):
        self.root = Node("root")
        self.sun = DirectionalLight()
        self.ambient = AmbientLight()
        self.background = (0.09, 0.1, 0.13)

    def add(self, node):
        return self.root.add(node)

    def update(self, dt, t):
        self.root.update(dt, t)

    def meshes(self):
        return [n for n in self.root.walk() if isinstance(n, MeshNode)]

    def bounds(self):
        """World-space AABB of all meshes (call after update)."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for n in self.meshes():
            p = n.geometry.positions
            pw = (n.world_matrix[:3, :3] @ p.T).T + n.world_matrix[:3, 3]
            lo, hi = np.minimum(lo, pw.min(0)), np.maximum(hi, pw.max(0))
        return lo, hi
