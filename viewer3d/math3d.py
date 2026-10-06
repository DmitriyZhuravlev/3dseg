"""Small 3D math library: 4x4 matrices and quaternions (numpy, float32).

Conventions: right-handed, Y-up world, column vectors (p' = M @ p), matrices
are stored row-major in numpy and transposed when uploaded to OpenGL.
"""
import math

import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def identity():
    return np.eye(4, dtype=np.float32)


def translate(t):
    m = identity()
    m[:3, 3] = t
    return m


def scale(s):
    s = np.broadcast_to(np.asarray(s, dtype=np.float32), (3,))
    return np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)


def perspective(fov_y_deg, aspect, near, far):
    """OpenGL perspective projection (clip z in [-w, w])."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left, right, bottom, top, near, far):
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    f = normalize(np.asarray(target, dtype=np.float64) - eye)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:  # looking straight along `up`
        s = np.cross(f, (0.0, 0.0, 1.0))
    s = normalize(s)
    u = np.cross(s, f)
    m = identity()
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Quaternion:
    """Unit quaternion (w, x, y, z) for rotations."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = float(w), float(x), float(y), float(z)

    @classmethod
    def from_axis_angle(cls, axis, angle_rad):
        ax = normalize(axis)
        s = math.sin(angle_rad / 2.0)
        return cls(math.cos(angle_rad / 2.0), ax[0] * s, ax[1] * s, ax[2] * s)

    @classmethod
    def from_euler(cls, x, y, z):
        """Intrinsic rotations applied in X, then Y, then Z order (radians)."""
        return (cls.from_axis_angle((0, 0, 1), z) * cls.from_axis_angle((0, 1, 0), y)
                * cls.from_axis_angle((1, 0, 0), x))

    def __mul__(self, o):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = o.w, o.x, o.y, o.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def normalized(self):
        n = math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def rotate(self, v):
        return self.to_matrix()[:3, :3] @ np.asarray(v, dtype=np.float32)

    def to_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)


def normal_matrix(model):
    """Inverse-transpose of the upper 3x3, for transforming normals."""
    return np.linalg.inv(model[:3, :3]).T.astype(np.float32)
