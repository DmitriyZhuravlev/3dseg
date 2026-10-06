"""Geometry: indexed triangle meshes with normals and UVs.

Primitives (box, sphere, torus, plane) plus the custom meshes this project
needs: cuboids lifted from image segments (bottom quad + height, the format
produced by the segmentation scripts), a height-field surface interpolated
over cuboid tops, and a minimal Wavefront OBJ loader.

All triangles are counter-clockwise when seen from the outside (front face).
"""
import math

import numpy as np


class Geometry:
    def __init__(self, positions, normals, uvs, indices):
        self.positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32).reshape(-1, 3)
        self.uvs = np.ascontiguousarray(uvs, dtype=np.float32).reshape(-1, 2)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
        n = len(self.positions)
        assert len(self.normals) == n and len(self.uvs) == n, "attribute count mismatch"
        assert self.indices.max(initial=0) < n, "index out of range"

    def interleaved(self):
        return np.hstack([self.positions, self.normals, self.uvs]).astype(np.float32)

    @property
    def triangle_count(self):
        return len(self.indices) // 3


def _quads_to_geometry(quads, uvs_per_quad=None):
    """Flat-shaded geometry from a list of CCW quads (4x3 each)."""
    pos, nrm, uv, idx = [], [], [], []
    default_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i, q in enumerate(quads):
        q = np.asarray(q, dtype=np.float64)
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if np.linalg.norm(n) < 1e-12:
            n = np.cross(q[2] - q[0], q[3] - q[0])
        n = n / (np.linalg.norm(n) or 1.0)
        base = len(pos)
        pos.extend(q)
        nrm.extend([n] * 4)
        uv.extend(uvs_per_quad[i] if uvs_per_quad else default_uv)
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return Geometry(pos, nrm, uv, idx)


def box(w=1.0, h=1.0, d=1.0):
    x, y, z = w / 2, h / 2, d / 2
    quads = [
        [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)],      # +Z
        [(x, -y, -z), (-x, -y, -z), (-x, y, -z), (x, y, -z)],  # -Z
        [(x, -y, z), (x, -y, -z), (x, y, -z), (x, y, z)],      # +X
        [(-x, -y, -z), (-x, -y, z), (-x, y, z), (-x, y, -z)],  # -X
        [(-x, y, z), (x, y, z), (x, y, -z), (-x, y, -z)],      # +Y
        [(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)],  # -Y
    ]
    return _quads_to_geometry(quads)


def plane(size=10.0, uv_repeat=1.0):
    s = size / 2
    q = [[(-s, 0, s), (s, 0, s), (s, 0, -s), (-s, 0, -s)]]
    r = uv_repeat
    return _quads_to_geometry(q, [[(0, 0), (r, 0), (r, r), (0, r)]])


def sphere(radius=1.0, segments=48, rings=24):
    pos, nrm, uv = [], [], []
    for r in range(rings + 1):
        v = r / rings
        phi = v * math.pi
        for s in range(segments + 1):
            u = s / segments
            theta = u * 2 * math.pi
            n = (math.sin(phi) * math.sin(theta), math.cos(phi), math.sin(phi) * math.cos(theta))
            nrm.append(n)
            pos.append([radius * c for c in n])
            uv.append((u, 1 - v))
    idx = []
    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a, b = r * row + s, (r + 1) * row + s
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return Geometry(pos, nrm, uv, idx)


def torus(radius=1.0, tube=0.3, radial=24, tubular=64):
    pos, nrm, uv = [], [], []
    for j in range(radial + 1):
        v = j / radial * 2 * math.pi
        for i in range(tubular + 1):
            u = i / tubular * 2 * math.pi
            cx, cz = radius * math.cos(u), -radius * math.sin(u)
            p = ((radius + tube * math.cos(v)) * math.cos(u), tube * math.sin(v),
                 -(radius + tube * math.cos(v)) * math.sin(u))
            pos.append(p)
            n = np.array([p[0] - cx, p[1], p[2] - cz])
            nrm.append(n / np.linalg.norm(n))
            uv.append((i / tubular, j / radial))
    idx = []
    row = tubular + 1
    for j in range(radial):
        for i in range(tubular):
            a, b = j * row + i, (j + 1) * row + i
            idx += [a, a + 1, b, a + 1, b + 1, b]
    return Geometry(pos, nrm, uv, idx)


def cuboid_from_bottom(bottom, height):
    """Prism from a bottom quad (4x3 or 4x2, Y-up) extruded by `height` along +Y.

    The bottom points may be in either winding order; faces are oriented
    outward automatically.
    """
    b = np.asarray(bottom, dtype=np.float64)
    if b.shape[1] == 2:
        b = np.column_stack([b[:, 0], np.zeros(4), b[:, 1]])
    t = b + np.array([0.0, float(height), 0.0])
    # Seen from above (+Y) the bottom must be CCW: signed area in the XZ plane
    # (x, -z) basis > 0 for CCW from above.
    area = sum(b[i, 0] * -b[(i + 1) % 4, 2] - b[(i + 1) % 4, 0] * -b[i, 2] for i in range(4))
    if area < 0:
        b, t = b[::-1], t[::-1]
    quads = [t.tolist(), b[::-1].tolist()]
    for i in range(4):
        j = (i + 1) % 4
        quads.append([b[i], b[j], t[j], t[i]])
    return _quads_to_geometry(quads)


def heightfield(points, resolution=48, method="linear"):
    """Smooth surface interpolated over scattered (x, y_height, z) points.

    Mirrors opengl_drawer.draw_interpolated_surface, but produces a real mesh
    with smooth normals and UVs instead of a matplotlib plot.
    """
    from scipy.interpolate import griddata

    p = np.asarray(points, dtype=np.float64)
    xs = np.linspace(p[:, 0].min(), p[:, 0].max(), resolution)
    zs = np.linspace(p[:, 2].min(), p[:, 2].max(), resolution)
    gx, gz = np.meshgrid(xs, zs)
    gy = griddata((p[:, 0], p[:, 2]), p[:, 1], (gx, gz), method=method)
    near = griddata((p[:, 0], p[:, 2]), p[:, 1], (gx, gz), method="nearest")
    gy = np.where(np.isnan(gy), near, gy)

    pos = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    uv = np.stack(np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution)), -1).reshape(-1, 2)
    idx = []
    for r in range(resolution - 1):
        for c in range(resolution - 1):
            a, b = r * resolution + c, (r + 1) * resolution + c
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return Geometry(pos, compute_smooth_normals(pos, idx), uv, idx)


def compute_smooth_normals(positions, indices):
    p = np.asarray(positions, dtype=np.float64)
    tri = np.asarray(indices).reshape(-1, 3)
    fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln < 1e-12, 1.0, ln)


def load_obj(path):
    """Minimal OBJ loader (v / vt / vn / f, polygons fan-triangulated)."""
    vs, vts, vns = [], [], []
    pos, uv, nrm, idx = [], [], [], []
    cache = {}
    has_normals = True
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vs.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                vts.append([float(x) for x in parts[1:3]])
            elif parts[0] == "vn":
                vns.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                face = []
                for corner in parts[1:]:
                    if corner not in cache:
                        ids = (corner.split("/") + ["", ""])[:3]
                        vi = int(ids[0])
                        pos.append(vs[vi - 1 if vi > 0 else vi])
                        uv.append(vts[int(ids[1]) - 1] if ids[1] else (0.0, 0.0))
                        if ids[2]:
                            nrm.append(vns[int(ids[2]) - 1])
                        else:
                            has_normals = False
                            nrm.append((0.0, 0.0, 0.0))
                        cache[corner] = len(pos) - 1
                    face.append(cache[corner])
                for k in range(1, len(face) - 1):
                    idx += [face[0], face[k], face[k + 1]]
    if not has_normals:
        nrm = compute_smooth_normals(pos, idx)
    return Geometry(pos, nrm, uv, idx)
