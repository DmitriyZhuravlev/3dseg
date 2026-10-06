"""Synthetic 3D ground truth for lift3d: known shapes rendered through the calibrated camera.

A silhouette only constrains what is visible, so the 3D accuracy of the lifting is
measured here on objects whose full 3D shape is known: each shape is an implicit
function inside(points) -> bool in world coordinates (Z up, ground Z = 0, units =
camera heights, placed where the real objects stand). Its mask is rendered with the
same camera as the photos, lifted with lift3d.lift_mask, and the cuboids are scored
by Monte-Carlo volume IoU against the shape.

    python synthetic3d.py
"""
import math

import cv2
import numpy as np

import lift3d

PIG_CENTRE = (0.98, -0.56)   # where pig.JPG's pig stands on the ground (lift3d box fit)


def _rot(points, centre, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = points.copy()
    p[:, 0] -= centre[0]
    p[:, 1] -= centre[1]
    x = c * p[:, 0] + s * p[:, 1]
    y = -s * p[:, 0] + c * p[:, 1]
    return np.c_[x, y, p[:, 2]]


def ellipsoid(c, r):
    return lambda p: (((p - c) / r) ** 2).sum(1) <= 1.0


def make_pig(centre=PIG_CENTRE, yaw=math.radians(-10)):
    """Body ellipsoid lying on the ground, head sphere, snout, two legs pairs folded, an ear."""
    body = ellipsoid(np.array([0.0, 0.0, 0.045]), np.array([0.20, 0.065, 0.045]))
    head = ellipsoid(np.array([-0.20, 0.0, 0.05]), np.array([0.06, 0.06, 0.05]))
    snout = ellipsoid(np.array([-0.26, 0.0, 0.035]), np.array([0.03, 0.035, 0.03]))
    ear = ellipsoid(np.array([-0.19, 0.035, 0.11]), np.array([0.015, 0.02, 0.04]))
    legs = ellipsoid(np.array([0.21, 0.0, 0.025]), np.array([0.06, 0.07, 0.025]))

    def inside(p):
        q = _rot(p, centre, yaw)
        return body(q) | head(q) | snout(q) | ear(q) | legs(q)

    bounds = (np.array([centre[0] - 0.35, centre[1] - 0.35, 0.0]), np.array([centre[0] + 0.35, centre[1] + 0.35, 0.16]))
    return inside, bounds


def make_cylinder(centre=PIG_CENTRE, yaw=math.radians(25), length=0.4, radius=0.06):
    def inside(p):
        q = _rot(p, centre, yaw)
        return (np.abs(q[:, 0]) <= length / 2) & (q[:, 1] ** 2 + (q[:, 2] - radius) ** 2 <= radius ** 2)

    bounds = (np.array([centre[0] - 0.3, centre[1] - 0.3, 0.0]), np.array([centre[0] + 0.3, centre[1] + 0.3, 2 * radius]))
    return inside, bounds


def make_box(centre=PIG_CENTRE, yaw=math.radians(20), dims=(0.3, 0.2, 0.15)):
    def inside(p):
        q = _rot(p, centre, yaw)
        return (np.abs(q[:, 0]) <= dims[0] / 2) & (np.abs(q[:, 1]) <= dims[1] / 2) & (q[:, 2] <= dims[2])

    r = max(dims[:2])
    bounds = (np.array([centre[0] - r, centre[1] - r, 0.0]), np.array([centre[0] + r, centre[1] + r, dims[2]]))
    return inside, bounds


SHAPES = {"pig": make_pig, "cylinder": make_cylinder, "box": make_box}


def render_mask(cam, inside, bounds, shape=(1760, 3036), n=3_000_000, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    pts = rng.uniform(lo, hi, size=(n, 3))
    pts = pts[inside(pts)]
    uv, depth = cam.project(pts)
    uv = np.round(uv[depth > 0]).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    mask = np.zeros(shape, np.uint8)
    mask[uv[ok, 1], uv[ok, 0]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask > 0


def volume_scores(cuboids, inside, bounds, n=600_000, seed=1):
    rng = np.random.default_rng(seed)
    corners = [np.vstack([np.asarray(c["bottom"]), np.asarray(c["bottom"]) + [0, 0, c["height"]]]) for c in cuboids]
    lo = np.minimum(bounds[0], np.min([c.min(0) for c in corners], axis=0))
    hi = np.maximum(bounds[1], np.max([c.max(0) for c in corners], axis=0))
    pts = rng.uniform(lo, hi, size=(n, 3))
    in_gt = inside(pts)
    in_model = np.zeros(n, bool)
    for c in cuboids:
        z0 = float(np.asarray(c["bottom"])[:, 2].min())
        in_model |= lift3d._inside_prism(pts, c["bottom"], z0, z0 + c["height"])
    inter, union = (in_gt & in_model).sum(), (in_gt | in_model).sum()
    return {"volume_iou": inter / union, "volume_precision": inter / max(in_model.sum(), 1),
            "volume_recall": inter / max(in_gt.sum(), 1)}


def evaluate(names=None, **lift_kwargs):
    cam = lift3d.default_camera()
    out = {}
    for name in names or SHAPES:
        inside, bounds = SHAPES[name]()
        mask = render_mask(cam, inside, bounds)
        r = lift3d.lift_mask(cam, mask, None, **lift_kwargs)
        out[name] = {**volume_scores(r["cuboids"], inside, bounds), "silhouette_iou": r["scores"]["iou"],
                     "shape": r["shape"], "cuboids": len(r["cuboids"])}
    return out


if __name__ == "__main__":
    for name, s in evaluate().items():
        print(f"{name:9s} [{s['shape']}] volume IoU {s['volume_iou']:.3f} (precision {s['volume_precision']:.3f}, "
              f"recall {s['volume_recall']:.3f}); silhouette IoU {s['silhouette_iou']:.3f}; {s['cuboids']} cuboids")
