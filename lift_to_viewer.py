"""Run the surf.py lifting pipeline headlessly on photos and open the result in 3D.

    python lift_to_viewer.py box.JPG pig.JPG               # writes <name>_cuboids.json + <name>_marked.png
    python -m viewer3d --scene box_cuboids.json            # view one in 3D

surf.py is a script (hard-coded data paths, cv2/plt windows, a process_images call
at the bottom), so this loads everything above that call, disables the GUI calls,
and captures the cuboids it would have drawn with draw_cubes_with_bounding_image.
The camera's vanishing points are hard-coded in surf.py for the reference.JPG set-up.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))


def load_pipeline():
    for name in ("imshow", "waitKey", "destroyAllWindows", "namedWindow"):
        setattr(cv2, name, lambda *a, **k: 0)
    plt.show = plt.pause = lambda *a, **k: None
    plt.waitforbuttonpress = lambda *a, **k: True

    path = os.path.join(REPO, "surf.py")
    src = open(path).read()
    cut = src.index('reference_image_path = "/home/dzhura')  # the script's own run section
    ns = {"__name__": "surf", "__file__": path}
    sys.path.insert(0, REPO)
    exec(compile(src[:cut], path, "exec"), ns)

    # The recursive cube growth recomputes the same box for the same mask many
    # times; memoise it (pure function of the mask for fixed vanishing points).
    original, cache = ns["get_projected_box"], {}

    def get_projected_box(mask, *a, **k):
        key = (hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=16).digest(), repr(a), repr(sorted(k.items())))
        if key not in cache:
            cache[key] = original(mask, *a, **k)
        return cache[key]

    ns["get_projected_box"] = get_projected_box
    return ns


def lift(ns, image_path, reference, out_dir, region_size, ruler):
    captured = []
    ns["draw_cubes_with_bounding_image"] = lambda img, bottoms, heights, colors, *a, **k: \
        captured.append((img.copy(), [np.asarray(b) for b in bottoms], list(heights), list(colors)))
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(image_path, tmp)
        ns["process_images"](reference, tmp, os.path.join(out_dir, "pipeline_out"),
                             region_size=region_size, ruler=ruler, method="otsu")
    if not captured:
        return None
    import opengl_drawer
    img, bottoms, heights, colors = captured[-1]  # final, most complete set
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0])
    cv2.imwrite(stem + "_marked.png", img)
    opengl_drawer.export_cuboids_json(stem + "_cuboids.json", bottoms, heights, colors)
    return stem, len(bottoms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--reference", default=os.path.join(REPO, "reference.JPG"), help="empty-scene photo")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--region-size", type=int, default=160, help="SLIC region size (surf.py uses 160-320)")
    ap.add_argument("--ruler", type=int, default=120, help="SLIC ruler")
    ap.add_argument("--show", action="store_true", help="open the last result in the 3D viewer")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ns = load_pipeline()
    last = None
    for path in args.images:
        t = time.time()
        res = lift(ns, os.path.abspath(path), os.path.abspath(args.reference), args.out,
                   args.region_size, args.ruler)
        if res is None:
            print(f"{path}: no cuboids were lifted")
            continue
        last, n = res
        print(f"{path}: {n} cuboids -> {last}_cuboids.json ({time.time() - t:.0f}s)")
    if args.show and last:
        from viewer3d.__main__ import main as viewer_main
        viewer_main(["--scene", last + "_cuboids.json", "--texture", "", "--no-showcase", "--no-surface"])


if __name__ == "__main__":
    main()
