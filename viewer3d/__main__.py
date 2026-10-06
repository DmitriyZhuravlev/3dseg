"""python -m viewer3d [--scene cuboids.json] [--headless --out frame.png]"""
import argparse
import sys

from .scenes import DEFAULT_GROUND_TEXTURE, build_cuboid_scene, demo_cuboids, load_cuboids_json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="viewer3d", description=__doc__)
    ap.add_argument("--scene", help="cuboid JSON written by opengl_drawer.export_cuboids_json "
                                    "(default: procedural demo scene)")
    ap.add_argument("--texture", default=DEFAULT_GROUND_TEXTURE, help="ground texture image")
    ap.add_argument("--no-surface", action="store_true", help="hide the interpolated top surface")
    ap.add_argument("--no-showcase", action="store_true", help="hide the sphere/torus primitives")
    ap.add_argument("--size", default="1280x720", help="window / image size, WxH")
    ap.add_argument("--msaa", type=int, default=4, help="MSAA samples (1 disables)")
    ap.add_argument("--headless", action="store_true", help="render offscreen (EGL) and save --out")
    ap.add_argument("--out", default="frame.png", help="headless output image")
    ap.add_argument("--frames", type=int, default=90, help="headless: frames to simulate before saving")
    ap.add_argument("--yaw", type=float, help="headless: camera yaw (rad)")
    ap.add_argument("--pitch", type=float, help="headless: camera pitch (rad)")
    ap.add_argument("--zoom", type=float, default=1.0, help="headless: distance multiplier")
    ap.add_argument("--max-seconds", type=float, help="windowed: quit after N seconds (smoke tests)")
    ap.add_argument("--screenshot-on-exit", help="windowed: save the last frame to this path")
    args = ap.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))
    data = load_cuboids_json(args.scene) if args.scene else demo_cuboids()
    scene = build_cuboid_scene(data, ground_texture=args.texture,
                               with_surface=not args.no_surface, with_showcase=not args.no_showcase)

    if args.headless:
        from .app import run_headless
        cam = {"zoom": args.zoom}
        if args.yaw is not None:
            cam["yaw"] = args.yaw
        if args.pitch is not None:
            cam["pitch"] = args.pitch
        _, info = run_headless(scene, args.out, (w, h), frames=args.frames, camera=cam, samples=args.msaa)
        print(f"wrote {args.out}: {info}")
    else:
        from .app import run_window
        run_window(scene, (w, h), samples=args.msaa, max_seconds=args.max_seconds,
                   screenshot_on_exit=args.screenshot_on_exit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
