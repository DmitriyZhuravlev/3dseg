"""Regenerate the lifting demo images in docs/:  PYTHONPATH=. python docs/make_demo.py"""
import math, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import lift3d
from viewer3d.app import run_headless
from viewer3d.scenes import build_cuboid_scene

F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
TW, TH = 600, 380
CROP = {"box": (0, 1760, 250, 2850), "pig": (560, 1640, 600, 2400)}

def tile(bgr_or_pil, caption):
    im = bgr_or_pil if isinstance(bgr_or_pil, Image.Image) else Image.fromarray(cv2.cvtColor(bgr_or_pil, cv2.COLOR_BGR2RGB))
    im = im.convert("RGB"); im.thumbnail((TW, TH))
    out = Image.new("RGB", (TW, TH), (24, 26, 32)); out.paste(im, ((TW - im.width) // 2, (TH - im.height) // 2))
    d = ImageDraw.Draw(out); d.rectangle([0, 0, d.textlength(caption, font=F) + 16, 30], fill=(0, 0, 0)); d.text((8, 4), caption, fill="white", font=F)
    return out

def crop(n, img): y0, y1, x0, x1 = CROP[n]; return img[y0:y1, x0:x1]

def view(data, yaw, pitch=0.42, zoom=1.0, size=(TW, TH)):
    scene = build_cuboid_scene(data, ground_texture=None, with_surface=False, with_showcase=False)
    img, _ = run_headless(scene, None, size, frames=1, camera={"yaw": yaw, "pitch": pitch, "zoom": zoom, "auto_rotate": 0})
    return img

def grid(tiles, cols, title):
    rows = math.ceil(len(tiles) / cols); pad = 8
    W = cols * TW + (cols + 1) * pad; H = rows * TH + (rows + 1) * pad + 44
    sheet = Image.new("RGB", (W, H), (40, 42, 50)); ImageDraw.Draw(sheet).text((pad + 4, 10), title, fill="white", font=F)
    for k, t in enumerate(tiles):
        sheet.paste(t, (pad + (k % cols) * (TW + pad), 44 + pad + (k // cols) * (TH + pad)))
    return sheet

cam = lift3d.default_camera()
vps = lift3d.vanishing_points()
results = {n: lift3d.lift(f"{n}.JPG") for n in ("box", "pig")}

# ---------- 1. pipeline, step by step ----------
for n in ("box", "pig"):
    r = results[n]; img = r["image"]; mask = r["mask"]
    t = [tile(crop(n, img), "1  photo")]
    m = img.copy(); m[~mask] = (m[~mask] * 0.25).astype(np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(m, cs, -1, (0, 0, 255), 6)
    t.append(tile(crop(n, m), "2  object mask (vs reference.JPG)"))
    # camera: VP lines + fitted box
    c = img.copy()
    for name, col in (("vertical", (255, 160, 0)), ("left", (0, 200, 255)), ("right", (255, 0, 200))):
        for l in lift3d.VP_LINES[name]:
            a, b = np.array(l[0], float), np.array(l[1], float); d = b - a
            cv2.line(c, tuple((a - 3 * d).astype(int)), tuple((b + 3 * d).astype(int)), col, 4, cv2.LINE_AA)
    box, _ = lift3d.fit_bounding_box(cam, mask)
    uv, _ = cam.project(lift3d.box_corners(*box))
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(c, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (60, 255, 60), 7, cv2.LINE_AA)
    t.append(tile(crop(n, c), "3  VP-calibrated camera + 3D box fit"))
    if r["shape"] == "round":
        # slices: reproject each disc outline
        (bt, tp), frame, discs = lift3d.round_model(cam, mask, box)
        cx, cy, ln, wd, yaw = frame; co, si = math.cos(yaw), math.sin(yaw); rot = np.array([[co, si], [-si, co]])
        s = img.copy()
        for i, vc, rr in discs:
            u = ((i + 0.5) / tp.shape[0] - 0.5) * ln; th = np.linspace(0, 2 * np.pi, 60)
            vv = vc + rr * np.cos(th); zz = rr + rr * np.sin(th)
            xy = np.c_[np.full(60, u), vv] @ rot + [cx, cy]
            p, _ = cam.project(np.c_[xy, zz]); cv2.polylines(s, [np.round(p).astype(np.int32)], True, (60, 255, 255), 3, cv2.LINE_AA)
        t.append(tile(crop(n, s), "4  round slice per section (incircle)"))
    else:
        rb = r["box"]; uv, _ = cam.project(lift3d.box_corners(*rb)); s = img.copy()
        corners = lift3d.box_corners(*rb)
        for a, b in lift3d.visible_box_edges(cam, corners):
            cv2.line(s, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (60, 255, 255), 7, cv2.LINE_AA)
        t.append(tile(crop(n, s), "4  box refined on its visible 3D edges"))
    rp = lift3d.overlay(r)
    t.append(tile(crop(n, rp), f"5  cuboids reprojected  IoU {r['scores']['iou']:.2f}"))
    data = {"cuboids": r["cuboids"], "z_up": True}
    t.append(tile(view(data, 2.3), "6  3D model (opposite side)"))
    grid(t, 3, f"{n}.JPG  ->  3D cuboids   ({r['shape']} model, {len(r['cuboids'])} cuboids)").save(f"docs/demo_pipeline_{n}.png")

# ---------- 2. before / after ----------
t = []
for n in ("box", "pig"):
    old = json.load(open(f"docs/surf_{n}_cuboids.json"))
    t.append(tile(crop(n, results[n]["image"]), f"{n}.JPG"))
    t.append(tile(view(old, 0.8), "before: surf.py (fixed 21x7x7 sizes)"))
    t.append(tile(view({"cuboids": results[n]["cuboids"], "z_up": True}, 0.8), "after: lift3d (metric, calibrated)"))
grid(t, 3, "Before / after: the same photos lifted by the original pipeline and by lift3d").save("docs/demo_before_after.png")

# ---------- 3. orbit GIF ----------
frames = []
for k in range(40):
    yaw = 0.6 + k * 2 * math.pi / 40
    a = view({"cuboids": results["box"]["cuboids"], "z_up": True}, yaw, size=(480, 320))
    b = view({"cuboids": results["pig"]["cuboids"], "z_up": True}, yaw, size=(480, 320))
    f = Image.new("RGB", (960, 320)); f.paste(a, (0, 0)); f.paste(b, (480, 0))
    frames.append(f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
frames[0].save("docs/demo_lift_orbit.gif", save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
print("done")
