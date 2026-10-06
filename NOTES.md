# NOTES: repo analysis, decisions, changes and verification

## 1. Repo analysis (before changes)

**Stack.** Python 3 research code: OpenCV, numpy/scipy, scikit-image/learn,
shapely and matplotlib. There was no `pyproject.toml`, no `requirements.txt`,
no build system and no tests. There are no `src/`, `lib/`, `app/`, `shaders/`
or `assets/` directories.

**Purpose.** This is monocular 3D scene segmentation: superpixel/graph
segmentation of a frame, then ground-plane calibration and inverse perspective
mapping (BEV), vanishing points, and lifting each segment to a **cuboid** given
as a bottom quad, a height and a colour (`lifting.py`, `cube.py`,
`img_rec_ipm.py`, `segment_rec.py`, `surf.py`, …).

**What "3D" existed.** Nothing was GPU-rendered. `opengl_drawer.py` uses
**matplotlib** `Poly3DCollection` / `plot_surface`, despite its name, and
contains no OpenGL. There was no real-time camera, lighting, depth buffer or
render loop.

**What was broken or weak:**

- **Dependencies.** There was no dependency list. `shape.py` and `ransac.py`
  failed to import until scikit-learn was installed. Several scripts need
  `cv2.ximgproc`, which comes from opencv-*contrib*.
- **Byte-compile.** Every `.py` file byte-compiles (`py_compile`). There are no
  syntax errors.
- **Hard-coded data and GUI scripts.** Most scripts are top-level programs with
  absolute data paths (e.g. `/home/dzhura/ComputerVision/data/...`).
  `normals.py` runs its whole pipeline at import time and crashes without that
  file. `main.py` needs a video plus an interactive `cv2.imshow` ROI click. Its
  segmentation is a placeholder: `labels = np.zeros_like(...)` and fixed
  `box_corners`.
- **TODOs.** `# TODO left, right ?` and `# TODO check orientation` in
  `cube.py` and the copies of that code in `img_*`.
- **Duplication.** Many near-duplicate 2–3k-line variants exist (`*_bk`,
  `*_cop`, `*_v2`, `*_extrem`, `surf/` copies of `cube.py`, `lifting.py`,
  `graph.py`, `main.py.bk*`).
- **Tracked junk.** `__pycache__/` was tracked in git.

**Missing for "real 3D".** A GPU renderer with perspective projection, a
camera with controls, depth testing, lighting, shadows, materials and
textures, anti-aliasing, and a timed render loop.

## 2. Decisions

- **Stay on Python, add a small OpenGL layer rather than rewrite.** The
  research code is the asset; the viewer *consumes* its output (cuboids) via
  `opengl_drawer.export_cuboids_json` / `show_cuboids_3d`. Nothing in the
  existing pipeline was rewritten.
- **moderngl + glfw.** moderngl is a thin, modern OpenGL 3.3+ wrapper (VAOs,
  FBOs, MSAA renderbuffers, depth textures) and can also create a headless EGL
  context, which made automated verification possible. glfw provides the
  window and input. Pillow loads textures and saves screenshots. Each is one
  small wheel with no heavy framework. I rejected PyOpenGL immediate mode (the
  legacy fixed pipeline) and pygame/three.js-style rewrites.
- **Own math and scene-graph module (`viewer3d/math3d.py`, `scene.py`)**
  instead of pyrr/glm. It is ~250 lines, unit-tested, and has no extra
  dependency.
- **Coordinates.** The pipeline is Z-up in BEV/pixel units. The viewer is
  Y-up. The conversion `(x, y, z) -> (x, z, -y)` is a rotation, not a mirror,
  so handedness is kept. A `world` node recentres and scales the data to about
  10 units, so lighting, shadows and camera are independent of input scale.
- **Demo data.** No saved pipeline output exists in the repo, and the scripts
  need private datasets. When no `--scene` is given, the viewer generates a
  deterministic procedural cuboid field (one cuboid per "superpixel"). The
  ground texture is the repo's `Background.bmp`.
- **Interpolated top surface.** This is the GPU equivalent of
  `draw_interpolated_surface`. It is drawn as a **wireframe** so it doesn't
  hide the cuboids. Pass `--no-surface` to hide it.
- **Research scripts left as is.** I did not delete or merge the duplicated
  scripts or remove hard-coded paths. They are the author's experiments, and
  deleting them is not reversible from their point of view. This file
  documents them instead.

## 3. Changes, one commit each

1. **`requirements.txt`.** All imports actually used, plus the viewer deps.
   Stopped tracking `__pycache__` (already in `.gitignore`).
2. **`viewer3d/math3d.py`, `scene.py`, `geometry.py`.** Matrices, a quaternion
   class, a scene graph with hierarchical transforms, a perspective camera and
   orbit controls (rotate/zoom/pan with clamping, auto-adjusted near/far).
   Geometry:
   - box, sphere, torus, plane
   - `cuboid_from_bottom`, which orients faces outward whatever the input
     winding
   - `heightfield`, using scipy `griddata` with smooth normals
   - a minimal OBJ loader
   - normals and UVs on every mesh
3. **`viewer3d/renderer.py`.** GLSL 330 Blinn-Phong with:
   - a directional sun plus hemisphere ambient light
   - shadow mapping: a 2048² depth texture, hardware compare, 3×3 PCF and
     slope-scaled bias
   - sRGB-correct colours and textures, with mipmaps and anisotropic filtering
   - 4× MSAA renderbuffers resolved into the target
   - depth test, back-face culling, double-sided and wireframe materials
4. **`viewer3d/app.py`, `scenes.py`, `__main__.py`.**
   - The `Viewer` has separate `update(dt)` and `render(fbo)` steps.
   - The glfw loop clamps dt and uses the framebuffer-size callback, so the
     framebuffer is sized in physical pixels (correct on HiDPI) while mouse
     deltas are normalised by window height.
   - Idle auto-orbit, hotkeys, an FPS/triangle count in the title, and
     screenshots rendered to an offscreen FBO.
   - Headless EGL mode.
   - Falls back to `libGL.so.1` when the unversioned `libGL.so` is missing.
5. **`tests/`.** 17 pytest tests:
   - matrix and quaternion maths, hierarchy composition and orbit clamping
   - outward CCW winding and unit normals for every closed primitive
   - heightfield and OBJ loading
   - GPU tests: the depth test is independent of draw order, there is no black
     screen, the camera reacts to orbit input, and a box's shadow darkens the
     ground on the correct side
   - JSON export round-trip from `opengl_drawer`
6. **`opengl_drawer.py`.** Added `export_cuboids_json()` and
   `show_cuboids_3d()`, which take the same arguments as
   `draw_cubes_with_bounding_image`.
7. **Docs.** `README.md` (viewer section), `RUN.md` and this file.

Dependencies added: `moderngl` (OpenGL 3.3 wrapper with headless EGL),
`glfw` (window/input), `Pillow` (textures/screenshots). Everything else in
`requirements.txt` was already imported by the existing code.

## 4. Verification

The environment was a Linux container with no GPU: Mesa llvmpipe, OpenGL 4.5,
4× MSAA available.

- `python -m pytest tests`: **17 passed**. `pyflakes viewer3d tests`: clean.
- `python -m viewer3d --headless --out docs/viewer.png`: 62 meshes and about
  7.5k triangles per frame (the count depends on the camera angle used).
- **What the image shows.** The textured ground plane (`Background.bmp`) is
  in correct perspective. About 60 coloured cuboids are lit from the upper
  left, and each has three visibly different face shades. Cuboids cast shadows
  onto the ground and onto each other. Edges are anti-aliased, with no
  z-fighting or shadow acne on the ground. The wireframe surface floats above
  the cuboid tops. The white sphere shows a specular highlight, and the
  tilted orange torus and blue cube orbit with it.
- **Shadow check without texture.** With the texture disabled (`--texture ""`),
  every box has a clearly offset shadow on the grey ground.
- **Real window under Xvfb.** `xvfb-run python -m viewer3d --max-seconds 6
  --screenshot-on-exit` opened a GLFW OpenGL 3.3 core window and rendered the
  same scene. The animation advanced between frames.
- **Real mouse input with xdotool.** I sent a left-drag, 4 wheel clicks and a
  right-drag to the Xvfb window. The camera orbited, zoomed in and panned. The
  mean absolute pixel difference against a run with no input was 36/255.
- **Not verifiable here.** I could not test a physical HiDPI monitor. The code
  uses `glfw.get_framebuffer_size` for rendering and window-size coordinates
  for input, which is the standard DPR-correct setup.
- **Bug found and fixed during verification.** Reading the window back buffer
  after `swap_buffers` returned garbage. Screenshots now render into their own
  FBO.

## 5. Real 2D → 3D lifting on the repo photos

`reference.JPG` (empty scene) and `box.JPG` / `pig.JPG` (3036×1760) match the
vanishing-point lines hard-coded in `surf.py`, so they are the intended input.
`lift_to_viewer.py` runs `surf.py`'s `process_images` headlessly on them and exports the
cuboids for the viewer (`docs/real_lifting.png`, `docs/box_cuboids.json`, `docs/pig_cuboids.json`).

- **`img_rec_ipm.py` is out of date.** It calls `graph.find_neighbors(labels)`,
  but `find_neighbors` now also needs the contour mask, as `surf.py` passes it.
  `surf.py` is the current pipeline.
- **Fixed: `graph.find_neighbors` never finished on full-size photos.** It
  looped over every contour pixel for every label pair. It is now vectorised,
  and `tests/test_graph.py` checks the results are identical.
- **Speed-up, in the script only.** `lift_to_viewer.py` memoises
  `get_projected_box` because the recursive growth recomputes the same masks.
  This takes `box.JPG` from >30 min to ~100 s and `pig.JPG` to ~30 s, with the
  same output.
- **Result: the lifting is only partial.** The object is segmented correctly,
  but out of ~115 superpixels only **6 (box)** and **8 (pig)** cuboids come out.
  About 350 "Error computing" messages show the 3D face is `None` for most
  segments. The resulting cuboids are small blocks, all in one colour, and do
  not form a recognisable box or pig shape. The lifting maths (`lifting.py`,
  `cube.py`, `process_segment` in `surf.py`) is where the work is needed;
  the viewer shows what the pipeline produces faithfully.
