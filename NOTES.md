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

## 6. Refined lifting (`lift3d.py`), verified in 3D

**Why the original output did not match the objects.** `surf.py`/`cube.py`
fit a box to the silhouette in 2D only, then map it through a homography onto
*hard-coded* object dimensions. The flag `pig = True` gives 21×7×7 and is
used for the box photo too. The cuboid coordinates therefore have no metric
relation to the object.

**Method.** All in `lift3d.py`; the module docstring has details.

1. **Calibrated camera from the three vanishing points.** The VP triangle's
   orthocentre is at (1487, 844) and f = 4471 px; all three pairwise focal
   estimates agree. The camera reproduces each VP exactly and looks down at
   38°. The world frame has Z up and ground Z = 0. Units are camera heights,
   so the model is metric up to one scale factor.
2. **Mask.** Same background subtraction as `surf.py`, keeping the largest
   component.
3. **Bounding box.** A yawed 3D box on the ground, fitted so its projection
   matches the silhouette.
4. **Shape model**, chosen automatically by how well a single box explains the
   silhouette (box-fit IoU ≥ 0.975):
   - **box**: one solid cuboid. It is refined so that its visible 3D edges,
     including interior ones the silhouette can't see (the near vertical edge
     and the top-front edges), lie on image edges. This is a coarse-to-fine
     truncated chamfer.
   - **round**: a generalised cylinder along the box's long axis. In each
     slice the visual hull is a wedge between the ground and the near and far
     lines of sight. The round cross-section resting on the ground and
     touching that wedge is unique: the wedge's incircle. Each cuboid spans its
     disc (its bottom z is above 0 where the disc curves underneath). Radii are
     median-smoothed, and slivers at the ends are dropped.
   - **free**: carved columns with a reflection-symmetry prior (the earlier
     approach; still available with `--shape free`).
5. **Colours** come from the photo pixels on the object.

**3D verification, not just the projection.** A silhouette says nothing about
how deep an object goes, so reprojection IoU alone can't show 3D correctness.
I checked the 3D shape in two independent ways:

- **Real box against hand-traced 3D edges.** The six VP lines in `surf.py` are
  traced box edges. Their intersections give six visible 3D corners, which
  are independent of the mask. The traced top and bottom edges disagree by a
  few percent (dented corner, bulging lid, manual tracing), and that
  disagreement is reported as the uncertainty:

  | | L | W | H |
  |---|---|---|---|
  | Ground truth (traced) | 0.348 ± 0.025 | 0.446 ± 0.012 | 0.282 ± 0.017 |
  | Lifted model | 0.362 | 0.447 | 0.279 |

  - Corner error is 2.4% of the box diagonal.
  - **Volume IoU is 0.89.** The carved-column model without these priors
    scored 0.65, and the original pipeline's dimensions were 21×7×7.

- **Synthetic objects with known 3D shape** (`synthetic3d.py`). Each shape is
  rendered through the same camera at the pig's position, lifted, and scored
  by Monte-Carlo volume IoU:

  | Shape | Model | Volume IoU | Silhouette IoU |
  |---|---|---|---|
  | Box 0.30×0.20×0.15, yaw 20° | box | **0.93** | 0.98 |
  | Lying cylinder r = 0.06 | round | **0.85** | 0.94 |
  | Pig (ellipsoid body, head, snout, ear, legs) | round | **0.66** | 0.91 |

  Before the round model, the cylinder scored 0.38 and the pig 0.38–0.47, with
  similar silhouette IoUs. That is the evidence that a matching projection
  does not mean a matching shape.

- **Real pig.** Reprojection IoU is 0.89 (precision 0.92, recall 0.97). The
  model is a rounded body of the right length with a raised head end
  (`docs/lift3d_results.png`). There is no 3D ground truth for the toy, so its
  3D accuracy is bounded by the synthetic pig result.

**Limits (single view).**

- **Cross-section aspect is unobservable.** Width vs height of a rounded cross
  section can't be determined from one silhouette. On the synthetic pig, the
  true aspect (1.44) and the round default (1.0) both give silhouette IoU
  0.925, while their volume IoU is 0.72 vs 0.66. `--aspect` sets this prior;
  the default is 1 (round).
- **Thin protrusions are not reconstructed.** Parts thinner than a slice, such
  as the pig's ear, are missed. A "fin" heuristic added spikes in the wrong
  places, so I removed it.
- **The hidden back side** is a prior (box or round), never observed.

**Tests.** `tests/test_lift3d.py` checks that:
- the camera reproduces the VPs;
- the box photo's dimensions fall within the traced uncertainty, with volume
  IoU > 0.85;
- the synthetic volume IoUs stay above 0.9 (box), 0.8 (cylinder) and
  0.6 (pig).

## 7. 3D from video (`video3d.py`, `video_demos.py`)

Silhouettes from a single photo leave the depth of the object to assumption.
Video gives many views of the same object.

### Rigid object, fixed camera (bike.mp4)

1. **Background.** Per-pixel median over the video; this is `surf.py`'s
   `reference.JPG`, computed instead of photographed.
2. **Masks.** Median difference. Thin see-through gaps are kept (spokes,
   frame). The cast shadow is stripped: in the lowest band of the silhouette
   only long vertical runs survive.
3. **Camera.**
   - Forward vanishing point of the painted parking lines, by LSD + RANSAC.
     It is found automatically, 3 px from a hand-picked value.
   - Metric scale from the 2.74 m stall width.
   - Focal length chosen so the stalls are 5.49 m deep (standard 9 × 18 ft).
     The silhouette fit alone hardly constrains it.
4. **Poses.**
   - Ground track from the silhouette's contact point.
   - Heading from the direction of travel.
   - Lean from the turn radius.
   - Each frame's (x, y, heading, lean) is then refined against its
     silhouette, first by IoU and then by truncated chamfer distance,
     alternating with carving.
5. **Shape.** A voxel grid in the bike's own frame. A voxel is kept if it
   projects inside the silhouette in ≥ 93 % of the training frames, which
   tolerates mask errors and the moving rider. Output is a marching-cubes mesh
   with per-vertex colour (median over the frames that see the vertex unoccluded).
6. **Verification.** Even frames carve; odd frames are held out.

| Frames used for carving | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 589 |
|---|---|---|---|---|---|---|---|---|---|
| Held-out silhouette IoU | 0.41 | 0.57 | 0.65 | 0.77 | 0.79 | 0.80 | 0.81 | 0.81 | 0.81 |

- One frame (what a single photo gives) explains other views poorly.
  Eight frames already reach 0.77.
- Model size with rider: 2.04 × 0.72 × 1.50 m. This is plausible for a large
  naked bike plus rider.
- **Limit:** the camera is only ~1.3 m high and never sees the top, so the
  shape is a visual hull. It is blobby where no silhouette carves (between
  the rider's legs and the tank, under the seat).

### Non-rigid subjects (horse, person)

A trotting horse or a kicking person changes shape between frames, so
carving across frames is not valid. Instead:

- **Inflation.** The cleanest side-view silhouette is inflated: every part is
  a tube whose half-depth at distance D from the outline is
  `thickness * sqrt(D (2R - D))`, with R the local inscribed-disc radius.
- **Thickness from other views.** Thickness is fitted on the frames where the
  subject turns, using a weak-perspective view fit of the model to each mask.
  Only the upper body is compared, so moving legs do not decide it.
  Thicknesses within 0.005 of the best score count as ties, broken towards
  round.
- **Check.** On held-out turning frames the result is compared with a flat
  cut-out.

### Per video

| Video | Method | Result |
|---|---|---|
| bike.mp4 | multi-view carving (above) | held-out IoU 0.81, metric |
| Horse.mp4 | inflation, thickness fitted | see metrics; width barely observable (horse seldom turns to camera); scale assumes back height 1.55 m |
| Yeop.mp4 | inflation, thickness fitted | see metrics; scale assumes body height 1.65 m |
| Air.mp4 | single frame, round assumed | outline traced by hand + GrabCut: hazy sky has the paint's colour, crowd hides the gear |
| Monument.mp4 | single frame, round assumed | camera barely moves; the fall is not usable as views (dust, partial) |
| Bender.MP4 | single frame, round assumed | film cuts, one face |
| Por.mp4 | single frame, round assumed | one viewpoint, 320×240 |

Metrics per video: `docs/video3d/*_metrics.json`. Pictures: `*_sheet.png`,
`bike_pipeline.png`, `bike_orbit.gif`.

**Reproduce.** `python video_demos.py <name> --video <file> --out docs/video3d`.
The videos and the generated `.ply` meshes are not committed.

**Tests.** `tests/test_video3d.py` covers:
- the ground camera's VP and scale;
- carving a box that drives in a circle (volume IoU > 0.8, held-out
  IoU > 0.8), and one view failing at it;
- heading and lean signs;
- inflation of a disc giving a sphere;
- recovery of the weak-view rotation.

## 8. Novelty and prior work (assessment)

Each building block is established:
- **Calibration:** vanishing-point camera calibration and single-view metrology
  (Caprile & Torre 1990; Cipolla et al. 1999; Criminisi et al. 2000).
- **Vehicle 3D boxes from vanishing points** in fixed traffic cameras (Dubská
  et al. 2014; Sochor et al., BoxCars). This is the closest prior work to the
  original `surf.py`/`cube.py` idea.
- **Visual hull** (Laurentini 1994; Szeliski 1993).
- **Shape from silhouette across time** for a moving rigid object (Cheung,
  Baker & Kanade 2003).
- **Silhouette inflation** (Teddy 1999; Monster Mash 2020).

A possible narrow contribution, for a workshop paper or technical report:
training-free metric shape of road users from one uncalibrated fixed camera.
It would rest on:
- self-calibration from painted ground lines and standard stall sizes;
- kinematic priors for two-wheelers (lean from the turn) inside
  silhouette-across-time carving;
- robustness measures: shadow stripping, a carving consensus threshold and
  chamfer pose refinement;
- a frames-vs-accuracy study.

Missing before writing:
- ground truth (synthetic CAD scenes, or vehicles of known dimensions);
- more videos;
- baselines: VP-box, single view, carving without pose refinement, learned
  methods (DUSt3R, TripoSR);
- ablations;
- a literature search confirming the lean-prior angle.

## 9. The original recursive segment method: drift fix and Make3D comparison

`segment3d.py` keeps the structure of `surf.py`/`cube.py`:
- a vanishing-point tangent box per superpixel;
- seeds at the object's six extreme points;
- recursive propagation between neighbouring superpixels.

The geometry is replaced with the calibrated camera from `lift3d.py`.

**Findings.**
- **The original's whole-object tangent box is metric once the camera is
  calibrated.** On `box.JPG` it gives 0.320 × 0.461 × 0.304 m against the
  traced 0.348 × 0.446 × 0.282 m, with zero reprojection error.
- **Recursive boxes.** With the calibrated camera, every superpixel gets a box
  (104/104 on `box.JPG`, against 6 with the original code). But the boxes
  drift in 3D. A superpixel is a patch of surface, while its box depth comes
  from its 2D extent, and the error accumulates along the propagation.
- **Drift fix (`patch_reconstruct`).** Each superpixel is a planar patch: its
  pixels back-projected onto an axis-aligned plane, so no depth is guessed.
  - Seeds sit on faces of the main box. That box is `lift3d`'s edge-refined
    fit, which matters: anchoring on the silhouette-only box gives 2.2 %
    median error, the edge-refined box 1.3 %.
  - A neighbour goes on the same plane or folds through the shared edge.
    Colour similarity and vanishing-point edge votes pick between the two
    (the orientation-map cue of Lee, Hebert & Kanade 2009).
  - A connectivity refinement then minimises 3D gaps across boundaries.
- **Make3D-style baseline (`make3d_planes`).** A global least-squares solve of
  one free plane per superpixel, with Make3D's connectivity and coplanarity
  terms (Saxena et al. 2009). The original Make3D code and its learned depth
  model are not reachable from this environment, and the model was trained on
  outdoor scenes. Its learned per-pixel term is therefore replaced by the same
  seed anchors the recursive method uses.

**Per-pixel depth error against 3D ground truth** (`python segment3d_eval.py`):

| Method | box.JPG (traced box): median / ≤3 % | synthetic pig: median / ≤3 % |
|---|---|---|
| Recursive boxes (original method, calibrated; segments may also attach to main-box planes) | 3.6 % / 46 % | 3.0 % / 49 % |
| **Recursive planar patches (drift fix)** | **1.3 % / 84 %** | 4.0 % / 40 % |
| — without the orientation cue | 1.2 % / 86 % | 4.0 % / 40 % |
| — without connectivity refinement | 1.3 % / 82 % | 5.0 % / 29 % |
| — with the silhouette-only main box | 2.2 % / 65 % | 4.8 % / 29 % |
| Make3D-style MRF (same anchors) | 1.8 % / 68 % | 5.1 % / 31 % |
| `lift3d` (box / round model), reference | 0.5 % / 100 % | 2.0 % / 68 % |

**Reading the results.**
- **Flat-faced objects.** On `box.JPG` the drift fix cuts the original
  recursive method's error by about 3× and beats the Make3D-style global
  solve. The edge-refined main box is the largest single gain; connectivity
  refinement removes gross outliers (mean error 3.4 % → 2.6 %).
- **Orientation cue.** It does not help on this photo (printed graphics give
  misleading edges) and has no effect on the untextured pig.
- **Curved objects.** Axis-aligned flat patches suit flat faces, not curves.
  On the pig the patches are worse than the original's solid segment boxes,
  and the Make3D-style solve, with few anchors, is worst.
- **Whole-object models.** For both objects, `lift3d`'s whole-object models
  are more accurate than any per-segment method. Per-segment methods only pay
  off when parts sit at different depths, which these test objects lack.

**Possible next steps.**
- Free (not axis-aligned) patch planes for curved objects.
- Combine the round model as anchors with segment-level detail.
- Test on multi-part objects.
