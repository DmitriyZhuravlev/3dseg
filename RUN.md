# Running 3dseg

Tested with Python 3.13 on Linux (Mesa llvmpipe, OpenGL 4.5). Any GPU/driver with
OpenGL 3.3 core works.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Headless Linux boxes also need the Mesa EGL/GL runtime
(`apt-get install libegl1 libgl1-mesa-dri`). No `libGL.so` dev symlink is required.

## Run the 3D viewer

```bash
python -m viewer3d                         # interactive window, procedural demo cuboids
python -m viewer3d --scene cuboids.json    # cuboids exported from the pipeline (see below)
python -m viewer3d --headless --out frame.png            # offscreen render via EGL, no display
python -m viewer3d --headless --out f.png --yaw 1.2 --pitch 0.4 --zoom 0.8 --size 1920x1080
python -m viewer3d --help
```

Controls:

| Input | Action |
|-------|--------|
| Left drag | orbit |
| Right / middle drag, or Shift + left drag | pan |
| Mouse wheel | zoom |
| `A` | toggle idle auto-orbit (starts after 3 s without input) |
| `Space` | pause/resume animation |
| `R` | reframe the scene |
| `S` | save `screenshot_<time>.png` |
| `Esc` / `Q` | quit |

## Feeding real pipeline output

The segmentation scripts (`img_rec_ipm.py`, `segment_rec.py`, `surf.py`, …) already
produce `bottoms/lower_faces`, `heights`, `colors` for `draw_cubes_with_bounding_image`.
Next to that call, add either:

```python
from opengl_drawer import export_cuboids_json, show_cuboids_3d
export_cuboids_json("cuboids.json", bottoms, heights, colors)   # then: python -m viewer3d --scene cuboids.json
show_cuboids_3d(bottoms, heights, colors)                       # or open the viewer directly
```

JSON format: `{"z_up": true, "cuboids": [{"bottom": [[x,y,z] x4], "height": h, "color": [r,g,b]}]}`
(colours in 0..1 or 0..255; bottoms may be 2D, then z = 0).

## Tests

```bash
pip install pytest
python -m pytest tests          # GPU tests auto-skip if no EGL context can be created
```

## Lift the repo photos to 3D

```bash
python lift_to_viewer.py box.JPG pig.JPG --out out/   # ~2 min; writes *_cuboids.json + *_marked.png
python -m viewer3d --scene out/box_cuboids.json --texture "" --no-showcase --no-surface
```

## Interactive smoke test without a display

```bash
apt-get install xvfb
xvfb-run -a python -m viewer3d --max-seconds 5 --screenshot-on-exit window.png
```
