# 3dseg

Experimental Python code for 3D scene understanding from monocular images and
video: ground-plane calibration, superpixel/graph-based segmentation, surface
normals, bird's-eye-view (inverse perspective mapping) reconstruction, and
fitting/rendering of 3D boxes.

> Research code: scripts are standalone experiments, several are near-duplicate
> variants (`*_bk`, `*_cop`, `*_v2`, `*_extrem`), and some have hard-coded paths.

## Requirements

Python 3 with: `numpy`, `scipy`, `opencv-python`, `scikit-image`,
`scikit-learn`, `shapely`, `matplotlib`, `flowiz`, and PyOpenGL (used by
`opengl_drawer.py`).

```bash
pip install numpy scipy opencv-python scikit-image scikit-learn shapely matplotlib flowiz PyOpenGL
```

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Video demo: pick ROI, build perspective matrix, background subtraction, draw cubes (`output_with_cubes.mp4`) |
| `Calib_GrndPlane.py` | Ground-plane calibration |
| `lifting.py`, `segment_3d.py` | Lifting 2D segments to 3D |
| `segment*.py`, `graph.py`, `seg.py` | Superpixel / graph-based segmentation (plain, BEV, DFS, RAFT variants) |
| `img_rec*.py`, `img_iter*.py` | Image reconstruction / iterative IPM experiments |
| `normals.py`, `ransac.py`, `shape.py` | Surface normals, RANSAC plane fitting, shape helpers |
| `cube.py`, `render.py`, `opengl_drawer.py` | Cube fitting and rendering |
| `surf.py`, `surf/` | Surface reconstruction experiments (with its own copies of `cube.py`, `lifting.py`, `main2.py`) |
| `*.png`, `*.JPG`, `*.bmp` | Sample/reference images and result figures |

## Usage

Most scripts are run directly, e.g.:

```bash
python main.py
```

`main.py` reads a video from a hard-coded path
(`/home/dzhura/ComputerVision/data/bike.mp4`); edit `input_video_path` to point
at your own file. Other scripts likewise expect local image/video paths to be
set near the top of the file.
