# RealSense 3-D Scanner

A minimal, two-script pipeline for scanning small objects (10–50 cm) with an
Intel RealSense depth camera.  No ROS, no special hardware rig — just Python,
Open3D, and a steady hand.

```
realsense_3d_scanner/
├── capture.py        # Stage 1 — record RGB-D frames from the camera
├── reconstruct.py    # Stage 2 — register frames and build a 3-D model
├── requirements.txt
├── README.md
└── output/
    ├── frames/       # created by capture.py  (PNG pairs + intrinsics.json)
    └── models/       # created by reconstruct.py  (.ply / .obj)
```

---

## Requirements

| Dependency      | Version tested | Notes                                  |
|-----------------|---------------|----------------------------------------|
| Python          | 3.9 +         |                                        |
| pyrealsense2    | 2.50 +        | RealSense SDK Python bindings          |
| open3d          | 0.17 +        | 3-D geometry and reconstruction        |
| numpy           | 1.21 +        |                                        |
| opencv-python   | 4.5 +         | display and image I/O                  |

**Tested cameras:** D435, D435i, D455.  Any D4xx series should work.

---

## Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **macOS / ARM note:** pre-built `pyrealsense2` wheels are only available for
> x86-64.  On Apple Silicon, either use Rosetta (`arch -x86_64 python …`) or
> build librealsense from source.

---

## Stage 1 — Capture

Plug in your RealSense camera, place the object on a plain surface, and run:

```bash
python capture.py
```

A live preview window opens (color on the left, depth heatmap on the right).

| Key       | Action                        |
|-----------|-------------------------------|
| `SPACE`   | Start / stop recording        |
| `Q / ESC` | Quit                          |

**Recommended capture procedure:**

1. Position the camera ~30–50 cm from the object.
2. Press `SPACE` to start recording.
3. Move the camera *slowly* in a continuous arc around the object — aim for
   about one full orbit in 20–30 seconds.
4. Try to keep the object in the centre of the frame at all times.
5. Press `SPACE` (or `Q`) to stop.  ~80–150 saved frames is usually plenty.

Saved files (in `output/frames/`):

```
frame_00000_color.png    # 8-bit RGB
frame_00000_depth.png    # 16-bit depth (values in mm)
frame_00001_color.png
frame_00001_depth.png
…
intrinsics.json          # camera parameters — required by reconstruct.py
meta.json                # frame count, resolution, depth scale
```

### Options

```
--output DIR      Frame output directory  (default: output/frames)
--width  N        Stream width in pixels  (default: 848)
--height N        Stream height in pixels (default: 480)
--fps    N        Stream frame rate       (default: 30)
--interval SECS   Min seconds between saved frames (default: 0.15)
```

---

## Stage 2 — Reconstruct

```bash
python reconstruct.py
```

This will:

1. Load every 3rd frame pair (adjustable with `--step`).
2. Register consecutive point clouds with point-to-plane ICP.
3. Merge all clouds into world space and run Poisson surface reconstruction.
4. Open an interactive viewer, then save outputs to `output/models/`.

Output files:

```
output/models/
├── pointcloud.ply   # fused, cleaned point cloud
├── mesh.ply         # Poisson surface mesh
└── mesh.obj         # same mesh, OBJ format (importable in Blender, MeshLab, …)
```

### Options

```
--input  DIR      Frame directory  (default: output/frames)
--output DIR      Model directory  (default: output/models)
--step   N        Use every Nth frame  (default: 3)
--voxel  M        Voxel size in metres (default: 0.002 = 2 mm)
--tsdf            Use TSDF volume integration — cleaner mesh, more memory
--no-mesh         Skip mesh; save point cloud only
```

**Tuning tips:**

| Situation | Suggestion |
|-----------|-----------|
| Too many gaps in the mesh | Reduce `--step` (use more frames) |
| Reconstruction is slow | Increase `--step` or `--voxel` |
| Drift/misalignment visible | Capture more slowly; re-run with `--step 1` |
| Thin floating artifacts | Already trimmed by Poisson density filter; try `--tsdf` for cleaner result |
| Object too small (< 5 cm) | Move closer (20–25 cm); try `--voxel 0.001` |

---

## Camera assumptions and calibration notes

### What the scripts assume

- **Pinhole camera model** — distortion coefficients from the camera are saved
  to `intrinsics.json` for reference but are *not* applied.  RealSense D-series
  distortion is small (< 1 %) at the distances used here, so the error is
  sub-millimetre for 10–50 cm objects.

- **Depth units = 1 mm / count** — the default for all RealSense D-series
  cameras.  If you changed the depth unit in firmware, update `DEPTH_SCALE` in
  `reconstruct.py`.

- **Depth range** — frames are truncated at 1 m (`DEPTH_MAX`).  This removes
  the table and background and keeps the ICP focused on the object.

- **Static object, moving camera** — ICP registers consecutive frames assuming
  only the camera moves.  If the object shifts, reconstruction will fail.

### Accuracy expectations

| Metric                    | Typical result (D435, 30–50 cm) |
|---------------------------|----------------------------------|
| Depth noise (1-sigma)     | ~1–2 mm                          |
| Reconstruction accuracy   | ~2–5 mm                          |
| Mesh resolution           | ~2–4 mm (controlled by `--voxel`)|

For better accuracy: use a tripod, capture slower, reduce `--voxel` to 0.001.

### Not covered (out of scope for this MVP)

- **Global loop closure** — long sequences may drift.  Open3D's `GlobalRegistration`
  or ORB-SLAM3 can add loop closure if needed.
- **Undistortion** — apply `cv2.undistort` to color frames using `coeffs` from
  `intrinsics.json` before reconstruction if your lens has high distortion.
- **Texture mapping** — the mesh is vertex-coloured.  UV unwrap in MeshLab or
  Blender for a full texture map.

---

## Viewing outputs

Any of these tools can open `.ply` / `.obj` files:

- **MeshLab** (free, recommended) — `meshlab output/models/mesh.ply`
- **Blender** — File → Import → Stanford (.ply)
- **Open3D viewer** — already opens automatically after reconstruction
- **CloudCompare** — good for point cloud analysis
