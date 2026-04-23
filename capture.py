"""
capture.py — RGB-D frame capture from Intel RealSense

Records aligned color + filtered depth frames to disk.
Each saved pair is:
  frame_NNNNN_color.png   — 8-bit RGB image
  frame_NNNNN_depth.png   — 16-bit depth image (values in mm, scale = 0.001 m/count)

Camera intrinsics are written once to intrinsics.json so the reconstruction
script can unproject depth pixels into 3-D points without any manual input.

Controls:
  SPACE    — toggle recording on / off
  Q / ESC  — quit
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_WIDTH    = 848    # px  — good FOV/resolution trade-off on D4xx cameras
DEFAULT_HEIGHT   = 480
DEFAULT_FPS      = 30
DEFAULT_OUTPUT   = "output/frames"
CAPTURE_INTERVAL = 0.15   # minimum seconds between saved frames while recording
WARMUP_FRAMES    = 60     # let auto-exposure settle before the user can record


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Capture RGB-D frames from RealSense")
    p.add_argument("--output",   default=DEFAULT_OUTPUT,
                   help="Output directory for frames (default: output/frames)")
    p.add_argument("--width",    type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height",   type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps",      type=int, default=DEFAULT_FPS)
    p.add_argument("--interval", type=float, default=CAPTURE_INTERVAL,
                   help="Min seconds between saved frames (default 0.15 ≈ 7 fps)")
    return p.parse_args()


# ── RealSense post-processing filters ─────────────────────────────────────────

def build_depth_filters():
    """
    Returns an ordered list of RealSense post-processing filters.

    Pipeline explanation:
      1. depth→disparity   required before spatial / temporal filters
      2. spatial           edge-preserving smoothing (reduces noise while keeping edges)
      3. temporal          smoothing across consecutive frames (reduces temporal flicker)
      4. disparity→depth   convert back to depth
      5. hole filling      fill small gaps left by the IR projector's blind spots

    Note: decimation filter is intentionally omitted so that the saved depth
    resolution matches the intrinsics (also at full resolution).
    """
    depth_to_disparity = rs.disparity_transform(True)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude,    2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal = rs.temporal_filter()

    disparity_to_depth = rs.disparity_transform(False)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)   # 1 = fill from farthest neighbour

    return [depth_to_disparity, spatial, temporal, disparity_to_depth, hole_filling]


def apply_filters(depth_frame, filters):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


# ── Intrinsics ─────────────────────────────────────────────────────────────────

def save_intrinsics(profile, output_dir: Path) -> dict:
    """
    Extract pinhole intrinsics from the active color stream and write to JSON.

    The values fx, fy (focal lengths in pixels) and cx, cy (principal point)
    define how 2-D pixel coordinates map to 3-D rays.  Open3D's
    PinholeCameraIntrinsic constructor accepts exactly these five numbers.

    Distortion coefficients are saved for reference but are NOT applied in
    reconstruction — Open3D's RGBD pipeline assumes a pinhole model.
    For the D435/D455 the distortion is small enough to ignore for this use case.
    """
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr   = stream.get_intrinsics()

    data = {
        "width":  intr.width,
        "height": intr.height,
        "fx":     intr.fx,
        "fy":     intr.fy,
        "cx":     intr.ppx,   # principal point x
        "cy":     intr.ppy,   # principal point y
        "model":  str(intr.model),
        "coeffs": list(intr.coeffs),   # distortion coefficients (for reference)
    }
    path = output_dir / "intrinsics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Start RealSense pipeline ───────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height,
                         rs.format.z16,  args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)

    print("Starting RealSense pipeline …")
    profile = pipeline.start(config)

    # Align depth pixels to the color camera's optical frame so that every
    # depth pixel maps to the correct color pixel in the saved pair.
    align = rs.align(rs.stream.color)

    # Save intrinsics once (based on the active color stream profile)
    intr = save_intrinsics(profile, out)
    print(f"  Resolution : {intr['width']} × {intr['height']}")
    print(f"  Focal length: fx={intr['fx']:.1f}  fy={intr['fy']:.1f}")
    print(f"  Principal pt: cx={intr['cx']:.1f}  cy={intr['cy']:.1f}")

    filters = build_depth_filters()

    # Allow auto-exposure / white-balance to settle
    print(f"Warming up ({WARMUP_FRAMES} frames) …")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()

    # ── Capture loop ───────────────────────────────────────────────────────────
    recording   = False
    frame_count = 0
    last_saved  = 0.0

    print("\n  SPACE   — start / stop recording")
    print("  Q / ESC — quit\n")

    try:
        while True:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)    # align depth → color frame

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Apply post-processing to the aligned depth frame
            depth_frame = apply_filters(depth_frame, filters)

            # Convert to numpy arrays
            color_bgr   = np.asanyarray(color_frame.get_data())   # uint8 BGR
            depth_mm    = np.asanyarray(depth_frame.get_data())   # uint16, millimetres

            # ── Live preview ──
            # Scale depth to [0, 255] for a colourful visualisation (alpha 0.03
            # maps ~8 m range to 255; reduce for closer max range)
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_mm, alpha=0.05), cv2.COLORMAP_JET)
            preview = np.hstack([color_bgr, depth_vis])

            label  = "  [REC]  " if recording else "  [PREVIEW]"
            colour = (0, 0, 255)  if recording else (0, 255, 0)
            cv2.putText(preview, f"{label}  saved: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.imshow("RealSense 3-D Scanner  |  SPACE=record  Q=quit", preview)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):           # Q or ESC
                break
            if key == ord(' '):
                recording = not recording
                state = "STARTED" if recording else "STOPPED"
                print(f"  Recording {state}  (saved {frame_count} frames so far)")

            # ── Save frame ────────────────────────────────────────────────────
            if recording:
                now = time.time()
                if now - last_saved >= args.interval:
                    stem = f"frame_{frame_count:05d}"

                    # Convert BGR → RGB before saving so Open3D reads correct colours
                    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(out / f"{stem}_color.png"), color_rgb)

                    # Save depth as 16-bit PNG (lossless, values in mm)
                    cv2.imwrite(str(out / f"{stem}_depth.png"), depth_mm)

                    frame_count += 1
                    last_saved  = now

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"\nCapture complete.  Saved {frame_count} frames → {out}/")

    # Write a small metadata file for the reconstruction script
    meta = {
        "frame_count": frame_count,
        "width":       args.width,
        "height":      args.height,
        "depth_scale": 0.001,    # 1 count = 1 mm = 0.001 m
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
