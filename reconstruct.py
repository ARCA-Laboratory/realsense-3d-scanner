"""
reconstruct.py — 3-D reconstruction from captured RGB-D frames

Pipeline:
  1. Load captured color + depth PNG pairs and camera intrinsics.
  2. Create an Open3D colored point cloud from each RGBD frame.
  3. Register consecutive point clouds with point-to-plane ICP, building a
     chain of camera poses relative to the first frame (world origin).
  4a. (default) Transform all point clouds into world space, merge, downsample,
      remove outliers, then run Poisson surface reconstruction for a mesh.
  4b. (--tsdf)  Feed each frame + its pose into a TSDF volume; extract a mesh
      directly from the implicit surface — cleaner but slower.
  5. Save pointcloud.ply, mesh.ply, and mesh.obj to the output directory.
  6. Open an interactive 3-D viewer.

Usage examples:
  python reconstruct.py
  python reconstruct.py --step 2 --voxel 0.003
  python reconstruct.py --tsdf
  python reconstruct.py --no-mesh --output output/models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ── Constants ──────────────────────────────────────────────────────────────────
DEPTH_SCALE = 0.001    # 1 count in saved PNG = 1 mm = 0.001 m
DEPTH_MAX   = 1.0      # discard depth beyond 1 m (ignore background)
VOXEL_SIZE  = 0.002    # 2 mm voxels — good balance for 10–50 cm objects


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct a 3-D model from RGB-D frames")
    p.add_argument("--input",   default="output/frames",
                   help="Directory containing captured frames (default: output/frames)")
    p.add_argument("--output",  default="output/models",
                   help="Directory for output files (default: output/models)")
    p.add_argument("--step",    type=int, default=3,
                   help="Use every Nth frame — reduces redundancy (default: 3)")
    p.add_argument("--voxel",   type=float, default=VOXEL_SIZE,
                   help="Voxel size for downsampling in metres (default: 0.002)")
    p.add_argument("--tsdf",    action="store_true",
                   help="Use TSDF volume integration for mesh (cleaner, slower)")
    p.add_argument("--no-mesh", action="store_true",
                   help="Skip mesh reconstruction (point cloud only)")
    return p.parse_args()


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_intrinsics(frames_dir: Path) -> o3d.camera.PinholeCameraIntrinsic:
    """Load the camera intrinsics saved by capture.py."""
    path = frames_dir / "intrinsics.json"
    if not path.exists():
        sys.exit(f"ERROR: intrinsics.json not found in {frames_dir}\n"
                 "       Run capture.py first.")
    with open(path) as f:
        d = json.load(f)
    return o3d.camera.PinholeCameraIntrinsic(
        d["width"], d["height"],
        d["fx"], d["fy"],
        d["cx"], d["cy"],
    )


def load_rgbd(color_path: Path, depth_path: Path) -> o3d.geometry.RGBDImage:
    """
    Load a color + depth PNG pair into an Open3D RGBDImage.

    depth_scale in Open3D's API means: depth_in_metres = raw_value / depth_scale
    Our PNGs store mm → raw_value / 1000 = metres → depth_scale = 1000.
    """
    color = o3d.io.read_image(str(color_path))
    depth = o3d.io.read_image(str(depth_path))   # 16-bit PNG in mm
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1.0 / DEPTH_SCALE,   # = 1000 — converts mm → metres
        depth_trunc=DEPTH_MAX,
        convert_rgb_to_intensity=False,
    )


def rgbd_to_pcd(rgbd: o3d.geometry.RGBDImage,
                intrinsic: o3d.camera.PinholeCameraIntrinsic
                ) -> o3d.geometry.PointCloud:
    """Unproject every valid depth pixel into a coloured 3-D point."""
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


# ── Registration ───────────────────────────────────────────────────────────────

def icp_register(source: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud,
                 voxel_size: float,
                 init: np.ndarray = None) -> tuple[np.ndarray, float]:
    """
    Point-to-plane ICP: find the rigid transform T such that
    T @ source_points  ≈  target_points.

    Returns (4×4 transformation matrix, fitness score ∈ [0, 1]).
    Fitness is the fraction of source points that found a correspondence;
    values below ~0.3 indicate poor overlap or divergence.
    """
    if init is None:
        init = np.eye(4)

    # Normals are required for point-to-plane ICP
    radius = voxel_size * 2
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel_size * 1.5,
        init=init,
        estimation_method=o3d.pipelines.registration
                           .TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration
                  .ICPConvergenceCriteria(max_iteration=50),
    )
    return result.transformation, result.fitness


# ── Reconstruction modes ───────────────────────────────────────────────────────

def fuse_point_clouds(frames_data: list, transforms: list[np.ndarray],
                      voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Transform every point cloud into world space (frame 0 as origin),
    concatenate them all, then voxel-downsample to remove duplicate points.
    """
    merged = o3d.geometry.PointCloud()
    for (_, pcd), T in zip(frames_data, transforms):
        # Copy so we don't mutate the stored cloud
        pcd_world = o3d.geometry.PointCloud(pcd)
        pcd_world.transform(T)
        merged += pcd_world
    return merged.voxel_down_sample(voxel_size)


def fuse_tsdf(frames_data: list, transforms: list[np.ndarray],
              intrinsic: o3d.camera.PinholeCameraIntrinsic,
              voxel_size: float) -> o3d.geometry.TriangleMesh:
    """
    Integrate all frames into a Truncated Signed Distance Function (TSDF) volume
    and extract the zero-level-set as a triangle mesh.

    TSDF integration averages depth observations in a volumetric grid, producing
    smoother surfaces than merging raw point clouds.  The trade-off is higher
    memory usage and the need for accurate camera poses.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size / 2,       # finer than the point-cloud voxel size
        sdf_trunc=voxel_size * 5,          # truncation band around surface
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print("  Integrating frames into TSDF volume …")
    for i, ((rgbd, _), T) in enumerate(zip(frames_data, transforms)):
        # TSDF integrate() takes the world-to-camera extrinsic (view matrix),
        # which is the inverse of our camera-to-world pose T.
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1} / {len(frames_data)} frames integrated")

    print("  Extracting mesh from TSDF volume …")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# ── Post-processing ────────────────────────────────────────────────────────────

def remove_outliers(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Statistical outlier removal: discard points whose average distance to their
    20 nearest neighbours is more than 2 standard deviations above the mean.
    This cleans up isolated floating points from depth noise.
    """
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd.select_by_index(ind)


def poisson_mesh(pcd: o3d.geometry.PointCloud,
                 depth: int = 9) -> o3d.geometry.TriangleMesh:
    """
    Poisson surface reconstruction from a coloured point cloud.

    Requires oriented normals.  depth=9 gives ~0.5 mm feature resolution;
    use depth=8 if the reconstruction is too slow or memory-heavy.

    The density-based trimming step removes low-confidence surface patches
    that Poisson tends to extrapolate beyond the actual scan coverage.
    """
    # Re-estimate and consistently orient normals before Poisson
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=100)

    mesh, densities = o3d.geometry.TriangleMesh \
        .create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1)

    # Remove the bottom 5 % by density — these are extrapolated phantom faces
    threshold = np.quantile(np.asarray(densities), 0.05)
    keep_mask = np.asarray(densities) > threshold
    mesh = mesh.select_by_index(np.where(keep_mask)[0])
    mesh.compute_vertex_normals()
    return mesh


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel   = args.voxel

    # ── Discover frame pairs ───────────────────────────────────────────────────
    color_files = sorted(in_dir.glob("*_color.png"))[::args.step]
    frame_pairs = [
        (cp, in_dir / cp.name.replace("_color.png", "_depth.png"))
        for cp in color_files
        if (in_dir / cp.name.replace("_color.png", "_depth.png")).exists()
    ]

    if len(frame_pairs) < 3:
        sys.exit(f"ERROR: found only {len(frame_pairs)} frame pairs in {in_dir}.\n"
                 "       Need at least 3.  Run capture.py first.")

    print(f"Found {len(frame_pairs)} frame pairs  (--step {args.step})")

    # ── Load intrinsics ────────────────────────────────────────────────────────
    intrinsic = load_intrinsics(in_dir)
    print(f"Intrinsics: {intrinsic.width}×{intrinsic.height}  "
          f"fx={intrinsic.intrinsic_matrix[0, 0]:.1f}  "
          f"fy={intrinsic.intrinsic_matrix[1, 1]:.1f}")

    # ── Load frames and create point clouds ────────────────────────────────────
    print("\nLoading frames and building point clouds …")
    frames_data = []   # list of (RGBDImage, PointCloud) for valid frames

    for cp, dp in frame_pairs:
        rgbd = load_rgbd(cp, dp)
        pcd  = rgbd_to_pcd(rgbd, intrinsic).voxel_down_sample(voxel)
        if len(pcd.points) < 100:
            continue   # skip frames with almost no depth data
        frames_data.append((rgbd, pcd))

    print(f"  {len(frames_data)} non-empty point clouds")

    if len(frames_data) < 3:
        sys.exit("ERROR: Too few valid frames after filtering.  "
                 "Check that the camera was aimed at the object during capture.")

    # ── ICP registration — build camera pose chain ─────────────────────────────
    # transforms[i] maps points in frame i's camera space → world space (frame 0).
    # We initialise with identity (frame 0 is the world origin) and accumulate
    # each frame-to-frame transform.
    print("\nRegistering frames with ICP …")
    transforms    = [np.eye(4)]
    cumulative_T  = np.eye(4)
    low_fitness   = 0

    for i in range(1, len(frames_data)):
        _, src = frames_data[i]
        _, tgt = frames_data[i - 1]

        T, fitness = icp_register(
            o3d.geometry.PointCloud(src),   # pass copies — ICP adds normals in-place
            o3d.geometry.PointCloud(tgt),
            voxel,
        )
        if fitness < 0.3:
            low_fitness += 1
            print(f"  WARNING: low ICP fitness ({fitness:.2f}) at frame {i} "
                  f"— consider reducing --step or capturing more slowly")

        # Chain: cumulative_T now maps frame i → world
        cumulative_T = cumulative_T @ T
        transforms.append(cumulative_T.copy())

        if (i + 1) % 10 == 0 or i == len(frames_data) - 1:
            print(f"  {i + 1} / {len(frames_data)} registered")

    if low_fitness > len(frames_data) // 3:
        print(f"\n  NOTE: {low_fitness} frames had low ICP fitness.  "
              "The reconstruction may show drift or gaps.  "
              "Try --step 1 or re-capture more slowly.")

    # ── Fuse into global representation ───────────────────────────────────────
    mesh = None

    if args.tsdf:
        print("\nRunning TSDF integration …")
        mesh = fuse_tsdf(frames_data, transforms, intrinsic, voxel)
        # Derive a point cloud from the mesh for outlier stats / saving
        pcd_global = mesh.sample_points_uniformly(number_of_points=200_000)
    else:
        print("\nMerging point clouds into world frame …")
        pcd_global = fuse_point_clouds(frames_data, transforms, voxel)

    # ── Post-process point cloud ───────────────────────────────────────────────
    print("Removing outliers …")
    pcd_global = remove_outliers(pcd_global)
    print(f"  Final point cloud: {len(pcd_global.points):,} points")

    # ── Save point cloud ───────────────────────────────────────────────────────
    pcd_path = out_dir / "pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_global)
    print(f"\nSaved point cloud → {pcd_path}")

    # ── Mesh reconstruction ────────────────────────────────────────────────────
    if not args.no_mesh:
        if not args.tsdf:
            print("Reconstructing mesh with Poisson surface reconstruction …")
            mesh = poisson_mesh(pcd_global)

        v_count = len(mesh.vertices)
        t_count = len(mesh.triangles)
        print(f"  Mesh: {v_count:,} vertices  {t_count:,} triangles")

        mesh_ply = out_dir / "mesh.ply"
        mesh_obj = out_dir / "mesh.obj"
        o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
        o3d.io.write_triangle_mesh(str(mesh_obj), mesh)
        print(f"Saved mesh → {mesh_ply}")
        print(f"Saved mesh → {mesh_obj}")

    # ── Interactive viewer ─────────────────────────────────────────────────────
    print("\nOpening 3-D viewer …  (close window to exit)")
    to_show = [pcd_global]
    if mesh is not None and not args.no_mesh:
        to_show.append(mesh)
    o3d.visualization.draw_geometries(
        to_show,
        window_name="3-D Scan Result",
        width=1280, height=720,
    )


if __name__ == "__main__":
    main()
