#!/usr/bin/env python3
"""
Point Cloud to Mesh Conversion for Tooth Surfaces

Converts reconstructed point clouds into smooth, anatomically plausible
tooth surface meshes using Open3D and PyVista.

Designed for:
- Open EDJ/dentine surfaces (not watertight/volumetric)
- Handling noise, cracks, and locally dense regions
- Preserving cuspal morphology
- Avoiding volumetric filling artifacts

Usage:
    python pointcloud_to_mesh.py --input reconstructed.ply --output mesh.ply
    python pointcloud_to_mesh.py --batch-dir output/reconstructions/
"""

import argparse
import os
import numpy as np
import open3d as o3d
from pathlib import Path

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load point cloud from PLY file."""
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {filepath}")
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, 
                           voxel_size: float = None,
                           remove_outliers: bool = True,
                           nb_neighbors: int = 30,
                           std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """
    Preprocess point cloud: downsample and remove outliers.
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling. If None, auto-computed.
        remove_outliers: Whether to remove statistical outliers
        nb_neighbors: Number of neighbors for outlier detection
        std_ratio: Standard deviation ratio for outlier detection
        
    Returns:
        Preprocessed point cloud
    """
    points = np.asarray(pcd.points)
    
    # Compute bounding box for scale estimation
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    
    # Auto voxel size: aim for ~20k-50k points for good mesh quality
    if voxel_size is None:
        n_points = len(points)
        if n_points > 50000:
            # Downsample large point clouds
            voxel_size = bbox_diag / 200
            pcd = pcd.voxel_down_sample(voxel_size)
            print(f"    Downsampled: {n_points} -> {len(pcd.points)} points")
    
    # Remove statistical outliers (noise, isolated points)
    if remove_outliers and len(pcd.points) > 100:
        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        removed = len(pcd.points) - len(pcd_clean.points)
        if removed > 0:
            print(f"    Removed {removed} outliers ({100*removed/len(pcd.points):.1f}%)")
        pcd = pcd_clean
    
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud,
                     radius: float = None,
                     max_nn: int = 50) -> o3d.geometry.PointCloud:
    """
    Estimate and orient normals for the point cloud.
    
    Args:
        pcd: Input point cloud
        radius: Search radius for normal estimation. If None, auto-computed.
        max_nn: Maximum number of neighbors for normal estimation
        
    Returns:
        Point cloud with normals
    """
    points = np.asarray(pcd.points)
    
    # Auto radius based on point density
    if radius is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        # Estimate average point spacing
        radius = bbox_diag / (len(points) ** (1/3)) * 3
    
    print(f"    Estimating normals (radius={radius:.4f}, max_nn={max_nn})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )
    
    # Orient normals consistently - use camera location method
    # Place camera at centroid and flip normals to point outward
    centroid = np.mean(points, axis=0)
    pcd.orient_normals_towards_camera_location(centroid)
    
    # Flip normals to point AWAY from centroid (outward from tooth surface)
    normals = np.asarray(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(-normals)
    
    # Additional orientation refinement using propagation
    try:
        pcd.orient_normals_consistent_tangent_plane(k=15)
    except Exception:
        pass  # Not critical if this fails
    
    return pcd


def poisson_reconstruction(pcd: o3d.geometry.PointCloud,
                          depth: int = 9,
                          scale: float = 1.1,
                          density_threshold_percentile: float = 5.0) -> o3d.geometry.TriangleMesh:
    """
    Perform Poisson surface reconstruction.
    
    Args:
        pcd: Point cloud with normals
        depth: Octree depth (higher = more detail, 8-11 typical)
        scale: Scale factor for bounding box
        density_threshold_percentile: Remove vertices below this density percentile
        
    Returns:
        Reconstructed triangle mesh
    """
    print(f"    Poisson reconstruction (depth={depth})...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=depth, 
        scale=scale,
        linear_fit=False
    )
    
    # Remove low-density vertices (artifacts from extrapolation)
    if density_threshold_percentile > 0:
        densities = np.asarray(densities)
        threshold = np.percentile(densities, density_threshold_percentile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"    Removed {vertices_to_remove.sum()} low-density vertices")
    
    return mesh


def ball_pivoting_reconstruction(pcd: o3d.geometry.PointCloud,
                                  radii: list = None) -> o3d.geometry.TriangleMesh:
    """
    Perform Ball Pivoting Algorithm (BPA) surface reconstruction.
    
    Better for open surfaces as it doesn't try to close the mesh.
    
    Args:
        pcd: Point cloud with normals
        radii: List of ball radii to try. If None, auto-computed.
        
    Returns:
        Reconstructed triangle mesh
    """
    points = np.asarray(pcd.points)
    
    # Auto radii based on point density
    if radii is None:
        # Estimate average nearest neighbor distance
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        nn_distances = []
        sample_indices = np.random.choice(len(points), min(1000, len(points)), replace=False)
        for i in sample_indices:
            [k, idx, dist] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)
            if k > 1:
                nn_distances.append(np.sqrt(dist[1]))
        avg_nn = np.mean(nn_distances)
        
        # Use more radii for better hole coverage (smaller to larger)
        radii = [avg_nn * 0.8, avg_nn * 1.0, avg_nn * 1.5, avg_nn * 2.0, 
                 avg_nn * 2.5, avg_nn * 3.0, avg_nn * 4.0, avg_nn * 5.0]
    
    print(f"    Ball Pivoting reconstruction...")
    print(f"    Radii: {[f'{r:.4f}' for r in radii[:4]]}...")
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    return mesh


def fill_holes(mesh: o3d.geometry.TriangleMesh, 
               hole_size: float = None) -> o3d.geometry.TriangleMesh:
    """
    Fill holes in the mesh using PyVista if available.
    
    Args:
        mesh: Input mesh with holes
        hole_size: Maximum hole size to fill. If None, fill all holes.
        
    Returns:
        Mesh with holes filled
    """
    if not PYVISTA_AVAILABLE:
        print("    [WARN] PyVista not available for hole filling")
        return mesh
    
    print("    Filling holes with PyVista...")
    
    # Convert Open3D mesh to PyVista
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # PyVista format: [n_points_in_face, idx1, idx2, idx3, ...]
    faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).flatten()
    pv_mesh = pv.PolyData(vertices, faces)
    
    # Fill holes
    filled = pv_mesh.fill_holes(hole_size=hole_size if hole_size else 1000)
    
    # Convert back to Open3D
    filled_vertices = np.array(filled.points)
    filled_faces = filled.faces.reshape(-1, 4)[:, 1:4]  # Remove the "3" prefix
    
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(filled_vertices)
    result.triangles = o3d.utility.Vector3iVector(filled_faces)
    
    n_new_triangles = len(filled_faces) - len(triangles)
    print(f"    Added {n_new_triangles} triangles to fill holes")
    
    return result


def pyvista_reconstruction(points: np.ndarray, 
                           method: str = "delaunay",
                           alpha: float = 0.0,
                           smooth_iter: int = 50) -> o3d.geometry.TriangleMesh:
    """
    Surface reconstruction using PyVista/VTK.
    
    Args:
        points: (N, 3) numpy array of points
        method: "delaunay", "surface", or "robust"
        alpha: Alpha value for Delaunay (0 = convex hull, larger = more concave)
        smooth_iter: Smoothing iterations
        
    Returns:
        Open3D TriangleMesh
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista not available")
    
    print(f"    PyVista reconstruction (method={method})...")
    
    # Create point cloud
    cloud = pv.PolyData(points)
    
    if method == "surface":
        # Surface reconstruction with normals
        cloud.compute_normals(point_normals=True, inplace=True)
        mesh = cloud.reconstruct_surface(nbr_sz=20)
    elif method == "robust":
        # More robust approach: Delaunay 2D on projected surface
        # First compute normals
        cloud.compute_normals(point_normals=True, inplace=True)
        
        # Use surface reconstruction with larger neighborhood
        mesh = cloud.reconstruct_surface(nbr_sz=30, sample_spacing=None)
        
        # Aggressively fill holes
        mesh = mesh.fill_holes(hole_size=10000)
        
        # Decimate to reduce complexity then subdivide for smoothness
        if mesh.n_faces > 50000:
            mesh = mesh.decimate(0.5)
        
        # Subdivide for smoothness
        mesh = mesh.subdivide(1, subfilter='loop')
        
    else:
        # Delaunay 3D with alpha shapes
        mesh = cloud.delaunay_3d(alpha=alpha)
        mesh = mesh.extract_surface()
    
    # Fill any holes
    mesh = mesh.fill_holes(hole_size=1000)
    
    # Smooth
    if smooth_iter > 0:
        mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)
    
    # Clean
    mesh = mesh.clean()
    
    # Convert to Open3D
    vertices = np.array(mesh.points)
    faces_raw = mesh.faces
    if len(faces_raw) > 0:
        faces = faces_raw.reshape(-1, 4)[:, 1:4]
    else:
        faces = np.array([]).reshape(0, 3)
    
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(vertices)
    result.triangles = o3d.utility.Vector3iVector(faces)
    result.compute_vertex_normals()
    
    return result


def alpha_shape_reconstruction(pcd: o3d.geometry.PointCloud,
                               alpha: float = None) -> o3d.geometry.TriangleMesh:
    """
    Perform Alpha Shape surface reconstruction.
    
    Good for preserving concavities and sharp features.
    
    Args:
        pcd: Point cloud
        alpha: Alpha value. If None, auto-computed.
        
    Returns:
        Reconstructed triangle mesh
    """
    points = np.asarray(pcd.points)
    
    # Auto alpha based on point density
    if alpha is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        alpha = bbox_diag / 30  # Conservative alpha for teeth
    
    print(f"    Alpha shape reconstruction (alpha={alpha:.4f})...")
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    return mesh


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh,
                 remove_degenerate: bool = True,
                 remove_duplicates: bool = True,
                 remove_non_manifold: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Clean up mesh by removing degenerate/duplicate triangles.
    
    Args:
        mesh: Input mesh
        remove_degenerate: Remove degenerate triangles
        remove_duplicates: Remove duplicate vertices/triangles
        remove_non_manifold: Remove non-manifold edges
        
    Returns:
        Cleaned mesh
    """
    print("    Cleaning mesh...")
    
    if remove_degenerate:
        mesh.remove_degenerate_triangles()
    
    if remove_duplicates:
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
    
    if remove_non_manifold:
        mesh.remove_non_manifold_edges()
    
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    
    return mesh


def smooth_mesh(mesh: o3d.geometry.TriangleMesh,
                method: str = "taubin",
                iterations: int = 10,
                lambda_filter: float = 0.5,
                mu: float = -0.53) -> o3d.geometry.TriangleMesh:
    """
    Smooth the mesh while preserving features.
    
    Args:
        mesh: Input mesh
        method: "taubin" (feature-preserving) or "laplacian" (aggressive)
        iterations: Number of smoothing iterations
        lambda_filter: Lambda parameter for smoothing
        mu: Mu parameter for Taubin smoothing (negative, < -lambda)
        
    Returns:
        Smoothed mesh
    """
    print(f"    Smoothing mesh ({method}, {iterations} iterations)...")
    
    if method == "taubin":
        # Taubin smoothing - preserves volume and features better
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter,
            mu=mu
        )
    else:
        # Laplacian smoothing - more aggressive
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter
        )
    
    # Recompute normals after smoothing
    mesh.compute_vertex_normals()
    
    return mesh


def point_cloud_to_mesh(input_path: str,
                        output_path: str,
                        method: str = "poisson",
                        poisson_depth: int = 9,
                        density_threshold: float = 5.0,
                        smooth_iterations: int = 10,
                        remove_outliers: bool = True,
                        fill_holes_flag: bool = True) -> bool:
    """
    Convert point cloud to smooth mesh.
    
    Args:
        input_path: Path to input point cloud PLY
        output_path: Path to output mesh PLY
        method: "poisson", "ball_pivoting", "alpha_shape", or "pyvista"
        poisson_depth: Octree depth for Poisson (8-11)
        density_threshold: Percentile for density-based vertex removal
        smooth_iterations: Number of smoothing iterations
        remove_outliers: Whether to remove outliers in preprocessing
        fill_holes_flag: Whether to fill holes after reconstruction
        
    Returns:
        True if successful
    """
    try:
        # Load point cloud
        print(f"  Loading: {input_path}")
        pcd = load_point_cloud(input_path)
        print(f"    {len(pcd.points)} points")
        
        # Preprocess
        pcd = preprocess_point_cloud(pcd, remove_outliers=remove_outliers)
        
        # Estimate normals
        pcd = estimate_normals(pcd)
        
        # Surface reconstruction
        if method == "ball_pivoting":
            mesh = ball_pivoting_reconstruction(pcd)
        elif method == "alpha_shape":
            mesh = alpha_shape_reconstruction(pcd)
        elif method == "pyvista":
            if not PYVISTA_AVAILABLE:
                print("    [WARN] PyVista not available, falling back to ball_pivoting")
                mesh = ball_pivoting_reconstruction(pcd)
            else:
                mesh = pyvista_reconstruction(np.asarray(pcd.points), method="surface")
        elif method == "robust":
            if not PYVISTA_AVAILABLE:
                print("    [WARN] PyVista not available, falling back to poisson")
                mesh = poisson_reconstruction(pcd, depth=poisson_depth)
            else:
                mesh = pyvista_reconstruction(np.asarray(pcd.points), method="robust", smooth_iter=smooth_iterations)
        else:  # poisson
            mesh = poisson_reconstruction(
                pcd, 
                depth=poisson_depth,
                density_threshold_percentile=density_threshold
            )
        
        print(f"    Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Fill holes if requested (works best with ball_pivoting and alpha_shape)
        if fill_holes_flag and method in ["ball_pivoting", "alpha_shape"]:
            mesh = fill_holes(mesh)
        
        # Cleanup
        mesh = cleanup_mesh(mesh)
        
        # Smooth
        if smooth_iterations > 0:
            mesh = smooth_mesh(mesh, method="taubin", iterations=smooth_iterations)
        
        # Save
        print(f"  Saving: {output_path}")
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        print(f"    Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch(input_dir: str, 
                  output_suffix: str = "_mesh.ply",
                  **kwargs) -> dict:
    """
    Process all reconstructed.ply files in a directory.
    
    Args:
        input_dir: Directory containing reconstruction subdirectories
        output_suffix: Suffix for output mesh files
        **kwargs: Arguments passed to point_cloud_to_mesh
        
    Returns:
        Dictionary with success/failure counts
    """
    results = {"success": 0, "failed": 0, "skipped": 0}
    
    input_path = Path(input_dir)
    
    # Find all reconstructed.ply files
    ply_files = list(input_path.glob("**/reconstructed.ply"))
    
    if not ply_files:
        print(f"No reconstructed.ply files found in {input_dir}")
        return results
    
    print(f"Found {len(ply_files)} point clouds to process\n")
    
    for ply_path in sorted(ply_files):
        # Output path: same directory, different name
        output_path = ply_path.parent / f"reconstructed{output_suffix}"
        
        # Skip if already exists
        if output_path.exists():
            print(f"  [SKIP] {ply_path.parent.name} - mesh already exists")
            results["skipped"] += 1
            continue
        
        print(f"\nProcessing: {ply_path.parent.name}")
        
        if point_cloud_to_mesh(str(ply_path), str(output_path), **kwargs):
            results["success"] += 1
        else:
            results["failed"] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert reconstructed point clouds to smooth tooth surface meshes"
    )
    
    parser.add_argument("--input", "-i", type=str,
                        help="Input point cloud PLY file")
    parser.add_argument("--output", "-o", type=str,
                        help="Output mesh PLY file")
    parser.add_argument("--batch-dir", "-b", type=str,
                        help="Directory for batch processing (looks for reconstructed.ply)")
    parser.add_argument("--method", "-m", type=str, default="poisson",
                        choices=["poisson", "ball_pivoting", "alpha_shape", "pyvista", "robust"],
                        help="Surface reconstruction method (robust = pyvista with hole filling)")
    parser.add_argument("--depth", "-d", type=int, default=9,
                        help="Poisson octree depth (8-11, higher=more detail)")
    parser.add_argument("--density-threshold", type=float, default=5.0,
                        help="Density percentile threshold for Poisson cleanup")
    parser.add_argument("--smooth-iterations", type=int, default=10,
                        help="Number of Taubin smoothing iterations")
    parser.add_argument("--no-outlier-removal", action="store_true",
                        help="Skip outlier removal in preprocessing")
    parser.add_argument("--fill-holes", action="store_true", default=True,
                        help="Fill holes after reconstruction (default: True)")
    parser.add_argument("--no-fill-holes", action="store_true",
                        help="Skip hole filling")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing mesh files in batch mode")
    
    args = parser.parse_args()
    
    if args.batch_dir:
        # Batch processing mode
        print("=" * 60)
        print("Batch Point Cloud to Mesh Conversion")
        print("=" * 60)
        print(f"Input directory: {args.batch_dir}")
        print(f"Method: {args.method}")
        print(f"Poisson depth: {args.depth}")
        print()
        
        results = process_batch(
            args.batch_dir,
            method=args.method,
            poisson_depth=args.depth,
            density_threshold=args.density_threshold,
            smooth_iterations=args.smooth_iterations,
            remove_outliers=not args.no_outlier_removal,
            fill_holes_flag=not args.no_fill_holes
        )
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Success: {results['success']}")
        print(f"  Failed:  {results['failed']}")
        print(f"  Skipped: {results['skipped']}")
        
    elif args.input and args.output:
        # Single file mode
        print("=" * 60)
        print("Point Cloud to Mesh Conversion")
        print("=" * 60)
        
        success = point_cloud_to_mesh(
            args.input,
            args.output,
            method=args.method,
            poisson_depth=args.depth,
            density_threshold=args.density_threshold,
            smooth_iterations=args.smooth_iterations,
            remove_outliers=not args.no_outlier_removal,
            fill_holes_flag=not args.no_fill_holes
        )
        
        if success:
            print("\nMesh generation complete!")
        else:
            print("\nMesh generation failed!")
            exit(1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Single file:")
        print("    python pointcloud_to_mesh.py -i reconstructed.ply -o mesh.ply")
        print("  Batch processing:")
        print("    python pointcloud_to_mesh.py --batch-dir output/reconstructions/")
        print("  With options:")
        print("    python pointcloud_to_mesh.py -i input.ply -o output.ply -m ball_pivoting")


if __name__ == "__main__":
    main()
