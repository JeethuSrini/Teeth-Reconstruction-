11#!/usr/bin/env python3
"""
Hybrid Tooth Reconstruction

Combines original high-resolution tooth mesh with SSM-reconstructed worn regions.

Strategy:
1. Load original high-res mesh (~119k vertices) 
2. Load the original wear mask (identifies worn vertices)
3. Load SSM reconstructed points (25k) and upsample/map to high-res
4. Replace worn regions in original with reconstructed points
5. Mesh the combined point cloud

This preserves the original tooth quality in intact regions while
only filling in the worn/missing areas with SSM predictions.

Usage:
    python hybrid_reconstruction.py --tooth-num 01 --wear-type mild_cusp1
    python hybrid_reconstruction.py --batch
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import trimesh


def load_point_cloud(filepath: str) -> np.ndarray:
    """Load point cloud from PLY file."""
    mesh = trimesh.load(filepath)
    if hasattr(mesh, 'vertices'):
        return np.array(mesh.vertices)
    else:
        return np.array(mesh)


def save_point_cloud(points: np.ndarray, filepath: str):
    """Save point cloud to PLY file."""
    # Create PLY content
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(filepath, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def normalize_to_unit_sphere(pts: np.ndarray) -> tuple:
    """Normalize points to unit sphere, return normalized points and transform params."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / scale if scale > 0 else centered
    return normalized, centroid, scale


def map_mask_to_highres(original_mask: np.ndarray,
                        original_mesh_vertices: np.ndarray,
                        target_vertices: np.ndarray) -> np.ndarray:
    """
    Map mask from original mesh to target vertices using normalized coordinates.
    
    Args:
        original_mask: Boolean mask for original mesh vertices
        original_mesh_vertices: Original mesh vertices (N, 3)
        target_vertices: Target vertices to map mask to (M, 3)
        
    Returns:
        Boolean mask for target vertices
    """
    # Normalize both to unit sphere for coordinate-invariant mapping
    orig_norm, _, _ = normalize_to_unit_sphere(original_mesh_vertices)
    target_norm, _, _ = normalize_to_unit_sphere(target_vertices)
    
    # Find nearest neighbor in original mesh for each target vertex
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(orig_norm)
    distances, indices = nn.kneighbors(target_norm)
    indices = indices.flatten()
    
    # Transfer mask values
    return original_mask[indices]


def icp_align_with_scale(source: np.ndarray, target: np.ndarray, 
                         max_iterations: int = 100, tolerance: float = 1e-6) -> tuple:
    """
    Align source point cloud to target using ICP with scale estimation.
    
    Args:
        source: Source points to align (N, 3)
        target: Target points to align to (M, 3)
        max_iterations: Maximum ICP iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (aligned_source, scale, rotation, translation)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # First, roughly align scales by matching bounding box sizes
    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)
    
    src_centered = source - src_center
    tgt_centered = target - tgt_center
    
    src_scale = np.max(np.abs(src_centered))
    tgt_scale = np.max(np.abs(tgt_centered))
    
    initial_scale = tgt_scale / src_scale
    
    # Initialize with scaled and centered source
    src = src_centered * initial_scale + tgt_center
    total_scale = initial_scale
    
    prev_error = float('inf')
    
    # Build target KD-tree
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    
    R_total = np.eye(3)
    t_total = tgt_center - initial_scale * src_center
    
    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, indices = nn.kneighbors(src)
        indices = indices.flatten()
        
        # Get corresponding points
        dst = target[indices]
        
        # Compute centroids
        src_centroid = src.mean(axis=0)
        dst_centroid = dst.mean(axis=0)
        
        # Center the points
        src_c = src - src_centroid
        dst_c = dst - dst_centroid
        
        # Compute scale for this iteration
        src_norm = np.sqrt(np.sum(src_c ** 2))
        dst_norm = np.sqrt(np.sum(dst_c ** 2))
        scale = dst_norm / src_norm if src_norm > 0 else 1.0
        
        # Normalize for rotation estimation
        src_n = src_c / src_norm if src_norm > 0 else src_c
        dst_n = dst_c / dst_norm if dst_norm > 0 else dst_c
        
        # Compute optimal rotation using SVD
        H = src_n.T @ dst_n
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply transformation: scale, rotate, translate
        src = scale * (R @ src_c.T).T + dst_centroid
        
        # Update totals
        total_scale *= scale
        R_total = R @ R_total
        t_total = dst_centroid - scale * (R @ src_centroid)
        
        # Check convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    print(f"      ICP converged: scale={total_scale:.4f}, error={mean_error:.6f}")
    
    return src, total_scale, R_total, t_total


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple:
    """
    Align source to target using Procrustes analysis (optimal rigid + scale).
    
    Args:
        source: Source points (N, 3)
        target: Target points (N, 3) - must be same size and correspond
        
    Returns:
        Tuple of (aligned_source, scale, rotation, translation)
    """
    # Center both
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    
    # Compute scale
    src_scale = np.sqrt(np.sum(src_centered ** 2))
    tgt_scale = np.sqrt(np.sum(tgt_centered ** 2))
    scale = tgt_scale / src_scale
    
    # Normalize
    src_normalized = src_centered / src_scale
    tgt_normalized = tgt_centered / tgt_scale
    
    # Compute rotation using SVD
    H = src_normalized.T @ tgt_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    aligned = (scale * (R @ src_centered.T).T) + tgt_centroid
    
    return aligned, scale, R, tgt_centroid - scale * (R @ src_centroid)


def extract_worn_region_from_reconstruction(reconstructed_25k: np.ndarray,
                                            ground_truth_25k: np.ndarray,
                                            threshold_percentile: float = 95) -> tuple:
    """
    Extract the worn/reconstructed region from the SSM output.
    
    Points where reconstruction differs significantly from ground truth
    are considered "reconstructed" (filled-in wear).
    
    Args:
        reconstructed_25k: SSM reconstructed points (25k, 3)
        ground_truth_25k: Ground truth corresponded points (25k, 3)
        threshold_percentile: Percentile threshold for identifying filled regions
        
    Returns:
        Tuple of (worn_region_points, worn_indices)
    """
    # Compute point-wise differences
    differences = np.linalg.norm(reconstructed_25k - ground_truth_25k, axis=1)
    
    # Points with large differences are in the worn/reconstructed region
    threshold = np.percentile(differences, threshold_percentile)
    worn_mask = differences > threshold
    
    return reconstructed_25k[worn_mask], np.where(worn_mask)[0]


def hybrid_reconstruct(tooth_num: str,
                       wear_type: str,
                       artificial_wear_dir: str,
                       correspondence_dir: str,
                       reconstruction_dir: str,
                       output_dir: str) -> dict:
    """
    Perform hybrid reconstruction for a single worn tooth.
    
    Args:
        tooth_num: Tooth number (e.g., "01")
        wear_type: Wear type (e.g., "mild_cusp1")
        artificial_wear_dir: Path to artificial_wear/output
        correspondence_dir: Path to correspondence output
        reconstruction_dir: Path to SSM reconstruction output
        output_dir: Path to save hybrid reconstruction
        
    Returns:
        Dictionary with reconstruction info
    """
    result = {
        "tooth_num": tooth_num,
        "wear_type": wear_type,
        "success": False,
        "error": None
    }
    
    # Construct paths
    worn_dir_name = f"tooth_{tooth_num}_wear_{wear_type}"
    
    # WORN high-res mesh (this is our INPUT - the damaged tooth we want to fix)
    worn_mesh_path = os.path.join(artificial_wear_dir, f"tooth_{tooth_num}", f"wear_{wear_type}.ply")
    
    # Original high-res mesh (ground truth - what we want to reconstruct towards)
    original_mesh_path = os.path.join(artificial_wear_dir, f"tooth_{tooth_num}", "original.ply")
    
    # Original mask (identifies which vertices in ORIGINAL mesh were removed to create wear)
    original_mask_path = os.path.join(artificial_wear_dir, f"tooth_{tooth_num}", f"removed_mask_{wear_type}.npy")
    
    # SSM reconstruction (25k) - the predicted complete tooth
    reconstruction_path = os.path.join(reconstruction_dir, worn_dir_name, "reconstructed.ply")
    
    # Check all files exist
    for path, name in [(worn_mesh_path, "worn mesh"),
                       (original_mesh_path, "original mesh"),
                       (original_mask_path, "original mask"),
                       (reconstruction_path, "SSM reconstruction")]:
        if not os.path.exists(path):
            result["error"] = f"Missing {name}: {path}"
            print(f"  [ERROR] {result['error']}")
            return result
    
    print(f"  Loading data...")
    
    # Load all data
    worn_highres = load_point_cloud(worn_mesh_path)  # The damaged input tooth
    original_highres = load_point_cloud(original_mesh_path)  # Ground truth (complete tooth)
    original_mask = np.load(original_mask_path).astype(bool)  # Which vertices were removed
    reconstructed_25k = load_point_cloud(reconstruction_path)  # SSM prediction
    
    print(f"    Worn mesh (input): {len(worn_highres)} vertices")
    print(f"    Original mesh (ground truth): {len(original_highres)} vertices")
    print(f"    Original mask: {original_mask.sum()}/{len(original_mask)} worn ({100*original_mask.mean():.1f}%)")
    print(f"    SSM reconstruction: {len(reconstructed_25k)} points")
    
    # Verify mask matches original mesh
    if len(original_mask) != len(original_highres):
        result["error"] = f"Mask size mismatch: {len(original_mask)} vs {len(original_highres)}"
        print(f"  [ERROR] {result['error']}")
        return result
    
    print(f"  Step 1: Loading worn tooth's transforms and applying inverse...")
    
    # Load the transforms that were applied to the worn tooth during correspondence
    worn_corr_dir = os.path.join(correspondence_dir, "artificial_worn", worn_dir_name)
    norm_path = os.path.join(worn_corr_dir, "normalization.json")
    icp_path = os.path.join(worn_corr_dir, "icp_transform.npy")
    
    if not os.path.exists(norm_path):
        result["error"] = f"Missing worn tooth normalization: {norm_path}"
        print(f"  [ERROR] {result['error']}")
        return result
    
    # Load normalization parameters
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    
    worn_centroid = np.array(norm_data["centroid"])
    worn_scale = norm_data["scale"]
    worn_pca_rotation = np.array(norm_data["pca_rotation"])
    
    print(f"    Worn tooth centroid: [{worn_centroid[0]:.2f}, {worn_centroid[1]:.2f}, {worn_centroid[2]:.2f}]")
    print(f"    Worn tooth scale: {worn_scale:.4f}")
    
    # Load ICP transform if exists
    if os.path.exists(icp_path):
        icp_transform = np.load(icp_path)
        has_icp = True
        print(f"    ICP transform loaded")
    else:
        has_icp = False
        print(f"    No ICP transform found")
    
    # Apply INVERSE transformation to SSM reconstruction
    # Forward was: center -> rotate -> scale -> ICP
    # Inverse is: inv_ICP -> inv_scale -> inv_rotate -> add_centroid
    
    reconstructed_aligned = reconstructed_25k.copy()
    
    # Step 1: Inverse ICP (if exists)
    if has_icp:
        R_icp = icp_transform[:3, :3]
        t_icp = icp_transform[:3, 3]
        # Inverse: R^T @ (p - t)
        reconstructed_aligned = (reconstructed_aligned - t_icp) @ R_icp
    
    # Step 2: Inverse scale (multiply by original scale)
    reconstructed_aligned = reconstructed_aligned * worn_scale
    
    # Step 3: Inverse PCA rotation (transpose)
    reconstructed_aligned = reconstructed_aligned @ worn_pca_rotation.T
    
    # Step 4: Add back centroid
    reconstructed_aligned = reconstructed_aligned + worn_centroid
    
    print(f"    Applied inverse transform to SSM ({len(reconstructed_aligned)} points)")
    
    print(f"  Step 1b: Fine-tuning alignment to worn mesh using ICP...")
    
    # The inverse transform gets us close, but we need to fine-tune
    # Use ICP to align the SSM reconstruction to the WORN mesh (not original)
    # This ensures the reconstructed region connects seamlessly to the intact surface
    reconstructed_aligned, fine_scale, fine_R, fine_t = icp_align_with_scale(
        reconstructed_aligned,
        worn_highres,
        max_iterations=50
    )
    print(f"    Fine-tuned alignment: scale={fine_scale:.4f}")
    
    print(f"  Step 2: Interpolating SSM to high-res for worn region...")
    
    # For each worn vertex in the original mesh, find the predicted position
    # from the aligned SSM reconstruction using nearest-neighbor interpolation
    worn_vertex_indices = np.where(original_mask)[0]
    worn_original_positions = original_highres[original_mask]  # Where they should be
    
    # Find nearest points in aligned SSM for interpolation
    nn = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(reconstructed_aligned)
    distances, indices = nn.kneighbors(worn_original_positions)
    
    # Inverse distance weighted interpolation
    weights = 1.0 / (distances + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Compute interpolated positions for worn vertices
    reconstructed_worn_region = np.zeros_like(worn_original_positions)
    for i in range(len(worn_original_positions)):
        reconstructed_worn_region[i] = np.sum(
            weights[i, :, np.newaxis] * reconstructed_aligned[indices[i]], axis=0
        )
    
    print(f"    Interpolated {len(reconstructed_worn_region)} worn vertices from SSM")
    
    print(f"  Step 3: Creating hybrid mesh...")
    
    # The worn mesh has FEWER vertices than original (vertices were removed during wear)
    # So we need to ADD the reconstructed worn region to the worn mesh
    
    # Compute the distance from original worn positions to reconstructed
    original_worn_positions = original_highres[original_mask]
    displacement = np.linalg.norm(reconstructed_worn_region - original_worn_positions, axis=1)
    avg_displacement = np.mean(displacement)
    
    # MERGE: worn mesh (intact) + reconstructed worn region (filled)
    hybrid_points = np.vstack([worn_highres, reconstructed_worn_region])
    
    print(f"    Worn mesh: {len(worn_highres)} vertices (intact surface)")
    print(f"    Reconstructed region: {len(reconstructed_worn_region)} vertices (filled)")
    print(f"    Hybrid mesh: {len(hybrid_points)} total vertices")
    print(f"    Reconstruction error (vs ground truth): {avg_displacement:.4f}")
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    
    hybrid_path = os.path.join(output_dir, worn_dir_name)
    os.makedirs(hybrid_path, exist_ok=True)
    
    # Save hybrid point cloud
    save_point_cloud(hybrid_points, os.path.join(hybrid_path, "hybrid_reconstructed.ply"))
    
    # Also save just the reconstructed worn region for visualization
    save_point_cloud(reconstructed_worn_region, os.path.join(hybrid_path, "worn_region_filled.ply"))
    
    # Save the aligned SSM reconstruction for debugging
    save_point_cloud(reconstructed_aligned, os.path.join(hybrid_path, "ssm_aligned.ply"))
    
    # Save metadata
    metadata = {
        "tooth_num": tooth_num,
        "wear_type": wear_type,
        "worn_mesh_vertices": len(worn_highres),
        "original_mesh_vertices": len(original_highres),
        "ssm_vertices": len(reconstructed_25k),
        "worn_vertices_replaced": int(original_mask.sum()),
        "worn_percentage": float(100 * original_mask.mean()),
        "avg_displacement": float(avg_displacement),
        "method": "hybrid_icp_interpolate"
    }
    with open(os.path.join(hybrid_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    Saved: {os.path.join(hybrid_path, 'hybrid_reconstructed.ply')}")
    print(f"    Hybrid mesh: {len(hybrid_points)} vertices")
    print(f"      - From worn input: {len(hybrid_points) - original_mask.sum()} intact")
    print(f"      - Replaced with SSM: {original_mask.sum()} worn")
    
    result["success"] = True
    result["output_path"] = os.path.join(hybrid_path, "hybrid_reconstructed.ply")
    result["n_vertices"] = len(hybrid_points)
    result["n_replaced"] = int(original_mask.sum())
    result["avg_displacement"] = float(avg_displacement)
    
    return result


def run_batch(artificial_wear_dir: str,
              correspondence_dir: str,
              reconstruction_dir: str,
              output_dir: str) -> list:
    """
    Run hybrid reconstruction on all available teeth.
    """
    results = []
    
    # Find all reconstruction directories
    recon_path = Path(reconstruction_dir)
    recon_dirs = sorted([d.name for d in recon_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(recon_dirs)} reconstruction directories\n")
    
    for worn_dir in recon_dirs:
        # Parse directory name: tooth_XX_wear_YYYYY
        parts = worn_dir.split("_wear_")
        if len(parts) != 2:
            print(f"[SKIP] Cannot parse: {worn_dir}")
            continue
        
        tooth_num = parts[0].replace("tooth_", "")
        wear_type = parts[1]
        
        print(f"Processing {worn_dir}...")
        
        result = hybrid_reconstruct(
            tooth_num=tooth_num,
            wear_type=wear_type,
            artificial_wear_dir=artificial_wear_dir,
            correspondence_dir=correspondence_dir,
            reconstruction_dir=reconstruction_dir,
            output_dir=output_dir
        )
        
        results.append(result)
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid tooth reconstruction: original mesh + SSM-filled worn regions"
    )
    
    parser.add_argument("--tooth-num", "-t", type=str,
                        help="Tooth number (e.g., 01)")
    parser.add_argument("--wear-type", "-w", type=str,
                        help="Wear type (e.g., mild_cusp1)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process all available teeth")
    
    # Directory paths
    parser.add_argument("--artificial-wear-dir", type=str,
                        default="../artificial_wear/output",
                        help="Path to artificial_wear/output")
    parser.add_argument("--correspondence-dir", type=str,
                        default="output/correspondence",
                        help="Path to correspondence output")
    parser.add_argument("--reconstruction-dir", type=str,
                        default="output/reconstructions",
                        help="Path to SSM reconstruction output")
    parser.add_argument("--output-dir", type=str,
                        default="output/hybrid_reconstructions",
                        help="Path to save hybrid reconstructions")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hybrid Tooth Reconstruction")
    print("=" * 60)
    print(f"Artificial wear dir: {args.artificial_wear_dir}")
    print(f"Correspondence dir: {args.correspondence_dir}")
    print(f"Reconstruction dir: {args.reconstruction_dir}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    if args.batch:
        # Batch mode
        results = run_batch(
            artificial_wear_dir=args.artificial_wear_dir,
            correspondence_dir=args.correspondence_dir,
            reconstruction_dir=args.reconstruction_dir,
            output_dir=args.output_dir
        )
        
        # Summary
        success = sum(1 for r in results if r["success"])
        failed = sum(1 for r in results if not r["success"])
        
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Success: {success}")
        print(f"  Failed:  {failed}")
        
    elif args.tooth_num and args.wear_type:
        # Single tooth mode
        result = hybrid_reconstruct(
            tooth_num=args.tooth_num,
            wear_type=args.wear_type,
            artificial_wear_dir=args.artificial_wear_dir,
            correspondence_dir=args.correspondence_dir,
            reconstruction_dir=args.reconstruction_dir,
            output_dir=args.output_dir
        )
        
        if result["success"]:
            print("\nHybrid reconstruction complete!")
        else:
            print(f"\nFailed: {result['error']}")
            exit(1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Single tooth:")
        print("    python hybrid_reconstruction.py -t 01 -w mild_cusp1")
        print("  Batch processing:")
        print("    python hybrid_reconstruction.py --batch")


if __name__ == "__main__":
    main()
