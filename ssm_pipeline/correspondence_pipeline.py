#!/usr/bin/env python3
"""
Correspondence Pipeline for EDJ Tooth Meshes

This script establishes point-to-point correspondence across all teeth using
template-based registration (ALPACA-like approach):
1. Sample uniform point clouds from meshes
2. Normalize (center, scale, PCA-align)
3. Rigid ICP alignment to template
4. Non-rigid CPD registration (template -> target) for correspondence

GPU acceleration via probreg/CuPy for CPD registration.

Author: Generated for Teeth-Reconstruction project
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh

# Try to import GPU-accelerated libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using CPU for linear algebra")

try:
    from probreg import cpd
    PROBREG_AVAILABLE = True
except ImportError:
    PROBREG_AVAILABLE = False
    print("probreg not available, falling back to pycpd")
    try:
        from pycpd import DeformableRegistration
    except ImportError:
        raise ImportError("Neither probreg nor pycpd available. Install one: pip install probreg or pip install pycpd")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CorrespondenceConfig:
    """Configuration for the correspondence pipeline."""
    # Sampling
    n_points: int = 20000
    
    # Normalization
    scale_method: str = "bbox_diagonal"  # "bbox_diagonal" or "unit_sphere"
    
    # ICP parameters
    icp_threshold: float = 0.02
    icp_max_iterations: int = 100
    icp_tolerance: float = 1e-6
    
    # CPD parameters
    cpd_alpha: float = 2.0       # Smoothness (higher = more rigid)
    cpd_beta: float = 2.0        # Motion coherence
    cpd_max_iterations: int = 150
    cpd_tolerance: float = 1e-5
    cpd_use_cuda: bool = True    # Use GPU if available
    
    # Template selection
    template_idx: int = 0        # Index of template tooth (0 = first)
    auto_select_template: bool = False  # Auto-select most central tooth
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# POINT CLOUD OPERATIONS
# =============================================================================

def sample_point_cloud(mesh_path: str, n_points: int, seed: int = 42) -> np.ndarray:
    """
    Uniformly sample points from a mesh surface.
    
    Args:
        mesh_path: Path to PLY mesh file
        n_points: Number of points to sample
        seed: Random seed for reproducibility
        
    Returns:
        Point cloud as (N, 3) numpy array
    """
    np.random.seed(seed)
    mesh = trimesh.load(mesh_path, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float64)


def normalize_point_cloud(points: np.ndarray, 
                         scale_method: str = "bbox_diagonal",
                         use_gpu: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Normalize point cloud: center, scale, and PCA-align.
    
    Args:
        points: Input point cloud (N, 3)
        scale_method: "bbox_diagonal" or "unit_sphere"
        use_gpu: Use CuPy for SVD if available
        
    Returns:
        Tuple of (normalized points, normalization parameters dict)
    """
    # Center to origin
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # Scale normalization
    if scale_method == "bbox_diagonal":
        bbox_min = centered.min(axis=0)
        bbox_max = centered.max(axis=0)
        scale = np.linalg.norm(bbox_max - bbox_min)
    else:  # unit_sphere
        scale = np.max(np.linalg.norm(centered, axis=1))
    
    if scale < 1e-10:
        scale = 1.0
    scaled = centered / scale
    
    # PCA alignment - align principal axes to XYZ
    if use_gpu and CUPY_AVAILABLE:
        scaled_gpu = cp.asarray(scaled)
        _, _, Vt = cp.linalg.svd(scaled_gpu, full_matrices=False)
        Vt = cp.asnumpy(Vt)
    else:
        _, _, Vt = np.linalg.svd(scaled, full_matrices=False)
    
    # Ensure right-handed coordinate system
    if np.linalg.det(Vt) < 0:
        Vt[2, :] *= -1
    
    aligned = scaled @ Vt.T
    
    params = {
        "centroid": centroid.tolist(),
        "scale": float(scale),
        "pca_rotation": Vt.tolist()
    }
    
    return aligned, params


def save_point_cloud(points: np.ndarray, filepath: str) -> None:
    """Save point cloud as PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)


def load_point_cloud(filepath: str) -> np.ndarray:
    """Load point cloud from PLY file."""
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points)


# =============================================================================
# REGISTRATION
# =============================================================================

def icp_align(source: np.ndarray, 
              target: np.ndarray,
              threshold: float = 0.02,
              max_iterations: int = 100,
              tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Rigid ICP alignment using Open3D.
    
    Args:
        source: Source point cloud (N, 3)
        target: Target point cloud (M, 3)
        threshold: Maximum correspondence distance
        max_iterations: Maximum ICP iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (aligned source, 4x4 transform matrix, metadata dict)
    """
    # Create Open3D point clouds
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source)
    
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target)
    
    # Estimate normals for better ICP
    src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Run ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=tolerance,
        relative_rmse=tolerance
    )
    
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    # Apply transform
    transform = result.transformation
    aligned = (transform[:3, :3] @ source.T).T + transform[:3, 3]
    
    metadata = {
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "iterations": max_iterations  # Open3D doesn't expose actual iteration count
    }
    
    return aligned, transform, metadata


def cpd_register(template: np.ndarray,
                 target: np.ndarray,
                 alpha: float = 2.0,
                 beta: float = 2.0,
                 max_iterations: int = 150,
                 tolerance: float = 1e-5,
                 use_cuda: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Non-rigid CPD registration: deform TEMPLATE onto TARGET.
    
    The output preserves template point ordering, establishing correspondence.
    
    Args:
        template: Template point cloud (N, 3) - will be deformed
        target: Target point cloud (M, 3) - registration target
        alpha: Smoothness parameter (higher = more rigid)
        beta: Motion coherence parameter
        max_iterations: Maximum CPD iterations
        tolerance: Convergence tolerance
        use_cuda: Use GPU acceleration if available
        
    Returns:
        Tuple of (deformed template with same N points, metadata dict)
    """
    start_time = time.time()
    
    if PROBREG_AVAILABLE:
        # Use probreg with optional GPU acceleration
        use_gpu = use_cuda and CUPY_AVAILABLE
        
        if use_gpu:
            # Convert to CuPy arrays for GPU processing
            template_arr = cp.asarray(template)
            target_arr = cp.asarray(target)
            
            # probreg CPD with CuPy backend
            tf_param, _, _ = cpd.registration_cpd(
                target_arr, template_arr,
                tf_type_name='nonrigid',
                w=0.0,
                maxiter=max_iterations,
                tol=tolerance
            )
            
            # Get deformed points
            deformed = cp.asnumpy(tf_param.transform(template_arr))
        else:
            # CPU version
            tf_param, _, _ = cpd.registration_cpd(
                target, template,
                tf_type_name='nonrigid',
                w=0.0,
                maxiter=max_iterations,
                tol=tolerance
            )
            deformed = tf_param.transform(template)
        
        method = "probreg_gpu" if use_gpu else "probreg_cpu"
    else:
        # Fallback to pycpd (CPU only)
        reg = DeformableRegistration(
            X=target, Y=template,
            alpha=alpha, beta=beta,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        deformed, _ = reg.register()
        method = "pycpd_cpu"
    
    elapsed = time.time() - start_time
    
    metadata = {
        "method": method,
        "alpha": alpha,
        "beta": beta,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "elapsed_seconds": elapsed
    }
    
    return deformed, metadata


# =============================================================================
# TEMPLATE SELECTION
# =============================================================================

def select_template_auto(point_clouds: List[np.ndarray]) -> int:
    """
    Automatically select the most "central" point cloud as template.
    
    Computes pairwise distances and selects the one with minimum
    total distance to all others.
    
    Args:
        point_clouds: List of normalized point clouds
        
    Returns:
        Index of the selected template
    """
    n = len(point_clouds)
    if n == 0:
        return 0
    
    # Compute pairwise centroid distances (fast approximation)
    centroids = np.array([pc.mean(axis=0) for pc in point_clouds])
    
    total_distances = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                total_distances[i] += np.linalg.norm(centroids[i] - centroids[j])
    
    return int(np.argmin(total_distances))


# =============================================================================
# PIPELINE
# =============================================================================

def process_single_tooth(mesh_path: str,
                        template_points: np.ndarray,
                        output_dir: str,
                        config: CorrespondenceConfig,
                        tooth_name: str) -> bool:
    """
    Process a single tooth through the correspondence pipeline.
    
    Args:
        mesh_path: Path to input mesh
        template_points: Template point cloud (normalized, ICP-aligned)
        output_dir: Output directory for this tooth
        config: Pipeline configuration
        tooth_name: Name for logging
        
    Returns:
        True if successful
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Processing {tooth_name}...")
    
    try:
        # Step 1: Sample point cloud
        print(f"    Sampling {config.n_points} points...")
        sampled = sample_point_cloud(mesh_path, config.n_points, config.seed)
        save_point_cloud(sampled, os.path.join(output_dir, "sampled_raw.ply"))
        
        # Step 2: Normalize
        print("    Normalizing...")
        normalized, norm_params = normalize_point_cloud(
            sampled, 
            config.scale_method,
            use_gpu=config.cpd_use_cuda and CUPY_AVAILABLE
        )
        save_point_cloud(normalized, os.path.join(output_dir, "normalized.ply"))
        
        with open(os.path.join(output_dir, "normalization.json"), 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        # Step 3: ICP alignment to template
        print("    ICP alignment...")
        icp_aligned, icp_transform, icp_meta = icp_align(
            normalized, template_points,
            threshold=config.icp_threshold,
            max_iterations=config.icp_max_iterations,
            tolerance=config.icp_tolerance
        )
        save_point_cloud(icp_aligned, os.path.join(output_dir, "icp_aligned.ply"))
        np.save(os.path.join(output_dir, "icp_transform.npy"), icp_transform)
        
        # Step 4: CPD non-rigid registration (template -> target)
        print("    CPD registration (template -> target)...")
        corresponded, cpd_meta = cpd_register(
            template_points, icp_aligned,  # Deform template onto this target
            alpha=config.cpd_alpha,
            beta=config.cpd_beta,
            max_iterations=config.cpd_max_iterations,
            tolerance=config.cpd_tolerance,
            use_cuda=config.cpd_use_cuda
        )
        save_point_cloud(corresponded, os.path.join(output_dir, "corresponded.ply"))
        
        # Save metadata
        metadata = {
            "source_mesh": os.path.basename(mesh_path),
            "n_points": config.n_points,
            "normalization": norm_params,
            "icp": icp_meta,
            "cpd": cpd_meta
        }
        with open(os.path.join(output_dir, "registration_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    Done! CPD took {cpd_meta['elapsed_seconds']:.2f}s ({cpd_meta['method']})")
        return True
        
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_correspondence_pipeline(good_teeth_dir: str,
                                artificial_wear_dir: str,
                                output_dir: str,
                                config: CorrespondenceConfig) -> None:
    """
    Run the full correspondence pipeline on all teeth.
    
    Args:
        good_teeth_dir: Directory containing good teeth PLY files
        artificial_wear_dir: Directory containing artificial wear output
        output_dir: Output directory
        config: Pipeline configuration
    """
    print("=" * 60)
    print("Correspondence Pipeline")
    print("=" * 60)
    print(f"Good teeth directory: {good_teeth_dir}")
    print(f"Artificial wear directory: {artificial_wear_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GPU acceleration: {CUPY_AVAILABLE and config.cpd_use_cuda}")
    print()
    
    # Find all good teeth
    good_teeth_files = sorted(glob(os.path.join(good_teeth_dir, "*.ply")))
    print(f"Found {len(good_teeth_files)} good teeth")
    
    if len(good_teeth_files) == 0:
        print("ERROR: No PLY files found in good teeth directory")
        return
    
    # Find all artificial wear teeth (original.ply files)
    artificial_original_files = sorted(glob(os.path.join(artificial_wear_dir, "tooth_*", "original.ply")))
    print(f"Found {len(artificial_original_files)} artificial wear originals")
    
    # Find all worn variants
    artificial_worn_files = sorted(glob(os.path.join(artificial_wear_dir, "tooth_*", "wear_*.ply")))
    print(f"Found {len(artificial_worn_files)} artificial wear variants")
    print()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    template_dir = os.path.join(output_dir, "template")
    good_output_dir = os.path.join(output_dir, "good_teeth")
    worn_output_dir = os.path.join(output_dir, "artificial_worn")
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(good_output_dir, exist_ok=True)
    os.makedirs(worn_output_dir, exist_ok=True)
    
    # Step 1: Sample and normalize all good teeth to select template
    print("Step 1: Sampling and normalizing good teeth...")
    good_normalized = []
    for i, filepath in enumerate(good_teeth_files):
        print(f"  [{i+1}/{len(good_teeth_files)}] {os.path.basename(filepath)}")
        sampled = sample_point_cloud(filepath, config.n_points, config.seed + i)
        normalized, _ = normalize_point_cloud(sampled, config.scale_method)
        good_normalized.append(normalized)
    
    # Step 2: Select template
    print("\nStep 2: Selecting template...")
    if config.auto_select_template:
        template_idx = select_template_auto(good_normalized)
        print(f"  Auto-selected template: tooth {template_idx + 1}")
    else:
        template_idx = min(config.template_idx, len(good_teeth_files) - 1)
        print(f"  Using specified template: tooth {template_idx + 1}")
    
    template_points = good_normalized[template_idx]
    save_point_cloud(template_points, os.path.join(template_dir, "template.ply"))
    
    template_meta = {
        "source_file": os.path.basename(good_teeth_files[template_idx]),
        "source_index": template_idx,
        "n_points": config.n_points,
        "auto_selected": config.auto_select_template
    }
    with open(os.path.join(template_dir, "template_metadata.json"), 'w') as f:
        json.dump(template_meta, f, indent=2)
    
    # Step 3: Process all good teeth
    print("\nStep 3: Processing good teeth...")
    good_success = 0
    for i, filepath in enumerate(good_teeth_files):
        tooth_dir = os.path.join(good_output_dir, f"tooth_{i+1:02d}")
        
        if i == template_idx:
            # Template tooth - just copy normalized as corresponded
            os.makedirs(tooth_dir, exist_ok=True)
            sampled = sample_point_cloud(filepath, config.n_points, config.seed + i)
            save_point_cloud(sampled, os.path.join(tooth_dir, "sampled_raw.ply"))
            
            normalized, norm_params = normalize_point_cloud(sampled, config.scale_method)
            save_point_cloud(normalized, os.path.join(tooth_dir, "normalized.ply"))
            save_point_cloud(normalized, os.path.join(tooth_dir, "icp_aligned.ply"))
            save_point_cloud(normalized, os.path.join(tooth_dir, "corresponded.ply"))
            
            np.save(os.path.join(tooth_dir, "icp_transform.npy"), np.eye(4))
            
            with open(os.path.join(tooth_dir, "normalization.json"), 'w') as f:
                json.dump(norm_params, f, indent=2)
            
            metadata = {
                "source_mesh": os.path.basename(filepath),
                "n_points": config.n_points,
                "is_template": True
            }
            with open(os.path.join(tooth_dir, "registration_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  [TEMPLATE] tooth_{i+1:02d}")
            good_success += 1
        else:
            if process_single_tooth(filepath, template_points, tooth_dir, config, f"tooth_{i+1:02d}"):
                good_success += 1
    
    print(f"\n  Good teeth: {good_success}/{len(good_teeth_files)} successful")
    
    # Step 4: Process artificial wear originals (ground truth for reconstruction)
    print("\nStep 4: Processing artificial wear originals...")
    original_success = 0
    for filepath in artificial_original_files:
        tooth_name = os.path.basename(os.path.dirname(filepath))
        tooth_dir = os.path.join(worn_output_dir, f"{tooth_name}_original")
        
        if process_single_tooth(filepath, template_points, tooth_dir, config, f"{tooth_name}_original"):
            original_success += 1
    
    print(f"\n  Originals: {original_success}/{len(artificial_original_files)} successful")
    
    # Step 5: Process all worn variants
    print("\nStep 5: Processing worn variants...")
    worn_success = 0
    for filepath in artificial_worn_files:
        parent_dir = os.path.basename(os.path.dirname(filepath))
        wear_name = os.path.splitext(os.path.basename(filepath))[0]
        tooth_dir = os.path.join(worn_output_dir, f"{parent_dir}_{wear_name}")
        
        if process_single_tooth(filepath, template_points, tooth_dir, config, f"{parent_dir}_{wear_name}"):
            worn_success += 1
    
    print(f"\n  Worn variants: {worn_success}/{len(artificial_worn_files)} successful")
    
    # Save pipeline config
    config_dict = {
        "config": {
            "n_points": config.n_points,
            "scale_method": config.scale_method,
            "icp_threshold": config.icp_threshold,
            "icp_max_iterations": config.icp_max_iterations,
            "icp_tolerance": config.icp_tolerance,
            "cpd_alpha": config.cpd_alpha,
            "cpd_beta": config.cpd_beta,
            "cpd_max_iterations": config.cpd_max_iterations,
            "cpd_tolerance": config.cpd_tolerance,
            "cpd_use_cuda": config.cpd_use_cuda,
            "template_idx": template_idx,
            "auto_select_template": config.auto_select_template,
            "seed": config.seed
        },
        "good_teeth_count": len(good_teeth_files),
        "artificial_originals_count": len(artificial_original_files),
        "artificial_worn_count": len(artificial_worn_files),
        "good_teeth_files": [os.path.basename(f) for f in good_teeth_files],
        "gpu_available": CUPY_AVAILABLE,
        "probreg_available": PROBREG_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Summary
    print()
    print("=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Template: tooth_{template_idx + 1:02d} ({os.path.basename(good_teeth_files[template_idx])})")
    print(f"Good teeth processed: {good_success}/{len(good_teeth_files)}")
    print(f"Artificial originals processed: {original_success}/{len(artificial_original_files)}")
    print(f"Worn variants processed: {worn_success}/{len(artificial_worn_files)}")
    print(f"Output saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Establish point correspondence across tooth meshes"
    )
    parser.add_argument(
        "--good-teeth", "-g",
        type=str,
        default=None,
        help="Directory containing good teeth PLY files"
    )
    parser.add_argument(
        "--artificial-wear", "-a",
        type=str,
        default=None,
        help="Directory containing artificial wear output"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--n-points", "-n",
        type=int,
        default=20000,
        help="Number of points to sample (default: 20000)"
    )
    parser.add_argument(
        "--template-idx",
        type=int,
        default=0,
        help="Index of template tooth (default: 0)"
    )
    parser.add_argument(
        "--auto-template",
        action="store_true",
        help="Auto-select most central tooth as template"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    good_teeth_dir = args.good_teeth or os.path.join(project_dir, "Good teeth")
    artificial_wear_dir = args.artificial_wear or os.path.join(project_dir, "artificial_wear", "output")
    output_dir = args.output or os.path.join(script_dir, "output", "correspondence")
    
    # Create config
    config = CorrespondenceConfig(
        n_points=args.n_points,
        template_idx=args.template_idx,
        auto_select_template=args.auto_template,
        cpd_use_cuda=not args.no_gpu,
        seed=args.seed
    )
    
    run_correspondence_pipeline(good_teeth_dir, artificial_wear_dir, output_dir, config)


if __name__ == "__main__":
    main()
