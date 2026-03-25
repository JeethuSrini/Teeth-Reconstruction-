#!/usr/bin/env python3
"""
SSM Reconstruction Pipeline for EDJ Tooth Meshes

This script builds a PCA-based Statistical Shape Model (SSM) from good teeth
and uses it to reconstruct missing anatomy in artificially worn teeth.

Pipeline:
1. Load corresponded point clouds from good teeth
2. Build SSM: mean shape + principal deformation modes (PCA)
3. For each worn tooth: fit SSM to observed points, reconstruct missing
4. Evaluate reconstruction accuracy against ground truth

GPU acceleration via CuPy/PyTorch for linear algebra operations.

Author: Generated for Teeth-Reconstruction project
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff

# Try to import GPU libraries
try:
    import cupy as cp
    from cupyx.scipy.linalg import solve as cp_solve
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using CPU for linear algebra")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReconstructionConfig:
    """Configuration for the reconstruction pipeline."""
    # SSM parameters
    n_components: int = None           # Number of PCA components (None = auto)
    variance_threshold: float = 0.99   # Keep modes explaining this much variance
    
    # Fitting parameters
    regularization: float = 1.0        # Tikhonov regularization strength
    
    # GPU settings
    use_gpu: bool = True
    
    # Output settings
    save_all_modes: bool = True        # Save individual mode visualizations
    
    # Evaluation split
    test_tooth: Optional[str] = None   # Held-out tooth ID (e.g., "01")
    
    # Real-worn mode
    skip_eval: bool = False            # Skip GT-based evaluation (for real worn teeth)
    proxy_missing_fraction: float = 0.15  # Fraction of points treated as missing in skip-eval mode
    real_merge_with_input: bool = False   # In real-worn mode, fuse inverse-transformed SSM with input mesh
    real_merge_knn: int = 3               # KNN used when mapping corresponded points back to input mesh


# =============================================================================
# I/O UTILITIES
# =============================================================================

def load_point_cloud(filepath: str) -> np.ndarray:
    """Load point cloud from PLY file."""
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points, dtype=np.float64)


def save_point_cloud(points: np.ndarray, filepath: str) -> None:
    """Save point cloud as PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)


def point_cloud_to_mesh(points: np.ndarray,
                        poisson_depth: int = 9,
                        density_quantile: float = 0.01,
                        smooth_iter: int = 30):
    """
    Convert point cloud to watertight smooth mesh using Open3D Screened Poisson.
    Returns a trimesh.Trimesh with outward normals.
    """
    import trimesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    nn_dist = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_spacing = np.mean(nn_dist)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=avg_spacing * 4.0, max_nn=30
        )
    )

    # Orient normals outward (away from centroid) before consistency pass
    centroid = np.mean(points, axis=0)
    normals = np.asarray(pcd.normals)
    dirs = points - centroid
    flip = np.sum(normals * dirs, axis=1) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k=20)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=True
    )

    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    mesh_o3d.remove_vertices_by_mask(densities < threshold)

    mesh_o3d = mesh_o3d.filter_smooth_taubin(
        number_of_iterations=smooth_iter, lambda_filter=0.5, mu=-0.53
    )

    # Convert to trimesh for robust normal handling and export
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(tm)
    trimesh.repair.fix_winding(tm)

    return tm


def save_mesh(mesh, filepath: str) -> None:
    """Save mesh to PLY file."""
    if hasattr(mesh, 'export'):
        mesh.export(filepath)
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(filepath, mesh)
    elif hasattr(mesh, 'save'):
        mesh.save(filepath)


# =============================================================================
# SSM BUILDING
# =============================================================================

def build_ssm(corresponded_paths: List[str],
              n_components: int = None,
              variance_threshold: float = 0.95,
              use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build PCA-based Statistical Shape Model from corresponded point clouds.
    
    The SSM represents shapes as: x(b) = μ + Σ bᵢφᵢ
    where μ is the mean shape and φᵢ are the principal modes.
    
    Args:
        corresponded_paths: List of paths to corresponded point cloud PLY files
        n_components: Number of components to keep (None = auto by variance)
        variance_threshold: Keep modes explaining this fraction of variance
        use_gpu: Use GPU for SVD if available
        
    Returns:
        Tuple of (mean_shape, eigenvectors, eigenvalues, variance_explained)
        - mean_shape: (3N,) flattened mean
        - eigenvectors: (K, 3N) principal modes
        - eigenvalues: (K,) variances per mode
        - variance_explained: fraction of variance captured
    """
    print("Building Statistical Shape Model...")
    
    # Load all corresponded point clouds
    shapes = []
    for path in corresponded_paths:
        points = load_point_cloud(path)
        shapes.append(points)
    
    shapes = np.array(shapes)  # (M, N, 3)
    M, N, _ = shapes.shape
    print(f"  Loaded {M} shapes with {N} points each")
    
    # Flatten to (M, 3N)
    X = shapes.reshape(M, -1)
    
    # Compute mean shape
    mu = X.mean(axis=0)
    
    # Center data
    D = X - mu
    
    # SVD for PCA
    if use_gpu and CUPY_AVAILABLE:
        print("  Using GPU (CuPy) for SVD...")
        D_gpu = cp.asarray(D)
        U, S, Vt = cp.linalg.svd(D_gpu, full_matrices=False)
        S = cp.asnumpy(S)
        Vt = cp.asnumpy(Vt)
    elif use_gpu and TORCH_AVAILABLE:
        print("  Using GPU (PyTorch) for SVD...")
        D_torch = torch.tensor(D, device='cuda', dtype=torch.float64)
        U, S, Vt = torch.linalg.svd(D_torch, full_matrices=False)
        S = S.cpu().numpy()
        Vt = Vt.cpu().numpy()
    else:
        print("  Using CPU for SVD...")
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
    
    # Compute eigenvalues (variances)
    eigenvalues = (S ** 2) / (M - 1)
    
    # Determine number of components
    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var
    
    if n_components is None:
        n_components = np.searchsorted(cumulative_var, variance_threshold) + 1
        n_components = min(n_components, M - 1)  # Can't have more than M-1 components
    
    print(f"  Keeping {n_components} components ({cumulative_var[n_components-1]*100:.1f}% variance)")
    
    # Extract top components
    eigenvectors = Vt[:n_components]  # (K, 3N)
    eigenvalues = eigenvalues[:n_components]  # (K,)
    variance_explained = cumulative_var[n_components - 1]
    
    return mu, eigenvectors, eigenvalues, variance_explained


def visualize_modes(mu: np.ndarray,
                   eigenvectors: np.ndarray,
                   eigenvalues: np.ndarray,
                   output_dir: str,
                   n_std: float = 2.0) -> None:
    """
    Save visualizations of each deformation mode.
    
    For each mode, saves mean ± n_std * sqrt(eigenvalue) * eigenvector
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(mu) // 3
    
    # Save mean shape
    mean_shape = mu.reshape(-1, 3)
    save_point_cloud(mean_shape, os.path.join(output_dir, "mean_shape.ply"))
    
    # Save each mode
    for i, (phi, lam) in enumerate(zip(eigenvectors, eigenvalues)):
        std = np.sqrt(lam)
        
        # Mean + mode
        plus = (mu + n_std * std * phi).reshape(-1, 3)
        save_point_cloud(plus, os.path.join(output_dir, f"mode_{i+1:02d}_plus.ply"))
        
        # Mean - mode
        minus = (mu - n_std * std * phi).reshape(-1, 3)
        save_point_cloud(minus, os.path.join(output_dir, f"mode_{i+1:02d}_minus.ply"))
    
    print(f"  Saved mode visualizations to {output_dir}")


# =============================================================================
# SSM FITTING
# =============================================================================

def fit_ssm_to_partial(mu: np.ndarray,
                       eigenvectors: np.ndarray,
                       eigenvalues: np.ndarray,
                       worn_points: np.ndarray,
                       observed_mask: np.ndarray,
                       regularization: float = 1.0,
                       use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit SSM to partial shape (worn tooth with missing vertices).
    
    Solves: min_b ||(μ + Φᵀb)_Ω - y_Ω||² + Σ bᵢ²/λᵢ
    
    where Ω is the set of observed (non-removed) vertices.
    
    Args:
        mu: Mean shape (3N,)
        eigenvectors: Principal modes (K, 3N)
        eigenvalues: Variances per mode (K,)
        worn_points: Worn tooth points (N, 3)
        observed_mask: Boolean mask (N,) - True = observed (not removed)
        regularization: Tikhonov regularization strength
        use_gpu: Use GPU if available
        
    Returns:
        Tuple of (reconstructed_points (N, 3), coefficients (K,))
    """
    K = len(eigenvalues)
    N = len(observed_mask)
    
    # Expand mask to 3N (x, y, z for each point)
    Omega = np.repeat(observed_mask, 3)
    n_observed = Omega.sum()
    
    # Extract observed portions
    mu_obs = mu[Omega]
    phi_obs = eigenvectors[:, Omega].T  # (n_observed, K)
    y_obs = worn_points.flatten()[Omega]
    
    # Build linear system: A @ b = y
    A = phi_obs  # (n_observed, K)
    y = y_obs - mu_obs  # (n_observed,)
    
    # Tikhonov regularization matrix
    reg_diag = regularization / eigenvalues  # (K,)
    
    if use_gpu and CUPY_AVAILABLE:
        # GPU solve
        A_gpu = cp.asarray(A)
        y_gpu = cp.asarray(y)
        reg_gpu = cp.diag(cp.asarray(reg_diag))
        
        AtA = A_gpu.T @ A_gpu + reg_gpu
        Aty = A_gpu.T @ y_gpu
        
        b = cp.asnumpy(cp.linalg.solve(AtA, Aty))
        
    elif use_gpu and TORCH_AVAILABLE:
        # PyTorch GPU solve
        A_torch = torch.tensor(A, device='cuda', dtype=torch.float64)
        y_torch = torch.tensor(y, device='cuda', dtype=torch.float64)
        reg_torch = torch.diag(torch.tensor(reg_diag, device='cuda', dtype=torch.float64))
        
        AtA = A_torch.T @ A_torch + reg_torch
        Aty = A_torch.T @ y_torch
        
        b = torch.linalg.solve(AtA, Aty).cpu().numpy()
        
    else:
        # CPU solve
        AtA = A.T @ A + np.diag(reg_diag)
        Aty = A.T @ y
        b = np.linalg.solve(AtA, Aty)
    
    sigma = np.sqrt(np.maximum(eigenvalues, 1e-12))
    b = np.clip(b, -4.0 * sigma, 4.0 * sigma)
    
    reconstructed_flat = mu + eigenvectors.T @ b  # (3N,)
    reconstructed = reconstructed_flat.reshape(-1, 3)
    
    return reconstructed, b


def refine_reconstruction_nonrigid(reconstructed: np.ndarray,
                                   worn_corr: np.ndarray,
                                   removed_mask: np.ndarray,
                                   k: int = 20) -> Tuple[np.ndarray, Dict]:
    """
    Non-rigid refinement: warp SSM reconstruction to exactly match observed
    regions and smoothly extrapolate the deformation into missing regions.

    At observed points the refined cloud equals the worn tooth exactly (0 error).
    At missing points a KNN inverse-distance-weighted displacement field
    carries the local shape correction smoothly into the gap.

    Returns (refined_points, refinement_metrics).
    """
    from sklearn.neighbors import NearestNeighbors

    observed = ~removed_mask
    obs_src = reconstructed[observed]
    obs_disp = worn_corr[observed] - obs_src
    obs_disp_norms = np.linalg.norm(obs_disp, axis=1)

    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(obs_src)
    distances, indices = nn.kneighbors(reconstructed[removed_mask])

    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    missing_disp = (weights[:, :, None] * obs_disp[indices]).sum(axis=1)
    missing_disp_norms = np.linalg.norm(missing_disp, axis=1)

    refined = reconstructed.copy()
    refined[observed] = worn_corr[observed]
    refined[removed_mask] += missing_disp

    # Self-consistency: how close is the final cloud to the worn input at observed pts?
    final_obs_err = np.linalg.norm(refined[observed] - worn_corr[observed], axis=1)
    all_err = np.linalg.norm(refined - worn_corr, axis=1)

    metrics = {
        "ssm_obs_rmse": float(np.sqrt(np.mean(obs_disp_norms ** 2))),
        "ssm_obs_mae": float(np.mean(obs_disp_norms)),
        "ssm_obs_max_error": float(np.max(obs_disp_norms)),
        "missing_avg_displacement": float(np.mean(missing_disp_norms)),
        "missing_max_displacement": float(np.max(missing_disp_norms)) if len(missing_disp_norms) > 0 else 0.0,
        "final_obs_rmse": float(np.sqrt(np.mean(final_obs_err ** 2))),
        "final_all_rmse": float(np.sqrt(np.mean(all_err ** 2))),
        "n_observed": int(observed.sum()),
        "n_missing": int(removed_mask.sum()),
    }

    print(f"    Non-rigid refinement:")
    print(f"      SSM fit error at observed pts: RMSE={metrics['ssm_obs_rmse']:.5f}, "
          f"MAE={metrics['ssm_obs_mae']:.5f}, max={metrics['ssm_obs_max_error']:.5f}")
    print(f"      Missing region displacement:   avg={metrics['missing_avg_displacement']:.5f}, "
          f"max={metrics['missing_max_displacement']:.5f}")
    print(f"      Final self-consistency:         obs RMSE={metrics['final_obs_rmse']:.8f} "
          f"(should be ~0)")

    return refined, metrics


def compute_geometric_comparison(recon_raw: np.ndarray,
                                 worn_raw: np.ndarray,
                                 scale: float) -> Dict:
    """
    Geometric comparison between reconstruction (in raw input space)
    and the original raw worn tooth mesh.
    
    All distance metrics are returned in mm (multiplied by scale if needed,
    but here both inputs are already in mm).

    Metrics:
      - Chamfer distance (symmetric mean NN distance)
      - Hausdorff distance (symmetric worst-case NN distance)
      - RMSE / MAE of NN distances (both directions)
      - R² (variance explained): how much of the worn tooth's spatial
        variance is captured by the reconstruction
      - Coverage: fraction of worn points within threshold of reconstruction
    """
    from scipy.spatial import cKDTree

    tree_recon = cKDTree(recon_raw)
    tree_worn = cKDTree(worn_raw)

    d_worn2recon, _ = tree_recon.query(worn_raw)
    d_recon2worn, _ = tree_worn.query(recon_raw)

    chamfer = float((np.mean(d_worn2recon) + np.mean(d_recon2worn)) / 2.0)
    hausdorff = float(max(np.max(d_worn2recon), np.max(d_recon2worn)))

    rmse_worn2recon = float(np.sqrt(np.mean(d_worn2recon ** 2)))
    mae_worn2recon = float(np.mean(d_worn2recon))
    rmse_recon2worn = float(np.sqrt(np.mean(d_recon2worn ** 2)))
    mae_recon2worn = float(np.mean(d_recon2worn))

    # R²: how much of the worn tooth's spatial variance is "explained"
    # by the reconstruction. SS_tot = variance of worn points around their
    # centroid. SS_res = mean squared NN distance from worn to reconstruction.
    worn_centered = worn_raw - worn_raw.mean(axis=0)
    ss_tot = float(np.sum(worn_centered ** 2))
    ss_res = float(np.sum(d_worn2recon ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Coverage: fraction of worn mesh points within various thresholds
    nn_worn = cKDTree(worn_raw)
    worn_spacing, _ = nn_worn.query(worn_raw, k=2)
    median_spacing = float(np.median(worn_spacing[:, 1]))

    cov_1x = float(np.mean(d_worn2recon < median_spacing * 1.0))
    cov_2x = float(np.mean(d_worn2recon < median_spacing * 2.0))
    cov_5x = float(np.mean(d_worn2recon < median_spacing * 5.0))

    m = {
        "chamfer_mm": chamfer,
        "hausdorff_mm": hausdorff,
        "rmse_worn_to_recon_mm": rmse_worn2recon,
        "mae_worn_to_recon_mm": mae_worn2recon,
        "rmse_recon_to_worn_mm": rmse_recon2worn,
        "mae_recon_to_worn_mm": mae_recon2worn,
        "r_squared": r_squared,
        "variance_explained_pct": r_squared * 100.0,
        "median_spacing_mm": median_spacing,
        "coverage_1x_spacing": cov_1x,
        "coverage_2x_spacing": cov_2x,
        "coverage_5x_spacing": cov_5x,
        "n_worn_pts": len(worn_raw),
        "n_recon_pts": len(recon_raw),
    }
    return m


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_reconstruction(reconstructed: np.ndarray,
                           ground_truth: np.ndarray,
                           removed_mask: np.ndarray) -> Dict:
    """
    Compute reconstruction accuracy metrics.
    
    Args:
        reconstructed: Reconstructed points (N, 3)
        ground_truth: Ground truth points (N, 3)
        removed_mask: Boolean mask (N,) - True = was removed (missing)
        
    Returns:
        Dictionary of metrics
    """
    # Point-wise differences
    diff = reconstructed - ground_truth
    diff_norms = np.linalg.norm(diff, axis=1)
    
    # Split by observed vs missing
    missing_idx = np.where(removed_mask)[0]
    observed_idx = np.where(~removed_mask)[0]
    
    # RMSE calculations
    rmse_all = np.sqrt(np.mean(diff_norms ** 2))
    rmse_missing = np.sqrt(np.mean(diff_norms[missing_idx] ** 2)) if len(missing_idx) > 0 else 0.0
    rmse_observed = np.sqrt(np.mean(diff_norms[observed_idx] ** 2)) if len(observed_idx) > 0 else 0.0
    
    # Maximum error
    max_error_all = np.max(diff_norms)
    max_error_missing = np.max(diff_norms[missing_idx]) if len(missing_idx) > 0 else 0.0
    
    # Hausdorff distance
    hausdorff_fwd = directed_hausdorff(reconstructed, ground_truth)[0]
    hausdorff_bwd = directed_hausdorff(ground_truth, reconstructed)[0]
    hausdorff = max(hausdorff_fwd, hausdorff_bwd)
    
    # Mean absolute error on missing region
    mae_missing = np.mean(diff_norms[missing_idx]) if len(missing_idx) > 0 else 0.0
    
    return {
        "rmse_all": float(rmse_all),
        "rmse_missing_region": float(rmse_missing),
        "rmse_observed_region": float(rmse_observed),
        "max_error_all": float(max_error_all),
        "max_error_missing": float(max_error_missing),
        "mae_missing": float(mae_missing),
        "hausdorff": float(hausdorff),
        "n_total_points": len(removed_mask),
        "n_missing_points": int(len(missing_idx)),
        "n_observed_points": int(len(observed_idx)),
        "missing_fraction": float(len(missing_idx) / len(removed_mask))
    }


def estimate_missing_mask_from_mean(worn_points: np.ndarray,
                                    mean_points: np.ndarray,
                                    missing_fraction: float = 0.15) -> np.ndarray:
    """
    Estimate missing (worn) region without ground truth.
    
    Heuristic: in normalized corresponded space, wear typically reduces occlusal
    height. Mark points with largest positive z-loss (mean_z - worn_z).
    """
    missing_fraction = float(np.clip(missing_fraction, 0.01, 0.9))
    z_loss = mean_points[:, 2] - worn_points[:, 2]
    
    # Prefer true wear-like direction (positive z-loss). Fallback to absolute
    # loss if there are too few positive candidates.
    positive = z_loss > 0
    target_n = max(1, int(round(len(worn_points) * missing_fraction)))
    
    if np.sum(positive) >= target_n:
        pos_idx = np.where(positive)[0]
        pos_loss = z_loss[pos_idx]
        top_local = np.argpartition(pos_loss, -target_n)[-target_n:]
        chosen = pos_idx[top_local]
    else:
        abs_loss = np.abs(z_loss)
        chosen = np.argpartition(abs_loss, -target_n)[-target_n:]
    
    mask = np.zeros(len(worn_points), dtype=bool)
    mask[chosen] = True
    return mask


def inverse_correspondence_transform(points: np.ndarray,
                                     normalization_path: str,
                                     icp_transform_path: str) -> np.ndarray:
    """
    Map points from correspondence space back to raw input space.
    
    Forward transform in correspondence pipeline:
      raw -> center -> scale -> rotate(PCA) -> ICP
    Inverse here:
      inv(ICP) -> inv(rotate) -> inv(scale) -> add centroid
    """
    with open(normalization_path, "r") as f:
        norm_data = json.load(f)
    
    centroid = np.array(norm_data["centroid"], dtype=np.float64)
    scale = float(norm_data["scale"])
    pca_rotation = np.array(norm_data["pca_rotation"], dtype=np.float64)
    
    restored = points.copy()
    
    if os.path.exists(icp_transform_path):
        icp = np.load(icp_transform_path)
        R = icp[:3, :3]
        t = icp[:3, 3]
        restored = (restored - t) @ R
    
    # Inverse PCA: forward was aligned = scaled @ Vt.T, so inverse is scaled = aligned @ Vt
    restored = restored @ pca_rotation
    restored = restored * scale
    restored = restored + centroid
    
    return restored


def _refine_icp_to_worn(reconstructed_raw: np.ndarray,
                        worn_raw: np.ndarray,
                        median_spacing: float,
                        max_iter: int = 30) -> np.ndarray:
    """
    Fine ICP alignment of SSM reconstruction onto the worn mesh.
    
    Pure numpy/scipy implementation (avoids open3d ICP segfaults).
    Uses Procrustes (SVD) per iteration on the overlapping region,
    then applies the accumulated rigid transform to ALL SSM points.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial import cKDTree
    
    tree = cKDTree(worn_raw)
    aligned = reconstructed_raw.copy()
    
    d_init, _ = tree.query(aligned)
    before_median = np.median(d_init)
    
    inlier_thresh = median_spacing * 10.0
    
    for _ in range(max_iter):
        d, idx = tree.query(aligned)
        inlier = d < inlier_thresh
        if inlier.sum() < 100:
            break
        
        src = aligned[inlier]
        tgt = worn_raw[idx[inlier]]
        
        src_c = src.mean(axis=0)
        tgt_c = tgt.mean(axis=0)
        H = (src - src_c).T @ (tgt - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_c - R @ src_c
        
        aligned = (aligned @ R.T) + t
        
        if np.linalg.norm(t) < median_spacing * 0.01:
            break
    
    d_after, _ = tree.query(aligned)
    after_median = np.median(d_after)
    print(f"      ICP refinement: median dist {before_median:.4f} → {after_median:.4f} mm")
    
    return aligned


def fuse_reconstructed_with_worn_input(reconstructed_corr: np.ndarray,
                                       worn_corr: np.ndarray,
                                       worn_raw: np.ndarray,
                                       normalization_path: str,
                                       icp_transform_path: str,
                                       knn: int = 3,
                                       removed_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Geometric merge with ICP refinement and smooth boundary blending.
    
    Steps:
      1. Inverse-transform SSM reconstruction to raw input space
      2. ICP-refine onto the worn mesh to remove residual rigid offset
      3. Three-zone fusion:
           Close  (d < inner): skip — real data covers this
           Middle (inner < d < outer): blend toward worn surface
           Far    (d > outer): pure SSM infill in the gap
    """
    from sklearn.neighbors import NearestNeighbors
    
    reconstructed_raw = inverse_correspondence_transform(
        reconstructed_corr, normalization_path, icp_transform_path
    )
    
    nn_worn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(worn_raw)
    worn_nn_dist, _ = nn_worn.kneighbors(worn_raw)
    median_spacing = np.median(worn_nn_dist[:, 1])
    
    reconstructed_raw = _refine_icp_to_worn(reconstructed_raw, worn_raw, median_spacing)
    
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(worn_raw)
    distances, indices = nn.kneighbors(reconstructed_raw)
    distances = distances.ravel()
    indices = indices.ravel()
    
    inner_thresh = median_spacing * 2.0
    outer_thresh = median_spacing * 6.0
    
    is_close = distances <= inner_thresh
    is_transition = (distances > inner_thresh) & (distances <= outer_thresh)
    is_gap = distances > outer_thresh
    
    transition_pts = reconstructed_raw[is_transition]
    transition_nearest = worn_raw[indices[is_transition]]
    transition_dist = distances[is_transition]
    t = (transition_dist - inner_thresh) / max(outer_thresh - inner_thresh, 1e-9)
    blended_transition = (1.0 - t[:, None]) * transition_nearest + t[:, None] * transition_pts
    
    gap_pts = reconstructed_raw[is_gap]
    
    parts = [worn_raw]
    if len(blended_transition) > 0:
        parts.append(blended_transition)
    if len(gap_pts) > 0:
        parts.append(gap_pts)
    fused = np.vstack(parts)
    
    print(f"      Geometric merge: {len(worn_raw)} worn + "
          f"{len(blended_transition)} transition + {len(gap_pts)} infill "
          f"= {len(fused)} total  "
          f"(spacing={median_spacing:.4f}, inner={inner_thresh:.4f}, "
          f"outer={outer_thresh:.4f} mm)")
    
    return fused, reconstructed_raw


# =============================================================================
# PIPELINE
# =============================================================================

def find_mask_for_worn(worn_dir: str, wear_name: str, artificial_wear_dir: str) -> Optional[str]:
    """
    Find the removal mask file for a worn tooth variant.
    
    Args:
        worn_dir: Directory name like "tooth_01_wear_mild_c0_spherical"
        wear_name: Wear variant name extracted from dir
        artificial_wear_dir: Base artificial wear directory
        
    Returns:
        Path to mask file or None if not found
    """
    # Parse tooth number and wear type from directory name
    # Format: tooth_XX_wear_YYY -> tooth_XX, wear_YYY
    parts = worn_dir.split("_")
    
    # Find tooth number
    tooth_idx = None
    wear_type = None
    for i, part in enumerate(parts):
        if part == "tooth" and i + 1 < len(parts):
            tooth_idx = parts[i + 1]
        if part == "wear" and i + 1 < len(parts):
            # Everything after "wear_" is the wear type
            wear_type = "_".join(parts[i:])
            break
    
    if tooth_idx is None or wear_type is None:
        return None
    
    # Construct mask filename
    # Mask files are named: removed_mask_mild_c0_spherical.npy (without "wear_" prefix)
    mask_name = wear_type.replace("wear_", "removed_mask_") + ".npy"
    
    tooth_dir = f"tooth_{tooth_idx}"
    mask_path = os.path.join(artificial_wear_dir, tooth_dir, mask_name)
    
    if os.path.exists(mask_path):
        return mask_path
    
    # Try alternative naming patterns
    # Sometimes mask might have slightly different name
    possible_masks = glob(os.path.join(artificial_wear_dir, tooth_dir, "removed_mask_*.npy"))
    
    # Try to match by wear type substring
    for mask in possible_masks:
        mask_basename = os.path.basename(mask)
        # Extract the part after "removed_mask_"
        mask_type = mask_basename.replace("removed_mask_", "").replace(".npy", "")
        if mask_type in wear_type or wear_type.replace("wear_", "") == mask_type:
            return mask
    
    return None


def run_reconstruction_pipeline(correspondence_dir: str,
                                artificial_wear_dir: str,
                                output_dir: str,
                                config: ReconstructionConfig) -> None:
    """
    Run the full reconstruction pipeline.
    
    Args:
        correspondence_dir: Directory containing correspondence outputs
        artificial_wear_dir: Directory containing artificial wear outputs (for masks)
        output_dir: Output directory
        config: Pipeline configuration
    """
    print("=" * 60)
    print("SSM Reconstruction Pipeline")
    print("=" * 60)
    print(f"Correspondence directory: {correspondence_dir}")
    print(f"Artificial wear directory: {artificial_wear_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GPU acceleration: {CUPY_AVAILABLE or TORCH_AVAILABLE}")
    print(f"Skip evaluation: {config.skip_eval}")
    print()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    ssm_dir = os.path.join(output_dir, "ssm")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(ssm_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Step 1: Find good teeth correspondences
    good_teeth_dir = os.path.join(correspondence_dir, "good_teeth")
    good_corresponded = sorted(glob(os.path.join(good_teeth_dir, "tooth_*", "corresponded.ply")))
    
    if len(good_corresponded) == 0:
        print("ERROR: No corresponded good teeth found!")
        print(f"  Expected at: {good_teeth_dir}/tooth_*/corresponded.ply")
        return
    
    # Map paths to tooth IDs so we can perform held-out splits reproducibly
    good_with_tooth = []
    for path in good_corresponded:
        tooth_dir = os.path.basename(os.path.dirname(path))  # tooth_XX
        parts = tooth_dir.split("_")
        if len(parts) == 2 and parts[0] == "tooth":
            good_with_tooth.append((parts[1], path))
        else:
            print(f"  [WARN] Could not parse tooth ID from path: {path}")
    
    if len(good_with_tooth) == 0:
        print("ERROR: Could not parse any training tooth IDs from correspondences")
        return
    
    all_training_teeth = sorted([tooth_id for tooth_id, _ in good_with_tooth])
    if config.test_tooth:
        good_with_tooth = [(tooth_id, path) for tooth_id, path in good_with_tooth
                           if tooth_id != config.test_tooth]
        if len(good_with_tooth) == 0:
            print(f"ERROR: No training teeth remain after excluding test tooth {config.test_tooth}")
            return
        print(f"Held-out test tooth: {config.test_tooth}")
        print(f"Training teeth ({len(good_with_tooth)}): "
              f"{', '.join(sorted([tooth_id for tooth_id, _ in good_with_tooth]))}")
    else:
        print(f"Found {len(good_with_tooth)} good teeth correspondences (no held-out test tooth)")
    
    good_corresponded = [path for _, path in good_with_tooth]
    
    # Step 2: Build SSM
    mu, phi, lambdas, var_explained = build_ssm(
        good_corresponded,
        n_components=config.n_components,
        variance_threshold=config.variance_threshold,
        use_gpu=config.use_gpu
    )
    
    # Save SSM
    np.save(os.path.join(ssm_dir, "mean_shape.npy"), mu)
    np.save(os.path.join(ssm_dir, "eigenvectors.npy"), phi)
    np.save(os.path.join(ssm_dir, "eigenvalues.npy"), lambdas)
    save_point_cloud(mu.reshape(-1, 3), os.path.join(ssm_dir, "mean_shape.ply"))
    
    ssm_meta = {
        "n_training_samples": len(good_corresponded),
        "n_points_per_sample": len(mu) // 3,
        "n_components": len(lambdas),
        "variance_explained": float(var_explained),
        "eigenvalues": lambdas.tolist(),
        "cumulative_variance": (np.cumsum(lambdas) / lambdas.sum()).tolist(),
        "test_tooth": config.test_tooth,
        "training_teeth": sorted([tooth_id for tooth_id, _ in good_with_tooth]),
        "available_good_teeth": all_training_teeth
    }
    with open(os.path.join(ssm_dir, "ssm_metadata.json"), 'w') as f:
        json.dump(ssm_meta, f, indent=2)
    
    # Save mode visualizations
    if config.save_all_modes:
        modes_dir = os.path.join(ssm_dir, "modes")
        visualize_modes(mu, phi, lambdas, modes_dir)
    
    print(f"\nSSM saved to {ssm_dir}")
    print(f"  Components: {len(lambdas)}")
    print(f"  Variance explained: {var_explained*100:.1f}%")
    
    # Step 3: Find worn teeth to reconstruct
    worn_teeth_dir = os.path.join(correspondence_dir, "artificial_worn")
    
    # Get all worn variants (not originals)
    all_worn_dirs = [d for d in os.listdir(worn_teeth_dir)
                     if os.path.isdir(os.path.join(worn_teeth_dir, d))
                     and "_wear_" in d]
    if config.test_tooth:
        prefix = f"tooth_{config.test_tooth}_wear_"
        all_worn_dirs = [d for d in all_worn_dirs if d.startswith(prefix)]
    
    # Get originals (ground truth)
    original_dirs = [d for d in os.listdir(worn_teeth_dir)
                    if os.path.isdir(os.path.join(worn_teeth_dir, d))
                    and d.endswith("_original")]
    
    print(f"\nFound {len(all_worn_dirs)} worn variants to reconstruct")
    print(f"Found {len(original_dirs)} originals (ground truth)")
    
    # Build mapping from tooth number to original
    original_map = {}
    for orig_dir in original_dirs:
        # Extract tooth number: tooth_01_original -> 01
        parts = orig_dir.split("_")
        if len(parts) >= 2:
            tooth_num = parts[1]  # "01"
            orig_path = os.path.join(worn_teeth_dir, orig_dir, "corresponded.ply")
            if os.path.exists(orig_path):
                original_map[tooth_num] = orig_path
    
    # Step 4: Reconstruct each worn tooth
    print("\nReconstructing worn teeth...")
    results = []
    
    for worn_dir in sorted(all_worn_dirs):
        worn_path = os.path.join(worn_teeth_dir, worn_dir, "corresponded.ply")
        
        if not os.path.exists(worn_path):
            print(f"  [SKIP] {worn_dir} - no corresponded.ply")
            continue
        
        # Extract tooth number
        parts = worn_dir.split("_")
        if len(parts) < 2:
            continue
        tooth_num = parts[1]
        
        ground_truth_path = original_map.get(tooth_num)
        if not config.skip_eval and ground_truth_path is None:
            print(f"  [SKIP] {worn_dir} - no ground truth for tooth {tooth_num}")
            continue
        
        print(f"  Processing {worn_dir}...")
        
        try:
            # Load data
            worn_points = load_point_cloud(worn_path)
            ground_truth = load_point_cloud(ground_truth_path) if ground_truth_path else None
            
            # Check dimensions match
            n_worn = len(worn_points)
            n_ssm = len(mu) // 3
            
            if config.skip_eval:
                if n_worn != n_ssm:
                    print(f"    [WARN] Dimension mismatch: worn={n_worn}, ssm={n_ssm}")
                    print(f"    [SKIP] Cannot proceed with mismatched dimensions")
                    continue
            else:
                n_gt = len(ground_truth)
                if not (n_worn == n_gt == n_ssm):
                    print(f"    [WARN] Dimension mismatch: worn={n_worn}, gt={n_gt}, ssm={n_ssm}")
                    if n_worn != n_ssm:
                        print(f"    [SKIP] Cannot proceed with mismatched dimensions")
                        continue
            
            # Load original mask from wear simulation and map to corresponded points
            # Extract wear type from directory name
            worn_name_parts = worn_dir.split("_wear_")
            if len(worn_name_parts) == 2:
                wear_type = worn_name_parts[1]
            else:
                wear_type = "_".join(worn_dir.split("_")[2:])
            
            original_mesh_path = os.path.join(
                artificial_wear_dir, f"tooth_{tooth_num}", "original.ply"
            )
            original_mask_path = os.path.join(
                artificial_wear_dir, f"tooth_{tooth_num}", f"removed_mask_{wear_type}.npy"
            )
            
            removed_mask = None
            if (not config.skip_eval
                and os.path.exists(original_mask_path)
                and os.path.exists(original_mesh_path)
                and ground_truth is not None):
                original_mask = np.load(original_mask_path).astype(bool)
                original_mesh_vertices = load_point_cloud(original_mesh_path)
                
                # Normalize both point clouds to unit sphere for coordinate-system-invariant mapping
                def normalize_to_unit_sphere(pts):
                    centroid = pts.mean(axis=0)
                    centered = pts - centroid
                    scale = np.max(np.linalg.norm(centered, axis=1))
                    return centered / scale if scale > 0 else centered
                
                orig_normalized = normalize_to_unit_sphere(original_mesh_vertices)
                gt_normalized = normalize_to_unit_sphere(ground_truth)
                
                # Map from corresponded ground truth to original mesh using normalized coordinates
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(orig_normalized)
                distances, indices = nn.kneighbors(gt_normalized)
                indices = indices.flatten()
                
                # Transfer mask values
                removed_mask = original_mask[indices]
                
                orig_pct = 100 * original_mask.sum() / len(original_mask)
                mapped_pct = 100 * removed_mask.sum() / len(removed_mask)
                print(f"    Original mask: {orig_pct:.1f}% -> Mapped: {mapped_pct:.1f}%")
            
            if removed_mask is None and not config.skip_eval:
                # Fallback: compute mask by comparing corresponded points
                print(f"    [WARN] Original mask not found, computing from point distances")
                point_distances = np.linalg.norm(worn_points - ground_truth, axis=1)
                scale = np.max(np.linalg.norm(ground_truth - ground_truth.mean(axis=0), axis=1))
                threshold = 0.01 * scale
                removed_mask = point_distances > threshold
            
            if removed_mask is None and config.skip_eval:
                mean_points = mu.reshape(-1, 3)
                removed_mask = estimate_missing_mask_from_mean(
                    worn_points, mean_points, config.proxy_missing_fraction
                )
                print(f"    Proxy mask from mean-shape z-loss "
                      f"({100*config.proxy_missing_fraction:.1f}% target)")
            
            n_removed = removed_mask.sum()
            print(f"    Computed mask: {n_removed}/{n_worn} points removed ({100*n_removed/n_worn:.1f}%)")
            
            # Fit SSM
            observed_mask = ~removed_mask
            reconstructed, coefficients = fit_ssm_to_partial(
                mu, phi, lambdas,
                worn_points, observed_mask,
                regularization=config.regularization,
                use_gpu=config.use_gpu
            )
            
            # Non-rigid refinement: warp SSM output onto observed regions,
            # smoothly extrapolate into the gap.
            reconstructed, refinement_metrics = refine_reconstruction_nonrigid(
                reconstructed, worn_points, removed_mask, k=20
            )
            
            # Evaluate (only when ground truth is available)
            metrics = None
            if not config.skip_eval and ground_truth is not None:
                metrics = evaluate_reconstruction(reconstructed, ground_truth, removed_mask)
            
            # Save outputs
            result_dir = os.path.join(recon_dir, worn_dir)
            os.makedirs(result_dir, exist_ok=True)
            
            save_point_cloud(worn_points, os.path.join(result_dir, "worn_input.ply"))
            save_point_cloud(reconstructed, os.path.join(result_dir, "reconstructed.ply"))
            if ground_truth is not None:
                save_point_cloud(ground_truth, os.path.join(result_dir, "ground_truth.ply"))
            np.save(os.path.join(result_dir, "coefficients.npy"), coefficients)
            np.save(os.path.join(result_dir, "removed_mask.npy"), removed_mask)
            
            # Inverse-transform refined reconstruction to raw input space
            corr_case_dir = os.path.join(worn_teeth_dir, worn_dir)
            normalization_path = os.path.join(corr_case_dir, "normalization.json")
            icp_transform_path = os.path.join(corr_case_dir, "icp_transform.npy")
            
            geo_metrics = None
            if os.path.exists(normalization_path):
                try:
                    with open(normalization_path, "r") as f:
                        norm_scale = float(json.load(f)["scale"])
                    
                    reconstructed_raw = inverse_correspondence_transform(
                        reconstructed, normalization_path, icp_transform_path
                    )
                    save_point_cloud(
                        reconstructed_raw,
                        os.path.join(result_dir, "reconstructed_in_input_space.ply")
                    )
                    print(f"    Saved reconstructed_in_input_space.ply ({len(reconstructed_raw)} pts)")
                    
                    # Geometric comparison with raw worn tooth
                    raw_worn_path = os.path.join(
                        artificial_wear_dir, f"tooth_{tooth_num}", f"wear_{wear_type}.ply"
                    )
                    if os.path.exists(raw_worn_path):
                        worn_raw = load_point_cloud(raw_worn_path)
                        geo_metrics = compute_geometric_comparison(
                            reconstructed_raw, worn_raw, norm_scale
                        )
                        print(f"    Geometric comparison (recon vs raw worn):")
                        print(f"      R² (variance explained): {geo_metrics['variance_explained_pct']:.2f}%")
                        print(f"      Chamfer dist:  {geo_metrics['chamfer_mm']:.4f} mm")
                        print(f"      Hausdorff:     {geo_metrics['hausdorff_mm']:.4f} mm")
                        print(f"      RMSE worn→rec: {geo_metrics['rmse_worn_to_recon_mm']:.4f} mm")
                        print(f"      MAE  worn→rec: {geo_metrics['mae_worn_to_recon_mm']:.4f} mm")
                        print(f"      Coverage (2x spacing): {geo_metrics['coverage_2x_spacing']*100:.1f}%")
                except Exception as inv_err:
                    print(f"    [WARN] Inverse transform failed: {inv_err}")
                    reconstructed_raw = None
            else:
                reconstructed_raw = None
            
            # Optional legacy merge path (kept for backward compatibility)
            if config.real_merge_with_input and reconstructed_raw is not None:
                raw_worn_path = os.path.join(
                    artificial_wear_dir, f"tooth_{tooth_num}", f"wear_{wear_type}.ply"
                )
                if os.path.exists(raw_worn_path):
                    try:
                        worn_raw = load_point_cloud(raw_worn_path)
                        fused_raw, _ = fuse_reconstructed_with_worn_input(
                            reconstructed_corr=reconstructed,
                            worn_corr=worn_points,
                            worn_raw=worn_raw,
                            normalization_path=normalization_path,
                            icp_transform_path=icp_transform_path,
                            knn=config.real_merge_knn,
                            removed_mask=removed_mask
                        )
                        save_point_cloud(
                            fused_raw,
                            os.path.join(result_dir, "fused_with_worn_input.ply")
                        )
                        print(f"    Legacy fusion saved "
                              f"(fused_with_worn_input.ply, {len(fused_raw)} points)")
                    except Exception as fusion_err:
                        print(f"    [WARN] Legacy fusion failed: {fusion_err}")
            
            # Smooth mesh from the refined reconstruction in input space
            if reconstructed_raw is not None:
                try:
                    mesh_obj = point_cloud_to_mesh(reconstructed_raw)
                    mesh_path = os.path.join(result_dir, "reconstructed_smooth.ply")
                    save_mesh(mesh_obj, mesh_path)
                    n_verts = len(mesh_obj.vertices)
                    n_tris = len(mesh_obj.faces)
                    print(f"    Smooth mesh: reconstructed_smooth.ply "
                          f"({n_verts} verts, {n_tris} tris)")
                except Exception as mesh_err:
                    print(f"    [WARN] Smooth mesh failed: {mesh_err}")
            
            eval_data = {
                "source_worn": worn_dir,
                "source_original": f"tooth_{tooth_num}_original" if ground_truth_path else None,
                "mask_computed": True,
                "skip_evaluation": bool(config.skip_eval),
                "n_points_removed": int(removed_mask.sum()),
                "n_components_used": len(lambdas),
                "coefficients": coefficients.tolist(),
                "regularization": config.regularization,
                "evaluation": metrics,
                "refinement": refinement_metrics,
                "geometric_comparison": geo_metrics
            }
            with open(os.path.join(result_dir, "evaluation.json"), 'w') as f:
                json.dump(eval_data, f, indent=2)
            
            if metrics is not None:
                results.append({
                    "name": worn_dir,
                    "rmse_missing": metrics["rmse_missing_region"],
                    "missing_fraction": metrics["missing_fraction"]
                })
                print(f"    RMSE (missing): {metrics['rmse_missing_region']:.4f}, "
                      f"Missing: {metrics['missing_fraction']*100:.1f}%")
            else:
                results.append({
                    "name": worn_dir,
                    "rmse_missing": None,
                    "missing_fraction": float(n_removed / n_worn),
                    "skip_evaluation": True
                })
                print(f"    Evaluation skipped (no ground truth). "
                      f"Missing proxy: {100*n_removed/n_worn:.1f}%")
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Generate summary report
    print("\n" + "=" * 60)
    print("Reconstruction Summary")
    print("=" * 60)
    
    if len(results) > 0 and not config.skip_eval:
        rmse_values = [r["rmse_missing"] for r in results if r["rmse_missing"] is not None]
        
        print(f"Reconstructed {len(results)} worn teeth")
        print(f"RMSE on missing regions:")
        print(f"  Mean: {np.mean(rmse_values):.4f}")
        print(f"  Std:  {np.std(rmse_values):.4f}")
        print(f"  Min:  {np.min(rmse_values):.4f}")
        print(f"  Max:  {np.max(rmse_values):.4f}")
        
        # Save summary
        summary = {
            "n_reconstructed": len(results),
            "ssm_components": len(lambdas),
            "ssm_variance_explained": float(var_explained),
            "regularization": config.regularization,
            "rmse_missing_mean": float(np.mean(rmse_values)),
            "rmse_missing_std": float(np.std(rmse_values)),
            "rmse_missing_min": float(np.min(rmse_values)),
            "rmse_missing_max": float(np.max(rmse_values)),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(output_dir, "reconstruction_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Rank by accuracy
        print("\nTop 5 best reconstructions:")
        sorted_results = sorted(results, key=lambda x: x["rmse_missing"])
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {r['name']}: RMSE={r['rmse_missing']:.4f}")
        
        print("\nTop 5 worst reconstructions:")
        for i, r in enumerate(sorted_results[-5:]):
            print(f"  {i+1}. {r['name']}: RMSE={r['rmse_missing']:.4f}")
    elif len(results) > 0 and config.skip_eval:
        missing_fracs = [r["missing_fraction"] for r in results]
        print(f"Reconstructed {len(results)} worn teeth (evaluation skipped)")
        print(f"Proxy missing fraction:")
        print(f"  Mean: {100*np.mean(missing_fracs):.2f}%")
        print(f"  Std:  {100*np.std(missing_fracs):.2f}%")
        print(f"  Min:  {100*np.min(missing_fracs):.2f}%")
        print(f"  Max:  {100*np.max(missing_fracs):.2f}%")
        
        summary = {
            "n_reconstructed": len(results),
            "ssm_components": len(lambdas),
            "ssm_variance_explained": float(var_explained),
            "regularization": config.regularization,
            "skip_evaluation": True,
            "proxy_missing_fraction_mean": float(np.mean(missing_fracs)),
            "proxy_missing_fraction_std": float(np.std(missing_fracs)),
            "proxy_missing_fraction_min": float(np.min(missing_fracs)),
            "proxy_missing_fraction_max": float(np.max(missing_fracs)),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(output_dir, "reconstruction_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("No reconstructions completed!")
    
    print(f"\nOutput saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build SSM and reconstruct worn teeth"
    )
    parser.add_argument(
        "--correspondence-dir", "-c",
        type=str,
        default=None,
        help="Directory containing correspondence outputs"
    )
    parser.add_argument(
        "--artificial-wear", "-a",
        type=str,
        default=None,
        help="Directory containing artificial wear outputs (for masks)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--n-components", "-n",
        type=int,
        default=None,
        help="Number of PCA components (default: auto by variance)"
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.99,
        help="Variance threshold for auto component selection (default: 0.99)"
    )
    parser.add_argument(
        "--regularization", "-r",
        type=float,
        default=1.0,
        help="Tikhonov regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--test-tooth",
        type=str,
        default=None,
        help="Held-out tooth ID (e.g., 01 or tooth_01). Excluded from SSM training."
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip ground-truth evaluation (for real worn teeth without originals)"
    )
    parser.add_argument(
        "--proxy-missing-fraction",
        type=float,
        default=0.15,
        help="Proxy missing-region fraction used when --skip-eval is enabled (default: 0.15)"
    )
    parser.add_argument(
        "--real-merge-with-input",
        action="store_true",
        help="In skip-eval mode, inverse-transform SSM and fuse with raw worn input mesh"
    )
    parser.add_argument(
        "--real-merge-knn",
        type=int,
        default=3,
        help="KNN neighbors for real-worn fusion mapping (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    correspondence_dir = args.correspondence_dir or os.path.join(script_dir, "output", "correspondence")
    artificial_wear_dir = args.artificial_wear or os.path.join(project_dir, "artificial_wear", "output")
    output_dir = args.output or os.path.join(script_dir, "output")
    
    # Normalize optional held-out tooth ID
    test_tooth = None
    if args.test_tooth:
        test_value = args.test_tooth.strip().lower()
        if test_value.startswith("tooth_"):
            test_value = test_value.replace("tooth_", "", 1)
        if not test_value.isdigit():
            parser.error("--test-tooth must be numeric, e.g., 01 or tooth_01")
        test_tooth = f"{int(test_value):02d}"
    
    # Create config
    config = ReconstructionConfig(
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
        regularization=args.regularization,
        use_gpu=not args.no_gpu,
        test_tooth=test_tooth,
        skip_eval=args.skip_eval,
        proxy_missing_fraction=args.proxy_missing_fraction,
        real_merge_with_input=args.real_merge_with_input,
        real_merge_knn=args.real_merge_knn
    )
    
    run_reconstruction_pipeline(correspondence_dir, artificial_wear_dir, output_dir, config)


if __name__ == "__main__":
    main()
