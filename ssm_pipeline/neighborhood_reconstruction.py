#!/usr/bin/env python3
"""
Neighborhood SSM Reconstruction

For each worn tooth, this script:
1. Projects it into a PCA space fit on good teeth
2. Selects ALL good-teeth within the adaptive threshold, capped at max_neighbors (default 5)
3. Builds a local SSM from those neighbors (even if only 1 is found)
4. Reconstructs using that local SSM

Falls back to global SSM only if ZERO good teeth are within the threshold.

Usage:
    python neighborhood_reconstruction.py \
        --correspondence-dir output/correspondence_all_100k \
        --artificial-wear ../../all_worn_input \
        --output output/recon_neighborhood

    python neighborhood_reconstruction.py \
        --correspondence-dir output/correspondence_all_100k \
        --output output/recon_neighborhood \
        --max-neighbors 5 \
        --threshold-iqr-mult 2.0
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from reconstruction_pipeline import (
    build_ssm,
    compute_geometric_comparison,
    estimate_missing_mask_from_mean,
    fit_ssm_to_partial,
    inverse_correspondence_transform,
    load_point_cloud,
    point_cloud_to_mesh,
    refine_reconstruction_nonrigid,
    save_mesh,
    save_point_cloud,
    _refine_icp_to_worn,
    CUPY_AVAILABLE,
    TORCH_AVAILABLE,
)

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------------ #
# Neighbor selection
# ------------------------------------------------------------------ #

def compute_neighbor_threshold(scores_good: np.ndarray,
                               iqr_mult: float = 2.0) -> Tuple[float, Dict]:
    """Compute adaptive distance threshold from pairwise good-teeth distances.

    Threshold = median(pairwise) + iqr_mult * IQR(pairwise).

    Returns (threshold, info_dict).
    """
    pw = pdist(scores_good, metric="euclidean")
    median_pw = float(np.median(pw))
    q25 = float(np.percentile(pw, 25))
    q75 = float(np.percentile(pw, 75))
    iqr = q75 - q25
    threshold = median_pw + iqr_mult * iqr

    info = {
        "median_pairwise": median_pw,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "iqr_multiplier": iqr_mult,
        "threshold": threshold,
        "n_pairs": len(pw),
        "min_pairwise": float(np.min(pw)),
        "max_pairwise": float(np.max(pw)),
    }
    return threshold, info


def select_neighbors(score_worn: np.ndarray,
                     scores_good: np.ndarray,
                     threshold: float,
                     max_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Select good-teeth neighbors for a worn tooth in PCA space.

    Takes ALL good teeth within the threshold, capped at max_neighbors.
    No minimum requirement — even 1 neighbor is valid.
    Falls back to global SSM only if ZERO teeth are within the threshold.

    Returns (neighbor_indices, neighbor_distances, fallback_reason).
    fallback_reason is None when at least 1 neighbor is found, otherwise a string.
    """
    dists = np.linalg.norm(scores_good - score_worn, axis=1)
    sorted_idx = np.argsort(dists)
    sorted_dists = dists[sorted_idx]

    # Take all teeth within threshold, capped at max_neighbors
    within = sorted_dists < threshold
    n_within = int(within.sum())
    n_select = min(n_within, max_neighbors)

    fallback_reason = None
    if n_select == 0:
        # Truly beyond all good teeth — fall back to global SSM
        fallback_reason = (
            f"no neighbors within threshold ({threshold:.4f}); "
            f"nearest good tooth is at {sorted_dists[0]:.4f}"
        )
        # Still return the single nearest so metadata is informative
        indices = sorted_idx[:1]
        distances = sorted_dists[:1]
    else:
        indices = sorted_idx[:n_select]
        distances = sorted_dists[:n_select]

    return indices, distances, fallback_reason


# ------------------------------------------------------------------ #
# Local SSM builder
# ------------------------------------------------------------------ #

def build_neighborhood_ssm(X_good: np.ndarray,
                           neighbor_indices: np.ndarray,
                           variance_threshold: float = 0.95,
                           use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a PCA-based SSM from a subset of good teeth.

    The local mean is the average of the selected neighbors.

    Returns (mu, eigenvectors, eigenvalues, variance_explained) with the
    same shapes as reconstruction_pipeline.build_ssm.
    """
    X_local = X_good[neighbor_indices]  # (K, 3N)
    K = len(neighbor_indices)

    mu = X_local.mean(axis=0)  # (3N,)
    D = X_local - mu

    if use_gpu and CUPY_AVAILABLE:
        D_gpu = cp.asarray(D)
        _, S, Vt = cp.linalg.svd(D_gpu, full_matrices=False)
        S = cp.asnumpy(S)
        Vt = cp.asnumpy(Vt)
    elif use_gpu and TORCH_AVAILABLE:
        D_t = torch.tensor(D, device="cuda", dtype=torch.float64)
        _, S, Vt = torch.linalg.svd(D_t, full_matrices=False)
        S = S.cpu().numpy()
        Vt = Vt.cpu().numpy()
    else:
        _, S, Vt = np.linalg.svd(D, full_matrices=False)

    eigenvalues = (S ** 2) / max(K - 1, 1)
    total_var = eigenvalues.sum()
    if total_var < 1e-15:
        # Degenerate case: all neighbors are nearly identical
        return mu, np.zeros((1, len(mu))), np.array([1e-10]), 1.0

    cumulative_var = np.cumsum(eigenvalues) / total_var
    n_components = int(np.searchsorted(cumulative_var, variance_threshold) + 1)
    n_components = min(n_components, K - 1)
    n_components = max(n_components, 1)

    eigenvectors = Vt[:n_components]
    eigenvalues = eigenvalues[:n_components]
    variance_explained = float(cumulative_var[min(n_components - 1, len(cumulative_var) - 1)])

    return mu, eigenvectors, eigenvalues, variance_explained


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #

def run_neighborhood_reconstruction(
    correspondence_dir: str,
    output_dir: str,
    global_recon_dir: Optional[str],
    artificial_wear_dir: Optional[str],
    worn_teeth_list: Optional[List[str]],
    max_neighbors: int,
    threshold_iqr_mult: float,
    pca_variance: float,
    ssm_variance: float,
    proxy_missing_fraction: float,
    regularization: float,
    use_gpu: bool,
) -> None:
    print("=" * 60)
    print("Neighborhood SSM Reconstruction")
    print("=" * 60)
    print(f"  Correspondence dir : {correspondence_dir}")
    print(f"  Output dir         : {output_dir}")
    print(f"  Global recon dir   : {global_recon_dir}")
    print(f"  Artificial wear    : {artificial_wear_dir}")
    print(f"  Max neighbors      : {max_neighbors}")
    print(f"  Threshold IQR mult : {threshold_iqr_mult}")
    print(f"  PCA variance       : {pca_variance}")
    print(f"  SSM variance       : {ssm_variance}")
    print(f"  Missing fraction   : {proxy_missing_fraction}")
    print(f"  Regularization     : {regularization}")
    print(f"  GPU                : {use_gpu and (CUPY_AVAILABLE or TORCH_AVAILABLE)}")
    print()

    good_teeth_dir = os.path.join(correspondence_dir, "good_teeth")
    worn_teeth_dir = os.path.join(correspondence_dir, "artificial_worn")
    ssm_out = os.path.join(output_dir, "ssm")
    recon_out = os.path.join(output_dir, "reconstructions")
    os.makedirs(ssm_out, exist_ok=True)
    os.makedirs(recon_out, exist_ok=True)

    # ---- 1. Load good teeth ----
    good_dirs = sorted(glob(os.path.join(good_teeth_dir, "tooth_*")))
    good_paths: List[str] = []
    good_ids: List[str] = []
    for d in good_dirs:
        cp_path = os.path.join(d, "corresponded.ply")
        if not os.path.exists(cp_path):
            continue
        tid = os.path.basename(d).split("_")[1]
        good_ids.append(tid)
        good_paths.append(cp_path)

    n_good = len(good_paths)
    if n_good < 3:
        sys.exit(f"Need at least 3 good teeth, found {n_good}")

    print(f"Good teeth ({n_good}): {', '.join(good_ids)}")

    # Load and flatten
    shapes = []
    for p in good_paths:
        shapes.append(load_point_cloud(p))
    shapes = np.array(shapes)  # (M, N, 3)
    n_points = shapes.shape[1]
    X_good = shapes.reshape(n_good, -1)  # (M, 3N)
    print(f"  Point clouds: {n_points} points each, feature dim: {X_good.shape[1]}")

    # ---- 2. PCA for neighbor finding ----
    n_pca = min(n_good - 1, 10)
    pca = PCA(n_components=n_pca)
    scores_good = pca.fit_transform(X_good)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    print(f"\nNeighbor-finding PCA: {n_pca} components, "
          f"{cum_var[-1]*100:.1f}% variance explained")
    for i, (v, c) in enumerate(zip(var_ratio, cum_var)):
        print(f"    PC{i+1}: {v*100:6.2f}%  (cumulative {c*100:6.2f}%)")

    # Save PCA artifacts
    np.save(os.path.join(ssm_out, "global_pca_scores.npy"), scores_good)
    np.save(os.path.join(ssm_out, "global_pca_components.npy"), pca.components_)
    np.save(os.path.join(ssm_out, "global_pca_mean.npy"), pca.mean_)

    # ---- 3. Adaptive threshold ----
    threshold, threshold_info = compute_neighbor_threshold(
        scores_good, iqr_mult=threshold_iqr_mult)
    threshold_info["good_tooth_ids"] = good_ids
    print(f"\nAdaptive threshold: {threshold:.4f}")
    print(f"  Pairwise distances: median={threshold_info['median_pairwise']:.4f}, "
          f"IQR={threshold_info['iqr']:.4f}, "
          f"range=[{threshold_info['min_pairwise']:.4f}, {threshold_info['max_pairwise']:.4f}]")

    with open(os.path.join(ssm_out, "neighbor_threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)

    # ---- 4. Pre-build global SSM (for fallback) ----
    print("\nPre-building global SSM (fallback)...")
    mu_global, phi_global, lam_global, var_global = build_ssm(
        good_paths, variance_threshold=ssm_variance, use_gpu=use_gpu)
    print(f"  Global SSM: {len(lam_global)} components, "
          f"{var_global*100:.1f}% variance")

    # ---- 5. Discover worn teeth ----
    if worn_teeth_list:
        all_worn_dirs = worn_teeth_list
    else:
        all_worn_dirs = sorted([
            d for d in os.listdir(worn_teeth_dir)
            if os.path.isdir(os.path.join(worn_teeth_dir, d))
            and "_wear_" in d
        ])

    print(f"\nFound {len(all_worn_dirs)} worn teeth to reconstruct")

    # ---- 6. Reconstruct each worn tooth ----
    ssm_cache: Dict[frozenset, Tuple] = {}
    neighbor_summary: List[Dict] = []
    results: List[Dict] = []

    for worn_name in all_worn_dirs:
        worn_path = os.path.join(worn_teeth_dir, worn_name, "corresponded.ply")
        if not os.path.exists(worn_path):
            print(f"\n  [SKIP] {worn_name} - no corresponded.ply")
            continue

        print(f"\n{'─'*50}")
        print(f"  Processing {worn_name}")

        worn_points = load_point_cloud(worn_path)
        if len(worn_points) != n_points:
            print(f"    [SKIP] dimension mismatch: worn={len(worn_points)}, ssm={n_points}")
            continue

        # Extract tooth number and wear type
        parts = worn_name.split("_")
        tooth_num = parts[1]
        worn_name_parts = worn_name.split("_wear_")
        wear_type = worn_name_parts[1] if len(worn_name_parts) == 2 else "_".join(parts[2:])

        # Project into PCA space
        worn_flat = worn_points.flatten().reshape(1, -1)
        score_worn = pca.transform(worn_flat)[0]  # (K_pca,)

        # Select neighbors
        nb_idx, nb_dists, fallback_reason = select_neighbors(
            score_worn, scores_good, threshold,
            max_neighbors=max_neighbors)

        nb_ids = [good_ids[i] for i in nb_idx]
        method = "neighborhood"
        n_neighbors = len(nb_idx)

        print(f"    Neighbors ({n_neighbors}): {', '.join(nb_ids)}")
        print(f"    Distances: {', '.join(f'{d:.4f}' for d in nb_dists)}")

        if fallback_reason:
            method = "global_fallback"
            print(f"    FALLBACK to global SSM: {fallback_reason}")

        # Record neighbor selection
        nb_record = {
            "worn_tooth": worn_name,
            "method": method,
            "n_neighbors": n_neighbors,
            "neighbor_ids": nb_ids,
            "neighbor_indices": nb_idx.tolist(),
            "neighbor_distances": nb_dists.tolist(),
            "fallback_reason": fallback_reason,
            "threshold": threshold,
        }
        neighbor_summary.append(nb_record)

        # Build or retrieve SSM
        if method == "global_fallback":
            mu, phi, lambdas, var_expl = mu_global, phi_global, lam_global, var_global
        else:
            cache_key = frozenset(nb_idx.tolist())
            if cache_key not in ssm_cache:
                ssm_cache[cache_key] = build_neighborhood_ssm(
                    X_good, nb_idx,
                    variance_threshold=ssm_variance, use_gpu=use_gpu)
                mu_c, phi_c, lam_c, var_c = ssm_cache[cache_key]
                print(f"    Local SSM: {len(lam_c)} components, "
                      f"{var_c*100:.1f}% variance (built fresh)")
            else:
                print(f"    Local SSM: cached")
            mu, phi, lambdas, var_expl = ssm_cache[cache_key]

        # Estimate missing mask
        mean_points = mu.reshape(-1, 3)
        removed_mask = estimate_missing_mask_from_mean(
            worn_points, mean_points, proxy_missing_fraction)
        n_removed = int(removed_mask.sum())
        print(f"    Proxy mask: {n_removed}/{n_points} points "
              f"({100 * n_removed / n_points:.1f}%)")

        # Fit SSM
        observed_mask = ~removed_mask
        try:
            reconstructed, coefficients = fit_ssm_to_partial(
                mu, phi, lambdas,
                worn_points, observed_mask,
                regularization=regularization,
                use_gpu=use_gpu)
        except Exception as e:
            print(f"    [ERROR] SSM fitting failed: {e}")
            continue

        # Non-rigid refinement
        reconstructed, refinement_metrics = refine_reconstruction_nonrigid(
            reconstructed, worn_points, removed_mask, k=20)

        # Save reconstruction artifacts
        result_dir = os.path.join(recon_out, worn_name)
        os.makedirs(result_dir, exist_ok=True)
        save_point_cloud(worn_points, os.path.join(result_dir, "worn_input.ply"))
        save_point_cloud(reconstructed, os.path.join(result_dir, "reconstructed.ply"))
        np.save(os.path.join(result_dir, "coefficients.npy"), coefficients)
        np.save(os.path.join(result_dir, "removed_mask.npy"), removed_mask)

        # Inverse-transform to raw input space + geometric comparison
        corr_case_dir = os.path.join(worn_teeth_dir, worn_name)
        norm_path = os.path.join(corr_case_dir, "normalization.json")
        icp_path = os.path.join(corr_case_dir, "icp_transform.npy")

        geo_metrics = None
        reconstructed_raw = None
        if os.path.exists(norm_path):
            try:
                with open(norm_path) as f:
                    norm_scale = float(json.load(f)["scale"])

                reconstructed_raw = inverse_correspondence_transform(
                    reconstructed, norm_path, icp_path)
                save_point_cloud(
                    reconstructed_raw,
                    os.path.join(result_dir, "reconstructed_in_input_space.ply"))
                print(f"    Saved reconstructed_in_input_space.ply "
                      f"({len(reconstructed_raw)} pts)")

                # Geometric comparison with raw worn mesh
                raw_worn_path = None
                if artificial_wear_dir:
                    candidate = os.path.join(
                        artificial_wear_dir, f"tooth_{tooth_num}",
                        f"wear_{wear_type}.ply")
                    if os.path.exists(candidate):
                        raw_worn_path = candidate

                if raw_worn_path:
                    worn_raw = load_point_cloud(raw_worn_path)
                    from scipy.spatial import cKDTree
                    tree = cKDTree(worn_raw)
                    spacing, _ = tree.query(worn_raw, k=2)
                    median_spacing = float(np.median(spacing[:, 1]))
                    reconstructed_raw = _refine_icp_to_worn(
                        reconstructed_raw, worn_raw, median_spacing)
                    geo_metrics = compute_geometric_comparison(
                        reconstructed_raw, worn_raw, norm_scale)
                    print(f"    Geometric comparison:")
                    print(f"      R²:          {geo_metrics['variance_explained_pct']:.2f}%")
                    print(f"      Chamfer:     {geo_metrics['chamfer_mm']:.4f} mm")
                    print(f"      Hausdorff:   {geo_metrics['hausdorff_mm']:.4f} mm")
                    print(f"      RMSE w->r:   {geo_metrics['rmse_worn_to_recon_mm']:.4f} mm")
                    print(f"      Coverage 2x: {geo_metrics['coverage_2x_spacing']*100:.1f}%")
                    print(f"      F-Score@0.5: {geo_metrics.get('fscore_0p5mm', 0):.4f}")
                    print(f"      F-Score@1.0: {geo_metrics.get('fscore_1p0mm', 0):.4f}")
                    emd = geo_metrics.get('emd_mm')
                    print(f"      EMD:         {emd:.4f} mm" if emd is not None else "      EMD:         N/A")
            except Exception as e:
                print(f"    [WARN] Inverse transform / geo comparison failed: {e}")
                reconstructed_raw = None

        # Smooth mesh
        if reconstructed_raw is not None:
            try:
                mesh_obj = point_cloud_to_mesh(reconstructed_raw)
                mesh_path = os.path.join(result_dir, "reconstructed_smooth.ply")
                save_mesh(mesh_obj, mesh_path)
                print(f"    Smooth mesh: {len(mesh_obj.vertices)} verts, "
                      f"{len(mesh_obj.faces)} tris")
            except Exception as me:
                print(f"    [WARN] Smooth mesh failed: {me}")

        # Save evaluation
        eval_data = {
            "source_worn": worn_name,
            "method": method,
            "neighbor_info": nb_record,
            "ssm_components": len(lambdas),
            "ssm_variance_explained": float(var_expl),
            "n_points_removed": int(removed_mask.sum()),
            "coefficients": coefficients.tolist(),
            "regularization": regularization,
            "refinement": refinement_metrics,
            "geometric_comparison": geo_metrics,
        }
        with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
            json.dump(eval_data, f, indent=2)

        results.append({
            "name": worn_name,
            "method": method,
            "n_neighbors": n_neighbors,
            "neighbor_ids": nb_ids,
            "missing_fraction": float(n_removed / n_points),
            "refinement": refinement_metrics,
            "geometric_comparison": geo_metrics,
        })

    # ---- 7. Save neighbor selection summary ----
    with open(os.path.join(output_dir, "neighbor_selection.json"), "w") as f:
        json.dump({
            "threshold_info": threshold_info,
            "max_neighbors": max_neighbors,
            "selections": neighbor_summary,
        }, f, indent=2)
    print(f"\nNeighbor selection summary saved to neighbor_selection.json")

    # ---- 8. Comparison with global reconstruction ----
    if global_recon_dir:
        _print_comparison(results, global_recon_dir, output_dir)

    # ---- 9. Final summary ----
    n_local = sum(1 for r in results if r["method"] == "neighborhood")
    n_fallback = sum(1 for r in results if r["method"] == "global_fallback")
    print(f"\n{'='*60}")
    print(f"Neighborhood Reconstruction Summary")
    print(f"{'='*60}")
    print(f"  Total reconstructed: {len(results)}")
    print(f"  Local neighborhood:  {n_local}")
    print(f"  Global fallback:     {n_fallback}")
    print(f"  Output: {output_dir}")


def _print_comparison(results: List[Dict],
                      global_recon_dir: str,
                      output_dir: str) -> None:
    """Print and save comparison between neighborhood and global reconstruction."""
    print(f"\n{'='*90}")
    print("COMPARISON: Global Mean  vs  Neighborhood")
    print(f"{'='*90}")

    header = (f"{'Tooth':<30} {'Method':<12} {'R²(%)':<10} {'SSM_RMSE':<10} "
              f"{'Chamfer':<10} {'Hausdorff':<10} {'Cov_2x(%)':<10}")
    print(header)
    print("-" * 90)

    comparison_rows = []

    for r in results:
        worn_name = r["name"]

        # Load global result
        gpath = os.path.join(global_recon_dir, "reconstructions",
                             worn_name, "evaluation.json")
        g_ref, g_geo = {}, {}
        if os.path.exists(gpath):
            with open(gpath) as f:
                gdata = json.load(f)
            g_ref = gdata.get("refinement") or {}
            g_geo = gdata.get("geometric_comparison") or {}

        # Neighborhood result
        n_ref = r.get("refinement") or {}
        n_geo = r.get("geometric_comparison") or {}

        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        def fmt_pct(v):
            return f"{v:.2f}" if v is not None else "N/A"

        # Global row
        g_r2  = g_geo.get("variance_explained_pct")
        g_ssm = g_ref.get("ssm_obs_rmse")
        g_ch  = g_geo.get("chamfer_mm")
        g_hd  = g_geo.get("hausdorff_mm")
        g_cov = g_geo.get("coverage_2x_spacing")
        g_fs05 = g_geo.get("fscore_0p5mm")
        g_fs10 = g_geo.get("fscore_1p0mm")
        g_emd  = g_geo.get("emd_mm")

        print(f"{worn_name:<30} {'Global':<12} "
              f"{fmt_pct(g_r2):<10} {fmt(g_ssm):<10} "
              f"{fmt(g_ch):<10} {fmt(g_hd):<10} "
              f"{fmt_pct(g_cov * 100 if g_cov is not None else None):<10} "
              f"{fmt(g_fs05):<10} {fmt(g_fs10):<10} {fmt(g_emd):<10}")

        # Neighborhood row
        n_r2  = n_geo.get("variance_explained_pct") if n_geo else None
        n_ssm = n_ref.get("ssm_obs_rmse") if n_ref else None
        n_ch  = n_geo.get("chamfer_mm") if n_geo else None
        n_hd  = n_geo.get("hausdorff_mm") if n_geo else None
        n_cov = n_geo.get("coverage_2x_spacing") if n_geo else None
        n_fs05 = n_geo.get("fscore_0p5mm") if n_geo else None
        n_fs10 = n_geo.get("fscore_1p0mm") if n_geo else None
        n_emd  = n_geo.get("emd_mm") if n_geo else None

        method_tag = f"Nbr({r['n_neighbors']})" if r["method"] == "neighborhood" else "Fallback"
        print(f"{'':<30} {method_tag:<12} "
              f"{fmt_pct(n_r2):<10} {fmt(n_ssm):<10} "
              f"{fmt(n_ch):<10} {fmt(n_hd):<10} "
              f"{fmt_pct(n_cov * 100 if n_cov is not None else None):<10} "
              f"{fmt(n_fs05):<10} {fmt(n_fs10):<10} {fmt(n_emd):<10}")
        print()

        comparison_rows.append({
            "tooth": worn_name,
            "method": r["method"],
            "n_neighbors": r["n_neighbors"],
            "neighbor_ids": r["neighbor_ids"],
            "global": {
                "r2_pct": g_r2, "ssm_obs_rmse": g_ssm,
                "chamfer_mm": g_ch, "hausdorff_mm": g_hd,
                "coverage_2x": g_cov,
                "fscore_0p5mm": g_fs05, "fscore_1p0mm": g_fs10,
                "emd_mm": g_emd,
            },
            "neighborhood": {
                "r2_pct": n_r2, "ssm_obs_rmse": n_ssm,
                "chamfer_mm": n_ch, "hausdorff_mm": n_hd,
                "coverage_2x": n_cov,
                "fscore_0p5mm": n_fs05, "fscore_1p0mm": n_fs10,
                "emd_mm": n_emd,
            },
        })

    comp_path = os.path.join(output_dir, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump({"rows": comparison_rows}, f, indent=2)
    print(f"Comparison saved to {comp_path}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Neighborhood SSM reconstruction: adaptive local shape priors")
    parser.add_argument("--correspondence-dir", type=str,
                        default="output/correspondence_all_100k",
                        help="Correspondence directory with good_teeth/ and artificial_worn/")
    parser.add_argument("--global-recon-dir", type=str, default=None,
                        help="Directory with global-mean reconstruction results (for comparison)")
    parser.add_argument("--output", type=str,
                        default="output/recon_neighborhood_v4",
                        help="Output directory")
    parser.add_argument("--worn-teeth", nargs="+", default=None,
                        help="Specific worn-tooth directory names to reconstruct "
                             "(default: all found)")
    parser.add_argument("--max-neighbors", type=int, default=5,
                        help="Max good-teeth neighbors to use (default: 5). "
                             "All teeth within threshold are used, up to this cap.")
    parser.add_argument("--threshold-iqr-mult", type=float, default=2.0,
                        help="IQR multiplier for adaptive distance threshold (default: 2.0)")
    parser.add_argument("--pca-variance", type=float, default=0.95,
                        help="Variance threshold for neighbor-finding PCA (default: 0.95)")
    parser.add_argument("--ssm-variance", type=float, default=0.95,
                        help="Variance threshold for local SSM PCA (default: 0.95)")
    parser.add_argument("--proxy-missing-fraction", type=float, default=0.15,
                        help="Fraction of points treated as missing (default: 0.15)")
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="Tikhonov regularization strength (default: 1.0)")
    parser.add_argument("--artificial-wear", type=str, default=None,
                        help="Directory with raw worn tooth meshes for geometric comparison")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only computation")
    args = parser.parse_args()

    run_neighborhood_reconstruction(
        correspondence_dir=args.correspondence_dir,
        output_dir=args.output,
        global_recon_dir=args.global_recon_dir,
        artificial_wear_dir=args.artificial_wear,
        worn_teeth_list=args.worn_teeth,
        max_neighbors=args.max_neighbors,
        threshold_iqr_mult=args.threshold_iqr_mult,
        pca_variance=args.pca_variance,
        ssm_variance=args.ssm_variance,
        proxy_missing_fraction=args.proxy_missing_fraction,
        regularization=args.regularization,
        use_gpu=not args.no_gpu,
    )


if __name__ == "__main__":
    main()
