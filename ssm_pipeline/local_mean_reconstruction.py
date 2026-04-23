#!/usr/bin/env python3
"""
Local-Mean SSM Reconstruction Experiment

Instead of using the global mean (average of all good teeth) as the SSM centre,
this script uses a single designated good tooth as the mean and rebuilds the PCA
modes around it.  Everything else (mask estimation, coefficient fitting, non-rigid
refinement, inverse transform, geometric comparison, smooth mesh) is identical to
the main reconstruction_pipeline.py.

Usage:
    python local_mean_reconstruction.py \
        --correspondence-dir output/correspondence_all_100k \
        --global-recon-dir  output/recon_all \
        --output            output/recon_local_t14 \
        --local-mean-tooth  14 \
        --worn-teeth tooth_03_wear_real tooth_06_wear_real tooth_07_wear_real \
        --proxy-missing-fraction 0.15
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np

from reconstruction_pipeline import (
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
# Local-mean SSM builder
# ------------------------------------------------------------------ #

def build_local_ssm(
    corresponded_paths: List[str],
    local_mean_idx: int,
    n_components: int = None,
    variance_threshold: float = 0.95,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a PCA-based SSM centred on a specific tooth instead of the global mean.

    Returns (mu, eigenvectors, eigenvalues, variance_explained) with the
    same shapes as reconstruction_pipeline.build_ssm.
    """
    print("Building Local-Mean SSM ...")

    shapes = []
    for p in corresponded_paths:
        shapes.append(load_point_cloud(p))
    shapes = np.array(shapes)  # (M, N, 3)
    M, N, _ = shapes.shape
    print(f"  Loaded {M} shapes with {N} points each")

    X = shapes.reshape(M, -1)  # (M, 3N)
    mu = X[local_mean_idx].copy()
    print(f"  Using tooth index {local_mean_idx} as local mean")

    D = X - mu  # centred on the chosen tooth

    if use_gpu and CUPY_AVAILABLE:
        print("  Using GPU (CuPy) for SVD ...")
        D_gpu = cp.asarray(D)
        _, S, Vt = cp.linalg.svd(D_gpu, full_matrices=False)
        S = cp.asnumpy(S)
        Vt = cp.asnumpy(Vt)
    elif use_gpu and TORCH_AVAILABLE:
        print("  Using GPU (PyTorch) for SVD ...")
        D_t = torch.tensor(D, device="cuda", dtype=torch.float64)
        _, S, Vt = torch.linalg.svd(D_t, full_matrices=False)
        S = S.cpu().numpy()
        Vt = Vt.cpu().numpy()
    else:
        print("  Using CPU for SVD ...")
        _, S, Vt = np.linalg.svd(D, full_matrices=False)

    eigenvalues = (S ** 2) / (M - 1)
    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var

    if n_components is None:
        n_components = int(np.searchsorted(cumulative_var, variance_threshold) + 1)
        n_components = min(n_components, M - 1)

    print(f"  Keeping {n_components} components "
          f"({cumulative_var[n_components - 1] * 100:.1f}% variance)")

    eigenvectors = Vt[:n_components]
    eigenvalues = eigenvalues[:n_components]
    variance_explained = float(cumulative_var[n_components - 1])

    return mu, eigenvectors, eigenvalues, variance_explained


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Local-mean SSM reconstruction experiment")
    parser.add_argument("--correspondence-dir", type=str,
                        default="output/correspondence_all_100k",
                        help="Correspondence directory with good_teeth/ and artificial_worn/")
    parser.add_argument("--global-recon-dir", type=str,
                        default="output/recon_all",
                        help="Directory with existing global-mean reconstruction results")
    parser.add_argument("--output", type=str,
                        default="output/recon_local_t14",
                        help="Output directory for local-mean results")
    parser.add_argument("--local-mean-tooth", type=str, default="14",
                        help="Good-tooth ID to use as local mean (e.g. '14')")
    parser.add_argument("--worn-teeth", nargs="+",
                        default=["tooth_03_wear_real",
                                 "tooth_06_wear_real",
                                 "tooth_07_wear_real"],
                        help="List of worn-tooth directory names to reconstruct")
    parser.add_argument("--proxy-missing-fraction", type=float, default=0.15,
                        help="Fraction of points treated as missing (default 0.15)")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                        help="Cumulative variance threshold for PCA (default 0.95)")
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="Tikhonov regularisation strength (default 1.0)")
    parser.add_argument("--artificial-wear", type=str, default=None,
                        help="Directory with raw worn tooth meshes "
                             "(e.g. ../all_worn_input). Used for geometric "
                             "comparison in input space.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only computation")
    args = parser.parse_args()

    corr_dir = args.correspondence_dir
    global_recon_dir = args.global_recon_dir
    output_dir = args.output
    use_gpu = not args.no_gpu

    print("=" * 60)
    print("Local-Mean SSM Reconstruction Experiment")
    print("=" * 60)
    print(f"  Correspondence dir : {corr_dir}")
    print(f"  Global recon dir   : {global_recon_dir}")
    print(f"  Output dir         : {output_dir}")
    print(f"  Local mean tooth   : tooth_{args.local_mean_tooth}")
    print(f"  Worn teeth         : {args.worn_teeth}")
    print(f"  Artificial wear dir: {args.artificial_wear}")
    print(f"  Missing fraction   : {args.proxy_missing_fraction}")
    print(f"  GPU                : {use_gpu and (CUPY_AVAILABLE or TORCH_AVAILABLE)}")
    print()

    # ---- directories ----
    good_teeth_dir = os.path.join(corr_dir, "good_teeth")
    worn_teeth_dir = os.path.join(corr_dir, "artificial_worn")
    ssm_dir = os.path.join(output_dir, "ssm")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(ssm_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # ---- 1. Discover good teeth ----
    good_dirs = sorted(glob(os.path.join(good_teeth_dir, "tooth_*")))
    good_paths: List[str] = []
    good_ids: List[str] = []
    local_mean_idx = None
    for d in good_dirs:
        cp_path = os.path.join(d, "corresponded.ply")
        if not os.path.exists(cp_path):
            continue
        tid = os.path.basename(d).split("_")[1]
        if tid == args.local_mean_tooth:
            local_mean_idx = len(good_paths)
        good_ids.append(tid)
        good_paths.append(cp_path)

    if local_mean_idx is None:
        print(f"ERROR: tooth_{args.local_mean_tooth} not found among good teeth "
              f"({', '.join(good_ids)})")
        sys.exit(1)

    print(f"Good teeth ({len(good_paths)}): {', '.join(good_ids)}")
    print(f"Local mean -> tooth_{args.local_mean_tooth} (index {local_mean_idx})")

    # ---- 2. Build local SSM ----
    mu, phi, lambdas, var_explained = build_local_ssm(
        good_paths,
        local_mean_idx,
        variance_threshold=args.variance_threshold,
        use_gpu=use_gpu,
    )

    np.save(os.path.join(ssm_dir, "mean_shape.npy"), mu)
    np.save(os.path.join(ssm_dir, "eigenvectors.npy"), phi)
    np.save(os.path.join(ssm_dir, "eigenvalues.npy"), lambdas)
    save_point_cloud(mu.reshape(-1, 3), os.path.join(ssm_dir, "mean_shape.ply"))

    ssm_meta = {
        "type": "local_mean",
        "local_mean_tooth": f"tooth_{args.local_mean_tooth}",
        "n_training_samples": len(good_paths),
        "n_points_per_sample": len(mu) // 3,
        "n_components": len(lambdas),
        "variance_explained": float(var_explained),
        "eigenvalues": lambdas.tolist(),
        "cumulative_variance": (np.cumsum(lambdas) / lambdas.sum()).tolist(),
        "training_teeth": good_ids,
    }
    with open(os.path.join(ssm_dir, "ssm_metadata.json"), "w") as f:
        json.dump(ssm_meta, f, indent=2)

    print(f"\nLocal SSM saved to {ssm_dir}")
    print(f"  Components        : {len(lambdas)}")
    print(f"  Variance explained: {var_explained * 100:.1f}%")

    # ---- 3. Reconstruct selected worn teeth ----
    mean_points = mu.reshape(-1, 3)
    n_ssm = len(mean_points)

    # Locate raw worn teeth directory (for geometric comparison)
    artificial_wear_base = args.artificial_wear

    local_results: Dict[str, dict] = {}

    print(f"\nReconstructing {len(args.worn_teeth)} worn teeth ...")
    for worn_name in args.worn_teeth:
        worn_path = os.path.join(worn_teeth_dir, worn_name, "corresponded.ply")
        if not os.path.exists(worn_path):
            print(f"  [SKIP] {worn_name} - no corresponded.ply")
            continue

        print(f"\n  Processing {worn_name} ...")
        worn_points = load_point_cloud(worn_path)
        if len(worn_points) != n_ssm:
            print(f"    [SKIP] dimension mismatch: worn={len(worn_points)}, ssm={n_ssm}")
            continue

        # Extract tooth number and wear type
        parts = worn_name.split("_")
        tooth_num = parts[1]
        wear_type = "_".join(parts[2:]).replace("wear_", "")

        # Proxy mask using local mean
        removed_mask = estimate_missing_mask_from_mean(
            worn_points, mean_points, args.proxy_missing_fraction)
        n_removed = int(removed_mask.sum())
        print(f"    Proxy mask: {n_removed}/{n_ssm} points "
              f"({100 * n_removed / n_ssm:.1f}%)")

        # Fit SSM
        observed_mask = ~removed_mask
        reconstructed, coefficients = fit_ssm_to_partial(
            mu, phi, lambdas,
            worn_points, observed_mask,
            regularization=args.regularization,
            use_gpu=use_gpu,
        )

        # Non-rigid refinement
        reconstructed, refinement_metrics = refine_reconstruction_nonrigid(
            reconstructed, worn_points, removed_mask, k=20)

        # Save reconstruction artefacts
        result_dir = os.path.join(recon_dir, worn_name)
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
                save_point_cloud(reconstructed_raw,
                                 os.path.join(result_dir, "reconstructed_in_input_space.ply"))
                print(f"    Saved reconstructed_in_input_space.ply ({len(reconstructed_raw)} pts)")

                raw_worn_path = None
                if artificial_wear_base:
                    raw_worn_path = os.path.join(
                        artificial_wear_base, f"tooth_{tooth_num}", f"wear_{wear_type}.ply")
                    if not os.path.exists(raw_worn_path):
                        raw_worn_path = None

                if raw_worn_path is not None:
                    worn_raw = load_point_cloud(raw_worn_path)
                    from scipy.spatial import cKDTree
                    tree = cKDTree(worn_raw)
                    spacing, _ = tree.query(worn_raw, k=2)
                    median_spacing = float(np.median(spacing[:, 1]))
                    reconstructed_raw = _refine_icp_to_worn(
                        reconstructed_raw, worn_raw, median_spacing)
                    geo_metrics = compute_geometric_comparison(
                        reconstructed_raw, worn_raw, norm_scale)
                    print(f"    Geometric comparison (recon vs raw worn):")
                    print(f"      R²:         {geo_metrics['variance_explained_pct']:.2f}%")
                    print(f"      Chamfer:    {geo_metrics['chamfer_mm']:.4f} mm")
                    print(f"      Hausdorff:  {geo_metrics['hausdorff_mm']:.4f} mm")
                    print(f"      RMSE w→r:   {geo_metrics['rmse_worn_to_recon_mm']:.4f} mm")
                    print(f"      Coverage 2x:{geo_metrics['coverage_2x_spacing'] * 100:.1f}%")
            except Exception as e:
                print(f"    [WARN] Inverse transform / geo comparison failed: {e}")
                reconstructed_raw = None

        # Smooth mesh
        if reconstructed_raw is not None:
            try:
                mesh_obj = point_cloud_to_mesh(reconstructed_raw)
                mesh_path = os.path.join(result_dir, "reconstructed_smooth.ply")
                save_mesh(mesh_obj, mesh_path)
                print(f"    Smooth mesh: reconstructed_smooth.ply "
                      f"({len(mesh_obj.vertices)} verts, {len(mesh_obj.faces)} tris)")
            except Exception as me:
                print(f"    [WARN] Smooth mesh failed: {me}")

        # Save evaluation.json
        eval_data = {
            "source_worn": worn_name,
            "method": "local_mean",
            "local_mean_tooth": f"tooth_{args.local_mean_tooth}",
            "n_points_removed": int(removed_mask.sum()),
            "n_components_used": len(lambdas),
            "coefficients": coefficients.tolist(),
            "regularization": args.regularization,
            "refinement": refinement_metrics,
            "geometric_comparison": geo_metrics,
        }
        with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
            json.dump(eval_data, f, indent=2)

        local_results[worn_name] = {
            "refinement": refinement_metrics,
            "geometric_comparison": geo_metrics,
        }

    # ---- 4. Load global results for comparison ----
    global_results: Dict[str, dict] = {}
    for worn_name in args.worn_teeth:
        gpath = os.path.join(global_recon_dir, "reconstructions",
                             worn_name, "evaluation.json")
        if os.path.exists(gpath):
            with open(gpath) as f:
                global_results[worn_name] = json.load(f)
        else:
            print(f"  [INFO] No global result found for {worn_name}")

    # ---- 5. Comparison table ----
    print("\n" + "=" * 90)
    print("COMPARISON: Global Mean  vs  Local Mean (tooth_{})".format(
        args.local_mean_tooth))
    print("=" * 90)

    header = (f"{'Tooth':<22} {'Method':<8} {'R²(%)':<10} {'SSM_RMSE':<10} "
              f"{'Chamfer':<10} {'Hausdorff':<10} {'Cov_2x(%)':<10}")
    print(header)
    print("-" * 90)

    comparison_rows = []

    for worn_name in args.worn_teeth:
        # Global row
        g = global_results.get(worn_name, {})
        g_ref = g.get("refinement") or {}
        g_geo = g.get("geometric_comparison") or {}
        g_r2 = g_geo.get("variance_explained_pct", g_geo.get("r_squared", 0) * 100
                         if "r_squared" in g_geo else None)
        g_ssm_rmse = g_ref.get("ssm_obs_rmse")
        g_chamfer = g_geo.get("chamfer_mm")
        g_hausdorff = g_geo.get("hausdorff_mm")
        g_cov2x = g_geo.get("coverage_2x_spacing")

        def fmt(v, mult=1):
            if v is None:
                return "N/A"
            return f"{v * mult:.4f}"

        def fmt_pct(v):
            if v is None:
                return "N/A"
            return f"{v:.2f}"

        print(f"{worn_name:<22} {'Global':<8} "
              f"{fmt_pct(g_r2):<10} "
              f"{fmt(g_ssm_rmse):<10} "
              f"{fmt(g_chamfer):<10} "
              f"{fmt(g_hausdorff):<10} "
              f"{fmt_pct(g_cov2x * 100 if g_cov2x is not None else None):<10}")

        # Local row
        loc = local_results.get(worn_name, {})
        l_ref = loc.get("refinement") or {}
        l_geo = loc.get("geometric_comparison") or {}
        l_r2 = l_geo.get("variance_explained_pct")
        l_ssm_rmse = l_ref.get("ssm_obs_rmse")
        l_chamfer = l_geo.get("chamfer_mm")
        l_hausdorff = l_geo.get("hausdorff_mm")
        l_cov2x = l_geo.get("coverage_2x_spacing")

        print(f"{'':<22} {'Local':<8} "
              f"{fmt_pct(l_r2):<10} "
              f"{fmt(l_ssm_rmse):<10} "
              f"{fmt(l_chamfer):<10} "
              f"{fmt(l_hausdorff):<10} "
              f"{fmt_pct(l_cov2x * 100 if l_cov2x is not None else None):<10}")

        # Deltas
        def delta(a, b):
            if a is None or b is None:
                return None
            return a - b

        d_r2 = delta(l_r2, g_r2)
        d_ssm = delta(l_ssm_rmse, g_ssm_rmse)
        d_chamfer = delta(l_chamfer, g_chamfer)
        d_hausdorff = delta(l_hausdorff, g_hausdorff)

        def fmt_delta(v, invert=False):
            if v is None:
                return ""
            sign = "+" if v >= 0 else ""
            better = (v > 0) if not invert else (v < 0)
            tag = " <--" if better else ""
            return f"{sign}{v:.5f}{tag}"

        print(f"{'':<22} {'Delta':<8} "
              f"{fmt_delta(d_r2):<10} "
              f"{fmt_delta(d_ssm, invert=True):<10} "
              f"{fmt_delta(d_chamfer, invert=True):<10} "
              f"{fmt_delta(d_hausdorff, invert=True):<10}")
        print()

        comparison_rows.append({
            "tooth": worn_name,
            "global": {"r2_pct": g_r2, "ssm_obs_rmse": g_ssm_rmse,
                        "chamfer_mm": g_chamfer, "hausdorff_mm": g_hausdorff,
                        "coverage_2x": g_cov2x},
            "local": {"r2_pct": l_r2, "ssm_obs_rmse": l_ssm_rmse,
                       "chamfer_mm": l_chamfer, "hausdorff_mm": l_hausdorff,
                       "coverage_2x": l_cov2x},
            "delta_r2_pct": d_r2,
            "delta_ssm_rmse": d_ssm,
            "delta_chamfer_mm": d_chamfer,
            "delta_hausdorff_mm": d_hausdorff,
        })

    # Save comparison
    comp_path = os.path.join(output_dir, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump({
            "local_mean_tooth": f"tooth_{args.local_mean_tooth}",
            "worn_teeth": args.worn_teeth,
            "rows": comparison_rows,
        }, f, indent=2)
    print(f"Comparison saved to {comp_path}")
    print("Done.")


if __name__ == "__main__":
    main()
