#!/usr/bin/env python3
"""
GPMM Posterior Shape-Completion Reconstruction  (Tier 1)
========================================================

Reconstruct a worn tooth as a single, globally-consistent complete shape by
*Gaussian-process posterior regression* under a shape prior — instead of
grafting reconstruction patches onto the raw scan (which produces seams/scabs).

The worn tooth's reliably-observed (unworn) corresponded points are treated as
GP observations; conditioning on them yields the posterior mean shape (the
reconstruction) plus per-point uncertainty. Key pieces:

  1. Prior = PCA SSM built from corresponded good teeth (the empirical kernel),
     optionally augmented with an analytic Gaussian kernel (-> a full GPMM) so
     the model can deform beyond the linear training span and capture local
     detail.
  2. Robust, data-driven detection of the worn region via IRLS: worn material is
     *recessed*, so points where the worn surface sits well inside the current
     fit are down-weighted and the posterior fills them from the prior. No
     geometric hole detection, no fixed "missing fraction".
  3. Detail-preserving refinement: trusted (unworn) points are snapped exactly to
     the real surface and the correction is carried smoothly into filled regions.

This is the principled, small-data-friendly route (the PCA SSM is a special case
of a GPMM; posterior shape models are standard for anatomical completion).

Usage:
    python gpmm_reconstruction.py \
        --correspondence-dir output/correspondence_all_100k \
        --output output/recon_gpmm \
        --variance 0.95 --kernel-modes 60 --kernel-sigma 2.0
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from reconstruction_pipeline import (  # noqa: E402
    build_ssm,
    refine_reconstruction_nonrigid,
    inverse_correspondence_transform,
    load_point_cloud,
    save_point_cloud,
    safe_mesh,
)


# ===========================================================================
# GPMM KERNEL AUGMENTATION (analytic Gaussian modes via Nyström)
# ===========================================================================

def gaussian_kernel_modes(ref_points: np.ndarray,
                          n_modes: int,
                          sigma_mult: float = 2.0,
                          scale_frac: float = 0.15,
                          n_landmarks: int = 1200,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Low-rank analytic Gaussian-kernel deformation modes over the reference
    shape (Nyström approximation), turning the PCA SSM into a full GPMM.

    Each smooth scalar eigenfunction of the Gaussian kernel becomes three
    vector-valued deformation modes (x/y/z), letting the model add local,
    correlated deformation the linear PCA span cannot represent.

    Args:
        ref_points: (N, 3) reference (mean) shape.
        n_modes: number of analytic modes to return (rounded to a multiple of 3).
        sigma_mult: kernel bandwidth as a multiple of the median NN spacing... no
            -- as a multiple of the shape's bbox diagonal * 0.1 (controls how
            local the deformation is; larger = smoother/global).
        scale_frac: per-mode std as a fraction of the bbox diagonal (deformation
            magnitude prior).
        n_landmarks: Nyström landmark count.
        seed: RNG seed for landmark sampling.

    Returns:
        (modes (M, 3N) unit-norm, variances (M,)).  M ~= n_modes.
    """
    from scipy.spatial.distance import cdist

    N = len(ref_points)
    diag = float(np.linalg.norm(ref_points.max(0) - ref_points.min(0)))
    sigma = sigma_mult * 0.1 * diag
    n_spatial = max(1, n_modes // 3)

    rng = np.random.default_rng(seed)
    m = min(n_landmarks, N)
    land_idx = rng.choice(N, m, replace=False)
    L = ref_points[land_idx]

    Kll = np.exp(-(cdist(L, L) ** 2) / (2.0 * sigma ** 2))
    evals, evecs = np.linalg.eigh(Kll)               # ascending
    order = np.argsort(evals)[::-1][:n_spatial]
    evals = np.maximum(evals[order], 1e-9)
    evecs = evecs[:, order]                           # (m, n_spatial)

    # Nyström extension of the eigenfunctions to all N points.
    Kal = np.exp(-(cdist(ref_points, L) ** 2) / (2.0 * sigma ** 2))  # (N, m)
    psi = (Kal @ evecs) / evals[None, :]              # (N, n_spatial)
    psi /= (np.linalg.norm(psi, axis=0, keepdims=True) + 1e-12)      # unit columns

    # variance per spatial eigenfunction (normalised), scaled to deformation mag
    var_spatial = evals / evals.max()
    mode_std = scale_frac * diag

    modes = []
    variances = []
    for j in range(n_spatial):
        for d in range(3):                            # x / y / z deformation
            v = np.zeros((N, 3))
            v[:, d] = psi[:, j]
            v = v.reshape(-1)
            v /= (np.linalg.norm(v) + 1e-12)
            modes.append(v)
            variances.append((mode_std ** 2) * var_spatial[j])
    return np.asarray(modes), np.asarray(variances)


# ===========================================================================
# ROBUST GP POSTERIOR (IRLS over the worn observation)
# ===========================================================================

def _posterior_b(mu: np.ndarray, Phi: np.ndarray, lam: np.ndarray,
                 worn_flat: np.ndarray, w3: np.ndarray, reg: float) -> np.ndarray:
    """Weighted Tikhonov posterior-mean coefficients:
        (Aᵀ W A + reg·diag(1/λ)) b = Aᵀ W (x - μ),   A = Φᵀ."""
    A = Phi.T                                # (3N, K)
    y = worn_flat - mu
    WA = w3[:, None] * A
    AtA = A.T @ WA + np.diag(reg / lam)
    Aty = A.T @ (w3 * y)
    b = np.linalg.solve(AtA, Aty)
    sigma = np.sqrt(np.maximum(lam, 1e-12))
    return np.clip(b, -4.0 * sigma, 4.0 * sigma)


def robust_posterior_reconstruct(mu: np.ndarray,
                                 Phi: np.ndarray,
                                 lam: np.ndarray,
                                 worn_pts: np.ndarray,
                                 reg: float = 1.0,
                                 n_iter: int = 12,
                                 welsch_c: float = 2.5,
                                 asym_factor: float = 0.2,
                                 trust_thresh: float = 0.5,
                                 refine: bool = True
                                 ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Robustly fit the GP posterior to the worn cloud and reconstruct.

    IRLS: trusted (unworn) points drive the fit; recessed/worn points are
    down-weighted (Welsch + asymmetric penalty for inward residuals) so the
    posterior fills them from the prior. Optionally refines so trusted points
    match the real surface exactly and the correction flows into filled regions.

    Returns (reconstructed (N,3), weight (N,), info).
    """
    N = len(worn_pts)
    worn_flat = worn_pts.reshape(-1)
    centroid = worn_pts.mean(0)
    outward = worn_pts - centroid
    outward /= np.linalg.norm(outward, axis=1, keepdims=True) + 1e-12

    w = np.ones(N)
    recon = worn_pts.copy()
    d = np.zeros(N)
    it = 0
    for it in range(n_iter):
        b = _posterior_b(mu, Phi, lam, worn_flat, np.repeat(w, 3), reg)
        recon = (mu + Phi.T @ b).reshape(-1, 3)
        diff = worn_pts - recon
        d = np.linalg.norm(diff, axis=1)
        scale = 1.4826 * np.median(d) + 1e-9
        w_new = np.exp(-(d / (welsch_c * scale)) ** 2)          # Welsch
        # asymmetric: worn material is recessed -> trust inward-residual pts less
        signed = np.einsum("ij,ij->i", diff, outward)           # worn - recon, outward
        recessed = signed < -0.5 * scale
        w_new[recessed] *= asym_factor
        if np.max(np.abs(w_new - w)) < 1e-3:
            w = w_new
            break
        w = w_new

    info = {
        "n_iter": int(it + 1),
        "frac_trusted": float((w > trust_thresh).mean()),
        "median_residual": float(np.median(d)),
        "n_modes": int(Phi.shape[0]),
    }

    if refine:
        removed = w <= trust_thresh                              # filled-from-prior
        if removed.any() and (~removed).any():
            recon, rinfo = refine_reconstruction_nonrigid(recon, worn_pts, removed)
            info["refine"] = rinfo
    return recon, w, info


# ===========================================================================
# DRIVER
# ===========================================================================

def _corr_paths(correspondence_dir: str) -> Tuple[List[str], List[str]]:
    good = sorted(glob(os.path.join(correspondence_dir, "good_teeth", "*", "corresponded.ply")))
    worn = sorted(glob(os.path.join(correspondence_dir, "artificial_worn", "*", "corresponded.ply")))
    return good, worn


def process_tooth(worn_corr_path: str, mu, Phi, lam, output_dir: str,
                  reg: float, robust_iters: int, skip_mesh: bool = False) -> Optional[Dict]:
    case_dir = os.path.dirname(worn_corr_path)
    name = os.path.basename(case_dir)
    worn = load_point_cloud(worn_corr_path)
    if worn.reshape(-1).shape[0] != mu.shape[0]:
        print(f"  [SKIP] {name}: point-count mismatch")
        return None

    print(f"\n  {name}")
    recon, w, info = robust_posterior_reconstruct(
        mu, Phi, lam, worn, reg=reg, n_iter=robust_iters)
    print(f"    trusted(unworn)={100*info['frac_trusted']:.1f}%  "
          f"iters={info['n_iter']}  median_resid={info['median_residual']:.4f}")

    result_dir = os.path.join(output_dir, "reconstructions", name)
    os.makedirs(result_dir, exist_ok=True)
    save_point_cloud(worn, os.path.join(result_dir, "worn_input.ply"))
    save_point_cloud(recon, os.path.join(result_dir, "reconstructed.ply"))
    np.save(os.path.join(result_dir, "trust_weight.npy"), w)

    # inverse-transform to input mm space + crash-safe mesh
    norm_path = os.path.join(case_dir, "normalization.json")
    icp_path = os.path.join(case_dir, "icp_transform.npy")
    depth = 0
    if os.path.exists(norm_path):
        try:
            recon_raw = inverse_correspondence_transform(recon, norm_path, icp_path)
            save_point_cloud(recon_raw, os.path.join(
                result_dir, "reconstructed_in_input_space.ply"))
            if not skip_mesh:
                depth = safe_mesh(recon_raw, os.path.join(result_dir, "reconstructed_smooth.ply"))
        except Exception as e:
            print(f"    [WARN] inverse transform / mesh failed: {e}")
    if skip_mesh:
        print(f"    mesh: skipped (--skip-mesh)")
    else:
        print(f"    mesh: {'reconstructed_smooth.ply (depth %d)' % depth if depth else 'skipped'}")

    info["name"] = name
    with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def run(correspondence_dir: str, output_dir: str, variance: float,
        kernel_modes: int, kernel_sigma: float, reg: float, robust_iters: int,
        worn_subset: Optional[List[str]], skip_mesh: bool = False) -> None:
    print("=" * 60)
    print("GPMM Posterior Reconstruction")
    print("=" * 60)
    good_paths, worn_paths = _corr_paths(correspondence_dir)
    n_before = len(worn_paths)
    worn_paths = [p for p in worn_paths
                  if not os.path.basename(os.path.dirname(p)).endswith("_original")]
    if len(worn_paths) != n_before:
        print(f"  Excluded {n_before - len(worn_paths)} unworn '_original' baseline dirs "
              f"(already complete -- nothing to reconstruct)")
    print(f"  Good teeth (prior): {len(good_paths)}   Worn teeth: {len(worn_paths)}")
    if worn_subset:
        worn_paths = [p for p in worn_paths
                      if os.path.basename(os.path.dirname(p)) in set(worn_subset)]
        print(f"  Restricted to {len(worn_paths)} worn teeth")

    # Prior: PCA SSM (empirical kernel)
    mu, Phi, lam, var_exp = build_ssm(good_paths, variance_threshold=variance, use_gpu=False)
    print(f"  PCA modes: {Phi.shape[0]} ({100*var_exp:.1f}% var)")

    # GPMM augmentation: analytic Gaussian modes
    if kernel_modes > 0:
        gm, gv = gaussian_kernel_modes(mu.reshape(-1, 3), n_modes=kernel_modes,
                                       sigma_mult=kernel_sigma)
        Phi = np.vstack([Phi, gm])
        lam = np.concatenate([lam, gv])
        print(f"  + {len(gm)} analytic Gaussian modes -> {Phi.shape[0]} total modes (GPMM)")

    os.makedirs(os.path.join(output_dir, "reconstructions"), exist_ok=True)
    results = []
    for wp in worn_paths:
        try:
            r = process_tooth(wp, mu, Phi, lam, output_dir, reg, robust_iters,
                              skip_mesh=skip_mesh)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [ERROR] {wp}: {e}")
            import traceback
            traceback.print_exc()

    summary = {"correspondence_dir": correspondence_dir, "n_reconstructed": len(results),
               "variance": variance, "kernel_modes": kernel_modes,
               "kernel_sigma": kernel_sigma, "regularization": reg,
               "robust_iters": robust_iters, "results": results}
    with open(os.path.join(output_dir, "gpmm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {len(results)} reconstructions -> {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="GPMM posterior shape-completion reconstruction")
    ap.add_argument("--correspondence-dir", default="output/correspondence_all_100k")
    ap.add_argument("--output", default="output/recon_gpmm")
    ap.add_argument("--variance", type=float, default=0.95, help="PCA variance kept")
    ap.add_argument("--kernel-modes", type=int, default=60,
                    help="Analytic Gaussian modes to append (0 = plain PCA SSM)")
    ap.add_argument("--kernel-sigma", type=float, default=2.0,
                    help="Gaussian kernel bandwidth (x 0.1 * bbox diagonal)")
    ap.add_argument("--regularization", type=float, default=1.0)
    ap.add_argument("--robust-iters", type=int, default=12)
    ap.add_argument("--worn-teeth", nargs="+", default=None)
    ap.add_argument("--skip-mesh", action="store_true",
                    help="Skip Poisson surface meshing (point clouds only; mesh later)")
    args = ap.parse_args()
    run(args.correspondence_dir, args.output, args.variance, args.kernel_modes,
        args.kernel_sigma, args.regularization, args.robust_iters, args.worn_teeth,
        skip_mesh=args.skip_mesh)


if __name__ == "__main__":
    main()
