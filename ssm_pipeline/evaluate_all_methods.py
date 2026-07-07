#!/usr/bin/env python3
"""
Consolidated Evaluation: All Reconstruction Methods x All Datasets
====================================================================

Computes comparable metrics (Chamfer, RMSE, MAE, Hausdorff, Coverage@2x) for
every reconstruction method we have built, split into:

  - FULL-SURFACE metrics : reconstruction vs. the true original tooth, over
    every point. Rewards keeping the real (lightly-worn) surface as much as
    rewarding genuine restoration -- can be misleading on its own.
  - WORN-REGION metrics  : the same comparison restricted to the points that
    were actually altered by wear. This is the fair test of restoration
    quality, since it is blind to how much of the surface was never touched.

Two datasets:
  - "old"  : TEST1/TEST2 (8 wear levels each) vs. the original full teeth
             n0245 / n0257 (raw meshes, ICP-aligned; not corresponded, so the
             worn region is located via each tooth's own wear mask).
  - "v5"   : 8 real specimens (N1063, N332, N4, N459, N705, N726, N728, N891)
             x 8 wear levels vs. their own corresponded original (exact,
             index-paired -- no ICP needed, no approximation).

Usage:
    conda activate teeth
    cd ssm_pipeline
    python evaluate_all_methods.py                 # both datasets, all methods
    python evaluate_all_methods.py --dataset old    # just the old dataset
    python evaluate_all_methods.py --dataset v5
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruction_pipeline import load_point_cloud, _refine_icp_to_worn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
ORIGINALS_DIR = "/Users/jeethu_srini/Downloads/Original models"


def chamfer_rmse_hausdorff(a: np.ndarray, b: np.ndarray) -> dict:
    """Symmetric Chamfer + directed RMSE/MAE/Hausdorff + coverage@2x."""
    tree_a, tree_b = cKDTree(a), cKDTree(b)
    d_a2b, _ = tree_b.query(a, k=1)
    d_b2a, _ = tree_a.query(b, k=1)
    msp = float(np.median(cKDTree(b).query(b[: min(20000, len(b))], k=2)[0][:, 1]))
    return {
        "chamfer": float((d_a2b.mean() + d_b2a.mean()) / 2.0),
        "rmse_b2a": float(np.sqrt((d_b2a ** 2).mean())),
        "mae_b2a": float(d_b2a.mean()),
        "hausdorff": float(max(d_a2b.max(), d_b2a.max())),
        "coverage_2x": float((d_b2a < 2 * msp).mean()),
    }


def directed_error(recon_subset: np.ndarray, gt: np.ndarray) -> dict:
    """One-directional restoration error: recon_subset -> nearest GT point."""
    if len(recon_subset) == 0:
        return {"n": 0, "rmse": None, "mae": None}
    d, _ = cKDTree(gt).query(recon_subset, k=1)
    return {"n": int(len(recon_subset)), "rmse": float(np.sqrt((d ** 2).mean())),
            "mae": float(d.mean())}


# ===========================================================================
# OLD DATASET (TEST1 -> n0245, TEST2 -> n0257)
# ===========================================================================

OLD_METHODS = {
    "global_ssm":        ("recon_all_v3",           "reconstructed_in_input_space.ply"),
    "neighborhood_ssm":   ("recon_neighborhood_v4",  "reconstructed_in_input_space.ply"),
    "blend":              ("recon_blend",            "reconstructed_blend_in_input_space.ply"),
    "patch_holes":        ("recon_holes_final",       "patched_holes_in_input_space.ply"),
    "gpmm":               ("recon_gpmm_test",         "reconstructed_in_input_space.ply"),
    "gpmm_kernel":        ("recon_gpmm_kernel_test",  "reconstructed_in_input_space.ply"),
}
OLD_GT = {"TEST1": "cprc_nyu_n0245_ULM3_EDJ.ply", "TEST2": "cprc_nyu_n0257_ULM3_EDJ.ply"}


def _sample_gt_mesh(path: str, n: int = 120000) -> np.ndarray:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    m.compute_vertex_normals()
    return np.asarray(m.sample_points_uniformly(n).points)


def _normalize(points: np.ndarray) -> np.ndarray:
    """center + scale by bbox-diagonal (matches correspondence_pipeline.py)."""
    c = points.mean(axis=0)
    p = points - c
    scale = np.linalg.norm(p.max(axis=0) - p.min(axis=0))
    return p / max(scale, 1e-12)


def _ground_truth_wear_mask(tooth: str, gt_norm: np.ndarray, corr_dir: str,
                            mult: float = 2.0) -> np.ndarray:
    """Ground-truth-anchored, method-agnostic wear mask: per corresponded-index
    point of the REAL worn scan (icp_aligned.ply, i.e. no method's opinion
    involved), True where that point sits recessed from the true original
    surface by more than ``mult`` x median spacing -- i.e. genuinely worn away.
    Using the same mask for every method (instead of one method's own guess of
    what's worn) avoids biasing the comparison toward whichever method's wear
    detector the mask happens to come from.
    """
    worn_path = os.path.join(corr_dir, tooth, "icp_aligned.ply")
    worn = load_point_cloud(worn_path)
    worn_n = _normalize(worn)
    worn_n_aligned = _refine_icp_to_worn(worn_n, gt_norm,
                                         float(np.median(cKDTree(gt_norm).query(
                                             gt_norm[:8000], k=2)[0][:, 1])))
    d, _ = cKDTree(gt_norm).query(worn_n_aligned, k=1)
    msp = float(np.median(cKDTree(worn_n_aligned).query(worn_n_aligned[:8000], k=2)[0][:, 1]))
    return d > mult * msp


def eval_old_dataset(methods=None) -> list:
    methods = methods or OLD_METHODS
    corr_dir = os.path.join(OUT_DIR, "correspondence_all_100k", "artificial_worn")
    rows = []
    for tset, gt_fname in OLD_GT.items():
        gt = _sample_gt_mesh(os.path.join(ORIGINALS_DIR, gt_fname))
        msp_gt = float(np.median(cKDTree(gt).query(gt[:15000], k=2)[0][:, 1]))
        gt_norm = _normalize(gt)
        for lvl in range(8):
            tooth = f"tooth_{tset}_wear_level{lvl}"
            try:
                mask = _ground_truth_wear_mask(tooth, gt_norm, corr_dir)
            except Exception as e:
                print(f"  [SKIP mask] {tooth}: {e}")
                mask = None

            for method, (d, fname) in methods.items():
                path = os.path.join(OUT_DIR, d, "reconstructions", tooth, fname)
                if not os.path.exists(path):
                    continue
                try:
                    recon = load_point_cloud(path)
                    recon_aligned = _refine_icp_to_worn(recon, gt, msp_gt)

                    # patch_holes appends new points (variable length) rather
                    # than replacing in-place, so the shared 100k-index wear
                    # mask doesn't apply -- use its OWN patch_mask.npy instead,
                    # which marks exactly the points it added (more precise: it
                    # is that method's true "what did I reconstruct" set).
                    own_mask_path = os.path.join(OUT_DIR, d, "reconstructions",
                                                 tooth, "patch_mask.npy")
                    if os.path.exists(own_mask_path):
                        own_mask = np.load(own_mask_path)
                        m = own_mask if len(own_mask) == len(recon_aligned) else None
                    else:
                        m = mask if mask is not None and len(mask) == len(recon_aligned) else None

                    full = chamfer_rmse_hausdorff(recon_aligned, gt)
                    worn = (directed_error(recon_aligned[m], gt)
                            if m is not None else {"n": 0, "rmse": None, "mae": None})
                    rows.append({
                        "dataset": "old", "set": tset, "level": lvl, "method": method,
                        "full_chamfer": round(full["chamfer"], 4),
                        "full_rmse": round(full["rmse_b2a"], 4),
                        "full_hausdorff": round(full["hausdorff"], 4),
                        "full_coverage_2x": round(full["coverage_2x"], 4),
                        "worn_n_pts": worn["n"],
                        "worn_rmse": round(worn["rmse"], 4) if worn["rmse"] is not None else "",
                        "worn_mae": round(worn["mae"], 4) if worn["mae"] is not None else "",
                    })
                    print(f"  [old] {tooth:26s} {method:16s} "
                          f"full_chamfer={full['chamfer']:.4f} worn_rmse="
                          f"{worn['rmse'] if worn['rmse'] is not None else float('nan'):.4f}")
                except Exception as e:
                    print(f"  [ERROR] {tooth} {method}: {e}")
    return rows


# ===========================================================================
# V5 DATASET (8 specimens x 8 levels, exact index-paired ground truth)
# ===========================================================================

V5_SPECIMENS = ["N1063", "N332", "N4", "N459", "N705", "N726", "N728", "N891"]

# Index-paired methods: recon is a fixed 10k cloud in the SAME corresponded
# ordering as `original`/`worn` -- no ICP or NN search needed, exact per-point
# comparison. (dir, filename) per method.
V5_METHODS = {
    "global_ssm":       ("recon_all_v5",          "reconstructed.ply"),
    "neighborhood_ssm":  ("recon_neighborhood_v5", "reconstructed.ply"),
    "gpmm":              ("recon_gpmm_v5",         "reconstructed.ply"),
    "blend":             ("recon_blend_v5",        "reconstructed_blend.ply"),
}
# NN-based methods: recon appends new points (variable length, e.g. patch/hole
# grafts), so it can't be index-paired -- compare via nearest-neighbor to the
# original, and use the method's OWN mask file to define its worn region.
V5_NN_METHODS = {
    # normalized (template-space) cloud -- NOT *_in_input_space.ply (mm), which
    # would be compared against `original`'s normalized frame and give bogus
    # (huge) distances from the unit mismatch.
    "patch_holes": ("recon_holes_v5", "patched_holes_points.ply", "patch_mask.npy"),
}
V5_CORR_DIR = os.path.join(OUT_DIR, "correspondence_v5_10k", "artificial_worn")


def eval_v5_dataset(methods=None, nn_methods=None) -> list:
    methods = methods if methods is not None else V5_METHODS
    nn_methods = nn_methods if nn_methods is not None else V5_NN_METHODS
    rows = []
    for spec in V5_SPECIMENS:
        orig_path = os.path.join(V5_CORR_DIR, f"tooth_{spec}_original", "corresponded.ply")
        if not os.path.exists(orig_path):
            print(f"  [SKIP] no corresponded original for {spec}")
            continue
        original = load_point_cloud(orig_path)
        msp = float(np.median(cKDTree(original).query(original[:8000], k=2)[0][:, 1]))
        for lvl in range(8):
            tooth = f"tooth_{spec}_wear_level{lvl}"
            worn_path = os.path.join(V5_CORR_DIR, tooth, "corresponded.ply")
            if not os.path.exists(worn_path):
                continue
            worn = load_point_cloud(worn_path)
            if len(worn) != len(original):
                print(f"  [SKIP] {tooth}: point-count mismatch"); continue
            # Exact, index-paired worn-region mask: where the true original
            # deviates from the worn input beyond a small noise floor.
            dev = np.linalg.norm(original - worn, axis=1)
            mask = dev > 1.5 * msp

            for method, (d, fname) in methods.items():
                path = os.path.join(OUT_DIR, d, "reconstructions", tooth, fname)
                if not os.path.exists(path):
                    continue
                try:
                    recon = load_point_cloud(path)
                    if len(recon) != len(original):
                        print(f"  [SKIP] {tooth} {method}: point-count mismatch"); continue
                    diff = recon - original
                    full_rmse = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
                    full_mae = float(np.linalg.norm(diff, axis=1).mean())
                    d_full, _ = cKDTree(original).query(recon, k=1)
                    full_chamfer = float(d_full.mean())  # exact frame, so this is already symmetric-ish; kept for cross-dataset comparability
                    if mask.sum() > 0:
                        wd = np.linalg.norm(diff[mask], axis=1)
                        worn_rmse = float(np.sqrt((wd ** 2).mean()))
                        worn_mae = float(wd.mean())
                    else:
                        worn_rmse = worn_mae = None
                    rows.append({
                        "dataset": "v5", "set": spec, "level": lvl, "method": method,
                        "full_chamfer": round(full_chamfer, 5),
                        "full_rmse": round(full_rmse, 5),
                        "full_mae": round(full_mae, 5),
                        "worn_n_pts": int(mask.sum()),
                        "worn_rmse": round(worn_rmse, 5) if worn_rmse is not None else "",
                        "worn_mae": round(worn_mae, 5) if worn_mae is not None else "",
                    })
                    print(f"  [v5] {tooth:26s} {method:16s} "
                          f"full_rmse={full_rmse:.5f} worn_rmse="
                          f"{worn_rmse if worn_rmse is not None else float('nan'):.5f}")
                except Exception as e:
                    print(f"  [ERROR] {tooth} {method}: {e}")

            # NN-based methods (variable-length recon: patch/hole grafts).
            for method, (d, fname, mask_fname) in nn_methods.items():
                path = os.path.join(OUT_DIR, d, "reconstructions", tooth, fname)
                mpath = os.path.join(OUT_DIR, d, "reconstructions", tooth, mask_fname)
                if not (os.path.exists(path) and os.path.exists(mpath)):
                    continue
                try:
                    recon = load_point_cloud(path)
                    own_mask = np.load(mpath)
                    if len(own_mask) != len(recon):
                        print(f"  [SKIP] {tooth} {method}: mask/cloud length mismatch"); continue
                    d_full, _ = cKDTree(original).query(recon, k=1)
                    d_orig, _ = cKDTree(recon).query(original, k=1)
                    full_chamfer = float((d_full.mean() + d_orig.mean()) / 2.0)
                    full_rmse = float(np.sqrt((d_full ** 2).mean()))
                    worn_pts = recon[own_mask]
                    if len(worn_pts) > 0:
                        wd, _ = cKDTree(original).query(worn_pts, k=1)
                        worn_rmse = float(np.sqrt((wd ** 2).mean()))
                        worn_mae = float(wd.mean())
                    else:
                        worn_rmse = worn_mae = None
                    rows.append({
                        "dataset": "v5", "set": spec, "level": lvl, "method": method,
                        "full_chamfer": round(full_chamfer, 5),
                        "full_rmse": round(full_rmse, 5),
                        "full_mae": "",
                        "worn_n_pts": int(own_mask.sum()),
                        "worn_rmse": round(worn_rmse, 5) if worn_rmse is not None else "",
                        "worn_mae": round(worn_mae, 5) if worn_mae is not None else "",
                    })
                    print(f"  [v5] {tooth:26s} {method:16s} "
                          f"full_chamfer={full_chamfer:.5f} worn_rmse="
                          f"{worn_rmse if worn_rmse is not None else float('nan'):.5f}")
                except Exception as e:
                    print(f"  [ERROR] {tooth} {method}: {e}")
    return rows


def save_csv(rows: list, path: str):
    if not rows:
        print(f"  (no rows to save for {path})")
        return
    cols = sorted({k for r in rows for k in r.keys()},
                  key=lambda k: (k != "dataset", k != "set", k != "level", k != "method", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {path} ({len(rows)} rows)")


def print_summary(rows: list, dataset: str):
    if not rows:
        return
    methods = sorted({r["method"] for r in rows})
    print(f"\n=== {dataset.upper()} — mean over all cases ===")
    print(f"{'method':18s} {'full_chamfer/rmse':>18s} {'worn_rmse':>12s} {'worn_mae':>10s}")
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        full_key = "full_chamfer" if dataset == "old" else "full_rmse"
        full_vals = [r[full_key] for r in sub if r.get(full_key) not in (None, "")]
        worn_rmse = [r["worn_rmse"] for r in sub if r.get("worn_rmse") not in (None, "")]
        worn_mae = [r["worn_mae"] for r in sub if r.get("worn_mae") not in (None, "")]
        fr = np.mean(full_vals) if full_vals else float("nan")
        wr = np.mean(worn_rmse) if worn_rmse else float("nan")
        wm = np.mean(worn_mae) if worn_mae else float("nan")
        print(f"{m:18s} {fr:18.5f} {wr:12.5f} {wm:10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["old", "v5", "both"], default="both")
    args = ap.parse_args()

    if args.dataset in ("old", "both"):
        print("=" * 70); print("OLD DATASET (TEST1 vs n0245, TEST2 vs n0257)"); print("=" * 70)
        rows_old = eval_old_dataset()
        save_csv(rows_old, os.path.join(OUT_DIR, "eval_old_dataset.csv"))
        print_summary(rows_old, "old")

    if args.dataset in ("v5", "both"):
        print("\n" + "=" * 70); print("V5 DATASET (8 real specimens vs. corresponded original)"); print("=" * 70)
        rows_v5 = eval_v5_dataset()
        save_csv(rows_v5, os.path.join(OUT_DIR, "eval_v5_dataset.csv"))
        print_summary(rows_v5, "v5")


if __name__ == "__main__":
    main()
