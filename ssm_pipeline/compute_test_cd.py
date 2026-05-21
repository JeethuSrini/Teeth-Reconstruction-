"""
Compute Chamfer Distance between:
  - Neighborhood SSM reconstructions of TEST1/TEST2 wear levels
  - The corresponded unworn original (same normalized template space)

All comparisons are done in normalized/corresponded space because:
  - reconstructed.ply  : SSM output in template space (same as corresponded.ply)
  - corresponded.ply   : CPD-registered points in template space
  - The raw worn PLY files are identical to the unworn scan at the point-cloud
    level, so only the corresponded versions capture actual wear geometry.

Usage:
  conda activate teeth
  cd ssm_pipeline
  python compute_test_cd.py
"""

import json
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from reconstruction_pipeline import load_point_cloud

# ── Paths ─────────────────────────────────────────────────────────────────────
NBR_RECON_DIR      = "output/recon_neighborhood_v3/reconstructions"
CORRESPONDENCE_DIR = "output/correspondence_all_100k"

LEVELS = [f"level{i}" for i in range(8)]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer Distance (mean of both directed distances)."""
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_a2b, _ = tree_b.query(a)
    d_b2a, _ = tree_a.query(b)
    return float((np.mean(d_a2b) + np.mean(d_b2a)) / 2.0)


def run():
    results = {}

    for test_id in ("TEST1", "TEST2"):
        # Unworn original in template/corresponded space
        orig_path = os.path.join(
            CORRESPONDENCE_DIR, "originals",
            f"original_{test_id}", "corresponded.ply"
        )
        if not os.path.exists(orig_path):
            print(f"[WARN] Original corresponded not found: {orig_path}")
            continue

        orig_pts = load_point_cloud(orig_path)
        print(f"\n{'='*68}")
        print(f"  {test_id}  —  reference: corresponded original "
              f"({len(orig_pts):,} pts, template space)")
        print(f"{'='*68}")
        print(f"  Note: all values in normalized template units "
              f"(scale ≈ 13.7 mm → multiply by ~13.7 for mm)")
        print(f"  {'Level':<12} {'Recon vs Original':>20}  "
              f"{'Worn vs Original':>20}  {'Recon improves?':>16}")
        print(f"  {'-'*72}")

        results[test_id] = {}

        for level in LEVELS:
            worn_dir  = f"tooth_{test_id}_wear_{level}"

            # reconstructed.ply is in template/corresponded space (same as corresponded.ply)
            recon_path = os.path.join(NBR_RECON_DIR, worn_dir, "reconstructed.ply")

            # corresponded worn input (template space, captures actual wear geometry)
            worn_corr_path = os.path.join(
                CORRESPONDENCE_DIR, "artificial_worn",
                f"tooth_{test_id}_wear_{level}", "corresponded.ply"
            )

            # CD: SSM reconstruction vs unworn original (how well recon restores shape)
            cd_recon = None
            if os.path.exists(recon_path):
                recon_pts = load_point_cloud(recon_path)
                cd_recon  = chamfer(recon_pts, orig_pts)
            else:
                print(f"  [WARN] Missing reconstruction: {recon_path}")

            # CD: worn input vs unworn original (how much wear degraded the shape)
            cd_worn = None
            if os.path.exists(worn_corr_path):
                worn_pts = load_point_cloud(worn_corr_path)
                cd_worn  = chamfer(worn_pts, orig_pts)
            else:
                print(f"  [WARN] Missing worn corresponded: {worn_corr_path}")

            improves = ""
            if cd_recon is not None and cd_worn is not None:
                improves = "YES" if cd_recon < cd_worn else "no"

            recon_str = f"{cd_recon:.6f}" if cd_recon is not None else "N/A"
            worn_str  = f"{cd_worn:.6f}"  if cd_worn  is not None else "N/A"
            print(f"  {level:<12} {recon_str:>20}  {worn_str:>20}  {improves:>16}")

            results[test_id][level] = {
                "cd_recon_vs_original": cd_recon,
                "cd_worn_vs_original":  cd_worn,
                "recon_improves_over_worn": (
                    bool(cd_recon < cd_worn)
                    if cd_recon is not None and cd_worn is not None else None
                ),
            }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  SUMMARY — avg CD across all levels (template-space units)")
    print(f"{'='*68}")
    for test_id, levels in results.items():
        recon_vals = [v["cd_recon_vs_original"] for v in levels.values()
                      if v["cd_recon_vs_original"] is not None]
        worn_vals  = [v["cd_worn_vs_original"]  for v in levels.values()
                      if v["cd_worn_vs_original"]  is not None]
        wins = sum(1 for v in levels.values()
                   if v.get("recon_improves_over_worn"))
        total = sum(1 for v in levels.values()
                    if v.get("recon_improves_over_worn") is not None)
        if recon_vals:
            print(f"  {test_id}  recon vs original:  "
                  f"avg={np.mean(recon_vals):.6f}  "
                  f"min={np.min(recon_vals):.6f}  "
                  f"max={np.max(recon_vals):.6f}")
        if worn_vals:
            print(f"  {test_id}  worn  vs original:  "
                  f"avg={np.mean(worn_vals):.6f}  "
                  f"min={np.min(worn_vals):.6f}  "
                  f"max={np.max(worn_vals):.6f}")
        print(f"  {test_id}  recon improves over worn: {wins}/{total} levels")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = "output/test_cd_vs_original.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Units: normalized template space (divide by ~13.7 to approximate mm)")


if __name__ == "__main__":
    run()
