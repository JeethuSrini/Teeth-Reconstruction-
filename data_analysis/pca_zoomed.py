"""
Zoomed-in PCA visualisation excluding the bottom-right outlier cluster,
showing BOTH global SSM and neighborhood SSM reconstructions.

Excluded items (from inspection of all_teeth_pca_2d.png):
  - Good tooth T14
  - Worn teeth T03/T05/T06/T07 (all wear levels) and their reconstructions

Categories plotted:
  - Good            (corresponded good-tooth scans)
  - Worn            (corresponded artificial-worn scans)
  - Recon (Global)  (reconstructed via the global SSM)
  - Recon (Nbr)     (reconstructed via the neighborhood SSM)
  - Original        (corresponded TEST1/TEST2 unworn originals)
  - TEST            (corresponded TEST1/TEST2 worn scans)

Outputs:
  data_analysis/plots_v2/pca_2d_zoomed.png
  data_analysis/plots_v2/pca_3d_zoomed.png

Usage:
  cd data_analysis
  conda activate teeth
  python pca_zoomed.py
"""

import os
from glob import glob

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_v2")
os.makedirs(PLOT_DIR, exist_ok=True)

CORR_DIR        = os.path.join(PROJECT_DIR, "ssm_pipeline", "output", "correspondence_all_100k")
RECON_GLOBAL    = os.path.join(PROJECT_DIR, "ssm_pipeline", "output", "recon_all_v3",          "reconstructions")
RECON_NBR       = os.path.join(PROJECT_DIR, "ssm_pipeline", "output", "recon_neighborhood_v4", "reconstructions")

GOOD_DIR = os.path.join(CORR_DIR, "good_teeth")
WORN_DIR = os.path.join(CORR_DIR, "artificial_worn")
ORIG_DIR = os.path.join(CORR_DIR, "originals")

# ── Exclusions (the outlier cluster) ────────────────────────────────────────
EXCLUDE_GOOD = {"tooth_14"}
EXCLUDE_WORN_TEETH = {"tooth_03", "tooth_05", "tooth_06", "tooth_07"}  # all wear levels


def load_ply(path):
    pc = trimesh.load(path, process=False)
    return np.asarray(
        pc.vertices if hasattr(pc, "vertices") else pc, dtype=np.float64
    )


def collect_specimens():
    """Return list of (category, label, path)."""
    specimens = []

    # Good teeth
    for td in sorted(glob(os.path.join(GOOD_DIR, "tooth_*"))):
        name = os.path.basename(td)
        if name in EXCLUDE_GOOD:
            print(f"  [skip good] {name}")
            continue
        ply = os.path.join(td, "corresponded.ply")
        if os.path.exists(ply):
            specimens.append(("Good", name.replace("tooth_", "T"), ply))

    # Worn teeth + both reconstruction types
    for td in sorted(glob(os.path.join(WORN_DIR, "tooth_*_wear_*"))):
        dname = os.path.basename(td)
        parts = dname.split("_wear_")
        if len(parts) != 2:
            continue
        tooth_id, wear_level = parts

        if tooth_id in EXCLUDE_WORN_TEETH:
            print(f"  [skip worn] {dname}")
            continue

        # Compact wear label: "level3" -> "3"
        wear_short = wear_level.replace("level", "")

        if "TEST" in tooth_id:
            tooth_short = tooth_id.replace("tooth_", "")           # TEST1 / TEST2
            cat_worn = "TEST"
        else:
            tooth_short = tooth_id.replace("tooth_", "T")          # T05
            cat_worn = "Worn"

        label_w = f"{tooth_short}_{wear_short}_w"
        label_g = f"{tooth_short}_{wear_short}_g"   # global recon
        label_n = f"{tooth_short}_{wear_short}_n"   # neighborhood recon

        worn_ply = os.path.join(td, "corresponded.ply")
        if os.path.exists(worn_ply):
            specimens.append((cat_worn, label_w, worn_ply))

        global_recon_ply = os.path.join(RECON_GLOBAL, dname, "reconstructed.ply")
        if os.path.exists(global_recon_ply):
            specimens.append(("Recon (Global)", label_g, global_recon_ply))

        nbr_recon_ply = os.path.join(RECON_NBR, dname, "reconstructed.ply")
        if os.path.exists(nbr_recon_ply):
            specimens.append(("Recon (Nbr)", label_n, nbr_recon_ply))

    # Corresponded TEST originals
    for td in sorted(glob(os.path.join(ORIG_DIR, "original_*"))):
        ply = os.path.join(td, "corresponded.ply")
        if os.path.exists(ply):
            name = os.path.basename(td).replace("original_", "")
            specimens.append(("Original", name, ply))

    return specimens


def main():
    print(f"Correspondence dir: {CORR_DIR}")
    print(f"Global recon dir:   {RECON_GLOBAL}")
    print(f"Nbr recon dir:      {RECON_NBR}")
    print()

    spec_meta = collect_specimens()
    print(f"\nLoading {len(spec_meta)} specimens (zoomed view)...\n")

    # Tally per category
    from collections import Counter
    print("  Counts per category:", dict(Counter(c for c, _, _ in spec_meta)))
    print()

    clouds = []
    for cat, lbl, path in tqdm(spec_meta, desc="Loading"):
        clouds.append(load_ply(path).flatten())

    n_pts = min(c.size for c in clouds)
    X = np.stack([c[:n_pts] for c in clouds], axis=0)
    print(f"\nData matrix: {X.shape}")

    pca = PCA(n_components=3, random_state=42)
    Z = pca.fit_transform(X)
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}  "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}  "
          f"PC3={pca.explained_variance_ratio_[2]:.3f}")

    cats = [c for c, _, _ in spec_meta]
    labels = [l for _, l, _ in spec_meta]

    # ── Style per category ──────────────────────────────────────────────────
    style = {
        "Good":           dict(c="tab:blue",   marker="o",  s=80,  alpha=0.85),
        "Worn":           dict(c="tab:red",    marker="^",  s=70,  alpha=0.85),
        "Recon (Global)": dict(c="tab:purple", marker="X",  s=90,  alpha=0.85),
        "Recon (Nbr)":    dict(c="tab:pink",   marker="*",  s=120, alpha=0.85),
        "Original":       dict(c="tab:orange", marker="p",  s=120, alpha=0.95),
        "TEST":           dict(c="tab:green",  marker="D",  s=80,  alpha=0.85),
    }

    # ── 2D plot — clean (no per-point labels, only category in legend) ─────
    fig, ax = plt.subplots(figsize=(12, 9))
    for cat in style:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax.scatter(Z[idx, 0], Z[idx, 1],
                   label=f"{cat} (n={len(idx)})",
                   edgecolors="black", linewidths=0.4,
                   **style[cat])
    # Annotate everything except the dense Worn cluster so it stays readable.
    # Worn points stay un-labelled — their paired g/n recons carry the same tooth id.
    annotate_color = {
        "Good":           "tab:blue",
        "Original":       "tab:orange",
        "Recon (Global)": "tab:purple",
        "Recon (Nbr)":    "deeppink",
        "TEST":           "tab:green",
    }
    for tag, col in annotate_color.items():
        for i, c in enumerate(cats):
            if c == tag:
                ax.annotate(labels[i], (Z[i, 0], Z[i, 1]),
                            fontsize=6, alpha=0.85, color=col,
                            xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA 2D — Global vs Neighborhood reconstruction (outlier cluster removed)")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out2d = os.path.join(PLOT_DIR, "pca_2d_zoomed.png")
    fig.tight_layout()
    fig.savefig(out2d, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out2d}")

    # ── 2D plot — labelled version (everything labelled, separate file) ────
    fig, ax = plt.subplots(figsize=(16, 11))
    for cat in style:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax.scatter(Z[idx, 0], Z[idx, 1],
                   label=f"{cat} (n={len(idx)})",
                   edgecolors="black", linewidths=0.4,
                   **style[cat])
        for i in idx:
            ax.annotate(labels[i], (Z[i, 0], Z[i, 1]),
                        fontsize=6, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA 2D (all labels) — Global vs Neighborhood reconstruction")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out2d_labelled = os.path.join(PLOT_DIR, "pca_2d_zoomed_labelled.png")
    fig.tight_layout()
    fig.savefig(out2d_labelled, dpi=200)
    plt.close(fig)
    print(f"Saved: {out2d_labelled}")

    # ── 3D plot — clean (no labels) ───────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    ax3 = fig.add_subplot(111, projection="3d")
    for cat in style:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax3.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2],
                    label=f"{cat} (n={len(idx)})",
                    edgecolors="black", linewidths=0.4,
                    **style[cat])
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax3.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)")
    ax3.set_title("PCA 3D — Global vs Neighborhood reconstruction (outlier cluster removed)")
    ax3.legend(loc="best", fontsize=10)
    out3d = os.path.join(PLOT_DIR, "pca_3d_zoomed.png")
    fig.tight_layout()
    fig.savefig(out3d, dpi=200)
    plt.close(fig)
    print(f"Saved: {out3d}")


if __name__ == "__main__":
    main()
