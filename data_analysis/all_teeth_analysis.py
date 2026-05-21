"""
Combined analysis of good, worn, and SSM-reconstructed teeth.

Loads corresponded point clouds for good and worn teeth, plus the
SSM reconstruction outputs, and visualises all three groups in:
  - PCA space (scree, 2-D, 3-D)
  - t-SNE from leading PCs
  - t-SNE from raw 300 k-D feature vectors
  - UMAP from leading PCs
  - UMAP from raw 300 k-D feature vectors
  - Distance bar chart (worn vs. reconstructed, relative to good centroid)

PCA is fit on good teeth only; worn and reconstructed are projected in.

Usage:
    python all_teeth_analysis.py
    python all_teeth_analysis.py --correspondence-dir ../ssm_pipeline/output/correspondence_all_100k
    python all_teeth_analysis.py --extra-recon-dir ../ssm_pipeline/output/recon_local_t14/reconstructions
    python all_teeth_analysis.py --no-arrows          # cleaner PCA 2-D plot
    python all_teeth_analysis.py --no-tsne            # skip t-SNE plots
    python all_teeth_analysis.py --no-umap            # skip UMAP plots
    python all_teeth_analysis.py --tsne-perplexity 10
    python all_teeth_analysis.py --umap-n-neighbors 10 --umap-min-dist 0.3
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import re
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

HAS_UMAP = False
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_v2")

DEFAULT_CORR_BASE = os.path.join(
    PROJECT_DIR, "ssm_pipeline", "output", "correspondence_all_100k"
)
RECON_DIR = os.path.join(
    PROJECT_DIR, "ssm_pipeline", "output", "recon_neighborhood_v4", "reconstructions"
)

# ── Style definitions ─────────────────────────────────────────────

ORIGINAL_DIR = os.path.join(PROJECT_DIR, "Original models")
ORIGINAL_MAP = {
    "cprc_nyu_n0245_ULM3_EDJ.ply": "TEST1",
    "cprc_nyu_n0257_ULM3_EDJ.ply": "TEST2",
}
ORIGINAL_CORR_MAP = {
    "original_TEST1": ("n0245", "TEST1"),
    "original_TEST2": ("n0257", "TEST2"),
}

STYLE = {
    "Good":            {"color": "#4C72B0", "marker": "o", "size": 120},
    "Original":        {"color": "#E5A030", "marker": "p", "size": 160},
    "Real worn":       {"color": "#C44E52", "marker": "^", "size": 110},
    "TEST1":           {"color": "#55A868", "marker": "D", "size": 100},
    "TEST2":           {"color": "#8172B2", "marker": "s", "size": 100},
    "Recon (Real)":    {"color": "#E8868B", "marker": "*", "size": 160},
    "Recon (TEST1)":   {"color": "#8FCC99", "marker": "*", "size": 160},
    "Recon (TEST2)":   {"color": "#B3A8D8", "marker": "*", "size": 160},
    # Local-mean / alternate reconstructions (e.g. recon_local_t14)
    "Recon local (Real)":  {"color": "#DD8452", "marker": "X", "size": 140},
    "Recon local (TEST1)": {"color": "#6BAF7A", "marker": "X", "size": 140},
    "Recon local (TEST2)": {"color": "#9B8FC9", "marker": "X", "size": 140},
}

WORN_TO_RECON_COLOR = {
    "Real worn": "Recon (Real)",
    "TEST1":     "Recon (TEST1)",
    "TEST2":     "Recon (TEST2)",
}

WORN_TO_LOCAL_RECON_COLOR = {
    "Real worn": "Recon local (Real)",
    "TEST1":     "Recon local (TEST1)",
    "TEST2":     "Recon local (TEST2)",
}


# ── Helpers ───────────────────────────────────────────────────────

def _load_ply(path: str) -> np.ndarray:
    pc = trimesh.load(path, process=False)
    return np.asarray(
        pc.vertices if hasattr(pc, "vertices") else pc, dtype=np.float64
    )


def _normalize(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ Vt.T
    for ax in range(3):
        if np.sum(rotated[:, ax] > 0) < len(rotated) // 2:
            rotated[:, ax] *= -1
    diag = rotated.max(axis=0) - rotated.min(axis=0)
    scale = np.linalg.norm(diag)
    if scale > 0:
        rotated /= scale
    return rotated


def _icp_align(source: np.ndarray, target: np.ndarray,
               max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    from scipy.spatial import cKDTree
    src = source.copy()
    tree = cKDTree(target)
    prev_err = np.inf
    for _ in range(max_iter):
        dists, idx = tree.query(src)
        corr = target[idx]
        src_c = src.mean(axis=0)
        corr_c = corr.mean(axis=0)
        H = (src - src_c).T @ (corr - corr_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = corr_c - R @ src_c
        src = (R @ src.T).T + t
        err = float(np.mean(dists))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return src


def load_originals_corresponded(corr_orig_dir):
    """Load corresponded original teeth (preferred -- exact same pipeline as all others)."""
    orig_clouds, orig_labels, orig_test_group = [], [], []
    for dirname, (label, test_name) in sorted(ORIGINAL_CORR_MAP.items()):
        ply = os.path.join(corr_orig_dir, dirname, "corresponded.ply")
        if not os.path.exists(ply):
            continue
        orig_clouds.append(_load_ply(ply))
        orig_labels.append(label)
        orig_test_group.append(test_name)
        print(f"  Loaded corresponded {dirname} -> {label} ({test_name})")
    return orig_clouds, orig_labels, orig_test_group


def load_originals_raw(original_dir, template_cloud, n_points=100000, seed=42):
    """Fallback: load raw meshes, sample, normalize, ICP-align (no CPD correspondence)."""
    orig_clouds, orig_labels, orig_test_group = [], [], []
    for fname, test_name in sorted(ORIGINAL_MAP.items()):
        fpath = os.path.join(original_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP original] {fname} -- not found")
            continue
        mesh = trimesh.load(fpath, force="mesh", process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
        pts = np.asarray(pts, dtype=np.float64)
        pts = _normalize(pts)
        pts = _icp_align(pts, template_cloud)
        orig_clouds.append(pts)
        m = re.search(r"n(\d+)", fname)
        orig_labels.append(f"n{m.group(1)}" if m else fname[:12])
        orig_test_group.append(test_name)
        print(f"  Loaded raw original {fname} -> {orig_labels[-1]} ({test_name})")
    return orig_clouds, orig_labels, orig_test_group


def classify_worn_dir(dirname: str):
    if "TEST1" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return "TEST1", int(m.group(1)) if m else 0
    if "TEST2" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return "TEST2", int(m.group(1)) if m else 0
    return "Real worn", 0


def worn_dir_label(dirname: str) -> str:
    if "TEST1" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return f"T1_L{m.group(1)}" if m else "TEST1"
    if "TEST2" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return f"T2_L{m.group(1)}" if m else "TEST2"
    m = re.search(r"tooth_(\d+)", dirname)
    return f"T{m.group(1)}_w" if m else dirname[:12]


def recon_label_from_worn(worn_label: str) -> str:
    """Same naming as global recon in load_all (star / triangle pairs in plots)."""
    if "_L" not in worn_label:
        return worn_label.replace("_w", "_r").replace("_L", "r_L")
    return worn_label + "r"


def local_recon_label_from_worn(worn_label: str) -> str:
    """Distinct annotation for local-mean (or other alternate) reconstructions."""
    return recon_label_from_worn(worn_label) + "L"


# ── Loading ───────────────────────────────────────────────────────

def load_all(corr_good_dir, corr_worn_dir, recon_dir):
    """Return good/worn/recon clouds, metadata, and worn_keys (folder names)."""

    # Good teeth
    good_dirs = sorted(glob(os.path.join(corr_good_dir, "tooth_*")))
    good_files, good_labels = [], []
    for td in good_dirs:
        ply = os.path.join(td, "corresponded.ply")
        if os.path.exists(ply):
            good_files.append(ply)
            good_labels.append(os.path.basename(td).replace("tooth_", "T"))

    # Worn teeth
    worn_dirs = sorted(glob(os.path.join(corr_worn_dir, "tooth_*")))
    worn_files, worn_labels, worn_groups, worn_keys = [], [], [], []
    for td in worn_dirs:
        ply = os.path.join(td, "corresponded.ply")
        if os.path.exists(ply):
            dname = os.path.basename(td)
            worn_files.append(ply)
            worn_labels.append(worn_dir_label(dname))
            worn_groups.append(classify_worn_dir(dname))
            worn_keys.append(dname)

    # Reconstructed teeth (matched by directory name)
    recon_files, recon_labels, recon_groups = [], [], []
    matched_worn_idx = []
    for wi, key in enumerate(worn_keys):
        rply = os.path.join(recon_dir, key, "reconstructed.ply")
        if os.path.exists(rply):
            recon_files.append(rply)
            recon_labels.append(recon_label_from_worn(worn_labels[wi]))
            recon_groups.append(worn_groups[wi])
            matched_worn_idx.append(wi)
        else:
            print(f"  [SKIP recon] {key} -- no reconstructed.ply")

    # Load point clouds
    good_clouds = [_load_ply(f) for f in tqdm(good_files, desc="Loading good teeth")]
    worn_clouds = [_load_ply(f) for f in tqdm(worn_files, desc="Loading worn teeth")]
    recon_clouds = [_load_ply(f) for f in tqdm(recon_files, desc="Loading reconstructed")]

    return (good_clouds, good_labels,
            worn_clouds, worn_labels, worn_groups,
            recon_clouds, recon_labels, recon_groups,
            matched_worn_idx, worn_keys)


def load_extra_reconstructions(extra_recon_dir, worn_keys, worn_labels, worn_groups):
    """
    Load alternate reconstructions (e.g. local-mean SSM) keyed by the same
    artificial_worn directory names as the primary recon set.
    """
    if not extra_recon_dir or not os.path.isdir(extra_recon_dir):
        return [], [], [], []

    extra_files, extra_labels, extra_groups, extra_wi = [], [], [], []
    for wi, key in enumerate(worn_keys):
        rply = os.path.join(extra_recon_dir, key, "reconstructed.ply")
        if not os.path.exists(rply):
            continue
        extra_files.append(rply)
        extra_labels.append(local_recon_label_from_worn(worn_labels[wi]))
        extra_groups.append(worn_groups[wi])
        extra_wi.append(wi)
        print(f"  [extra recon] {key} -> {extra_labels[-1]}")

    if not extra_files:
        return [], [], [], []

    extra_clouds = [_load_ply(f) for f in tqdm(extra_files, desc="Loading extra recon")]
    return extra_clouds, extra_labels, extra_groups, extra_wi


# ── Plotting helpers ──────────────────────────────────────────────

def _legend_handles(include_recon=True, include_original=True,
                    include_local_recon=False):
    keys = ["Good"]
    if include_original:
        keys.append("Original")
    keys += ["Real worn", "TEST1", "TEST2"]
    if include_recon:
        keys += ["Recon (Real)", "Recon (TEST1)", "Recon (TEST2)"]
    if include_local_recon:
        keys += ["Recon local (Real)", "Recon local (TEST1)", "Recon local (TEST2)"]
    return [
        Line2D([0], [0], marker=STYLE[k]["marker"], color="w",
               markerfacecolor=STYLE[k]["color"], markersize=10,
               markeredgecolor="k", label=k)
        for k in keys
    ]


def _plot_paired_distance(worn_2d, recon_2d, worn_labels, worn_groups,
                          matched_worn_idx, filename, title, ax_label):
    """Bar chart: Euclidean distance from each reconstructed tooth to its worn counterpart."""
    n_pairs = len(matched_worn_idx)
    dists = np.array([
        np.linalg.norm(recon_2d[ri] - worn_2d[matched_worn_idx[ri]])
        for ri in range(n_pairs)
    ])
    labels = [worn_labels[matched_worn_idx[ri]] for ri in range(n_pairs)]
    colors = [STYLE[worn_groups[matched_worn_idx[ri]][0]]["color"]
              for ri in range(n_pairs)]

    sort_idx = np.argsort(dists)[::-1]
    fig, ax = plt.subplots(figsize=(11, max(5, n_pairs * 0.32)))
    y_pos = range(len(sort_idx))
    ax.barh(y_pos, dists[sort_idx],
            color=[colors[i] for i in sort_idx],
            edgecolor="k", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in sort_idx], fontsize=8)
    ax.set_xlabel(f"Euclidean Distance ({ax_label})", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.invert_yaxis()

    legend_bar = [
        Line2D([0], [0], color=STYLE[k]["color"], linewidth=8, label=k)
        for k in ["Real worn", "TEST1", "TEST2"]
    ]
    ax.legend(handles=legend_bar, fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close(fig)
    print(f"Saved {filename}")

    return dists, labels


def _scatter_group(ax, pts_2d, labels, group_key, annotate=True,
                   fontsize=7, text_offset=(6, 6)):
    s = STYLE[group_key]
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1],
               c=s["color"], marker=s["marker"], s=s["size"],
               edgecolors="k", linewidths=0.5, zorder=4 if group_key == "Good" else 3)
    if annotate:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (pts_2d[i, 0], pts_2d[i, 1]),
                        textcoords="offset points", xytext=text_offset,
                        fontsize=fontsize, color=s["color"])


def _scatter_extra_recons_2d(ax, sc_extra, extra_labels, extra_groups,
                             fontsize=5, text_offset=(6, -10)):
    for ei in range(len(sc_extra)):
        grp = extra_groups[ei][0]
        rkey = WORN_TO_LOCAL_RECON_COLOR[grp]
        s = STYLE[rkey]
        ax.scatter(sc_extra[ei, 0], sc_extra[ei, 1],
                   c=s["color"], marker=s["marker"], s=s["size"],
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(extra_labels[ei], (sc_extra[ei, 0], sc_extra[ei, 1]),
                    textcoords="offset points", xytext=text_offset,
                    fontsize=fontsize, color=s["color"])


def _scatter_extra_recons_3d(ax3, sc_extra, extra_groups):
    for ei in range(len(sc_extra)):
        grp = extra_groups[ei][0]
        rkey = WORN_TO_LOCAL_RECON_COLOR[grp]
        rs = STYLE[rkey]
        ax3.scatter(sc_extra[ei, 0], sc_extra[ei, 1], sc_extra[ei, 2],
                    c=rs["color"], marker=rs["marker"], s=rs["size"],
                    edgecolors="k", linewidths=0.3, depthshade=True)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined PCA / t-SNE analysis of good, worn, and reconstructed teeth")
    parser.add_argument("--correspondence-dir", type=str, default=DEFAULT_CORR_BASE,
                        help="Correspondence output root (good_teeth/, artificial_worn/)")
    parser.add_argument("--recon-dir", type=str, default=RECON_DIR,
                        help="Path to reconstruction output directory")
    parser.add_argument("--extra-recon-dir", type=str, default=None,
                        help="Optional second recon root (e.g. .../recon_local_t14/reconstructions). "
                             "Same folder names as primary recon; plotted as orange X markers "
                             "(labels end with L, e.g. T03_rL).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE plots")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (clamped to n_samples-1)")
    parser.add_argument("--tsne-pc-dims", type=int, default=10,
                        help="Number of leading PCs to feed t-SNE (default 10)")
    parser.add_argument("--no-umap", action="store_true",
                        help="Skip UMAP plots")
    parser.add_argument("--umap-n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default 0.1)")
    parser.add_argument("--no-arrows", action="store_true",
                        help="Skip worn->recon arrows on PCA 2-D plot")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    corr_base = os.path.abspath(args.correspondence_dir)
    corr_good_dir = os.path.join(corr_base, "good_teeth")
    corr_worn_dir = os.path.join(corr_base, "artificial_worn")
    corr_orig_dir = os.path.join(corr_base, "originals")

    # ── Load data ─────────────────────────────────────────────────
    print(f"Correspondence dir : {corr_base}")
    print(f"Reconstruction dir : {args.recon_dir}")
    if not os.path.isdir(corr_good_dir):
        sys.exit(f"Good-teeth correspondence dir not found: {corr_good_dir}")

    (good_clouds, good_labels,
     worn_clouds, worn_labels, worn_groups,
     recon_clouds, recon_labels, recon_groups,
     matched_worn_idx, worn_keys) = load_all(corr_good_dir, corr_worn_dir, args.recon_dir)

    extra_recon_dir = args.extra_recon_dir
    if extra_recon_dir:
        extra_recon_dir = os.path.abspath(extra_recon_dir)
        print(f"Extra reconstruction dir : {extra_recon_dir}")
    extra_clouds, extra_labels, extra_groups, extra_wi = load_extra_reconstructions(
        extra_recon_dir, worn_keys, worn_labels, worn_groups)

    n_good = len(good_clouds)
    n_worn = len(worn_clouds)
    n_recon = len(recon_clouds)
    n_extra = len(extra_clouds)
    n_pts = good_clouds[0].shape[0]

    print(f"\nGood teeth : {n_good}  ({n_pts} pts each)")
    print(f"Worn teeth : {n_worn}  "
          f"({sum(1 for g, _ in worn_groups if g == 'Real worn')} real, "
          f"{sum(1 for g, _ in worn_groups if g == 'TEST1')} TEST1, "
          f"{sum(1 for g, _ in worn_groups if g == 'TEST2')} TEST2)")
    print(f"Reconstructed : {n_recon}")
    if n_extra:
        print(f"Extra (local) reconstructions : {n_extra}")

    # Load original (unworn) teeth for TEST1/TEST2
    # Prefer corresponded outputs; fall back to raw mesh + ICP
    if os.path.isdir(corr_orig_dir):
        print(f"\nLoading corresponded originals from: {corr_orig_dir}")
        orig_clouds, orig_labels, orig_test_group = load_originals_corresponded(corr_orig_dir)
    else:
        orig_clouds, orig_labels, orig_test_group = [], [], []

    if not orig_clouds:
        template_cloud = good_clouds[0]
        print(f"  No corresponded originals found -- falling back to raw meshes + ICP")
        print(f"  (Run ssm_pipeline/correspond_originals.py first for exact results)")
        orig_clouds, orig_labels, orig_test_group = load_originals_raw(
            ORIGINAL_DIR, template_cloud, n_points=n_pts, seed=args.seed)
    n_orig = len(orig_clouds)
    print(f"Originals : {n_orig}")

    if n_good < 3:
        sys.exit("Need at least 3 good teeth for PCA")

    # Feature matrices
    X_good = np.array([c.flatten() for c in good_clouds])
    X_worn = np.array([c.flatten() for c in worn_clouds])
    X_recon = np.array([c.flatten() for c in recon_clouds])
    X_extra = np.array([c.flatten() for c in extra_clouds]) if n_extra else np.empty((0, X_good.shape[1]))
    X_orig = np.array([c.flatten() for c in orig_clouds]) if n_orig > 0 else np.empty((0, X_good.shape[1]))

    # ── PCA (fit on good, project worn + recon + originals) ───────
    n_comp = min(n_good - 1, 10)
    pca = PCA(n_components=n_comp)
    sc_good = pca.fit_transform(X_good)
    sc_worn = pca.transform(X_worn)
    sc_recon = pca.transform(X_recon)
    sc_extra = pca.transform(X_extra) if n_extra else np.empty((0, n_comp))
    sc_orig = pca.transform(X_orig) if n_orig > 0 else np.empty((0, n_comp))
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print(f"\nPCA on {n_good} good teeth ({n_comp} components):")
    for i in range(min(8, n_comp)):
        print(f"  PC{i+1}: {var_ratio[i]*100:6.2f}%  (cumulative {cum_var[i]*100:6.2f}%)")

    # ── Plot 1: Scree ─────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))
    pcs = np.arange(1, len(var_ratio) + 1)
    ax1.bar(pcs, var_ratio * 100, color="#4C72B0", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Variance Explained (%)", fontsize=12)
    ax1.set_xticks(pcs)

    ax2 = ax1.twinx()
    ax2.plot(pcs, cum_var * 100, "o-", color="#C44E52", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12)
    ax2.set_ylim(0, 105)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=10)
    ax1.set_title(f"PCA Scree Plot  ({n_good} Good Teeth)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "all_teeth_scree.png"), dpi=200)
    plt.close(fig)
    print("Saved all_teeth_scree.png")

    # ── Plot 2: PCA 2-D (PC1 vs PC2) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9))

    # Good
    _scatter_group(ax, sc_good[:, :2], good_labels, "Good")

    # Worn
    for wi in range(n_worn):
        grp = worn_groups[wi][0]
        s = STYLE[grp]
        ax.scatter(sc_worn[wi, 0], sc_worn[wi, 1],
                   c=s["color"], marker=s["marker"], s=s["size"],
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(worn_labels[wi], (sc_worn[wi, 0], sc_worn[wi, 1]),
                    textcoords="offset points", xytext=(6, -8),
                    fontsize=6, color=s["color"])

    # Reconstructed
    for ri in range(n_recon):
        grp = recon_groups[ri][0]
        rkey = WORN_TO_RECON_COLOR[grp]
        s = STYLE[rkey]
        ax.scatter(sc_recon[ri, 0], sc_recon[ri, 1],
                   c=s["color"], marker=s["marker"], s=s["size"],
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(recon_labels[ri], (sc_recon[ri, 0], sc_recon[ri, 1]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=5, color=s["color"])

    if n_extra:
        _scatter_extra_recons_2d(ax, sc_extra[:, :2], extra_labels, extra_groups)

    # Originals
    if n_orig > 0:
        _scatter_group(ax, sc_orig[:, :2], orig_labels, "Original",
                       fontsize=8, text_offset=(6, 8))

    # Arrows worn -> recon
    if not args.no_arrows:
        for ri, wi in enumerate(matched_worn_idx):
            grp = worn_groups[wi][0]
            ax.annotate("",
                        xy=(sc_recon[ri, 0], sc_recon[ri, 1]),
                        xytext=(sc_worn[wi, 0], sc_worn[wi, 1]),
                        arrowprops=dict(arrowstyle="->", color=STYLE[grp]["color"],
                                        lw=1.0, linestyle="--", alpha=0.6))
        for ei, wi in enumerate(extra_wi):
            grp = worn_groups[wi][0]
            ax.annotate("",
                        xy=(sc_extra[ei, 0], sc_extra[ei, 1]),
                        xytext=(sc_worn[wi, 0], sc_worn[wi, 1]),
                        arrowprops=dict(arrowstyle="->", color=STYLE[grp]["color"],
                                        lw=1.2, linestyle=":", alpha=0.75))

    ax.legend(handles=_legend_handles(include_local_recon=n_extra > 0), fontsize=9, loc="best")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Good + Worn + Reconstructed + Originals  (PC1 vs PC2)", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "all_teeth_pca_2d.png"), dpi=200)
    plt.close(fig)
    print("Saved all_teeth_pca_2d.png")

    # Paired worn<->recon distance in PC1-PC2
    _plot_paired_distance(
        sc_worn[:, :2], sc_recon[:, :2],
        worn_labels, worn_groups, matched_worn_idx,
        "paired_dist_pca_2d.png",
        "Worn-to-Recon Distance  (PC1-PC2)", "PC1-PC2 space")

    if n_extra:
        _plot_paired_distance(
            sc_worn[:, :2], sc_extra[:, :2],
            worn_labels, worn_groups, extra_wi,
            "paired_dist_pca_2d_local.png",
            "Worn-to-Local-Recon Distance  (PC1-PC2)", "PC1-PC2 space")

    # ── Plot 3: PCA 3-D ──────────────────────────────────────────
    if n_comp >= 3:
        fig = plt.figure(figsize=(12, 9))
        ax3 = fig.add_subplot(111, projection="3d")

        s = STYLE["Good"]
        ax3.scatter(sc_good[:, 0], sc_good[:, 1], sc_good[:, 2],
                    c=s["color"], marker=s["marker"], s=s["size"],
                    edgecolors="k", linewidths=0.4, depthshade=True)
        for i, lbl in enumerate(good_labels):
            ax3.text(sc_good[i, 0], sc_good[i, 1], sc_good[i, 2],
                     f"  {lbl}", fontsize=6, color=s["color"])

        for wi in range(n_worn):
            grp = worn_groups[wi][0]
            ws = STYLE[grp]
            ax3.scatter(sc_worn[wi, 0], sc_worn[wi, 1], sc_worn[wi, 2],
                        c=ws["color"], marker=ws["marker"], s=ws["size"],
                        edgecolors="k", linewidths=0.3, depthshade=True)

        for ri in range(n_recon):
            grp = recon_groups[ri][0]
            rkey = WORN_TO_RECON_COLOR[grp]
            rs = STYLE[rkey]
            ax3.scatter(sc_recon[ri, 0], sc_recon[ri, 1], sc_recon[ri, 2],
                        c=rs["color"], marker=rs["marker"], s=rs["size"],
                        edgecolors="k", linewidths=0.3, depthshade=True)

        if n_extra:
            _scatter_extra_recons_3d(ax3, sc_extra[:, :3], extra_groups)

        if n_orig > 0:
            os_ = STYLE["Original"]
            ax3.scatter(sc_orig[:, 0], sc_orig[:, 1], sc_orig[:, 2],
                        c=os_["color"], marker=os_["marker"], s=os_["size"],
                        edgecolors="k", linewidths=0.4, depthshade=True)
            for i, lbl in enumerate(orig_labels):
                ax3.text(sc_orig[i, 0], sc_orig[i, 1], sc_orig[i, 2],
                         f"  {lbl}", fontsize=7, color=os_["color"])

        ax3.legend(handles=_legend_handles(include_local_recon=n_extra > 0),
                   fontsize=8, loc="upper left")
        ax3.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax3.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax3.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
        ax3.set_title("All Teeth  (PC1-PC2-PC3)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "all_teeth_pca_3d.png"), dpi=200)
        plt.close(fig)
        print("Saved all_teeth_pca_3d.png")

    # ── t-SNE from PCs ────────────────────────────────────────────
    n_all = n_good + n_worn + n_recon + n_extra + n_orig
    if not args.no_tsne and n_all >= 3:
        d_tsne = min(max(2, args.tsne_pc_dims), n_comp)
        stack_parts = [sc_good[:, :d_tsne], sc_worn[:, :d_tsne],
                       sc_recon[:, :d_tsne]]
        if n_extra:
            stack_parts.append(sc_extra[:, :d_tsne])
        if n_orig > 0:
            stack_parts.append(sc_orig[:, :d_tsne])
        X_stack = np.vstack(stack_parts)
        max_perp = max(2.0, float(n_all - 1) - 1e-6)
        perp = float(np.clip(args.tsne_perplexity, 2.0, max_perp))
        print(f"\nRunning t-SNE on PCs ({n_all} specimens, dims={d_tsne}, "
              f"perplexity={perp:.1f})...")
        kw = dict(n_components=2, perplexity=perp, random_state=args.seed,
                  init="pca")
        try:
            tsne = TSNE(**kw, learning_rate="auto")
        except TypeError:
            tsne = TSNE(**kw, learning_rate=200)
        Z = tsne.fit_transform(X_stack)
        off = 0
        Zg = Z[off:off + n_good]
        off += n_good
        Zw = Z[off:off + n_worn]
        off += n_worn
        Zr = Z[off:off + n_recon]
        off += n_recon
        Zrl = Z[off:off + n_extra] if n_extra else np.empty((0, 2))
        off += n_extra
        Zo = Z[off:] if n_orig > 0 else np.empty((0, 2))

        fig, ax = plt.subplots(figsize=(13, 9))
        _scatter_group(ax, Zg, good_labels, "Good")
        for wi in range(n_worn):
            grp = worn_groups[wi][0]
            s = STYLE[grp]
            ax.scatter(Zw[wi, 0], Zw[wi, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(worn_labels[wi], (Zw[wi, 0], Zw[wi, 1]),
                        textcoords="offset points", xytext=(6, -8),
                        fontsize=6, color=s["color"])
        for ri in range(n_recon):
            grp = recon_groups[ri][0]
            rkey = WORN_TO_RECON_COLOR[grp]
            s = STYLE[rkey]
            ax.scatter(Zr[ri, 0], Zr[ri, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(recon_labels[ri], (Zr[ri, 0], Zr[ri, 1]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=5, color=s["color"])
        if n_extra:
            _scatter_extra_recons_2d(ax, Zrl, extra_labels, extra_groups)
        if n_orig > 0:
            _scatter_group(ax, Zo, orig_labels, "Original",
                           fontsize=8, text_offset=(6, 8))

        ax.legend(handles=_legend_handles(include_local_recon=n_extra > 0),
                  fontsize=9, loc="best")
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(f"All Teeth t-SNE  (first {d_tsne} PCs, perplexity={perp:.1f})",
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "all_teeth_tsne.png"), dpi=200)
        plt.close(fig)
        print("Saved all_teeth_tsne.png")

        _plot_paired_distance(
            Zw, Zr, worn_labels, worn_groups, matched_worn_idx,
            "paired_dist_tsne.png",
            f"Worn-to-Recon Distance  (t-SNE, {d_tsne} PCs)", "t-SNE space")

        if n_extra:
            _plot_paired_distance(
                Zw, Zrl, worn_labels, worn_groups, extra_wi,
                "paired_dist_tsne_local.png",
                f"Worn-to-Local-Recon Distance  (t-SNE, {d_tsne} PCs)",
                "t-SNE space")

    # ── t-SNE from raw 300 k-D features ──────────────────────────
    if not args.no_tsne and n_all >= 3:
        raw_parts = [X_good, X_worn, X_recon]
        if n_extra:
            raw_parts.append(X_extra)
        if n_orig > 0:
            raw_parts.append(X_orig)
        X_all_raw = np.vstack(raw_parts)
        max_perp = max(2.0, float(n_all - 1) - 1e-6)
        perp = float(np.clip(args.tsne_perplexity, 2.0, max_perp))
        n_feat = X_all_raw.shape[1]
        print(f"\nRunning t-SNE on raw features ({n_feat}D, {n_all} specimens, "
              f"perplexity={perp:.1f})...")
        kw2 = dict(n_components=2, perplexity=perp, random_state=args.seed,
                   metric="euclidean")
        try:
            tsne2 = TSNE(**kw2, learning_rate="auto", init="pca")
        except TypeError:
            tsne2 = TSNE(**kw2, learning_rate=200, init="random")
        Z2 = tsne2.fit_transform(X_all_raw)
        off2 = 0
        Z2g = Z2[off2:off2 + n_good]
        off2 += n_good
        Z2w = Z2[off2:off2 + n_worn]
        off2 += n_worn
        Z2r = Z2[off2:off2 + n_recon]
        off2 += n_recon
        Z2rl = Z2[off2:off2 + n_extra] if n_extra else np.empty((0, 2))
        off2 += n_extra
        Z2o = Z2[off2:] if n_orig > 0 else np.empty((0, 2))

        fig, ax = plt.subplots(figsize=(13, 9))
        _scatter_group(ax, Z2g, good_labels, "Good")
        for wi in range(n_worn):
            grp = worn_groups[wi][0]
            s = STYLE[grp]
            ax.scatter(Z2w[wi, 0], Z2w[wi, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(worn_labels[wi], (Z2w[wi, 0], Z2w[wi, 1]),
                        textcoords="offset points", xytext=(6, -8),
                        fontsize=6, color=s["color"])
        for ri in range(n_recon):
            grp = recon_groups[ri][0]
            rkey = WORN_TO_RECON_COLOR[grp]
            s = STYLE[rkey]
            ax.scatter(Z2r[ri, 0], Z2r[ri, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(recon_labels[ri], (Z2r[ri, 0], Z2r[ri, 1]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=5, color=s["color"])
        if n_extra:
            _scatter_extra_recons_2d(ax, Z2rl, extra_labels, extra_groups)
        if n_orig > 0:
            _scatter_group(ax, Z2o, orig_labels, "Original",
                           fontsize=8, text_offset=(6, 8))

        ax.legend(handles=_legend_handles(include_local_recon=n_extra > 0),
                  fontsize=9, loc="best")
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(f"All Teeth t-SNE  (full {n_feat}D features, "
                     f"perplexity={perp:.1f})", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "all_teeth_tsne_raw.png"), dpi=200)
        plt.close(fig)
        print("Saved all_teeth_tsne_raw.png")

        _plot_paired_distance(
            Z2w, Z2r, worn_labels, worn_groups, matched_worn_idx,
            "paired_dist_tsne_raw.png",
            f"Worn-to-Recon Distance  (t-SNE, raw {n_feat}D)", "t-SNE space")

        if n_extra:
            _plot_paired_distance(
                Z2w, Z2rl, worn_labels, worn_groups, extra_wi,
                "paired_dist_tsne_raw_local.png",
                f"Worn-to-Local-Recon Distance  (t-SNE, raw {n_feat}D)",
                "t-SNE space")

    # ── UMAP from PCs ─────────────────────────────────────────────
    run_umap = (not args.no_umap) and HAS_UMAP and n_all >= 3
    if not args.no_umap and not HAS_UMAP:
        print("\n[SKIP] UMAP not available -- install with: pip install umap-learn")

    if run_umap:
        d_umap = min(max(2, args.tsne_pc_dims), n_comp)
        umap_pc_parts = [sc_good[:, :d_umap], sc_worn[:, :d_umap],
                         sc_recon[:, :d_umap]]
        if n_extra:
            umap_pc_parts.append(sc_extra[:, :d_umap])
        if n_orig > 0:
            umap_pc_parts.append(sc_orig[:, :d_umap])
        X_stack_u = np.vstack(umap_pc_parts)
        nn = min(args.umap_n_neighbors, n_all - 1)
        print(f"\nRunning UMAP on PCs ({n_all} specimens, dims={d_umap}, "
              f"n_neighbors={nn}, min_dist={args.umap_min_dist})...")
        reducer = UMAP(n_components=2, n_neighbors=nn,
                       min_dist=args.umap_min_dist, random_state=args.seed)
        U = reducer.fit_transform(X_stack_u)
        uo = 0
        Ug = U[uo:uo + n_good]
        uo += n_good
        Uw = U[uo:uo + n_worn]
        uo += n_worn
        Ur = U[uo:uo + n_recon]
        uo += n_recon
        Url = U[uo:uo + n_extra] if n_extra else np.empty((0, 2))
        uo += n_extra
        Uo = U[uo:] if n_orig > 0 else np.empty((0, 2))

        fig, ax = plt.subplots(figsize=(13, 9))
        _scatter_group(ax, Ug, good_labels, "Good")
        for wi in range(n_worn):
            grp = worn_groups[wi][0]
            s = STYLE[grp]
            ax.scatter(Uw[wi, 0], Uw[wi, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(worn_labels[wi], (Uw[wi, 0], Uw[wi, 1]),
                        textcoords="offset points", xytext=(6, -8),
                        fontsize=6, color=s["color"])
        for ri in range(n_recon):
            grp = recon_groups[ri][0]
            rkey = WORN_TO_RECON_COLOR[grp]
            s = STYLE[rkey]
            ax.scatter(Ur[ri, 0], Ur[ri, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(recon_labels[ri], (Ur[ri, 0], Ur[ri, 1]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=5, color=s["color"])
        if n_extra:
            _scatter_extra_recons_2d(ax, Url, extra_labels, extra_groups)
        if n_orig > 0:
            _scatter_group(ax, Uo, orig_labels, "Original",
                           fontsize=8, text_offset=(6, 8))

        ax.legend(handles=_legend_handles(include_local_recon=n_extra > 0),
                  fontsize=9, loc="best")
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(f"All Teeth UMAP  (first {d_umap} PCs, "
                     f"n_neighbors={nn})", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "all_teeth_umap.png"), dpi=200)
        plt.close(fig)
        print("Saved all_teeth_umap.png")

        _plot_paired_distance(
            Uw, Ur, worn_labels, worn_groups, matched_worn_idx,
            "paired_dist_umap.png",
            f"Worn-to-Recon Distance  (UMAP, {d_umap} PCs)", "UMAP space")

        if n_extra:
            _plot_paired_distance(
                Uw, Url, worn_labels, worn_groups, extra_wi,
                "paired_dist_umap_local.png",
                f"Worn-to-Local-Recon Distance  (UMAP, {d_umap} PCs)",
                "UMAP space")

    # ── UMAP from raw 300 k-D features ───────────────────────────
    if run_umap:
        umap_raw_parts = [X_good, X_worn, X_recon]
        if n_extra:
            umap_raw_parts.append(X_extra)
        if n_orig > 0:
            umap_raw_parts.append(X_orig)
        X_all_raw_u = np.vstack(umap_raw_parts)
        nn = min(args.umap_n_neighbors, n_all - 1)
        n_feat = X_all_raw_u.shape[1]
        print(f"\nRunning UMAP on raw features ({n_feat}D, {n_all} specimens, "
              f"n_neighbors={nn}, min_dist={args.umap_min_dist})...")
        reducer2 = UMAP(n_components=2, n_neighbors=nn,
                        min_dist=args.umap_min_dist, random_state=args.seed,
                        metric="euclidean")
        U2 = reducer2.fit_transform(X_all_raw_u)
        u2o = 0
        U2g = U2[u2o:u2o + n_good]
        u2o += n_good
        U2w = U2[u2o:u2o + n_worn]
        u2o += n_worn
        U2r = U2[u2o:u2o + n_recon]
        u2o += n_recon
        U2rl = U2[u2o:u2o + n_extra] if n_extra else np.empty((0, 2))
        u2o += n_extra
        U2o = U2[u2o:] if n_orig > 0 else np.empty((0, 2))

        fig, ax = plt.subplots(figsize=(13, 9))
        _scatter_group(ax, U2g, good_labels, "Good")
        for wi in range(n_worn):
            grp = worn_groups[wi][0]
            s = STYLE[grp]
            ax.scatter(U2w[wi, 0], U2w[wi, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(worn_labels[wi], (U2w[wi, 0], U2w[wi, 1]),
                        textcoords="offset points", xytext=(6, -8),
                        fontsize=6, color=s["color"])
        for ri in range(n_recon):
            grp = recon_groups[ri][0]
            rkey = WORN_TO_RECON_COLOR[grp]
            s = STYLE[rkey]
            ax.scatter(U2r[ri, 0], U2r[ri, 1], c=s["color"], marker=s["marker"],
                       s=s["size"], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(recon_labels[ri], (U2r[ri, 0], U2r[ri, 1]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=5, color=s["color"])
        if n_extra:
            _scatter_extra_recons_2d(ax, U2rl, extra_labels, extra_groups)
        if n_orig > 0:
            _scatter_group(ax, U2o, orig_labels, "Original",
                           fontsize=8, text_offset=(6, 8))

        ax.legend(handles=_legend_handles(include_local_recon=n_extra > 0),
                  fontsize=9, loc="best")
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(f"All Teeth UMAP  (full {n_feat}D features, "
                     f"n_neighbors={nn})", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "all_teeth_umap_raw.png"), dpi=200)
        plt.close(fig)
        print("Saved all_teeth_umap_raw.png")

        _plot_paired_distance(
            U2w, U2r, worn_labels, worn_groups, matched_worn_idx,
            "paired_dist_umap_raw.png",
            f"Worn-to-Recon Distance  (UMAP, raw {n_feat}D)", "UMAP space")

        if n_extra:
            _plot_paired_distance(
                U2w, U2rl, worn_labels, worn_groups, extra_wi,
                "paired_dist_umap_raw_local.png",
                f"Worn-to-Local-Recon Distance  (UMAP, raw {n_feat}D)",
                "UMAP space")

    # ── Distance bar chart ────────────────────────────────────────
    good_centroid = sc_good.mean(axis=0)
    dist_worn = np.linalg.norm(sc_worn - good_centroid, axis=1)
    dist_recon = np.linalg.norm(sc_recon - good_centroid, axis=1)
    dist_extra = (np.linalg.norm(sc_extra - good_centroid, axis=1)
                  if n_extra else np.array([]))
    dist_orig = np.linalg.norm(sc_orig - good_centroid, axis=1) if n_orig > 0 else np.array([])

    bar_labels, bar_dists, bar_colors, bar_types = [], [], [], []
    for oi in range(n_orig):
        bar_labels.append(orig_labels[oi])
        bar_dists.append(dist_orig[oi])
        bar_colors.append(STYLE["Original"]["color"])
        bar_types.append("original")
    for wi in range(n_worn):
        grp = worn_groups[wi][0]
        bar_labels.append(worn_labels[wi])
        bar_dists.append(dist_worn[wi])
        bar_colors.append(STYLE[grp]["color"])
        bar_types.append("worn")
    for ri in range(n_recon):
        grp = recon_groups[ri][0]
        rkey = WORN_TO_RECON_COLOR[grp]
        bar_labels.append(recon_labels[ri])
        bar_dists.append(dist_recon[ri])
        bar_colors.append(STYLE[rkey]["color"])
        bar_types.append("recon")
    for ei in range(n_extra):
        grp = extra_groups[ei][0]
        rkey = WORN_TO_LOCAL_RECON_COLOR[grp]
        bar_labels.append(extra_labels[ei])
        bar_dists.append(dist_extra[ei])
        bar_colors.append(STYLE[rkey]["color"])
        bar_types.append("recon_local")

    sort_idx = np.argsort(bar_dists)[::-1]
    fig, ax = plt.subplots(figsize=(13, max(8, len(bar_labels) * 0.28)))
    y_pos = range(len(sort_idx))
    ax.barh(y_pos, [bar_dists[i] for i in sort_idx],
            color=[bar_colors[i] for i in sort_idx],
            edgecolor="k", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([bar_labels[i] for i in sort_idx], fontsize=7)
    ax.set_xlabel("Euclidean Distance to Good-Teeth Centroid (PC space)", fontsize=11)
    ax.set_title("All Teeth: Distance from Good-Teeth Distribution",
                 fontsize=13)
    ax.invert_yaxis()

    legend_keys = ["Original", "Real worn", "TEST1", "TEST2",
                   "Recon (Real)", "Recon (TEST1)", "Recon (TEST2)"]
    if n_extra:
        legend_keys += ["Recon local (Real)", "Recon local (TEST1)",
                        "Recon local (TEST2)"]
    legend_bar = [
        Line2D([0], [0], color=STYLE[k]["color"], linewidth=8, label=k)
        for k in legend_keys
    ]
    ax.legend(handles=legend_bar, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "all_teeth_distance.png"), dpi=200)
    plt.close(fig)
    print("Saved all_teeth_distance.png")

    # ── Terminal summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Distance Summary  (to good-teeth centroid in PC space)")
    print(f"{'='*60}")
    print(f"{'Label':>18s}  {'Group':>14s}  {'Type':>6s}  {'Dist':>8s}")
    print(f"{'-'*60}")
    for i in sort_idx:
        print(f"{bar_labels[i]:>18s}  "
              f"{'':>14s}  "
              f"{bar_types[i]:>6s}  "
              f"{bar_dists[i]:8.4f}")

    # Paired comparison
    print(f"\n{'='*60}")
    print("Paired Comparison  (worn -> recon distance shift)")
    print(f"{'='*60}")
    print(f"{'Worn':>18s}  {'d_worn':>8s}  {'d_recon':>8s}  {'delta':>8s}  {'Closer?':>8s}")
    print(f"{'-'*60}")
    for ri, wi in enumerate(matched_worn_idx):
        dw = dist_worn[wi]
        dr = dist_recon[ri]
        delta = dr - dw
        closer = "Yes" if delta < 0 else "No"
        print(f"{worn_labels[wi]:>18s}  {dw:8.4f}  {dr:8.4f}  "
              f"{delta:+8.4f}  {closer:>8s}")

    n_closer = sum(1 for ri, wi in enumerate(matched_worn_idx)
                   if dist_recon[ri] < dist_worn[wi])
    print(f"\n{n_closer}/{n_recon} reconstructions are closer to good-teeth centroid "
          f"than their worn input.")

    # Direct worn-to-global-recon distance for ALL worn teeth
    print(f"\n{'='*70}")
    print("Worn-to-Global-Recon Distance  (direct, full PC space)")
    print("  Lower = reconstruction stays closer to the worn tooth")
    print(f"{'='*70}")
    print(f"{'Worn':>18s}  {'Group':>10s}  {'d(worn,recon)':>14s}")
    print(f"{'-'*70}")
    paired_dists_global = []
    for ri, wi in enumerate(matched_worn_idx):
        d = float(np.linalg.norm(sc_recon[ri] - sc_worn[wi]))
        grp = worn_groups[wi][0]
        paired_dists_global.append((worn_labels[wi], grp, d))
    paired_dists_global.sort(key=lambda x: x[2])
    for lbl, grp, d in paired_dists_global:
        print(f"{lbl:>18s}  {grp:>10s}  {d:14.4f}")
    avg_d = np.mean([d for _, _, d in paired_dists_global])
    print(f"\n{'Average':>18s}  {'':>10s}  {avg_d:14.4f}")

    if n_extra:
        print(f"\n{'='*60}")
        print("Local recon: distance to good-teeth centroid")
        print(f"{'='*60}")
        print(f"{'Worn':>18s}  {'d_worn':>8s}  {'d_local':>8s}  {'delta':>8s}  {'Closer?':>8s}")
        print(f"{'-'*60}")
        for ei, wi in enumerate(extra_wi):
            dw = dist_worn[wi]
            dl = dist_extra[ei]
            delta = dl - dw
            closer = "Yes" if delta < 0 else "No"
            print(f"{worn_labels[wi]:>18s}  {dw:8.4f}  {dl:8.4f}  "
                  f"{delta:+8.4f}  {closer:>8s}")

        # Direct worn-to-recon distance: how close is each recon to its worn input?
        # Build lookup: worn_key -> index in sc_recon (global recon)
        global_ri_by_wi = {wi: ri for ri, wi in enumerate(matched_worn_idx)}

        print(f"\n{'='*80}")
        print("Worn-to-Recon Distance  (direct, in full PC space)")
        print("  Lower = reconstruction stays closer to the worn tooth")
        print(f"{'='*80}")
        print(f"{'Worn':>18s}  {'d(w,global)':>12s}  {'d(w,local)':>12s}  "
              f"{'delta':>10s}  {'Local closer?':>14s}")
        print(f"{'-'*80}")
        for ei, wi in enumerate(extra_wi):
            d_local = float(np.linalg.norm(sc_extra[ei] - sc_worn[wi]))
            gri = global_ri_by_wi.get(wi)
            if gri is not None:
                d_global = float(np.linalg.norm(sc_recon[gri] - sc_worn[wi]))
                delta = d_local - d_global
                closer = "YES" if delta < 0 else "no"
                print(f"{worn_labels[wi]:>18s}  {d_global:12.4f}  {d_local:12.4f}  "
                      f"{delta:+10.4f}  {closer:>14s}")
            else:
                print(f"{worn_labels[wi]:>18s}  {'N/A':>12s}  {d_local:12.4f}  "
                      f"{'':>10s}  {'':>14s}")

        n_local_closer = sum(
            1 for ei, wi in enumerate(extra_wi)
            if wi in global_ri_by_wi
            and np.linalg.norm(sc_extra[ei] - sc_worn[wi])
                < np.linalg.norm(sc_recon[global_ri_by_wi[wi]] - sc_worn[wi]))
        n_comparable = sum(1 for wi in [w for _, w in enumerate(extra_wi)]
                           if wi in global_ri_by_wi)
        print(f"\n{n_local_closer}/{n_comparable} local reconstructions are "
              f"closer to their worn input than the global recon.")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
