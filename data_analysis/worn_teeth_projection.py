"""
Project worn teeth into the PCA space of good (unworn) teeth.

Default mode: loads corresponded point clouds from the SSM correspondence
pipeline output. These have consistent point ordering, so PCA captures
true anatomical shape variation.

Raw mode (--raw): loads original PLY meshes, samples points, normalizes,
and ICP-aligns. GPU-accelerated with CuPy on multiple V100s.

Usage:
    python worn_teeth_projection.py                       # corresponded + t-SNE plot
    python worn_teeth_projection.py --no-tsne             # skip t-SNE
    python worn_teeth_projection.py --raw --n-points 100000  # raw + GPU ICP
    python worn_teeth_projection.py --raw --no-gpu        # raw + CPU ICP
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import re
import sys
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

HAS_CUPY = False
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

GOOD_TEETH_DIRS = [
    os.path.join(PROJECT_DIR, "Good teeth"),
    os.path.join(PROJECT_DIR, "Good teeth-part2"),
]
WORN_TEETH_DIR = os.path.join(PROJECT_DIR, "Worn teeth")
ARTIFICIAL_WORN_DIR = os.path.join(PROJECT_DIR, "Artificially worn part 2")

CORR_BASE = os.path.join(
    PROJECT_DIR, "ssm_pipeline", "output", "correspondence_real_100k_v2"
)
CORR_GOOD_DIR = os.path.join(CORR_BASE, "good_teeth")
CORR_WORN_DIR = os.path.join(CORR_BASE, "artificial_worn")


# ── Preprocessing helpers (raw mode) ─────────────────────────────

def sample_points(mesh_path: str, n_points: int, seed: int) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def normalize(points: np.ndarray) -> np.ndarray:
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


# ── ICP implementations ──────────────────────────────────────────

def icp_align_cpu(source: np.ndarray, target: np.ndarray,
                  max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """CPU fallback using scipy cKDTree."""
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


def icp_align_gpu(source: np.ndarray, target: np.ndarray,
                  gpu_id: int = 0, max_iter: int = 100,
                  tol: float = 1e-6, batch_size: int = 10000) -> np.ndarray:
    """GPU-accelerated ICP using CuPy batched brute-force nearest neighbor.

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b to leverage fast GPU matmul.
    Default batch_size 10000 uses ~4 GB for 100k-point clouds (safe for V100 16 GB).
    """
    with cp.cuda.Device(gpu_id):
        src = cp.asarray(source, dtype=cp.float32)
        tgt = cp.asarray(target, dtype=cp.float32)
        tgt_sq = cp.sum(tgt ** 2, axis=1)
        n = src.shape[0]

        prev_err = float("inf")
        for _ in range(max_iter):
            min_idx = cp.empty(n, dtype=cp.int64)
            sum_dist = cp.float32(0.0)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = src[start:end]
                dist_sq = (cp.sum(batch ** 2, axis=1, keepdims=True)
                           + tgt_sq[None, :]
                           - 2.0 * batch @ tgt.T)
                cp.maximum(dist_sq, 0, out=dist_sq)
                idx = cp.argmin(dist_sq, axis=1)
                min_idx[start:end] = idx
                sum_dist += cp.sum(cp.sqrt(dist_sq[cp.arange(end - start), idx]))

            corr = tgt[min_idx]
            src_c = src.mean(axis=0)
            corr_c = corr.mean(axis=0)
            H = (src - src_c).T @ (corr - corr_c)
            U, _, Vt = cp.linalg.svd(H)
            R = Vt.T @ U.T
            if float(cp.linalg.det(R)) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = corr_c - R @ src_c
            src = (R @ src.T).T + t

            err = float(sum_dist) / n
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        return cp.asnumpy(src).astype(np.float64)


def parallel_icp_gpu(clouds, template, n_gpus=4,
                     max_iter=100, tol=1e-6, batch_size=10000):
    """Align all clouds to template using multiple GPUs in parallel threads."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_avail = cp.cuda.runtime.getDeviceCount()
    n_use = min(n_gpus, n_avail)

    def _align(i):
        gpu_id = (i - 1) % n_use
        return i, icp_align_gpu(clouds[i], template, gpu_id=gpu_id,
                                max_iter=max_iter, tol=tol,
                                batch_size=batch_size)

    n_to_align = len(clouds) - 1
    with ThreadPoolExecutor(max_workers=n_use) as executor:
        futures = {executor.submit(_align, i): i
                   for i in range(1, len(clouds))}
        with tqdm(total=n_to_align,
                  desc=f"ICP aligning ({n_use} GPU{'s' if n_use > 1 else ''})") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                clouds[idx] = result
                pbar.update(1)
    return clouds


# ── Label / classification helpers ────────────────────────────────

def extract_specimen_id(filename: str) -> str:
    m = re.search(r"n(\d+)", filename)
    if m:
        return f"n{m.group(1)}"
    m2 = re.search(r"(TEST\d+).*?(\d+)\.ply", filename)
    if m2:
        return f"{m2.group(1)}_L{m2.group(2)}"
    return os.path.splitext(filename)[0][:15]


def classify_worn_dir(dirname: str):
    """Classify a correspondence worn dir name like 'tooth_TEST1_wear_level3'."""
    if "TEST1" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return "TEST1", int(m.group(1)) if m else 0
    if "TEST2" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return "TEST2", int(m.group(1)) if m else 0
    return "Real worn", 0


def classify_worn_file(filename: str):
    """Classify a raw worn file."""
    if "TEST1" in filename:
        level = int(re.search(r"(\d+)\.ply", filename).group(1))
        return "TEST1", level
    if "TEST2" in filename:
        level = int(re.search(r"(\d+)\.ply", filename).group(1))
        return "TEST2", level
    return "Real worn", 0


def worn_dir_label(dirname: str) -> str:
    """Create a display label from correspondence worn dir name."""
    if "TEST1" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return f"TEST1_L{m.group(1)}" if m else "TEST1"
    if "TEST2" in dirname:
        m = re.search(r"level(\d+)", dirname)
        return f"TEST2_L{m.group(1)}" if m else "TEST2"
    m = re.search(r"tooth_(\d+)", dirname)
    return f"T{m.group(1)}_worn" if m else dirname[:15]


# ── Loading strategies ────────────────────────────────────────────

def _load_ply_points(path: str) -> np.ndarray:
    pc = trimesh.load(path, process=False)
    return np.asarray(pc.vertices if hasattr(pc, "vertices") else pc, dtype=np.float64)


def load_corresponded(corr_good_dir: str, corr_worn_dir: str):
    """Load corresponded point clouds for good and worn teeth."""
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
    worn_files, worn_labels, worn_groups = [], [], []
    for td in worn_dirs:
        ply = os.path.join(td, "corresponded.ply")
        if os.path.exists(ply):
            worn_files.append(ply)
            dname = os.path.basename(td)
            worn_labels.append(worn_dir_label(dname))
            worn_groups.append(classify_worn_dir(dname))

    # Load all
    good_clouds = []
    for f in tqdm(good_files, desc="Loading good teeth"):
        good_clouds.append(_load_ply_points(f))

    worn_clouds = []
    for f in tqdm(worn_files, desc="Loading worn teeth"):
        worn_clouds.append(_load_ply_points(f))

    return good_clouds, good_labels, worn_clouds, worn_labels, worn_groups


def load_raw(good_dirs_list, worn_dir, artificial_dir,
             n_points, seed, use_gpu, n_gpus):
    """Load raw PLY meshes, sample, normalize, ICP-align."""
    # Good
    good_files = []
    for d in good_dirs_list:
        good_files.extend(sorted(glob(os.path.join(d, "*.ply"))))
    good_files.sort(key=lambda f: os.path.basename(f))
    good_labels = [extract_specimen_id(os.path.basename(f)) for f in good_files]

    # Worn
    worn_files = sorted(glob(os.path.join(worn_dir, "*.ply")),
                        key=lambda f: os.path.basename(f))
    worn_files += sorted(glob(os.path.join(artificial_dir, "*.ply")),
                         key=lambda f: os.path.basename(f))
    worn_labels = [extract_specimen_id(os.path.basename(f)) for f in worn_files]
    worn_groups = [classify_worn_file(os.path.basename(f)) for f in worn_files]

    # Preprocess all together
    all_files = good_files + worn_files
    clouds = []
    for i, f in enumerate(tqdm(all_files, desc="Sampling & normalizing")):
        pts = sample_points(f, n_points, seed + i)
        pts = normalize(pts)
        clouds.append(pts)

    template = clouds[0]
    t0 = time.time()
    if use_gpu:
        parallel_icp_gpu(clouds, template, n_gpus=n_gpus)
    else:
        for i in tqdm(range(1, len(clouds)), desc="ICP aligning (CPU)"):
            clouds[i] = icp_align_cpu(clouds[i], template)
    print(f"ICP alignment took {time.time() - t0:.1f}s")

    n_good = len(good_files)
    return (clouds[:n_good], good_labels,
            clouds[n_good:], worn_labels, worn_groups)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Worn teeth PCA projection (uses corresponded point clouds by default)")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw PLY meshes instead of corresponded outputs")
    parser.add_argument("--n-points", type=int, default=10000,
                        help="Points to sample per tooth (raw mode only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs for parallel ICP (raw mode, default: 4)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only ICP (raw mode)")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE embedding plot")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (clamped to n_samples-1, default 30)")
    parser.add_argument("--tsne-pc-dims", type=int, default=10,
                        help="Number of leading PCs to feed t-SNE (default 10)")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    if args.raw:
        use_gpu = HAS_CUPY and not args.no_gpu
        if use_gpu:
            n_avail = cp.cuda.runtime.getDeviceCount()
            print(f"Raw mode + GPU: {n_avail} GPU(s), using {min(args.n_gpus, n_avail)}")
        else:
            reason = "CuPy not installed" if not HAS_CUPY else "--no-gpu flag"
            print(f"Raw mode + CPU ({reason})")
        (good_clouds, good_labels,
         worn_clouds, worn_labels, worn_groups) = load_raw(
            GOOD_TEETH_DIRS, WORN_TEETH_DIR, ARTIFICIAL_WORN_DIR,
            args.n_points, args.seed, use_gpu, args.n_gpus)
    else:
        print(f"Loading corresponded point clouds from:\n  {CORR_BASE}")
        if not os.path.isdir(CORR_GOOD_DIR):
            sys.exit(f"Correspondence dir not found: {CORR_GOOD_DIR}\n"
                     f"Run with --raw to use original PLY meshes.")
        (good_clouds, good_labels,
         worn_clouds, worn_labels, worn_groups) = load_corresponded(
            CORR_GOOD_DIR, CORR_WORN_DIR)

    n_good = len(good_clouds)
    n_worn = len(worn_clouds)
    n_pts = good_clouds[0].shape[0]
    print(f"Good teeth: {n_good}  ({n_pts} pts each)")
    print(f"Worn teeth: {n_worn}  "
          f"({sum(1 for g,_ in worn_groups if g=='Real worn')} real, "
          f"{sum(1 for g,_ in worn_groups if g=='TEST1')} TEST1, "
          f"{sum(1 for g,_ in worn_groups if g=='TEST2')} TEST2)")

    if n_good < 3:
        sys.exit("Need at least 3 good teeth")

    X_good = np.array([c.flatten() for c in good_clouds])
    X_worn = np.array([c.flatten() for c in worn_clouds])

    # ── PCA on good teeth, project worn ───────────────────────────
    n_comp = min(n_good - 1, 10)
    pca = PCA(n_components=n_comp)
    scores_good = pca.fit_transform(X_good)
    scores_worn = pca.transform(X_worn)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print(f"\nPCA on {n_good} good teeth ({n_comp} components):")
    for i in range(min(8, n_comp)):
        print(f"  PC{i+1}: {var_ratio[i]*100:6.2f}%  (cumulative {cum_var[i]*100:6.2f}%)")

    # Group colors and markers
    group_style = {
        "Good":      {"color": "#4C72B0", "marker": "o", "size": 120},
        "Real worn": {"color": "#C44E52", "marker": "^", "size": 110},
        "TEST1":     {"color": "#55A868", "marker": "D", "size": 100},
        "TEST2":     {"color": "#8172B2", "marker": "s", "size": 100},
    }

    # ── Plot 1: 2D scatter good + worn in PC1-PC2 ────────────────
    fig, ax = plt.subplots(figsize=(11, 8))

    gs = group_style["Good"]
    ax.scatter(scores_good[:, 0], scores_good[:, 1],
               c=gs["color"], marker=gs["marker"], s=gs["size"],
               edgecolors="k", linewidths=0.5, zorder=4, label="Good teeth")
    for i, lbl in enumerate(good_labels):
        ax.annotate(lbl, (scores_good[i, 0], scores_good[i, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7,
                    color=gs["color"])

    for gi, (grp, _) in enumerate(worn_groups):
        ws = group_style[grp]
        ax.scatter(scores_worn[gi, 0], scores_worn[gi, 1],
                   c=ws["color"], marker=ws["marker"], s=ws["size"],
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(worn_labels[gi],
                    (scores_worn[gi, 0], scores_worn[gi, 1]),
                    textcoords="offset points", xytext=(6, -8), fontsize=6,
                    color=ws["color"])

    legend_handles = [
        Line2D([0], [0], marker=v["marker"], color="w", markerfacecolor=v["color"],
               markersize=10, markeredgecolor="k", label=k)
        for k, v in group_style.items()
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="best")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    mode_tag = "corresponded" if not args.raw else "raw"
    ax.set_title(f"Worn Teeth Projected onto Good-Teeth PCA  (PC1 vs PC2, {mode_tag})",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "worn_vs_good_pca_2d.png"), dpi=200)
    plt.close(fig)
    print("Saved worn_vs_good_pca_2d.png")

    # ── t-SNE on stacked good + worn PC scores ────────────────────
    n_all = n_good + n_worn
    if not args.no_tsne and n_all >= 3:
        d_tsne = min(max(2, args.tsne_pc_dims), n_comp)
        X_stack = np.vstack([scores_good[:, :d_tsne], scores_worn[:, :d_tsne]])
        max_perp = max(2.0, float(n_all - 1) - 1e-6)
        perp = float(np.clip(args.tsne_perplexity, 2.0, max_perp))
        print(f"\nRunning t-SNE on good+worn ({n_all} specimens, input dims={d_tsne}, "
              f"perplexity={perp:.1f})...")
        _tsne_kw = dict(
            n_components=2,
            perplexity=perp,
            random_state=args.seed,
            init="pca",
        )
        try:
            tsne = TSNE(**_tsne_kw, learning_rate="auto")
        except TypeError:
            tsne = TSNE(**_tsne_kw, learning_rate=200)
        Z = tsne.fit_transform(X_stack)
        Z_good = Z[:n_good]
        Z_worn = Z[n_good:]

        fig, ax = plt.subplots(figsize=(11, 8))
        gs = group_style["Good"]
        ax.scatter(Z_good[:, 0], Z_good[:, 1],
                   c=gs["color"], marker=gs["marker"], s=gs["size"],
                   edgecolors="k", linewidths=0.5, zorder=4, label="Good teeth")
        for i, lbl in enumerate(good_labels):
            ax.annotate(lbl, (Z_good[i, 0], Z_good[i, 1]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color=gs["color"])

        for gi, (grp, _) in enumerate(worn_groups):
            ws = group_style[grp]
            ax.scatter(Z_worn[gi, 0], Z_worn[gi, 1],
                       c=ws["color"], marker=ws["marker"], s=ws["size"],
                       edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(worn_labels[gi],
                        (Z_worn[gi, 0], Z_worn[gi, 1]),
                        textcoords="offset points", xytext=(6, -8), fontsize=6,
                        color=ws["color"])

        legend_handles_tsne = [
            Line2D([0], [0], marker=v["marker"], color="w", markerfacecolor=v["color"],
                   markersize=10, markeredgecolor="k", label=k)
            for k, v in group_style.items()
        ]
        ax.legend(handles=legend_handles_tsne, fontsize=10, loc="best")
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        mode_tag = "corresponded" if not args.raw else "raw"
        ax.set_title(
            f"Good + Worn t-SNE  (first {d_tsne} PCs, perplexity={perp:.1f}, {mode_tag})",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "worn_vs_good_tsne.png"), dpi=200)
        plt.close(fig)
        print("Saved worn_vs_good_tsne.png")

    # ── t-SNE directly on full feature vectors (no PCA reduction) ─
    if not args.no_tsne and n_all >= 3:
        X_all = np.vstack([X_good, X_worn])
        max_perp = max(2.0, float(n_all - 1) - 1e-6)
        perp = float(np.clip(args.tsne_perplexity, 2.0, max_perp))
        n_feat = X_all.shape[1]
        print(f"\nRunning t-SNE on raw features ({n_feat}D, {n_all} specimens, "
              f"perplexity={perp:.1f})...")
        _tsne_kw2 = dict(
            n_components=2,
            perplexity=perp,
            random_state=args.seed,
            metric="euclidean",
        )
        try:
            tsne2 = TSNE(**_tsne_kw2, learning_rate="auto", init="pca")
        except TypeError:
            tsne2 = TSNE(**_tsne_kw2, learning_rate=200, init="random")
        Z2 = tsne2.fit_transform(X_all)
        Z2_good = Z2[:n_good]
        Z2_worn = Z2[n_good:]

        fig, ax = plt.subplots(figsize=(11, 8))
        gs = group_style["Good"]
        ax.scatter(Z2_good[:, 0], Z2_good[:, 1],
                   c=gs["color"], marker=gs["marker"], s=gs["size"],
                   edgecolors="k", linewidths=0.5, zorder=4, label="Good teeth")
        for i, lbl in enumerate(good_labels):
            ax.annotate(lbl, (Z2_good[i, 0], Z2_good[i, 1]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color=gs["color"])
        for gi, (grp, _) in enumerate(worn_groups):
            ws = group_style[grp]
            ax.scatter(Z2_worn[gi, 0], Z2_worn[gi, 1],
                       c=ws["color"], marker=ws["marker"], s=ws["size"],
                       edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(worn_labels[gi],
                        (Z2_worn[gi, 0], Z2_worn[gi, 1]),
                        textcoords="offset points", xytext=(6, -8), fontsize=6,
                        color=ws["color"])
        legend_handles_raw = [
            Line2D([0], [0], marker=v["marker"], color="w", markerfacecolor=v["color"],
                   markersize=10, markeredgecolor="k", label=k)
            for k, v in group_style.items()
        ]
        ax.legend(handles=legend_handles_raw, fontsize=10, loc="best")
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        mode_tag = "corresponded" if not args.raw else "raw"
        ax.set_title(
            f"Good + Worn t-SNE  (full {n_feat}D features, perplexity={perp:.1f}, {mode_tag})",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "worn_vs_good_tsne_raw.png"), dpi=200)
        plt.close(fig)
        print("Saved worn_vs_good_tsne_raw.png")

    # ── Plot 2: 3D scatter PC1-PC2-PC3 ───────────────────────────
    if n_comp >= 3:
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")

        gs = group_style["Good"]
        ax.scatter(scores_good[:, 0], scores_good[:, 1], scores_good[:, 2],
                   c=gs["color"], marker=gs["marker"], s=gs["size"],
                   edgecolors="k", linewidths=0.4, label="Good teeth", depthshade=True)
        for i, lbl in enumerate(good_labels):
            ax.text(scores_good[i, 0], scores_good[i, 1], scores_good[i, 2],
                    f"  {lbl}", fontsize=6, color=gs["color"])

        for gi, (grp, _) in enumerate(worn_groups):
            ws = group_style[grp]
            ax.scatter(scores_worn[gi, 0], scores_worn[gi, 1], scores_worn[gi, 2],
                       c=ws["color"], marker=ws["marker"], s=ws["size"],
                       edgecolors="k", linewidths=0.3, depthshade=True)

        ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
        ax.set_title("Good + Worn Teeth  (PC1-PC2-PC3)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "worn_vs_good_pca_3d.png"), dpi=200)
        plt.close(fig)
        print("Saved worn_vs_good_pca_3d.png")

    # ── Plot 3: Distance bar chart ────────────────────────────────
    good_centroid = scores_good.mean(axis=0)
    distances = np.linalg.norm(scores_worn - good_centroid, axis=1)

    sort_idx = np.argsort(distances)
    sorted_labels = [worn_labels[i] for i in sort_idx]
    sorted_dists = distances[sort_idx]
    sorted_groups = [worn_groups[i][0] for i in sort_idx]

    bar_colors = [group_style[g]["color"] for g in sorted_groups]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.barh(range(len(sorted_labels)), sorted_dists, color=bar_colors,
            edgecolor="k", linewidth=0.4)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.set_xlabel("Euclidean Distance to Good-Teeth Centroid (in PC space)", fontsize=11)
    ax.set_title("Worn Teeth: Distance from Good-Teeth Distribution", fontsize=13)
    ax.invert_yaxis()

    legend_handles_bar = [
        Line2D([0], [0], color=v["color"], linewidth=8, label=k)
        for k, v in group_style.items() if k != "Good"
    ]
    ax.legend(handles=legend_handles_bar, fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "worn_distance_to_good.png"), dpi=200)
    plt.close(fig)
    print("Saved worn_distance_to_good.png")

    # ── Plot 4: Wear trajectories for TEST1 and TEST2 ────────────
    fig, ax = plt.subplots(figsize=(11, 8))

    gs = group_style["Good"]
    ax.scatter(scores_good[:, 0], scores_good[:, 1],
               c=gs["color"], marker=gs["marker"], s=gs["size"],
               edgecolors="k", linewidths=0.5, alpha=0.4, zorder=2, label="Good teeth")

    for test_name, color in [("TEST1", "#55A868"), ("TEST2", "#8172B2")]:
        idx_levels = [(i, worn_groups[i][1]) for i in range(n_worn)
                      if worn_groups[i][0] == test_name]
        idx_levels.sort(key=lambda x: x[1])
        if not idx_levels:
            continue
        traj_x = [scores_worn[i, 0] for i, _ in idx_levels]
        traj_y = [scores_worn[i, 1] for i, _ in idx_levels]
        levels = [lv for _, lv in idx_levels]

        ax.plot(traj_x, traj_y, "-", color=color, linewidth=2, alpha=0.7, zorder=3)

        for j, (xi, yi, lv) in enumerate(zip(traj_x, traj_y, levels)):
            alpha = 0.4 + 0.6 * (j / max(len(levels) - 1, 1))
            ax.scatter(xi, yi, c=color, s=100, edgecolors="k", linewidths=0.5,
                       alpha=alpha, zorder=4)
            ax.annotate(f"L{lv}", (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=color, fontweight="bold")

        if len(traj_x) >= 2:
            ax.annotate("", xy=(traj_x[-1], traj_y[-1]),
                        xytext=(traj_x[-2], traj_y[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))

    legend_handles_traj = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0",
               markersize=10, markeredgecolor="k", label="Good teeth"),
        Line2D([0], [0], color="#55A868", linewidth=2, marker="o",
               markerfacecolor="#55A868", markersize=8, label="TEST1 wear path"),
        Line2D([0], [0], color="#8172B2", linewidth=2, marker="o",
               markerfacecolor="#8172B2", markersize=8, label="TEST2 wear path"),
    ]
    ax.legend(handles=legend_handles_traj, fontsize=10, loc="best")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Wear Progression Trajectories in PCA Space", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "wear_trajectory.png"), dpi=200)
    plt.close(fig)
    print("Saved wear_trajectory.png")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Distance Summary (worn -> good centroid):")
    print(f"{'='*50}")
    for i in sort_idx:
        grp = worn_groups[i][0]
        print(f"  {worn_labels[i]:>15s}  ({grp:>9s})  dist = {distances[i]:.4f}")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
