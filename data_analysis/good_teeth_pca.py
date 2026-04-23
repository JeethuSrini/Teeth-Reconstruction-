"""
PCA variance analysis and clustering of good (unworn) teeth.

Default mode: loads corresponded point clouds from the SSM correspondence
pipeline output. These have consistent point ordering across teeth, so
PCA captures true anatomical shape variation (not random sampling noise).

Raw mode (--raw): loads original PLY meshes, samples points, normalizes,
and ICP-aligns. GPU-accelerated with CuPy on multiple V100s.

Usage:
    python good_teeth_pca.py                              # corresponded + t-SNE plot
    python good_teeth_pca.py --k-means-k 3                # K-Means with k=3 (not auto)
    python good_teeth_pca.py --no-tsne                    # skip t-SNE
    python good_teeth_pca.py --raw --n-points 100000      # raw mesh + GPU ICP
    python good_teeth_pca.py --raw --no-gpu               # raw mesh + CPU ICP
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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
CORRESPONDENCE_DIR = os.path.join(
    PROJECT_DIR, "ssm_pipeline", "output", "correspondence_real_100k_v2", "good_teeth"
)


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
    batch_size controls GPU memory: each batch allocates (batch_size x N_target x 4) bytes.
    Default 10000 uses ~4 GB for 100k-point clouds (safe for V100 16 GB).
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


# ── Label helpers ─────────────────────────────────────────────────

def extract_specimen_id(filename: str) -> str:
    m = re.search(r"n(\d+)", filename)
    return f"n{m.group(1)}" if m else os.path.splitext(filename)[0][:12]


def extract_tooth_label(dirname: str) -> str:
    """Extract a label from a correspondence directory name like 'tooth_01'."""
    return dirname.replace("tooth_", "T")


# ── Loading strategies ────────────────────────────────────────────

def load_corresponded(corr_dir: str):
    """Load corresponded point clouds (already in correspondence)."""
    tooth_dirs = sorted(glob(os.path.join(corr_dir, "tooth_*")))
    files, labels, sources = [], [], []
    for td in tooth_dirs:
        ply = os.path.join(td, "corresponded.ply")
        if not os.path.exists(ply):
            print(f"  [SKIP] {os.path.basename(td)} -- no corresponded.ply")
            continue
        files.append(ply)
        labels.append(extract_tooth_label(os.path.basename(td)))

        norm_json = os.path.join(td, "normalization.json")
        if os.path.exists(norm_json):
            import json
            with open(norm_json) as f:
                meta = json.load(f)
            src_file = meta.get("source_file", "")
            sources.append("Part 2" if "part2" in src_file.lower() or "part 2" in src_file.lower() else "Part 1")
        else:
            sources.append("Part 1")

    clouds = []
    for f in tqdm(files, desc="Loading corresponded point clouds"):
        pc = trimesh.load(f, process=False)
        pts = np.asarray(pc.vertices if hasattr(pc, "vertices") else pc, dtype=np.float64)
        clouds.append(pts)

    return clouds, labels, sources


def load_raw_meshes(good_dirs, n_points, seed, use_gpu, n_gpus):
    """Load raw PLY meshes, sample, normalize, ICP-align."""
    files, sources = [], []
    for d in good_dirs:
        for f in sorted(glob(os.path.join(d, "*.ply"))):
            files.append(f)
            sources.append("Part 1" if "Good teeth-part2" not in d else "Part 2")

    labels = [extract_specimen_id(os.path.basename(f)) for f in files]

    clouds = []
    for i, f in enumerate(tqdm(files, desc="Sampling & normalizing")):
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

    return clouds, labels, sources


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCA of good teeth (uses corresponded point clouds by default)")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw PLY meshes instead of corresponded outputs")
    parser.add_argument("--n-points", type=int, default=10000,
                        help="Points to sample per tooth (raw mode only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs for parallel ICP (raw mode, default: 4)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only ICP (raw mode)")
    parser.add_argument("--correspondence-dir", type=str, default=CORRESPONDENCE_DIR,
                        help="Path to correspondence good_teeth/ output dir")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE embedding plot")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (clamped to n_teeth-1, default 30)")
    parser.add_argument("--tsne-pc-dims", type=int, default=10,
                        help="Number of leading PCs to feed t-SNE (default 10)")
    parser.add_argument("--k-means-k", type=int, default=None, metavar="K",
                        help="Fix number of K-Means clusters (e.g. 3). "
                             "If omitted, k is chosen by best silhouette score.")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    if args.raw:
        use_gpu = HAS_CUPY and not args.no_gpu
        if use_gpu:
            n_avail = cp.cuda.runtime.getDeviceCount()
            print(f"Raw mode + GPU: {n_avail} GPU(s) detected, "
                  f"using {min(args.n_gpus, n_avail)}")
        else:
            reason = "CuPy not installed" if not HAS_CUPY else "--no-gpu flag"
            print(f"Raw mode + CPU ({reason})")
        clouds, labels, sources = load_raw_meshes(
            GOOD_TEETH_DIRS, args.n_points, args.seed, use_gpu, args.n_gpus)
    else:
        print(f"Loading corresponded point clouds from:\n  {args.correspondence_dir}")
        if not os.path.isdir(args.correspondence_dir):
            sys.exit(f"Correspondence dir not found: {args.correspondence_dir}\n"
                     f"Run with --raw to use original PLY meshes instead.")
        clouds, labels, sources = load_corresponded(args.correspondence_dir)

    n_teeth = len(clouds)
    n_pts = clouds[0].shape[0]
    print(f"Found {n_teeth} good teeth  ({n_pts} points each)")
    if n_teeth < 3:
        sys.exit("Need at least 3 teeth for meaningful PCA")

    # Build feature matrix  (N_teeth x 3*n_points)
    X = np.array([c.flatten() for c in clouds])
    print(f"Feature matrix: {X.shape}")

    # PCA
    n_components = min(n_teeth, 10)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print("\nVariance explained per component:")
    for i, (v, c) in enumerate(zip(var_ratio, cum_var)):
        print(f"  PC{i+1}: {v*100:6.2f}%  (cumulative {c*100:6.2f}%)")

    # ── Plot 1: Scree plot ────────────────────────────────────────
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

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
    mode_tag = "raw" if args.raw else "corresponded"
    ax1.set_title(f"PCA Scree Plot  ({n_teeth} Good Teeth, {mode_tag})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "good_teeth_scree.png"), dpi=200)
    plt.close(fig)
    print("Saved good_teeth_scree.png")

    # ── Plot 2: PC1 vs PC2 colored by source group ───────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for src, marker, color in [("Part 1", "o", "#4C72B0"), ("Part 2", "s", "#DD8452")]:
        idx = [i for i, s in enumerate(sources) if s == src]
        if not idx:
            continue
        ax.scatter(scores[idx, 0], scores[idx, 1], c=color, marker=marker,
                   s=120, edgecolors="k", linewidths=0.5, label=src, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (scores[i, 0], scores[i, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Good Teeth in PCA Space (PC1 vs PC2)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "good_teeth_pca_2d.png"), dpi=200)
    plt.close(fig)
    print("Saved good_teeth_pca_2d.png")

    # ── Plot 3: 3D scatter PC1-PC2-PC3 ───────────────────────────
    if n_components >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for src, marker, color in [("Part 1", "o", "#4C72B0"), ("Part 2", "s", "#DD8452")]:
            idx = [i for i, s in enumerate(sources) if s == src]
            if not idx:
                continue
            ax.scatter(scores[idx, 0], scores[idx, 1], scores[idx, 2],
                       c=color, marker=marker, s=100, edgecolors="k",
                       linewidths=0.4, label=src, depthshade=True)
        for i, lbl in enumerate(labels):
            ax.text(scores[i, 0], scores[i, 1], scores[i, 2], f"  {lbl}", fontsize=7)
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
        ax.set_title("Good Teeth  (PC1-PC2-PC3)", fontsize=14)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "good_teeth_pca_3d.png"), dpi=200)
        plt.close(fig)
        print("Saved good_teeth_pca_3d.png")

    # ── t-SNE on PCA scores (2D visualization) ───────────────────
    if not args.no_tsne and n_teeth >= 3:
        d_tsne = min(max(2, args.tsne_pc_dims), n_components)
        X_tsne_in = scores[:, :d_tsne]
        max_perp = max(2.0, float(n_teeth - 1) - 1e-6)
        perp = float(np.clip(args.tsne_perplexity, 2.0, max_perp))
        print(f"\nRunning t-SNE (input dims={d_tsne}, perplexity={perp:.1f})...")
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
        Z = tsne.fit_transform(X_tsne_in)

        fig, ax = plt.subplots(figsize=(9, 7))
        for src, marker, color in [("Part 1", "o", "#4C72B0"), ("Part 2", "s", "#DD8452")]:
            idx = [i for i, s in enumerate(sources) if s == src]
            if not idx:
                continue
            ax.scatter(Z[idx, 0], Z[idx, 1], c=color, marker=marker,
                       s=120, edgecolors="k", linewidths=0.5, label=src, zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (Z[i, 0], Z[i, 1]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(
            f"Good Teeth t-SNE  (from first {d_tsne} PCs, perplexity={perp:.1f})",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "good_teeth_tsne.png"), dpi=200)
        plt.close(fig)
        print("Saved good_teeth_tsne.png")

    # ── t-SNE directly on full feature vectors (no PCA reduction) ─
    if not args.no_tsne and n_teeth >= 3:
        max_perp = max(2.0, float(n_teeth - 1) - 1e-6)
        perp = float(np.clip(args.tsne_perplexity, 2.0, max_perp))
        n_feat = X.shape[1]
        print(f"\nRunning t-SNE on raw features ({n_feat}D, perplexity={perp:.1f})...")
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
        Z2 = tsne2.fit_transform(X)

        fig, ax = plt.subplots(figsize=(9, 7))
        for src, marker, color in [("Part 1", "o", "#4C72B0"), ("Part 2", "s", "#DD8452")]:
            idx = [i for i, s in enumerate(sources) if s == src]
            if not idx:
                continue
            ax.scatter(Z2[idx, 0], Z2[idx, 1], c=color, marker=marker,
                       s=120, edgecolors="k", linewidths=0.5, label=src, zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (Z2[i, 0], Z2[i, 1]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(
            f"Good Teeth t-SNE  (full {n_feat}D features, perplexity={perp:.1f})",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "good_teeth_tsne_raw.png"), dpi=200)
        plt.close(fig)
        print("Saved good_teeth_tsne_raw.png")

    # ── Plot 4: Clustering + silhouette analysis ──────────────────
    # Default sweep: k = 2 .. min(5, n_teeth-1) (same as old range(2, min(n_teeth, 6)))
    k_hi_sweep = min(n_teeth - 1, 5)
    if args.k_means_k is not None:
        if args.k_means_k < 2 or args.k_means_k > n_teeth - 1:
            sys.exit(f"--k-means-k must be between 2 and {n_teeth - 1} (n_teeth={n_teeth})")
        k_hi_sweep = max(k_hi_sweep, args.k_means_k)
    k_range = range(2, k_hi_sweep + 1)
    sil_scores = []
    cluster_results = {}
    pca_dims = min(5, n_components)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        cluster_labels = km.fit_predict(scores[:, :pca_dims])
        s = silhouette_score(scores[:, :pca_dims], cluster_labels)
        sil_scores.append(s)
        cluster_results[k] = cluster_labels
        print(f"  k={k}: silhouette={s:.3f}")

    if args.k_means_k is not None:
        best_k = args.k_means_k
        best_labels = cluster_results[best_k]
        print(f"Using k={best_k} (--k-means-k); silhouette={sil_scores[list(k_range).index(best_k)]:.3f}")
    else:
        best_k = list(k_range)[int(np.argmax(sil_scores))]
        best_labels = cluster_results[best_k]
        print(f"Best k={best_k} (silhouette={max(sil_scores):.3f})")

    fig, (ax_sil, ax_clust) = plt.subplots(1, 2, figsize=(14, 6))

    ax_sil.plot(list(k_range), sil_scores, "o-", linewidth=2, color="#4C72B0")
    vline_label = f"k={best_k} (fixed)" if args.k_means_k is not None else f"Best k={best_k}"
    ax_sil.axvline(best_k, linestyle="--", color="#C44E52", alpha=0.7, label=vline_label)
    ax_sil.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax_sil.set_ylabel("Silhouette Score", fontsize=12)
    ax_sil.set_title("Silhouette Analysis", fontsize=13)
    ax_sil.legend(fontsize=11)
    ax_sil.grid(True, alpha=0.3)

    cmap = plt.cm.Set2
    for ci in range(best_k):
        idx = np.where(best_labels == ci)[0]
        ax_clust.scatter(scores[idx, 0], scores[idx, 1], c=[cmap(ci)] * len(idx),
                         s=120, edgecolors="k", linewidths=0.5,
                         label=f"Cluster {ci+1}", zorder=3)
    for i, lbl in enumerate(labels):
        ax_clust.annotate(lbl, (scores[i, 0], scores[i, 1]),
                          textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax_clust.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax_clust.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax_clust.set_title(f"K-Means Clustering (k={best_k})", fontsize=13)
    ax_clust.legend(fontsize=10)
    ax_clust.grid(True, alpha=0.3)

    fig.suptitle(f"Good Teeth Clustering  ({n_teeth} specimens)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "good_teeth_clusters.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved good_teeth_clusters.png")

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
