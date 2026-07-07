#!/usr/bin/env python3
"""
Wear-Margin Patch Reconstruction Pipeline

A reconstruction mode that consumes existing Stage 2 outputs and produces a
"patched" tooth instead of fitting a fresh SSM:

1. Detect the worn region on the input by per-point DEVIATION between the
   index-aligned worn input (`worn_input.ply`) and the clean SSM
   reconstruction (`reconstructed.ply`). Both clouds share the same 100k
   point ordering, so detection is a per-index operation.
2. Trace a closed "wear ring" landmark/annotation around the detected region.
3. Cut the clean reconstruction along that ring and graft its cap onto the
   worn input:
     - mode "swap"   : index-space graft (inside ring = recon, outside = input)
                       with transition-band seam blending, then Poisson re-mesh.
     - mode "surgery": mesh-level cut of the recon cap + stitch onto the
                       input mesh (optional, built on top of swap).

The script reuses the inverse-transform, meshing and evaluation helpers from
``reconstruction_pipeline.py``.

Usage:
    python patch_reconstruction.py \
        --recon-dir output/recon_neighborhood_v4 \
        --correspondence-dir output/correspondence_all_100k \
        --artificial-wear ../all_worn_input \
        --output output/recon_patch

    python patch_reconstruction.py \
        --recon-dir output/recon_all_v3 \
        --correspondence-dir output/correspondence_all_100k \
        --output output/recon_patch \
        --worn-teeth tooth_01_wear_real tooth_08_wear_real \
        --deviation-percentile 80 \
        --mode surgery
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

# Make sibling modules importable regardless of the working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from reconstruction_pipeline import (  # noqa: E402
    compute_geometric_comparison,
    inverse_correspondence_transform,
    load_point_cloud,
    point_cloud_to_mesh,
    save_mesh,
    save_point_cloud,
    _refine_icp_to_worn,
)


# ===========================================================================
# SUBPROCESS-ISOLATED MESHING
# ===========================================================================
# Open3D's Screened Poisson reconstruction can intermittently abort the whole
# process ("Failed to close loop") without raising a catchable Python error.
# We isolate every meshing call in a spawned subprocess so such an abort is
# contained: the parent detects failure via the child exit code + whether the
# output file was actually written, then continues.

def _mesh_worker(points: np.ndarray, out_path: str, poisson_depth: int,
                 density_quantile: float = 0.01) -> None:
    """Child-process entry point: mesh a cloud and save it. Must be top-level
    (picklable) for the 'spawn' start method.

    The child's stdout/stderr are redirected to /dev/null at the C file-descriptor
    level so Open3D's noisy C++ Poisson logging ("Failed to close loop") and the
    "CuPy not available" import banner don't flood the parent log. Success is
    signalled purely by the written file + clean exit code. A degenerate mesh
    (no faces) is treated as failure (file is not written) so the parent retries.

    ``density_quantile`` trims low-density (extrapolated) Poisson vertices; pass
    0.0 to keep every vertex -- needed when meshing a cloud with genuine holes,
    because the watertight bridge Poisson lays over a hole is itself low-density
    and would otherwise be trimmed away (reopening the hole).
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)   # stdout
        os.dup2(devnull, 2)   # stderr
    except Exception:
        pass
    from reconstruction_pipeline import point_cloud_to_mesh as _ptm, save_mesh as _sm
    mesh = _ptm(points, poisson_depth=poisson_depth,
                density_quantile=density_quantile)
    n_faces = len(getattr(mesh, "faces", []))
    if n_faces == 0:
        return  # leave out_path absent -> parent sees failure and retries
    _sm(mesh, out_path)


def safe_mesh_to_file(points: np.ndarray,
                      out_path: str,
                      poisson_depths: Tuple[int, ...] = (9, 8, 7, 6),
                      attempts_per_depth: int = 6,
                      timeout: float = 300.0,
                      density_quantile: float = 0.01) -> int:
    """Mesh ``points`` to ``out_path`` in an isolated process.

    Open3D's Screened Poisson aborts non-deterministically, so we (a) retry the
    same depth a few times (a fresh run usually succeeds) and (b) fall back to
    progressively lower octree depths, which are markedly more stable.

    Returns the octree depth that actually produced the mesh (a truthy int), or
    ``0`` on total failure. Reporting the depth matters because a silent
    fallback to depth 6-7 coarsens the WHOLE surface (faceted, rigid look), not
    just the patch -- the caller should log it. The depth-9 abort is roughly a
    coin-flip per run on these clouds, so ``attempts_per_depth`` defaults to 6:
    that pushes the chance of holding the full depth-9 detail above ~99% before
    we ever drop to a coarser octree. (Voxel-uniform downsampling was tried as
    an alternative stabiliser and made the abort *more* frequent -- avoid it.)
    """
    pts = np.asarray(points)
    ctx = mp.get_context("spawn")
    for depth in poisson_depths:
        for _ in range(max(1, attempts_per_depth)):
            if os.path.exists(out_path):
                os.remove(out_path)
            proc = ctx.Process(target=_mesh_worker,
                               args=(pts, out_path, depth, density_quantile))
            proc.start()
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                continue
            if proc.exitcode == 0 and os.path.exists(out_path):
                return depth
    return 0


# ===========================================================================
# 1. WEAR DETECTION (deviation between index-aligned input and reconstruction)
# ===========================================================================

def detect_wear_mask(worn_points: np.ndarray,
                     recon_points: np.ndarray,
                     deviation_percentile: float = 75.0,
                     deviation_threshold: Optional[float] = None
                     ) -> Tuple[np.ndarray, Dict]:
    """Detect the worn region from per-point deviation.

    ``worn_points`` and ``recon_points`` are index-aligned (N, 3) clouds in the
    normalized correspondence frame. Wear is where the clean reconstruction
    sits *above/outside* the worn surface, i.e. the signed outward component of
    the displacement ``recon - worn`` is large and positive.

    Args:
        worn_points: Worn input cloud (N, 3).
        recon_points: Clean SSM reconstruction (N, 3), same ordering.
        deviation_percentile: Percentile of the outward-deviation score used as
            the threshold when ``deviation_threshold`` is None.
        deviation_threshold: Absolute outward-deviation floor (normalized units).
            When provided, points must clear BOTH the percentile and this floor.

    Returns:
        (wear_mask (N,) bool, info dict).
    """
    disp = recon_points - worn_points                      # (N, 3)
    disp_mag = np.linalg.norm(disp, axis=1)                 # (N,)

    # Outward direction at each worn point (away from the tooth centroid).
    centroid = worn_points.mean(axis=0)
    outward = worn_points - centroid
    outward_norm = np.linalg.norm(outward, axis=1, keepdims=True)
    outward_unit = outward / np.maximum(outward_norm, 1e-12)

    # Signed outward component of the displacement. Positive => recon is
    # outside the worn surface (true wear that needs filling).
    signed_outward = np.einsum("ij,ij->i", disp, outward_unit)

    # Score combines outward push with raw magnitude so oblique wear still
    # registers, while clamping the outward term at 0 to ignore inward noise.
    score = np.maximum(signed_outward, 0.0) + 0.25 * disp_mag

    # Rank-based selection of the top fraction. This is robust to the fact
    # that the reconstruction is refined to match the worn input exactly at
    # observed points: most points then have ~0 deviation, which would make a
    # value-based percentile threshold collapse to "select everything". By
    # ranking and then dropping anything below a small noise floor we keep only
    # the genuinely-deviating cap.
    n = len(score)
    target_frac = max(0.0, min(1.0, (100.0 - deviation_percentile) / 100.0))
    k = int(round(target_frac * n))
    order = np.argsort(score)[::-1]
    candidates = order[:k]

    noise_floor = max(1e-9, 1e-3 * float(score.max()))
    candidates = candidates[score[candidates] > noise_floor]

    wear_mask = np.zeros(n, dtype=bool)
    wear_mask[candidates] = True
    if deviation_threshold is not None:
        wear_mask = wear_mask & (signed_outward >= float(deviation_threshold))

    info = {
        "deviation_percentile": deviation_percentile,
        "deviation_threshold": deviation_threshold,
        "target_fraction": target_frac,
        "noise_floor": noise_floor,
        "disp_mag_mean": float(disp_mag.mean()),
        "disp_mag_max": float(disp_mag.max()),
        "signed_outward_mean": float(signed_outward.mean()),
        "signed_outward_max": float(signed_outward.max()),
        "n_nonzero_score": int((score > noise_floor).sum()),
        "n_raw_wear": int(wear_mask.sum()),
        "raw_wear_fraction": float(wear_mask.mean()),
    }
    return wear_mask, info


# ===========================================================================
# 2. MASK CLEANUP (largest connected component + hole filling)
# ===========================================================================

def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) neighbor indices EXCLUDING the point itself."""
    k_eff = min(k + 1, len(points))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(points)
    _, knn = nn.kneighbors(points)
    return knn[:, 1:]  # drop self column


def clean_wear_mask(points: np.ndarray,
                    mask: np.ndarray,
                    k: int = 8,
                    min_component_size: int = 200,
                    fill_k: int = 12,
                    fill_frac: float = 0.7,
                    fill_iters: int = 3) -> Tuple[np.ndarray, Dict]:
    """Keep the largest contiguous wear region and fill interior holes.

    Args:
        points: Full point cloud (N, 3).
        mask: Raw boolean wear mask (N,).
        k: KNN used to build the connectivity graph over masked points.
        min_component_size: Drop wear components smaller than this.
        fill_k: KNN used for hole filling.
        fill_frac: An unworn point is absorbed if at least this fraction of its
            neighbors are worn.
        fill_iters: Max hole-filling sweeps.

    Returns:
        (cleaned_mask (N,) bool, info dict).
    """
    info: Dict = {"n_input": int(mask.sum())}
    idx = np.where(mask)[0]
    if len(idx) == 0:
        info["status"] = "empty"
        return mask, info

    # Connected components over the masked sub-cloud.
    sub = points[idx]
    knn = _knn_indices(sub, k)
    rows = np.repeat(np.arange(len(idx)), knn.shape[1])
    cols = knn.ravel()
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(len(idx), len(idx)))
    n_comp, labels = connected_components(graph, directed=False)

    comp_sizes = np.bincount(labels)
    largest = int(np.argmax(comp_sizes))
    keep_components = {largest}
    # Keep any other reasonably large components too (multi-cusp wear).
    for c, size in enumerate(comp_sizes):
        if c != largest and size >= min_component_size:
            keep_components.add(c)

    keep_local = np.isin(labels, list(keep_components))
    cleaned = np.zeros(len(mask), dtype=bool)
    cleaned[idx[keep_local]] = True

    info["n_components"] = int(n_comp)
    info["largest_component_size"] = int(comp_sizes[largest])
    info["kept_components"] = sorted(int(c) for c in keep_components)
    info["n_after_components"] = int(cleaned.sum())

    # Hole filling: absorb unworn points surrounded by worn neighbors.
    knn_full = _knn_indices(points, fill_k)
    filled = cleaned.copy()
    sweeps = 0
    for _ in range(fill_iters):
        neigh_worn_frac = filled[knn_full].mean(axis=1)
        newly = (~filled) & (neigh_worn_frac >= fill_frac)
        if not newly.any():
            break
        filled = filled | newly
        sweeps += 1

    info["fill_sweeps"] = sweeps
    info["n_after_fill"] = int(filled.sum())
    info["status"] = "ok"
    return filled, info


# ===========================================================================
# 3. RING LANDMARK / ANNOTATION (ordered wear-margin loop)
# ===========================================================================

def extract_wear_ring(points: np.ndarray,
                      mask: np.ndarray,
                      k: int = 12) -> Tuple[np.ndarray, Dict]:
    """Extract the wear margin as an ordered closed loop of point indices.

    Boundary points are worn points that have at least one unworn KNN
    neighbor. They are ordered into a loop by projecting onto the local
    tangent plane (top-2 PCA axes of the boundary) and sorting by angle.

    Args:
        points: Full cloud (N, 3).
        mask: Cleaned boolean wear mask (N,).
        k: KNN neighborhood size.

    Returns:
        (ordered_indices (R,) into ``points``, info dict).
    """
    info: Dict = {}
    if mask.sum() == 0:
        info["status"] = "empty"
        return np.array([], dtype=np.int64), info

    knn = _knn_indices(points, k)
    has_unworn_neighbor = (~mask[knn]).any(axis=1)
    boundary = np.where(mask & has_unworn_neighbor)[0]

    if len(boundary) < 3:
        info["status"] = "too_small"
        info["n_boundary"] = int(len(boundary))
        return boundary, info

    bpts = points[boundary]
    center = bpts.mean(axis=0)
    centered = bpts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    u = centered @ Vt[0]
    v = centered @ Vt[1]
    angles = np.arctan2(v, u)
    order = np.argsort(angles)
    ordered = boundary[order]

    info["status"] = "ok"
    info["n_boundary"] = int(len(boundary))
    info["ring_center"] = center.tolist()
    info["ring_plane_axes"] = Vt[:2].tolist()
    return ordered, info


# ===========================================================================
# 3b. GEOMETRIC HOLE DETECTION (rim tracing on the real worn surface)
# ===========================================================================
# Worn EDJ surfaces have genuine open holes (missing/damaged material). We find
# their rims geometrically: a point on a true boundary has all its neighbors
# bunched on one side, leaving a large empty angular wedge (~180 deg) in the
# local tangent plane. Folds/ridges do NOT trigger this (neighbors still wrap
# all the way around), which is why this beats a curvature/centroid heuristic.

def estimate_point_normals(points: np.ndarray, knn: int = 30) -> np.ndarray:
    """Estimate per-point unit normals via Open3D (PCA over knn neighbors)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals)


def angular_gap_rim(points: np.ndarray,
                    normals: np.ndarray,
                    k: int = 22,
                    gap_thresh: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Flag boundary/rim points by the largest angular gap among their
    tangent-plane-projected neighbors.

    Args:
        points: (N, 3) cloud.
        normals: (N, 3) unit normals.
        k: neighbors per point.
        gap_thresh: a point is a rim point if its max angular gap (radians)
            exceeds this (default 2.0 rad ~= 115 deg).

    Returns:
        (rim_mask (N,) bool, max_gap (N,) float).
    """
    knn = _knn_indices(points, k)                       # (N, k)
    v = points[knn] - points[:, None, :]                # (N, k, 3)
    nrm = normals[:, None, :]
    v = v - (v * nrm).sum(-1, keepdims=True) * nrm      # project to tangent plane

    # Build a consistent tangent basis (t1, t2) per point from the normal.
    a = np.tile(np.array([1.0, 0.0, 0.0]), (len(points), 1))
    flip = np.abs((normals * a).sum(1)) > 0.9
    a[flip] = [0.0, 1.0, 0.0]
    t1 = np.cross(normals, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(normals, t1)

    x = (v * t1[:, None, :]).sum(-1)
    y = (v * t2[:, None, :]).sum(-1)
    ang = np.sort(np.arctan2(y, x), axis=1)             # (N, k)
    gaps = np.diff(ang, axis=1)
    wrap = ang[:, 0] + 2 * np.pi - ang[:, -1]           # wrap-around gap
    max_gap = np.maximum(gaps.max(axis=1), wrap)
    return max_gap > gap_thresh, max_gap


def _order_loop(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Order a set of rim point indices into a closed loop by angle in their
    local tangent plane."""
    if len(idx) < 3:
        return idx
    p = points[idx]
    center = p.mean(axis=0)
    centered = p - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    u = centered @ Vt[0]
    v = centered @ Vt[1]
    order = np.argsort(np.arctan2(v, u))
    return idx[order]


def _loop_perimeter(points: np.ndarray, ordered_idx: np.ndarray) -> float:
    """Closed-loop perimeter length."""
    if len(ordered_idx) < 2:
        return 0.0
    p = points[ordered_idx]
    seg = np.linalg.norm(np.diff(np.vstack([p, p[0]]), axis=0), axis=1)
    return float(seg.sum())


def detect_holes(points: np.ndarray,
                 cell_mult: float = 2.5,
                 min_hole_cells: int = 6,
                 close_iter: int = 1,
                 rim_dilate: int = 1,
                 outer_gap_thresh: float = 2.7,
                 outer_gap_k: int = 22,
                 outer_denoise_radius: float = 4.0) -> Dict:
    """Detect *enclosed* open holes on a worn surface and trace their rims.

    Holes (interior) are found as **enclosed empty regions** of the occlusal
    projection, which is robust to noise and density variation:

      1. Project the cloud onto its occlusal plane (top-2 PCA axes).
      2. Rasterize to an occupancy grid (cell = ``cell_mult`` x median spacing),
         with a light morphological close to bridge sampling micro-gaps.
      3. Flood-fill empty cells from the grid border: cells reachable from the
         border are the *exterior*; enclosed empty cells are *holes*.
      4. Each enclosed empty component (>= ``min_hole_cells``) is a hole; its rim
         = occupied points in cells bordering that component.

    The **outer perimeter** is found differently: as the true open-sheet edge via
    angular-gap boundary detection at a high threshold (``outer_gap_thresh``).
    The occupancy-silhouette boundary would instead capture whole vertical wall
    columns (the rim wall is near-vertical), which looks scattered in side
    views; the sheet-edge approach yields a thin, clean 3D loop.

    Returns a dict with:
      outer_rim (ordered idx), interior_loops (list of ordered idx),
      hole_mask (union of interior-hole rim points), and an info sub-dict that
      includes the projection basis + grid so a later patch step can fill holes.
    """
    from scipy import ndimage

    n = len(points)
    centroid = points.mean(axis=0)
    cen = points - centroid
    _, _, Vt = np.linalg.svd(cen, full_matrices=False)
    axes = Vt[:2]                                  # occlusal in-plane axes
    p2 = cen @ axes.T                              # (n, 2) occlusal coords

    # Median nearest-neighbour spacing -> grid cell size.
    nn = NearestNeighbors(n_neighbors=2).fit(points)
    dists, _ = nn.kneighbors(points)
    msp = float(np.median(dists[:, 1]))
    cell = cell_mult * msp

    mins = p2.min(axis=0)
    ij = np.floor((p2 - mins) / cell).astype(int)
    gi = ij[:, 0] + 1                              # +1 pad border
    gj = ij[:, 1] + 1
    H = int(ij[:, 0].max()) + 3
    W = int(ij[:, 1].max()) + 3

    occ = np.zeros((H, W), dtype=bool)
    occ[gi, gj] = True
    if close_iter > 0:
        occ = ndimage.binary_closing(occ, iterations=close_iter)
    empty = ~occ

    lab, n_lab = ndimage.label(empty)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    interior_labels = [l for l in range(1, n_lab + 1) if l not in border]
    hole_region = np.isin(lab, interior_labels) if interior_labels \
        else np.zeros_like(empty)

    cell_area = cell * cell
    result: Dict = {
        "outer_rim": np.array([], dtype=np.int64),
        "interior_loops": [],
        "hole_mask": np.zeros(n, dtype=bool),
        "info": {
            "status": "ok",
            "median_spacing": msp,
            "cell_size": cell,
            "grid_shape": [H, W],
            "projection_axes": axes.tolist(),
            "projection_centroid": centroid.tolist(),
            "grid_origin": mins.tolist(),
        },
    }

    # Interior holes.
    hlab, hn = ndimage.label(hole_region)
    hole_mask = np.zeros(n, dtype=bool)
    kept_grid = np.zeros((H, W), dtype=bool)   # union of kept hole cells (for patching)
    interior_loops: List[np.ndarray] = []
    loops_info: List[Dict] = []
    for hid in range(1, hn + 1):
        cells = (hlab == hid)
        n_cells = int(cells.sum())
        if n_cells < min_hole_cells:
            continue
        ring_cells = ndimage.binary_dilation(cells, iterations=rim_dilate) & occ
        on_rim = ring_cells[gi, gj]
        idx = np.where(on_rim)[0]
        if len(idx) < 3:
            continue
        ordered = _order_loop(points, idx)
        hole_mask[idx] = True
        kept_grid |= cells
        interior_loops.append(ordered)
        loops_info.append({
            "n_rim_points": int(len(ordered)),
            "n_cells": n_cells,
            "area_normalized": float(n_cells * cell_area),
            "perimeter": _loop_perimeter(points, ordered),
            "centroid": points[idx].mean(axis=0).tolist(),
        })

    result["interior_loops"] = interior_loops
    result["hole_mask"] = hole_mask
    result["hole_grid"] = kept_grid

    # Outer perimeter = true open-sheet edge (thin 3D loop), via angular-gap
    # boundary at a high threshold. Exclude hole-rim points so holes stay red.
    normals = estimate_point_normals(points, knn=30)
    rim_mask, _ = angular_gap_rim(points, normals, k=outer_gap_k,
                                  gap_thresh=outer_gap_thresh)
    rim_mask &= ~hole_mask
    outer_idx = np.where(rim_mask)[0]
    # Light denoise: drop isolated specks (need >=2 rim neighbours nearby).
    if len(outer_idx) > 10:
        sub = points[outer_idx]
        dd, _ = NearestNeighbors(n_neighbors=2).fit(sub).kneighbors(sub)
        mspr = float(np.median(dd[:, 1]))
        neigh = NearestNeighbors(radius=outer_denoise_radius * mspr).fit(sub)
        counts = neigh.radius_neighbors(sub, return_distance=False)
        keep = np.array([len(c) - 1 >= 2 for c in counts])
        outer_idx = outer_idx[keep]
    if len(outer_idx) >= 3:
        result["outer_rim"] = _order_loop(points, outer_idx)

    result["info"].update({
        "n_rim_points": int(hole_mask.sum() + len(outer_idx)),
        "outer_rim_size": int(len(outer_idx)),
        "n_interior_holes": len(interior_loops),
        "interior_holes": loops_info,
    })
    return result


# ===========================================================================
# 3d. EXTRAS CLEANING (recon-prior: outward shells, veils, spurs)
# ===========================================================================
# Uses reconstructed.ply as a geometric prior for a healthy tooth. Worn points
# are compared to the recon SURFACE (local tangent planes), not single recon
# points. Extras are outward duplicate layers / thin sheets; cusp bumps that
# merely sit outside a coarse SSM fit are rejected unless they look sheet-like.

def _recon_signed_distance(points: np.ndarray,
                           recon: np.ndarray,
                           plane_knn: int = 25) -> np.ndarray:
    """Signed distance from each worn point to the recon surface via local
    tangent planes fitted to the nearest recon neighbours (point-to-surface)."""
    k = min(plane_knn, len(recon))
    _, idx = NearestNeighbors(n_neighbors=k).fit(recon).kneighbors(points)
    neighbors = recon[idx]                                       # (N, k, 3)
    centers = neighbors.mean(axis=1)
    X = neighbors - centers[:, None, :]
    cov = np.einsum("nki,nkj->nij", X, X) / max(k, 1)
    _, evecs = np.linalg.eigh(cov)                               # ascending
    normals = evecs[:, 0]                                        # smallest var
    to_pt = points - centers
    flip = (np.sum(normals * to_pt, axis=1) < 0)
    normals[flip] *= -1
    return np.sum(to_pt * normals, axis=1)


def _cluster_sheet_aspect(pts: np.ndarray) -> float:
    """Aspect ratio of the cluster PCA (large = thin sheet / veil)."""
    if len(pts) < 4:
        return 1.0
    X = pts - pts.mean(axis=0)
    evals = np.linalg.eigh(X.T @ X / len(pts))[0]
    evals = np.sort(evals)
    return float(evals[2] / (evals[0] + 1e-12))


def _occlusal_radius_percentile(cluster_idx: np.ndarray,
                                all_points: np.ndarray) -> float:
    """Percentile rank of a cluster's mean occlusal-plane radius (high = margin)."""
    ctr = all_points.mean(axis=0)
    _, _, Vt = np.linalg.svd(all_points - ctr, full_matrices=False)
    axes = Vt[:2]
    r_all = np.linalg.norm((all_points - ctr) @ axes.T, axis=1)
    r_cl = float(np.linalg.norm((all_points[cluster_idx].mean(axis=0) - ctr) @ axes.T))
    return float(100.0 * (r_all < r_cl).mean())


def _cluster_extent_tail_percentile(cluster_idx: np.ndarray,
                                    all_points: np.ndarray) -> float:
    """How far toward an axis extreme the cluster centroid sits (0–100)."""
    ctr = all_points[cluster_idx].mean(axis=0)
    tails = []
    for ax in range(3):
        p = 100.0 * (all_points[:, ax] < ctr[ax]).mean()
        tails.append(max(p, 100.0 - p))
    return float(max(tails))


def _has_inner_surface(idx: int,
                       points: np.ndarray,
                       signed: np.ndarray,
                       worn_tree,
                       protrude_thresh: float,
                       signed_median: float,
                       shell_radius_mult: float,
                       min_inner_neighbors: int,
                       msp: float) -> bool:
    """True when this outward point sits on a duplicate outer shell: several
    nearby worn neighbours lie on (or inside) the recon surface below it."""
    p = points[idx]
    nbrs = worn_tree.query_ball_point(p, r=shell_radius_mult * msp)
    nbrs = [j for j in nbrs if j != idx]
    if not nbrs:
        return False
    nbrs = np.asarray(nbrs, dtype=int)
    # Inner layer = close to the recon prior (not merely lower on the same bump).
    inner_cut = min(protrude_thresh * 0.35, signed_median + 2.0 * msp)
    n_inner = int((signed[nbrs] < inner_cut).sum())
    return n_inner >= min_inner_neighbors


def clean_extras(points: np.ndarray,
                 recon: Optional[np.ndarray] = None,
                 extra_dist_mult: float = 6.0,
                 envelope_mad_mult: float = 3.0,
                 connect_radius_mult: float = 3.0,
                 min_extra_cluster: int = 150,
                 sheet_aspect_min: float = 5.0,
                 shell_radius_mult: float = 4.0,
                 min_inner_neighbors: int = 4,
                 shell_cluster_frac: float = 0.12,
                 periphery_pct: float = 58.0,
                 extent_tail_pct: float = 82.0,
                 grow_iters: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Remove continuous micro-CT extras using reconstructed.ply as a prior.

    Pipeline:
      1. Point-to-surface signed distance (local planes on recon).
      2. Flag outward points beyond a robust envelope threshold.
      3. Keep only thin-shell / duplicate-layer points (inner worn surface
         exists below) OR points in sheet-like clusters (veils / flaps).
      4. Remove only large connected clusters (no scattered point-wise drops).

    Returns (cleaned_points, keep_mask, info); keep_mask is over ORIGINAL
    points (True = kept).
    """
    from scipy.spatial import cKDTree

    n = len(points)
    nn = NearestNeighbors(n_neighbors=2).fit(points)
    dd, _ = nn.kneighbors(points)
    msp = float(np.median(dd[:, 1]))
    info: Dict = {"n_input": n, "median_spacing": msp, "stages": {}}

    if recon is None or extra_dist_mult <= 0:
        keep_mask = np.ones(n, dtype=bool)
        info["stages"]["skipped"] = "no_recon"
        info["n_kept"] = n
        info["n_removed"] = 0
        return points.copy(), keep_mask, info

    signed = _recon_signed_distance(points, recon)
    med = float(np.median(signed))
    mad = float(np.median(np.abs(signed - med))) + 1e-12
    robust = med + envelope_mad_mult * 1.4826 * mad
    thresh = max(extra_dist_mult * msp, robust)
    protrude = signed > thresh
    info["envelope_thresh"] = thresh
    info["envelope_signed_median"] = med
    info["stages"]["protrude_candidates"] = int(protrude.sum())

    worn_tree = cKDTree(points)
    thin_shell = np.zeros(n, dtype=bool)
    protrude_idx = np.where(protrude)[0]
    for i in protrude_idx:
        thin_shell[i] = _has_inner_surface(
            i, points, signed, worn_tree, thresh, med,
            shell_radius_mult, min_inner_neighbors, msp)
    info["stages"]["thin_shell_candidates"] = int(thin_shell.sum())

    # Per-cluster: sheet-like blobs (veils) OR mostly duplicate-shell points.
    candidate = np.zeros(n, dtype=bool)
    cand_idx = protrude_idx
    remove = np.zeros(n, dtype=bool)
    cluster_sizes: List[int] = []
    cluster_aspects: List[float] = []

    if len(cand_idx) > 0:
        sub = points[cand_idx]
        cg = NearestNeighbors(radius=connect_radius_mult * msp).fit(sub) \
            .radius_neighbors_graph(sub)
        cnc, clab = connected_components(cg, directed=False)
        for c in range(cnc):
            sel = clab == c
            size = int(sel.sum())
            if size < min_extra_cluster:
                continue
            glob_idx = cand_idx[sel]
            aspect = _cluster_sheet_aspect(points[glob_idx])
            is_sheet = aspect >= sheet_aspect_min
            shell_frac = float(thin_shell[glob_idx].mean())
            has_shell = shell_frac >= shell_cluster_frac
            periph = _occlusal_radius_percentile(glob_idx, points)
            extent_tail = _cluster_extent_tail_percentile(glob_idx, points)
            at_margin = (periph >= periphery_pct
                         or extent_tail >= extent_tail_pct)
            # Duplicate-shell layers: remove anywhere on the tooth.
            # Sheet-like veils/flaps: remove only at the margin (occlusal rim
            # or root/cervical extremes) so interior cusp bumps are preserved.
            if has_shell or (is_sheet and at_margin):
                remove[glob_idx] = True
                cluster_sizes.append(size)
                cluster_aspects.append(aspect)
                candidate[glob_idx] = True

    if grow_iters > 0 and remove.any():
        tree = cKDTree(points)
        protrude_set = protrude.copy()
        for _ in range(grow_iters):
            seed = points[remove]
            nbrs = tree.query_ball_point(seed, r=connect_radius_mult * msp)
            add = np.unique(np.concatenate(nbrs)) if len(nbrs) else np.array([], int)
            grow = np.zeros(n, dtype=bool)
            grow[add.astype(int)] = True
            remove |= grow & protrude_set

    keep_mask = ~remove
    info["stages"]["removed"] = int(remove.sum())
    info["n_extra_clusters"] = len(cluster_sizes)
    info["extra_cluster_sizes"] = sorted(cluster_sizes, reverse=True)
    info["cluster_aspects"] = sorted(cluster_aspects, reverse=True)
    info["n_kept"] = int(keep_mask.sum())
    info["n_removed"] = int(remove.sum())
    return points[keep_mask], keep_mask, info


# ===========================================================================
# 3c. HOLE PATCHING (fuse the reconstruction cap into the detected holes)
# ===========================================================================
# The reconstruction is in the same frame as the worn surface, so we keep every
# real worn point and *add* reconstruction points wherever they project into a
# detected hole. Each graft is locally ICP-aligned to the worn rim collar and
# feathered into the seam so the patch meets the surrounding surface smoothly.

def _occlusal_basis(info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axes (2,3), normal (3,), centroid (3,)) of the occlusal frame the
    hole grid was built in. (u, v) span the occlusal plane, normal is height."""
    axes = np.array(info["projection_axes"])
    centroid = np.array(info["projection_centroid"])
    normal = np.cross(axes[0], axes[1])
    normal /= np.linalg.norm(normal) + 1e-12
    return axes, normal, centroid


def _to_uvh(pts: np.ndarray, axes: np.ndarray, normal: np.ndarray,
            centroid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World points -> (uv (N,2) occlusal coords, h (N,) height along normal)."""
    d = pts - centroid
    uv = d @ axes.T
    h = d @ normal
    return uv, h


def _uvh_to_world(uv: np.ndarray, h: np.ndarray, axes: np.ndarray,
                  normal: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Inverse of ``_to_uvh``."""
    return centroid + uv @ axes + h[:, None] * normal


def patch_holes(real_worn: np.ndarray,
                recon: np.ndarray,
                hole_result: Dict,
                fill_dilate: int = 2,
                align: bool = True,
                collar_iter: int = 5,
                patch_cell_mult: float = 0.9,
                support_radius_mult: float = 3.5,
                min_support: int = 4,
                tps_smooth_mult: float = 0.4,
                jitter_frac: float = 0.4,
                tps_neighbors: int = 48) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Fill detected holes with a dense, seamless, smooth graft.

    Strategy (per hole): treat the surface around the hole as a height field
    fitted with a thin-plate spline in a LOCAL frame, then resample it:

      1. Locally rigid-ICP the reconstruction (hole + a rim collar) onto the
         worn rim collar so its interior shape sits at the right place.
      2. Build a support set from BOTH the worn rim points (which anchor the
         seam so the patch meets the surrounding surface) and the aligned
         reconstruction interior (which supplies the missing shape).
      3. Fit the height field z = f(a, b) in a frame whose (a, b) plane is the
         PCA plane of the supports and whose height axis is their thin
         (smallest-variance) direction. Doing this per hole keeps the field
         single-valued even on a tilted cusp wall, where a fixed global-occlusal
         height field would collapse a near-vertical surface to a ramp.
      4. Interpolate with a thin-plate-spline RBF (curvature-minimizing, so the
         patch is genuinely smooth and joins the collar with continuous slope),
         then sample it on a jittered grid at the LOCAL worn-collar spacing so
         the patch density matches the surrounding real surface (no density step
         for Poisson to facet on at the seam).

    Args:
        real_worn: (N, 3) real worn surface (icp_aligned), with holes.
        recon: (M, 3) complete reconstruction, SAME frame as real_worn.
        hole_result: output of ``detect_holes`` (provides the occlusal grid).
        fill_dilate: dilate each hole region by this many grid cells so the
            patch overlaps the worn rim (closes the seam line).
        align: per-hole local rigid ICP of the recon graft onto the worn collar.
        collar_iter: rim-collar band width (grid cells) used as ICP target and
            height-field anchor around each hole.
        patch_cell_mult: fallback resample spacing as a multiple of the global
            median spacing, used only when a hole's worn collar is too small to
            estimate a local spacing.
        support_radius_mult: max distance (x median spacing) a sample node may
            sit from a support point; nodes beyond it are dropped so the spline
            is never extrapolated past the available data.
        min_support: minimum support points required to resample a hole.
        tps_smooth_mult: thin-plate-spline smoothing as a multiple of the local
            spacing (RBF residual tolerance); >0 absorbs worn-collar noise so the
            patch stays smooth instead of chasing every speckle.
        jitter_frac: random jitter (x local spacing) applied to grid nodes to
            break the axis-aligned lattice signature that meshes as a rigid edge.
        tps_neighbors: nearest supports used per query for the local TPS solve
            (keeps the fit tractable when a hole has thousands of supports).

    Returns:
        (patched (N + P, 3), is_patch (N + P,) bool, info).
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from scipy.interpolate import RBFInterpolator

    info = hole_result["info"]
    hole_grid = hole_result.get("hole_grid")
    if hole_grid is None or not hole_grid.any():
        return real_worn.copy(), np.zeros(len(real_worn), dtype=bool), {
            "status": "no_holes", "n_patch_points": 0}

    msp = float(info.get("median_spacing", 1.0))
    cell = float(info["cell_size"])
    mins = np.array(info["grid_origin"])
    H, W = info["grid_shape"]
    axes, normal, centroid = _occlusal_basis(info)

    worn_uv, worn_h = _to_uvh(real_worn, axes, normal, centroid)
    recon_uv, recon_h = _to_uvh(recon, axes, normal, centroid)

    def _coarse_cell(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.floor((uv - mins) / cell).astype(int)
        gi = ij[:, 0] + 1
        gj = ij[:, 1] + 1
        valid = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
        return gi, gj, valid

    wgi, wgj, wv = _coarse_cell(worn_uv)
    rgi, rgj, rv = _coarse_cell(recon_uv)

    hlab, hn = ndimage.label(hole_grid)
    patches: List[np.ndarray] = []
    per_hole: List[Dict] = []

    for h in range(1, hn + 1):
        cells_h = (hlab == h)
        fill_h = ndimage.binary_dilation(cells_h, iterations=fill_dilate) \
            if fill_dilate > 0 else cells_h
        collar_h = ndimage.binary_dilation(cells_h, iterations=collar_iter) & ~cells_h
        region_h = fill_h | collar_h

        r_region = np.zeros(len(recon), dtype=bool)
        r_region[rv] = region_h[rgi[rv], rgj[rv]]
        w_collar = np.zeros(len(real_worn), dtype=bool)
        w_collar[wv] = collar_h[wgi[wv], wgj[wv]]
        if not r_region.any():
            continue

        # 1. Align the reconstruction region onto the worn rim collar.
        region_pts = recon[r_region]
        aligned = False
        if align and int(w_collar.sum()) >= 50 and int(r_region.sum()) >= 10:
            region_pts = _refine_icp_to_worn(region_pts, real_worn[w_collar], msp)
            aligned = True

        # Keep only the aligned recon points that fall inside the hole fill
        # region (interior shape); the collar recon was only an ICP handle.
        reg_uv, reg_h = _to_uvh(region_pts, axes, normal, centroid)
        rgi2, rgj2, rv2 = _coarse_cell(reg_uv)
        is_fill = np.zeros(len(region_pts), dtype=bool)
        is_fill[rv2] = fill_h[rgi2[rv2], rgj2[rv2]]

        # 2. Supports: aligned recon interior (shape) + worn rim collar (seam
        # anchor), kept as world points so we can build a per-hole local frame.
        recon_fill_pts = region_pts[is_fill]
        collar_pts = real_worn[w_collar]
        supp_world = np.vstack([recon_fill_pts, collar_pts])

        def _bail(reason: str):
            patches.append(recon_fill_pts)
            per_hole.append({"hole_id": h - 1, "n_fill": int(is_fill.sum()),
                             "aligned": aligned, "resampled": False,
                             "reason": reason})

        if len(supp_world) < max(min_support, 4):
            _bail("too_few_supports")
            continue

        # 3. Local height-field frame from the supports: (a, b) span their PCA
        # plane, height is the thin (smallest-variance) direction -> the field
        # stays single-valued even on a near-vertical cusp wall.
        s_ctr = supp_world.mean(axis=0)
        _, _, s_vt = np.linalg.svd(supp_world - s_ctr, full_matrices=False)
        s_axes = s_vt[:2]
        s_nrm = np.cross(s_axes[0], s_axes[1])
        s_nrm /= np.linalg.norm(s_nrm) + 1e-12

        def _to_ab(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            d = p - s_ctr
            return d @ s_axes.T, d @ s_nrm

        def _ab_to_world(ab: np.ndarray, c: np.ndarray) -> np.ndarray:
            return s_ctr + ab @ s_axes + c[:, None] * s_nrm

        supp_ab, supp_c = _to_ab(supp_world)

        # Local spacing = worn-collar NN spacing so patch density matches the
        # surrounding real surface (Poisson facets on density steps at a seam).
        if len(collar_pts) >= 5:
            cd, _ = NearestNeighbors(n_neighbors=2).fit(collar_pts).kneighbors(collar_pts)
            local_sp = max(float(np.median(cd[:, 1])), 1e-6)
        else:
            local_sp = max(patch_cell_mult * msp, 1e-6)

        # Jittered grid over the supports' (a, b) extent. The jitter breaks the
        # axis-aligned lattice that otherwise meshes as a hard, rigid edge.
        rng = np.random.default_rng(h)
        a0, b0 = supp_ab.min(axis=0)
        a1, b1 = supp_ab.max(axis=0)
        aa = np.arange(a0, a1 + local_sp, local_sp)
        bb = np.arange(b0, b1 + local_sp, local_sp)
        ga, gb = np.meshgrid(aa, bb, indexing="ij")
        node_ab = np.column_stack([ga.ravel(), gb.ravel()])
        node_ab = node_ab + rng.uniform(-jitter_frac, jitter_frac,
                                        node_ab.shape) * local_sp
        if len(node_ab) == 0:
            _bail("empty_grid")
            continue

        # 4. Thin-plate-spline height field (curvature-minimizing => smooth and
        # C1 across the seam). Smoothing absorbs worn-collar noise.
        smoothing = (tps_smooth_mult * local_sp) ** 2
        n_nbr = int(min(tps_neighbors, len(supp_ab)))
        try:
            rbf = RBFInterpolator(supp_ab, supp_c,
                                  kernel="thin_plate_spline",
                                  smoothing=smoothing, neighbors=n_nbr)
            node_c = rbf(node_ab)
        except Exception:
            _bail("rbf_failed")
            continue

        node_world = _ab_to_world(node_ab, node_c)

        # Gate nodes: (i) close to a support (no TPS extrapolation past data)
        # and (ii) world position lands in the hole fill region of the grid.
        ab_tree = cKDTree(supp_ab)
        d0, _ = ab_tree.query(node_ab, k=1)
        near_ok = d0 < support_radius_mult * msp
        nuv, _ = _to_uvh(node_world, axes, normal, centroid)
        ngi, ngj, nv = _coarse_cell(nuv)
        in_fill = np.zeros(len(node_world), dtype=bool)
        in_fill[nv] = fill_h[ngi[nv], ngj[nv]]
        keep = near_ok & in_fill
        if not keep.any():
            _bail("no_nodes_kept")
            continue

        patch_pts = node_world[keep]
        patches.append(patch_pts)
        per_hole.append({"hole_id": h - 1, "n_fill": int(len(patch_pts)),
                         "aligned": aligned, "resampled": True,
                         "local_spacing": local_sp})

    patch = np.vstack(patches) if patches else np.empty((0, 3))
    patched = np.vstack([real_worn, patch])
    is_patch = np.zeros(len(patched), dtype=bool)
    is_patch[len(real_worn):] = True

    pinfo = {
        "status": "ok",
        "n_worn_points": int(len(real_worn)),
        "n_patch_points": int(len(patch)),
        "n_total_points": int(len(patched)),
        "fill_dilate": fill_dilate,
        "align": bool(align),
        "collar_iter": collar_iter,
        "patch_cell_mult": patch_cell_mult,
        "n_holes_patched": len(patches),
        "per_hole": per_hole,
    }
    return patched, is_patch, pinfo


# ===========================================================================
# 3e. MESH-SPACE HOLE PATCHING (harmonic recon blend -- no proud scab/seam)
# ===========================================================================
# The point-resample-then-Poisson route grafts a *separate* sheet of recon
# points: ~30-45% of them sit proud of the worn surface, and re-meshing that
# raised sheet against the worn cloud produces a crusty rim. Instead we mesh the
# worn cloud ONCE (Open3D Poisson is watertight, so every gap is already bridged
# by a smooth membrane), then lift only the *fabricated* bridge vertices onto
# the reconstruction shape, weighted by a harmonic field that is exactly 0 on
# the surrounding real surface and 1 deep inside the fill. The seam therefore
# stays welded to real geometry (no crusty ring) and the fill rises smoothly to
# the recon cusp (no proud step).

def _uniform_laplacian(n_vert: int, faces: np.ndarray):
    """Sparse graph Laplacian ``L = D - A`` and binary adjacency ``A`` (degree
    ``deg``) from triangle edges."""
    from scipy.sparse import coo_matrix, diags
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                   shape=(n_vert, n_vert)).tocsr()
    A = (A > 0).astype(float)                         # dedupe -> binary
    deg = np.asarray(A.sum(1)).ravel()
    return diags(deg) - A, A, deg


def _boundary_loops(faces: np.ndarray, min_len: int = 8) -> List[np.ndarray]:
    """Ordered open-boundary vertex loops of a triangle mesh (no networkx).

    A boundary edge belongs to exactly one face; manifold boundary vertices have
    two such edges, so each loop is recovered by a simple walk.
    """
    from collections import defaultdict
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    if len(bnd) == 0:
        return []
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in bnd:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    loops: List[np.ndarray] = []
    seen = set()
    for start in list(adj):
        if start in seen or len(adj[start]) != 2:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = -1, start
        while True:
            nb = adj[cur]
            nxt = nb[0] if nb[0] != prev else (nb[1] if len(nb) > 1 else -1)
            if nxt < 0 or nxt == start or nxt in seen or len(adj[nxt]) != 2:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= min_len:
            loops.append(np.array(loop, dtype=np.int64))
    return loops


def _triangulate_loop(loop_xyz: np.ndarray, loop_gidx: np.ndarray,
                      spacing: float, base: int):
    """Triangulate a 3D boundary loop with a densified flat cap in its own PCA
    plane. Returns (interior_xyz (k,3), faces (m,3) global) or (None, None).

    The cap is intentionally flat -- the harmonic blend reshapes it to the recon
    cusp afterwards; here we only need a closed, evenly-sampled triangulation.
    """
    from scipy.spatial import Delaunay
    from matplotlib.path import Path
    c = loop_xyz.mean(0)
    _, _, Vt = np.linalg.svd(loop_xyz - c, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    uv = np.column_stack([(loop_xyz - c) @ e1, (loop_xyz - c) @ e2])
    poly = Path(uv)
    lo, hi = uv.min(0), uv.max(0)
    xs = np.arange(lo[0] + spacing, hi[0], spacing)
    ys = np.arange(lo[1] + spacing, hi[1], spacing)
    if len(xs) and len(ys):
        gx, gy = np.meshgrid(xs, ys)
        g = np.column_stack([gx.ravel(), gy.ravel()])
        g = g[poly.contains_points(g, radius=-0.5 * spacing)]
    else:
        g = np.empty((0, 2))
    allp = np.vstack([uv, g])
    if len(allp) < 3:
        return None, None
    try:
        tri = Delaunay(allp)
    except Exception:
        return None, None
    keep = poly.contains_points(allp[tri.simplices].mean(1))
    simp = tri.simplices[keep]
    if len(simp) == 0:
        return None, None
    nloop = len(loop_gidx)
    gmap = np.empty(len(allp), dtype=np.int64)
    gmap[:nloop] = loop_gidx
    gmap[nloop:] = base + np.arange(len(g))
    faces = gmap[simp]
    new_xyz = c + g[:, 0:1] * e1 + g[:, 1:2] * e2
    return new_xyz, faces


def mesh_space_patch_holes(real_worn: np.ndarray,
                           recon: np.ndarray,
                           hole_result: Dict,
                           tmp_dir: str,
                           collar_iter: int = 6,
                           real_mult: float = 1.3,
                           deep_mult: float = 3.0,
                           align: bool = True,
                           smooth_iter: int = 12,
                           occlusal_gate_frac: float = 0.30):
    """Fill holes in the mesh domain via a harmonic reconstruction blend.

    Mesh the worn cloud once at the normal density trim (clean surface; only the
    genuine wear holes survive as open boundary loops). Triangulate each wear
    hole with a densified flat cap, then lift the cap vertices onto the recon
    shape with a harmonic weight that is exactly 0 on the surrounding real rim
    and 1 deep inside the fill -- so the seam stays welded to real geometry (no
    crusty ring) and the fill rises smoothly to the cusp (no proud step).

    Args:
        real_worn: (N, 3) real worn surface (icp_aligned), with holes.
        recon: (M, 3) reconstruction, same frame as real_worn.
        hole_result: output of ``detect_holes`` (occlusal grid + holes).
        tmp_dir: scratch dir for the intermediate worn-only mesh.
        collar_iter: rim-collar width (grid cells) used for the per-hole recon
            ICP target and to bound the fill region on the mesh.
        real_mult: a cap vertex counts as fill (free to move) when its nearest
            real worn point is farther than this x median spacing.
        deep_mult: fill vertices farther than this x median spacing are "deep"
            (blend weight pinned to 1 -> full recon shape).
        align: per-hole rigid ICP of the recon region onto the worn collar.
        smooth_iter: volume-preserving (Taubin) relaxation iterations applied to
            the fill + its 1-ring only; real anatomy is never touched.

    Returns:
        (trimesh or None, info dict).
    """
    import trimesh
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    from scipy import ndimage

    info = hole_result["info"]
    hole_grid = hole_result.get("hole_grid")
    if hole_grid is None or not hole_grid.any():
        return None, {"status": "no_holes"}

    msp = float(info.get("median_spacing", 1.0))
    cell = float(info["cell_size"])
    mins = np.array(info["grid_origin"])
    H, W = info["grid_shape"]
    axes, normal, centroid = _occlusal_basis(info)

    def _cell_of(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ij = np.floor((_to_uvh(pts, axes, normal, centroid)[0] - mins) / cell).astype(int)
        return (np.clip(ij[:, 0] + 1, 0, H - 1), np.clip(ij[:, 1] + 1, 0, W - 1))

    # 1. Clean worn-only mesh at the standard density trim: the real surface
    # meshes cleanly and only the genuine wear holes remain as open boundary
    # loops (the outer skirt + pinholes are removed by the trim as usual).
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, "_worn_only_mesh.ply")
    depth = safe_mesh_to_file(real_worn, tmp)
    if not depth or not os.path.exists(tmp):
        return None, {"status": "worn_mesh_failed"}
    M = trimesh.load(tmp, process=False)
    try:
        os.remove(tmp)
    except OSError:
        pass
    M.update_faces(M.nondegenerate_faces())
    M.remove_unreferenced_vertices()
    if len(M.faces):
        lbl = trimesh.graph.connected_component_labels(M.face_adjacency)
        keep = lbl == int(np.argmax(np.bincount(lbl)))
        if not keep.all():
            M.update_faces(keep)
            M.remove_unreferenced_vertices()

    worn_tree = cKDTree(real_worn)
    hole_region = ndimage.binary_dilation(hole_grid, iterations=collar_iter + 2)

    # Occlusal height gate. The wear holes are occlusal (high along the tooth
    # axis); the cervical base is the low, open bottom of the EDJ shell. In the
    # 2D occlusal projection the open base lands in the SAME (u,v) cells as the
    # holes above it, so without a height gate the base loop gets triangulated /
    # blended -- the "patching the base" artifact. Orient the normal so the
    # detected wear holes are HIGH, then only ever fill above ``occl_cut`` and
    # trim any fabricated base cap below it (re-opening the cervical base).
    h_worn = (real_worn - centroid) @ normal
    rim = hole_result.get("interior_loops") or []
    rim = np.concatenate(rim) if len(rim) else np.array([], dtype=int)
    if len(rim) and rim.max() < len(real_worn):
        if h_worn[rim].mean() < h_worn.mean():
            normal = -normal
            h_worn = -h_worn
    lo_h, hi_h = np.percentile(h_worn, 1.0), np.percentile(h_worn, 99.0)
    occl_cut = lo_h + occlusal_gate_frac * (hi_h - lo_h)

    # 2. Triangulate each wear-hole boundary loop with a densified flat cap. The
    # large outer-rim loop and any tiny pinholes are skipped; only loops whose
    # centroid sits over a detected hole are filled. The new cap vertices are
    # what the harmonic blend then lifts onto the recon shape.
    V = np.asarray(M.vertices, dtype=float)
    F = np.asarray(M.faces)
    new_pts: List[np.ndarray] = []
    new_faces: List[np.ndarray] = []
    offset = len(V)
    n_caps = 0
    for loop in _boundary_loops(F):
        lc = V[loop].mean(0)
        if (lc - centroid) @ normal < occl_cut:   # cervical base / low loop
            continue
        ci, cj = _cell_of(lc[None])
        if not hole_region[ci[0], cj[0]]:          # outer rim / off-hole loop
            continue
        cap_xyz, cap_faces = _triangulate_loop(V[loop], loop, msp, base=offset)
        if cap_xyz is None:
            continue
        new_pts.append(cap_xyz)
        new_faces.append(cap_faces)
        offset += len(cap_xyz)
        n_caps += 1

    # Big holes reopen under the trim and are filled by the caps above; small
    # holes stay bridged by Poisson. Either way the fill vertices are "fabricated"
    # (far from real data) and the harmonic blend below lifts them, so we always
    # continue rather than bailing when no cap was added.
    if new_faces:
        V = np.vstack([V] + new_pts)
        F = np.vstack([F] + new_faces)
        M = trimesh.Trimesh(V, F, process=False)
    nV = len(V)

    # 3. Per-vertex nearest real-worn distance + occlusal cell.
    dvert, _ = worn_tree.query(V, k=1)
    vgi, vgj = _cell_of(V)
    h_vert = (V - centroid) @ normal
    occlusal = h_vert >= occl_cut
    fabricated = (dvert > real_mult * msp) & occlusal   # never move the base

    wgi, wgj = _cell_of(real_worn)
    rgi, rgj = _cell_of(recon)

    # 4. Per hole: ICP the recon region to the worn collar, then set the recon
    #    target for the fabricated vertices over that hole.
    hlab, hn = ndimage.label(hole_grid)
    target = V.copy()
    over_any = np.zeros(nV, dtype=bool)
    n_holes_done = 0
    for h in range(1, hn + 1):
        cells_h = (hlab == h)
        fillcol = ndimage.binary_dilation(cells_h, iterations=collar_iter)
        collar = fillcol & ~cells_h
        v_over = fillcol[vgi, vgj] & fabricated
        if int(v_over.sum()) < 10:
            continue
        region_pts = recon[fillcol[rgi, rgj]]
        if len(region_pts) < 10:
            continue
        w_collar = collar[wgi, wgj]
        if align and int(w_collar.sum()) >= 50:
            region_pts = _refine_icp_to_worn(region_pts, real_worn[w_collar], msp)
        idx = cKDTree(region_pts).query(V[v_over], k=1)[1]
        target[v_over] = region_pts[idx]
        over_any |= v_over
        n_holes_done += 1

    if not over_any.any():
        M.fix_normals()
        return M, {"status": "no_fill_vertices", "mesh_poisson_depth": int(depth)}

    # 4. Harmonic blend weight: 0 on the real surface, 1 deep inside the fill.
    L, A, deg = _uniform_laplacian(nV, F)
    deep = over_any & (dvert > deep_mult * msp)
    free = over_any & ~deep
    pinned = ~free
    w = np.zeros(nV)
    w[deep] = 1.0
    if free.any():
        try:
            w[free] = spsolve(L[free][:, free].tocsc(),
                              -(L[free][:, pinned] @ w[pinned]))
        except Exception:
            w[free] = float(deep.any())   # degenerate: snap fill to recon
    w = np.clip(w, 0.0, 1.0)

    # 5. Displace fabricated fill vertices toward the recon target.
    V2 = V.copy()
    V2[over_any] = V[over_any] + w[over_any, None] * (target[over_any] - V[over_any])

    # 6. Volume-preserving (Taubin) relaxation on the fill + its 1-ring only, so
    #    the blend and any projection speckle settle without flattening the cusp
    #    or disturbing real anatomy.
    ring = over_any | ((np.asarray((A[over_any].sum(0))).ravel() > 0) & fabricated)
    ring_idx = np.where(ring)[0]
    if len(ring_idx) and smooth_iter > 0:
        invdeg = 1.0 / np.maximum(deg, 1)
        for i in range(smooth_iter):
            lap = (A @ V2) * invdeg[:, None] - V2
            V2[ring_idx] += (0.5 if i % 2 == 0 else -0.53) * lap[ring_idx]

    M.vertices = V2

    # 7. Re-open the cervical base: drop fabricated faces that sit below the
    # occlusal gate (the Poisson base cap + any low skirt). Real lower-flank
    # surface has support points nearby (not fabricated) and is kept.
    fab_low = (dvert > real_mult * msp) & (h_vert < occl_cut)
    drop_face = fab_low[F].any(axis=1)
    n_base_trimmed = int(drop_face.sum())
    if drop_face.any():
        M.update_faces(~drop_face)
        M.remove_unreferenced_vertices()
        if len(M.faces):
            lbl = trimesh.graph.connected_component_labels(M.face_adjacency)
            keepc = lbl == int(np.argmax(np.bincount(lbl)))
            if not keepc.all():
                M.update_faces(keepc)
                M.remove_unreferenced_vertices()
    M.fix_normals()
    return M, {
        "status": "ok",
        "method": "mesh_space_harmonic_blend",
        "mesh_poisson_depth": int(depth),
        "n_caps_triangulated": int(n_caps),
        "n_holes_filled": int(n_holes_done),
        "n_fill_vertices": int(over_any.sum()),
        "n_deep_vertices": int(deep.sum()),
        "n_smoothed_vertices": int(len(ring_idx)),
        "n_base_faces_trimmed": n_base_trimmed,
        "occl_cut": float(occl_cut),
    }


def render_patch_png(patched: np.ndarray,
                     is_patch: np.ndarray,
                     real_worn: np.ndarray,
                     interior_loops: List[np.ndarray],
                     filepath: str,
                     title: str,
                     max_bg: int = 30000) -> None:
    """Before/after PNG: left = worn with red hole rims, right = patched with
    the green graft filling the holes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worn = np.asarray(real_worn)
    bg = np.arange(len(worn))
    if len(worn) > max_bg:
        bg = np.random.default_rng(0).choice(len(worn), max_bg, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    # Before
    axes[0].scatter(worn[bg, 0], worn[bg, 1], s=1, c="lightgray", linewidths=0)
    for loop in interior_loops:
        axes[0].scatter(worn[loop, 0], worn[loop, 1], s=6, c="#dc3c32", linewidths=0)
    axes[0].set_title(f"before: {len(interior_loops)} hole(s) detected")
    axes[0].set_aspect("equal"); axes[0].axis("off")
    # After
    keep = ~is_patch
    kb = np.where(keep)[0]
    if len(kb) > max_bg:
        kb = np.random.default_rng(0).choice(kb, max_bg, replace=False)
    pb = np.where(is_patch)[0]
    axes[1].scatter(patched[kb, 0], patched[kb, 1], s=1, c="lightgray", linewidths=0)
    axes[1].scatter(patched[pb, 0], patched[pb, 1], s=4, c="#2ca02c", linewidths=0)
    axes[1].set_title(f"after: holes filled ({int(is_patch.sum())} graft pts)")
    axes[1].set_aspect("equal"); axes[1].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_extras_png(points: np.ndarray,
                      keep_mask: np.ndarray,
                      filepath: str,
                      title: str,
                      max_bg: int = 30000) -> None:
    """3-view PNG of the worn cloud with removed extras highlighted in orange
    (kept surface in grey)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points)
    removed = np.where(~keep_mask)[0]
    kept = np.where(keep_mask)[0]
    if len(kept) > max_bg:
        kept = np.random.default_rng(0).choice(kept, max_bg, replace=False)

    views = [((0, 1), "top (XY)"), ((0, 2), "side (XZ)"), ((1, 2), "front (YZ)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ((a, b), name) in zip(axes, views):
        ax.scatter(pts[kept, a], pts[kept, b], s=1, c="lightgray", linewidths=0)
        if len(removed):
            ax.scatter(pts[removed, a], pts[removed, b], s=4, c="#f08c1e", linewidths=0)
        ax.set_aspect("equal"); ax.set_title(name); ax.axis("off")
    fig.suptitle(f"{title}  (grey=kept, orange={len(removed)} removed extras)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 4. PATCH GRAFT (index-space swap with transition-band seam blending)
# ===========================================================================

def graft_index_swap(worn_points: np.ndarray,
                     recon_points: np.ndarray,
                     mask: np.ndarray,
                     transition_band: float = 3.0) -> Tuple[np.ndarray, Dict]:
    """Graft the reconstruction cap onto the worn input by point index.

    Inside the wear mask, points come from the reconstruction; outside, from
    the worn input. Unworn points within ``transition_band`` x median-spacing
    of the wear region are linearly blended to hide the seam.

    Args:
        worn_points: Worn input cloud (N, 3).
        recon_points: Reconstruction cloud (N, 3), same ordering.
        mask: Boolean wear mask (N,).
        transition_band: Blend band width as a multiple of median spacing.

    Returns:
        (patched (N, 3), info dict).
    """
    patched = worn_points.copy()
    patched[mask] = recon_points[mask]

    info: Dict = {"transition_band_mult": transition_band}

    worn_idx = np.where(mask)[0]
    unworn_idx = np.where(~mask)[0]
    if len(worn_idx) == 0 or len(unworn_idx) == 0 or transition_band <= 0:
        info["n_blended"] = 0
        return patched, info

    # Median point spacing of the worn input.
    nn2 = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(worn_points)
    spacing, _ = nn2.kneighbors(worn_points)
    median_spacing = float(np.median(spacing[:, 1]))
    band = transition_band * median_spacing

    # Distance from each unworn point to the nearest worn point.
    nn_worn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(worn_points[worn_idx])
    dist, _ = nn_worn.kneighbors(worn_points[unworn_idx])
    dist = dist.ravel()

    in_band = dist < band
    band_local = np.where(in_band)[0]
    if len(band_local) > 0:
        # alpha=0 at the margin (use recon), alpha=1 at band edge (use input).
        alpha = np.clip(dist[band_local] / max(band, 1e-12), 0.0, 1.0)[:, None]
        global_pts = unworn_idx[band_local]
        patched[global_pts] = (alpha * worn_points[global_pts]
                               + (1.0 - alpha) * recon_points[global_pts])

    info["median_spacing"] = median_spacing
    info["band_width"] = band
    info["n_blended"] = int(len(band_local))
    return patched, info


# ===========================================================================
# 5b. CORRESPONDENCE-SPACE CONFIDENCE BLEND (recommended reconstruction)
# ===========================================================================
# worn_input.ply and reconstructed.ply share the 100k correspondence ordering,
# so reconstruction is a per-index blend: keep the real worn point where the
# surface is intact, fade smoothly to the SSM reconstruction only where material
# has worn away (large outward deviation). Because nothing is meshed from the
# raw holed scan, the cervical base is never closed or "patched", and there are
# no graft seams/scabs -- the result is as clean as reconstructed.ply while
# retaining the real worn surface's fine detail everywhere it is intact.

def correspondence_blend(worn_points: np.ndarray,
                         recon_points: np.ndarray,
                         lo_pct: float = 70.0,
                         hi_pct: float = 92.0,
                         smooth_k: int = 12,
                         smooth_iters: int = 2) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Blend the index-aligned worn cloud with the SSM reconstruction.

    Args:
        worn_points: (N, 3) corresponded worn cloud (worn_input.ply).
        recon_points: (N, 3) SSM reconstruction, SAME ordering.
        lo_pct/hi_pct: outward-deviation percentiles mapped to blend weight 0/1;
            points below ``lo_pct`` keep the real worn surface, above ``hi_pct``
            use the recon, with a smoothstep ramp between.
        smooth_k/smooth_iters: KNN averaging of the weight field so the blend has
            no speckle or hard seam.

    Returns:
        (blended (N, 3), weight (N,) in [0, 1], info dict).
    """
    centroid = worn_points.mean(axis=0)
    outward = worn_points - centroid
    outward /= np.linalg.norm(outward, axis=1, keepdims=True) + 1e-12
    disp = recon_points - worn_points
    s = np.maximum(np.einsum("ij,ij->i", disp, outward), 0.0)   # outward deviation

    lo, hi = np.percentile(s, lo_pct), np.percentile(s, hi_pct)
    w = np.clip((s - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)                                 # smoothstep

    if smooth_k > 0 and smooth_iters > 0:
        knn = _knn_indices(worn_points, smooth_k)
        for _ in range(smooth_iters):
            w = 0.5 * w + 0.5 * w[knn].mean(axis=1)

    blended = worn_points + w[:, None] * disp
    info = {
        "lo_deviation": float(lo),
        "hi_deviation": float(hi),
        "frac_blended": float((w > 0.05).mean()),
        "frac_full_recon": float((w > 0.95).mean()),
        "mean_weight": float(w.mean()),
    }
    return blended, w, info


# ===========================================================================
# 6. OPTIONAL MESH SURGERY (cut recon cap, stitch onto input mesh)
# ===========================================================================

def _label_mesh_vertices(mesh, source_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Label each mesh vertex worn/unworn via its nearest corresponded point."""
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(source_raw)
    _, idx = nn.kneighbors(np.asarray(mesh.vertices))
    return mask[idx.ravel()]


def mesh_surgery_patch(input_mesh,
                       recon_mesh,
                       worn_raw: np.ndarray,
                       recon_raw: np.ndarray,
                       mask: np.ndarray):
    """Cut the recon cap along the wear region and stitch it onto the input.

    Approximate surgery (operates on pre-built meshes; meshing happens in an
    isolated process beforehand for stability):
      1. Label vertices worn/unworn by nearest corresponded point.
      2. Open a hole in the input mesh by removing faces touching the wear
         region; keep only the cap faces (fully worn) of the recon mesh.
      3. Concatenate and weld nearby boundary vertices, then fill residual
         gaps to recover a closed surface.

    Args:
        input_mesh: trimesh of the worn input (raw mm space).
        recon_mesh: trimesh of the reconstruction (raw mm space).
        worn_raw: corresponded worn cloud in raw space (for vertex labeling).
        recon_raw: corresponded recon cloud in raw space (for vertex labeling).
        mask: boolean wear mask over the corresponded points.

    Returns:
        (stitched_trimesh, info dict).
    """
    import trimesh

    info: Dict = {}
    input_vert_worn = _label_mesh_vertices(input_mesh, worn_raw, mask)
    recon_vert_worn = _label_mesh_vertices(recon_mesh, recon_raw, mask)

    # Input: drop any face that touches the wear region -> opens a clean hole.
    in_faces = np.asarray(input_mesh.faces)
    face_touches_wear = input_vert_worn[in_faces].any(axis=1)
    holed_input = input_mesh.copy()
    holed_input.update_faces(~face_touches_wear)
    holed_input.remove_unreferenced_vertices()

    # Recon: keep only faces fully inside the wear region -> the cap.
    rc_faces = np.asarray(recon_mesh.faces)
    cap_faces = recon_vert_worn[rc_faces].all(axis=1)
    cap = recon_mesh.copy()
    cap.update_faces(cap_faces)
    cap.remove_unreferenced_vertices()

    info["holed_input_faces"] = int(len(holed_input.faces))
    info["cap_faces"] = int(len(cap.faces))

    combined = trimesh.util.concatenate([holed_input, cap])

    # Weld coincident/near-coincident vertices to stitch the two loops, then
    # fill residual gaps along the seam to recover a closed surface.
    try:
        combined.merge_vertices()
        trimesh.repair.fill_holes(combined)
        trimesh.repair.fix_normals(combined)
        trimesh.repair.fix_winding(combined)
    except Exception:
        pass

    info["stitched_vertices"] = int(len(combined.vertices))
    info["stitched_faces"] = int(len(combined.faces))
    info["watertight"] = bool(combined.is_watertight)
    return combined, info


# ===========================================================================
# PER-TOOTH DRIVER
# ===========================================================================

def _save_ring(points: np.ndarray,
               ordered_idx: np.ndarray,
               result_dir: str,
               raw_points: Optional[np.ndarray],
               filename: str = "wear_ring.ply") -> None:
    """Save the ordered ring as a point cloud (normalized) and JSON metadata."""
    if len(ordered_idx) == 0:
        return
    ring_pts = points[ordered_idx]
    save_point_cloud(ring_pts, os.path.join(result_dir, filename))
    ring_json = {
        "n_points": int(len(ordered_idx)),
        "ordered_indices": ordered_idx.tolist(),
        "coordinates_normalized": ring_pts.tolist(),
    }
    if raw_points is not None:
        ring_raw = raw_points[ordered_idx]
        save_point_cloud(ring_raw, os.path.join(result_dir, "wear_ring_in_input_space.ply"))
        ring_json["coordinates_input_space"] = ring_raw.tolist()
    with open(os.path.join(result_dir, "wear_ring.json"), "w") as f:
        json.dump(ring_json, f, indent=2)


# --- Annotation / visualization helpers --------------------------------------

# RGB colors for the wear annotation (uint8).
_COLOR_UNWORN = (170, 170, 170)   # grey   - intact surface
_COLOR_WEAR = (220, 60, 50)       # red    - detected wear / hole
_COLOR_PATCH = (40, 170, 70)      # green  - reconstruction graft filling a hole
_COLOR_EXTRA = (240, 140, 30)     # orange - removed micro-CT extra / artifact
_COLOR_RING = (40, 190, 70)       # green - wear-margin ring


def write_colored_ply(points: np.ndarray, colors: np.ndarray, filepath: str) -> None:
    """Write a point cloud with per-vertex RGB to a binary PLY (no Open3D)."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    n = len(points)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def make_annotation_colors(n_points: int,
                           wear_mask: np.ndarray,
                           ring_idx: np.ndarray) -> np.ndarray:
    """Per-point RGB: grey intact, red wear, green ring (drawn on top)."""
    colors = np.tile(np.array(_COLOR_UNWORN, dtype=np.uint8), (n_points, 1))
    colors[wear_mask] = _COLOR_WEAR
    if len(ring_idx) > 0:
        colors[ring_idx] = _COLOR_RING
    return colors


def render_annotation_png(points: np.ndarray,
                          wear_mask: np.ndarray,
                          ring_idx: np.ndarray,
                          filepath: str,
                          title: str,
                          max_bg: int = 25000) -> None:
    """Render a 3-view (top/side/front) PNG of the wear annotation for a quick
    visual check without opening a 3D viewer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points)
    worn = np.where(wear_mask)[0]
    unworn = np.where(~wear_mask)[0]
    if len(unworn) > max_bg:
        sel = np.random.default_rng(0).choice(unworn, max_bg, replace=False)
    else:
        sel = unworn

    views = [((0, 1), "top (XY)"), ((0, 2), "side (XZ)"), ((1, 2), "front (YZ)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ((a, b), name) in zip(axes, views):
        ax.scatter(pts[sel, a], pts[sel, b], s=1, c="lightgray", linewidths=0)
        if len(worn) > 0:
            ax.scatter(pts[worn, a], pts[worn, b], s=2, c="#dc3c32", linewidths=0)
        if len(ring_idx) > 0:
            ax.scatter(pts[ring_idx, a], pts[ring_idx, b], s=5, c="#28be46",
                       linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{title}  (grey=intact, red=wear, green=ring)", fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_annotation(worn_points: np.ndarray,
                    worn_raw: Optional[np.ndarray],
                    wear_mask: np.ndarray,
                    ring_idx: np.ndarray,
                    result_dir: str,
                    title: str,
                    annotation_png: bool = True) -> None:
    """Save the color-coded wear annotation (normalized + input space PLY) and
    an optional multi-view PNG."""
    colors = make_annotation_colors(len(worn_points), wear_mask, ring_idx)
    write_colored_ply(worn_points, colors,
                      os.path.join(result_dir, "wear_annotation.ply"))
    if worn_raw is not None:
        write_colored_ply(worn_raw, colors,
                          os.path.join(result_dir, "wear_annotation_in_input_space.ply"))
    if annotation_png:
        try:
            render_annotation_png(worn_points, wear_mask, ring_idx,
                                  os.path.join(result_dir, "wear_annotation.png"),
                                  title)
        except Exception as e:
            print(f"    [WARN] annotation PNG failed: {e}")


# --- Hole annotation (red interior-hole outlines) ----------------------------

_COLOR_OUTER_RIM = (60, 110, 220)   # blue  - outer perimeter


def make_hole_colors(n_points: int,
                     outer_rim: np.ndarray,
                     interior_loops: List[np.ndarray]) -> np.ndarray:
    """Per-point RGB: grey surface, blue outer rim, red interior hole rims."""
    colors = np.tile(np.array(_COLOR_UNWORN, dtype=np.uint8), (n_points, 1))
    if len(outer_rim) > 0:
        colors[outer_rim] = _COLOR_OUTER_RIM
    for loop in interior_loops:
        colors[loop] = _COLOR_WEAR
    return colors


def render_holes_png(points: np.ndarray,
                     outer_rim: np.ndarray,
                     interior_loops: List[np.ndarray],
                     filepath: str,
                     title: str,
                     max_bg: int = 30000) -> None:
    """3-view PNG: grey surface, blue outer rim, red interior-hole outlines
    (drawn as connected closed loops)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points)
    n = len(pts)
    bg = np.arange(n)
    if n > max_bg:
        bg = np.random.default_rng(0).choice(n, max_bg, replace=False)

    views = [((0, 1), "top (XY)"), ((0, 2), "side (XZ)"), ((1, 2), "front (YZ)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ((a, b), name) in zip(axes, views):
        ax.scatter(pts[bg, a], pts[bg, b], s=1, c="lightgray", linewidths=0)
        if len(outer_rim) > 0:
            ax.scatter(pts[outer_rim, a], pts[outer_rim, b], s=2,
                       c="#3c6edc", linewidths=0)
        # Scatter the rim points (no connecting polyline: angular-sort ordering
        # zig-zags on elongated cracks; dense rim points read as a clean outline).
        for loop in interior_loops:
            ax.scatter(pts[loop, a], pts[loop, b], s=6, c="#dc3c32", linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.axis("off")
    n_holes = len(interior_loops)
    fig.suptitle(f"{title}  (grey=surface, blue=outer rim, red={n_holes} hole rims)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_hole_annotation(points: np.ndarray,
                         points_raw: Optional[np.ndarray],
                         hole_result: Dict,
                         result_dir: str,
                         title: str,
                         annotation_png: bool = True) -> None:
    """Save geometric-hole annotation: colored PLY (normalized + input space),
    multi-view PNG, and holes.json with each interior-hole loop."""
    outer_rim = hole_result["outer_rim"]
    interior_loops = hole_result["interior_loops"]
    n = len(points)
    colors = make_hole_colors(n, outer_rim, interior_loops)

    write_colored_ply(points, colors, os.path.join(result_dir, "hole_annotation.ply"))
    if points_raw is not None:
        write_colored_ply(points_raw, colors,
                          os.path.join(result_dir, "hole_annotation_in_input_space.ply"))

    holes_json = {
        "n_interior_holes": len(interior_loops),
        "outer_rim": {
            "n_points": int(len(outer_rim)),
            "ordered_indices": outer_rim.tolist(),
        },
        "interior_holes": [],
        "info": hole_result["info"],
    }
    for i, loop in enumerate(interior_loops):
        entry = {
            "hole_id": i,
            "n_points": int(len(loop)),
            "ordered_indices": loop.tolist(),
            "perimeter": _loop_perimeter(points, loop),
            "coordinates_normalized": points[loop].tolist(),
        }
        if points_raw is not None:
            entry["coordinates_input_space"] = points_raw[loop].tolist()
        holes_json["interior_holes"].append(entry)
    with open(os.path.join(result_dir, "holes.json"), "w") as f:
        json.dump(holes_json, f, indent=2)

    if annotation_png:
        try:
            render_holes_png(points, outer_rim, interior_loops,
                             os.path.join(result_dir, "hole_annotation.png"), title)
        except Exception as e:
            print(f"    [WARN] hole annotation PNG failed: {e}")


def process_tooth(worn_name: str,
                  recon_dir: str,
                  correspondence_dir: str,
                  artificial_wear_dir: Optional[str],
                  output_dir: str,
                  deviation_percentile: float,
                  deviation_threshold: Optional[float],
                  min_component_size: int,
                  transition_band: float,
                  mode: str,
                  detect_only: bool = False,
                  annotation_png: bool = True,
                  detect_mode: str = "deviation",
                  hole_cell_mult: float = 2.5,
                  hole_min_cells: int = 6,
                  clean_extras_flag: bool = False,
                  extra_dist_mult: float = 6.0,
                  min_extra_cluster: int = 150,
                  patch_align: bool = True,
                  patch_collar: int = 4,
                  no_mesh: bool = False) -> Optional[Dict]:
    """Run the full patch pipeline for one worn tooth. Returns a summary dict.

    detect_mode:
      - "deviation": wear from worn_input vs reconstructed (index-aligned).
      - "holes":     on the REAL worn surface (icp_aligned): clean micro-CT
                     extras (Part A), geometrically detect open holes (Part 1),
                     then fill each hole with locally-aligned reconstruction
                     points (Part 2). Use --detect-only to stop after detection.

    When ``detect_only`` is True the pipeline stops after detection +
    annotation (no graft, mesh or surgery), so detection can be inspected.

    ``no_mesh`` skips Poisson meshing entirely. The point-cloud outputs used by
    every downstream evaluation (Chamfer/RMSE/etc.) are saved before meshing
    runs regardless, so this only affects the `*_mesh.ply` visualization files
    -- useful to skip when Poisson's occasional aborts/slowness aren't needed
    (e.g. batch runs whose only purpose is numeric evaluation).
    """
    src_dir = os.path.join(recon_dir, "reconstructions", worn_name)
    worn_path = os.path.join(src_dir, "worn_input.ply")
    recon_path = os.path.join(src_dir, "reconstructed.ply")

    if not (os.path.exists(worn_path) and os.path.exists(recon_path)):
        print(f"  [SKIP] {worn_name}: missing worn_input.ply or reconstructed.ply")
        return None

    print(f"\n{'-' * 50}")
    print(f"  Processing {worn_name}")

    worn_points = load_point_cloud(worn_path)
    recon_points = load_point_cloud(recon_path)
    if len(worn_points) != len(recon_points):
        print(f"    [SKIP] point-count mismatch: worn={len(worn_points)}, "
              f"recon={len(recon_points)}")
        return None
    n_points = len(worn_points)

    # Correspondence metadata (needed for both detect modes).
    corr_case_dir = os.path.join(correspondence_dir, "artificial_worn", worn_name)
    norm_path = os.path.join(corr_case_dir, "normalization.json")
    icp_path = os.path.join(corr_case_dir, "icp_transform.npy")

    # ======================================================================
    # detect_mode == "blend": correspondence-space confidence blend. Keep the
    # real worn surface, fade to the SSM recon only where worn away. No raw-scan
    # meshing -> no base patching, no graft seams. Recommended reconstruction.
    # ======================================================================
    if detect_mode == "blend":
        result_dir = os.path.join(output_dir, "reconstructions", worn_name)
        os.makedirs(result_dir, exist_ok=True)

        blended, weight, binfo = correspondence_blend(worn_points, recon_points)
        print(f"    Blend: {100 * binfo['frac_blended']:.1f}% of points blended, "
              f"{100 * binfo['frac_full_recon']:.1f}% full-recon "
              f"(deviation lo={binfo['lo_deviation']:.4f}, hi={binfo['hi_deviation']:.4f})")

        save_point_cloud(blended, os.path.join(result_dir, "reconstructed_blend.ply"))
        save_point_cloud(worn_points, os.path.join(result_dir, "worn_input.ply"))
        np.save(os.path.join(result_dir, "blend_weight.npy"), weight)

        # Colour-coded annotation: grey intact surface -> green where recon used.
        colors = np.tile(np.array(_COLOR_UNWORN, dtype=np.uint8), (len(blended), 1))
        green = np.array(_COLOR_PATCH, dtype=np.float64)
        grey = np.array(_COLOR_UNWORN, dtype=np.float64)
        colors = (grey[None] + weight[:, None] * (green - grey)[None]).astype(np.uint8)
        write_colored_ply(blended, colors,
                          os.path.join(result_dir, "blend_annotation.ply"))

        # Inverse-transform to input mm space and mesh once (clean, no base fill).
        blended_raw = None
        if os.path.exists(norm_path):
            try:
                blended_raw = inverse_correspondence_transform(blended, norm_path, icp_path)
                save_point_cloud(blended_raw, os.path.join(
                    result_dir, "reconstructed_blend_in_input_space.ply"))
            except Exception as e:
                print(f"    [WARN] inverse transform (blend) failed: {e}")

        # Mesh in NORMALIZED space -- Poisson reliably holds depth 9 there, while
        # meshing the mm-scaled, origin-offset input-space cloud is far more
        # abort-prone (falls back to a coarse depth 7-8 octree). Then transform
        # the resulting mesh vertices into input mm space.
        depth = 0
        if no_mesh:
            print("    Blend mesh: skipped (--no-mesh)")
        else:
            import trimesh
            out_mesh = os.path.join(result_dir, "blend_mesh.ply")
            tmp_mesh = os.path.join(result_dir, "_blend_norm_mesh.ply")
            depth = safe_mesh_to_file(blended, tmp_mesh)
            if depth and os.path.exists(tmp_mesh):
                bm = trimesh.load(tmp_mesh, process=False)
                os.remove(tmp_mesh)
                if blended_raw is not None and os.path.exists(norm_path):
                    try:
                        bm.vertices = inverse_correspondence_transform(
                            np.asarray(bm.vertices), norm_path, icp_path)
                    except Exception as e:
                        print(f"    [WARN] mesh inverse transform failed: {e}")
                save_mesh(bm, out_mesh)
                print(f"    Blend mesh: blend_mesh.ply (Poisson depth {depth}"
                      f"{' -- COARSE fallback' if depth < 9 else ''})")
            else:
                depth = 0
                print("    Blend mesh: FAILED")

        eval_data = {
            "source_worn": worn_name,
            "detect_mode": "blend",
            "n_points": len(blended),
            "blend": binfo,
            "mesh_poisson_depth": int(depth),
        }
        with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
            json.dump(eval_data, f, indent=2)

        return {
            "name": worn_name,
            "mode": "blend",
            "wear_fraction": float(binfo["frac_blended"]),
            "n_ring_points": 0,
            "geometric_comparison": None,
        }

    # ======================================================================
    # detect_mode == "holes": geometric open-hole detection on the REAL worn
    # surface (icp_aligned), Part 1 of the hole-patch workflow.
    # ======================================================================
    if detect_mode == "holes":
        icp_path_ply = os.path.join(corr_case_dir, "icp_aligned.ply")
        if not os.path.exists(icp_path_ply):
            print(f"    [SKIP] {worn_name}: missing icp_aligned.ply "
                  f"(needed for hole detection)")
            return None
        real_worn_full = load_point_cloud(icp_path_ply)
        print(f"    Real worn surface (icp_aligned): {len(real_worn_full)} pts")

        result_dir = os.path.join(output_dir, "reconstructions", worn_name)
        os.makedirs(result_dir, exist_ok=True)
        save_point_cloud(real_worn_full,
                         os.path.join(result_dir, "real_worn_input.ply"))

        # --- Part A: clean micro-CT extras (floaters, veils, spurs) ---
        extras_info: Dict = {"status": "skipped"}
        if clean_extras_flag:
            real_worn, keep_mask, extras_info = clean_extras(
                real_worn_full, recon_points, extra_dist_mult=extra_dist_mult,
                min_extra_cluster=min_extra_cluster)
            print(f"    Cleaned extras: removed {extras_info['n_removed']} pts "
                  f"-> {extras_info['n_kept']} kept "
                  f"(stages={extras_info['stages']})")
            np.save(os.path.join(result_dir, "extras_mask.npy"), ~keep_mask)
            save_point_cloud(real_worn,
                             os.path.join(result_dir, "cleaned_worn.ply"))
            ecolors = np.tile(np.array(_COLOR_UNWORN, dtype=np.uint8),
                              (len(real_worn_full), 1))
            ecolors[~keep_mask] = _COLOR_EXTRA
            write_colored_ply(real_worn_full, ecolors,
                              os.path.join(result_dir, "extras_annotation.ply"))
            if annotation_png:
                try:
                    render_extras_png(
                        real_worn_full, keep_mask,
                        os.path.join(result_dir, "extras_annotation.png"),
                        f"{worn_name}: micro-CT extras removal")
                except Exception as e:
                    print(f"    [WARN] extras PNG failed: {e}")
        else:
            real_worn = real_worn_full

        hole_result = detect_holes(real_worn, cell_mult=hole_cell_mult,
                                   min_hole_cells=hole_min_cells)
        n_holes = hole_result["info"].get("n_interior_holes", 0)
        print(f"    Geometric holes: {n_holes} enclosed hole(s), "
              f"outer rim={hole_result['info'].get('outer_rim_size', 0)} pts")

        np.save(os.path.join(result_dir, "hole_mask.npy"), hole_result["hole_mask"])

        # Inverse-transform the real worn surface to input mm space (for an
        # input-space colored annotation).
        real_worn_raw = None
        if os.path.exists(norm_path):
            try:
                real_worn_raw = inverse_correspondence_transform(
                    real_worn, norm_path, icp_path)
            except Exception as e:
                print(f"    [WARN] inverse transform (real worn) failed: {e}")

        save_hole_annotation(real_worn, real_worn_raw, hole_result,
                             result_dir, worn_name, annotation_png=annotation_png)
        print(f"    Annotation: hole_annotation.ply"
              f"{' + .png' if annotation_png else ''} "
              f"(red outlines = {n_holes} interior holes)")

        # --- Part 2: patch the holes with the reconstruction cap ---
        patch_info: Dict = {"status": "skipped_detect_only"}
        if not detect_only:
            if n_holes > 0:
                patched, is_patch, patch_info = patch_holes(
                    real_worn, recon_points, hole_result,
                    align=patch_align, collar_iter=patch_collar)
                print(f"    Patched: +{patch_info['n_patch_points']} recon pts "
                      f"-> {len(patched)} total"
                      f"{' (aligned)' if patch_align else ''}")

                pcolors = np.tile(np.array(_COLOR_UNWORN, dtype=np.uint8),
                                  (len(patched), 1))
                pcolors[is_patch] = _COLOR_PATCH
                write_colored_ply(patched, pcolors,
                                  os.path.join(result_dir, "patched_holes.ply"))
                save_point_cloud(patched,
                                 os.path.join(result_dir, "patched_holes_points.ply"))
                np.save(os.path.join(result_dir, "patch_mask.npy"), is_patch)

                # Patched cloud in original input mm space.
                if os.path.exists(norm_path):
                    try:
                        patched_raw = inverse_correspondence_transform(
                            patched, norm_path, icp_path)
                        save_point_cloud(patched_raw, os.path.join(
                            result_dir, "patched_holes_in_input_space.ply"))
                    except Exception as e:
                        print(f"    [WARN] inverse transform (patched) failed: {e}")

                if annotation_png:
                    try:
                        render_patch_png(
                            patched, is_patch, real_worn,
                            hole_result["interior_loops"],
                            os.path.join(result_dir, "patched_holes.png"),
                            f"{worn_name}: hole patching")
                    except Exception as e:
                        print(f"    [WARN] patch PNG failed: {e}")

                # Patched mesh: mesh-space harmonic blend (no proud scab/seam).
                # Mesh the worn cloud once and lift the fabricated bridge
                # vertices onto the recon shape; fall back to meshing the point
                # patch if the mesh-space route can't run.
                mesh_path = os.path.join(result_dir, "patched_holes_mesh.ply")
                if no_mesh:
                    print("    Patched mesh: skipped (--no-mesh)")
                    patch_info["mesh_poisson_depth"] = 0
                else:
                    try:
                        ms_mesh, ms_info = mesh_space_patch_holes(
                            real_worn, recon_points, hole_result, result_dir)
                        patch_info["mesh_patch"] = ms_info
                        if ms_mesh is not None and ms_info.get("status") in ("ok", "no_fill_vertices"):
                            save_mesh(ms_mesh, mesh_path)
                            d = ms_info.get("mesh_poisson_depth", 0)
                            print(f"    Patched mesh: patched_holes_mesh.ply "
                                  f"(mesh-space blend, Poisson depth {d}, "
                                  f"{ms_info.get('n_fill_vertices', 0)} fill verts"
                                  f"{' -- COARSE fallback' if d and d < 9 else ''})")
                        else:
                            depth = safe_mesh_to_file(patched, mesh_path)
                            patch_info["mesh_poisson_depth"] = int(depth)
                            print(f"    Patched mesh: point-patch fallback "
                                  f"(mesh-space {ms_info.get('status')}, Poisson depth {depth})")
                    except Exception as e:
                        print(f"    [WARN] mesh-space patch failed ({e}); point-patch fallback")
                        try:
                            depth = safe_mesh_to_file(patched, mesh_path)
                            patch_info["mesh_poisson_depth"] = int(depth)
                            print(f"    Patched mesh: depth {depth}")
                        except Exception as e2:
                            print(f"    [WARN] patched meshing failed: {e2}")
            else:
                patch_info = {"status": "no_holes", "n_patch_points": 0}
                print("    No enclosed holes -> nothing to patch")

        eval_data = {
            "source_worn": worn_name,
            "detect_mode": "holes",
            "worn_source": "icp_aligned",
            "n_points": len(real_worn),
            "n_interior_holes": n_holes,
            "extras": extras_info,
            "holes": hole_result["info"],
            "patch": patch_info,
        }
        with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
            json.dump(eval_data, f, indent=2)

        return {
            "name": worn_name,
            "mode": "holes",
            "n_interior_holes": n_holes,
            "n_patch_points": int(patch_info.get("n_patch_points", 0)),
            "wear_fraction": float(hole_result["hole_mask"].mean()),
            "n_ring_points": int(hole_result["info"].get("n_rim_points", 0)),
            "geometric_comparison": None,
        }

    # --- 1. Wear detection ---
    wear_mask, detect_info = detect_wear_mask(
        worn_points, recon_points,
        deviation_percentile=deviation_percentile,
        deviation_threshold=deviation_threshold)
    print(f"    Raw wear: {detect_info['n_raw_wear']}/{n_points} "
          f"({100 * detect_info['raw_wear_fraction']:.1f}%)")

    # --- 2. Mask cleanup ---
    wear_mask, clean_info = clean_wear_mask(
        worn_points, wear_mask, min_component_size=min_component_size)
    n_wear = int(wear_mask.sum())
    print(f"    Cleaned wear: {n_wear}/{n_points} ({100 * n_wear / n_points:.1f}%), "
          f"components={clean_info.get('n_components')}, "
          f"kept={clean_info.get('kept_components')}")

    # --- 3. Ring annotation ---
    ring_idx, ring_info = extract_wear_ring(worn_points, wear_mask)
    print(f"    Wear ring: {ring_info.get('n_boundary', 0)} boundary points "
          f"({ring_info.get('status')})")

    # --- Output dir + base artifacts ---
    result_dir = os.path.join(output_dir, "reconstructions", worn_name)
    os.makedirs(result_dir, exist_ok=True)
    save_point_cloud(worn_points, os.path.join(result_dir, "worn_input.ply"))
    save_point_cloud(recon_points, os.path.join(result_dir, "reconstructed.ply"))
    np.save(os.path.join(result_dir, "wear_mask.npy"), wear_mask)

    # Correspondence metadata for inverse transforms.
    parts = worn_name.split("_")
    tooth_num = parts[1] if len(parts) > 1 else "??"
    worn_name_parts = worn_name.split("_wear_")
    wear_type = worn_name_parts[1] if len(worn_name_parts) == 2 else "_".join(parts[2:])

    corr_case_dir = os.path.join(correspondence_dir, "artificial_worn", worn_name)
    norm_path = os.path.join(corr_case_dir, "normalization.json")
    icp_path = os.path.join(corr_case_dir, "icp_transform.npy")

    # Inverse-transform the worn input early so annotations can be saved in
    # the original mm space too (also reused later for graft / surgery).
    worn_raw_corr = None
    norm_scale = None
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as f:
                norm_scale = float(json.load(f)["scale"])
            worn_raw_corr = inverse_correspondence_transform(worn_points, norm_path, icp_path)
        except Exception as e:
            print(f"    [WARN] inverse transform (worn input) failed: {e}")
            worn_raw_corr = None

    # --- Annotation outputs (BEFORE patching) ---
    save_annotation(worn_points, worn_raw_corr, wear_mask, ring_idx,
                    result_dir, worn_name, annotation_png=annotation_png)
    _save_ring(worn_points, ring_idx, result_dir, worn_raw_corr)
    print(f"    Annotation: wear_annotation.ply"
          f"{' + .png' if annotation_png else ''} "
          f"(red={n_wear} wear, green={ring_info.get('n_boundary', 0)} ring)")

    # --- Detect-only: stop before patching ---
    if detect_only:
        eval_data = {
            "source_worn": worn_name,
            "source_recon_dir": recon_dir,
            "mode": "detect_only",
            "n_points": n_points,
            "n_wear_points": n_wear,
            "wear_fraction": float(n_wear / n_points),
            "detection": detect_info,
            "mask_cleanup": clean_info,
            "ring": ring_info,
        }
        with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
            json.dump(eval_data, f, indent=2)
        return {
            "name": worn_name,
            "mode": "detect_only",
            "wear_fraction": float(n_wear / n_points),
            "n_ring_points": int(ring_info.get("n_boundary", 0)),
            "geometric_comparison": None,
        }

    # --- 4. Index-swap graft ---
    patched, graft_info = graft_index_swap(
        worn_points, recon_points, wear_mask, transition_band=transition_band)
    print(f"    Graft: {n_wear} cap points + {graft_info['n_blended']} blended")
    save_point_cloud(patched, os.path.join(result_dir, "patched.ply"))

    # --- 5/6. Inverse-transform + mesh ---
    patched_raw = None
    recon_raw_corr = None
    if worn_raw_corr is not None:
        try:
            patched_raw = inverse_correspondence_transform(patched, norm_path, icp_path)
            recon_raw_corr = inverse_correspondence_transform(recon_points, norm_path, icp_path)
            save_point_cloud(patched_raw,
                             os.path.join(result_dir, "patched_in_input_space.ply"))
            print(f"    Saved patched_in_input_space.ply ({len(patched_raw)} pts)")
        except Exception as e:
            print(f"    [WARN] inverse transform (patched) failed: {e}")
            patched_raw = None

    # Smooth watertight mesh from the patched cloud in input space
    # (isolated process to survive Open3D Poisson aborts).
    if no_mesh:
        print("    Smooth mesh: skipped (--no-mesh)")
    elif patched_raw is not None:
        smooth_path = os.path.join(result_dir, "patched_smooth.ply")
        if safe_mesh_to_file(patched_raw, smooth_path):
            print(f"    Smooth mesh: saved patched_smooth.ply")
        else:
            print(f"    [WARN] smooth mesh failed (Open3D Poisson); skipped")

    # --- Optional mesh surgery ---
    surgery_info = None
    if mode == "surgery" and patched_raw is not None and worn_raw_corr is not None:
        try:
            import trimesh
            tmp_in = os.path.join(result_dir, "_tmp_input_mesh.ply")
            tmp_re = os.path.join(result_dir, "_tmp_recon_mesh.ply")

            ok_in = safe_mesh_to_file(worn_raw_corr, tmp_in)

            # Prefer the clean watertight reconstruction mesh that already
            # exists in the source recon folder (generated upstream); only
            # re-mesh as a fallback.
            src_recon_mesh = os.path.join(src_dir, "reconstructed_smooth.ply")
            ok_re = False
            if os.path.exists(src_recon_mesh):
                ok_re = True
            else:
                ok_re = safe_mesh_to_file(recon_raw_corr, tmp_re)
                src_recon_mesh = tmp_re

            if ok_in and ok_re:
                input_mesh = trimesh.load(tmp_in, process=False)
                recon_mesh = trimesh.load(src_recon_mesh, process=False)
                stitched, surgery_info = mesh_surgery_patch(
                    input_mesh, recon_mesh, worn_raw_corr, recon_raw_corr, wear_mask)
                save_mesh(stitched, os.path.join(result_dir, "patched_surgery.ply"))
                print(f"    Surgery mesh: {surgery_info['stitched_faces']} tris, "
                      f"watertight={surgery_info['watertight']}")
            else:
                print(f"    [WARN] mesh surgery skipped: base meshing failed "
                      f"(input ok={ok_in}, recon ok={ok_re})")
            for tmp in (tmp_in, tmp_re):
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:
            print(f"    [WARN] mesh surgery failed: {e}")

    # --- 7. Evaluation vs raw worn mesh ---
    geo_metrics = None
    if (artificial_wear_dir and patched_raw is not None and norm_scale is not None):
        raw_worn_path = os.path.join(
            artificial_wear_dir, f"tooth_{tooth_num}", f"wear_{wear_type}.ply")
        if os.path.exists(raw_worn_path):
            try:
                worn_raw_mesh = load_point_cloud(raw_worn_path)
                from scipy.spatial import cKDTree
                tree = cKDTree(worn_raw_mesh)
                spacing, _ = tree.query(worn_raw_mesh, k=2)
                median_spacing = float(np.median(spacing[:, 1]))
                aligned = _refine_icp_to_worn(patched_raw, worn_raw_mesh, median_spacing)
                geo_metrics = compute_geometric_comparison(
                    aligned, worn_raw_mesh, norm_scale)
                print(f"    Geo vs worn: R²={geo_metrics['variance_explained_pct']:.2f}%, "
                      f"Chamfer={geo_metrics['chamfer_mm']:.4f} mm, "
                      f"Cov2x={geo_metrics['coverage_2x_spacing'] * 100:.1f}%")
            except Exception as e:
                print(f"    [WARN] geometric comparison failed: {e}")

    eval_data = {
        "source_worn": worn_name,
        "source_recon_dir": recon_dir,
        "mode": mode,
        "n_points": n_points,
        "n_wear_points": n_wear,
        "wear_fraction": float(n_wear / n_points),
        "detection": detect_info,
        "mask_cleanup": clean_info,
        "ring": ring_info,
        "graft": graft_info,
        "surgery": surgery_info,
        "geometric_comparison": geo_metrics,
    }
    with open(os.path.join(result_dir, "evaluation.json"), "w") as f:
        json.dump(eval_data, f, indent=2)

    return {
        "name": worn_name,
        "mode": mode,
        "wear_fraction": float(n_wear / n_points),
        "n_ring_points": int(ring_info.get("n_boundary", 0)),
        "geometric_comparison": geo_metrics,
    }


# ===========================================================================
# PIPELINE RUNNER
# ===========================================================================

def run_patch_pipeline(recon_dir: str,
                       correspondence_dir: str,
                       artificial_wear_dir: Optional[str],
                       output_dir: str,
                       worn_teeth_list: Optional[List[str]],
                       deviation_percentile: float,
                       deviation_threshold: Optional[float],
                       min_component_size: int,
                       transition_band: float,
                       mode: str,
                       detect_only: bool = False,
                       annotation_png: bool = True,
                       detect_mode: str = "deviation",
                       hole_cell_mult: float = 2.5,
                       hole_min_cells: int = 6,
                       clean_extras_flag: bool = False,
                       extra_dist_mult: float = 6.0,
                       min_extra_cluster: int = 150,
                       patch_align: bool = True,
                       patch_collar: int = 4,
                       no_mesh: bool = False) -> None:
    print("=" * 60)
    print("Wear-Margin Patch Reconstruction")
    print("=" * 60)
    print(f"  Recon dir          : {recon_dir}")
    print(f"  Correspondence dir : {correspondence_dir}")
    print(f"  Artificial wear    : {artificial_wear_dir}")
    print(f"  Output dir         : {output_dir}")
    print(f"  Detect mode        : {detect_mode}")
    if detect_mode == "holes":
        print(f"  Clean extras       : {clean_extras_flag} "
              f"(envelope x{extra_dist_mult}, min cluster={min_extra_cluster})")
        print(f"  Patch align        : {patch_align} (collar={patch_collar})")
    print(f"  Mode               : {'detect-only' if detect_only else mode}")
    print(f"  Deviation pct      : {deviation_percentile}")
    print(f"  Deviation thresh   : {deviation_threshold}")
    print(f"  Min component size : {min_component_size}")
    print(f"  Transition band    : {transition_band} x median spacing")
    print(f"  Annotation PNG     : {annotation_png}")
    print()

    recon_reco_dir = os.path.join(recon_dir, "reconstructions")
    if not os.path.isdir(recon_reco_dir):
        sys.exit(f"ERROR: {recon_reco_dir} not found. Run Stage 2 first.")

    if worn_teeth_list:
        worn_dirs = worn_teeth_list
    else:
        worn_dirs = sorted(
            d for d in os.listdir(recon_reco_dir)
            if os.path.isdir(os.path.join(recon_reco_dir, d))
        )

    print(f"Found {len(worn_dirs)} worn teeth to patch")
    os.makedirs(os.path.join(output_dir, "reconstructions"), exist_ok=True)

    results: List[Dict] = []
    for worn_name in worn_dirs:
        try:
            res = process_tooth(
                worn_name=worn_name,
                recon_dir=recon_dir,
                correspondence_dir=correspondence_dir,
                artificial_wear_dir=artificial_wear_dir,
                output_dir=output_dir,
                deviation_percentile=deviation_percentile,
                deviation_threshold=deviation_threshold,
                min_component_size=min_component_size,
                transition_band=transition_band,
                mode=mode,
                detect_only=detect_only,
                annotation_png=annotation_png,
                detect_mode=detect_mode,
                hole_cell_mult=hole_cell_mult,
                hole_min_cells=hole_min_cells,
                clean_extras_flag=clean_extras_flag,
                extra_dist_mult=extra_dist_mult,
                min_extra_cluster=min_extra_cluster,
                patch_align=patch_align,
                patch_collar=patch_collar,
                no_mesh=no_mesh)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"    [ERROR] {worn_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Top-level summary ---
    wear_fracs = [r["wear_fraction"] for r in results]
    chamfers = [r["geometric_comparison"]["chamfer_mm"] for r in results
                if r.get("geometric_comparison")]
    summary = {
        "recon_dir": recon_dir,
        "correspondence_dir": correspondence_dir,
        "mode": "detect_only" if detect_only else mode,
        "n_patched": len(results),
        "deviation_percentile": deviation_percentile,
        "deviation_threshold": deviation_threshold,
        "min_component_size": min_component_size,
        "transition_band": transition_band,
        "wear_fraction_mean": float(np.mean(wear_fracs)) if wear_fracs else None,
        "wear_fraction_min": float(np.min(wear_fracs)) if wear_fracs else None,
        "wear_fraction_max": float(np.max(wear_fracs)) if wear_fracs else None,
        "chamfer_mm_mean": float(np.mean(chamfers)) if chamfers else None,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "patch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Patch Reconstruction Summary")
    print(f"{'=' * 60}")
    label = "Annotated teeth" if detect_only else "Patched teeth"
    print(f"  {label}: {len(results)}")
    if wear_fracs:
        print(f"  Wear fraction: mean={100 * np.mean(wear_fracs):.1f}%, "
              f"range=[{100 * np.min(wear_fracs):.1f}%, "
              f"{100 * np.max(wear_fracs):.1f}%]")
    if chamfers:
        print(f"  Chamfer (mm) mean: {np.mean(chamfers):.4f}")
    print(f"  Output: {output_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wear-margin patch reconstruction: detect wear, ring "
                    "annotation, graft the clean reconstruction cap onto the "
                    "worn input.")
    parser.add_argument("--recon-dir", type=str,
                        default="output/recon_neighborhood_v4",
                        help="Stage 2 output dir containing reconstructions/<tooth>/ "
                             "with worn_input.ply + reconstructed.ply "
                             "(default: output/recon_neighborhood_v4)")
    parser.add_argument("--correspondence-dir", type=str,
                        default="output/correspondence_all_100k",
                        help="Correspondence dir with artificial_worn/<tooth>/ "
                             "normalization.json + icp_transform.npy")
    parser.add_argument("--artificial-wear", type=str, default=None,
                        help="Directory with raw worn meshes (tooth_XX/wear_*.ply) "
                             "for geometric evaluation (optional)")
    parser.add_argument("--output", type=str, default="output/recon_patch",
                        help="Output directory (default: output/recon_patch)")
    parser.add_argument("--worn-teeth", nargs="+", default=None,
                        help="Specific worn-tooth directory names to patch "
                             "(default: all found under recon-dir)")
    parser.add_argument("--deviation-percentile", type=float, default=75.0,
                        help="Percentile of the outward-deviation score used as "
                             "the wear threshold (default: 75)")
    parser.add_argument("--deviation-threshold", type=float, default=None,
                        help="Absolute outward-deviation floor in normalized units "
                             "(optional, combined with the percentile)")
    parser.add_argument("--min-component-size", type=int, default=200,
                        help="Drop wear components smaller than this (default: 200)")
    parser.add_argument("--transition-band", type=float, default=3.0,
                        help="Seam-blend band width as multiple of median spacing "
                             "(default: 3.0; set 0 to disable)")
    parser.add_argument("--mode", choices=["swap", "surgery"], default="swap",
                        help="Patch mode: 'swap' (index-space graft + re-mesh) or "
                             "'surgery' (mesh cut + stitch). Default: swap")
    parser.add_argument("--detect-mode", choices=["deviation", "holes", "blend"],
                        default="blend",
                        help="Reconstruction method. 'blend' (default, recommended): "
                             "correspondence-space confidence blend of worn_input "
                             "and reconstructed -- keeps the real worn surface, fades "
                             "to the SSM recon only where worn away; no raw-scan "
                             "meshing so the base is never patched and there are no "
                             "graft seams. 'holes': geometric open-hole rim tracing "
                             "+ mesh-space fill on the real scan. 'deviation': "
                             "index-space cap swap.")
    parser.add_argument("--hole-cell-mult", type=float, default=2.5,
                        help="Occupancy-grid cell size as a multiple of the cloud's "
                             "median point spacing (default: 2.5). Larger = coarser "
                             "grid, ignores smaller holes.")
    parser.add_argument("--hole-min-cells", type=int, default=6,
                        help="Minimum enclosed-region area (in grid cells) to count "
                             "as a hole; drops tiny sampling gaps. Default: 6")
    parser.add_argument("--clean-extras", action="store_true",
                        help="[holes mode] Enable micro-CT extras removal (floaters, "
                             "veils, spurs) before hole detection. Off by default: "
                             "the pipeline goes straight from hole detection to "
                             "patching.")
    parser.add_argument("--extra-dist-mult", type=float, default=6.0,
                        help="[holes mode] Extras envelope: flag worn points that "
                             "protrude beyond this x median spacing past the "
                             "reconstruction as extra CANDIDATES. 0 disables. "
                             "Default: 6")
    parser.add_argument("--min-extra-cluster", type=int, default=150,
                        help="[holes mode] Only remove an extra candidate if it "
                             "belongs to a connected cluster of at least this many "
                             "points, so cleaning removes continuous blobs (spurs/"
                             "veils/flaps) and never scattered surface noise. "
                             "Default: 150")
    parser.add_argument("--no-patch-align", action="store_true",
                        help="[holes mode] Skip per-hole local ICP alignment + "
                             "feather; graft reconstruction points as-is.")
    parser.add_argument("--patch-collar", type=int, default=4,
                        help="[holes mode] Rim collar width (grid cells) used as the "
                             "ICP target / feather anchor around each hole. Default: 4")
    parser.add_argument("--detect-only", action="store_true",
                        help="Stop after wear detection + ring annotation; do not "
                             "graft, mesh or run surgery. Use to inspect detection.")
    parser.add_argument("--no-annotation-png", action="store_true",
                        help="Skip the multi-view annotation PNG (saves time).")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Skip Poisson meshing entirely (no *_mesh.ply / "
                             "*_smooth.ply). The point-cloud outputs used by all "
                             "evaluation scripts are saved regardless, so this is "
                             "safe for batch runs whose only purpose is numeric "
                             "evaluation -- avoids Poisson's slowness/aborts.")
    args = parser.parse_args()

    run_patch_pipeline(
        recon_dir=args.recon_dir,
        correspondence_dir=args.correspondence_dir,
        artificial_wear_dir=args.artificial_wear,
        output_dir=args.output,
        worn_teeth_list=args.worn_teeth,
        deviation_percentile=args.deviation_percentile,
        deviation_threshold=args.deviation_threshold,
        min_component_size=args.min_component_size,
        transition_band=args.transition_band,
        mode=args.mode,
        detect_only=args.detect_only,
        annotation_png=not args.no_annotation_png,
        detect_mode=args.detect_mode,
        hole_cell_mult=args.hole_cell_mult,
        hole_min_cells=args.hole_min_cells,
        clean_extras_flag=args.clean_extras,
        extra_dist_mult=args.extra_dist_mult,
        min_extra_cluster=args.min_extra_cluster,
        patch_align=not args.no_patch_align,
        patch_collar=args.patch_collar,
        no_mesh=args.no_mesh)


if __name__ == "__main__":
    main()
