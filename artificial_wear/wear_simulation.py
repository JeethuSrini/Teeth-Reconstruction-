#!/usr/bin/env python3
"""
EDJ Artificial Wear Simulation Pipeline

This script generates realistic, artificially worn EDJ tooth meshes from unworn
3D EDJ surfaces. The artificial wear preferentially affects sharp, high-curvature
(cusp) regions, reflecting how wear is distributed in real tooth datasets.

Key Features:
- Cusp detection using curvature, elevation, and normal variation
- Localized wear simulation (spherical, planar, height-clipping)
- Multiple wear severity levels (mild, moderate)
- Fully reproducible with random seeds
- Outputs: worn meshes, vertex masks, metadata JSON

Author: Generated for Teeth-Reconstruction project
"""

import json
import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CuspRegion:
    """Represents a detected cusp region on the tooth surface."""
    index: int                          # Cusp identifier
    centroid: np.ndarray               # 3D centroid of the cusp region
    apex: np.ndarray                   # Highest point (peak) of the cusp
    vertex_indices: np.ndarray         # Indices of vertices in this cusp
    mean_likelihood: float             # Average cusp likelihood score
    radius: float                      # Approximate radius of the cusp region


@dataclass
class WearResult:
    """Contains the result of a wear simulation."""
    name: str                          # e.g., "wear_mild_cusp1"
    mesh: trimesh.Trimesh              # The worn mesh
    removed_mask: np.ndarray           # Boolean mask: True = removed vertex
    wear_type: str                     # "spherical", "planar", "height_clip", "mixed"
    wear_depth_mm: float               # Actual depth of wear applied
    cusps_affected: List[int]          # Indices of cusps that were worn
    occlusal_direction: np.ndarray     # Direction used for wear
    random_seed: int                   # Seed for reproducibility


@dataclass
class WearConfig:
    """Configuration parameters for wear simulation."""
    # Wear depth ranges (in mm)
    mild_depth_range: Tuple[float, float] = (0.3, 0.8)
    moderate_depth_range: Tuple[float, float] = (1.0, 1.5)
    
    # Number of cusps to affect
    mild_cusps: Tuple[int, int] = (1, 2)
    moderate_cusps: Tuple[int, int] = (2, 4)
    
    # Cusp detection parameters
    curvature_weight: float = 0.4
    elevation_weight: float = 0.3
    normal_variation_weight: float = 0.3
    n_cusps_to_detect: int = 4
    
    # Realism parameters
    surface_noise_sigma: float = 0.02  # mm
    max_mild_removal_fraction: float = 0.30
    plane_tilt_range: Tuple[float, float] = (5.0, 15.0)  # degrees
    
    # Neighborhood parameters
    cusp_neighborhood_percentile: float = 95.0
    falloff_sigma_factor: float = 0.5


# =============================================================================
# MESH LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess(filepath: str, repair: bool = True) -> trimesh.Trimesh:
    """
    Load a PLY mesh and optionally repair it for watertightness.
    
    Args:
        filepath: Path to the PLY file
        repair: Whether to attempt watertight repair
        
    Returns:
        Loaded and optionally repaired trimesh object
    """
    mesh = trimesh.load(filepath, force='mesh')
    
    if repair:
        # Fill holes to improve watertightness (non-aggressive)
        # This requires networkx - gracefully skip if not available
        try:
            trimesh.repair.fill_holes(mesh)
        except (ImportError, ModuleNotFoundError):
            pass  # Skip hole filling if networkx not installed
        # Fix winding and normals
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        # Remove degenerate faces (API changed in newer trimesh versions)
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        else:
            # Use nondegenerate_faces mask to filter
            mask = mesh.nondegenerate_faces()
            if not mask.all():
                mesh.update_faces(mask)
        # Remove duplicate faces
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
        else:
            # Use unique_faces for newer API
            mesh.update_faces(mesh.unique_faces())
    
    # Ensure vertex normals are computed
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.vertex_normals
    
    return mesh


# =============================================================================
# GEOMETRIC FEATURE COMPUTATION
# =============================================================================

def compute_mean_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute mean curvature at each vertex using discrete curvature measure.
    
    The mean curvature is estimated using the angle defect method and
    cotangent weights, providing a robust measure of local surface bending.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Array of mean curvature values per vertex (higher = sharper)
    """
    # Use trimesh's discrete mean curvature measure
    # This computes the integral of mean curvature in a small ball around each vertex
    try:
        curvature = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=None
        )
    except Exception:
        # Fallback: estimate curvature from vertex normals and positions
        curvature = _estimate_curvature_fallback(mesh)
    
    return np.abs(curvature)


def _estimate_curvature_fallback(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Fallback curvature estimation using Laplacian smoothing difference.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Estimated curvature per vertex
    """
    # Build vertex adjacency
    edges = mesh.edges_unique
    adj = {i: [] for i in range(len(mesh.vertices))}
    for e in edges:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    
    # Compute Laplacian (difference from neighborhood mean)
    laplacian = np.zeros(len(mesh.vertices))
    for i, neighbors in adj.items():
        if len(neighbors) > 0:
            neighbor_mean = mesh.vertices[neighbors].mean(axis=0)
            laplacian[i] = np.linalg.norm(mesh.vertices[i] - neighbor_mean)
    
    return laplacian


def compute_elevation_score(mesh: trimesh.Trimesh, 
                           occlusal_direction: np.ndarray = None) -> np.ndarray:
    """
    Compute elevation score based on vertex height along occlusal direction.
    
    Higher vertices (cusps) get higher scores. The occlusal direction is
    typically +Z but can be auto-detected from the mesh principal axis.
    
    Args:
        mesh: Input trimesh object
        occlusal_direction: Direction of occlusal surface (default: +Z)
        
    Returns:
        Elevation score per vertex, normalized to [0, 1]
    """
    if occlusal_direction is None:
        # Use +Z as default occlusal direction
        occlusal_direction = np.array([0.0, 0.0, 1.0])
    
    occlusal_direction = occlusal_direction / np.linalg.norm(occlusal_direction)
    
    # Project vertices onto occlusal direction
    heights = mesh.vertices @ occlusal_direction
    
    # Normalize to [0, 1] using percentile ranking
    min_h, max_h = heights.min(), heights.max()
    if max_h - min_h > 1e-10:
        elevation_score = (heights - min_h) / (max_h - min_h)
    else:
        elevation_score = np.zeros(len(heights))
    
    return elevation_score


def compute_normal_variation(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute local normal variation as a sharpness proxy.
    
    High normal variation indicates sharp ridges and cusp tips where
    adjacent face normals diverge significantly.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Normal variation score per vertex, normalized to [0, 1]
    """
    # Get face normals and vertex-face adjacency
    face_normals = mesh.face_normals
    
    # For each vertex, compute variance of adjacent face normals
    variation = np.zeros(len(mesh.vertices))
    
    # Build vertex-to-face mapping
    vertex_faces = [[] for _ in range(len(mesh.vertices))]
    for fi, face in enumerate(mesh.faces):
        for vi in face:
            vertex_faces[vi].append(fi)
    
    for vi in range(len(mesh.vertices)):
        adj_faces = vertex_faces[vi]
        if len(adj_faces) < 2:
            continue
        
        adj_normals = face_normals[adj_faces]
        
        # Compute angular deviation from mean normal
        mean_normal = adj_normals.mean(axis=0)
        mean_normal_norm = np.linalg.norm(mean_normal)
        if mean_normal_norm > 1e-10:
            mean_normal /= mean_normal_norm
            # Compute angles between each normal and mean
            dots = np.clip(adj_normals @ mean_normal, -1, 1)
            angles = np.arccos(dots)
            variation[vi] = angles.std()
    
    # Normalize to [0, 1]
    if variation.max() > 1e-10:
        variation = variation / variation.max()
    
    return variation


def compute_cusp_likelihood(mesh: trimesh.Trimesh, 
                           config: WearConfig = None) -> np.ndarray:
    """
    Compute combined cusp likelihood score for each vertex.
    
    Combines three geometric criteria:
    1. Mean curvature (weight: 0.4) - Sharp regions have high curvature
    2. Elevation (weight: 0.3) - Cusps are at top of occlusal surface
    3. Normal variation (weight: 0.3) - Cusp tips have divergent normals
    
    Args:
        mesh: Input trimesh object
        config: Wear configuration parameters
        
    Returns:
        Cusp likelihood score per vertex, in [0, 1]
    """
    if config is None:
        config = WearConfig()
    
    # Compute individual scores
    curvature = compute_mean_curvature(mesh)
    elevation = compute_elevation_score(mesh)
    normal_var = compute_normal_variation(mesh)
    
    # Normalize curvature to [0, 1]
    if curvature.max() > 1e-10:
        curvature_norm = curvature / curvature.max()
    else:
        curvature_norm = np.zeros_like(curvature)
    
    # Weighted combination
    likelihood = (
        config.curvature_weight * curvature_norm +
        config.elevation_weight * elevation +
        config.normal_variation_weight * normal_var
    )
    
    # Normalize final score
    if likelihood.max() > 1e-10:
        likelihood = likelihood / likelihood.max()
    
    return likelihood


# =============================================================================
# CUSP REGION DETECTION
# =============================================================================

def identify_cusp_regions(mesh: trimesh.Trimesh, 
                         likelihood: np.ndarray,
                         n_cusps: int = 4,
                         min_separation: float = None) -> List[CuspRegion]:
    """
    Identify distinct cusp regions from the cusp likelihood scores.
    
    Uses a peak-finding approach with spatial clustering to identify
    separate cusp regions. Each region is defined by its apex (highest
    likelihood vertex) and surrounding high-likelihood vertices.
    
    Args:
        mesh: Input trimesh object
        likelihood: Cusp likelihood score per vertex
        n_cusps: Number of cusps to detect (typically 4 for molars)
        min_separation: Minimum distance between cusp centers (auto-computed if None)
        
    Returns:
        List of CuspRegion objects representing detected cusps
    """
    vertices = mesh.vertices
    
    # Auto-compute minimum separation based on mesh extent
    if min_separation is None:
        bbox_extent = mesh.bounding_box.extents
        min_separation = min(bbox_extent[:2]) * 0.2  # 20% of smaller XY dimension
    
    # Find high-likelihood vertices (top 20%)
    threshold = np.percentile(likelihood, 80)
    high_likelihood_mask = likelihood >= threshold
    high_likelihood_indices = np.where(high_likelihood_mask)[0]
    
    if len(high_likelihood_indices) == 0:
        return []
    
    # Find local maxima using iterative peak finding
    cusps = []
    remaining_indices = set(high_likelihood_indices)
    
    for _ in range(n_cusps):
        if not remaining_indices:
            break
        
        # Find highest likelihood vertex among remaining
        remaining_list = list(remaining_indices)
        remaining_likelihoods = likelihood[remaining_list]
        peak_idx = remaining_list[np.argmax(remaining_likelihoods)]
        peak_pos = vertices[peak_idx]
        
        # Find all vertices within neighborhood of this peak
        tree = cKDTree(vertices[list(remaining_indices)])
        remaining_arr = np.array(list(remaining_indices))
        
        # Neighborhood radius based on likelihood falloff
        neighborhood_radius = min_separation * 1.5
        neighbor_local_indices = tree.query_ball_point(peak_pos, neighborhood_radius)
        neighbor_indices = remaining_arr[neighbor_local_indices]
        
        if len(neighbor_indices) == 0:
            remaining_indices.discard(peak_idx)
            continue
        
        # Compute cusp properties
        cusp_vertices = vertices[neighbor_indices]
        centroid = cusp_vertices.mean(axis=0)
        
        # Find apex (highest point along Z)
        z_values = cusp_vertices[:, 2]
        apex_local_idx = np.argmax(z_values)
        apex = cusp_vertices[apex_local_idx]
        
        # Estimate cusp radius
        distances = np.linalg.norm(cusp_vertices - centroid, axis=1)
        radius = np.percentile(distances, 90)
        
        cusp = CuspRegion(
            index=len(cusps),
            centroid=centroid,
            apex=apex,
            vertex_indices=neighbor_indices,
            mean_likelihood=likelihood[neighbor_indices].mean(),
            radius=max(radius, min_separation * 0.3)
        )
        cusps.append(cusp)
        
        # Remove these vertices from consideration
        remaining_indices -= set(neighbor_indices)
    
    # Sort cusps by likelihood (highest first)
    cusps.sort(key=lambda c: c.mean_likelihood, reverse=True)
    for i, cusp in enumerate(cusps):
        cusp.index = i
    
    return cusps


# =============================================================================
# WEAR SIMULATION METHODS
# =============================================================================

def apply_spherical_wear(mesh: trimesh.Trimesh,
                        cusp: CuspRegion,
                        depth_mm: float,
                        rng: np.random.Generator,
                        noise_sigma: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply spherical/ellipsoidal wear centered on a cusp.
    
    Places a sphere at the cusp apex and marks vertices within the sphere
    for removal. The sphere is slightly ellipsoidal (flattened along Z)
    to create more realistic wear patterns.
    
    Args:
        mesh: Input trimesh object
        cusp: Target cusp region
        depth_mm: Wear depth in millimeters
        rng: Random number generator for reproducibility
        noise_sigma: Surface noise standard deviation
        
    Returns:
        Tuple of (modified vertices, removal mask)
    """
    vertices = mesh.vertices.copy()
    
    # Sphere center slightly below apex
    center = cusp.apex.copy()
    center[2] -= depth_mm * 0.3  # Offset down
    
    # Ellipsoid radii (flattened along Z for realistic wear)
    rx = depth_mm * 1.2 + rng.uniform(-0.1, 0.1) * depth_mm
    ry = depth_mm * 1.2 + rng.uniform(-0.1, 0.1) * depth_mm
    rz = depth_mm * 0.8  # Flatter in Z direction
    
    # Compute ellipsoidal distance
    diff = vertices - center
    ellipsoid_dist = np.sqrt(
        (diff[:, 0] / rx) ** 2 +
        (diff[:, 1] / ry) ** 2 +
        (diff[:, 2] / rz) ** 2
    )
    
    # Mark vertices inside ellipsoid for removal
    removal_mask = ellipsoid_dist < 1.0
    
    # For vertices near the boundary, apply noise for irregularity
    boundary_mask = (ellipsoid_dist >= 0.8) & (ellipsoid_dist < 1.2)
    if np.any(boundary_mask):
        noise = rng.normal(0, noise_sigma, size=np.sum(boundary_mask))
        # Vertices with noise pushing them "in" get removed
        boundary_indices = np.where(boundary_mask)[0]
        for i, idx in enumerate(boundary_indices):
            if ellipsoid_dist[idx] + noise[i] * 0.5 < 1.0:
                removal_mask[idx] = True
    
    return vertices, removal_mask


def apply_planar_cut(mesh: trimesh.Trimesh,
                    cusp: CuspRegion,
                    depth_mm: float,
                    rng: np.random.Generator,
                    tilt_range: Tuple[float, float] = (5.0, 15.0),
                    neighborhood_radius: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a localized tilted planar cut centered on a cusp.
    
    Creates a plane at the cusp apex with slight random tilt for
    irregularity. Only affects vertices within the cusp neighborhood.
    
    Args:
        mesh: Input trimesh object
        cusp: Target cusp region
        depth_mm: Wear depth in millimeters
        rng: Random number generator
        tilt_range: Range of tilt angles in degrees
        neighborhood_radius: Radius of effect (default: cusp radius * 2)
        
    Returns:
        Tuple of (modified vertices, removal mask)
    """
    vertices = mesh.vertices.copy()
    
    if neighborhood_radius is None:
        neighborhood_radius = cusp.radius * 2.0
    
    # Plane center at apex minus depth
    plane_center = cusp.apex.copy()
    plane_center[2] -= depth_mm
    
    # Random tilt angle
    tilt_deg = rng.uniform(tilt_range[0], tilt_range[1])
    tilt_rad = np.radians(tilt_deg)
    
    # Random tilt direction in XY plane
    tilt_direction = rng.uniform(0, 2 * np.pi)
    
    # Compute tilted plane normal
    # Start with Z-up, then rotate
    plane_normal = np.array([
        np.sin(tilt_rad) * np.cos(tilt_direction),
        np.sin(tilt_rad) * np.sin(tilt_direction),
        np.cos(tilt_rad)
    ])
    
    # Compute signed distance to plane
    diff = vertices - plane_center
    signed_dist = diff @ plane_normal
    
    # Distance from cusp center in XY
    xy_diff = vertices[:, :2] - cusp.centroid[:2]
    xy_dist = np.linalg.norm(xy_diff, axis=1)
    
    # Gaussian falloff for localization
    falloff = np.exp(-(xy_dist ** 2) / (2 * neighborhood_radius ** 2))
    
    # Mark vertices above plane AND within neighborhood
    removal_mask = (signed_dist > 0) & (falloff > 0.1)
    
    return vertices, removal_mask


def apply_height_clipping(mesh: trimesh.Trimesh,
                         cusp: CuspRegion,
                         depth_mm: float,
                         rng: np.random.Generator,
                         falloff_sigma_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply height-based clipping restricted to cusp neighborhood.
    
    Clips vertices above a certain height threshold, but only within
    the cusp's spatial neighborhood using Gaussian falloff.
    
    Args:
        mesh: Input trimesh object
        cusp: Target cusp region
        depth_mm: Wear depth in millimeters
        rng: Random number generator
        falloff_sigma_factor: Controls neighborhood size
        
    Returns:
        Tuple of (modified vertices, removal mask)
    """
    vertices = mesh.vertices.copy()
    
    # Height threshold
    clip_height = cusp.apex[2] - depth_mm
    
    # Add slight randomness to threshold
    clip_height += rng.uniform(-0.05, 0.05) * depth_mm
    
    # Distance from cusp center in 3D
    diff = vertices - cusp.centroid
    dist_3d = np.linalg.norm(diff, axis=1)
    
    # Gaussian falloff
    sigma = cusp.radius * (1.0 + falloff_sigma_factor)
    falloff = np.exp(-(dist_3d ** 2) / (2 * sigma ** 2))
    
    # Effective clip height varies with falloff
    # Vertices far from cusp have higher effective threshold (less likely to be clipped)
    effective_threshold = clip_height + (1 - falloff) * depth_mm * 2
    
    # Mark vertices above effective threshold
    removal_mask = vertices[:, 2] > effective_threshold
    
    return vertices, removal_mask


def apply_mixed_wear(mesh: trimesh.Trimesh,
                    cusps: List[CuspRegion],
                    depth_mm: float,
                    rng: np.random.Generator,
                    config: WearConfig = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Apply a combination of wear methods to multiple cusps.
    
    Randomly selects wear method for each cusp and combines results.
    
    Args:
        mesh: Input trimesh object
        cusps: List of cusp regions to wear
        depth_mm: Base wear depth in millimeters
        rng: Random number generator
        config: Wear configuration
        
    Returns:
        Tuple of (modified vertices, combined removal mask, wear type string)
    """
    if config is None:
        config = WearConfig()
    
    vertices = mesh.vertices.copy()
    combined_mask = np.zeros(len(vertices), dtype=bool)
    wear_types_used = []
    
    methods = [
        ('spherical', apply_spherical_wear),
        ('planar', apply_planar_cut),
        ('height_clip', apply_height_clipping)
    ]
    
    for cusp in cusps:
        # Random depth variation per cusp
        cusp_depth = depth_mm * rng.uniform(0.7, 1.3)
        
        # Random method selection
        method_name, method_func = methods[rng.integers(0, len(methods))]
        
        if method_name == 'spherical':
            _, mask = apply_spherical_wear(mesh, cusp, cusp_depth, rng, config.surface_noise_sigma)
        elif method_name == 'planar':
            _, mask = apply_planar_cut(mesh, cusp, cusp_depth, rng, config.plane_tilt_range)
        else:
            _, mask = apply_height_clipping(mesh, cusp, cusp_depth, rng, config.falloff_sigma_factor)
        
        combined_mask |= mask
        wear_types_used.append(method_name)
    
    # Determine overall wear type
    unique_types = list(set(wear_types_used))
    if len(unique_types) == 1:
        wear_type = unique_types[0]
    else:
        wear_type = "mixed"
    
    return vertices, combined_mask, wear_type


# =============================================================================
# MESH RECONSTRUCTION
# =============================================================================

def create_worn_mesh(original_mesh: trimesh.Trimesh,
                    removal_mask: np.ndarray,
                    smooth_boundary: bool = True) -> trimesh.Trimesh:
    """
    Create a new mesh with specified vertices removed.
    
    Removes faces that contain any removed vertex, then optionally
    smooths the boundary for better appearance.
    
    Args:
        original_mesh: Original trimesh object
        removal_mask: Boolean mask (True = remove vertex)
        smooth_boundary: Whether to smooth wear boundaries
        
    Returns:
        New trimesh object with wear applied
    """
    # Identify faces to keep (faces with no removed vertices)
    faces = original_mesh.faces
    face_removed = removal_mask[faces].any(axis=1)
    keep_faces = ~face_removed
    
    if not np.any(keep_faces):
        # All faces removed - return empty mesh
        return trimesh.Trimesh()
    
    # Create new mesh with remaining faces
    new_faces = faces[keep_faces]
    
    # Remap vertex indices
    keep_vertices = np.zeros(len(original_mesh.vertices), dtype=bool)
    keep_vertices[new_faces.flatten()] = True
    
    old_to_new = np.full(len(original_mesh.vertices), -1, dtype=int)
    old_to_new[keep_vertices] = np.arange(np.sum(keep_vertices))
    
    new_vertices = original_mesh.vertices[keep_vertices]
    remapped_faces = old_to_new[new_faces]
    
    worn_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
    
    # Smooth boundary if requested
    if smooth_boundary:
        worn_mesh = _smooth_boundary_vertices(worn_mesh)
    
    # Fix mesh issues
    trimesh.repair.fix_winding(worn_mesh)
    trimesh.repair.fix_normals(worn_mesh)
    
    return worn_mesh


def _smooth_boundary_vertices(mesh: trimesh.Trimesh, 
                             iterations: int = 2,
                             factor: float = 0.3) -> trimesh.Trimesh:
    """
    Smooth vertices near the mesh boundary (wear edge).
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
        factor: Smoothing strength (0-1)
        
    Returns:
        Mesh with smoothed boundary
    """
    if len(mesh.vertices) == 0:
        return mesh
    
    # Find boundary edges (edges with only one adjacent face)
    edges = mesh.edges_unique
    edge_faces = mesh.edges_unique_inverse
    
    # Count faces per edge
    edge_face_count = np.bincount(edge_faces, minlength=len(edges))
    boundary_edges = edges[edge_face_count == 1]
    
    if len(boundary_edges) == 0:
        return mesh
    
    # Get boundary vertices
    boundary_vertices = np.unique(boundary_edges.flatten())
    
    # Build adjacency for smoothing
    adj = {i: [] for i in range(len(mesh.vertices))}
    for e in mesh.edges_unique:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    
    vertices = mesh.vertices.copy()
    
    for _ in range(iterations):
        new_vertices = vertices.copy()
        for vi in boundary_vertices:
            neighbors = adj[vi]
            if len(neighbors) > 0:
                neighbor_mean = vertices[neighbors].mean(axis=0)
                new_vertices[vi] = (1 - factor) * vertices[vi] + factor * neighbor_mean
        vertices = new_vertices
    
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# =============================================================================
# ADVANCED WEAR METHODS
# =============================================================================

def apply_faceted_wear(mesh: trimesh.Trimesh,
                      cusp: CuspRegion,
                      depth_mm: float,
                      rng: np.random.Generator,
                      n_facets: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple small planar cuts creating faceted wear surface.
    
    Creates more realistic wear with multiple intersecting planes,
    mimicking the faceted appearance of real tooth wear.
    
    Args:
        mesh: Input trimesh object
        cusp: Target cusp region
        depth_mm: Base wear depth in millimeters
        rng: Random number generator
        n_facets: Number of facets to create
        
    Returns:
        Tuple of (modified vertices, removal mask)
    """
    vertices = mesh.vertices.copy()
    combined_mask = np.zeros(len(vertices), dtype=bool)
    
    for i in range(n_facets):
        # Vary depth for each facet
        facet_depth = depth_mm * rng.uniform(0.6, 1.4)
        
        # Offset center from apex
        offset = rng.uniform(-cusp.radius * 0.5, cusp.radius * 0.5, size=2)
        facet_center = cusp.apex.copy()
        facet_center[:2] += offset
        facet_center[2] -= facet_depth
        
        # Random tilt for each facet
        tilt_deg = rng.uniform(5, 25)
        tilt_rad = np.radians(tilt_deg)
        tilt_direction = rng.uniform(0, 2 * np.pi)
        
        plane_normal = np.array([
            np.sin(tilt_rad) * np.cos(tilt_direction),
            np.sin(tilt_rad) * np.sin(tilt_direction),
            np.cos(tilt_rad)
        ])
        
        # Signed distance to plane
        diff = vertices - facet_center
        signed_dist = diff @ plane_normal
        
        # Localization
        xy_diff = vertices[:, :2] - cusp.centroid[:2]
        xy_dist = np.linalg.norm(xy_diff, axis=1)
        neighborhood_radius = cusp.radius * rng.uniform(1.0, 2.0)
        falloff = np.exp(-(xy_dist ** 2) / (2 * neighborhood_radius ** 2))
        
        # Mark for removal
        facet_mask = (signed_dist > 0) & (falloff > 0.15)
        combined_mask |= facet_mask
    
    return vertices, combined_mask


def apply_asymmetric_wear(mesh: trimesh.Trimesh,
                         cusps: List[CuspRegion],
                         base_depth_mm: float,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Apply highly asymmetric wear across multiple cusps.
    
    Different cusps receive very different wear depths and types,
    creating realistic asymmetric wear patterns.
    
    Args:
        mesh: Input trimesh object
        cusps: List of cusp regions
        base_depth_mm: Base wear depth
        rng: Random number generator
        
    Returns:
        Tuple of (vertices, combined mask, affected cusp indices)
    """
    vertices = mesh.vertices.copy()
    combined_mask = np.zeros(len(vertices), dtype=bool)
    affected = []
    
    # Randomly select which cusps to wear (not all)
    n_to_wear = rng.integers(1, min(len(cusps), 3) + 1)
    cusp_indices = rng.choice(len(cusps), size=n_to_wear, replace=False)
    
    for idx in cusp_indices:
        cusp = cusps[idx]
        affected.append(int(idx))
        
        # Highly variable depth per cusp
        depth_multiplier = rng.uniform(0.3, 2.0)
        cusp_depth = base_depth_mm * depth_multiplier
        
        # Random method per cusp
        method = rng.integers(0, 3)
        
        if method == 0:
            _, mask = apply_spherical_wear(mesh, cusp, cusp_depth, rng)
        elif method == 1:
            _, mask = apply_planar_cut(mesh, cusp, cusp_depth, rng, (10, 30))
        else:
            _, mask = apply_faceted_wear(mesh, cusp, cusp_depth, rng, rng.integers(2, 5))
        
        combined_mask |= mask
    
    return vertices, combined_mask, affected


def apply_erosive_wear(mesh: trimesh.Trimesh,
                      cusps: List[CuspRegion],
                      depth_mm: float,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Apply irregular erosive wear pattern using noise-based removal.
    
    Creates organic, irregular wear patterns that look like
    natural erosion or acid wear.
    
    Args:
        mesh: Input trimesh object
        cusps: List of cusp regions
        depth_mm: Wear depth
        rng: Random number generator
        
    Returns:
        Tuple of (vertices, mask, affected cusps)
    """
    vertices = mesh.vertices.copy()
    
    # Select primary cusp
    primary_idx = rng.integers(0, len(cusps))
    primary_cusp = cusps[primary_idx]
    
    # Height threshold with noise
    base_threshold = primary_cusp.apex[2] - depth_mm
    
    # Create noise field based on vertex positions
    noise_scale = rng.uniform(0.5, 2.0)
    noise = np.sin(vertices[:, 0] * noise_scale) * np.cos(vertices[:, 1] * noise_scale)
    noise = noise * depth_mm * 0.5  # Scale noise amplitude
    
    # Effective threshold varies spatially
    effective_threshold = base_threshold + noise
    
    # Distance-based weighting from multiple cusps
    weights = np.zeros(len(vertices))
    affected = []
    
    n_cusps_involved = rng.integers(1, min(len(cusps), 3) + 1)
    involved_indices = rng.choice(len(cusps), size=n_cusps_involved, replace=False)
    
    for idx in involved_indices:
        cusp = cusps[idx]
        affected.append(int(idx))
        dist = np.linalg.norm(vertices - cusp.centroid, axis=1)
        sigma = cusp.radius * rng.uniform(1.5, 3.0)
        weights += np.exp(-(dist ** 2) / (2 * sigma ** 2))
    
    weights = weights / weights.max()
    
    # Apply threshold only where weights are significant
    removal_mask = (vertices[:, 2] > effective_threshold) & (weights > 0.1)
    
    return vertices, removal_mask, affected


def apply_localized_damage(mesh: trimesh.Trimesh,
                          cusp: CuspRegion,
                          depth_mm: float,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply small, localized damage that looks like chipping or fracture.
    
    Creates small, sharp-edged removal areas typical of mechanical damage.
    
    Args:
        mesh: Input trimesh object
        cusp: Target cusp region
        depth_mm: Damage depth
        rng: Random number generator
        
    Returns:
        Tuple of (vertices, mask)
    """
    vertices = mesh.vertices.copy()
    
    # Multiple small damage sites near the cusp
    n_sites = rng.integers(1, 4)
    combined_mask = np.zeros(len(vertices), dtype=bool)
    
    for _ in range(n_sites):
        # Random offset from apex
        offset_xy = rng.uniform(-cusp.radius, cusp.radius, size=2)
        offset_z = rng.uniform(-depth_mm * 0.3, 0)
        
        damage_center = cusp.apex.copy()
        damage_center[:2] += offset_xy
        damage_center[2] += offset_z
        
        # Small, irregular removal volume
        damage_radius = depth_mm * rng.uniform(0.5, 1.5)
        
        # Elongated ellipsoid for chip-like appearance
        rx = damage_radius * rng.uniform(0.5, 1.5)
        ry = damage_radius * rng.uniform(0.5, 1.5)
        rz = damage_radius * rng.uniform(0.3, 0.8)
        
        diff = vertices - damage_center
        ellipsoid_dist = np.sqrt(
            (diff[:, 0] / rx) ** 2 +
            (diff[:, 1] / ry) ** 2 +
            (diff[:, 2] / rz) ** 2
        )
        
        combined_mask |= (ellipsoid_dist < 1.0)
    
    return vertices, combined_mask


# =============================================================================
# WEAR APPLICATION
# =============================================================================

def generate_wear_variants(mesh: trimesh.Trimesh,
                          cusps: List[CuspRegion],
                          base_seed: int,
                          config: WearConfig = None) -> List[WearResult]:
    """
    Generate multiple diverse wear variants for a single tooth.
    
    Creates varied wear patterns with randomized:
    - Which cusps are affected (not always the same ones)
    - Wear types (spherical, planar, faceted, erosive, damage)
    - Wear depths and extents
    - Number and combination of affected cusps
    
    Args:
        mesh: Original tooth mesh
        cusps: Detected cusp regions
        base_seed: Base random seed for reproducibility
        config: Wear configuration parameters
        
    Returns:
        List of WearResult objects
    """
    if config is None:
        config = WearConfig()
    
    results = []
    occlusal_dir = np.array([0.0, 0.0, 1.0])
    
    if len(cusps) == 0:
        return results
    
    # Master RNG for this tooth
    master_rng = np.random.default_rng(base_seed)
    
    # Shuffle cusp order for this specific tooth (adds variety across teeth)
    cusp_order = master_rng.permutation(len(cusps))
    
    # --- VARIANT 1: Single cusp mild wear (random cusp, random method) ---
    rng = np.random.default_rng(base_seed + master_rng.integers(0, 1000))
    target_idx = int(cusp_order[0])
    target_cusp = cusps[target_idx]
    depth = rng.uniform(*config.mild_depth_range)
    
    # Random method selection
    method = rng.integers(0, 3)
    if method == 0:
        _, mask = apply_spherical_wear(mesh, target_cusp, depth, rng, config.surface_noise_sigma)
        wear_type = "spherical"
    elif method == 1:
        _, mask = apply_planar_cut(mesh, target_cusp, depth, rng, config.plane_tilt_range)
        wear_type = "planar"
    else:
        _, mask = apply_faceted_wear(mesh, target_cusp, depth, rng, rng.integers(2, 4))
        wear_type = "faceted"
    
    removal_fraction = mask.sum() / len(mask)
    if removal_fraction <= config.max_mild_removal_fraction and removal_fraction > 0.001:
        worn_mesh = create_worn_mesh(mesh, mask)
        results.append(WearResult(
            name=f"wear_mild_c{target_idx}_{wear_type}",
            mesh=worn_mesh,
            removed_mask=mask,
            wear_type=wear_type,
            wear_depth_mm=depth,
            cusps_affected=[target_idx],
            occlusal_direction=occlusal_dir,
            random_seed=base_seed
        ))
    
    # --- VARIANT 2: Different cusp, different method ---
    if len(cusps) >= 2:
        rng = np.random.default_rng(base_seed + 100 + master_rng.integers(0, 1000))
        target_idx = int(cusp_order[1])
        target_cusp = cusps[target_idx]
        depth = rng.uniform(*config.mild_depth_range)
        
        # Different method than variant 1
        method = rng.integers(0, 4)
        if method == 0:
            _, mask = apply_spherical_wear(mesh, target_cusp, depth * 1.2, rng, config.surface_noise_sigma)
            wear_type = "spherical"
        elif method == 1:
            _, mask = apply_height_clipping(mesh, target_cusp, depth, rng, config.falloff_sigma_factor)
            wear_type = "height_clip"
        elif method == 2:
            _, mask = apply_faceted_wear(mesh, target_cusp, depth, rng, rng.integers(3, 6))
            wear_type = "faceted"
        else:
            _, mask = apply_localized_damage(mesh, target_cusp, depth, rng)
            wear_type = "damage"
        
        removal_fraction = mask.sum() / len(mask)
        if removal_fraction <= config.max_mild_removal_fraction and removal_fraction > 0.001:
            worn_mesh = create_worn_mesh(mesh, mask)
            results.append(WearResult(
                name=f"wear_mild_c{target_idx}_{wear_type}",
                mesh=worn_mesh,
                removed_mask=mask,
                wear_type=wear_type,
                wear_depth_mm=depth,
                cusps_affected=[target_idx],
                occlusal_direction=occlusal_dir,
                random_seed=base_seed + 100
            ))
    
    # --- VARIANT 3: Asymmetric multi-cusp moderate wear ---
    rng = np.random.default_rng(base_seed + 200 + master_rng.integers(0, 1000))
    depth = rng.uniform(*config.moderate_depth_range)
    
    _, mask, affected = apply_asymmetric_wear(mesh, cusps, depth, rng)
    
    if mask.sum() > 0:
        worn_mesh = create_worn_mesh(mesh, mask)
        results.append(WearResult(
            name=f"wear_moderate_asymmetric_c{'_'.join(map(str, affected))}",
            mesh=worn_mesh,
            removed_mask=mask,
            wear_type="asymmetric",
            wear_depth_mm=depth,
            cusps_affected=affected,
            occlusal_direction=occlusal_dir,
            random_seed=base_seed + 200
        ))
    
    # --- VARIANT 4: Erosive/irregular wear pattern ---
    rng = np.random.default_rng(base_seed + 300 + master_rng.integers(0, 1000))
    depth = rng.uniform(*config.moderate_depth_range)
    
    _, mask, affected = apply_erosive_wear(mesh, cusps, depth, rng)
    
    if mask.sum() > 0:
        worn_mesh = create_worn_mesh(mesh, mask)
        results.append(WearResult(
            name=f"wear_moderate_erosive_c{'_'.join(map(str, affected))}",
            mesh=worn_mesh,
            removed_mask=mask,
            wear_type="erosive",
            wear_depth_mm=depth,
            cusps_affected=affected,
            occlusal_direction=occlusal_dir,
            random_seed=base_seed + 300
        ))
    
    # --- VARIANT 5: Heavy localized damage on random cusp ---
    rng = np.random.default_rng(base_seed + 400 + master_rng.integers(0, 1000))
    target_idx = int(rng.choice(cusp_order))
    target_cusp = cusps[target_idx]
    depth = rng.uniform(config.moderate_depth_range[0], config.moderate_depth_range[1] * 1.2)
    
    _, mask = apply_localized_damage(mesh, target_cusp, depth, rng)
    
    if mask.sum() > 0:
        worn_mesh = create_worn_mesh(mesh, mask)
        results.append(WearResult(
            name=f"wear_damage_c{target_idx}",
            mesh=worn_mesh,
            removed_mask=mask,
            wear_type="damage",
            wear_depth_mm=depth,
            cusps_affected=[target_idx],
            occlusal_direction=occlusal_dir,
            random_seed=base_seed + 400
        ))
    
    # --- VARIANT 6: Combined multi-method wear (most realistic) ---
    if len(cusps) >= 2:
        rng = np.random.default_rng(base_seed + 500 + master_rng.integers(0, 1000))
        combined_mask = np.zeros(len(mesh.vertices), dtype=bool)
        affected = []
        total_depth = 0
        
        # Apply different wear to 2-3 cusps
        n_cusps = min(rng.integers(2, 4), len(cusps))
        selected_cusps = rng.choice(len(cusps), size=n_cusps, replace=False)
        
        for idx in selected_cusps:
            cusp = cusps[idx]
            affected.append(int(idx))
            
            # Random depth per cusp
            cusp_depth = rng.uniform(config.mild_depth_range[0], config.moderate_depth_range[1])
            total_depth = max(total_depth, cusp_depth)
            
            # Random method per cusp
            method = rng.integers(0, 4)
            if method == 0:
                _, mask = apply_spherical_wear(mesh, cusp, cusp_depth, rng)
            elif method == 1:
                _, mask = apply_planar_cut(mesh, cusp, cusp_depth, rng, (5, 20))
            elif method == 2:
                _, mask = apply_faceted_wear(mesh, cusp, cusp_depth, rng, rng.integers(2, 4))
            else:
                _, mask = apply_height_clipping(mesh, cusp, cusp_depth, rng)
            
            combined_mask |= mask
        
        if combined_mask.sum() > 0:
            worn_mesh = create_worn_mesh(mesh, combined_mask)
            results.append(WearResult(
                name=f"wear_combined_c{'_'.join(map(str, sorted(affected)))}",
                mesh=worn_mesh,
                removed_mask=combined_mask,
                wear_type="combined",
                wear_depth_mm=total_depth,
                cusps_affected=sorted(affected),
                occlusal_direction=occlusal_dir,
                random_seed=base_seed + 500
            ))
    
    return results


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_outputs(output_dir: str,
                original_mesh: trimesh.Trimesh,
                original_filename: str,
                wear_results: List[WearResult]) -> None:
    """
    Save all outputs for a single tooth.
    
    Creates directory structure and saves:
    - Original mesh
    - Worn mesh variants
    - Removal masks
    - Metadata JSON
    
    Args:
        output_dir: Output directory path
        original_mesh: Original unworn mesh
        original_filename: Original PLY filename
        wear_results: List of wear simulation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original mesh
    original_path = os.path.join(output_dir, "original.ply")
    original_mesh.export(original_path)
    
    # Prepare metadata
    metadata = {
        "original_file": original_filename,
        "original_vertices": len(original_mesh.vertices),
        "original_faces": len(original_mesh.faces),
        "wear_variants": []
    }
    
    # Save each wear variant
    for result in wear_results:
        # Save worn mesh
        mesh_path = os.path.join(output_dir, f"{result.name}.ply")
        result.mesh.export(mesh_path)
        
        # Save removal mask
        mask_path = os.path.join(output_dir, f"removed_mask_{result.name.replace('wear_', '')}.npy")
        np.save(mask_path, result.removed_mask)
        
        # Add to metadata
        variant_meta = {
            "name": result.name,
            "wear_type": result.wear_type,
            "wear_depth_mm": float(result.wear_depth_mm),
            "cusps_affected": result.cusps_affected,
            "occlusal_direction": result.occlusal_direction.tolist(),
            "random_seed": result.random_seed,
            "vertices_removed": int(result.removed_mask.sum()),
            "removal_fraction": float(result.removed_mask.sum() / len(result.removed_mask)),
            "output_vertices": len(result.mesh.vertices),
            "output_faces": len(result.mesh.faces)
        }
        metadata["wear_variants"].append(variant_meta)
    
    # Save metadata JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved to {output_dir}")
    print(f"    - Original: {len(original_mesh.vertices)} vertices")
    for result in wear_results:
        print(f"    - {result.name}: {len(result.mesh.vertices)} vertices "
              f"({result.removed_mask.sum()} removed, {result.wear_type})")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_tooth(ply_path: str,
                 output_dir: str,
                 tooth_index: int,
                 base_seed: int = 42,
                 config: WearConfig = None) -> bool:
    """
    Process a single tooth through the wear simulation pipeline.
    
    Args:
        ply_path: Path to input PLY file
        output_dir: Base output directory
        tooth_index: Index for output naming
        base_seed: Base random seed
        config: Wear configuration
        
    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = WearConfig()
    
    filename = os.path.basename(ply_path)
    print(f"\nProcessing tooth {tooth_index + 1}: {filename}")
    
    try:
        # Load and preprocess
        mesh = load_and_preprocess(ply_path)
        print(f"  Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Compute cusp likelihood
        likelihood = compute_cusp_likelihood(mesh, config)
        print(f"  Cusp likelihood computed (max: {likelihood.max():.3f})")
        
        # Identify cusps
        cusps = identify_cusp_regions(mesh, likelihood, config.n_cusps_to_detect)
        print(f"  Detected {len(cusps)} cusp regions")
        
        for i, cusp in enumerate(cusps):
            print(f"    Cusp {i}: {len(cusp.vertex_indices)} vertices, "
                  f"apex Z={cusp.apex[2]:.2f}, likelihood={cusp.mean_likelihood:.3f}")
        
        # Generate wear variants
        seed = base_seed + tooth_index * 100
        wear_results = generate_wear_variants(mesh, cusps, seed, config)
        print(f"  Generated {len(wear_results)} wear variants")
        
        # Save outputs
        tooth_dir = os.path.join(output_dir, f"tooth_{tooth_index + 1:02d}")
        save_outputs(tooth_dir, mesh, filename, wear_results)
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main(input_dir: str = None,
        output_dir: str = None,
        base_seed: int = 42,
        config: WearConfig = None) -> None:
    """
    Main entry point for the wear simulation pipeline.
    
    Processes all PLY files in the input directory and generates
    artificially worn variants for each.
    
    Args:
        input_dir: Directory containing unworn PLY meshes
        output_dir: Output directory for results
        base_seed: Base random seed for reproducibility
        config: Wear configuration parameters
    """
    # Default paths relative to script location
    if input_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(os.path.dirname(script_dir), "Good teeth")
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
    
    if config is None:
        config = WearConfig()
    
    print("=" * 60)
    print("EDJ Artificial Wear Simulation Pipeline")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base seed: {base_seed}")
    print()
    
    # Find all PLY files
    ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))
    
    if len(ply_files) == 0:
        print(f"ERROR: No PLY files found in {input_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "pipeline_config.json")
    config_dict = {
        "base_seed": base_seed,
        "mild_depth_range": config.mild_depth_range,
        "moderate_depth_range": config.moderate_depth_range,
        "mild_cusps": config.mild_cusps,
        "moderate_cusps": config.moderate_cusps,
        "curvature_weight": config.curvature_weight,
        "elevation_weight": config.elevation_weight,
        "normal_variation_weight": config.normal_variation_weight,
        "n_cusps_to_detect": config.n_cusps_to_detect,
        "surface_noise_sigma": config.surface_noise_sigma,
        "max_mild_removal_fraction": config.max_mild_removal_fraction,
        "plane_tilt_range": config.plane_tilt_range
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Process each tooth
    successful = 0
    failed = 0
    
    for i, ply_path in enumerate(ply_files):
        if process_tooth(ply_path, output_dir, i, base_seed, config):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print()
    print("=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Processed: {successful} successful, {failed} failed")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate artificially worn EDJ tooth meshes"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory containing unworn PLY meshes (default: 'Good teeth')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results (default: 'artificial_wear/output')"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n-cusps",
        type=int,
        default=4,
        help="Number of cusps to detect (default: 4)"
    )
    parser.add_argument(
        "--mild-depth-min",
        type=float,
        default=0.3,
        help="Minimum mild wear depth in mm (default: 0.3)"
    )
    parser.add_argument(
        "--mild-depth-max",
        type=float,
        default=0.8,
        help="Maximum mild wear depth in mm (default: 0.8)"
    )
    parser.add_argument(
        "--moderate-depth-min",
        type=float,
        default=1.0,
        help="Minimum moderate wear depth in mm (default: 1.0)"
    )
    parser.add_argument(
        "--moderate-depth-max",
        type=float,
        default=1.5,
        help="Maximum moderate wear depth in mm (default: 1.5)"
    )
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = WearConfig(
        mild_depth_range=(args.mild_depth_min, args.mild_depth_max),
        moderate_depth_range=(args.moderate_depth_min, args.moderate_depth_max),
        n_cusps_to_detect=args.n_cusps
    )
    
    main(
        input_dir=args.input,
        output_dir=args.output,
        base_seed=args.seed,
        config=config
    )
