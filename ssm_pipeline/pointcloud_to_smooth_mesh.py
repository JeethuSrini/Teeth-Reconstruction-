#!/usr/bin/env python3
"""
Point Cloud to Smooth Mesh - GeoMagic-style Smoothing

Creates a smooth surface mesh by:
1. Loading the original mesh (with face connectivity)
2. Loading the wear mask to identify worn region
3. Mapping point cloud positions to mesh vertices
4. Applying heavy smoothing ONLY to worn region
5. Keeping intact surface untouched
6. Blending smoothly at the boundary

Usage:
    python pointcloud_to_smooth_mesh.py -i hybrid_reconstructed.ply -o smooth_mesh.ply --tooth-num 01 --wear-type mild_cusp1
"""

import argparse
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trimesh
from scipy.sparse import lil_matrix


def load_mesh_with_faces(filepath: str) -> trimesh.Trimesh:
    """Load mesh preserving faces."""
    return trimesh.load(filepath, process=False)


def load_point_cloud(filepath: str) -> np.ndarray:
    """Load point cloud as numpy array."""
    mesh = trimesh.load(filepath, process=False)
    if hasattr(mesh, 'vertices'):
        return np.array(mesh.vertices)
    return np.array(mesh)


def build_adjacency(faces: np.ndarray, n_verts: int):
    """Build adjacency matrix from faces."""
    adjacency = lil_matrix((n_verts, n_verts))
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i], face[j]] = 1
    return adjacency.tocsr()


def laplacian_hole_fill(vertices: np.ndarray, faces: np.ndarray,
                        worn_mask: np.ndarray,
                        ssm_points: np.ndarray = None,
                        ssm_weight: float = 0.1) -> np.ndarray:
    """
    Fill worn region using Laplacian interpolation (like GeoMagic).
    
    Solves for vertex positions that:
    1. Match boundary vertices exactly
    2. Minimize Laplacian (creates smooth surface)
    3. Optionally guided by SSM points as soft constraints
    
    This creates a C1-continuous surface at boundaries.
    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    
    n_verts = len(vertices)
    adjacency = build_adjacency(faces, n_verts)
    
    worn_indices = np.where(worn_mask)[0]
    intact_indices = np.where(~worn_mask)[0]
    
    print(f"    Total vertices: {n_verts}")
    print(f"    Worn (to fill): {len(worn_indices)}")
    print(f"    Intact (fixed): {len(intact_indices)}")
    
    # Build Laplacian matrix
    # L = D - A, where D is degree matrix, A is adjacency
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = diags(degrees)
    L = D - adjacency
    
    # Create mapping: old index -> new index for worn vertices
    worn_idx_map = {idx: i for i, idx in enumerate(worn_indices)}
    n_worn = len(worn_indices)
    
    # Build system: for each worn vertex, Laplacian = 0 (smooth)
    # Boundary conditions: neighbors that are intact contribute to RHS
    
    # Extract submatrix for worn vertices
    L_worn = lil_matrix((n_worn, n_worn))
    rhs = np.zeros((n_worn, 3))
    
    for i, idx in enumerate(worn_indices):
        neighbors = adjacency[idx].nonzero()[1]
        degree = len(neighbors)
        L_worn[i, i] = degree
        
        for n_idx in neighbors:
            if worn_mask[n_idx]:
                # Neighbor is also worn - add to matrix
                j = worn_idx_map[n_idx]
                L_worn[i, j] = -1
            else:
                # Neighbor is intact - add to RHS
                rhs[i] += vertices[n_idx]
    
    # Add soft constraint from SSM points if provided
    if ssm_points is not None and ssm_weight > 0:
        print(f"    Adding SSM guidance (weight={ssm_weight})")
        
        # Find nearest SSM point for each worn vertex
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ssm_points)
        distances, indices = nn.kneighbors(vertices[worn_mask])
        
        # Add regularization: (1 + w) * x = Laplacian_constraint + w * ssm_point
        for i in range(n_worn):
            L_worn[i, i] += ssm_weight
            rhs[i] += ssm_weight * ssm_points[indices[i, 0]]
    
    L_worn = L_worn.tocsr()
    
    print(f"    Solving Laplacian system...")
    
    # Solve for each coordinate
    result = vertices.copy()
    for dim in range(3):
        result[worn_mask, dim] = spsolve(L_worn, rhs[:, dim])
    
    print(f"    Hole filling complete")
    
    return result


def smooth_worn_region(vertices: np.ndarray, faces: np.ndarray,
                       worn_mask: np.ndarray,
                       iterations: int = 100,
                       lambda_factor: float = 0.7,
                       mu_factor: float = -0.72,
                       boundary_rings: int = 8) -> np.ndarray:
    """
    Apply heavy smoothing to worn region only, with boundary blending.
    """
    n_verts = len(vertices)
    adjacency = build_adjacency(faces, n_verts)
    
    worn_indices = set(np.where(worn_mask)[0])
    intact_indices = set(np.where(~worn_mask)[0])
    
    # Find boundary rings
    rings = []
    current_ring = set()
    
    for idx in worn_indices:
        neighbors = set(adjacency[idx].nonzero()[1])
        if neighbors & intact_indices:
            current_ring.add(idx)
    
    if current_ring:
        rings.append(current_ring)
    
    all_boundary = current_ring.copy()
    for r in range(1, boundary_rings):
        next_ring = set()
        for idx in current_ring:
            neighbors = set(adjacency[idx].nonzero()[1])
            for n in neighbors:
                if n in worn_indices and n not in all_boundary:
                    next_ring.add(n)
        if not next_ring:
            break
        rings.append(next_ring)
        all_boundary.update(next_ring)
        current_ring = next_ring
    
    # Compute smoothing weights
    smooth_weight = np.zeros(n_verts)
    
    interior = worn_indices - all_boundary
    for idx in interior:
        smooth_weight[idx] = 1.0
    
    for ring_idx, ring in enumerate(rings):
        t = (ring_idx + 1) / (len(rings) + 1)
        weight = 1.0 - t * t * t
        for idx in ring:
            smooth_weight[idx] = weight
    
    smoothed = vertices.copy()
    smoothable = list(worn_indices)
    
    for iteration in range(iterations):
        new_pos = smoothed.copy()
        
        for idx in smoothable:
            if smooth_weight[idx] > 0:
                neighbors = adjacency[idx].nonzero()[1]
                if len(neighbors) > 0:
                    neighbor_avg = smoothed[neighbors].mean(axis=0)
                    delta = neighbor_avg - smoothed[idx]
                    new_pos[idx] = smoothed[idx] + lambda_factor * smooth_weight[idx] * delta
        smoothed = new_pos
        
        new_pos = smoothed.copy()
        for idx in smoothable:
            if smooth_weight[idx] > 0:
                neighbors = adjacency[idx].nonzero()[1]
                if len(neighbors) > 0:
                    neighbor_avg = smoothed[neighbors].mean(axis=0)
                    delta = neighbor_avg - smoothed[idx]
                    new_pos[idx] = smoothed[idx] + mu_factor * smooth_weight[idx] * delta
        smoothed = new_pos
    
    return smoothed


def create_smooth_mesh(
    pointcloud_path: str,
    output_path: str,
    original_mesh_path: str,
    mask_path: str,
    smooth_iterations: int = 100,
    method: str = "laplacian",
    ssm_weight: float = 0.1
) -> dict:
    """
    Create smooth mesh from point cloud with region-aware smoothing.
    
    Methods:
    - laplacian: Solve Laplacian equation for smooth hole fill (best quality)
    - taubin: Iterative Taubin smoothing (faster, less smooth)
    """
    result = {"success": False}
    
    print(f"\n{'='*60}")
    print(f"Point Cloud to Smooth Mesh ({method.upper()} method)")
    print(f"{'='*60}")
    
    # Check files
    for path, name in [(pointcloud_path, "Point cloud"),
                       (original_mesh_path, "Original mesh"),
                       (mask_path, "Wear mask")]:
        if not os.path.exists(path):
            print(f"  [ERROR] {name} not found: {path}")
            return result
    
    print(f"  Loading point cloud: {pointcloud_path}")
    point_cloud = load_point_cloud(pointcloud_path)
    print(f"    {len(point_cloud)} points")
    
    print(f"  Loading original mesh: {original_mesh_path}")
    original_mesh = load_mesh_with_faces(original_mesh_path)
    print(f"    {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")
    
    print(f"  Loading wear mask: {mask_path}")
    worn_mask = np.load(mask_path).astype(bool)
    print(f"    {worn_mask.sum()} worn vertices ({100*worn_mask.mean():.1f}%)")
    
    # Verify mask matches mesh
    if len(worn_mask) != len(original_mesh.vertices):
        print(f"  [ERROR] Mask size mismatch: {len(worn_mask)} vs {len(original_mesh.vertices)}")
        return result
    
    original_vertices = np.array(original_mesh.vertices)
    
    if method == "laplacian":
        print(f"\n  Step 1: Laplacian hole filling with SSM guidance...")
        
        # Use SSM points as soft constraints for the Laplacian solve
        # The boundary is fixed, interior minimizes curvature while staying near SSM
        smoothed_vertices = laplacian_hole_fill(
            original_vertices,
            original_mesh.faces,
            worn_mask,
            ssm_points=point_cloud,
            ssm_weight=ssm_weight
        )
        
        # Optional: light post-smoothing for extra polish
        if smooth_iterations > 0:
            print(f"\n  Step 2: Light post-smoothing ({smooth_iterations} iterations)...")
            smoothed_vertices = smooth_worn_region(
                smoothed_vertices,
                original_mesh.faces,
                worn_mask,
                iterations=smooth_iterations,
                lambda_factor=0.5,
                mu_factor=-0.52,
                boundary_rings=5
            )
    else:
        # Original approach: map points then smooth
        print(f"\n  Step 1: Mapping point cloud to mesh vertices...")
        
        nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(point_cloud)
        distances, indices = nn.kneighbors(original_vertices)
        
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        new_vertices = original_vertices.copy()
        worn_indices = np.where(worn_mask)[0]
        for i in worn_indices:
            new_vertices[i] = np.sum(
                weights[i, :, np.newaxis] * point_cloud[indices[i]], axis=0
            )
        
        print(f"\n  Step 2: Smoothing worn region ({smooth_iterations} iterations)...")
        
        smoothed_vertices = smooth_worn_region(
            new_vertices,
            original_mesh.faces,
            worn_mask,
            iterations=smooth_iterations,
            lambda_factor=0.7,
            mu_factor=-0.72,
            boundary_rings=8
        )
    
    # Create output mesh
    output_mesh = trimesh.Trimesh(
        vertices=smoothed_vertices,
        faces=original_mesh.faces,
        process=False
    )
    output_mesh.fix_normals()
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    output_mesh.export(output_path)
    
    print(f"\n  Saved: {output_path}")
    print(f"    {len(output_mesh.vertices)} vertices, {len(output_mesh.faces)} faces")
    
    result["success"] = True
    result["output_path"] = output_path
    
    print(f"\n  Done!")
    
    return result


def run_batch(hybrid_reconstructions_dir: str,
              artificial_wear_dir: str,
              smooth_iterations: int = 20,
              method: str = "laplacian",
              ssm_weight: float = 0.3) -> list:
    """
    Run smooth mesh on all hybrid_reconstructions subdirectories.
    """
    from pathlib import Path
    
    base_path = Path(hybrid_reconstructions_dir)
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {hybrid_reconstructions_dir}")
        return []
    
    # Find all subdirs like tooth_01_wear_mild_cusp1
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    results = []
    
    print(f"Found {len(subdirs)} hybrid reconstruction directories\n")
    
    for subdir in subdirs:
        parts = subdir.name.split("_wear_")
        if len(parts) != 2:
            print(f"[SKIP] Cannot parse: {subdir.name}")
            continue
        
        tooth_num = parts[0].replace("tooth_", "")
        wear_type = parts[1]
        
        input_ply = subdir / "hybrid_reconstructed.ply"
        output_ply = subdir / "smooth_mesh.ply"
        
        if not input_ply.exists():
            print(f"[SKIP] Missing input: {input_ply}")
            continue
        
        original_mesh_path = Path(artificial_wear_dir) / f"tooth_{tooth_num}" / "original.ply"
        mask_path = Path(artificial_wear_dir) / f"tooth_{tooth_num}" / f"removed_mask_{wear_type}.npy"
        
        if not original_mesh_path.exists():
            print(f"[SKIP] Missing original mesh: {original_mesh_path}")
            continue
        if not mask_path.exists():
            print(f"[SKIP] Missing mask: {mask_path}")
            continue
        
        print(f"Processing {subdir.name}...")
        result = create_smooth_mesh(
            str(input_ply),
            str(output_ply),
            str(original_mesh_path),
            str(mask_path),
            smooth_iterations,
            method,
            ssm_weight
        )
        results.append(result)
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert point cloud to smooth mesh with region-aware smoothing"
    )
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Input point cloud PLY file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output mesh PLY file")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process all teeth in hybrid_reconstructions dir")
    parser.add_argument("--hybrid-dir", type=str, default="output/hybrid_reconstructions",
                        help="Hybrid reconstructions dir (for --batch)")
    parser.add_argument("--artificial-wear-dir", type=str, default="../artificial_wear/output",
                        help="Artificial wear output dir (for --batch)")
    parser.add_argument("--tooth-num", "-t", type=str, default="01",
                        help="Tooth number")
    parser.add_argument("--wear-type", "-w", type=str, default="mild_cusp1",
                        help="Wear type (for finding mask)")
    parser.add_argument("--original-mesh", type=str, default=None,
                        help="Path to original mesh")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path to wear mask")
    parser.add_argument("--smooth-iterations", "-s", type=int, default=20,
                        help="Number of post-smoothing iterations")
    parser.add_argument("--method", "-m", type=str, default="laplacian",
                        choices=["laplacian", "taubin"],
                        help="Hole filling method: laplacian (best) or taubin")
    parser.add_argument("--ssm-weight", type=float, default=0.3,
                        help="Weight for SSM guidance in Laplacian method (0-1)")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.batch:
        # Batch: process all subdirs in hybrid_reconstructions
        hybrid_dir = args.hybrid_dir if os.path.isabs(args.hybrid_dir) else os.path.join(base_dir, args.hybrid_dir)
        aw_dir = args.artificial_wear_dir if os.path.isabs(args.artificial_wear_dir) else os.path.join(base_dir, args.artificial_wear_dir)
        
        results = run_batch(
            hybrid_reconstructions_dir=hybrid_dir,
            artificial_wear_dir=aw_dir,
            smooth_iterations=args.smooth_iterations,
            method=args.method,
            ssm_weight=args.ssm_weight
        )
        
        success = sum(1 for r in results if r.get("success"))
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Success: {success} / {len(results)}")
        return
    
    if not args.input or not args.output:
        parser.error("--input and --output are required (or use --batch)")
    
    # Single run
    if args.original_mesh:
        original_mesh_path = args.original_mesh
    else:
        original_mesh_path = os.path.join(
            base_dir, "..", "artificial_wear", "output",
            f"tooth_{args.tooth_num}", "original.ply"
        )
    
    if args.mask:
        mask_path = args.mask
    else:
        mask_path = os.path.join(
            base_dir, "..", "artificial_wear", "output",
            f"tooth_{args.tooth_num}", f"removed_mask_{args.wear_type}.npy"
        )
    
    create_smooth_mesh(
        args.input,
        args.output,
        original_mesh_path,
        mask_path,
        args.smooth_iterations,
        args.method,
        args.ssm_weight
    )


if __name__ == "__main__":
    main()
