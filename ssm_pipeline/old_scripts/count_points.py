#!/usr/bin/env python3
"""
Count points in PLY files before and after correspondence.

Usage:
    python count_points.py
"""

import os
from glob import glob
import trimesh
import numpy as np


def count_ply_points(filepath):
    """Count vertices/points in a PLY file."""
    try:
        mesh = trimesh.load(filepath)
        if hasattr(mesh, 'vertices'):
            return len(mesh.vertices)
        else:
            return len(mesh)
    except Exception as e:
        return f"Error: {e}"


def main():
    project_dir = "/gpfs/home/ja5163/Cayo/Teeth-Reconstruction-"
    
    print("=" * 70)
    print("Point Counts BEFORE Correspondence (Original Meshes)")
    print("=" * 70)
    
    # Good teeth (original meshes)
    print("\n--- Good Teeth (Original) ---")
    good_teeth_dir = os.path.join(project_dir, "Good teeth")
    good_files = sorted(glob(os.path.join(good_teeth_dir, "*.ply")))
    
    for f in good_files:
        count = count_ply_points(f)
        print(f"  {os.path.basename(f)}: {count} points")
    
    # Artificial wear originals
    print("\n--- Artificial Wear Originals ---")
    wear_dir = os.path.join(project_dir, "artificial_wear", "output")
    original_files = sorted(glob(os.path.join(wear_dir, "tooth_*", "original.ply")))
    
    for f in original_files:
        count = count_ply_points(f)
        tooth_name = os.path.basename(os.path.dirname(f))
        print(f"  {tooth_name}/original.ply: {count} points")
    
    # Artificial wear variants (sample)
    print("\n--- Artificial Wear Variants (sample) ---")
    wear_files = sorted(glob(os.path.join(wear_dir, "tooth_*", "wear_*.ply")))[:5]
    
    for f in wear_files:
        count = count_ply_points(f)
        tooth_name = os.path.basename(os.path.dirname(f))
        wear_name = os.path.basename(f)
        print(f"  {tooth_name}/{wear_name}: {count} points")
    
    print("\n" + "=" * 70)
    print("Point Counts AFTER Correspondence")
    print("=" * 70)
    
    # Corresponded point clouds
    corr_dir = os.path.join(project_dir, "ssm_pipeline", "output", "correspondence")
    
    print("\n--- Good Teeth (Corresponded) ---")
    good_corr = sorted(glob(os.path.join(corr_dir, "good_teeth", "tooth_*", "corresponded.ply")))
    
    for f in good_corr:
        count = count_ply_points(f)
        tooth_name = os.path.basename(os.path.dirname(f))
        print(f"  {tooth_name}/corresponded.ply: {count} points")
    
    print("\n--- Artificial Worn (Corresponded, sample) ---")
    worn_corr = sorted(glob(os.path.join(corr_dir, "artificial_worn", "*", "corresponded.ply")))[:5]
    
    for f in worn_corr:
        count = count_ply_points(f)
        tooth_name = os.path.basename(os.path.dirname(f))
        print(f"  {tooth_name}/corresponded.ply: {count} points")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if good_files:
        orig_count = count_ply_points(good_files[0])
        print(f"Original mesh points (typical): {orig_count}")
    
    if good_corr:
        corr_count = count_ply_points(good_corr[0])
        print(f"Corresponded points (all same): {corr_count}")


if __name__ == "__main__":
    import sys
    
    # If a file path is provided as argument, just count that file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Counting points in: {filepath}")
        count = count_ply_points(filepath)
        print(f"Points: {count}")
    else:
        main()
