#!/usr/bin/env python3
"""
Run the correspondence pipeline on the two original (unworn) teeth
(n0245 = TEST1 source, n0257 = TEST2 source) using the SAME template
and parameters as the chosen correspondence run (default: correspondence_real_100k_v2).

Outputs are saved into <correspondence-dir>/originals/ so the analysis
script can load them as properly corresponded data.

Usage:
    python correspond_originals.py                    # sequential, GPU
    python correspond_originals.py --n-gpus 2           # parallel on 2 GPUs
    python correspond_originals.py --no-gpu               # CPU only
    python correspond_originals.py --correspondence-dir output/correspondence_all_100k
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_CORR_BASE = os.path.join(SCRIPT_DIR, "output", "correspondence_real_100k_v2")
ORIGINAL_MESH_DIR = os.path.join(PROJECT_DIR, "Original models")

ORIGINAL_TEETH = {
    "original_TEST1": "cprc_nyu_n0245_ULM3_EDJ.ply",
    "original_TEST2": "cprc_nyu_n0257_ULM3_EDJ.ply",
}

sys.path.insert(0, SCRIPT_DIR)
from correspondence_pipeline import (
    CorrespondenceConfig,
    load_point_cloud,
    process_single_tooth,
    process_teeth_parallel,
)


def _merge_cpd_options_from_metadata(corr_base: str, cfg: dict) -> dict:
    """Fill registration_mode / cpd_points / displacement_knn from a sample tooth if missing."""
    if all(k in cfg for k in ("registration_mode", "cpd_points", "displacement_knn")):
        return cfg
    meta_paths = sorted(glob(os.path.join(corr_base, "good_teeth", "*", "registration_metadata.json")))
    meta_paths += sorted(glob(os.path.join(corr_base, "artificial_worn", "*", "registration_metadata.json")))
    for mp in meta_paths:
        with open(mp) as f:
            meta = json.load(f)
        if meta.get("is_template"):
            continue
        cpd = meta.get("cpd", {})
        if not cpd:
            continue
        mode = cpd.get("mode", "direct")
        cfg.setdefault("registration_mode", mode)
        if mode == "coarse2fine":
            cfg.setdefault("cpd_points", cpd.get("cpd_points", 25000))
            cfg.setdefault("displacement_knn", cpd.get("displacement_knn", 3))
        break
    else:
        cfg.setdefault("registration_mode", "direct")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Correspond the original (unworn) teeth for TEST1/TEST2")
    parser.add_argument("--correspondence-dir", type=str, default=DEFAULT_CORR_BASE,
                        help="Correspondence run root (template/, pipeline_config.json)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for parallel processing (default 1)")
    args = parser.parse_args()

    corr_base = os.path.abspath(args.correspondence_dir)
    template_dir = os.path.join(corr_base, "template")

    # Load existing pipeline config
    config_path = os.path.join(corr_base, "pipeline_config.json")
    if not os.path.exists(config_path):
        sys.exit(f"Pipeline config not found: {config_path}\n"
                 f"Run the main correspondence pipeline first.")
    with open(config_path) as f:
        saved = json.load(f)
    cfg = dict(saved["config"])
    cfg = _merge_cpd_options_from_metadata(corr_base, cfg)

    config = CorrespondenceConfig(
        n_points=cfg["n_points"],
        scale_method=cfg["scale_method"],
        icp_threshold=cfg["icp_threshold"],
        icp_max_iterations=cfg["icp_max_iterations"],
        icp_tolerance=cfg["icp_tolerance"],
        cpd_alpha=cfg["cpd_alpha"],
        cpd_beta=cfg["cpd_beta"],
        cpd_max_iterations=cfg["cpd_max_iterations"],
        cpd_tolerance=cfg["cpd_tolerance"],
        cpd_use_cuda=not args.no_gpu,
        seed=cfg["seed"],
        template_idx=cfg.get("template_idx", 0),
        registration_mode=cfg.get("registration_mode", "direct"),
        cpd_points=cfg.get("cpd_points", 25000),
        displacement_knn=cfg.get("displacement_knn", 3),
    )

    # Load existing template
    template_ply = os.path.join(template_dir, "template.ply")
    template_Vt_path = os.path.join(template_dir, "template_Vt.npy")
    if not os.path.exists(template_ply):
        sys.exit(f"Template not found: {template_ply}")

    template_points = load_point_cloud(template_ply)
    template_Vt = np.load(template_Vt_path) if os.path.exists(template_Vt_path) else None

    print("=" * 60)
    print("Correspond Original (Unworn) Teeth")
    print("=" * 60)
    print(f"Template       : {template_ply}")
    print(f"Template points: {template_points.shape}")
    print(f"n_points       : {config.n_points}")
    print(f"CPD mode       : {config.registration_mode}  "
          f"(cpd_points={config.cpd_points}, knn={config.displacement_knn})")
    print(f"GPU            : {config.cpd_use_cuda}")
    print(f"Parallel GPUs  : {args.n_gpus}")
    print()

    output_dir = os.path.join(corr_base, "originals")
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for tooth_name, filename in sorted(ORIGINAL_TEETH.items()):
        mesh_path = os.path.join(ORIGINAL_MESH_DIR, filename)
        if not os.path.exists(mesh_path):
            print(f"  [SKIP] {filename} not found at {mesh_path}")
            continue
        tooth_dir = os.path.join(output_dir, tooth_name)
        tasks.append((mesh_path, tooth_dir, tooth_name))
        print(f"  Queued: {tooth_name}  ({filename})")

    if not tasks:
        sys.exit("No original teeth found to process.")

    print()

    if args.n_gpus > 1 and len(tasks) > 1:
        n_success = process_teeth_parallel(
            tasks, template_points, config, n_gpus=args.n_gpus,
            template_Vt=template_Vt)
    else:
        n_success = 0
        for mesh_path, tooth_dir, tooth_name in tasks:
            ok = process_single_tooth(
                mesh_path, template_points, tooth_dir, config, tooth_name,
                template_Vt=template_Vt)
            if ok:
                n_success += 1

    print()
    print("=" * 60)
    print(f"Done: {n_success}/{len(tasks)} original teeth corresponded")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
