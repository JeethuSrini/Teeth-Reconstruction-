# Teeth Reconstruction

A project for reconstructing worn/damaged EDJ (enamel-dentine junction) tooth surfaces from 3D mesh data using Statistical Shape Models (SSM).

## Project Structure

```
Teeth-Reconstruction/
├── Good teeth/              # Unworn EDJ crown surfaces (ground truth)
│   └── *.ply               # 8 upper-left third molar meshes
├── Worn teeth/              # Real worn/damaged examples (reference only)
│   └── *.ply               # 9 worn tooth meshes
├── artificial_wear/         # Wear simulation pipeline
│   ├── wear_simulation.py  # Generate artificial wear
│   ├── requirements.txt
│   └── output/             # Generated worn meshes + masks
├── ssm_pipeline/            # Correspondence + SSM reconstruction
│   ├── correspondence_pipeline.py  # Point correspondence (ICP + CPD)
│   ├── reconstruction_pipeline.py  # SSM building + reconstruction
│   ├── requirements.txt
│   └── output/
│       ├── correspondence/  # Aligned point clouds
│       ├── ssm/             # Mean shape + eigenvectors
│       └── reconstructions/ # Reconstructed teeth + evaluation
└── README.md
```

## Full Pipeline Workflow

```
1. artificial_wear/     →  Generate worn teeth with known ground truth
2. correspondence/      →  Establish point correspondence across all teeth
3. reconstruction/      →  Build SSM, reconstruct worn teeth, evaluate accuracy
```

## Artificial Wear Simulation Pipeline

Generates realistic, artificially worn EDJ tooth meshes from unworn 3D surfaces. The artificial wear preferentially affects sharp, high-curvature (cusp) regions, reflecting biological wear patterns.

### Features

- **Cusp Detection**: Automatically identifies cusps using curvature, elevation, and normal variation
- **Multiple Wear Types**:
  - Spherical/ellipsoidal removal
  - Tilted planar cuts
  - Faceted wear (multiple planes)
  - Erosive/irregular patterns
  - Localized damage (chipping)
- **Varied Patterns**: Each tooth receives unique, randomized wear across different cusps
- **Reproducible**: All operations seeded for deterministic results

### Installation

```bash
cd artificial_wear
pip install -r requirements.txt
```

### Usage

```bash
# Run with defaults (processes "Good teeth" folder)
python wear_simulation.py

# Custom input/output directories
python wear_simulation.py -i "../Good teeth" -o "./output" --seed 42

# Adjust wear depth parameters (in mm)
python wear_simulation.py --mild-depth-min 0.3 --mild-depth-max 0.8 \
                          --moderate-depth-min 1.0 --moderate-depth-max 1.5
```

### Output Structure

```
output/
├── pipeline_config.json     # Configuration used
├── tooth_01/
│   ├── original.ply         # Copy of unworn mesh
│   ├── wear_mild_c0_spherical.ply
│   ├── wear_mild_c2_faceted.ply
│   ├── wear_moderate_asymmetric_c1_3.ply
│   ├── wear_moderate_erosive_c0_2.ply
│   ├── wear_damage_c1.ply
│   ├── wear_combined_c0_2_3.ply
│   ├── removed_mask_*.npy   # Boolean masks of removed vertices
│   └── metadata.json        # Wear parameters and statistics
└── tooth_02/
    └── ...
```

### Metadata JSON

Each tooth folder contains metadata with full reproducibility information:

```json
{
  "original_file": "cprc_nyu_n0047_ULM3_EDJ_GEO.ply",
  "wear_variants": [
    {
      "name": "wear_mild_c0_spherical",
      "wear_type": "spherical",
      "wear_depth_mm": 0.52,
      "cusps_affected": [0],
      "random_seed": 42
    }
  ]
}
```

---

## SSM Pipeline (Correspondence + Reconstruction)

The SSM pipeline establishes point correspondence across all teeth and builds a PCA-based Statistical Shape Model for reconstruction.

### Installation

```bash
cd ssm_pipeline
pip install -r requirements.txt
```

**GPU acceleration** requires CuPy (adjust for your CUDA version):
```bash
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x
```

### Step 1: Correspondence Pipeline

Establishes point-to-point correspondence using template-based registration:

```bash
python correspondence_pipeline.py \
    --good-teeth "../Good teeth" \
    --artificial-wear "../artificial_wear/output" \
    --n-points 20000

# Options:
#   --auto-template     Auto-select most central tooth as template
#   --no-gpu            Disable GPU acceleration
```

**What it does:**
1. Sample uniform point clouds (N=20,000 points)
2. Normalize (center, scale, PCA-align)
3. ICP rigid alignment to template
4. CPD non-rigid registration (GPU-accelerated)

**Output:** `output/correspondence/` with corresponded point clouds where vertex i = same anatomical location across all teeth.

### Step 2: Reconstruction Pipeline

Builds SSM and reconstructs missing anatomy:

```bash
python reconstruction_pipeline.py \
    --correspondence-dir "./output/correspondence" \
    --artificial-wear "../artificial_wear/output" \
    --n-components 5 \
    --regularization 1.0

# Options:
#   --variance-threshold 0.95  # Auto-select components by variance
#   --no-gpu                   # Disable GPU acceleration
```

**What it does:**
1. Build SSM from good teeth (PCA: mean + eigenvectors)
2. For each worn tooth:
   - Load removal mask (from artificial_wear)
   - Fit SSM to observed points (regularized least squares)
   - Reconstruct full shape
3. Evaluate against ground truth (RMSE, Hausdorff)

**Output:** `output/ssm/` (model) + `output/reconstructions/` (results)

### Evaluation Metrics

For each reconstruction:
- **RMSE (missing region)**: Error on reconstructed anatomy
- **RMSE (observed region)**: Sanity check on fitting
- **Hausdorff distance**: Worst-case error
- **Missing fraction**: % of points that were removed

---

## Data Format

All meshes are in PLY format containing:
- Vertex positions (3D coordinates)
- Face indices (triangular mesh)
- Vertex normals

## Dependencies

**Wear Simulation:**
- Python 3.8+
- trimesh, numpy, scipy, networkx

**SSM Pipeline:**
- trimesh, open3d, numpy, scipy
- probreg (GPU-accelerated CPD)
- cupy (GPU linear algebra - optional but recommended)
