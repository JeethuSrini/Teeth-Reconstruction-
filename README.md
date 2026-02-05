# Tooth Wear Reconstruction using Statistical Shape Models

A complete pipeline for reconstructing worn/damaged EDJ (enamel-dentine junction) tooth surfaces from 3D mesh data using **Statistical Shape Models (SSM)**.

## Results

| Original Worn Tooth | Reconstructed Tooth |
|:---:|:---:|
| ![Original with wear](original_with_wear.png) | ![Reconstruction final](reconstruction_final.png) |

**Interactive 3D Preview** (GitHub's PLY viewer):
- [View Hybrid Reconstructed Tooth (PLY)](ssm_pipeline/output/hybrid_reconstructions/tooth_01_wear_combined_c0_2_3/hybrid_reconstructed.ply)

---

## What This Project Does

This project addresses a common problem in dental anthropology and paleontology: **worn tooth surfaces lose anatomical information** that is critical for analysis. We use machine learning (PCA-based Statistical Shape Models) to **reconstruct the original unworn tooth surface** from partial observations.

### The Problem

Teeth wear down over time due to:
- Mastication (chewing)
- Bruxism (grinding)
- Dietary abrasion
- Erosion (chemical wear)

This wear removes the original cusp geometry, making morphological analysis difficult.

### Our Solution

We combine:
1. **Statistical Shape Models** trained on unworn teeth to learn "what teeth should look like"
2. **Hybrid reconstruction** that preserves the intact (unworn) regions and only fills in the worn areas
3. **Laplacian smoothing** for seamless, natural-looking surface reconstruction

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOOTH RECONSTRUCTION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │  Good Teeth  │         │ Worn Teeth   │         │  Artificial  │
    │  (Unworn)    │         │ (Real/Sim)   │         │    Wear      │
    │   8 PLY      │         │   9 PLY      │         │  Simulation  │
    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
           │                        │                        │
           │                        │                        ▼
           │                        │              ┌──────────────────┐
           │                        │              │ Generate worn    │
           │                        │              │ teeth + masks    │
           │                        │              │ (known GT)       │
           │                        │              └────────┬─────────┘
           │                        │                       │
           ▼                        ▼                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              STEP 1: CORRESPONDENCE PIPELINE                 │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │ 1. Sample uniform point clouds (25,000 points)      │    │
    │  │ 2. Normalize (center, scale, PCA-align)             │    │
    │  │ 3. ICP rigid alignment to template                  │    │
    │  │ 4. CPD non-rigid registration (GPU-accelerated)     │    │
    │  └─────────────────────────────────────────────────────┘    │
    │  Output: Corresponded point clouds where vertex i =          │
    │          same anatomical location across ALL teeth           │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              STEP 2: SSM RECONSTRUCTION                      │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │ 1. Build SSM from good teeth (PCA: mean + modes)    │    │
    │  │ 2. For worn tooth: fit SSM to observed (unworn) pts │    │
    │  │ 3. Predict missing anatomy from SSM coefficients    │    │
    │  └─────────────────────────────────────────────────────┘    │
    │  Output: 25k reconstructed point cloud                       │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              STEP 3: HYBRID RECONSTRUCTION                   │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │ 1. Load original high-res mesh (~119k vertices)     │    │
    │  │ 2. Apply inverse transforms to SSM output           │    │
    │  │ 3. Align SSM to worn mesh using ICP with scale      │    │
    │  │ 4. Merge: intact regions + SSM-filled worn regions  │    │
    │  └─────────────────────────────────────────────────────┘    │
    │  Output: Hybrid point cloud preserving original quality      │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              STEP 4: SMOOTH MESH GENERATION                  │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │ 1. Map points to original mesh topology (faces)     │    │
    │  │ 2. Laplacian hole-filling for smooth worn region    │    │
    │  │ 3. Boundary blending for seamless transition        │    │
    │  └─────────────────────────────────────────────────────┘    │
    │  Output: Final smooth mesh with reconstructed anatomy        │
    └─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Teeth-Reconstruction/
├── Good teeth/                 # Unworn EDJ crown surfaces (ground truth)
│   └── *.ply                  # 8 upper-left third molar meshes
├── Worn teeth/                 # Real worn/damaged examples
│   └── *.ply                  # 9 worn tooth meshes
├── artificial_wear/            # Wear simulation pipeline
│   ├── wear_simulation.py     # Generate artificial wear
│   └── output/                # Generated worn meshes + masks
├── ssm_pipeline/               # Main reconstruction pipeline
│   ├── correspondence_pipeline.py   # Point correspondence
│   ├── reconstruction_pipeline.py   # SSM building + reconstruction
│   ├── hybrid_reconstruction.py     # Combine SSM with original mesh
│   ├── pointcloud_to_smooth_mesh.py # Generate smooth final mesh
│   └── output/
│       ├── correspondence/          # Aligned point clouds
│       ├── ssm/                     # Mean shape + eigenvectors
│       ├── reconstructions/         # SSM predictions
│       └── hybrid_reconstructions/  # Final outputs
├── original_with_wear.png      # Example worn tooth
├── reconstruction_final.png    # Example reconstruction
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Teeth-Reconstruction.git
cd Teeth-Reconstruction

# Install dependencies
pip install -r ssm_pipeline/requirements.txt

# GPU acceleration (optional but recommended)
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x
```

---

## Quick Start

### Option 1: Process Artificial Wear (with ground truth)

```bash
cd ssm_pipeline

# Step 1: Correspondence (align all teeth to common template)
python correspondence_pipeline.py --n-points 25000

# Step 2: Build SSM and reconstruct worn teeth
python reconstruction_pipeline.py --n-components 5

# Step 3: Hybrid reconstruction (merge SSM with original mesh)
python hybrid_reconstruction.py --batch

# Step 4: Generate smooth final meshes
python pointcloud_to_smooth_mesh.py --batch -m laplacian --ssm-weight 0.1
```

### Option 2: Process a Single Tooth

```bash
cd ssm_pipeline

# Single tooth reconstruction
python hybrid_reconstruction.py -t 01 -w mild_cusp1

# Generate smooth mesh
python pointcloud_to_smooth_mesh.py \
    -i output/hybrid_reconstructions/tooth_01_wear_mild_cusp1/hybrid_reconstructed.ply \
    -o output/hybrid_reconstructions/tooth_01_wear_mild_cusp1/smooth_mesh.ply \
    --tooth-num 01 --wear-type mild_cusp1 \
    -m laplacian --ssm-weight 0.1
```

---

## Detailed Pipeline Steps

### Step 1: Artificial Wear Simulation

Generates synthetic worn teeth with known ground truth for training and evaluation.

```bash
cd artificial_wear
python wear_simulation.py
```

**Wear types generated:**
- Spherical/ellipsoidal cusp removal
- Tilted planar cuts
- Erosive/irregular patterns
- Localized damage (chipping)
- Combined multi-cusp wear

**Output:** `output/tooth_XX/` with worn PLY files and boolean masks (`removed_mask_*.npy`)

### Step 2: Correspondence Pipeline

Establishes point-to-point correspondence across all teeth using GPU-accelerated registration.

```bash
cd ssm_pipeline
python correspondence_pipeline.py --n-points 25000 --n-gpus 4
```

**What it does:**
1. Samples uniform point clouds (25,000 points per tooth)
2. Normalizes (center, scale, PCA-align)
3. ICP rigid alignment to template
4. CPD (Coherent Point Drift) non-rigid registration

**Output:** `output/correspondence/` with corresponded point clouds

### Step 3: SSM Reconstruction

Builds a Statistical Shape Model and reconstructs missing anatomy.

```bash
python reconstruction_pipeline.py --n-components 5 --regularization 1.0
```

**What it does:**
1. PCA on corresponded good teeth → mean shape + principal modes
2. For worn teeth: fit SSM coefficients to observed (intact) points
3. Reconstruct full shape using fitted coefficients

**Output:** `output/ssm/` (model) + `output/reconstructions/` (25k point clouds)

### Step 4: Hybrid Reconstruction

Combines high-resolution original mesh with SSM predictions.

```bash
python hybrid_reconstruction.py --batch
```

**Why this step?**
- SSM outputs only 25k points (low resolution)
- Original meshes have ~119k vertices (high resolution)
- This step preserves original quality in intact regions and fills worn areas with SSM predictions

**What it does:**
1. Loads original high-res mesh (~119k vertices)
2. Applies inverse transforms to align SSM output back to original space
3. Fine-tunes alignment using ICP with scale estimation
4. Merges: intact regions (original) + worn regions (SSM)

**Output:** `output/hybrid_reconstructions/*/hybrid_reconstructed.ply`

### Step 5: Smooth Mesh Generation

Creates final smooth meshes using Laplacian hole-filling.

```bash
python pointcloud_to_smooth_mesh.py --batch -m laplacian --ssm-weight 0.1
```

**Methods available:**
- `laplacian` (recommended): Solves Laplacian equation for C1-continuous surface
- `taubin`: Iterative Taubin smoothing

**Key parameters:**
- `--ssm-weight 0.1`: Lower = smoother fill, Higher = follows SSM more closely
- `--smooth-iterations 20`: Post-smoothing iterations

**Output:** `output/hybrid_reconstructions/*/smooth_mesh.ply`

---

## Key Algorithms

### Statistical Shape Model (SSM)

We use PCA to learn the principal modes of shape variation from unworn teeth:

```
Shape(α) = μ + Σ αᵢ · φᵢ
```

Where:
- `μ` = mean shape (25k × 3)
- `φᵢ` = principal components (eigenvectors)
- `αᵢ` = shape coefficients

For reconstruction, we fit coefficients to observed points and use the full model to predict missing anatomy.

### Laplacian Hole Filling

For smooth surface reconstruction, we solve:

```
L · x = b
```

Where `L` is the graph Laplacian, with boundary conditions from intact vertices. This creates a minimal-curvature surface that smoothly fills the worn region.

### Hybrid Reconstruction

The key insight: preserve original mesh quality where possible.

1. **Intact regions**: Keep original high-resolution vertices
2. **Worn regions**: Fill with SSM predictions mapped back to original space
3. **Boundary**: Blend smoothly using weighted Laplacian

---

## Evaluation Metrics

For each reconstruction (when ground truth is available):

| Metric | Description |
|--------|-------------|
| RMSE (missing) | Error on reconstructed worn region |
| RMSE (observed) | Fitting error on intact region |
| Hausdorff | Worst-case distance |
| Missing % | Fraction of vertices removed |

---

## GPU Acceleration

The pipeline supports multi-GPU processing:

```bash
# Use 4 GPUs for correspondence
srun --gres=gpu:4 python correspondence_pipeline.py --n-gpus 4

# Single GPU reconstruction
srun --gres=gpu:1 python reconstruction_pipeline.py
```

---

## Data Format

All meshes are in PLY format:
- Vertex positions (X, Y, Z)
- Face indices (triangular mesh)
- Vertex normals

---

## Dependencies

- Python 3.8+
- trimesh, open3d, numpy, scipy
- scikit-learn, sklearn
- probreg (GPU-accelerated CPD)
- cupy (GPU linear algebra)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tooth_reconstruction,
  title = {Tooth Wear Reconstruction using Statistical Shape Models},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/Teeth-Reconstruction}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
