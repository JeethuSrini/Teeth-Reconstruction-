# Tooth Wear Reconstruction using Statistical Shape Models

Reconstruct worn/damaged EDJ (enamel-dentine junction) tooth surfaces from 3D mesh data using PCA-based **Statistical Shape Models (SSM)** with per-tooth **neighborhood-adaptive priors** and non-rigid refinement.

## Results

| Original Worn Tooth | Reconstructed Smooth Mesh |
|:---:|:---:|
| ![Worn input](worn.png) | ![Reconstruction](reconstruction.png) |

### Global SSM vs Neighborhood SSM (25 worn teeth)

The neighborhood approach builds a **per-tooth local SSM** from the nearest anatomically-similar good teeth, instead of one global SSM built from all 15. Averaged across all 25 worn teeth:

| Metric | Global avg | Neighborhood avg | Δ | Verdict |
|---|---:|---:|---:|:---|
| R² (%) | 99.739 | 99.728 | -0.01% | tie |
| **Chamfer (mm)** | 0.0882 | 0.0866 | **-1.9%** | **Neighborhood better** |
| Hausdorff (mm) | 0.665 | 0.674 | +1.3% | Global better |
| RMSE worn→recon (mm) | 0.1256 | 0.1274 | +1.5% | Global better |
| **RMSE recon→worn (mm)** | 0.1017 | 0.0987 | **-3.0%** | **Neighborhood better** |
| MAE worn→recon (mm) | 0.0968 | 0.0964 | -0.4% | Neighborhood better |
| **MAE recon→worn (mm)** | 0.0797 | 0.0768 | **-3.6%** | **Neighborhood better** |
| Coverage @1x spacing | 0.135 | 0.136 | +0.6% | Neighborhood better |
| **Coverage @2x spacing** | 0.411 | 0.422 | **+2.7%** | **Neighborhood better** |
| **Coverage @5x spacing** | 0.774 | 0.789 | **+1.9%** | **Neighborhood better** |
| SSM fit RMSE | 0.0161 | 0.0183 | +13.3% | Global (expected — fewer modes) |

**Neighborhood wins on 7 of 11 metrics**, with the strongest gains on surface-coverage metrics:

| Metric | Neighborhood wins (out of 25) |
|---|:---:|
| Coverage @5x | **22 / 25** |
| RMSE recon→worn | 21 / 25 |
| R² | 20 / 25 |
| RMSE worn→recon | 20 / 25 |
| Chamfer | 18 / 25 |
| MAE recon→worn | 18 / 25 |
| Coverage @2x | 18 / 25 |
| Hausdorff | 16 / 25 |
| MAE worn→recon | 16 / 25 |
| Coverage @1x | 16 / 25 |
| SSM fit RMSE | 7 / 25 |

SSM trained on 15 unworn ULM3 teeth. Evaluation set: 9 real worn teeth + 8 artificially-worn levels of TEST1 + 8 artificially-worn levels of TEST2 = 25 teeth total.

---

## Data Analysis — Shape Space Visualizations

These figures are produced by `data_analysis/all_teeth_analysis.py` after Stage 2, and give a geometric view of how the 15 good teeth, 25 worn teeth, and their reconstructions distribute in shape space.

### PCA Scree Plot — How Much Variance Each Mode Captures

![PCA Scree](data_analysis/plots_v2/all_teeth_scree.png)

The first two PCA modes alone capture **~75 %** of total variance across the 15 good teeth, and ~95 % is reached by mode 8. This is why a 2D PCA plot is a faithful summary of tooth shape — and why local SSMs (K ≤ 5 neighbors) only need 1–4 modes.

### PCA 2D — Good Teeth, Worn Teeth, and Reconstructions

![PCA 2D all teeth](data_analysis/plots_v2/all_teeth_pca_2d.png)

Each tooth is one point in the PCA shape space of the 15 good teeth. Blue circles = good teeth, orange = originals, red triangles = real worn inputs, green diamonds = TEST1 wear levels, purple squares = TEST2 wear levels. Stars and X-markers overlay the corresponding **reconstructions**.

Two things jump out:
- **TEST1** and **TEST2** form tight per-tooth clusters — progressive wear barely moves a tooth in PCA space, which is why reconstructions stay close to the input.
- **Real worn teeth 03, 06, 07** sit in the upper-left corner far from the main cluster — this is exactly why the neighborhood algorithm selects only **tooth_14** (K=1) for these: they are outliers and averaging across all 15 good teeth would pull the reconstruction back toward a centroid they don't belong to.

### t-SNE (Raw 300,000-D features) — Non-linear Shape Embedding

![t-SNE raw](data_analysis/plots_v2/all_teeth_tsne_raw.png)

t-SNE on the full 300,000-D point-cloud features (before PCA) shows the same neighborhood structure that PCA exposes, but non-linearly. TEST2 forms a very tight cluster in the lower-right — every wear level of TEST2 shares nearly identical overall shape. TEST1 fans out along a roughly linear "wear trajectory" on the left side. Real worn teeth are scattered among the good teeth, confirming each one has its own anatomically-similar subset — the core motivation for the neighborhood approach.

### Worn-to-Reconstruction Distance in PCA Space

![Paired distance PCA 2D](data_analysis/plots_v2/paired_dist_pca_2d.png)

Euclidean distance in PC1–PC2 space between each worn tooth and its reconstruction. Small bars mean the reconstruction lands near its input — which is what we want. **T07, T06, T03** show the largest displacements: these are exactly the teeth the neighborhood algorithm flagged as outliers (K=1 with tooth_14 as sole neighbor), because the SSM has to pull a heavily-worn tooth toward a complete-anatomy prior. The TEST1 and TEST2 reconstructions cluster near zero, confirming the pipeline is stable on clean inputs.

A full gallery of paired-distance plots (t-SNE, UMAP, local variants) is in [`data_analysis/plots_v2/`](data_analysis/plots_v2/).

---

## What This Project Does

Teeth wear down over time through mastication, bruxism, dietary abrasion, and chemical erosion. This wear removes original cusp geometry from the enamel-dentine junction (EDJ), making morphological analysis difficult in dental anthropology and paleontology.

This pipeline learns the statistical shape variation of **unworn upper-left third molars (ULM3)** from 15 complete specimens, then reconstructs missing anatomy on worn teeth in two stages:

1. **Global SSM reconstruction** — one PCA model built from all 15 good teeth, used as baseline.
2. **Neighborhood SSM reconstruction** — for each worn tooth, the pipeline automatically selects its anatomically-nearest good teeth and builds a *local* SSM tailored to that tooth's shape family. This gives a tighter shape prior and measurably better surface coverage.

Output per worn tooth: a 100k-point reconstructed point cloud, a smooth watertight triangle mesh, the reconstruction in the original millimeter input space, and a full evaluation JSON.

---

## Pipeline Overview

```
                TOOTH RECONSTRUCTION PIPELINE (3 STAGES)

  +-------------------+        +-------------------+
  |  Good Teeth (15)  |        |  Worn Teeth (25)  |
  |  Unworn ULM3 PLY  |        |  Real + Test PLY  |
  +---------+---------+        +---------+---------+
            |                            |
            v                            v
  +----------------------------------------------------------+
  |  STAGE 1: CORRESPONDENCE                                 |
  |  correspondence_pipeline.py                              |
  |                                                          |
  |   1. Sample 100,000 uniform points per tooth             |
  |   2. Normalize (center, scale to bbox diag, PCA-align)   |
  |   3. ICP rigid alignment to auto-selected template       |
  |   4. Coarse-to-fine CPD non-rigid registration           |
  |      (CPD on 43k subset -> KNN upsample to 100k)         |
  |                                                          |
  |  Output: corresponded.ply per tooth                      |
  |          (100k points in shared anatomical frame)        |
  +-------------------------+--------------------------------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
  +---------------------------+  +---------------------------+
  |  STAGE 2a: GLOBAL SSM     |  |  STAGE 2b: NEIGHBORHOOD   |
  |  reconstruction_          |  |  neighborhood_            |
  |      pipeline.py          |  |      reconstruction.py    |
  |                           |  |                           |
  |  - PCA on all 15 teeth    |  |  For EACH worn tooth:     |
  |  - Fit SSM coefficients   |  |   1. Project into PCA     |
  |  - Non-rigid refinement   |  |      shape space          |
  |  - Poisson mesh           |  |   2. Pick good teeth      |
  |                           |  |      within threshold (<=5)|
  |                           |  |   3. Build LOCAL SSM      |
  |                           |  |   4. Fit + refine + mesh  |
  +-------------+-------------+  +-------------+-------------+
                |                              |
                +--------------+---------------+
                               |
                               v
  +----------------------------------------------------------+
  |  STAGE 3: COMPARISON & ANALYSIS                          |
  |  data_analysis/all_teeth_analysis.py                     |
  |                                                          |
  |  PCA scores, pairwise distances, % improvement plots     |
  +----------------------------------------------------------+
```

---

## How Neighborhood Selection Works (4 Steps)

For every worn tooth, the pipeline decides *which* good teeth to build the local SSM from:

1. **PCA on good teeth.** All 15 good teeth are flattened and projected into a low-dimensional PCA shape space (up to 10 components, 95% variance). Each tooth becomes one point in that space.
2. **Adaptive distance threshold.** All pairwise distances between the 15 good teeth are computed in PCA space. The threshold is set at `median + 2 × IQR` of those pairwise distances (≈ 13.72 for this dataset). Anything farther is considered anatomically "too different."
3. **Nearest-neighbors within threshold.** The worn tooth is projected into the same PCA space. Distances to all 15 good teeth are computed, and every tooth closer than the threshold is selected — capped at 5.
4. **No minimum required.** Even a single neighbor is used (that tooth's shape becomes the local mean, 0 PCA modes). Only if *zero* good teeth are within the threshold does the method fall back to the global SSM. In practice this fallback triggered on only **1 of 25** worn teeth (tooth_05, whose nearest good tooth sat at distance 30 — well past the 13.72 threshold).

**Example neighbor assignments from this dataset:**
- `tooth_01` → neighbors `[08, 10, 04, 15, 11]` (K=5)
- `tooth_03`, `tooth_06`, `tooth_07` → neighbor `[14]` only (K=1, local mean = tooth_14)
- `tooth_05` → global fallback (no neighbor within threshold)
- `tooth_TEST2_level7` → neighbors `[09, 03, 05, 12, 06]` (K=5)

---

## Project Structure

```
Teeth-Reconstruction-/
├── all_good_teeth/                 # 15 unworn ULM3 EDJ meshes (SSM training set)
│   └── cprc_nyu_*.ply
├── all_worn_input/                 # 25 worn tooth inputs (9 real + TEST1 + TEST2)
│   ├── tooth_01/wear_real.ply      # Real worn teeth
│   │   ...
│   ├── tooth_09/wear_real.ply
│   ├── tooth_TEST1/
│   │   ├── wear_level0.ply         # Artificially-worn (8 levels each)
│   │   │   ...
│   │   └── wear_level7.ply
│   └── tooth_TEST2/
│       └── wear_level0..7.ply
│
├── ssm_pipeline/
│   ├── correspondence_pipeline.py       # STAGE 1: point correspondence
│   ├── reconstruction_pipeline.py       # STAGE 2a: global SSM reconstruction
│   ├── neighborhood_reconstruction.py   # STAGE 2b: neighborhood-adaptive SSM
│   ├── local_mean_reconstruction.py     # (legacy) single-tooth local mean
│   ├── correspond_originals.py          # utility: correspond original-space outputs
│   ├── run_correspondence_a100_short.slurm
│   ├── run_full_pipeline.slurm          # correspondence + global recon
│   ├── run_neighborhood_recon.slurm     # neighborhood reconstruction only
│   └── output/
│       ├── correspondence_all_100k/     # Stage 1 output (15 good + 25 worn corresponded)
│       ├── recon_all/                   # Stage 2a global reconstructions
│       └── recon_neighborhood/          # Stage 2b neighborhood reconstructions
│           ├── ssm/
│           ├── neighbor_selection.json  # which neighbors each worn tooth used
│           ├── comparison.json          # global vs neighborhood per tooth
│           └── reconstructions/tooth_XX/
│               ├── worn_input.ply
│               ├── reconstructed.ply
│               ├── reconstructed_in_input_space.ply
│               ├── reconstructed_smooth.ply
│               ├── coefficients.npy, removed_mask.npy
│               └── evaluation.json
│
├── data_analysis/
│   ├── all_teeth_analysis.py       # PCA + distance + comparison plots across all recon outputs
│   ├── good_teeth_pca.py           # PCA on just the good-tooth training set
│   ├── worn_teeth_projection.py    # project worn teeth into good-teeth PCA space
│   └── plots/                      # generated figures
│
├── requirements.txt                # pip freeze of `teeth` conda env
├── worn.png / reconstruction.png   # example visualization (input vs output)
└── README.md
```

---

## Installation

Requires **Python 3.9–3.12** (Open3D does not yet support 3.13+).

### Option A — Clone + pip install (matches the cluster environment exactly)

```bash
git clone https://github.com/yourusername/Teeth-Reconstruction.git
cd Teeth-Reconstruction
pip install -r requirements.txt
```

### Option B — HPC cluster with conda (NYU-Langone setup)

```bash
source /gpfs/data/davolilab/software/conda-envs/miniconda3/etc/profile.d/conda.sh
conda activate teeth
```

All SLURM scripts in `ssm_pipeline/` assume this `teeth` environment.

### GPU support

CPD non-rigid registration and SSM SVD can be GPU-accelerated via CuPy. The pinned `requirements.txt` installs `cupy-cuda12x`. For a CUDA 11 system, replace with `cupy-cuda11x`.

---

## Quick Start

All reconstruction commands assume you are in `ssm_pipeline/` with the environment active.

```bash
cd ssm_pipeline
conda activate teeth    # or: source venv/bin/activate
```

### Stage 1 — Correspondence (GPU, ~1–4 hours for 40 teeth)

Establishes point-to-point anatomical correspondence across all 15 good + 25 worn teeth.

```bash
python correspondence_pipeline.py \
  --good-teeth "../all_good_teeth" \
  --artificial-wear "../all_worn_input" \
  --output "output/correspondence_all_100k" \
  --n-points 100000 \
  --registration-mode coarse2fine \
  --cpd-points 43000 \
  --displacement-knn 3 \
  --auto-template
```

On the HPC cluster:

```bash
sbatch run_correspondence_a100_short.slurm
```

### Stage 2a — Global SSM Reconstruction (CPU, ~10 min for 25 teeth)

Baseline: one SSM built from all 15 good teeth, applied to every worn tooth.

```bash
python reconstruction_pipeline.py \
  --correspondence-dir "output/correspondence_all_100k" \
  --artificial-wear "../all_worn_input" \
  --output "output/recon_all" \
  --skip-eval \
  --proxy-missing-fraction 0.25 \
  --variance-threshold 0.99
```

Or combined with Stage 1 via SLURM:

```bash
sbatch run_full_pipeline.slurm
```

### Stage 2b — Neighborhood SSM Reconstruction (GPU recommended, ~20 min)

Per-worn-tooth local SSM from adaptively-selected nearest good-tooth neighbors.

```bash
python neighborhood_reconstruction.py \
  --correspondence-dir "output/correspondence_all_100k" \
  --global-recon-dir "output/recon_all" \
  --output "output/recon_neighborhood" \
  --artificial-wear "../all_worn_input" \
  --max-neighbors 5 \
  --threshold-iqr-mult 2.0 \
  --proxy-missing-fraction 0.15 \
  --ssm-variance 0.95 \
  --regularization 1.0
```

On the HPC cluster:

```bash
sbatch run_neighborhood_recon.slurm
```

Outputs land in `output/recon_neighborhood/`. Two JSONs summarize the run:
- `neighbor_selection.json` — which good teeth each worn tooth used, distances, and fallback reasons
- `comparison.json` — side-by-side metrics vs the global reconstruction

### Stage 3 — Cross-method Analysis

```bash
cd ../data_analysis
python all_teeth_analysis.py \
  --correspondence-dir ../ssm_pipeline/output/correspondence_all_100k \
  --extra-recon-dir ../ssm_pipeline/output/recon_neighborhood/reconstructions
```

Produces PCA plots, pairwise-distance matrices, and comparison figures in `data_analysis/plots_v2/`.

---

## Parameters Reference

### correspondence_pipeline.py

| Flag | Default | Description |
|------|---------|-------------|
| `--good-teeth`, `-g` | `../all_good_teeth` | Directory of unworn tooth PLY files |
| `--artificial-wear`, `-a` | `../all_worn_input` | Directory of worn-tooth subdirectories (each with a `wear_*.ply`) |
| `--output`, `-o` | `output/correspondence` | Output directory |
| `--n-points`, `-n` | 20000 | Points sampled per tooth. Use **100000** for high-fidelity reconstruction |
| `--registration-mode` | `direct` | `direct`: CPD on all points. `coarse2fine`: CPD on subset then KNN upsample (required for 100k) |
| `--cpd-points` | 25000 | Points used for CPD in `coarse2fine` mode. 43000 recommended for 100k total |
| `--displacement-knn` | 3 | KNN neighbors for upsampling coarse CPD deformation |
| `--auto-template` | off | Auto-select the most central tooth as template (recommended) |
| `--template-idx` | 0 | Manual template index (ignored if `--auto-template`) |
| `--n-gpus` | 1 | Number of GPUs for parallel processing |
| `--no-gpu` | off | Force CPU-only registration |
| `--seed` | 42 | Random seed |

### reconstruction_pipeline.py (Global SSM)

| Flag | Default | Description |
|------|---------|-------------|
| `--correspondence-dir`, `-c` | `output/correspondence` | Directory with correspondence outputs from Stage 1 |
| `--artificial-wear`, `-a` | `../all_worn_input` | Directory of worn-tooth inputs |
| `--output`, `-o` | `output/` | Output directory |
| `--variance-threshold` | 0.99 | PCA variance threshold |
| `--n-components`, `-n` | auto | Override number of PCA components |
| `--regularization`, `-r` | 1.0 | Tikhonov regularization strength |
| `--skip-eval` | off | Skip ground-truth evaluation (required for real worn teeth) |
| `--proxy-missing-fraction` | 0.15 | Fraction of points treated as "missing" when `--skip-eval` is on. Range 0.10–0.30 |
| `--test-tooth` | none | Hold out a tooth by ID for leave-one-out evaluation |
| `--no-gpu` | off | Force CPU-only SVD |

### neighborhood_reconstruction.py (Neighborhood SSM)

| Flag | Default | Description |
|------|---------|-------------|
| `--correspondence-dir` | `output/correspondence_all_100k` | Stage 1 outputs |
| `--global-recon-dir` | `output/recon_all` | Stage 2a outputs (used for global fallback + side-by-side comparison) |
| `--output` | `output/recon_neighborhood` | Output directory |
| `--artificial-wear` | `../all_worn_input` | Worn-tooth input directory |
| `--worn-teeth` | all found | Restrict to specific worn tooth IDs |
| `--max-neighbors` | 5 | Cap on neighbors per worn tooth |
| `--threshold-iqr-mult` | 2.0 | Threshold = median + this × IQR of pairwise good-tooth distances |
| `--pca-variance` | 0.95 | PCA variance threshold for the neighbor-finding PCA |
| `--ssm-variance` | 0.95 | PCA variance threshold for the local SSM |
| `--proxy-missing-fraction` | 0.15 | Same as global pipeline |
| `--regularization` | 1.0 | Tikhonov regularization |
| `--no-gpu` | off | Force CPU-only SVD |

---

## Output Files

Each worn-tooth directory (`output/<run>/reconstructions/tooth_XX/`) contains:

| File | Description |
|------|-------------|
| `worn_input.ply` | Worn tooth sampled to 100k points in normalized SSM space |
| `reconstructed.ply` | SSM reconstruction in normalized space (100k points) |
| `reconstructed_in_input_space.ply` | Reconstruction mapped back to original millimeter coordinates |
| `reconstructed_smooth.ply` | Watertight triangle mesh (Screened Poisson + Taubin smoothing) |
| `removed_mask.npy` | Boolean (100k): `True` = point treated as missing |
| `coefficients.npy` | Fitted PCA coefficients |
| `evaluation.json` | All metrics: refinement, geometric comparison, SSM info, neighbor info (neighborhood run only) |

The SSM itself is saved to `output/<run>/ssm/`:

| File | Description |
|------|-------------|
| `mean_shape.ply` / `mean_shape.npy` | Mean tooth shape (100k × 3) |
| `eigenvectors.npy`, `eigenvalues.npy` | PCA modes |
| `ssm_metadata.json` | Training info |
| `modes/` | Visualizations of each PCA mode (±2σ) |

Neighborhood run also produces two top-level JSONs:
- `neighbor_selection.json` — per-tooth neighbor assignments, distances, threshold info
- `comparison.json` — per-tooth global vs neighborhood metric comparison

---

## Evaluation Metrics

All metrics compare the reconstruction (in original input space) against the raw worn tooth.

| Metric | What it measures |
|--------|-----------------|
| **R² (variance explained)** | Fraction of the worn tooth's spatial variance captured by the reconstruction. 99.9% = nearly identical overall |
| **Chamfer distance (mm)** | Symmetric average nearest-neighbor distance. Best single "overall accuracy" number. Lower is better |
| **Hausdorff distance (mm)** | Worst-case nearest-neighbor distance. Sensitive to outliers |
| **RMSE worn→recon (mm)** | RMS of nearest-neighbor distances from worn points to the reconstruction |
| **RMSE recon→worn (mm)** | RMS in the opposite direction — catches over-extrapolated reconstruction points |
| **MAE worn→recon / recon→worn** | Mean absolute variant of the above. When close to RMSE, errors are uniform |
| **Coverage @1x / 2x / 5x spacing** | Fraction of worn points with a reconstruction point within 1×, 2×, or 5× the median worn-point spacing. Measures how tightly the reconstruction follows the worn surface |
| **SSM fit RMSE** | Residual after fitting the SSM to observed points only. Lower is better, but neighborhood SSMs inherently have fewer modes |

---

## Key Algorithms

### Statistical Shape Model (SSM)

PCA learns the principal modes of shape variation:

```
Shape(b) = μ + V · b
```

`μ` is the mean shape (100k × 3 flattened to 300k), `V` the eigenvectors, `b` the fitted coefficients. Coefficients are fitted to observed (non-missing) points via Tikhonov-regularized least squares:

```
b = (VᵀV + λΛ⁻¹)⁻¹ Vᵀ (x_obs - μ_obs)
```

Coefficients are clipped to ±4σ to prevent extreme extrapolation.

### Neighborhood SSM

The same PCA math, but `μ` and `V` are rebuilt per worn tooth from only its K selected neighbors (`K ≤ 5`). For `K = 1`, `μ` is that single neighbor's shape and `V` is empty (pure mean-shape fit). Built local SSMs are cached by `frozenset(neighbor_indices)` so teeth with identical neighbor sets don't recompute the SVD.

### Non-Rigid Refinement

After SSM fitting:
1. **Observed points** are replaced exactly with the worn input coordinates (observation error → 0).
2. **Missing points** are displaced by KNN-interpolated corrections from nearby observed points, producing smooth transitions.

### Screened Poisson Surface Reconstruction

1. Estimate + orient surface normals (outward from centroid)
2. Open3D Screened Poisson (octree depth 9) → watertight triangle mesh
3. Trim bottom 1% of Poisson density (removes extrapolated fringe)
4. 30 iterations Taubin smoothing (λ=0.5, μ=-0.53)
5. Fix normals and triangle winding via trimesh

---

## Tooth-to-Specimen Mapping

Real worn teeth (`all_worn_input/tooth_01..09/wear_real.ply`):

| Pipeline ID | Original Specimen |
|-------------|-------------------|
| tooth_01 | cprc_nyu_n0225_ULM3_EDJ_damage.ply |
| tooth_02 | cprc_nyu_n0265_teeth_ULM3_WS_EDJ_WEAR&Damage.ply |
| tooth_03 | cprc_nyu_n0274_ULM3_EDJ_damage.ply |
| tooth_04 | cprc_nyu_n0294_ULM3_WS_EDJ_WEAR.ply |
| tooth_05 | cprc_nyu_n0295_ULM3_Dentine&damage.ply |
| tooth_06 | cprc_nyu_n0296_ULM3_WS_ed_dentine&damage.ply |
| tooth_07 | cprc_nyu_n0299_ULM3_EDJ_damage.ply |
| tooth_08 | cprc_nyu_n0311_ULM3_WS_EDJ_WORN.ply |
| tooth_09 | cprc_nyu_n0312_ULM3_WS_EDJ_WORN.ply |

Artificially-worn test teeth: `tooth_TEST1/wear_level0..7.ply` and `tooth_TEST2/wear_level0..7.ply` — 8 progressively-worn versions of two source teeth, used to test reconstruction robustness at increasing wear levels.

Good teeth (SSM training set, `all_good_teeth/`): 15 unworn ULM3 EDJ meshes including `n0043, n0047, n0049, n0256, n0258, n0264, n0266, n0269, n0291, n0292, n0293, n0298, n0300, n0307, n0350`.

---

## Dependencies

Full pinned list: [`requirements.txt`](requirements.txt). Headlines:

- Python 3.9–3.12
- trimesh >= 4.0.0
- open3d >= 0.18.0
- numpy, scipy, scikit-learn
- probreg >= 0.3.0 (GPU-accelerated CPD)
- pycpd >= 2.0.0 (CPU fallback)
- cupy-cuda12x (optional, GPU linear algebra)

---

## Citation

```bibtex
@software{tooth_reconstruction_ssm,
  title  = {Tooth Wear Reconstruction using Statistical Shape Models
            with Neighborhood-Adaptive Priors},
  author = {Jeevan Ananth},
  year   = {2026},
  url    = {https://github.com/yourusername/Teeth-Reconstruction}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
