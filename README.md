# Teeth Reconstruction

A project for reconstructing worn/damaged EDJ (enamel-dentine junction) tooth surfaces from 3D mesh data.

## Project Structure

```
Teeth-Reconstruction/
├── Good teeth/              # Unworn EDJ crown surfaces (ground truth)
│   └── *.ply               # 8 upper-left third molar meshes
├── Worn teeth/              # Real worn/damaged examples for reference
│   └── *.ply               # 9 worn tooth meshes
├── artificial_wear/         # Wear simulation pipeline
│   ├── wear_simulation.py  # Main simulation script
│   ├── requirements.txt    # Python dependencies
│   └── output/             # Generated worn meshes
└── README.md
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

## Data Format

All meshes are in PLY format containing:
- Vertex positions (3D coordinates)
- Face indices (triangular mesh)
- Vertex normals

## Dependencies

- Python 3.8+
- trimesh
- numpy
- scipy
- networkx