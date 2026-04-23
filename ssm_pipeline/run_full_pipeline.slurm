#!/bin/bash
#SBATCH --job-name=tooth_full_pipeline
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/full_pipeline_%j.log
#SBATCH --error=logs/full_pipeline_%j.err

set -euo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO with exit code $?"' ERR

mkdir -p logs

echo "=============================================================="
echo "Tooth Full Pipeline (Correspondence + SSM Reconstruction)"
echo "  15 good teeth + 25 worn teeth (real + artificial)"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo ""

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
module purge
module load cuda/12.9
export LD_LIBRARY_PATH=/gpfs/share/apps/cuda/12.9/lib64:$LD_LIBRARY_PATH

CONDA_SH="/gpfs/data/davolilab/software/conda-envs/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "[ERROR] conda.sh not found at: $CONDA_SH"
  exit 1
fi
source "$CONDA_SH"
conda activate teeth
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
which python
python --version

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_DIR="/gpfs/home/ja5163/Cayo/Teeth-Reconstruction-"
SSM_DIR="$PROJECT_DIR/ssm_pipeline"

GOOD_TEETH_DIR="$PROJECT_DIR/all_good_teeth"
WORN_DIR="$PROJECT_DIR/all_worn_input"
CORR_OUTPUT="$SSM_DIR/output/correspondence_all_100k"
RECON_OUTPUT="$SSM_DIR/output/recon_all"

cd "$SSM_DIR"

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================================================================
# STEP 1: Correspondence Pipeline
# ==================================================================
echo "=============================================================="
echo "STEP 1: Correspondence Pipeline"
echo "=============================================================="
echo "  Good teeth dir:   $GOOD_TEETH_DIR  (15 teeth)"
echo "  Worn teeth dir:   $WORN_DIR  (25 worn variants)"
echo "  Output dir:       $CORR_OUTPUT"
echo "  n_points:         100000"
echo "  n_gpus:           4"
echo "  registration:     coarse2fine"
echo "  cpd_points:       43000"
echo "  displacement_knn: 3"
echo "  template_idx:     1  (cprc_nyu_n0047)"
echo ""

python correspondence_pipeline.py \
  --good-teeth "$GOOD_TEETH_DIR" \
  --artificial-wear "$WORN_DIR" \
  --output "$CORR_OUTPUT" \
  --n-points 100000 \
  --n-gpus 4 \
  --registration-mode coarse2fine \
  --cpd-points 43000 \
  --displacement-knn 3 \
  --template-idx 1

echo ""
echo "Correspondence completed at: $(date)"
echo ""

# ==================================================================
# STEP 2: SSM Reconstruction Pipeline
# ==================================================================
echo "=============================================================="
echo "STEP 2: SSM Reconstruction Pipeline"
echo "=============================================================="
echo "  Correspondence dir: $CORR_OUTPUT"
echo "  Worn teeth dir:     $WORN_DIR"
echo "  Output dir:         $RECON_OUTPUT"
echo "  variance_threshold: 0.99"
echo "  proxy_missing:      0.25"
echo "  skip_eval:          yes"
echo ""

python reconstruction_pipeline.py \
  --correspondence-dir "$CORR_OUTPUT" \
  --artificial-wear "$WORN_DIR" \
  --output "$RECON_OUTPUT" \
  --skip-eval \
  --proxy-missing-fraction 0.25 \
  --variance-threshold 0.99

echo ""
echo "=============================================================="
echo "Full pipeline completed successfully."
echo "  Correspondence: $CORR_OUTPUT"
echo "  Reconstruction: $RECON_OUTPUT"
echo "End time: $(date)"
echo "=============================================================="
