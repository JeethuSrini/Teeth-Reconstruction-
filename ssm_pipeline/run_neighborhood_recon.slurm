#!/bin/bash
#SBATCH --job-name=tooth_neighborhood_recon
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/neighborhood_recon_%j.log
#SBATCH --error=logs/neighborhood_recon_%j.err

set -euo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO with exit code $?"' ERR

mkdir -p logs

echo "=============================================================="
echo "Neighborhood SSM Reconstruction"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

CORR_DIR="$SSM_DIR/output/correspondence_all_100k"
GLOBAL_RECON="$SSM_DIR/output/recon_all"
OUTPUT_DIR="$SSM_DIR/output/recon_neighborhood"
WORN_INPUT="$PROJECT_DIR/all_worn_input"

cd "$SSM_DIR"

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================================================================
# Neighborhood Reconstruction
# ==================================================================
echo "=============================================================="
echo "Running Neighborhood SSM Reconstruction"
echo "=============================================================="
echo "  Correspondence dir: $CORR_DIR"
echo "  Global recon dir:   $GLOBAL_RECON"
echo "  Output dir:         $OUTPUT_DIR"
echo "  Worn input dir:     $WORN_INPUT"
echo ""

python neighborhood_reconstruction.py \
  --correspondence-dir "$CORR_DIR" \
  --global-recon-dir "$GLOBAL_RECON" \
  --output "$OUTPUT_DIR" \
  --artificial-wear "$WORN_INPUT" \
  --max-neighbors 5 \
  --threshold-iqr-mult 2.0 \
  --proxy-missing-fraction 0.15 \
  --ssm-variance 0.95 \
  --regularization 1.0

echo ""
echo "=============================================================="
echo "Neighborhood reconstruction completed."
echo "  Output: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================================="
