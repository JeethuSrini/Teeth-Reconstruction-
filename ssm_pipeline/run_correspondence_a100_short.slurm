#!/bin/bash
#SBATCH --job-name=tooth_corr_real_100k
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/correspondence_real_100k_%j.log
#SBATCH --error=logs/correspondence_real_100k_%j.err

set -euo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO with exit code $?"' ERR

mkdir -p logs

echo "=============================================================="
echo "Tooth Correspondence Pipeline (Real Worn) - GPU"
echo "Mode: coarse2fine (high-point correspondence, CPD on subset)"
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

# Initialize conda in a non-interactive batch-safe way.
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

PROJECT_DIR="/gpfs/home/ja5163/Cayo/Teeth-Reconstruction-"
SSM_DIR="$PROJECT_DIR/ssm_pipeline"
GOOD_TEETH_DIR="$PROJECT_DIR/Good teeth"
REAL_WEAR_DIR="$PROJECT_DIR/real_wear_input"
OUTPUT_DIR="$SSM_DIR/output/correspondence_real_100k_v3"

cd "$SSM_DIR"

echo "Configuration:"
echo "  - Script: correspondence_pipeline.py"
echo "  - Good teeth dir: $GOOD_TEETH_DIR"
echo "  - Real worn dir:  $REAL_WEAR_DIR"
echo "  - Output dir:     $OUTPUT_DIR"
echo "  - n_points:       100000"
echo "  - n_gpus:         4"
echo "  - registration:   coarse2fine"
echo "  - cpd_points:     43000"
echo "  - displacement_knn: 3"
echo ""

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

echo "Running correspondence..."
python correspondence_pipeline.py \
  --good-teeth "$GOOD_TEETH_DIR" \
  --artificial-wear "$REAL_WEAR_DIR" \
  --output "$OUTPUT_DIR" \
  --n-points 100000 \
  --n-gpus 4 \
  --registration-mode coarse2fine \
  --cpd-points 43000 \
  --displacement-knn 3

EXIT_CODE=$?

echo ""
echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Correspondence completed successfully."
    echo "Output: $OUTPUT_DIR"
else
    echo "Correspondence failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=============================================================="

exit $EXIT_CODE
