#!/bin/bash
# ============================================================
# Batch launcher for warmup sweep experiments
# ============================================================
# Submits multiple warmup values × 3 seeds to the cluster
#
# Usage:
#   bash launch_warmup_sweep.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/sync_sq_warmup_sweep.sh"

# Warmup values to sweep
WARMUP_VALUES=(10 20 30 40)

# Seeds
SEEDS=(0 1 42)

# Base LR (fixed from previous coordinate descent)
BASE_LR="1e-5"

echo "Starting warmup sweep: ${#WARMUP_VALUES[@]} warmup values × ${#SEEDS[@]} seeds = $((${#WARMUP_VALUES[@]} * ${#SEEDS[@]})) jobs"
echo "Warmup values: ${WARMUP_VALUES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Base LR: ${BASE_LR}"
echo ""

for warmup in "${WARMUP_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Submitting: SQ_WARMUP=${warmup} SEED=${seed}"
    
    # Submit job (adjust this command based on your cluster scheduler)
    # Example for SLURM:
    # sbatch --job-name="warmup${warmup}_s${seed}" \
    #        --output="logs/warmup${warmup}_seed${seed}_%j.out" \
    #        --wrap="SQ_WARMUP=${warmup} SQ_BASE_LR=${BASE_LR} SEED=${seed} bash ${SWEEP_SCRIPT}"
    
    # Example for direct execution (sequential, for testing):
    SQ_WARMUP=${warmup} SQ_BASE_LR=${BASE_LR} SEED=${seed} bash "${SWEEP_SCRIPT}" &
    
    # Add delay to avoid Ray port conflicts if running sequentially
    sleep 2
  done
done

echo ""
echo "All jobs submitted. Monitor with 'squeue' or check logs in verl/logs/"
