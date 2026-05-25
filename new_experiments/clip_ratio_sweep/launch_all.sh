#!/bin/bash
# Launch all clip ratio sweep experiments
# Usage: bash launch_all.sh
# Or run individual: CLIP_LOW=0.2 CLIP_HIGH=0.2 SEED=0 NCPUS=32 bash sync_clip_ratio_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/sync_clip_ratio_sweep.sh"

SEEDS=(0 1 42)

# Symmetric configs: low=high
SYMMETRIC_VALUES=(0.1 0.2 0.3 0.4)

# Asymmetric configs: (low, high)
ASYMMETRIC_CONFIGS=(
  "0.2 0.28"
  "0.2 0.4"
  "0.1 0.3"
)

echo "=== Clip Ratio Sweep ==="
echo "Symmetric: ${SYMMETRIC_VALUES[*]}"
echo "Asymmetric: ${ASYMMETRIC_CONFIGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total experiments: $(( (${#SYMMETRIC_VALUES[@]} + ${#ASYMMETRIC_CONFIGS[@]}) * ${#SEEDS[@]} ))"
echo ""

# Symmetric
for val in "${SYMMETRIC_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> CLIP_LOW=${val} CLIP_HIGH=${val} SEED=${seed}"
    CLIP_LOW="${val}" CLIP_HIGH="${val}" SEED="${seed}" NCPUS=32 bash "${SCRIPT}"
  done
done

# Asymmetric
for cfg in "${ASYMMETRIC_CONFIGS[@]}"; do
  read -r low high <<< "${cfg}"
  for seed in "${SEEDS[@]}"; do
    echo ">>> CLIP_LOW=${low} CLIP_HIGH=${high} SEED=${seed}"
    CLIP_LOW="${low}" CLIP_HIGH="${high}" SEED="${seed}" NCPUS=32 bash "${SCRIPT}"
  done
done

echo "All experiments completed!"
