#!/bin/bash
set -xeuo pipefail

# Baseline 1 (seed0): constant LR = 3.10e-6
SEED=0 \
EXP_NAME=deepseek1.5b_sync_8gpu_matched_alpha_3.10e-6_seed0 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_matched_alpha_3.10e-6.sh
