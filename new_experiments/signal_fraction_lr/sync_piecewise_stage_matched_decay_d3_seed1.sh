#!/bin/bash
set -xeuo pipefail

# Baseline D3 (seed1): piecewise stage-matched decay.
SEED=1 \
EXP_NAME=deepseek1.5b_sync_8gpu_piecewise_stage_matched_decay_d3_seed1 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3.sh
