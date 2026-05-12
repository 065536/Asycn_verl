#!/bin/bash
set -xeuo pipefail

# Diagnostic: W10 with a lower base scale to roughly match B's mean LR.
# Existing W10 mean alpha was about 4.81e-6 vs B about 3.25e-6, so
# base_lr is scaled by 3.25 / 4.81 ≈ 0.676.

export SEED=42
export SIGFRAC_RUN_SUFFIX="_windowr_w10_meanmatch_seed42"
export SIGFRAC_BASE_LR=8.45e-6
export SIGFRAC_R_WINDOW_SIZE=10
export SIGFRAC_R_WINDOW_MODE="replace_ema"

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
