#!/bin/bash
set -xeuo pipefail

export SEED=42
export SIGFRAC_RUN_SUFFIX="_windowr_w10_alpharlim0.05_seed42"
export SIGFRAC_R_WINDOW_SIZE=10
export SIGFRAC_R_WINDOW_MODE="replace_ema"
export SIGFRAC_ALPHA_RATE_LIMIT=0.05
export VAL_BEFORE_TRAIN=False

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
