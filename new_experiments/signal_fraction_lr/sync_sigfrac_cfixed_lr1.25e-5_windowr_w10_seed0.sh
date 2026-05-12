#!/bin/bash
set -xeuo pipefail

export SEED=0
export SIGFRAC_RUN_SUFFIX="_windowr_w10_seed0"
export SIGFRAC_R_WINDOW_SIZE=10
export SIGFRAC_R_WINDOW_MODE="replace_ema"

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
