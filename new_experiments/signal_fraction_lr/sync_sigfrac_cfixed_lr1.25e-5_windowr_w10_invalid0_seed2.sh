#!/bin/bash
set -xeuo pipefail

# Diagnostic: W10 with invalid/misaligned steps entering the window as 0.
# Tests whether the original W10 gain depends on valid-only positive alignment bias.

export SEED=2
export SIGFRAC_RUN_SUFFIX="_windowr_w10_invalid0_seed2"
export SIGFRAC_R_WINDOW_SIZE=10
export SIGFRAC_R_WINDOW_MODE="replace_ema"
export SIGFRAC_R_WINDOW_INVALID_VALUE=0.0

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
