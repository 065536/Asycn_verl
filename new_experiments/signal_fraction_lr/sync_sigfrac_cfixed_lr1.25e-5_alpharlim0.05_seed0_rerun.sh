#!/bin/bash
set -xeuo pipefail

export SEED=0
export SIGFRAC_RUN_SUFFIX="_alpharlim0.05_seed0_rerun"
export SIGFRAC_ALPHA_RATE_LIMIT=0.05

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
