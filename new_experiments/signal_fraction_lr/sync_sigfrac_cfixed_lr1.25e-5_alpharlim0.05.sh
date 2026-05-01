#!/bin/bash
set -xeuo pipefail

# Alpha-rate-limit ablation for noisy split-gradient alignment.
# Limit signal-fraction alpha to +/-5% relative change per optimizer step.
export SIGFRAC_RUN_SUFFIX="_alpharlim0.05"
export SIGFRAC_ALPHA_RATE_LIMIT=0.05

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
