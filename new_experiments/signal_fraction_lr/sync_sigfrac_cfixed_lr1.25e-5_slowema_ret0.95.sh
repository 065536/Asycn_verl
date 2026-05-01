#!/bin/bash
set -xeuo pipefail

# Slow-EMA ablation for noisy split-gradient alignment.
# Code betas are observation weights, so retention 0.95 -> beta 0.05.
export SIGFRAC_RUN_SUFFIX="_slowema_ret0.95"
export SIGFRAC_R_EMA_BETA_SYM=0.05
export SIGFRAC_R_EMA_BETA_DOWN=0.05
export SIGFRAC_R_EMA_BETA_UP=0.05

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
