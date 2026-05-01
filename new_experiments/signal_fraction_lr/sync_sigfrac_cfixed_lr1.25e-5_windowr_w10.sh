#!/bin/bash
set -xeuo pipefail

# Windowed continuous-r ablation.
# Replace the per-step EMA/fast-drop r controller with the mean of the latest
# 10 valid r observations.
export SIGFRAC_RUN_SUFFIX="_windowr_w10"
export SIGFRAC_R_WINDOW_SIZE=10
export SIGFRAC_R_WINDOW_MODE="replace_ema"

bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
