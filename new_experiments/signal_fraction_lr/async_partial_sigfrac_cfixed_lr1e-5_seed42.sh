#!/bin/bash
set -xeuo pipefail

# Async Phase 1 c_fixed sweep: middle scale.
SEED=42 \
BASE_LR=1e-5 \
LR_TAG=1e-5 \
EXP_NAME=deepseek1.5b_fa_partial_8gpu_sigfrac_cfixed_lr1e-5_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed.sh
