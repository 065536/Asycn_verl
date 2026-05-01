#!/bin/bash
set -xeuo pipefail

# Async Phase 1 c_fixed sweep: low scale.
SEED=42 \
BASE_LR=7.5e-6 \
LR_TAG=7.5e-6 \
EXP_NAME=deepseek1.5b_fa_partial_8gpu_sigfrac_cfixed_lr7.5e-6_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed.sh
