#!/bin/bash
set -xeuo pipefail

# Async Phase 1 c_fixed sweep: high scale, same nominal scale as sync B.
SEED=42 \
BASE_LR=1.25e-5 \
LR_TAG=1.25e-5 \
EXP_NAME=deepseek1.5b_fa_partial_8gpu_sigfrac_cfixed_lr1.25e-5_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed.sh
