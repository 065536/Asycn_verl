#!/bin/bash
set -xeuo pipefail

BASE_LR=1e-5 \
LR_TAG=1e-5 \
SEED=42 \
EXP_NAME=deepseek7b_sync_16gpu_sigfrac_cfixed_lr1e-5_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed.sh
