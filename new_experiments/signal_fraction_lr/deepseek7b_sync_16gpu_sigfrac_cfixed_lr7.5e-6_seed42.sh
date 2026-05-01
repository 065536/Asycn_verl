#!/bin/bash
set -xeuo pipefail

BASE_LR=7.5e-6 \
LR_TAG=7.5e-6 \
SEED=42 \
EXP_NAME=deepseek7b_sync_16gpu_sigfrac_cfixed_lr7.5e-6_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed.sh
