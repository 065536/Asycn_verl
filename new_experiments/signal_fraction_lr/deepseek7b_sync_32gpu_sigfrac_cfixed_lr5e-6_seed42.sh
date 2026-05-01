#!/bin/bash
set -xeuo pipefail

NNODES=4 \
NGPUS_PER_NODE=8 \
BASE_LR=5e-6 \
LR_TAG=5e-6 \
SEED=42 \
EXP_NAME=deepseek7b_sync_32gpu_sigfrac_cfixed_lr5e-6_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed.sh
