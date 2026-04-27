#!/bin/bash
set -xeuo pipefail

SEED=42 \
SHUFFLE_SEED=42 \
ALPHA_REPLAY_PATH=/data/250010176/codes/verl/exp_data/4.26/alpha_replay/b_alpha_seed42_step21_300.json \
EXP_NAME=deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpha_shuffled_seed42 \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpha_shuffled.sh
