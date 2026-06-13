#!/bin/bash
A2Q_MODE=rollout_only A2Q_LR=1e-6 A2Q_NORMALIZE=False SEED=1 bash "$(dirname "$0")/sync_a2q_stage1.sh"
