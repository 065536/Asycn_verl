#!/bin/bash
bash -n /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh && echo "OK: sync_constant_lr_diagnostic.sh"
bash -n /data/250010176/codes/verl/new_experiments/lr_schedule/sync_cd_lr_decay_5e-6_to_3.1e-6.sh && echo "OK: sync_cd_lr_decay_5e-6_to_3.1e-6.sh"
bash -n /data/250010176/codes/verl/new_experiments/lr_schedule/sync_cd_lr_decay_1e-5_to_3.1e-6.sh && echo "OK: sync_cd_lr_decay_1e-5_to_3.1e-6.sh"
bash -n /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_signal_quality_lr.sh && echo "OK: sync_signal_quality_lr.sh"
