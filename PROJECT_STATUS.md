# Project Status

**Last updated**: 2026-04-30

## Current Focus

The signal-fraction controller is now being tested as a **low-frequency
reliability controller**, not a precise single-step LR oracle.

Current claim boundary:

```text
split-batch alignment is noisy but informative;
its useful part should be extracted through smoothing / windowing.
```

Single-step `g_A1^T g_A2` sign is too weak to use as a hard decision signal
(`g_dot_positive` around 55%), so the next 1.5B experiments test whether
temporal aggregation improves the r-side controller.

## Implemented Today

Updated the 1.5B c-fixed signal-fraction path:

- slow EMA wrappers
- alpha change-rate limit
- 3A windowed continuous-r controller

Main new scripts:

```bash
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w5.sh
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10.sh
```

Implementation files:

- `verl/workers/config/optimizer.py`
- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/fsdp_workers.py`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh`

New metrics:

- `actor/r_window`
- `actor/r_window_count`
- `actor/r_window_enabled`
- `actor/alpha_rate_limited`

Verification passed:

- `python3 -m py_compile` on changed Python files
- `bash -n` on base/W5/W10 scripts

## Next Runs

Run 3A first:

| script | purpose |
|---|---|
| `sync_sigfrac_cfixed_lr1.25e-5_windowr_w5.sh` | short-window continuous-r |
| `sync_sigfrac_cfixed_lr1.25e-5_windowr_w10.sh` | stronger low-pass continuous-r |

Expected local outputs:

- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w5.jsonl`
- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10.jsonl`

## Detailed Log

The longer historical project status remains in:

- `paper/PROJECT_STATUS.md`
- `memory/algorithm_design.md`
- `memory/engineering_impl.md`
