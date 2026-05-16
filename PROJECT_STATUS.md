# Project Status

**Last updated**: 2026-05-15 (ratio_of_sums W10 rerun results after Bug 17 fix)

## 2026-05-15 ratio_of_sums W10 Rerun Results (Post Bug-17 Fix)

### Run status

Four seeds relaunched after Bug 17 fix. Three completed, one failed:

| seed | status | steps | file size |
|---:|---|---:|---|
| 0 | ✅ complete | 0-300 | 3.8 MB |
| 1 | ✅ complete | 0-300 | 3.8 MB |
| 2 | ✅ complete | 0-300 | 3.8 MB |
| 42 | ❌ failed | 0 (empty file) | 0 bytes |

seed42 JSONL is empty — the run either never started or crashed before writing
any metrics. Needs investigation or relaunch.

### Mechanism sanity check (PASSED)

- `actor/r_window_enabled` = 1.0 throughout (ratio_of_sums mode active)
- `actor/r_window_count` stabilizes at 10 by mid-training (window fully filled)
- `actor/r_window` ∈ [0.014, 0.037] — correct range, no longer inverted
- `actor/c_t` = 2.5e-4 post-handoff (frozen as expected with eta_c=0)
- `actor/alpha_t` post-warmup mean ≈ 4.6e-6, range [2.5e-6, 8.8e-6]

### Core avg5 results

| seed | best avg5 | @step | final avg5 | @step | drop |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3365 | 300 | 0.3365 | 300 | +0.0000 |
| 1 | 0.3352 | 270 | 0.3257 | 300 | -0.0095 |
| 2 | 0.3463 | 230 | 0.3414 | 300 | -0.0049 |
| **mean** | **0.3393** | | **0.3345** | | **-0.0048** |
| std | 0.0049 | | 0.0065 | | 0.0039 |

### Comparison with references

| method | seeds | mean best avg5 | mean final avg5 | mean drop |
|---|---:|---:|---:|---:|
| **B-current** | 3 | **0.3466** | **0.3440** | -0.0023 |
| W10 replace_ema | 3 | 0.3513 | 0.3413 | -0.0101 |
| **ratio_of_sums W10** | 3 | 0.3393 | 0.3345 | -0.0048 |

- ratio_of_sums vs B-current: **-0.0095** final avg5
- ratio_of_sums vs W10 replace_ema: **-0.0068** final avg5
- ratio_of_sums is the weakest of the three

### Key diagnostic findings

1. **r_hat_raw ≈ 0 in all post-warmup phases**: even after pooling 10 steps of
   cross-power and auto-power, the signal is noise-dominated. Phase means are
   O(1e-4) to O(1e-3), fluctuating around zero.

2. **g_dot_positive ≈ 50%**: warmup 62% → late 50%. ratio_of_sums pooling did
   not improve the sign signal reliability compared to replace_ema or B-current.

3. **alpha_t scale is higher than B**: post-warmup mean ~4.6e-6 vs B's ~3.1e-6.
   The ratio_of_sums estimator produces higher r_window values than the EMA
   path, inflating effective LR. This may explain part of the performance gap.

4. **PPO stability is fine**: KL mean ~3.8e-4, ratio_p95 ~1.03, entropy
   declining normally (0.55→0.32), grad_norm ~0.040. No safety concern.

5. **Training score improving normally**: score/mean from -0.60 to +0.12,
   response length stable ~2600-2700. The method trains, just less effectively.

### Interpretation

The theoretical motivation for ratio_of_sums (Welch-style pooled estimator for
more stable r̂_t) is valid, but empirically it **does not outperform** the
simpler replace_ema windowed mean or even B-current's fast EMA. Possible
explanations:

- Pooling cross-power sums does not solve the fundamental problem that
  E[ĝ_A1^T ĝ_A2] ≈ ||g||² is very small relative to noise variance when
  r_t ≈ 0.02. The denominator pooling helps slightly, but the numerator
  remains noise-dominated.
- The ratio_of_sums estimator produces systematically higher r_window values
  than per-step ratio averaging, causing alpha_t to be ~50% larger than B.
  This excess LR may explain the performance gap.

### Decision

ratio_of_sums W10 does not warrant further investigation. The replace_ema
windowed controller remains the better variant in the temporal aggregation
family. Detailed analysis report: `exp_data/5.15_ratio_of_sums_w10_rerun.md`.

---

## 2026-05-14 ratio_of_sums W10: Bug 17 — numerator/denominator swapped → 4 runs destroyed

### Bug 17 summary

The `ratio_of_sums` r-window estimator had **numerator and denominator swapped**
at the call site. All four ratio_of_sums W10 runs (seed 0/1/2/42) are invalid.

Bug location: `verl/workers/engine/fsdp/transformer_impl.py:1493-1494`

```python
# Before (wrong):
r_window_num=denom,   # auto-power → was treated as numerator
r_window_den=g_dot,   # cross-power → was treated as denominator

# After (fixed):
r_window_num=g_dot,   # cross-power = Σ ĝ_A1ᵀ ĝ_A2
r_window_den=denom,   # auto-power  = (||ĝ_A1||² + ||ĝ_A2||²) / 2
```

Consequence: `r_window = sum(auto-power) / sum(cross-power)` ≈ 2–80 instead of
the correct ∈ (0, 1]. This caused `alpha_t = c_t × r_ctrl` to explode
100–5000× above normal, destroying the policy within the first few post-warmup
steps.

Fix: single swap of the two arguments. Committed 2026-05-14.

### Destroyed run postmortem

Four runs completed 300 steps each but produced no usable training:

| seed | alpha_t (post-warmup mean) | r_window range | response_len (early→late) | score (early→late) |
|---:|---:|---|---|---|
| 0 | 6.35e-4 (212× normal) | [2.5, 130] | 3897 → 1 | -0.88 → -1.00 |
| 1 | 2.34e-3 (780× normal) | [0.04, 34] | 3845 → 759 | -0.88 → -1.00 |
| 2 | 2.60e-3 (867× normal) | [10, 23] | 3940 → 1 | -0.89 → -1.00 |
| 42 | 1.95e-2 (6500× normal) | [58, 80] | 3600 → 182 | -0.88 → -1.00 |

All four seeds: entropy collapsed to ~0 (or diverged to uniform), response
length dropped to 1 token, g_dot_positive = 0% post-warmup, score = -1.0.
Validation on all benchmarks = 0 at step 300.

Data files (invalid, kept for reference):

```text
deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_ratio_of_sums_w10_seed{0,1,2,42}.jsonl
```

### Next step

Four seeds have been relaunched with the bug fix. Awaiting results.

## 2026-05-12 Progress Update (superseded by Bug 17 postmortem above)

Prepared multi-seed ratio_of_sums W10 runs. All four completed but results are
invalid due to Bug 17. See above.

## Current Focus

The signal-fraction controller is now being tested as a **low-frequency
reliability controller**, not a precise single-step LR oracle.

Current claim boundary:

```text
split-batch alignment is noisy but informative;
its useful part should be extracted through smoothing / windowing.
```

Single-step `g_A1^T g_A2` sign is too weak to use as a hard decision signal
(`g_dot_positive` around 55%). The current question is whether low-frequency
aggregation can turn that weak signal into a useful controller.

## 2026-05-06 W10 Multi-Seed Readout

The new 1.5B multi-seed controller runs changed the current priority.

Main result:

```text
windowed continuous-r with W=10 is the most promising direction, but not yet a
settled improvement over B-current because seed42 had a large late drop.
```

### B-current reference

B-current already has three historical seeds:

| run | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|
| B seed42 old | 0.3470 | 0.3470 | +0.0000 |
| B seed0 old | 0.3458 | 0.3419 | -0.0040 |
| B seed1 old | 0.3471 | 0.3442 | -0.0029 |
| **mean** | **0.3466** | **0.3440** | **-0.0023** |

There is also a newer B-current diagnostic rerun:

| run | best avg5 | final avg5 |
|---|---:|---:|
| B seed42 rerun | 0.3385 | 0.3385 |

Do not say B-current lacks seeds. It does not. The only caveat is that the
historical B runs and the newest W10 runs were not all launched in the same
batch of code/logging changes.

### W10 status

Current W10 results:

| run | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|
| W10 seed0 new | 0.3456 | 0.3419 | -0.0037 |
| W10 seed1 new | 0.3624 | 0.3600 | -0.0024 |
| W10 seed42 old | 0.3460 | 0.3219 | -0.0241 |
| **mean** | **0.3513** | **0.3413** | **-0.0101** |

Interpretation:

- W10 has the best observed peak among the recent controller variants.
- W10 seed1 is a strong positive result.
- W10 seed42 old has a large late-stage drop, so final-score stability is not
  confirmed.
- The right claim is not "W10 solves noise"; the right claim is:

```text
temporal aggregation is the most promising way found so far to use the noisy
split-alignment signal, but W10 still needs stability confirmation.
```

### Other controller variants

Recent completed seed0/seed1 readout:

| group | seeds | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|---:|
| W10 | 2 | 0.3540 | 0.3510 | -0.0030 |
| slow EMA 0.95 | 2 | 0.3408 | 0.3344 | -0.0064 |
| alpharlim0.05 | 1 complete | 0.3405 | 0.3366 | -0.0039 |
| matched constant 3.10e-6 | 1 | 0.3407 | 0.3320 | -0.0087 |

Current reading:

- Slow EMA alone is not a clean solution. It can raise the effective LR but is
  seed-sensitive and does not reliably preserve gains.
- Alpha rate limiting is best understood as a safety/stability constraint, not
  a main source of improvement.
- W10 is currently the only new variant worth prioritizing.

### Next actions

Run only targeted confirmation, not a broad new sweep:

```bash
RESUME_MODE=disable \
SEED=42 \
SIGFRAC_RUN_SUFFIX="_windowr_w10_seed42_rerun" \
SIGFRAC_R_WINDOW_SIZE=10 \
SIGFRAC_R_WINDOW_MODE="replace_ema" \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh

RESUME_MODE=disable \
SEED=2 \
SIGFRAC_RUN_SUFFIX="_windowr_w10_seed2" \
SIGFRAC_R_WINDOW_SIZE=10 \
SIGFRAC_R_WINDOW_MODE="replace_ema" \
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
```

Do not prioritize new 7B experiments or 3B/3C controller variants until W10
stability is clarified.

## 2026-05-05 Diagnostic Runs Launched

After adding PPO tail / clip / dispersion diagnostics, started diagnostic runs
to collect the new metrics. These runs are not new algorithm variants; they are
for explaining late-stage risk and deciding whether a future controller should
use safety-tail signals.

Launched:

| run | purpose |
|---|---|
| current B / `sync_sigfrac_cfixed_lr1.25e-5.sh` | collect new diagnostics for the main adaptive baseline |
| matched constant / `sync_matched_alpha_3.10e-6.sh` | compare tail/safety dynamics against same-scale constant LR |

Primary readout:

- `actor/ratio_p95`, `actor/ratio_p99`
- `actor/ratio_frac_gt_1p2`, `actor/ratio_frac_lt_0p8`, `actor/ratio_frac_gt_1p5`
- `actor/pg_clipfrac_high`, `actor/pg_clipfrac_low`
- `critic/score/std`, `critic/rewards/std`
- `critic/advantages/std`, `critic/advantages/abs_mean`

Question:

```text
Do ratio tails / clip fractions / batch dispersion explain late-stage drop
better than KL mean, ratio std, or single-step r_hat sign?
```

## 2026-05-06 alpharlim0.05 seed0 failure

Run:

```bash
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0.sh
```

Status:

- stopped at `global_step=39`;
- no `global_step_*` checkpoint was written because `save_freq=50`;
- local JSONL is partial:
  `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0.jsonl`;
- log did not show a Python traceback; likely external interruption / job stop.

Observed issue in the log:

```text
swanlab failed to log training/rollout_probs_diff_top*_token_text because those
fields are strings, while swanlab expects scalar numeric chart values.
```

Fix:

- `verl/utils/tracking.py` now filters non-numeric metrics only for the
  `swanlab` backend.
- `console` and `file` still receive the original metrics, including string
  token text diagnostics.
- Added rerun wrapper with a clean suffix:
  `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0_rerun.sh`.

Recommended relaunch:

```bash
bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0_rerun.sh
```

## 2026-05-06 7B 32GPU status and fix

The three 7B 32GPU signal-fraction runs all failed for the same reason:

```text
RuntimeError: quantile() input tensor is too large
```

Cause:

- the new diagnostic metric `actor/ratio_p95` / `actor/ratio_p99` used
  `torch.quantile()` directly on a very large token tensor;
- this is fine on 1.5B-sized batches but can exceed backend limits on the 7B
  32GPU long-response setup.

Fix:

- `verl/trainer/ppo/core_algos.py` now uses deterministic strided subsampling
  inside `_masked_quantile()` when the valid tensor has more than 1,000,000
  elements;
- this only affects diagnostic quantile metrics, not training loss or updates;
- `python3 -m py_compile verl/trainer/ppo/core_algos.py` passed;
- `verl2` smoke test on a 2.1M-element tensor passed.

Current 7B progress:

| run | last training step seen in log | latest checkpoint | restart point |
|---|---:|---:|---:|
| `lr1e-5` | 170 | `global_step_170` | 170 |
| `lr7.5e-6` | 146 | `global_step_140` | 140 |
| `lr5e-6` | 146 | `global_step_140` | 140 |

The 7B template uses `RESUME_MODE=auto`, so rerunning the same scripts should
resume from the latest checkpoint directories above.

## 2026-05-05 Multi-Seed Follow-Up

Current interpretation after single-seed controller ablations:

```text
The new low-frequency controllers are close enough that seed noise is a serious
confound. Do not make a conclusion from seed42 alone.
```

The next priority is not adding more controller variants. It is to make the
main contenders comparable across seeds.

Launched follow-up runs:

| group | seeds launched | purpose |
|---|---:|---|
| `alpharlim0.05` | 0, 1 | test whether the most stable seed42 variant is consistently stable |
| `slowema_ret0.95` | 0, 1 | test whether high mid-training peak is real or seed-specific |
| `windowr_w10` | 0, 1 | test whether stronger low-pass/windowing reliably helps exploration or hurts late stability |

Scripts:

```bash
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0.sh
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed1.sh
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_seed0.sh
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_seed1.sh
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed0.sh
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed1.sh
```

Engineering update:

- `sync_sigfrac_cfixed_lr1.25e-5.sh` now supports `SEED=${SEED:-42}`.
- Existing behavior is unchanged when `SEED` is not set.
- New seed wrappers use unique `SIGFRAC_RUN_SUFFIX` values, so outputs should
  not overwrite the existing seed42 runs.

Readout target:

- compare mean / std over seeds for `best avg5`, `final avg5`, and `final-best`;
- inspect alpha dynamics (`alpha mean`, `p95 |delta alpha|`) and
  `g_dot_positive`;
- avoid claiming that the noise problem is solved unless the multi-seed mean
  clearly improves stability or final score over B.

## 2026-05-05 Diagnostic Pivot

Because single-step `r_hat` / `g_dot_positive` is noisy, the next analysis
focuses on whether `r_hat` becomes useful when combined with PPO learning
pressure.

Generated artifacts:

- `paper/figures/rhat_pg_loss_opportunity_diagnostic.png`
- `paper/analysis/rhat_pg_loss_opportunity_diagnostic.md`

Main readout:

```text
high r_hat + high pg_loss predicts the strongest future improvement over all
stages, but it does not dominate in late stage.
```

Interpretation:

- `r_hat` alone is weak.
- `pg_loss` is a stronger opportunity signal.
- The combination is useful mainly as an early/mid opportunity diagnostic.
- Late-stage control should not simply increase LR when `r_hat` and `pg_loss`
  are both high; it needs safety/tail diagnostics.

Added diagnostics for future runs without changing training behavior:

- PPO ratio tails:
  - `actor/ratio_p95`
  - `actor/ratio_p99`
  - `actor/ratio_frac_gt_1p2`
  - `actor/ratio_frac_lt_0p8`
  - `actor/ratio_frac_gt_1p5`
- PPO clipping split:
  - `actor/pg_clipfrac_high`
  - `actor/pg_clipfrac_low`
- train-batch dispersion:
  - `critic/score/std`
  - `critic/rewards/std`
  - `critic/advantages/std`
  - `critic/advantages/abs_mean`
  - `critic/returns/std`

Verification:

- `python3 -m py_compile verl/trainer/ppo/core_algos.py verl/trainer/ppo/metric_utils.py`
- `verl2` smoke test for ratio-tail metrics passed.

## 2026-05-05 Result Analysis

Local files currently available:

- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.jsonl`

Single-seed controller ablations are close overall; this is why the multi-seed
follow-up above was launched.

| run | max step | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|---:|
| `alpharlim0.05` | 300 | `0.3428 @290` | `0.3403 @300` | `-0.0025` |
| `slowema_ret0.95` | 300 | `0.3483 @250` | `0.3407 @300` | `-0.0075` |
| `slowema_ret0.95_alpharlim0.05` | 300 | `0.3365 @270` | `0.3324 @300` | `-0.0041` |
| `slowema_ret0.98` | 279 | `0.3378 @240` | `0.3240 @270` | `-0.0138` |

Current conclusion:

```text
The noise problem is not solved.
These low-frequency ablations are close to each other and do not clearly beat B.
```

Interpretation:

- `slowema_ret0.95` reaching `0.3483` is useful: larger effective alpha / slower
  smoothing may improve mid-training exploration.
- It does not hold the gain, so it is not a clean improvement.
- `alpharlim0.05` is stable but not better than B.
- `ret0.98` and `slowema+alpharlim` are too damped / underperform.

Next clean test remains true 3A W5/W10, with explicit sanity checks:

```text
actor/r_window_enabled = 1.0
actor/r_window_count grows toward W
JSONL/log/checkpoint name contains windowr_w5 or windowr_w10
```

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

## 4.30 Result Analysis (Superseded By 2026-05-05 Readout)

Analyzed completed runs in `deepseek1.5b_lr/`:

- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.jsonl`

Core metric definition:

- At each validation step, compute mean over 5 tasks:
  `val-core/{AIME,AIME2025,Idavidrein/gpqa,MINERVA,OLYMPIAD_BENCH}/acc/mean@16`.

Summary (revised for scale confound):

- Best `core_final` inside the 4.30 batch: `slowema_ret0.95` = `0.3408`.
- Most stable late phase (`final-best` closest to 0): `alpharlim0.05` = `-0.0025`.
- Largest `net_gain` (`final-step0`): `alpharlim0.05` = `+0.1155`.
- `slowema_ret0.98` underperforms in both final (`0.3240`) and late regression (`-0.0138`).
- Combined control (`slowema_ret0.95 + alpharlim0.05`) underperforms at current setting, but should NOT be labeled simply “more conservative”.

Control-side observations:

- `alpharlim0.05` strongly suppresses alpha oscillation (`mean|Δalpha| ~= 1.55e-7`) and keeps near-best final quality.
- `slowema_ret0.95` has much larger alpha scale (`alpha_mean ~= 6.17e-6`), which is a confound against B-current scale.
- Therefore `slowema_ret0.95` gains cannot yet be attributed solely to low-frequency smoothing.

Previous tentative actionable decision (now weakened by 2026-05-05 readout):

- Priority 1: multi-seed on `alpharlim0.05` (cleanest stability candidate).
- Priority 2: run scale-matched `slowema_ret0.95` rerun first, then consider multi-seed.
- Priority 3: continue W5/W10 runs to directly answer temporal-aggregation effectiveness.

Revised note: do not treat these four ablations as significant wins. They are
too close to B and each other. The main unanswered experiment is still actual
windowed continuous-r W5/W10.

## Detailed Log

The longer historical project status remains in:

- `paper/PROJECT_STATUS.md`
- `memory/algorithm_design.md`
- `memory/engineering_impl.md`
