# Detailed Signal-Fraction Analysis (2026-04-30, Scale-Confound Revision)

## Executive Summary

The 4.30 phase should not be interpreted as “low-frequency control is already proven better”.
The key reason is a **new scale confound**: slow-EMA variants run at substantially higher mean alpha than
B-current-level scale. Under this confound, `slowema_ret0.95` being highest-final among 4.30 variants is
insufficient to claim low-frequency smoothing gain by itself.

Current strongest conclusion:

- `alpharlim0.05` is the **cleanest stability-improvement candidate** in 4.30:
  near-best final core, smallest late regression, and strongly reduced high-frequency alpha jitter.

## Scope and Data

This report analyzes completed 1.5B runs in `deepseek1.5b_lr/`:

- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.jsonl`

Core metric:

`core(step) = mean(val-core/{AIME,AIME2025,Idavidrein/gpqa,MINERVA,OLYMPIAD_BENCH}/acc/mean@16)`

## Aggregated Comparison (4.30 Intra-Batch)

| Run | core_init | core_best@step | core_final | net_gain | drop |
|---|---:|---:|---:|---:|---:|
| slowema_ret0.95 | 0.230333 | 0.348250@250 | **0.340750** | +0.110417 | -0.007500 |
| alpharlim0.05 | 0.224833 | 0.342833@290 | 0.340333 | **+0.115500** | **-0.002500** |
| slowema_ret0.95 + alpharlim0.05 | 0.223500 | 0.336458@270 | 0.332375 | +0.108875 | -0.004083 |
| slowema_ret0.98 | 0.222417 | 0.337833@240 | 0.324042 | +0.101625 | -0.013792 |

## Scale-Confound Check (Critical)

Reference (historical): B-current typical alpha is approximately `3.1e-6` (project context).

| Run | alpha_mean | alpha_last | vs B-current alpha_mean | interpretation |
|---|---:|---:|---:|---|
| slowema_ret0.95 | 6.17e-6 | 4.87e-6 | ~2.0x | severe scale confound |
| alpharlim0.05 | 3.42e-6 | 2.50e-6 | ~1.1x | mild scale shift (acceptable with caveat) |
| slowema_ret0.95 + alpharlim0.05 | 6.29e-6 | 2.64e-6 | ~2.0x | severe scale confound |
| slowema_ret0.98 | 5.45e-6 | 4.52e-6 | ~1.8x | severe scale confound |

### Claim boundary (must enforce)

Can say:

- `alpharlim0.05` gives the cleanest stability signature in 4.30.
- `slowema_ret0.95` is top-final among 4.30 runs.

Cannot yet say:

- “slow EMA improves low-frequency control” as a causal claim.
- “combined strategy is better/worse due to conservativeness” without disentangling scale and interaction.

## Baseline Context Missing from 4.30 Table

4.30 table is intra-batch only. It must be read alongside historical baselines:

- B-current (main controller baseline)
- D3 (coarse schedule baseline)
- C310 (mean-alpha-matched constant)
- S-shuffled (alpha-distribution shuffled)
- M-2.97e-6 (fixed LR baseline)

Without this context, controller-improvement claims are incomplete.

## Task-Level Notes

### Why alpharlim0.05 is the cleanest signal

- `core_final=0.340333` (near best)
- `drop=-0.002500` (best late stability)
- `mean|delta(alpha)|=1.553e-7` (strong high-frequency suppression)
- `alpha_rate_limited ratio=0.57` (bandwidth control is actively used)
- More balanced behavior on MINERVA than slowema_ret0.95.

### Why slowema_ret0.95 is still ambiguous

- Best final among 4.30 (`0.340750`), but with `alpha_mean=6.17e-6` (~2x B-current typical).
- Therefore gains are confounded by update scale, not isolating low-frequency effect.

### Combined strategy interpretation correction

Do **not** describe it as simply “more conservative”.

- `alpha_mean=6.29e-6` is actually highest in the batch.
- More plausible explanation: slow-EMA + rate-limit interaction distorts alpha trajectory (high early scale plus constrained late correction).

### ret0.98 interpretation correction

Do **not** summarize only as “over-smoothed”.

- It appears **too slow to adapt**: late alpha remains relatively high (`alpha_last=4.52e-6`) while late regression is larger.
- This aligns with delayed response to late-stage risk.

## Revised Findings (for direct citation)

1. `alpharlim0.05` gives the cleanest stability signal.
   It achieves near-best final core with the smallest late regression and substantially lower alpha step-to-step variation.
2. `slowema_ret0.95` has the highest final core among these four runs, but is confounded by a much larger mean alpha.
   Its gain cannot yet be attributed solely to low-frequency smoothing.
3. `ret0.98` is likely too slow to adapt.
   It keeps alpha relatively high late in training and shows larger late regression.
4. Combining slow EMA and rate limit is not beneficial at the current setting.
   The interaction is nontrivial and should not be treated as simply “more conservative”.

## Revised Next-Step Priority

1. **Priority 1:** multi-seed for `alpharlim0.05`.
2. **Priority 2:** run **scale-matched** `ret0.95` reruns first (adjust c_fixed so mean alpha returns near B-current range, e.g. `3.1e-6` to `3.4e-6`), then consider multi-seed.
3. **Priority 3:** run W5/W10 (`windowed-r`) because temporal-aggregation effectiveness remains unanswered by this 4.30 set.

## Appendix: Control Metrics

| Run | alpha_mean | alpha_last | mean\|delta(alpha)\| | g_dot_positive mean | alpha_rate_limited ratio | r_hat mean |
|---|---:|---:|---:|---:|---:|---:|
| slowema_ret0.95 | 0.0000061702 | 0.0000048679 | 0.0000015353 | 0.506667 | 0.000000 | 0.0289972801 |
| alpharlim0.05 | 0.0000034164 | 0.0000025000 | 0.0000001553 | 0.573333 | 0.570000 | 0.0118456459 |
| slowema_ret0.95 + alpharlim0.05 | 0.0000062884 | 0.0000026436 | 0.0000002138 | 0.486667 | 0.463333 | 0.0268618241 |
| slowema_ret0.98 | 0.0000054541 | 0.0000045247 | 0.0000017073 | 0.559140 | 0.000000 | 0.0274106898 |
