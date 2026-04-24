# Proposed Experiments: Mitigating LR-Induced Failure Modes

This directory contains training scripts for four experimental directions designed to address the failure modes identified in our study of learning rate sensitivity in synchronous vs. asynchronous RL training.

## Background

Our baseline experiments revealed two distinct failure modes when using high learning rate ($10^{-5}$):

| Config | Failure Mode | Root Cause | Key Indicator |
|--------|-------------|------------|---------------|
| `sync_lr1e-5` | **Type I: Entropy Collapse** | Positive feedback loop destroys sampling diversity | Entropy drops to 0.07 |
| `async_partial_lr1e-5` | **Type II: Gradient Truncation** | Clip systematically discards high-information tokens | Clip fraction 10-100x elevated |

Both low-LR configs (`lr=1e-6`) train stably. The goal of these experiments is to find interventions that allow high-LR benefits (faster initial learning) while preventing the pathological dynamics.

---

## Direction 1: LR Schedule (`lr_schedule/`)

**Hypothesis**: Decaying LR over training captures early fast learning while preventing late-stage failure.

| Script | Base LR | Schedule | Warmup | Key Change |
|--------|---------|----------|--------|------------|
| `sync_cosine_decay_lr1e-5.sh` | 1e-5 | Cosine | 10 steps | `lr_scheduler_type=cosine` |
| `sync_warmup_decay_lr1e-5.sh` | 1e-5 | Cosine | 30 steps | + `lr_warmup_steps=30` |
| `async_partial_cosine_decay_lr1e-5.sh` | 1e-5 | Cosine | 10 steps | `lr_scheduler_type=cosine` |
| `async_partial_warmup_decay_lr1e-5.sh` | 1e-5 | Cosine | 30 steps | + `lr_warmup_steps=30` |

**Expected outcomes**:
- **Sync**: Should preserve the fast rise to ~0.48 accuracy (steps 0-20) while preventing the V-shaped collapse after step 150. By step 150, cosine decay will have reduced LR to ~0.5x peak, significantly slowing the entropy collapse feedback loop.
- **Async**: Reduced late-stage LR means slower $\pi_\theta$ divergence from $\pi_{\text{gen}}$, leading to lower clip fraction in the second half of training. Extended warmup (30 steps) is particularly important for async since early steps have the highest off-policy mismatch.

**What to watch**: Entropy trajectory (should decline slower than constant-LR baseline), clip fraction (async, should decrease in later stages), accuracy stability in steps 100-300.

---

## Direction 2: Entropy Regularization (`entropy_regularization/`)

**Hypothesis**: Adding entropy bonus directly counteracts the Type I failure loop.

| Script | Entropy Coeff | LR | Purpose |
|--------|---------------|-----|---------|
| `sync_entropy_bonus_lr1e-5.sh` | 0.01 | 1e-5 | Primary: break entropy collapse |
| `async_partial_entropy_bonus_lr1e-5.sh` | 0.01 | 1e-5 | Secondary: broader $\pi$ may keep $r_t$ closer to 1 |
| `sync_entropy_bonus_lr1e-6.sh` | 0.01 | 1e-6 | Control: effect on already-stable config |
| `async_partial_entropy_bonus_lr1e-6.sh` | 0.01 | 1e-6 | Control |

**Expected outcomes**:
- **`sync_lr1e-5` + entropy bonus**: This is the **most critical experiment**. If entropy collapse is truly the bottleneck, entropy bonus should rescue the accuracy curve. Expect entropy to stabilize above 0.2, preventing gradient death and enabling continued learning.
- **`async_lr1e-5` + entropy bonus**: Modest effect expected. Async failure is not entropy-driven, but broader $\pi$ may reduce $r_t$ deviation, slightly lowering clip fraction.
- **Controls (lr=1e-6)**: May provide marginal improvement by maintaining exploration longer, or may hurt by preventing useful specialization. The comparison is informative either way.

**Tuning guidance**: If entropy stays above 0.5 at step 300 with coeff=0.01, reduce to 0.005. If collapse still occurs, increase to 0.02.

**What to watch**: Entropy trajectory (primary), accuracy vs. entropy tradeoff (does higher entropy translate to better accuracy?), whether the model still converges or remains too diffuse.

---

## Direction 3: Adaptive Clip Range (`adaptive_clip/`)

**Hypothesis**: Tighter clip range limits per-step policy change, acting as implicit LR reduction.

| Script | Clip Range | LR | Key Change |
|--------|-----------|-----|------------|
| `sync_adaptive_clip_lr1e-5.sh` | [0.9, 1.1] | 1e-5 | `clip_ratio_low=0.1, clip_ratio_high=0.1` |
| `async_partial_adaptive_clip_lr1e-5.sh` | [0.9, 1.1] | 1e-5 | Same |

**Expected outcomes**:
- **Sync**: Tighter clip constrains the sampled actions' ratio more aggressively, slowing the rate at which $\pi(a^*)$ increases. This should delay (but may not prevent) entropy collapse. The fundamental limitation remains: unsampled actions are unconstrained regardless of clip range.
- **Async**: Nuanced effect. More tokens will hit the clip boundary (lower threshold), but the maximum update magnitude per token is reduced. Whether this helps or hurts depends on the balance between information loss (more clip) and update stability (smaller effective steps).

**What to watch**: Clip fraction (will increase with tighter clip — this is expected, not a problem), entropy trajectory (sync), accuracy stability, whether the behavior more closely resembles lr=1e-6 (which would confirm that clip range and LR have partially interchangeable effects).

---

## Direction 4: Fine-Grained LR Sweep (`lr_sweep/`)

**Hypothesis**: There exists a sharp phase transition between stable and pathological training, predictable from the positive feedback loop theory.

| Script | LR | Mode | Purpose |
|--------|-----|------|---------|
| `sync_lr3e-6.sh` | 3e-6 | Sync | Below predicted critical point |
| `sync_lr5e-6.sh` | 5e-6 | Sync | Near predicted critical point |
| `async_partial_lr3e-6.sh` | 3e-6 | Async | Async may have lower critical LR |
| `async_partial_lr5e-6.sh` | 5e-6 | Async | Near async critical point |

**Expected outcomes**:
- **`sync_lr3e-6`**: Likely stable. 3x increase from baseline should remain within stability boundary. Expect similar trajectory to `sync_lr1e-6` but ~3x faster improvement rate.
- **`sync_lr5e-6`**: This is the **most theoretically informative** experiment. If the phase transition is sharp (as predicted), this will either be clearly stable or clearly failing. A gradual degradation would suggest additional stabilizing mechanisms.
- **Async LR sweep**: The async critical LR may differ from sync due to the additional off-policy amplification factor. `async_lr5e-6` may show elevated clip fraction even if sync_lr5e-6 is stable.

**What to watch**: Entropy trajectory at step 50 (early divergence indicator), accuracy at step 300 vs. lr=1e-6 baseline, whether the transition is sharp or gradual, whether sync and async critical LRs differ.

---

## Running the Experiments

Each script is self-contained and follows the same pattern as the baseline scripts. To run:

```bash
# Example: run the sync cosine decay experiment
bash new_experiments/lr_schedule/sync_cosine_decay_lr1e-5.sh
```

Ensure the following environment variables are set if needed:
- `RESUME_MODE`: "disable" (default) for fresh training
- Hardware: 8 GPUs on a single node
- Conda environment: `verl2`

## Priority Order

If compute budget is limited, we recommend running experiments in this order:

1. **`sync_entropy_bonus_lr1e-5.sh`** — Highest expected impact, directly tests root cause
2. **`sync_lr5e-6.sh`** — Most theoretically informative, cheap (just LR change)
3. **`sync_cosine_decay_lr1e-5.sh`** — Practical intervention with broad applicability
4. **`async_partial_cosine_decay_lr1e-5.sh`** — Tests whether LR decay helps async
5. Remaining experiments as compute permits
