# Project Status

**Last updated**: 2026-04-20 (bug-fixed + experiment scripts added)

---

## Current logical thread (confirmed, no gaps)

### 1. Problem

Fixed LR fails in on-policy RL because the optimal learning rate is time-varying and decreasing:

$$\alpha^*(t) = \frac{r_t}{L}$$

As policy concentrates → $n_\text{eff}$ decreases → $r_t$ decreases → $\alpha^*(t)$ decreases.
Fixed LR eventually exceeds $2\alpha^*(t)$, causing expected-negative updates.

### 2. The improvement bound (full derivation in `main_modified.tex` Section 3)

$$\mathbb{E}[J(\theta_{t+1})] - J(\theta_t) \geq D_t \cdot \alpha \left[r_t - \frac{\alpha L}{2}\right]$$

Expected improvement > 0 iff $\alpha < 2r_t/L$. The safe-zone condition is determined entirely by $r_t$:

$$r_t = \frac{\|g\|^2}{\|g\|^2 + \sigma^2/n_\text{eff}} \in (0,1]$$

The individual values of $\|g\|^2$ and $\sigma^2/n_\text{eff}$ don't matter — only their ratio.

### 3. $r_t$ can be estimated: split-batch estimator

For any two independent unbiased estimators $\hat{g}_A$, $\hat{g}_B$ of $g_t$ (by independence):

$$\mathbb{E}[\hat{g}_A^\top \hat{g}_B] = \|g_t\|^2$$

Therefore:

$$\hat{r}_t = \frac{\hat{g}_A^\top \hat{g}_B}{\dfrac{\|\hat{g}_A\|^2 + \|\hat{g}_B\|^2}{2}}$$

**Only requirement**: both halves generated under the same $\theta_t$. Prompts can differ entirely.

### 4. Shape vs scale: what $r_t$ solves and what it doesn't

Tracking $r_t$ gives $\alpha(t) = C \cdot r_t$ where $C = \alpha(0)/r_0$.

The ratio $\alpha(t)/\alpha^*(t) = C \cdot L$ is **constant** — initial errors are preserved forever.
$r_t$ correctly captures the **shape** (how $\alpha^*(t)$ changes over time) but not the **scale** ($1/L$).

### 5. Proposed framework

$$\alpha_t = c_t \cdot \hat{r}_t$$

| Component | Role | Target |
|-----------|------|--------|
| $\hat{r}_t$ | feedforward, theory-grounded | shape of $\alpha^*(t)$ |
| $c_t$ | feedback-corrected scale | converge to $1/L$ |

**Why not directly estimate $L$**:
1. L is a global constant but training needs local effective curvature — it drifts as policy/sampling/advantage distribution change
2. Finite-difference $\hat{L}_t$ amplifies gradient noise; worst exactly when $n_\text{eff}$ is small
3. Lipschitz bounds are conservative → systematically suppresses $\alpha_t$

### 6. Feedback signal for $c_t$: held-out surrogate realization ratio $\phi_t$

Define using half-batch B (held out from the update step, used only for measurement):

$$p_t := \alpha_t \hat{g}_B^\top \hat{g}_A \quad \text{(predicted gain)}$$
$$a_t := \mathcal{L}_B(\theta_{t+1}) - \mathcal{L}_B(\theta_t) \quad \text{(actual gain, on held-out B)}$$
$$\phi_t := \frac{a_t}{p_t + \varepsilon} \quad \text{(realization ratio)}$$

**Why $\phi_t$ is the right signal**:
- $p_t$ is the first-order predicted improvement from step direction $\hat{g}_A$, magnitude $\alpha_t$
- $a_t$ is actual improvement on an independent batch B → unbiased surrogate (no gradient leakage)
- $\phi_t$ measures "how much of the predicted improvement was realized"

### 7. Target $\phi^* = 1/2$ (derived, not heuristic)

**4-step derivation** (from improvement bound):

1. From improvement bound maximized at $\alpha^* = r_t/L$:
$$f(\alpha) = \alpha \|g\|^2 \left(1 - \frac{\alpha L}{2 r_t}\right)$$

2. First-order approximation (valid near $\alpha^*(t)$):
$$\phi_t \approx 1 - \frac{\alpha_t L}{2 r_t}$$

3. Substitute $\alpha_t = c_t \cdot r_t$:
$$\phi_t \approx 1 - \frac{c_t L}{2} \quad \leftarrow r_t \text{ cancels exactly}$$

4. At $c_t = 1/L$ (target):
$$\phi^* = 1 - \frac{1}{2} = \frac{1}{2}$$

**Critical property**: $\phi^*$ does not depend on $r_t$. The scale controller is decoupled from the shape signal.

**Position encoding** (where $\beta = \alpha_t/\alpha^*(t) = c_t L$):

| $\beta$ | $\phi_t$ | Interpretation |
|---------|---------|----------------|
| 0.5 | 0.75 | Step too small |
| 1.0 | 0.5 | Optimal |
| 2.0 | 0 | At stability boundary |
| > 2.0 | < 0 | Diverging |

### 8. Controller update rule

$$c_{t+1} = c_t \cdot \exp\!\left(\eta_c \left(\bar{\phi}_t - \frac{1}{2}\right)\right)$$

where $\bar{\phi}_t$ is an EMA of recent $\phi_t$ values (smoothing out per-step noise).

- $\bar{\phi}_t > 1/2$: step is too conservative → increase $c_t$
- $\bar{\phi}_t < 1/2$: step is too aggressive → decrease $c_t$

**Three practical protections**:
1. Skip update when $|p_t|$ is too small (estimate unreliable, e.g. near convergence)
2. Clip or EMA-smooth $\phi_t$ before feeding into controller (prevent single-step spikes)
3. Hard bounds on $c_t$: $c_t \in [c_\min, c_\max]$ to avoid divergence during early training

---

## Engineering implementation (completed 2026-04-20)

### Files modified (all syntax-verified)

| File | Change |
|------|--------|
| `verl/workers/config/optimizer.py` | 8 new `signal_fraction_*` fields in `FSDPOptimizerConfig`; updated `assert` |
| `verl/workers/engine/fsdp/transformer_impl.py` | `SignalFractionLRScheduler` class |
| `verl/workers/fsdp_workers.py` | `signal_fraction` branch in scheduler builder; attach to actor; metrics extraction |
| `verl/workers/actor/dp_actor.py` | Redirect in `update_policy`; `_update_policy_signal_fraction` method |

**To enable**: set `actor_rollout_ref.actor.optim.lr_scheduler_type: signal_fraction` in experiment config.

### Algorithm flow (per DP rank, every step)

```
Non-calibration step:
1. Split local batch at group level:
     A1 = first 50% of groups,  A2 = last 50% of groups
2. zero_grad → backward(A1, scale=1/n_mb_A1) → save grad_A1
3. zero_grad → backward(A2, scale=1/n_mb_A2)  [p.grad = grad_A2]
4. local_stats = [Σ g1·g2,  Σ ||g1||²,  Σ ||g2||²]  per FSDP shard
   all_reduce(local_stats)  →  r̂_t = stats[0] / ((stats[1]+stats[2])/2)
   r̂_t = clamp(r̂_t, r_min=0.01, 1.0)
5. p.grad = (grad_A1 + p.grad) / 2   [ĝ_upd]
6. α_t = c_t · r̂_t  →  set on optimizer
7. _optimizer_step()

Calibration step (every K=5 steps, additionally):
   [split batch: A1=37.5%, A2=37.5%, C=25%]
6. save g_upd = [p.grad.clone() for all params]
7. zero_grad → backward(C, scale=1/n_mb_C)  →  save grad_C,  L_C_old
8. p_t = α_t · Σ(grad_C · g_upd),  all_reduce
9. restore p.grad = g_upd  →  _optimizer_step()
10. forward-only on C at θ_{t+1}  →  L_C_new
11. a_t = L_C_old - L_C_new    [improvement = loss decrease]
    φ_t = a_t / (p_t + ε)
12. if |p_t| > p_min  and  dot(grad_C, g_upd) > 0:
        φ̄_t ← EMA(φ̄_t, φ_t,  β=0.9)
        c_t ← clip(c_t · exp(η_c · (φ̄_t − 0.5)),  c_min, c_max)
```

### FSDP gradient dot product correctness

After FSDP backward, `p.grad` on each GPU = globally-reduced shard gradient.
Local dot products summed via `dist.all_reduce` give the global dot product.
Scale factors cancel in r̂_t ratio → r̂_t is FSDP-correct. ✓

### φ* = 1/2 preserved

ĝ_upd = (ĝ_A1 + ĝ_A2)/2 is unbiased for g_t → same derivation applies → φ* = 1/2. ✓

### Default hyperparameters for first experiment

| Param | Value | Rationale |
|-------|-------|-----------|
| `lr` (= c_init) | 1e-5 | = base_lr; controller drives c_t downward to ~1/L |
| `signal_fraction_eta_c` | 0.1 | ~30 calibration steps to go from 1e-5 → 6e-6 |
| `signal_fraction_calib_freq` | 5 | c_t updated every 5 steps |
| `signal_fraction_calib_frac` | 0.25 | 25% of batch as C on calibration steps |
| `signal_fraction_r_min` | 0.01 | Lower clamp: prevents near-zero LR from noise |
| `signal_fraction_phi_ema_beta` | 0.9 | Smooth φ_t over ~10 calibration steps |

### SwanLab metrics

`actor/r_hat`, `actor/r_hat_raw`, `actor/c_t`, `actor/alpha_t`, `actor/phi_bar`,
`actor/phi_t` (calibration only), `actor/p_t`, `actor/a_t`, `actor/g_A1_dot_A2`,
`actor/is_calibration_step`

### Design choices and known limitations

1. **C source: Method B** — held-out from existing batch (no trainer changes). Original design (Method A) intended extra C rollouts from trainer; can upgrade later.
2. **No ppo_epochs > 1**: signal-fraction is on-policy single-pass. Correct for the theory.
3. **No KL loss / entropy bonus** in `_update_policy_signal_fraction`. Pure policy loss only.
4. **Small C on low DP ranks**: with 4 DP ranks × 8 local groups and calib_frac=0.25, C = 2 groups per rank. May be noisy; monitor `actor/p_t` to check.
5. **`ppo_micro_batch_size_per_gpu` not required**: `use_dynamic_bsz=True` → field is None; `_make_mb_list` uses `prepare_dynamic_batch` instead.

### Bugs fixed (2026-04-20, post-implementation review)

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | ĝ_upd build (line ~736) | `p.grad.data = x` crashes when `p.grad is None` | `p.grad = x` |
| 2 | calibration restore (line ~777) | same `p.grad.data = gu` crash | `p.grad = gu` |
| 3 | data_C creation | `calib_groups` clamped to 0 but empty DataProto assigned; passes `is not None` → backward on empty batch | `if is_calibration and calib_groups > 0: data_C = ...` |
| 4 | `set_lr_from_rt` warmup check | `step_count <= N` → warmup runs 1 extra step | `step_count < N` |

Also fixed: `use_dynamic_bsz=True` with `micro_bsz=None` — 5 direct `.split(None)` calls crash + OOM risk from single-pass on full partition. Replaced with `_make_mb_list` helper using `prepare_dynamic_batch`.

### Experiment scripts

| Script | lr / c_init | c_max | SwanLab name |
|--------|-------------|-------|--------------|
| `new_experiments/signal_fraction_lr/sync_signal_fraction_lr1e-5.sh` | 1e-5 | 1e-2 | `deepseek1.5b_sync_8gpu_sigfrac_lr1e-5` |
| `new_experiments/signal_fraction_lr/sync_signal_fraction_lr1e-6.sh` | 1e-6 | 1e-3 | `deepseek1.5b_sync_8gpu_sigfrac_lr1e-6` |

Both in SwanLab project `deepseek1.5b_lr` (same as entropy_adaptive baseline) for direct comparison.

---

## Files

| File | Contents |
|------|----------|
| `theory_derivation.tex` | Full formal derivations (ground truth), all steps proved |
| `main_modified.tex` | Reconstructed paper: Section 3 = full math derivation + shape/scale; Section 8 = proposed framework + φ_t + φ*=1/2 derivation + controller |
| `core_problem_to_algorithm_v2.tex` | Clean logical chain from problem statement to complete algorithm (17 sections) |
| `engineering_design_adaptive_lr_grpo_en.tex` | Engineering design doc with pseudocode and GRPO constraints |
| `discussion_log.md` | Session logs from 2026-04-19 |
| `PROJECT_STATUS.md` | This file |
