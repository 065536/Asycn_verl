---
name: Engineering Implementation — Signal-Fraction Adaptive LR
description: Concrete implementation of α_t = c_t·r̂_t; code status as of 2026-04-22; p_min guard removed; phi_t degenerate-loss bug fixed
type: project
---

## Core abstraction

Three independent data streams under the same θ_t:

- **A1, A2**: both participate in update gradient AND r̂_t estimation
- **C**: held-out only on calibration steps, never touches α_t or θ_{t+1}

**Key invariant**: C must not influence α_t. Specifically:
C → r̂_t (forbidden), C → ĝ_upd (forbidden), C → θ_{t+1} (forbidden)

---

## Implementation status (2026-04-22 第二轮): phi_t 退化为零 bug 修复

**Bug：_forward_loss_nograd 使用自指 log_probs，导致 a_t ≡ 0（2026-04-22 诊断 + 修复）**

4.22 第一轮修复之后跑的实验（~139 步）显示 phi_t = 0 全程，phi_bar 从 0.5 单调下降至 0.16，c_t 被持续驱动向下。根因：

- `_backward_split`（算 L_C_old at θ_t）：`old_log_prob = log_prob.detach()` → ratio=1 → L = -mean(A_C)
- `_forward_loss_nograd`（算 L_C_new at θ_{t+1}）：同样使用 `old_log_prob = log_prob.detach()`（θ_{t+1} 时刻的）→ ratio=1 → L = -mean(A_C)

两者都等于 -mean(A_C)（A_C 在 θ 更新前已固定），**与 θ 无关**，所以 a_t = L_C_old - L_C_new ≡ 0 恒成立。

**根本问题**：4.22 第一轮修复解决了"两个函数用不同基准"的问题，但引入了"两个函数都退化成常数"的新问题。phi_t = 0 < 0.5 → phi_bar → 0 → c_t 持续下降，c_t controller 方向完全错误。

**正确设计**：L_C_new 必须真正依赖 θ_{t+1}。方法：用 θ_t 时的 log_probs 作为 old_log_probs 传给 `_forward_loss_nograd`，使 ratio = π_{θ_{t+1}} / π_{θ_t}，L_C_new 才是 θ_{t+1} 的函数。

**修复**：`_backward_split` 新增 `collect_log_probs=True` 参数，在 C backward 时保存各 micro-batch 的 θ_t log_probs；`_forward_loss_nograd` 新增 `old_log_probs_list` 参数接收并使用它们。

变更文件：
- `verl/workers/actor/dp_actor.py`：
  - `_backward_split` 加 `collect_log_probs=False` 参数，返回 `(loss, saved_log_probs)` 或 `loss`
  - `_forward_loss_nograd` 加 `old_log_probs_list=None` 参数，有值时用 θ_t log_probs 替代 `log_prob.detach()`
  - 调用处：`L_C_old, log_probs_C_theta_t = _backward_split(data_C, collect_log_probs=True)`
  - 调用处：`L_C_new = _forward_loss_nograd(data_C, old_log_probs_list=log_probs_C_theta_t)`

修复后：L_C_old = -mean(A_C)（ratio=1 at θ_t），L_C_new = -mean(clip(π_{θ_{t+1}}/π_{θ_t}) × A_C)，a_t ≠ 0，φ_t 回到 O(1) 量级。

---

## Implementation status (2026-04-22 第一轮): phi_t 量级异常 bug 修复（已被上方 bug 取代）

**Bug：_forward_loss_nograd 与 _backward_split 使用不一致的 loss function**

phi_t ≈ 1e5。根因：`_forward_loss_nograd` 误用 rollout 的 `old_log_probs`（ratio = π_{θ_{t+1}}/π_rollout），`_backward_split` 用 ratio=1。差值放大 5 个数量级。

**修复**：`_forward_loss_nograd` 改为 `old_log_prob = log_prob.detach()`（ratio=1）。
→ 此修复本身正确，但导致了上方描述的退化 bug，需要进一步修复。

---

## Implementation status (2026-04-21): p_min removed — c_t guard now only g_dot > 0

**Bug：p_min 绝对阈值破坏 scale-invariance（2026-04-21 诊断 + 修复）**

4.21 实验数据显示 c_t 全程只有 handoff 初始化值（2e-4 / 2e-5），从未更新。根因：
- 实际 p_t = α_t × g_dot ≈ 1e-6 × 7e-6 = 7e-12
- p_min = 1e-8（固定绝对阈值）→ p_t 比 p_min 小 4 个量级 → 300/300 步全跳过

**根本问题**：p_min 是"缺乏归一化的绝对阈值"，与系统尺度（LR、梯度范数、维度）耦合，不具备 scale-invariance。不同 base LR 的实验对应完全不同的 p_t 量级，但 p_min 不随之共变。这使 c_t controller 被人为绑回了初始 LR，破坏了 c → 1/L 的 factorization 含义。

**修复**：删除 p_min guard，只保留方向 guard（`g_dot > 0`）。

变更文件：
- `verl/workers/actor/dp_actor.py`：删除 p_min 变量读取；`if abs(p_t_val) > p_min and g_dot > 0` → `if g_dot > 0`
- `verl/workers/engine/fsdp/transformer_impl.py`：`__init__` 删除 p_min 参数和 `self.p_min`；`update_ct` 删除 `abs(p_t) < self.p_min or` 条件
- `verl/workers/fsdp_workers.py`：删除 `p_min=` 传参
- 两个实验脚本：删除 `signal_fraction_p_min=1e-8` 行

### Files modified

| File | Change |
|------|--------|
| `verl/workers/config/optimizer.py` | Added 11 signal_fraction fields to FSDPOptimizerConfig; updated assert |
| `verl/workers/engine/fsdp/transformer_impl.py` | Full rewrite of `SignalFractionLRScheduler` with warmup EMA, handoff init, asymmetric EMA, fast-drop state machine, validity guard |
| `verl/workers/fsdp_workers.py` | signal_fraction branch passes all new params to scheduler; attach scheduler to actor; metrics extraction |
| `verl/workers/actor/dp_actor.py` | `_update_policy_signal_fraction`: g_rms_t computation, n_params global all_reduce, `update_r_and_set_lr` call, `get_signal_fraction_metrics()` |

To enable: set `actor_rollout_ref.actor.optim.lr_scheduler_type: signal_fraction` in config.

---

## Key design choices in current implementation

### C source: Method B (held-out from existing batch)
- Non-calibration steps: full batch → A1 (50%) + A2 (50%)
- Calibration steps: A1 + A2 get (1 - calib_frac) of groups; C gets calib_frac (default 0.25)
- No trainer changes needed; zero extra rollout generation
- Trade-off: slightly smaller update batch on calibration steps (every 5 steps → ~5% average reduction)
- **Note**: original design (PROJECT_STATUS.md) specified generating *extra* C rollouts (Method A). Chose Method B for simplicity of first experiment. Can upgrade to Method A later.

### FSDP gradient dot product correctness
After FSDP backward, `p.grad` on each GPU = globally-reduced gradient shard. Local dot products summed via `dist.all_reduce` give the global dot product. Scale factors cancel in r̂_t ratio → r̂_t is FSDP-correct without additional normalization.

### g_rms_t computation (Bug 1 fix)
```python
# In the ĝ_upd build loop, also accumulate shard-local ||ĝ_upd||²
upd_norm_sq_local += g_upd.to(torch.float64).norm().item() ** 2

# All-reduce to get global squared norm
upd_norm_sq_t = torch.tensor(upd_norm_sq_local, ..., dtype=torch.float64)
torch.distributed.all_reduce(upd_norm_sq_t)

# n_params: cached after first all_reduce to avoid repeated comms
if not hasattr(self, "_sf_n_params_global"):
    n_params_t = torch.tensor(local_count, ..., dtype=torch.int64)
    torch.distributed.all_reduce(n_params_t)
    self._sf_n_params_global = int(n_params_t.item())

g_rms_t = sqrt(upd_norm_sq_t / n_params_global)
```

### Calibration flow (ordering)
```
1. Backward A1 → save grad_A1
2. Backward A2 → p.grad = grad_A2; compute r̂_t stats (g_dot, d_t, r_hat_raw)
3. Build ĝ_upd = (grad_A1 + grad_A2) / 2 → write into p.grad; accumulate upd_norm_sq
4. all_reduce upd_norm_sq; compute g_rms_t
5. sched.update_r_and_set_lr(g_dot, d_t, g_rms_t, r_hat_raw)  → sets α_t on optimizer
[CALIBRATION ONLY:]
6. Save g_upd = [p.grad.clone() for all params]
7. Backward on C at θ_t → grad_C, L_C_old (from loss value)
8. Compute p_t = α_t · Σ(grad_C · g_upd)  [all_reduce]
9. Restore g_upd → p.grad
10. _optimizer_step() → θ_{t+1}
11. Forward-only on C at θ_{t+1} → L_C_new
12. a_t = L_C_old - L_C_new;  φ_t = a_t / (p_t + ε)
13. sched.update_ct(φ_t, p_t, g_dot_C_upd)  if |p_t| > p_min and g_dot_C_upd > 0
```

---

## Default hyperparameters

| Param | Value | Meaning |
|-------|-------|---------|
| `lr` (= c_init) | 1e-5 | Initial scale factor; controller adjusts c_T at handoff |
| `signal_fraction_eta_c` | 0.1 | Controller step size |
| `signal_fraction_calib_freq` | 5 | c_t updated every 5 steps |
| `signal_fraction_calib_frac` | 0.25 | 25% of batch held out as C on calibration steps |
| `signal_fraction_r_min` | 0.01 | r_ctrl safety floor (replaces old r̂_t clamp) |
| `signal_fraction_phi_ema_beta` | 0.9 | φ_t EMA smoothing |
| ~~`signal_fraction_p_min`~~ | ~~1e-8~~ | **已删除（2026-04-21）** — 绝对阈值破坏 scale-invariance |
| `signal_fraction_c_min` | 1e-8 | Hard floor on c_t |
| `signal_fraction_c_max` | 1e-2 | Hard ceiling on c_t |
| `signal_fraction_r_boot` | 0.05 | warmup EMA initial value / c_T denominator floor |
| `signal_fraction_r_ema_beta_sym` | 0.3 | Warmup symmetric EMA β |
| `signal_fraction_r_ema_beta_down` | 0.5 | Post-handoff fast downward EMA β |
| `signal_fraction_r_ema_beta_up` | 0.1 | Post-handoff slow upward EMA β |
| `signal_fraction_g_rms_ema_beta` | 0.05 | Background g_rms baseline EMA β (slow, stable) |
| `signal_fraction_d_min_abs` | 1e-30 | Absolute floor on d_t (prevent NaN) |
| `signal_fraction_tau_rms` | 0.05 | Relative RMS validity threshold |
| `signal_fraction_fast_drop_rho` | 0.7 | Fast-drop trigger threshold |
| `signal_fraction_cooldown_steps` | 5 | Cooldown duration in training steps (not calibration steps) |
| `signal_fraction_handoff_steps` | 10 | Handoff interpolation steps |

---

## SwanLab metrics logged (Bug 1 fix adds new entries)

| Metric | Description |
|--------|-------------|
| `actor/r_hat_raw` | r̂_t before validity/EMA |
| `actor/r_hat` | r̄_t (EMA tracked value) |
| `actor/r_ctrl` | r_t_ctrl used in α_t = c_t · r_ctrl |
| `actor/c_t` | Current scale factor |
| `actor/alpha_t` | Actual LR applied |
| `actor/phi_bar` | EMA of φ_t |
| `actor/phi_t` | Per-calibration-step realization ratio (when available) |
| `actor/p_t` | Predicted gain (calibration steps only) |
| `actor/a_t` | Actual gain on C (calibration steps only) |
| `actor/g_rms` | Per-param gradient RMS of ĝ_upd (global) |
| `actor/g_rms_ema` | Background g_rms EMA baseline |
| `actor/cooldown` | 1.0 if in cooldown (c_t frozen), else 0 |
| `actor/in_handoff` | 1.0 during handoff interpolation, else 0 |
| `actor/g_A1_dot_A2` | Raw dot product (numerator of r̂_t) |
| `actor/g_norm_A1_sq` | \|\|ĝ_A1\|\|² |
| `actor/g_norm_A2_sq` | \|\|ĝ_A2\|\|² |
| `actor/is_calibration_step` | 1.0 on calibration steps |

---

## Algorithm per step (current implementation)

### Every step
```
1. Split local batch into A1 (50%) and A2 (50%) at group level
2. A1 → backward → ĝ_A1;  A2 → backward → ĝ_A2
3. r̂_raw = dot(ĝ_A1, ĝ_A2) / d_t;  g_rms_t = sqrt(||ĝ_upd||² / n_params_global)
4. sched.update_r_and_set_lr(g_dot, d_t, g_rms_t, r_hat_raw):
     - update g_rms_ema (unconditional)
     - validity check (3 conditions)
     - warmup EMA or asymmetric EMA
     - handoff init (first post-warmup step)
     - fast-drop detection → cooldown counter
     - compute r_ctrl, α_t (with handoff interpolation)
     - set optimizer LR; decrement cooldown counter
5. ĝ_upd = (ĝ_A1 + ĝ_A2) / 2;  θ_{t+1} = θ_t - α_t · ĝ_upd
```

### Calibration step (every K=5 steps, additionally)
```
6. Hold out C (25% of groups) from A1/A2
7. Backward on C at θ_t → ĝ_C, L_C(θ_t)
8. p_t = α_t · ĝ_C^T ĝ_upd;  [do update];  forward on C at θ_{t+1} → L_C(θ_{t+1})
9. a_t = L_C(θ_t) - L_C(θ_{t+1})
10. φ_t = a_t / (p_t + ε)
11. if ĝ_C^T ĝ_upd > 0 and not in_cooldown:
        c_{t+1} = clip(c_t · exp(η_c · (φ̄_t − 1/2)), c_min, c_max)
```

---

## GRPO constraint

Split at prompt-group level (not response level). Each half independently normalizes advantages within its own groups. With rollout_n=8 and local_bsz=64 (4 DP ranks): 8 groups per rank → A1=4, A2=4 (non-calibration) or A1=3, A2=3, C=2 (calibration, calib_frac=0.25).

**Edge case guard**: `calib_groups = max(0, min(calib_groups, n_groups - 2))` prevents negative calib_groups when local_bsz ≤ rollout_n (e.g., at large scale with many GPUs).

---

## φ* = 1/2 preserved

ĝ_upd = (ĝ_A1 + ĝ_A2)/2 is unbiased for g_t. Using EMA proxy r̄_t: φ_t ≈ 1 - c_t·L/2·(r_t/r̄_t). Quasi-statically r̄_t ≈ r_t → φ* = 1/2 ↔ c_t = 1/L. ✓

---

## Known limitations / TODO

1. **No ppo_epochs > 1 support**: Signal-fraction uses a single pass over the batch (no PPO re-use). Correct for the theory (on-policy) but different from standard GRPO setup.
2. **No KL loss / entropy bonus**: `_update_policy_signal_fraction` uses pure policy loss. Add later if needed.
3. **Trainer not modified**: No changes to `ray_trainer.py`. Method B (held-out from batch) was chosen to avoid trainer complexity. Method A (extra rollouts) would improve C quality.
4. **`step()` called after optimizer step** (standard PyTorch convention): warmup effectively runs `num_warmup_steps + 1` optimizer steps, with the last step at full base_lr. Off-by-one is harmless.

---

## Sign-gate mode（2026-04-23 新增）

Phase 2 三角对照引入 sign-gate 参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `signal_fraction_sign_gate_gamma` | `Optional[float]` | `None` | None → 连续 r-shaping；1.0 → 常数 LR（M）；0.5 → 两级门控（A） |
| `signal_fraction_sign_gate_alpha_plus` | `Optional[float]` | `None` | 若设定，handoff 时 c_T = alpha_plus/r_ref，解耦 warmup 目标与 post-handoff 全速 LR |

**新增 metric**：`actor/g_dot_positive`（`dp_actor.py:829`），每步 0/1，用于核对实际 p(g_dot>0) 分布。

**Bug 16（2026-04-23 修复）**：两个新参数漏加 `FSDPOptimizerConfig`，Hydra 实例化时报 `unexpected keyword argument`。修复：在 `verl/workers/config/optimizer.py` 的 `FSDPOptimizerConfig` 末尾（`signal_fraction_handoff_steps` 之后）添加两个 `Optional[float] = None` 字段。
