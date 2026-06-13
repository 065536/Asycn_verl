---
name: Engineering Implementation — RL Gradient Diagnostics + Signal-Fraction History
description: 6.12 Pos/neg gradient decomposition (g⁺ vs g⁻ at lm_head)；6.10 Exact lm_head gradient norm；6.8b Gradient preconditioning diagnostic；历史：A²Q reweighting, SQ-LR, noise decomposition
type: project
---

## Pos/Neg Gradient Decomposition (2026-06-12)

**目的**：将 batch gradient 按 advantage sign 分解为 g⁺ 和 g⁻，测量谁主导真实 update。在 lm_head 层精确计算。

**核心公式**：对每个 response b，lm_head score function gradient:

```
G_b = Σ_t mask_t · (e_{y_t} - π_t) h_tᵀ,  π_t = softmax(logits_t / τ)
```

按 advantage sign 累加:
```
g⁺ = Σ_{A_b > 0} A_b · G_b    (positive consolidation)
g⁻ = Σ_{A_b < 0} A_b · G_b    (negative correction)
```

**实现**：

| File | Change |
|---|---|
| `verl/trainer/ppo/lm_head_grad_norm.py` | +`compute_single_response_lm_head_grad(h_b, logits_b, y_b, chunk)` → (V,d) float32 |
| `verl/workers/actor/dp_actor.py` | +`compute_pos_neg_grad_decomposition(data, max_responses)` |
| `verl/workers/actor/dp_actor.py` | `update_policy()` 集成：训练前 eval-mode 诊断 |
| `verl/workers/config/actor.py` | +3 config fields |

**`compute_single_response_lm_head_grad` 细节**：
- 输入: (T_valid, d) hidden states, (T_valid, V) logits (已除 temperature), (T_valid,) token ids
- 按 token_chunk_size=256 分块避免 (T, V) 全量 softmax
- 每 chunk: π = softmax(logits_chunk) → scatter_add(-1 at y) → grad -= pi.T @ h
- 输出: (V, d) float32 gradient matrix

**`compute_pos_neg_grad_decomposition` 细节**：
- 在 prompt-group 级别子采样（保证同 prompt 的 K 个 response 完整，便于按 success rate 分桶）
- Micro-batch size = 2, forward with `output_hidden_states=True`
- 4 个 (V, d) float32 accumulator: g⁺_low, g⁻_low, g⁺_high, g⁻_high (按 prompt success rate < 0.5 / ≥ 0.5 分桶)
- Global g⁺/g⁻ = sum of bucket accumulators
- Memory: 4 × V × d × 4 bytes ≈ 3.7GB (V=151936, d=1536 for Qwen-1.5B)

**配置**：

```yaml
actor:
  calculate_pos_neg_grad_decomp: true
  pos_neg_grad_decomp_freq: 5
  pos_neg_grad_decomp_max_responses: 64
```

**Metrics** (prefix `grad_decomp/`):
- Global: `g_pos_norm`, `g_neg_norm`, `g_total_norm`, `cos_pos_neg`, `pos_neg_norm_ratio`, `n_pos`, `n_neg`, `n_zero`
- Per bucket: `high_p/g_pos_norm`, `high_p/g_neg_norm`, `high_p/cos_pos_neg`, ... (同理 `low_p/`)
- Summary: `n_prompts_sampled`, `n_prompts_mixed`, `frac_uninformative`

**时序位置**：在 `update_policy()` 开头、训练循环之前运行。此时 advantages 已可用，模型参数尚未被本步 update。对 signal_fraction 和 standard 两条路径均有效。

---

## Exact lm_head Gradient Norm (2026-06-10 — replaces A²Q proxy)

**问题**：A²Q 分解假设不同 token 的 score function 梯度在参数空间正交 (||Σ_t J_t||² = Σ_t ||J_t||²)，丢掉了 cross terms。实际上所有 token 共享参数，cross terms 可以很大。

**方案**：精确计算 lm_head 层的 per-response gradient norm ||∂S_b/∂W||²_F，使用 Gram 矩阵公式：

```
||∂S_b/∂W||²_F = Σ_{t,s} K_δ[t,s] · K_h[t,s]
K_h[t,s] = h_t · h_s          (hidden-space Gram)
K_δ[t,s] = 1_{y_t=y_s} - π_s[y_t] - π_t[y_s] + π_t·π_s  (vocab-space Gram of error vectors)
```

K_δ 的对角线恢复 q_per_token = 1 - 2π_t[y_t] + Σ_v π_t[v]²，所以 A²Q 等于 A²·trace(K_h ⊙ diag(K_δ))。

**不需要 lm_head 权重**：公式只用 hidden states 和 softmax distributions (from logits)，在 FSDP 下不需要 unsharding。

**验证**：analytical 公式与 PyTorch autograd 精确一致 (diff < 1e-6)。chunked 和 non-chunked 版本一致。

**文件变更**：

| File | Change |
|---|---|
| `verl/trainer/ppo/lm_head_grad_norm.py` | 新文件：core computation (`compute_response_lm_head_grad_norms`) |
| `verl/workers/actor/dp_actor.py` | +`compute_lm_head_grad_norms()` 方法 + 集成到 `compute_log_prob` |
| `verl/workers/config/actor.py` | +`calculate_lm_head_grad_norm`, `lm_head_grad_norm_freq`, `lm_head_grad_norm_max_responses` |
| `verl/trainer/ppo/metric_utils.py` | +exact_grad decomposition metrics (between/within, cross_term_ratio, corr) |
| `verl/trainer/ppo/ray_trainer.py` | +`global_step` in meta_info before compute_log_prob |

**配置**：

```yaml
actor:
  calculate_lm_head_grad_norm: true
  lm_head_grad_norm_freq: 5        # 每 5 步做一次
  lm_head_grad_norm_max_responses: 64  # 子采样 64 个 response
  calculate_sum_pi_squared: true   # 仍需 sum_pi_squared 用于 A²Q 对比
```

**新 metrics**（前缀 `noise_decomp/exact_grad/`）：

| Metric | Description |
|---|---|
| `n_sampled` | 本步采样的 response 数 |
| `mean_gnorm_sq` | E[||∂S/∂W||²] |
| `std_gnorm_sq` | Std[||∂S/∂W||²] |
| `mean_h_exact` | E[A² · ||∂S/∂W||²]（exact gradient energy） |
| `cross_term_ratio_mean` | E[||∂S/∂W||² / Q]（>1 说明 cross terms 正贡献，<1 说明互相抵消） |
| `corr_gnorm_q` | Corr(||∂S/∂W||², Q)（exact norm 与 A²Q proxy 的相关性） |
| `gnorm/between_var`, `within_var`, `between_frac`, `within_frac` | ||∂S/∂W||² 的 prompt 间/内方差分解 |
| `h_exact/between_var`, etc. | A²·||∂S/∂W||² 的方差分解 |

**限制**：

- 仅支持 `use_fused_kernels=False`（fused 路径不返回 logits）
- Peak memory: (T, V) softmax matrix per response (~800MB for T=2000, V=100K in float32)。chunked 模式用 `token_chunk_size=512` 降至 ~200MB
- 子采样引入采样噪声，需多步平均

## Gradient Preconditioning Diagnostic (2026-06-08b — P0 新增)

**目的**：诊断 Adam D_t 是否改变 gradient 有效方向，使其偏离 positive consolidation。
纯诊断脚本，不改训练代码，不跑训练。

### 实现方案

新脚本 `exp_data/adam_direction_diagnostic.py`（待实现）：

**输入**：
- Step 200/250 FSDP checkpoint（含 optimizer state: exp_avg, exp_avg_sq）
- 训练 batch（含 boundary prompts, 从 replay buffer 或在线采样）
- Per-problem eval 结果（用于确定 boundary/consolidated/dead bucket）

**计算流程**：
1. 加载 checkpoint → 提取 m_t (exp_avg), v_t (exp_avg_sq) per param
2. Forward + backward on batch → 得到 token-level gradients
3. 按 advantage_sign × problem_bucket 聚合：
   - g⁺_boundary = Σ_{A>0, boundary} A·∇logπ
   - g⁻_boundary = Σ_{A<0, boundary} A·∇logπ
   - （flatten 为全参数向量）
4. 计算 D_t = 1/(√v_t + ε), u_t = D_t * m_t
5. 计算所有诊断指标（见 algorithm_design.md diagnostic framework）

**关键工程约束**：
- FSDP: 需要 full gather 或 per-shard 计算 + all_reduce inner products
- 内存: g⁺/g⁻ 各是完整参数维度，需 float32；1.5B 模型 ≈ 2×6GB
- 方案 A: 用 FSDP `summon_full_params` + 逐层计算 inner product（内存友好）
- 方案 B: 单 GPU 加载 merged checkpoint + 单卡 backward（简单但需大显存）

**输出**：
```
cos(m, g⁺_boundary), cos(Dm, g⁺_boundary), Δ_dir⁺
cos(m, g⁻_boundary), cos(Dm, g⁻_boundary), Δ_dir⁻
κ(g⁺), κ(g⁻), κ_ratio = κ(g⁺)/κ(g⁻)
d̄(g⁺), d̄(g⁻)
‖g⁺‖, ‖g⁻‖, ‖m‖, ‖u‖
```

Per bucket × per batch → mean ± std 矩阵。

### Counterfactual Variants (同一脚本内)

| Variant | D' 构造 | cos(D'·m, g⁺) |
|---|---|---|
| Actual Adam | D_t = 1/(√v_t + ε) | baseline |
| Flatten (= raw m direction) | D_flat = scalar · I | = cos(m, g⁺) |
| Normalize | D̃ = D / mean(D) | 去全局 scale |
| Cap q95 | √v_ℓ ← min(√v_ℓ, q95) | 测试极端坐标影响 |

### 状态

- [ ] 脚本框架
- [ ] 单 GPU merged checkpoint 路径
- [ ] Per-shard FSDP 路径
- [ ] 在 step 200 checkpoint 上跑
- [ ] 多 batch 稳定性验证

---

## Adam Diagnostics + v-scale Continuation (2026-06-08)

**注意 (6.8b 修正)**：v*=0.1 continuation 从 P0 降级为 P2。原因：uniform v-scale 不改变 u_t 方向，等价于 LR × √10。只有 P0 gradient diagnostic 确认路径 B（cos(Dm,g⁺) < cos(m,g⁺)）后，才值得跑非 uniform continuation（如 cap-D）。

### Adam State Diagnostic Logging (P0)

`verl/workers/actor/dp_actor.py` — 新增 `_compute_adam_diagnostics()` 方法。

在 `optimizer.step()` 后、`zero_grad()` 前调用，读取 optimizer state 计算：

| Metric | 计算方式 | 用途 |
|---|---|---|
| `adam/v_rms` | RMS(√v) across all params | Denominator 整体大小 |
| `adam/v_mean` | mean(v) | Denominator 均值 |
| `adam/v_max` | max(v) | 极端 denominator |
| `adam/m_rms` | RMS(m) | Numerator 整体大小 |
| `adam/update_norm` | ‖m/(√v+ε)‖ | Preconditioned update 大小 |
| `adam/raw_grad_norm` | ‖g‖ post-clip | 原始梯度大小 |
| `adam/update_to_grad_ratio` | update_norm / grad_norm | 关键：下降说明 denominator 在压步长 |
| `adam/eta_eff` | ⟨u, g⟩ / ‖g‖² | 在梯度方向的有效 LR |
| `adam/cos_update_grad` | cos(u, g) | Update 与 gradient 对齐度 |

零额外 forward/backward。所有统计量通过 all_reduce 跨 FSDP shard 汇总。
已集成到标准 update_policy 路径 (line ~1206) 和 signal-fraction 路径 (line ~923)。

### Adam v-scale / beta2 Override (P1)

**Config** (`verl/workers/config/optimizer.py`):
- `adam_v_scale: Optional[float] = None` — checkpoint load 后对 exp_avg_sq 乘以此因子
- `adam_override_beta2: Optional[float] = None` — 切换 β₂ 并做 bias-correction rescale

**执行点** (`verl/workers/fsdp_workers.py` → `load_checkpoint`):
- 在 `checkpoint_manager.load_checkpoint()` 返回后立即执行
- v_scale: 遍历所有 param state, `state["exp_avg_sq"].mul_(v_scale)`
- beta2 override: 计算 `rescale = (1-β₂_new^t) / (1-β₂_old^t)`，先 rescale state，再修改 param_group betas

### Continuation 实验脚本

`exp_data/run_adam_continuation.sh` — 模板脚本
`new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh` — 新增 `${EXTRA_HYDRA:-}` 支持

Hydra override 方式：
```
++actor_rollout_ref.actor.optim.adam_v_scale=0.1
++actor_rollout_ref.actor.optim.adam_override_beta2=0.99
```

### Per-Problem Eval 脚本

`exp_data/per_problem_eval.py` — 独立评测脚本：
- 输入：base model + FSDP checkpoints + AIME parquet
- 自动 merge FSDP shards → HF 格式
- 用 vLLM 生成 N samples/题
- 输出：per-problem accuracy table, category classification, answer distribution, NPZ arrays

`exp_data/run_per_problem_eval.sh` — 便捷启动封装

---

## A²Q Hierarchical Gradient Reweighting (2026-06-02)

### Bug Fix: Gini Sign (2026-06-03)

`_gini()` used `(2*ranks - n - 1)` which is the ascending-sort formula, but data is sorted descending.
Fix: `(n + 1 - 2*ranks)`. Now Gini ∈ [0, 1] as expected.

### New Energy-Weighted Metrics (2026-06-03)

Added to `apply_a2q_reweighting()` return dict:

| Metric | Formula | Purpose |
|---|---|---|
| `a2q_reweight/h_sum_ratio` | Σ(w²H) / ΣH | Overall energy reduction |
| `a2q_reweight/energy_weighted_w2_mean` | same as above | Alias |
| `a2q_reweight/energy_weighted_w_mean` | Σ(wH) / ΣH | Energy-weighted mean weight |
| `a2q_reweight/h_top5_share_before` | Σ(H_top5) / ΣH | Top5 energy share, original |
| `a2q_reweight/h_top5_share_after` | Σ(w²H_top5) / Σ(w²H) | Top5 energy share, after reweight |
| `a2q_reweight/neff_before` | (ΣH)² / Σ(H²) | Effective sample count, original |
| `a2q_reweight/neff_after` | (Σw²H)² / Σ(w²H)² | Effective sample count, after |
| `a2q_reweight/prompt_top1_share_before` | E_max / ΣE | Top prompt energy share, original |
| `a2q_reweight/prompt_top1_share_after` | E'_max / ΣE' | Top prompt energy share, after |
| `a2q_reweight/neff_prompt_before` | (ΣE)² / Σ(E²) | Prompt neff, original |
| `a2q_reweight/neff_prompt_after` | (ΣE')² / Σ(E'²) | Prompt neff, after |

These are logged regardless of mode (even vanilla logs them for comparison).

### 核心模块

`verl/trainer/ppo/a2q_reweighting.py` — 新建文件，包含：

| 函数 | 作用 |
|---|---|
| `compute_q_response()` | 从 sum_pi_squared + old_log_probs 算 per-response Q_{i,k} |
| `compute_response_a2()` | 从 token-level advantages 算 per-response A²_{i,k} |
| `compute_h_and_e()` | H_{i,k} = A² × Q, E_i = Σ_k H_{i,k} |
| `compute_thresholds()` | global percentile τ_r, τ_p |
| `compute_weights()` | w_{i,k} = min(1, √(τ_r/H)), w_i = min(1, √(τ_p/E)) |
| `build_prompt_indices()` | uid → integer prompt index mapping |
| `apply_a2q_reweighting()` | 主入口，修改 batch advantages in-place |
| `A2QReweightState` | EMA state dataclass (tau_r_ema, tau_p_ema, step) |

### 集成点

`verl/trainer/ppo/ray_trainer.py`:
- 在 `compute_advantage()` 返回后、`_update_actor(batch)` 之前调用
- 此时 batch 是完整全局 batch，global percentile 直接算，不需要 distributed gather
- `self._a2q_reweight_state` 持久化 EMA state

### 配置字段

`verl/trainer/config/algorithm.py` → `AlgoConfig` 新增 6 个字段：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `a2q_reweight_enabled` | bool | False | 主开关 |
| `a2q_reweight_mode` | str | "hierarchical" | vanilla/rollout_only/prompt_only/hierarchical |
| `a2q_reweight_percentile_r` | float | 95.0 | rollout-level threshold percentile |
| `a2q_reweight_percentile_p` | float | 90.0 | prompt-level threshold percentile |
| `a2q_reweight_ema_beta` | float | 0.9 | EMA smoothing for thresholds |
| `a2q_reweight_warmup_steps` | int | 10 | warmup 期间不 reweight |

### 前提条件

实验脚本须设 `actor.calculate_sum_pi_squared=True`（默认 False）。
该选项与 `use_fused_kernels` 互斥。

### Logged metrics (32 个)

**Concentration:**
- `a2q_reweight/h_{mean,max}`, `e_{mean,max}`
- `a2q_reweight/h_rollout_{top1,top5,top10}_share`, `neff`, `neff_ratio`, `gini`
- `a2q_reweight/h_prompt_{top1,top5,top10}_share`, `neff`, `neff_ratio`, `gini`

**Thresholds:**
- `a2q_reweight/tau_{r,p}` (EMA), `tau_{r,p}_batch` (raw)

**Weights:**
- `a2q_reweight/{rollout,prompt}_weight_{mean,min}`, `{rollout,prompt}_clipped_frac`
- `a2q_reweight/combined_weight_mean`

**Control:**
- `a2q_reweight/{skipped,active,effective_adv_scale}`

### 实验脚本

`new_experiments/a2q_reweighting/`:
- `sync_a2q_stage1.sh` — 参数化主脚本
- 12 个 wrapper: `sync_a2q_{vanilla,rollout_only,prompt_only,hierarchical}_seed{0,1,42}.sh`
- LR = 3.1e-6 constant, N=128, K=8, 300 步
- `RESUME_MODE=auto` 支持断点续训
- `trainer.project_name="deepseek1.5b_a2q"`

### Verification

- `python3 -m py_compile` passed: a2q_reweighting.py, algorithm.py, ray_trainer.py
- `bash -n` passed: main script + 12 wrappers

---

## Signal-Quality-Aware LR Scaling (2026-05-20)

New LR scheduler: `lr_scheduler_type = "signal_quality"`.

### Config fields (optimizer.py)

| Field | Type | Default | Description |
|---|---|---|---|
| `signal_quality_info_source` | str | `"reward_std_median"` | Info indicator: reward_std_median / frac_informative / bernoulli_var / mean_a2 |
| `signal_quality_info_ema_beta` | float | 0.9 | EMA beta for info signal |
| `signal_quality_info_q_min` | float | 0.3 | Floor for q_info. Design: q_min ≈ α_safe / α_base |
| `signal_quality_warmup_steps` | int | 20 | Steps to compute I_ref |
| `signal_quality_use_concentration` | bool | False | Enable q_conc penalty |
| `signal_quality_conc_source` | str | `"cv2_h"` | Conc indicator: cv2_h / cv2_a2 / top10_h_share |
| `signal_quality_conc_ema_beta` | float | 0.9 | EMA beta for conc signal |
| `signal_quality_conc_q_min` | float | 0.5 | Floor for q_conc |
| `signal_quality_conc_gamma` | float | 0.5 | Exponent for conc penalty |

### SignalQualityLRScheduler (transformer_impl.py)

- Placed between EntropyAdaptiveLRScheduler and SignalFractionLRScheduler
- `update_signal_quality(info_value, conc_value)` — called every step
- `get_signal_quality_metrics()` — returns 12 metrics (sq_q_info, sq_q_conc, sq_warmup_done, sq_step, etc.)
- Full state_dict / load_state_dict for checkpoint resume

### Distributed consistency (critical fix)

`_compute_signal_quality_indicators()` uses `all_reduce(SUM)` so all DP ranks get identical q_t:

1. Each rank computes local sums: reward_var_sum, bernoulli_var_sum, informative_count, n_prompts
2. `torch.distributed.all_reduce(buf, op=SUM)` on CPU tensor
3. Global means computed from global sums
4. **No median** (can't all_reduce). `reward_std_median` config actually computes `mean_i[sqrt(Var_k(R_i))]`

Prompt group aggregation always uses uid, never assumes batch ordering.

### Scheduler timing

q_t from step t's batch applies to step t+1's LR.
Flow: `update_policy()` → metrics → `update_signal_quality(I_t)` → `scheduler.step()`

### Logged metrics per step

| Metric | Description |
|---|---|
| `actor/sq_q_info` | Current info gate value |
| `actor/sq_q_conc` | Current concentration gate value (1.0 if disabled) |
| `actor/sq_q_total` | Combined q = q_info × q_conc |
| `actor/sq_info_ref` | Reference I_ref from warmup |
| `actor/sq_info_ema` | Current EMA of info signal |
| `actor/sq_info_raw` | Raw I_t this step (before EMA) |
| `actor/sq_conc_ref` | Reference C_ref from warmup |
| `actor/sq_conc_ema` | Current EMA of conc signal |
| `actor/sq_conc_raw` | Raw C_t this step (if concentration enabled) |
| `actor/sq_alpha_t` | Actual LR applied |
| `actor/sq_warmup_done` | 1.0 after warmup complete |
| `actor/sq_step` | Scheduler step count |
| `actor/sq_alt_reward_var_mean` | Alternative: mean Var(R) (logged, not used for control) |
| `actor/sq_alt_reward_std_mean` | Alternative: mean sqrt(Var(R)) |
| `actor/sq_alt_bernoulli_var` | Alternative: mean p_i(1-p_i) |
| `actor/sq_alt_frac_informative` | Alternative: fraction informative prompts |

### Experiment script

`new_experiments/signal_fraction_lr/sync_signal_quality_lr.sh`

Env vars: SQ_BASE_LR, SQ_TAG, SQ_INFO_SRC, SQ_USE_CONC, SQ_CONC_SRC, SEED, etc.

### Analysis scripts

- `exp_data/signal_quality_lr_offline_replay.py` — offline validation on existing logs
- `exp_data/signal_quality_lr_sensitivity.py` — 4 sources × 4 betas sensitivity analysis

### Verification

`python3 -m py_compile` passed for all 4 modified Python files.
`bash -n` passed for experiment script.

---

## Full noise decomposition overhaul (2026-05-19)

Complete rewrite of `compute_variance_decomposition_metrics()` in
`verl/trainer/ppo/metric_utils.py` plus new `save_noise_decomp_tables()`.
Called from `verl/trainer/ppo/ray_trainer.py`.

### Motivation

Previous a2q_between_frac / a2q_within_frac had a normalization mismatch:
between_var was computed on prompt-level means (already ÷K), but within_var
was response-level raw (not ÷K). This made within-prompt fraction appear
~87% when the corrected batch-gradient-level fraction is ~56/44.

Also: need per-response/per-prompt raw arrays saved to disk so any future
decomposition / normalization / downsampling can be done offline without retraining.

### New scalar metrics (113 total, logged every step)

**Reward informativeness (extended):**
- `noise_decomp/reward_std/{mean,median,p25,p75}`
- `noise_decomp/frac_informative`, `frac_all_correct`, `frac_all_wrong`
- `noise_decomp/success_rate/{mean,std}`
- `noise_decomp/success_rate_bin_{0,0_25,25_75,75_1,1}` — success rate histogram

**Three-way between/within for A², Q, H=A²Q:**

For each X ∈ {a2, q, h, h_cfact_a, h_cfact_q}:
- `noise_decomp/factor_{X}/between_var` — Var_i(μ_i)
- `noise_decomp/factor_{X}/within_var_raw` — E_i[Var_k(X)]
- `noise_decomp/factor_{X}/within_var_corrected` — within_raw / K
- `noise_decomp/factor_{X}/batch_between` — between_var / N
- `noise_decomp/factor_{X}/batch_within` — within_raw / (NK)
- `noise_decomp/factor_{X}/batch_total` — sum of above
- `noise_decomp/factor_{X}/batch_between_frac` — corrected fraction
- `noise_decomp/factor_{X}/batch_within_frac` — corrected fraction
- `noise_decomp/factor_{X}/raw_between_frac` — legacy-style uncorrected
- `noise_decomp/factor_{X}/raw_within_frac` — legacy-style uncorrected

**Factor attribution:**
- `noise_decomp/{mean,std,cv2}_{a2,q,h}` — per-factor statistics
- `noise_decomp/corr_a2_q` — Corr(A², Q)
- `noise_decomp/interaction_ratio_a2q` — E[A²Q] / (E[A²]·E[Q])
- `noise_decomp/corr_{a2,q}_t` — correlation with response length

**Counterfactual variance attribution:**
- `noise_decomp/cfact_var_{h,a_only,q_only,interaction}`
- `noise_decomp/cfact_frac_{a_only,q_only,interaction}`

**Response length:**
- `noise_decomp/resp_length/{mean,std}`
- `noise_decomp/q_over_t/{mean,std}`

**Legacy metrics preserved:** all existing `a2q_between_frac`, `a2q_within_frac`,
`adv_between_frac`, `prompt_h/*`, etc. remain for backward compatibility.

### Per-response/per-prompt npz tables

New function `save_noise_decomp_tables(batch, global_step, save_dir)` saves
compressed npz every 5 steps to `{checkpoint_dir}/noise_tables/`.

Per-response arrays (shape [bsz]):
- `resp_prompt_idx`, `resp_reward`, `resp_advantage`, `resp_advantage_sq`
- `resp_score_energy_Q`, `resp_Q_per_token`, `resp_response_length`, `resp_H_a2q`

Per-prompt arrays (shape [N]):
- `prompt_id`, `prompt_reward_{mean,std}`, `prompt_success_rate`
- `prompt_num_unique_rewards`, `prompt_is_informative`
- `prompt_{a2,q,h}_{mean,var}`

Metadata: `global_step`, `N`, `K`.

### ray_trainer.py changes

- Import `save_noise_decomp_tables`
- After `compute_variance_decomposition_metrics()`, call
  `save_noise_decomp_tables()` every 5 steps (try/except wrapped)
- Save directory: `{trainer.default_local_dir}/noise_tables/`

### Verification

- `python3 -m py_compile` passed for metric_utils.py and ray_trainer.py
- Smoke test in verl2 env: 113 metrics, batch fractions sum to 1,
  counterfactual reconstruction exact, npz arrays correct shape, legacy metrics present

---

## Variance decomposition diagnostics (2026-05-16)

Added `compute_variance_decomposition_metrics()` to `verl/trainer/ppo/metric_utils.py`.
Called every step from `verl/trainer/ppo/ray_trainer.py` after existing `compute_variance_proxy_metrics()`.

Requires: `token_level_scores`, `advantages`, `response_mask` in batch, `uid` in non_tensor_batch.
Optional: `sum_pi_squared`, `old_log_probs` for score-energy metrics.

New metrics (all prefixed `noise_decomp/`):

- `n_prompts`, `n_responses_per_prompt`
- `reward_std/mean`, `reward_std/median`, `reward_n_unique/mean`
- `success_rate/mean`, `frac_informative`
- `adv_energy/mean`, `adv_energy/std`, `adv_scalar/abs_mean`
- `score_energy_q/mean`, `q_response/mean`, `q_response/std`
- `a2q/mean`, `a2q/std`
- `a2q_between_prompt_var`, `a2q_within_prompt_var`, `a2q_between_frac`, `a2q_within_frac`
- `prompt_h/mean`, `prompt_h/std`, `prompt_h/max`, `prompt_h/p90`

Verification: `python3 -m py_compile` passed for both files.

### Diagnostic run script

New: `new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh`

Parameterized constant-LR diagnostic script. Key env vars:

- `DIAG_LR` — constant learning rate (required)
- `DIAG_TAG` — short tag for filenames (required)
- `SEED` — random seed (default 42)
- `NCPUS` — Ray CPU count (default: `$(nproc)`)
- `RESUME_MODE` — auto/disable (default disable)

Uses `signal_fraction + sign_gate_gamma=1.0` path so split-batch metrics are
logged as diagnostics but LR is constant.

`bash -n` passed.

---

## Bug 17 (2026-05-14): ratio_of_sums r_window_num/r_window_den swapped

**Symptom**: all four ratio_of_sums W10 runs (seed 0/1/2/42) had `r_window` in
range [2, 130] instead of (0, 1]. `alpha_t` exploded 100–6500× above normal.
Models collapsed to 1-token output within the first few post-warmup steps.

**Root cause**: call site in the new-engine signal-fraction path passed arguments
in wrong order:

```python
# transformer_impl.py:1493-1494 (before fix)
r_window_num=denom,   # auto-power (should be denominator)
r_window_den=g_dot,   # cross-power (should be numerator)
```

The scheduler computes `r_window = sum(num_list) / sum(den_list)`. With the swap,
it computed `sum(auto-power) / sum(cross-power)` ≈ denom/g_dot >> 1.

**Fix**: swap the two arguments:

```python
# transformer_impl.py:1493-1494 (after fix)
r_window_num=g_dot,   # cross-power = Σ ĝ_A1ᵀ ĝ_A2
r_window_den=denom,   # auto-power  = (||ĝ_A1||² + ||ĝ_A2||²) / 2
```

**Verification**: `python3 -m py_compile` passed.

**Note**: the DP actor path (`dp_actor.py`) was not checked for the same issue;
it may have the same swap if it also passes `r_window_num`/`r_window_den`. Check
before running ratio_of_sums on the legacy sync path.

---

## Script note (2026-05-06): W10 confirmation commands

No new script is required to rerun W10 seed42 or launch W10 seed2. The base
script already supports `SEED`, `SIGFRAC_RUN_SUFFIX`, and window controls.

Use explicit suffixes so new logs/checkpoints do not collide with earlier W10
runs:

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

Expected names:

- `sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed42_rerun_*.log`
- `sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed2_*.log`
- checkpoint dirs with matching `DeepSeek1.5B-Sync-8gpu-sigfrac-cfixed-lr1.25e-5_windowr_w10_*` suffixes.

## Script update (2026-05-05): multi-seed wrappers for controller ablations

The base 1.5B c-fixed signal-fraction script now accepts a seed from the
environment:

```bash
SEED=${SEED:-42}
...
data.seed=${SEED}
```

Default behavior is unchanged for existing scripts because the default remains
`42`.

Added multi-seed wrappers for the highest-priority controller follow-up:

```text
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed1.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_seed0.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_seed1.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed0.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed1.sh
```

Each wrapper sets both `SEED` and a unique `SIGFRAC_RUN_SUFFIX`, so logs,
checkpoints, experiment names, and JSONL outputs should remain separated from
seed42 and from each other.

Bash syntax checks passed for the edited base script and representative wrappers.

## Diagnostic logging update (2026-05-05): PPO tails and dispersion metrics

Added future-run diagnostics without changing training behavior.

Files:

- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/metric_utils.py`

New PPO ratio-tail metrics:

```text
actor/ratio_p95
actor/ratio_p99
actor/ratio_frac_gt_1p2
actor/ratio_frac_lt_0p8
actor/ratio_frac_gt_1p5
```

New split PPO clip metrics:

```text
actor/pg_clipfrac_high
actor/pg_clipfrac_low
```

New train-batch dispersion metrics:

```text
critic/score/std
critic/rewards/std
critic/advantages/std
critic/advantages/abs_mean
critic/returns/std
```

Verification:

- `python3 -m py_compile verl/trainer/ppo/core_algos.py verl/trainer/ppo/metric_utils.py`
- `verl2` smoke test for `_build_ratio_kl_metrics()` passed.

Diagnostic runs started after this logging change:

```bash
bash new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh
bash new_experiments/signal_fraction_lr/sync_matched_alpha_3.10e-6.sh
```

Expected use: compare new tail/safety metrics between current B and matched
constant LR. These runs should be read as diagnostics, not as new algorithm
variants.

## Local run status (2026-05-05): low-frequency ablation results available, window results missing

Available local JSONL outputs:

- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.jsonl`
- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.jsonl`
- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.jsonl`
- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.jsonl`

Not found locally:

- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w5.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10.jsonl`
- any matching `windowr` log or checkpoint.

This means the available analysis is for slow EMA / alpha-rate-limit ablations,
not the true 3A windowed continuous-r experiment.

Controller diagnostics from available runs:

| run | alpha mean | alpha std | mean `|Δalpha|` | p95 `|Δalpha|` | `g_dot_positive` |
|---|---:|---:|---:|---:|---:|
| `alpharlim0.05` | `3.08e-6` | `0.80e-6` | `0.11e-6` | `0.21e-6` | `0.561` |
| `slowema_ret0.95` | `6.02e-6` | `2.55e-6` | `1.56e-6` | `6.35e-6` | `0.489` |
| `slowema_ret0.95_alpharlim0.05` | `6.12e-6` | `2.17e-6` | `0.17e-6` | `0.44e-6` | `0.479` |
| `slowema_ret0.98` | `5.27e-6` | `2.24e-6` | `1.74e-6` | `5.82e-6` | `0.544` |

Key engineering check for future 3A runs:

```text
actor/r_window_enabled must be 1.0
actor/r_window_count should grow toward W
output JSONL name should contain windowr_w5 or windowr_w10
```

If those metrics are absent or `actor/r_window_enabled=0`, the run is not a 3A
windowed controller run.

## Implementation status (2026-04-30): low-frequency r-side controller variants

The 1.5B signal-fraction c-fixed script now supports three low-frequency
controller variants while preserving the old B-current default behavior.

### 1. Slow EMA wrappers

Base script updated:

- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh`

New environment controls:

- `SIGFRAC_RUN_SUFFIX`
- `SIGFRAC_R_EMA_BETA_SYM`
- `SIGFRAC_R_EMA_BETA_DOWN`
- `SIGFRAC_R_EMA_BETA_UP`

Wrappers:

- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.sh`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.sh`

### 2. Alpha change-rate limit

Config and scheduler additions:

- `verl/workers/config/optimizer.py`
  - `signal_fraction_alpha_rate_limit: float = 0.0`
- `verl/workers/engine/fsdp/transformer_impl.py`
  - applies a post-computation limiter:
    `alpha_t in [(1-rate) * alpha_{t-1}, (1+rate) * alpha_{t-1}]`
  - alpha replay bypasses this limiter.
  - logs `actor/alpha_rate_limited`.
- `verl/workers/fsdp_workers.py`
  - passes the new optimizer config field.

Wrappers:

- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.sh`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.sh`

### 3A. Windowed continuous-r

Config and scheduler additions:

- `verl/workers/config/optimizer.py`
  - `signal_fraction_r_window_size: int = 0`
  - `signal_fraction_r_window_mode: str = "off"`
  - valid modes: `off`, `replace_ema`
- `verl/workers/engine/fsdp/transformer_impl.py`
  - stores the last `W` valid `r_obs` values only.
  - invalid observations do not enter the window.
  - when enabled:

```python
if valid:
    r_window.append(float(r_obs))
    r_window = r_window[-W:]

if r_window:
    r_ctrl = max(mean(r_window), r_min_ctrl)
else:
    r_ctrl = previous_r_ctrl
```

  - this mode replaces the fast EMA / fast-drop r-control branch for a clean
    windowing ablation.
  - logs:
    - `actor/r_window`
    - `actor/r_window_count`
    - `actor/r_window_enabled`
  - scheduler state dict persists the window so checkpoint resume is consistent.
- `verl/workers/fsdp_workers.py`
  - passes `signal_fraction_r_window_size` and
    `signal_fraction_r_window_mode`.

Base script additions:

- `SIGFRAC_R_WINDOW_SIZE`
- `SIGFRAC_R_WINDOW_MODE`

Wrappers:

- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w5.sh`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10.sh`

Expected local JSONL outputs:

- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w5.jsonl`
- `deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10.jsonl`

Verification completed:

```text
python3 -m py_compile verl/workers/config/optimizer.py \
  verl/workers/engine/fsdp/transformer_impl.py verl/workers/fsdp_workers.py

bash -n new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh \
  new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w5.sh \
  new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10.sh
```

Both checks passed.

### 7B operational notes from 2026-04-30

The 7B sync restart errors for
`deepseek7b_sync_16gpu_sigfrac_cfixed_lr1e-5_seed42` and
`deepseek7b_sync_16gpu_sigfrac_cfixed_lr7.5e-6_seed42` were caused by Hydra
struct overrides for fields not present in the base config. The fix was to use
`+actor_rollout_ref.actor.use_kl_loss=False` and
`+actor_rollout_ref.actor.kl_loss_coef=0.0`.

The `data/250010176` disk pressure was also identified as a blocker; old 7B
logs/checkpoints/local tracking artifacts were cleaned before relaunching.

## Implementation status (2026-04-27): fully async c-fixed r-side port to new FSDP engine

**Context**: fully async partial rollout cannot use legacy worker implementation. Setting
`trainer.use_legacy_worker_impl="enable"` fails before model init with:

```text
NotImplementedError: Fully async policy or One step off policy does not support legacy worker implementation
```

Therefore the correct path is not to force legacy workers, but to make the new
`TrainingWorker/FSDP engine` path understand the r-side of `signal_fraction`.

**Implemented for current async Phase 1 only**:

- `verl/workers/engine/fsdp/transformer_impl.py`
  - new FSDP engine `_build_lr_scheduler()` now supports `lr_scheduler_type="signal_fraction"` by constructing the same `SignalFractionLRScheduler`.
  - `FSDPEngine.train_batch()` has a `signal_fraction` branch:
    1. split local batch into A1/A2 at prompt-group boundaries;
    2. backward A1 and save `ĝ_A1`;
    3. backward A2 and read `ĝ_A2`;
    4. all-reduce `g_dot`, norm squares, and update-gradient RMS;
    5. call `sched.update_r_and_set_lr(g_dot, denom, g_rms, r_hat_raw)`;
    6. write `(ĝ_A1 + ĝ_A2)/2` to grads and optimizer step.
  - adds runtime diagnostics including `actor/signal_fraction_new_engine`, `actor/g_A1_dot_A2`, `actor/g_dot_positive`, `actor/r_hat`, `actor/r_ctrl`, `actor/alpha_t`.
  - adds a `uid` contiguity guard: every consecutive `rollout_n` entries must share the same uid; otherwise fail fast. This prevents silently splitting at response level if batch balancing reorders samples.
- `verl/workers/config/optimizer.py`
  - adds `signal_fraction_rollout_n` so the new engine knows prompt-group size.
- `verl/workers/engine_workers.py`
  - propagates `actor_rollout_ref.rollout.n` into `signal_fraction_rollout_n`.
- `new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed.sh`
  - uses `trainer.use_legacy_worker_impl="disable"` (required by fully async).
  - sets `trainer.balance_batch=False` so async assembly preserves the contiguous 8 responses per prompt.

**Important boundary**:

- This port is **only valid for the current async c-fixed sweep** where:
  - `signal_fraction_eta_c=0.0`
  - `signal_fraction_calib_frac=0.0`
  - `ppo_epochs=1`
- C-side calibration (`φ_t`, `p_t`, `a_t`, `c_t` feedback) is **not ported** to the new async engine. Do not use this implementation to claim full two-timescale controller behavior.
- If `trainer.balance_batch=True` or any group-reordering logic is reintroduced, the uid guard should catch it. If it fires, do not bypass it; use group-preserving balancing or keep balancing disabled.

**Sanity-check after restart**:

1. config prints `trainer.use_legacy_worker_impl: disable`
2. no `LR scheduler type signal_fraction is not supported`
3. no `does not support legacy worker implementation`
4. logs contain:
   - `actor/signal_fraction_new_engine = 1`
   - `actor/g_A1_dot_A2`
   - `actor/g_dot_positive`
   - `actor/r_hat`, `actor/r_ctrl`
   - `actor/alpha_t`
5. verify `actor/alpha_t` warmup/handoff/post-handoff before looking at validation scores.

**2026-04-28 code review fix: async new-engine scheduler step ordering**

Issue found after reading the async port: the new FSDP engine `signal_fraction`
branch called `update_r_and_set_lr()` before advancing scheduler step state, while
the generic `TrainingWorker.train_batch()` advanced the scheduler only on the last
mini-batch after the optimizer step. That made async new-engine `step_count`,
warmup, and handoff semantics diverge from the legacy sync path.

Fix:

- `verl/workers/engine/fsdp/transformer_impl.py`
  - signal-fraction `train_batch()` now calls `self.lr_scheduler.step()` at the
    start of each optimizer step, before A1/A2 backward and before
    `update_r_and_set_lr()`.
- `verl/workers/engine_workers.py`
  - generic post-batch scheduler stepping is skipped when
    `engine.optimizer_config.lr_scheduler_type == "signal_fraction"`, avoiding a
    second step increment.
- `tests/workers/test_signal_fraction_new_engine_on_cpu.py`
  - added CPU tests for scheduler step semantics and the outer-worker skip guard.

Verification:

- `python -m py_compile verl/workers/engine/fsdp/transformer_impl.py verl/workers/engine_workers.py tests/workers/test_signal_fraction_new_engine_on_cpu.py`
- In `verl2`:
  `PYTHONPATH=/data/250010176/codes/verl pytest -q -p no:rerunfailures tests/workers/test_signal_fraction_new_engine_on_cpu.py`
  → 2 passed.

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

---

## Alpha replay controls and D3 baseline（2026-04-27）

### Alpha replay mechanism

为实现 alpha shuffled control 和 hand-crafted schedule control，scheduler 已支持：

- `signal_fraction_alpha_replay_path`
- `signal_fraction_alpha_replay_shuffle`
- `signal_fraction_alpha_replay_seed`
- `signal_fraction_alpha_replay_start_step`
- `signal_fraction_alpha_replay_end_step`

支持 JSON list / JSON dict / CSV / TXT。replay override 在 `update_r_and_set_lr()` 中 signal-fraction alpha 计算之后应用，并通过 `_set_optimizer_lr(alpha_t)` 写入 optimizer，因此 `actor/alpha_t` 理论上记录的是 override 后的真实 LR。

相关 metrics：

- `actor/alpha_replay_applied`
- `actor/alpha_replay_enabled`

注意：4.27 SwanLab 导出的 actor CSV 没包含这两个 replay metrics，但 `actor/alpha_t` 可用于复核。

### Observed replay logging offset

4.27 shuffled control 发现一个重要日志/step offset：

- 配置 `REPLAY_START_STEP=21` 时，日志 `actor/alpha_t` 的 step22-300 与 shuffled replay 序列逐点一致；
- 日志 step21 是 replay 前的原始 signal-fraction alpha。

因此后续使用 alpha replay 做精确 step21-300 schedule 时，默认应使用：

- `REPLAY_START_STEP=20`
- `REPLAY_END_STEP=299`

目标是让**日志 step21-300** 对应 280 个 replay 值。跑完后必须先检查 `actor/alpha_t`，不能直接看分数。

### D3 implementation

D3 baseline 用 alpha replay 实现，不改训练核心代码。

新增 replay 文件：

- `exp_data/4.27/alpha_replay/d3_piecewise_stage_matched_step21_300.json`

该 JSON 长度 280，值为：

- 前 80 个：`3.286384633e-6`（日志 step21-100）
- 中 100 个：`3.370729357e-6`（日志 step101-200）
- 后 100 个：`2.667058466e-6`（日志 step201-300）
- weighted mean: `3.0953198319e-6`

新增脚本：

- `new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3.sh`
- `new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3_seed42.sh`
- `new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3_seed0.sh`
- `new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3_seed1.sh`

D3 主脚本设置：

- `actor_rollout_ref.actor.optim.lr=3.0953198319e-6`
- `lr_scheduler_type=signal_fraction`
- `signal_fraction_eta_c=0.0`
- `signal_fraction_calib_frac=0.0`
- `signal_fraction_sign_gate_gamma=1.0`
- `signal_fraction_alpha_replay_shuffle=False`
- `REPLAY_START_STEP=20`
- `REPLAY_END_STEP=299`

保持 `signal_fraction + sign_gate_gamma=1.0` 路径，是为了与 C310/M 一样保留 split-batch diagnostics，同时避免换 scheduler 路径引入实现混杂。

启动命令：

```bash
./new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3_seed42.sh
./new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3_seed0.sh
./new_experiments/signal_fraction_lr/sync_piecewise_stage_matched_decay_d3_seed1.sh
```

第一优先级 sanity-check：

1. `actor/alpha_t` step21-100 是否全为 `3.286384633e-6`
2. step101-200 是否全为 `3.370729357e-6`
3. step201-300 是否全为 `2.667058466e-6`
4. 若 `actor/alpha_replay_applied` 被导出，step21-300 是否全为 1
5. 若仍有 offset，按实际日志窗口重算，不要先看 validation score

### D3 run result（2026-04-28）

用户上传 D3 三 seed 数据到 `exp_data/4.28/`：

- `val-core/`: five validation tasks, steps 0..300 every 10 steps
- `actor/`: alpha/lr/r_hat/r_ctrl/gradient diagnostics, steps 1..300
- `critic/`: reward/return/advantage/response length

Sanity-check:

- `actor/alpha_t` equals replayed D3 schedule for all three seeds:
  - step21-100: `3.286384633e-6`
  - step101-200: `3.370729357e-6`
  - step201-300: `2.667058466e-6`
- weighted post mean step21-300: `3.0953198319e-6`

Outcome:

- D3 core avg final: `0.3360`, best `0.3372@270`, final-best `-0.0011`.
- B core avg final: `0.3443`.
- matched alpha 3.10e-6 final: `0.3322`.
- alpha shuffled final: `0.3324`.

Engineering interpretation:

- alpha replay mechanism is working for exact stage schedules when using `REPLAY_START_STEP=20`, `REPLAY_END_STEP=299`.
- D3 confirms that a state-independent stage schedule can recover much of B, so the code path is not hiding a special signal-fraction-only optimization.
- B still outperforming D3 means the continuous adaptive path remains worth preserving; D3 is a strong baseline, not a replacement.

---

## DeepSeek-7B sync c_fixed sweep scripts（2026-04-28）

新增 7B 16-GPU signal-fraction c_fixed sweep。

Files:

- `new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed.sh`
- `new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed_lr5e-6_seed42.sh`
- `new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed_lr7.5e-6_seed42.sh`
- `new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed_lr1e-5_seed42.sh`

Resource rationale:

- 16 GPUs = 2 nodes * 8 GPUs.
- Under a 128 GPU-hour per-job cap, 16 GPUs gives ~8h wall-clock vs 32 GPUs ~4h.
- Default `RESUME_MODE=auto` so repeated submissions continue from the latest matching checkpoint.
- Ray multi-node setup follows existing sync scripts: rank-0 node starts head at `${HEAD_IP}:6379`; workers join via `ray start --address="$HEAD_IP:6379"`.
- The 7B script adds a worker-side readiness loop (`ray status --address="$HEAD_IP:6379"`) before joining, reducing failures when worker nodes start before the head is ready.

Sweep:

| BASE_LR | c_fixed approx |
|---:|---:|
| `5e-6` | `1.0e-4` |
| `7.5e-6` | `1.5e-4` |
| `1e-5` | `2.0e-4` |

Main config:

- `MODEL_PATH=/data/250010176/codes/models/DeepSeek-R1-Distill-Qwen-7B`
- `max_response_length=10240`
- `overlong_buffer_cfg.len=2048`
- `train_batch_size=256`
- `rollout.n=16`
- `ppo_mini_batch_size=64`
- `gen_tp=4`
- `lr_scheduler_type=signal_fraction`
- `signal_fraction_eta_c=0.0`
- `signal_fraction_calib_frac=0.0`
- `trainer.balance_batch=False`
- `trainer.save_freq=10`
- `trainer.max_actor_ckpt_to_keep=2`
- `trainer.max_critic_ckpt_to_keep=2`

Validation:

- `bash -n` passed for the main script and all wrappers.
- Scripts are executable.

First launch bug/fix:

- Three first 7B sync submissions all failed after Ray/vLLM init and step-0 validation, before the first training update.
- Stack root:
  - `verl/trainer/ppo/ray_trainer.py` calls `calculate_debug_metrics()`
  - `verl/utils/debug/metrics.py` calls `torch.quantile(diff_f, 0.90)`
  - PyTorch raises `RuntimeError: quantile() input tensor is too large`
- Interpretation: this was optional metric logging on a very large flattened token tensor (`7B`, `rollout.n=16`, long responses), not a Ray setup issue and not a signal-fraction algorithm failure.
- Fix implemented on 2026-04-28:
  - `verl/utils/debug/metrics.py`: removed optional rollout-diff p90/p99 and response-length p90/p99.
  - `verl/trainer/ppo/rollout_corr_helper.py`: replaced rollout-diff p95/p99 with mean/max.
  - `verl/trainer/ppo/core_algos.py`: ratio/KL stats now use mean/std/max only; p95/p99 removed.
- Validation:
  - `python3 -m py_compile verl/utils/debug/metrics.py verl/trainer/ppo/rollout_corr_helper.py verl/trainer/ppo/core_algos.py`
  - no remaining `torch.quantile` under `verl/utils/debug`, `verl/trainer/ppo`, or `verl/experimental/fully_async_policy`
  - `bash -n` passed for 7B main/wrapper scripts.

Local experiment-parameter logging:

- Implemented in `verl/utils/tracking.py` via the existing `file` logger backend.
- Whenever `trainer.logger` includes `"file"`, local metrics still write to `<experiment_name>.jsonl`, and the resolved config is additionally saved as:
  - `<experiment_name>.config.json` (latest pointer)
  - `<experiment_name>.config.YYYYMMDD_HHMMSS.pid<PID>.json` (immutable run snapshot)
- `new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed.sh` now sets:
  - `VERL_FILE_LOGGER_ROOT="${VERL_ROOT}/logs/local_tracking"`
- Therefore the 7B sync c-fixed sweep writes local copies under:
  - `/data/250010176/codes/verl/verl/logs/local_tracking/DeepSeek7B/`
- Smoke test with `/data/250010176/yrh/miniconda3/envs/verl2/bin/python` generated the expected `.config.json`, timestamped config snapshot, and `.jsonl` metric file.

7B runtime adjustment:

- After the quantile fix, all three 16k 7B runs reached `global_step_10` and saved checkpoints.
- Runtime at 16k was too high for c-fixed sweep:
  - per train step roughly `935-1067s`
  - step-10 validation roughly `1165-1182s`
  - 10 steps roughly `3.1h`, making 300 steps impractical under a 128 GPU-hour cap.
- Main 7B c-fixed script default now uses:
  - `MAX_RESPONSE_LENGTH=10240`
  - `OVERLONG_BUFFER_LEN=2048`
- The 10k setting is for scale search; it preserves more headroom than 8k while reducing token cost vs 16k.
- New 10k launches should use `RESUME_MODE=disable` if a fresh run is intended, because `RESUME_MODE=auto` may resume the previous 16k checkpoint directory.

---

## Async signal-fraction c_fixed sweep（2026-04-27）

### Purpose

同步 B 的 `c_fixed=2.5e-4` 来自同步 rollout/update 动力学，不能直接假设迁移到 fully async partial rollout。异步会改变：

- staleness distribution
- effective update cadence
- gradient noise
- KL / entropy trajectory

因此 async 主线第一步是重新 sweep `c_fixed`，且只测试一个自由度。

### Implementation

新增 async 参数化主脚本：

- `new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed.sh`

新增 seed42 sweep wrappers：

- `new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed_lr7.5e-6_seed42.sh`
- `new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed_lr1e-5_seed42.sh`
- `new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed_lr1.25e-5_seed42.sh`

主脚本沿用现有 fully async partial rollout baseline：

- `python3 -m verl.experimental.fully_async_policy.fully_async_main`
- rollout/training GPU split: 4 / 4
- `rollout_mode=async`
- `partial_rollout=True`
- `staleness_threshold=4`
- `trigger_parameter_sync_step=1`
- `require_batches=4`
- `total_rollout_steps=$((128*300))`

Signal-fraction 配置：

- `actor_rollout_ref.actor.optim.lr=${BASE_LR}`
- `actor_rollout_ref.actor.optim.lr_scheduler_type=signal_fraction`
- `signal_fraction_eta_c=0.0`
- `signal_fraction_calib_frac=0.0`
- `signal_fraction_r_min=0.01`
- `signal_fraction_c_min=1e-8`
- `signal_fraction_c_max=1e-2`

不用 Method B / held-out C calibration，因为本轮是 cfixed sweep，`eta_c=0`，不更新 c-side。保留 `calib_frac=0.0`，避免改变 update batch composition。

### Sweep points

| wrapper | BASE_LR | implied c_fixed |
|---|---:|---:|
| `async_partial_sigfrac_cfixed_lr7.5e-6_seed42.sh` | `7.5e-6` | `1.5e-4` |
| `async_partial_sigfrac_cfixed_lr1e-5_seed42.sh` | `1e-5` | `2.0e-4` |
| `async_partial_sigfrac_cfixed_lr1.25e-5_seed42.sh` | `1.25e-5` | `2.5e-4` |

Status: 3-run seed42 coarse sweep started on 2026-04-27. This is only for locating the async scale regime, not for final claims.

Launch failures and final fix:

- Logs: `verl/logs/async_partial_sigfrac_cfixed_lr{7.5e-6,1e-5,1.25e-5}_seed42_20260427_033*.log`
- Error: `NotImplementedError: LR scheduler type signal_fraction is not supported`
- Failure point: model initialization in `verl/workers/engine/fsdp/transformer_impl.py::_build_lr_scheduler()`.
- Root cause: fully async default config sets `trainer.use_legacy_worker_impl=disable`, so it uses the new `TrainingWorker/FSDP engine` path. This path does not implement `signal_fraction` scheduler nor the split-batch A1/A2 r-side update. The validated signal-fraction implementation lives in legacy `DataParallelPPOActor._update_policy_signal_fraction()`.
- Rejected attempted fix: setting `use_legacy_worker_impl="enable"` does **not** work; fully async rejects legacy workers with `Fully async policy or One step off policy does not support legacy worker implementation`.
- Final fix:
  - keep `use_legacy_worker_impl="disable"`;
  - port r-side signal-fraction to the new FSDP engine;
  - set `trainer.balance_batch=False` so prompt rollout groups remain contiguous;
  - pass `signal_fraction_rollout_n=rollout.n`;
  - add uid contiguity guard and `actor/signal_fraction_new_engine` diagnostic.
- Boundary: this only supports the current c-fixed async sweep (`eta_c=0`, `calib_frac=0`, `ppo_epochs=1`). C-side calibration is not ported.
- After relaunch, first check that the run passes model initialization and reaches the first actor update before analyzing alpha/metrics.

Follow-up bug (2026-04-27 04:27 relaunch):

- Latest low-scale log: `verl/logs/async_partial_sigfrac_cfixed_lr7.5e-6_seed42_20260427_042737.log`
- Error: `FileExistsError: ... outputs/2026-04-27/04-28-45/fully_async_main.log`, then `ValueError: Unable to configure handler 'file'`.
- Root cause: the three 8GPU async wrappers were launched concurrently on one machine. They share Hydra's second-resolution default output dir, fixed Ray ports `6379/8265`, and `/tmp/ray`; concurrent launches can collide or stop each other's Ray cluster.
- Fix:
  - main async cfixed script now creates a unique `RUN_ID="${EXP_NAME}_${TIMESTAMP}_pid$$"`;
  - passes `hydra.run.dir="${LOG_DIR}/hydra/${RUN_ID}"` and `hydra.job.name="${RUN_ID}"`;
  - wraps the run in `flock` on `${LOG_DIR}/async_partial_sigfrac_cfixed_8gpu.lock`, so the three 8GPU sweep wrappers queue instead of racing on Ray/Hydra resources.
- Additional sanity print: `Actor optimizer config: lr_scheduler_type=..., signal_fraction_rollout_n=...` in `verl/workers/engine_workers.py`, so relaunch logs directly show whether the actual new-engine actor optimizer received `signal_fraction` and `rollout_n=8`.
- Note: a second printed `actor.optim.lr_scheduler_type: constant` in the full config dump can be the top-level trainer template, not the actual `actor_rollout_ref.actor` used by `DetachActorWorker`; use the new sanity print and `actor/signal_fraction_new_engine` metric for confirmation.

Follow-up bug (2026-04-27 FrozenInstanceError):

- Latest mid/high logs: `verl/logs/async_partial_sigfrac_cfixed_lr{1e-5,1.25e-5}_seed42_20260427_042737.log`
- Error: `dataclasses.FrozenInstanceError: Field 'signal_fraction_rollout_n' is frozen and cannot be modified`.
- Root cause: `FSDPOptimizerConfig` inherits `BaseConfig`; after `omega_conf_to_dataclass(self.config.actor)`, fields are frozen except `_mutable_fields`. The previous patch tried to set `actor_training_config.optimizer_config.signal_fraction_rollout_n = self.config.rollout.n` after dataclass construction.
- Fix: set `self.config.actor.optim.signal_fraction_rollout_n = self.config.rollout.n` under `open_dict(self.config.actor.optim)` before calling `omega_conf_to_dataclass(self.config.actor)`. The resulting frozen dataclass now receives `rollout_n` through construction, not mutation.

启动：

```bash
./new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed_lr7.5e-6_seed42.sh
./new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed_lr1e-5_seed42.sh
./new_experiments/signal_fraction_lr/async_partial_sigfrac_cfixed_lr1.25e-5_seed42.sh
```

### Analysis order

先机制，后分数：

1. `actor/alpha_t` step21-300 mean/range
2. `actor/r_hat`, `actor/r_ctrl` 是否长期卡 floor
3. `actor/ppo_kl_mean`, `actor/grad_norm` 是否明显高于同步
4. entropy 是否过早塌陷
5. val final/peak/drop 最后看

选择规则：

- high 不炸且表现最好：继续用 `1.25e-5`
- high 有 KL/grad norm/entropy 异常：退到 `1e-5`
- mid 仍激进：退到 `7.5e-6`
- low 学不动：排除

Second-round densification rule:

- If `1.25e-5` is best and stable, add `1.5e-5` / `1.75e-5`.
- If `1e-5` is best, add `8.75e-6` / `1.125e-5`.
- If `7.5e-6` is best or high is unstable, add `5e-6` / `6.25e-6` / `8.75e-6`.

---

## 2026-04-29 7B sync sweep runtime and launch updates

### Runtime diagnosis

The 7B c-fixed sync sweep was still very slow after reducing response length from 16k to 10k. Latest 10k logs showed this is mostly expected workload, not a clear hang:

- `MAX_RESPONSE_LENGTH=10240` was confirmed in the resolved config.
- Per train step on 16 GPUs was roughly `589-745s` depending on run/step.
- Typical breakdown:
  - `timing_s/gen`: about `316-400s`
  - `timing_s/old_log_prob`: about `53-68s`
  - `timing_s/update_actor`: about `203-275s`
  - `timing_s/step`: about `590-740s`
- Each step processes roughly `20M-24M` tokens:
  - `TRAIN_BATCH_SIZE=256`
  - `N_RESP_PER_PROMPT=16`
  - average response length often `4.7k-5.7k`

Conclusion: the main bottleneck is the huge long-response rollout/update workload for 7B, with additional validation/checkpoint overhead. A full 300-step 16-GPU run can still take multiple days wall-clock.

### Validation/checkpoint changes

To reduce avoidable overhead without changing the training trajectory:

- `new_experiments/signal_fraction_lr/deepseek7b_sync_16gpu_sigfrac_cfixed.sh`
  - added `VAL_N_RESP_PER_PROMPT`
  - added `TEST_FREQ`
  - added `SAVE_FREQ`
  - added `VAL_BEFORE_TRAIN`
- Defaults now:
  - `VAL_N_RESP_PER_PROMPT=8` instead of validation also using train `n=16`
  - `TEST_FREQ=30` instead of `10`
  - `SAVE_FREQ=10` kept unchanged because 8-hour jobs may not reach 30 steps
  - `VAL_BEFORE_TRAIN=True`

Important rationale: validation is expensive because it runs 5 eval sets with multiple sampled responses, but `save_freq` should stay at 10 under short wall-clock job windows to avoid losing too much progress.

### 32-GPU support

The main 7B sync script now supports dynamic GPU tagging:

- defaults remain `NNODES=2`, `NGPUS_PER_NODE=8` -> `16gpu`
- setting `NNODES=4`, `NGPUS_PER_NODE=8` -> `32gpu`
- `EXP_NAME_BASE`, `CKPT_PREFIX`, and `LOG_FILE` now use `${GPU_TAG}` / total GPU count, so 32-GPU runs do not accidentally share 16-GPU checkpoint/log prefixes.

New 32-GPU wrappers were added:

- `new_experiments/signal_fraction_lr/deepseek7b_sync_32gpu_sigfrac_cfixed_lr5e-6_seed42.sh`
- `new_experiments/signal_fraction_lr/deepseek7b_sync_32gpu_sigfrac_cfixed_lr7.5e-6_seed42.sh`
- `new_experiments/signal_fraction_lr/deepseek7b_sync_32gpu_sigfrac_cfixed_lr1e-5_seed42.sh`

Recommended launch order:

1. Start `lr7.5e-6` first and inspect the first few `timing_s/step`.
2. If 32 GPUs clearly improves wall-clock versus 16 GPUs, launch the other two.

Under a 128 GPU-hour per-job cap, 32 GPUs only gives about 4 hours wall-clock, so it may require more checkpoint/resume submissions even if each step is faster.

### Cleanup note

Old 7B logs/local tracking/checkpoints were cleared to start fresh:

- `verl/logs/deepseek7b_sync_*.log`
- `verl/logs/local_tracking/DeepSeek7B/deepseek7b_sync_*`
- `DeepSeek7B/deepseek7b_sync_*.jsonl`
- `verl/ckpts/DeepSeek7B/DeepSeek7B-Sync-16gpu-sigfrac-cfixed-*`
- `verl/ckpts/DeepSeek7B/DeepSeek7B-Sync-32gpu-sigfrac-cfixed-*`

After cleanup:

- `verl/ckpts/DeepSeek7B`: about `40K`
- `verl/logs/local_tracking/DeepSeek7B`: about `4K`
- `DeepSeek7B`: about `4K`

Caveat: the cleanup also removed any just-created 32-GPU run artifacts matching those patterns. If the scheduler was already running the three new 32-GPU jobs, verify whether they recreated logs/checkpoints or need relaunching.
