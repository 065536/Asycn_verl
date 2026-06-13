# Project Status

**Last updated**: 2026-06-12 (Direction pivot: from adaptive LR to RL-aware optimizer design; pos/neg gradient decomposition diagnostic)

## 2026-06-12 Direction Pivot — From Adaptive LR to RL-Aware Optimizer

### 核心认知转变

LR 不是本质问题。所有 adaptive LR 方案（signal-fraction, signal-quality gate, entropy gate）都在调标量 α，但 RL gradient 的结构性问题发生在**方向**层面，而非步长层面。Adam 作为通用优化器，没有利用 RL gradient 的特有结构：

1. Gradient 是 positive-advantage（巩固好 response）和 negative-advantage（惩罚差 response）的向量和
2. 不同 prompt 的 success rate 不同，gradient composition 随 prompt 类型变化
3. Adam 的 coordinate-wise preconditioning 可能系统性地改变这个 composition

### |A⁻| 大 ≠ negative update 占主导

之前"negative advantage 占主导"的说法过粗。三件事必须分开：

| 层面 | 说的是什么 | 结论 |
|---|---|---|
| 标量 advantage 分布 | 一阶 ΣA+ vs Σ|A⁻| | **平衡**（GRPO group norm 结构保证） |
| 二阶 advantage energy | ΣA²+ vs ΣA²⁻ | 高成功率 prompt 中**负侧更大** |
| 真实梯度向量和 | ‖g+‖ vs ‖g⁻‖, cos(g+,g⁻) | **未知 — 需要实验** |

关键推导（p = success rate, K responses per prompt）：
- 正样本总 advantage: pK·A⁺ = K√(p(1-p))
- 负样本总 advantage 绝对值: (1-p)K·|A⁻| = K√(p(1-p))
- **一阶总量平衡**

- 正样本总平方量: pK·(A⁺)² = K(1-p)
- 负样本总平方量: (1-p)K·(A⁻)² = Kp
- p>0.5 时**负侧二阶 energy 更大**

但二阶 energy 大不等于梯度向量和大：正样本数量多可能方向更一致，负样本方向分散可能互相抵消。

### 新的实验方向

不再追求 adaptive LR。转向诊断 RL gradient 结构，为设计 RL-aware optimizer 提供依据。

| Priority | 内容 | 状态 |
|---|---|---|
| **P0** | Pos/neg lm_head gradient decomposition | **已实现，实验启动中** |
| **P1** | Adam preconditioning direction diagnostic | 已设计，待 P0 数据 |
| P2 | RL-aware optimizer 设计（基于 P0/P1 结论） | blocked on P0/P1 |

### P0: Pos/Neg Gradient Decomposition（本次实现）

将 batch gradient 分解为 g⁺（正 advantage 样本贡献）和 g⁻（负 advantage 样本贡献），在 lm_head 层精确计算。

**方法**：对子采样 response，forward pass with `output_hidden_states=True`，解析计算 per-response gradient G_b = Σ_t (e_{y_t} - π_t) h_tᵀ，按 advantage 符号累加到 g⁺/g⁻。

**按 prompt success rate 分桶**（p<0.5 vs p≥0.5）分别计算，回答：
- 高成功率 mixed prompt 中，负样本是否真的主导真实更新？
- 低成功率 mixed prompt 中，正样本方向是否更集中？
- g⁺ 和 g⁻ 是否近似反向（cos ≈ -1）还是有大角度偏差？

### 实现

| File | Change |
|---|---|
| `verl/trainer/ppo/lm_head_grad_norm.py` | +`compute_single_response_lm_head_grad()` — 解析计算 (V,d) gradient vector |
| `verl/workers/actor/dp_actor.py` | +`compute_pos_neg_grad_decomposition()` — 子采样 + 累加 + metrics |
| `verl/workers/actor/dp_actor.py` | `update_policy()` 集成：在训练前运行诊断 |
| `verl/workers/config/actor.py` | +`calculate_pos_neg_grad_decomp`, `pos_neg_grad_decomp_freq/max_responses` |
| `new_experiments/.../sync_constant_lr_diagnostic.sh` | lm_head_grad_norm + pos_neg_grad_decomp 直接写入脚本 |

### 新 metrics

| Metric | 含义 |
|---|---|
| `grad_decomp/g_pos_norm` | ‖g⁺‖ |
| `grad_decomp/g_neg_norm` | ‖g⁻‖ |
| `grad_decomp/g_total_norm` | ‖g⁺ + g⁻‖ |
| `grad_decomp/cos_pos_neg` | cos(g⁺, g⁻) |
| `grad_decomp/pos_neg_norm_ratio` | ‖g⁺‖ / ‖g⁻‖ |
| `grad_decomp/high_p/*` | 高成功率 prompt (p≥0.5) 的同组 metrics |
| `grad_decomp/low_p/*` | 低成功率 prompt (p<0.5) 的同组 metrics |
| `grad_decomp/frac_uninformative` | 无信息 prompt (p=0 or p=1) 比例 |

### 首要分析目标

1. ‖g⁺‖ vs ‖g⁻‖ → 谁主导 update magnitude？
2. cos(g⁺, g⁻) → 正负方向关系（反向 ≈ -1？正交 ≈ 0？）
3. high_p vs low_p 分桶的 norm ratio 差异 → 二阶 energy 不对称是否传导到真实梯度？
4. 以上指标随训练的演变趋势

### Bug fixes

1. `dp_actor.py:compute_log_prob` 中 `select_keys` 遗漏 `response_mask`（lm_head_grad_norm 需要）→ 已修复
2. `sync_constant_lr_diagnostic.sh` 中 `IS_HEAD` 判断在单节点（无 rank 环境变量）时错误走 worker 分支 → 已修复

---

## 2026-06-10 Exact lm_head Gradient Norm — A²Q Proxy Replacement

### 核心问题

A²Q 分解假设不同 token 的 score function 梯度在参数空间正交：

```
||Σ_t J_t||² = Σ_t ||J_t||²  (A²Q 假设)
```

实际上所有 token 共享参数，cross terms `Σ_{t≠s} ⟨J_t, J_s⟩` 可以很大。A²Q 的 H = A²·Q 只保留了对角项，等价于 `A² · trace(K_h ⊙ diag(K_δ))`。

### 解决方案：lm_head 层精确梯度范数

对 lm_head 层（最大参数矩阵，V×d），per-response gradient 有解析公式：

```
∂S_b/∂W = Σ_t (e_{y_t} - π_t) h_tᵀ
||∂S_b/∂W||²_F = Σ_{t,s} K_δ[t,s] · K_h[t,s]
```

其中：
- `K_h[t,s] = h_t · h_s`（hidden-space Gram matrix）
- `K_δ[t,s] = 1_{y_t=y_s} - π_s[y_t] - π_t[y_s] + π_t·π_s`（vocab-space error vector Gram matrix）

**精度**：对 lm_head 层精确，无任何近似。autograd 验证误差 < 1e-6。

**不需要 FSDP unsharding**：只用 hidden states 和 logits（forward 产物），不需要访问权重。

### 实现

| File | Change |
|---|---|
| `verl/trainer/ppo/lm_head_grad_norm.py` | 新文件：Gram 矩阵公式 + chunked 版本 |
| `verl/workers/actor/dp_actor.py` | +`compute_lm_head_grad_norms()` + 集成到 `compute_log_prob` |
| `verl/workers/config/actor.py` | +`calculate_lm_head_grad_norm`, `lm_head_grad_norm_freq`, `lm_head_grad_norm_max_responses` |
| `verl/trainer/ppo/metric_utils.py` | +exact_grad metrics（between/within, cross_term_ratio, corr） |
| `verl/trainer/ppo/ray_trainer.py` | +`global_step` in meta_info |

### 新 metrics

| Metric | 含义 |
|---|---|
| `noise_decomp/exact_grad/cross_term_ratio_mean` | E[||∂S/∂W||²/Q]，>1 = cross terms 正贡献，<1 = 互相抵消 |
| `noise_decomp/exact_grad/corr_gnorm_q` | exact norm 与 Q 的相关性（A²Q 作为 proxy 有多好） |
| `noise_decomp/exact_grad/mean_h_exact` | E[A²·||∂S/∂W||²]（exact gradient energy） |
| `noise_decomp/exact_grad/gnorm/between_frac` | ||∂S/∂W||² 的 prompt 间方差占比 |
| `noise_decomp/exact_grad/h_exact/between_frac` | A²·||∂S/∂W||² 的 prompt 间方差占比 |

### 实验已启动

3 seed baseline，LR=1e-6，每 5 步采样 64 个 response 计算 exact gradient norm：

```bash
DIAG_LR=1e-6 DIAG_TAG=lr1e-6_exactgrad SEED={42,1,0} NCPUS=32 \
EXTRA_HYDRA="++actor_rollout_ref.actor.calculate_lm_head_grad_norm=true \
  ++actor_rollout_ref.actor.lm_head_grad_norm_freq=5 \
  ++actor_rollout_ref.actor.lm_head_grad_norm_max_responses=64" \
bash sync_constant_lr_diagnostic.sh
```

SwanLab: `deepseek1.5b_sync_8gpu_diag_constant_lr1e-6_exactgrad_seed{42,1,0}`

### 首要分析目标

1. `cross_term_ratio_mean` 偏离 1 多少 → cross terms 有多大
2. `corr_gnorm_q` → A²Q 和 exact norm 相关性如何
3. exact grad 的 between/within 分解 vs A²Q 的分解是否一致
4. cross_term_ratio 随训练的演变趋势

### Checkpoint 清理

清理了 10 个老实验 ckpt（diag_aqh, sq_info/ent, cosine decay），释放 ~320G：

- 1.2T → 776G（21 个目录保留）
- 保留所有 a2q lr1e-6 和 lr3.1e-6 实验 ckpt

---

## 2026-06-08b Pivot: Gradient Direction Diagnostic

### 核心修正

从 "Adam denominator 压步长 → 调 LR" 转向 "Adam D_t 是否改变 gradient 方向，削弱 positive consolidation"。

**之前的错误框架**：v*=0.1 释放 denominator → 看分数变不变 → 本质是 LR 实验。
**修正后的框架**：先诊断 cos(m, g⁺) vs cos(Dm, g⁺) → 确认方向扭转再设计干预。

### 为什么 uniform v*=0.1 不是好证据

v_t ← 0.1·v_t 后：m/(√(0.1v)+ε) = m/(√0.1·√v+ε) ≈ √10 · m/(√v+ε)。

方向不变，只是全局放大 √10 ≈ 3.16×。等价于 LR×3.16。不能区分 "denominator 改变方向" vs "步长变大"。

### 新的实验优先级

| Priority | 内容 | 性质 | 状态 |
|---|---|---|---|
| **P0** | Gradient preconditioning diagnostic | 纯诊断，1 checkpoint + 几个 batch | **next** |
| P1 | Counterfactual D analysis (flatten/cap/normalize) | 纯计算 | pending |
| P2 | Non-uniform continuation (仅当 P0 确认路径 B) | 训练 | blocked on P0 |
| ~~旧 P0~~ | ~~v*=0.1 continuation~~ | ~~等价 LR change~~ | ~~降级~~ |

### P0 Diagnostic 指标

```
cos(m, g⁺_boundary)     — numerator 是否对齐 positive consolidation
cos(Dm, g⁺_boundary)    — preconditioned update 是否对齐
Δ_dir⁺ = cos(Dm,g⁺) - cos(m,g⁺)  — 方向扭转量
κ(g⁺) / κ(g⁻)          — positive vs negative 的 preconditioning strength ratio
d̄(g⁺) vs d̄(g⁻)        — positive 方向遭遇的平均 denominator
```

### 判断标准

- **路径 A (numerator 问题)**：cos(m, g⁺) ≤ 0 → 正样本不足或被冲掉 → 转向 positive auxiliary
- **路径 B (denominator 方向扭转)**：cos(m, g⁺) > 0 且 Δ_dir⁺ < 0 且 κ(g⁺) < κ(g⁻) → D 削弱 positive → 非 uniform D 干预有价值

### Per-problem 分桶（来自 6.8 per-problem eval）

| 分桶 | 用途 |
|---|---|
| Boundary (p∈[0.4,0.7]) | 核心检测对象 |
| Consolidated (p>0.8) | 对照：positive 应已巩固 |
| Problem 20 (退化) | 区分 cos(m,g⁺)<0 vs cos(Dm,g⁺)<0 |

### Implementation (completed, from 6.8)

| File | Change |
|---|---|
| `verl/workers/actor/dp_actor.py` | +`_compute_adam_diagnostics()` 方法 (9 metrics), 集成到两条 update 路径 |
| `verl/workers/config/optimizer.py` | +`adam_v_scale`, `adam_override_beta2` 配置字段 |
| `verl/workers/fsdp_workers.py` | `load_checkpoint` 后自动执行 v-scale / beta2 override |
| `new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh` | +`${EXTRA_HYDRA:-}` 支持 |
| `exp_data/per_problem_eval.py` | Per-problem AIME 评测脚本 (vLLM + FSDP merge) |
| `exp_data/run_per_problem_eval.sh` | 便捷启动封装 |
| `exp_data/per_problem_structure_inference.md` | 完整分析报告 |

### Per-Problem Structure Inference (retained from 6.8, still valid)

从 76 个 run 的聚合 AIME 指标推断 per-problem 结构。用于确定 gradient diagnostic 的 problem bucket 分组。

**高置信度结论**：
1. 增益集中在 ~5 道边界题 (p: 0.1-0.3 → 0.5-0.7)
2. ~11/30 题始终不动 (p≈0)，RL 完全无效
3. 一致性提升 > 能力提升：Δmaj@16 = 2.0× Δmean@16 (SNR=4.24)
4. 行为变化先于能力变化：overlong 84%→51%，先于 acc 提升 ~10 steps
5. AIME2025 提升仅为 AIME2024 的 60%，存在题型偏向

**结论**：GRPO 主要把 near-boundary 问题稳定化，而非学会全新解法。Per-problem 分桶直接服务于 P0 gradient diagnostic。

---

---

## 2026-06-01 AQH Closed-Loop Experiment Results

### Executive summary

A4 signal-quality LR (reward_std gate, base 1e-5) **failed to achieve its goal**. The reward-space gate only reduced LR by 10–15% (effective 8.6–9.0e-6), allowing entropy collapse (0.45→0.08) and per-token score energy collapse (77%). The A²/Q/H variance decomposition is structurally stable across runs — the failure is not a new H-concentration pathology but a self-masking feedback loop where the reward-only gate cannot detect policy-space collapse. Offline replay confirms an entropy-ratio gate would have triggered at step 58–66 and hit the 3.1e-6 safe floor by step 92–108.

### What we discovered

**1e-5 is an unsafe high-variance regime**: A2 seed0 runs successfully to 300 steps (best 0.348), but A2 seed42 catastrophically collapses (0.30→0.10 by step 140). Not "always fails" — but unacceptable seed variance.

**A4 prevents catastrophic collapse but introduces slow policy collapse**: both A4 seeds complete 300 steps, peak early (step 90–120), then continuously degrade (final 0.27–0.28). Safer than A2-seed42, worse than A2-seed0.

**The true failure mode is entropy collapse, not reward signal degradation**: A4 late entropy 0.08–0.09 (vs A2-seed0: 0.35, A1: 0.42). The causal chain:
```
high LR + weak gate → policy sharpening → entropy ↓
→ p(a) → 1 → per-token q = 1-2p+Σp² → 0
→ score-gradient energy vanishes → recovery barrier
→ validation continuously drops
```

### Self-masking feedback loop (key mechanism finding)

The gate's control action (slight LR reduction) prevents the reward-std indicator from declining, so the gate never triggers strong reduction. Meanwhile, the actual failure (entropy collapse → Q collapse) proceeds through a channel the gate does not monitor.

> Controller optimizes the observability of its own failure signal.

### Two failure channels identified

| Type | Name | Indicators | A4 status |
|---|---|---|---|
| I | Reward-side signal degradation | reward_std, frac_informative, A², CV²(A²) | Stable (gate protected) |
| II | Policy-side score-energy collapse | entropy, per-token q, Q_response | **Collapsed** (gate blind) |

Conclusion: a gate using only Type I indicators will miss Type II collapse.

### A²/Q/H structural findings (stable across all runs)

- CV²(A²) >> CV²(Q): advantage side is the primary noise driver
- Counterfactual: A²-only ~50% of H variance, Q-only ~3–5%, interaction ~40–45%
- Q between-prompt fraction >92%: Q is strongly prompt-level
- Per-token q / entropy are policy-health sensors, distinct from H variance drivers
- These findings are consistent across A1/A2/A4, confirming Stage 2 results

### Offline replay validation

| Gate | seed0 first<5e-6 | seed0 first≤3.1e-6 | seed42 first<5e-6 | seed42 first≤3.1e-6 |
|---|---|---|---|---|
| Entropy | step 58 | step 92 | step 66 | step 108 |
| Per-token Q | step 68 | step 102 | step 77 | step 132 |

Entropy gate triggers earlier and is computationally cheaper (no `sum_pi_squared`).

### Corrections applied in analysis

1. **Terminology**: "E[Q]" → "per-token score energy q" (0.2 scale is per-token, not response-level)
2. **Phrasing**: "categorically too high" → "unsafe high-variance regime" (A2-seed0 succeeds)
3. **Causality**: "Q collapse is root cause" → "Q collapse is irreversible lock-in mechanism"
4. **Implementation**: `reward_std_median` config name actually computes distributed `sqrt(mean_i[Var_k(R_i)])` — failure is in the entire reward-space indicator class

### Next method: dual-channel gate

```
q_entropy = clip(EMA(entropy) / entropy_ref, 0.31, 1.0)
q_total = min(q_info, q_entropy)     # min, not product
lr = base_lr * q_total
```

Parameters: entropy_ref = warmup steps 10–20 mean, EMA β=0.9, q_entropy_min=0.31.

### Next experiments (priority order)

| Config | Base LR | Gate | Priority | Purpose |
|---|---|---|---|---|
| **A4c** | **5e-6** | **reward + entropy** | **Highest** | Lower max LR + policy collapse detection |
| A4d | 1e-5 | reward + entropy | High | Test if policy gate saves aggressive LR |
| A4b | 5e-6 | reward only | Medium | Isolate base LR effect |
| A1 relaunch | 3.1e-6 | none | Needed | Safe baseline (both seeds failed) |
| A3 relaunch | 5e-6→3.1e-6 cosine | none | Needed | Decay baseline (both seeds failed) |

If two configs: A4c + A4d (separates "lower base LR" vs "policy gate" contributions).

### Data availability

| Run | Steps | AQH data | Status |
|---|---|---|---|
| A1 lr3.1e-6 seed0 | 46 | ✓ | ❌ partial |
| A1 lr3.1e-6 seed42 | 164 | ✓ | ⚠️ partial |
| A2 lr1e-5 seed0 | 300 | ✓ | ✅ complete |
| A2 lr1e-5 seed42 | 145 | ✓ | ⚠️ catastrophic collapse |
| A4 sq_info seed0 | 300 | ✓ | ✅ complete |
| A4 sq_info seed42 | 300 | ✓ | ✅ complete |
| A3 cosine seed0/42 | 8/0 | — | ❌ failed |

### Files

| File | Contents |
|---|---|
| `exp_data/aqh_closedloop_report.md` | Full analysis report with all tables and decomposition |
| `exp_data/aqh_closedloop_analysis.py` | Analysis script (6 runs, 9 sections) |
| `exp_data/aqh_gate_offline_replay.py` | Offline entropy/Q gate replay on A4 data |
| `exp_data/aqh_gate_replay_results.txt` | Raw replay output |
| `exp_data/aqh_closedloop_report_raw.txt` | Raw analysis output |

---

## 2026-05-20 Signal-Quality-Aware LR Scaling

```
α_t = α_base · q_t,    q_t ∈ [q_min, 1.0]
```

Design principle: `q_min = α_safe / α_base`. With α_base=1e-5, q_min=0.3 → min LR = 3e-6 ≈ known-safe LR.
Meaning: use aggressive LR early; automatically retreat to safe LR when signal degrades.

### Offline validation (completed)

Ran `exp_data/signal_quality_lr_offline_replay.py` on Stage 1 constant-LR logs:

| Run | val drop | q_trend | Status |
|---|---:|---:|---|
| lr1e-5_seed0 | -0.0847 | -0.3603 | ✓ predictive |
| lr1e-5_seed1 | -0.0746 | -0.0598 | ✓ predictive |
| lr1e-5_seed42 | -0.0053 | -0.4226 | ✓ predictive |
| lr3.1e-6 (stable) | ~0 | ~0 | ○ no drop to predict |

Key interpretation:
- **q_info is a risk indicator, not a deterministic predictor.** seed42: q drops to floor 0.30 but val only drops -0.005.
- **q_t↓ means current batch signal quality is worse**; whether it causes val drop depends on LR magnitude, model state, KL/ratio tail.
- **Stable LR: q naturally declines but doesn't cause harm** — reward informativeness drops as model improves (more all-correct/all-wrong), but low LR makes this safe.

### Sensitivity analysis (completed)

Ran `exp_data/signal_quality_lr_sensitivity.py` comparing 4 I_t sources × 4 β_I values:

| Source | high-LR mean q_trend | stable-LR mean q_late | Verdict |
|---|---:|---:|---|
| **reward_std_median** | **-0.28** | **0.82** | ★ **best: brakes when needed, doesn't brake when safe** |
| frac_informative | -0.025 | 1.00 | too weak — barely gates LR at all |
| bernoulli_var (p(1-p)) | **+0.02** | 1.00 | **fails completely** — q goes UP during degradation |
| reward_var_mean | -0.20 | 0.64 | usable but high false-alarm on stable LR |

**bernoulli_var failure**: uses global success_rate/mean → p(1-p) is highest at p=0.5, decreases monotonically as model improves OR collapses. Cannot distinguish the two.

**β_I sensitivity**: 0.70–0.95 all work; differences are small. β=0.95 has lowest false-alarm on stable LR (q_late=0.85).

Effective LR trajectory (β=0.95, base=1e-5):

| run | α_early | α_late | α_min | stable ref |
|---|---|---|---|---|
| seed0 | 8.8e-6 | **5.6e-6** | 3.0e-6 | 3.1e-6 |
| seed1 | 9.3e-6 | **8.6e-6** | 5.5e-6 | 3.1e-6 |
| seed42 | 8.2e-6 | **3.7e-6** | 3.0e-6 | 3.1e-6 |

seed1 note: info-only gate is weak for seed1 (q only drops to 0.88). This seed's degradation may need concentration gate or is driven by ratio-tail/KL risk rather than reward informativeness.

### Critical engineering fixes (completed)

**Distributed consistency**: `_compute_signal_quality_indicators()` rewritten with `all_reduce`.
Each rank computes local sums (reward_var_sum, bernoulli_var_sum, informative_count, n_prompts),
then `all_reduce(SUM)` → global mean. All ranks get identical q_t.
**No longer uses median** (can't all_reduce). `reward_std_median` config now computes
`mean_i[sqrt(Var_k(R_i))]` — semantically similar, distributedly correct.

**Prompt group aggregation**: always uses uid-based grouping, never assumes batch ordering.

**Scheduler timing**: q_t computed from step t's batch, applied to step t+1's LR.
Flow: `update_policy()` → metrics → `update_signal_quality(I_t)` → `scheduler.step()`.

**Alternative indicators logged without controlling LR**:
- `actor/sq_alt_reward_var_mean` — mean_i[Var_k(R)]
- `actor/sq_alt_bernoulli_var` — mean_i[p_i(1-p_i)]
- `actor/sq_alt_frac_informative` — fraction of informative prompts
- `actor/sq_info_raw` — raw I_t before EMA

These allow offline comparison without rerunning.

**Concentration gate disabled by default**: `signal_quality_use_concentration=False` → no H=A²Q computation overhead.

### Implementation (completed)

Files changed:

| File | Change |
|---|---|
| `verl/workers/config/optimizer.py` | 10 new `signal_quality_*` config fields; `lr_scheduler_type="signal_quality"` |
| `verl/workers/engine/fsdp/transformer_impl.py` | `SignalQualityLRScheduler` class (~150 lines) + `update_signal_quality_for_lr()` + `get_signal_quality_metrics()` on engine |
| `verl/workers/fsdp_workers.py` | `_compute_signal_quality_indicators()` with all_reduce + integration in `update_actor` |
| `verl/workers/engine_workers.py` | new-engine path integration |
| `new_experiments/signal_fraction_lr/sync_signal_quality_lr.sh` | parameterized experiment script |
| `exp_data/signal_quality_lr_offline_replay.py` | offline validation script |
| `exp_data/signal_quality_lr_sensitivity.py` | sensitivity analysis (4 sources × 4 betas) |

Logged metrics per step (12+):

| Metric | Description |
|---|---|
| `actor/sq_q_info` | current info gate value |
| `actor/sq_q_conc` | current concentration gate (1.0 if disabled) |
| `actor/sq_q_total` | combined q = q_info × q_conc |
| `actor/sq_alpha_t` | actual LR applied |
| `actor/sq_info_raw` | raw I_t this step (before EMA) |
| `actor/sq_info_ema` | EMA-smoothed I_t |
| `actor/sq_info_ref` | warmup reference I_ref |
| `actor/sq_warmup_done` | 1.0 after warmup complete |
| `actor/sq_step` | scheduler step count |
| `actor/sq_alt_*` | 4 alternative indicators for offline comparison |

Default parameters:

| Param | Value | Rationale |
|---|---|---|
| I_t source | reward_std_median | best sensitivity analysis score |
| β_I | 0.9 | conservative; 0.95 also works |
| q_min | 0.3 | = α_safe / α_base = 3.1e-6 / 1e-5 |
| T_warm | 20 | collect I_ref before gating |
| use_concentration | False | info-only first; add if needed |

### Experiment plan

**Group A**: stable fixed LR 3.1e-6 (already have Stage 1 data, 3 seeds)

**Group B**: aggressive fixed LR 1e-5 (already have Stage 1 data, 3 seeds)

**Group C** (to launch): aggressive LR + info gate

```bash
SQ_BASE_LR=1e-5 SQ_TAG=sq_info SEED=42 NCPUS=32 bash sync_signal_quality_lr.sh
SQ_BASE_LR=1e-5 SQ_TAG=sq_info SEED=1  NCPUS=32 bash sync_signal_quality_lr.sh
SQ_BASE_LR=1e-5 SQ_TAG=sq_info SEED=0  NCPUS=32 bash sync_signal_quality_lr.sh
```

SwanLab names: `deepseek1.5b_sync_8gpu_sq_info_seed{42,1,0}`

**Group D** (after C, if needed): aggressive LR + info + concentration gate

```bash
SQ_BASE_LR=1e-5 SQ_TAG=sq_info_conc SQ_USE_CONC=true SEED={42,1,0} NCPUS=32 bash sync_signal_quality_lr.sh
```

### Post-run analysis order

1. **Mechanism first**: check `actor/sq_alpha_t`, `actor/sq_q_info`, `actor/sq_info_ref` — verify warmup, gating, and LR trajectory
2. **Compare effective LR**: C's late-stage α should be between 3e-6 and 6e-6
3. **Validation**: peak-to-final drop, seed variance, early learning speed
4. **Mechanistic**: entropy, KL, ratio_p95 — does reduced late LR prevent these from diverging?
5. **Alternative indicators**: check `actor/sq_alt_*` — would a different I_t have been better?

### Success criteria

1. **Peak-to-final drop**: C < B (most important)
2. **Seed variance**: std_seed(final_val of C) < std_seed(final_val of B)
3. **Early learning speed**: C ≈ B in first 50–100 steps
4. **Late effective LR**: C's α_t auto-descends to 3–6e-6 range

### Expected outcomes by seed

- **seed0**: likely strongest improvement (offline shows q drops to 0.50, val drop is -0.085)
- **seed42**: may not improve much (original drop only -0.005; q already gates aggressively)
- **seed1**: key test — if C doesn't fix seed1, concentration gate (Group D) becomes necessary

---

## 2026-05-19 Stage 2 Analysis, Normalization Fix, Full Noise Logging Overhaul

### Stage 2 partial results (6 of 9 runs)

| Config | K | N | Seeds OK | Seeds failed/missing |
|---|---:|---:|---|---|
| n256k4 | 4 | 256 | seed42 (301 steps) | seed1 (3 lines), seed0 missing |
| n128k8 | 8 | 128 | seed42 (280), seed1 (301) | seed0 (3 lines) |
| n64k16 | 16 | 64 | seed42 (301), seed1 (301), seed0 (270) | — |

Failed runs killed externally during initialization (no Python traceback).

### Critical finding: a2q normalization correction

The raw `a2q_between_frac` / `a2q_within_frac` had a normalization mismatch.
`between_var` = Var_x(μ_i) where μ_i is already 1/K averaged, but `within_var`
= E_x[Var_k(z)] is response-level raw. Batch gradient variance requires:

```
Var(ĝ) = (1/N)·between_var + (1/NK)·within_var
```

The within term needs an extra 1/K factor. Corrected fractions:

| Config | K | raw within_frac | **corrected between_frac** |
|---|---:|---:|---:|
| n256k4 | 4 | 0.67 | **0.67** (prompt noise dominates) |
| n128k8 | 8 | 0.87 | **0.56** (~50/50) |
| n64k16 | 16 | 0.94 | **0.50** (exact 50/50) |

**Conclusion: at baseline K=8, prompt-sampling and rollout noise contribute
roughly equally (~56/44), not 87/13 as originally reported.**

### Validation: n64k16 best

| Config | Mean best avg5 | Mean final avg5 |
|---|---:|---:|
| n256k4 (1 seed) | 0.3354 | 0.3354 |
| n128k8 (2 seeds) | 0.3333 | 0.3265 |
| n64k16 (3 seeds) | **0.3446** | **0.3369** |

### Two-dimensional noise framework established

Cannot compare prompt/rollout noise with A²/Q noise as four additive terms.
Correct hierarchy:

- **Dimension A (sampling axis)**: between-prompt vs within-prompt variance of h=A²Q
- **Dimension B (factor axis)**: CV²(A²) vs CV²(Q) vs Corr(A²,Q) within each sampling component

### Full noise logging overhaul

Rewrote `compute_variance_decomposition_metrics()` in `metric_utils.py`:

- **113 scalar metrics** per step (up from ~35)
- Three-way A²/Q/H decomposition with **batch-level corrected fractions**
- CV², Corr(A²,Q), interaction_ratio, counterfactual attribution
- Success rate histogram (5 bins)
- Response length and Q/T diagnostics

New `save_noise_decomp_tables()`:

- Per-response arrays: reward, advantage, A², Q, H=A²Q, response_length, Q/T
- Per-prompt arrays: reward stats, success rate, A²/Q/H mean+var
- Compressed npz every 5 steps → offline reanalysis without retraining

### All 9 runs relaunched

All 9 Stage 2 runs (3 configs × 3 seeds) relaunched from scratch with new code:

```bash
# n256k4
DIAG_N=256 DIAG_K=4 SEED={42,1,0} NCPUS=32 bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh

# n128k8
DIAG_N=128 DIAG_K=8 SEED={42,1,0} NCPUS=32 bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh

# n64k16
DIAG_N=64 DIAG_K=16 SEED={42,1,0} NCPUS=32 bash /data/250010176/codes/verl/new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
```

### Files changed

- `verl/trainer/ppo/metric_utils.py` — full rewrite of noise decomposition + new `save_noise_decomp_tables()`
- `verl/trainer/ppo/ray_trainer.py` — import new function, call npz saving every 5 steps
- `exp_data/stage2_nk_noise_analysis.md` — Stage 2 analysis report (corrected)
- `exp_data/stage2_nk_noise_analysis.py` — analysis script
- `memory/algorithm_design.md` — updated with Stage 2 findings + two-dimensional noise framework
- `memory/engineering_impl.md` — updated with full logging overhaul details
- `memory/feedback_style.md` — added: log raw components, don't conflate decomposition axes
- `memory/MEMORY.md` — updated index
- `PROJECT_STATUS.md` — this update

### Analysis plan after rerun

With full factor decomposition + npz tables:

1. **Dimension A**: Confirm corrected between/within fractions for A², Q, H separately
2. **Dimension B**: CV²(A²) vs CV²(Q) — is noise driven by advantage informativeness or score/length energy?
3. **Interaction**: Corr(A², Q) — do high-advantage samples also have high score energy?
4. **Counterfactual**: Var(A²Q̄) vs Var(Ā²Q) vs interaction — which factor dominates H variance?
5. **Cross N/K**: How do factor attributions shift with N/K configuration?
6. **Offline validation**: Use npz tables to verify all scalar metrics match, test alternative normalizations

---

## 2026-05-17 Between/Within Decomposition Fix & Stage 2 Ready

### Bug fix: between/within variance decomposition now works without sum_pi_squared

**Root cause**: all a2q / between-within / score_energy / prompt_h metrics were
gated by `has_score_energy = "sum_pi_squared" in batch.batch`. Stage 1 runs
never set `calculate_sum_pi_squared=True`, so the entire block was silently
skipped — prompt grouping (`uid`) was fine, only `sum_pi_squared` was missing.

**Fix** (`verl/trainer/ppo/metric_utils.py`): added a pure advantage-based
between/within-prompt variance decomposition that runs **unconditionally** —
only needs `uid` + `advantages` (always present in batch). Implements the exact
law of total variance:

```
Var(A) = Var_x(E[A|x]) + E_x[Var(A|x)]
       = between_var     + within_var
```

Verified: `py_compile` passed; smoke test confirms `between + within = Var(A)`
exactly.

New metrics logged every step (no config change needed):

| Metric | Meaning |
|---|---|
| `noise_decomp/adv_between_prompt_var` | Var_x(E[A\|x]) — prompt-sampling noise |
| `noise_decomp/adv_within_prompt_var` | E_x[Var(A\|x)] — rollout-sampling noise |
| `noise_decomp/adv_total_var` | Sum of above |
| `noise_decomp/adv_between_frac` | between / total — fraction from prompt sampling |
| `noise_decomp/adv_within_frac` | within / total — fraction from rollout sampling |
| `noise_decomp/prompt_adv_mean/{std,abs_mean,max}` | Prompt-level gradient proxy |

The existing a2q decomposition (score-function weighted) is **preserved** and
will additionally log when `calculate_sum_pi_squared=True` (enabled in Stage 2
script).

### Stage 1 diagnostic results (5 of 6 runs complete)

Detailed analysis: `exp_data/stage1_variance_decomposition_analysis.md`

Usable runs: lr3.1e-6 seed1 (300 steps), seed42 (278 steps); lr1e-5 seed0/1/42 (all 300 steps).
lr3.1e-6 seed0 only 90 steps (partial, excluded from group stats).

Validation:

| Group | Mean best avg5 | Mean final avg5 | Mean drop |
|---|---:|---:|---:|
| lr3.1e-6 (2 seeds) | 0.3378 | 0.3327 | -0.0051 |
| lr1e-5 (3 seeds) | 0.3313 | 0.2764 | -0.0549 |

lr1e-5 seed0/seed1 suffered catastrophic late-stage collapse (drop -0.08/-0.07).

#### Key findings

1. **Reward sparsity is the dominant noise source**: `frac_informative` ≈ 50-55%
   throughout — nearly half of prompt groups contribute zero gradient signal
   (all responses same reward). This is the structural baseline problem.

2. **`reward_std/median` is the best leading indicator**: in high-LR, median
   within-prompt reward std drops to 0.13-0.21 by step 100 (vs stable LR 0.44-0.63).
   Late-phase group-mean gap = -24.9%. This divergence leads entropy collapse
   by 50-100 steps.

3. **SNR is fundamentally low**: r_hat ≈ 0.01-0.02, g_dot_positive ≈ 50-56%
   (coin-flip level). SNR_split frequently negative in 50-step windows.

4. **Between/within decomposition was NOT logged in Stage 1** — now fixed (see
   above). Stage 2 runs will have both the advantage-based decomposition
   (unconditional) and the a2q decomposition (`calculate_sum_pi_squared=True`).

5. **Divergence timeline**: reward sparsity metrics diverge at step 10-50;
   entropy diverges at step ~120; r_hat at step ~40.

#### Answers to pre-registered questions

| # | Question | Answer |
|---|---|---|
| Q1 | frac_informative declines faster in high-LR? | Yes, -4.2% in late phase |
| Q2 | between vs within-prompt variance? | Cannot answer from Stage 1 (fixed for Stage 2) |
| Q3 | adv_energy drop + SNR collapse? | Confirmed: adv_energy -20%, auto_power can 3× in high-LR |
| Q4 | r_hat correlates with frac_informative? | Weakly (r=-0.36), confounded by training stage |
| Q5 | Which metrics diverge first? | reward_std/median (-24.9% late), then frac_informative (-4.2%) |

### Stage 2: N×K decomposition experiment design

**Goal**: separate prompt-sampling noise from rollout-sampling noise by varying
N (prompts) and K (responses per prompt) while holding total responses = 1024.

**Fixed LR**: 3.1e-6 (stable regime) — only N×K varies.

| Config | N (prompts) | K (responses/prompt) | Total | ppo_mini_batch_size | Purpose |
|---|---:|---:|---:|---:|---|
| high-N low-K | 256 | 4 | 1024 | 64 | More prompt diversity, less within-prompt averaging |
| baseline | 128 | 8 | 1024 | 32 | Same as Stage 1 |
| low-N high-K | 64 | 16 | 1024 | 16 | Fewer prompts, stronger within-prompt signal |

Each config × 3 seeds (42, 1, 0) = **9 runs, 72 GPUs**.

**Key diagnostics** (all enabled in Stage 2):

- Advantage-based decomposition (unconditional, from fix above):
  `adv_between_frac`, `adv_within_frac`
- a2q decomposition (from `calculate_sum_pi_squared=True`):
  `a2q_between_frac`, `a2q_within_frac`, `score_energy_q/mean`, `prompt_h/*`
- All Stage 1 metrics continue: `frac_informative`, `reward_std/*`, `r_hat`, etc.

**Predictions to test**:

| Hypothesis | Metric to watch | If true |
|---|---|---|
| Prompt noise dominates | `adv_between_frac` high in baseline; high-N reduces it | Batch stratification most valuable |
| Rollout noise dominates | `adv_within_frac` high; high-K reduces it + raises frac_informative | Increase K per prompt most valuable |
| Both matter equally | No clear winner between high-N and high-K | Need combined approach |

Launch commands:

```bash
# high-N low-K (N=256, K=4)
DIAG_N=256 DIAG_K=4 SEED=42 NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
DIAG_N=256 DIAG_K=4 SEED=1  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
DIAG_N=256 DIAG_K=4 SEED=0  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh

# baseline (N=128, K=8)
DIAG_N=128 DIAG_K=8 SEED=42 NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
DIAG_N=128 DIAG_K=8 SEED=1  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
DIAG_N=128 DIAG_K=8 SEED=0  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh

# low-N high-K (N=64, K=16)
DIAG_N=64 DIAG_K=16 SEED=42 NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
DIAG_N=64 DIAG_K=16 SEED=1  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
DIAG_N=64 DIAG_K=16 SEED=0  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh
```

SwanLab experiment names:
- `deepseek1.5b_sync_8gpu_stage2_n256k4_seed{42,1,0}`
- `deepseek1.5b_sync_8gpu_stage2_n128k8_seed{42,1,0}`
- `deepseek1.5b_sync_8gpu_stage2_n64k16_seed{42,1,0}`

### Analysis plan after Stage 2

1. Does `frac_informative` increase with K? (More responses → more likely to have
   mixed rewards within a prompt group)
2. Does `adv_between_frac` vs `adv_within_frac` shift with N/K? (Direct answer to Q2)
3. Does `a2q_between_frac` confirm the advantage-based decomposition?
4. Does high-K improve r_hat / SNR_split? (More within-prompt averaging)
5. Does high-N improve validation score? (More diverse prompt sampling)
6. Which config gives the best score-improvement per unit of gradient noise?

### Files changed

- `verl/trainer/ppo/metric_utils.py` — added unconditional advantage between/within decomposition
- `new_experiments/signal_fraction_lr/sync_stage2_nk_diagnostic.sh` — new Stage 2 script
- `exp_data/stage1_variance_decomposition_analysis.md` — Stage 1 full analysis report
- `PROJECT_STATUS.md` — updated with fix, Stage 1 results, and Stage 2 design

---

## 2026-05-16 Direction Pivot: Variance Source Decomposition

### Motivation

All r_t estimators tried so far (split-batch, EMA window, ratio_of_sums, logit-space
M₂, parameter-space momentum) have fundamental issues. Before designing the next
estimator, we need to answer:

> Where does RL gradient noise actually come from — prompt sampling, rollout
> sampling, reward signal sparsity, or score-function variance?

Running adaptive LR while diagnosing noise conflates the two problems: the
controller changes the policy trajectory, which changes the noise structure,
making it impossible to identify root causes.

### Parameter-space momentum estimator: rejected

A new estimator using A_t = EMA(||ĝ_t||²) and B_t = EMA(||m_t^Adam||²) was
proposed. Analysis revealed a **non-stationarity bias**: the derivation assumes
g_t is stable within Adam's momentum window (~10 steps), but in RL:

- When signal decays (late stage): ||ḡ_t||² > ||g_t||² → r_t systematically
  overestimated → LR stays too high
- This is exactly the dangerous regime the project cares most about
- Bias magnitude: ~30% overestimate when ||g_t|| decays 20% over 10 steps at
  r_t = 0.02

Additional issues: low-r_t resolution (B/A dynamic range too narrow near κ),
double-EMA response delay (15-20 steps), Adam bias correction in early steps.

### New approach: diagnostic-first experiment design

Instead of building another estimator, decompose gradient noise into four layers
using law of total variance:

```
Var(ĝ) = (1/N)[Var_x(μ(x)) + E_x[Var(U|x)]]
```

1. **Prompt sampling noise** — between-prompt variance of gradient contributions
2. **Rollout sampling noise** — within-prompt, between-response variance
3. **Reward/advantage signal sparsity** — fraction of informative prompt groups
4. **Score-function / trajectory-length variance** — logit-space energy

### Implemented diagnostics

Added `compute_variance_decomposition_metrics()` to
`verl/trainer/ppo/metric_utils.py`, called every step from `ray_trainer.py`.

New metrics logged per step:

| Category | Metrics |
|---|---|
| Reward informativeness | `noise_decomp/frac_informative`, `reward_std/mean`, `success_rate/mean`, `reward_n_unique/mean` |
| Advantage energy | `noise_decomp/adv_energy/mean`, `adv_energy/std`, `adv_scalar/abs_mean` |
| Score energy | `noise_decomp/score_energy_q/mean`, `q_response/mean`, `a2q/mean`, `a2q/std` |
| Between/within decomposition | `noise_decomp/a2q_between_prompt_var`, `a2q_within_prompt_var`, `a2q_between_frac`, `a2q_within_frac` |
| Prompt-level proxy | `noise_decomp/prompt_h/mean`, `prompt_h/std`, `prompt_h/max`, `prompt_h/p90` |

Split-batch metrics (`actor/r_hat`, `actor/g_A1_dot_A2`, `actor/g_dot_positive`)
continue to be logged as diagnostics but do NOT control LR.

### Stage 1 experiment: constant LR diagnostic runs

Created `new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh` — a
parameterized script for pure constant-LR diagnostic runs.

Two LR conditions, 3 seeds each (6 runs total, 48 GPUs):

| Run | LR | Purpose | Seeds |
|---|---|---|---|
| A (stable) | 3.10e-6 | Normal training noise structure | 42, 0, 1 |
| B (high) | 1e-5 | Pre-collapse noise structure | 42, 0, 1 |

Launch commands:

```bash
# Run A — stable LR
DIAG_LR=3.10e-6 DIAG_TAG=lr3.1e-6 SEED=42 NCPUS=32 bash new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh
DIAG_LR=3.10e-6 DIAG_TAG=lr3.1e-6 SEED=0  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh
DIAG_LR=3.10e-6 DIAG_TAG=lr3.1e-6 SEED=1  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh

# Run B — high LR (known to degrade)
DIAG_LR=1e-5 DIAG_TAG=lr1e-5 SEED=42 NCPUS=32 bash new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh
DIAG_LR=1e-5 DIAG_TAG=lr1e-5 SEED=0  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh
DIAG_LR=1e-5 DIAG_TAG=lr1e-5 SEED=1  NCPUS=32 bash new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh
```

SwanLab experiment names:

- `deepseek1.5b_sync_8gpu_diag_constant_lr3.1e-6_seed{42,0,1}`
- `deepseek1.5b_sync_8gpu_diag_constant_lr1e-5_seed{42,0,1}`

### Analysis plan after Stage 1

Compare stable vs high-LR runs across early/mid/late phases:

1. Does `frac_informative` decline faster in high-LR runs?
2. Does `a2q_between_frac` or `a2q_within_frac` dominate?
3. Does `adv_energy` drop while `a2q` stays high (SNR collapse)?
4. Does `r_hat_split` correlate with `frac_informative`?
5. Which noise-source metrics diverge first before validation drops?

### Future stages (not yet started)

**Stage 2: N,K decomposition** — fix total responses, vary prompt count vs
completions per prompt (e.g., 64×32 vs 128×16 vs 256×8) to separate prompt
noise from rollout noise.

**Stage 3: algorithm direction** — choose based on Stage 1-2 findings:

| Finding | Algorithm direction | Stage 2 verdict |
|---|---|---|
| Informative prompts decline | Reward-informativeness-aware LR decay | **✓ confirmed**: frac_informative 从 early 40-66% 降到 late 38-63% |
| Score energy stays high + signal weak | M₂-based noise diagnostic gate | **✗ refuted**: A² 和 Q 同步衰减 (各 -17~27%), 无 "Q stays high" 模式; interaction 衰减最慢 |
| Between-prompt variance dominates | Batch construction / prompt stratification | **Partial**: K=4 时 between 占 66%, K=8 时 55%, K=16 时 50/50; 不是一方绝对主导 |
| Within-prompt variance dominates | Increase K / temperature / advantage estimator | **Partial**: K=16 时 within 50%, 但总方差 9× 大于 K=4; 盲目增 K 无效 |

### Files changed

- `verl/trainer/ppo/metric_utils.py` — added `compute_variance_decomposition_metrics()`
- `verl/trainer/ppo/ray_trainer.py` — import and call new function
- `new_experiments/signal_fraction_lr/sync_constant_lr_diagnostic.sh` — new diagnostic script
- `memory/algorithm_design.md` — updated with direction pivot
- `memory/pg_moment_signal_fraction_lr_design.md` — marked as paused
- `memory/MEMORY.md` — updated index

---

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
