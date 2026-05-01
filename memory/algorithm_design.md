---
name: Algorithm Design — Current Framework and Open Questions
description: entadapt_initial 缺陷 + α_t=c_t·r̂_t 框架 + 4.21 诊断：c_t dead → p_min guard 删除；guard 设计原则确立
type: project
---

## 2026-04-30 controller redesign after noisy split-alignment diagnosis

The current interpretation of split-batch alignment is deliberately weaker and
cleaner:

```text
single-step alignment sign is weak;
temporally aggregated alignment may still be useful.
```

Empirical diagnostic: `g_dot_positive` is only around `55%`, close to a random
coin flip at the single-step sign level. Therefore the method should not be
presented as a precise per-step reliability classifier, and future controllers
should not trust one raw `g_A1^T g_A2` observation as a high-confidence decision.

Updated claim:

> split-batch alignment is a noisy but informative proxy. Its value is to induce
> a state-aware, low-frequency LR trajectory, not to exactly estimate the
> optimal LR at every PPO mini-update.

This preserves the support-matched update-scale story from the D3/C310/shuffled
controls:

- D3 recovering much of B means coarse update-scale schedule explains a large
  part of the gain.
- B still beating D3 means state-dependent residual allocation may still matter.
- Shuffled alpha is now treated as a secondary/confusing control, not the main
  evidence source. The cleaner question is whether temporal aggregation of the
  same signal improves the controller.

### Next controller sequence

Priority is to validate low-bandwidth control without introducing artificial
stage priors:

1. **Slow EMA**: reduce direct transmission of raw `r_obs` noise.
2. **Alpha change-rate limit**: constrain per-step alpha jitter after computing
   the signal-fraction LR.
3. **3A windowed continuous-r**: replace the fast EMA input with the mean of the
   last `W` valid `r_obs` values.

Deferred:

- stage-level baseline + residual adaptive: rejected as next main step because
  it needs D3/stage prior and weakens the clean claim.
- 3B windowed sign reliability: deferred because it introduces new mapping
  hyperparameters (`lambda`, multiplier bounds, window size).
- 3C hysteresis sign-gate: still useful later as a sign-gate repair, but not the
  main controller line.

### 3A experimental interpretation

Run:

| group | purpose |
|---|---|
| B-current repeat | same-batch baseline |
| 3A-W5 | short temporal aggregation; should filter one-step noise while keeping responsiveness |
| 3A-W10 | stronger low-pass test; checks whether low-frequency stage trend dominates |

Interpretation:

- W5/W10 > B: current B was overreacting to single-step noise; temporal
  aggregation improves alignment-conditioned gain control.
- W5 ~= B, W10 < B: moderate smoothing is safe, excessive low-pass loses useful
  adaptivity.
- W5/W10 < B: current B smoothing is already adequate or windowing over-damps.
- W10 ~= D3: long window is mostly becoming a coarse schedule and losing
  state-dependent residual value.

## 2026-04-27 async c-fixed sweep: design boundary after new-engine port

Current async Phase 1 goal is **only** to calibrate `c_fixed` under fully async partial rollout:

```text
α_t = c_fixed · r_ctrl,    c_fixed ≈ BASE_LR / r_boot,    r_boot=0.05
```

This is deliberately not a full `c_t` controller experiment. The active sweep remains:

| BASE_LR | implied c_fixed |
|---:|---:|
| `7.5e-6` | `1.5e-4` |
| `1e-5` | `2.0e-4` |
| `1.25e-5` | `2.5e-4` |

**Why this separation matters**:

- fully async changes staleness, update cadence, gradient noise, KL/entropy trajectories, and partial-rollout truncation.
- Therefore sync `c_fixed=2.5e-4` cannot be assumed to transfer.
- This phase should choose a scale region only; it should not be used to make a final claim about r-side causality or full two-timescale feedback.

**Current implementation semantics**:

- r-side split-batch logic has been ported to the new async `TrainingWorker/FSDP engine` path.
- A1/A2 are split at prompt-group boundaries, provided `trainer.balance_batch=False` preserves contiguous rollout groups.
- C-side calibration is not ported. This is acceptable for this sweep because `eta_c=0` and `calib_frac=0`.

**First analysis order after runs start**:

1. mechanism:
   - `actor/signal_fraction_new_engine=1`
   - `actor/alpha_t` warmup/handoff/post-handoff
   - `actor/r_hat`, `actor/r_ctrl`, floor frequency
   - `actor/g_A1_dot_A2`, `actor/g_dot_positive`
2. stability:
   - `actor/ppo_kl_mean`
   - `actor/grad_norm`
   - entropy trajectory
3. validation:
   - final/peak/drop only after mechanism is confirmed.

**Decision rule**:

- `1.25e-5` stable and best → densify upward (`1.5e-5`, `1.75e-5`)
- `1e-5` best → densify around mid (`8.75e-6`, `1.125e-5`)
- `7.5e-6` best or higher points unstable → densify lower (`5e-6`, `6.25e-6`, `8.75e-6`)

---

## Current algorithm: entadapt_initial (flawed)

Formula: α(t) = α₀ × H(t)/H(0)

**Flaw 1 — α₀ dependency**: only works because α₀=1e-5 was chosen knowing 3e-6 is empirically optimal.
**Flaw 2 — Wrong proxy**: theory says ratio should be exp(H(t))/exp(H(0)) ≈ n_eff(t)/n_eff(0), not H(t)/H(0).
**Flaw 3 — Not scale-independent**: α_max doesn't transfer across different models/tasks/batch sizes.

---

## Theoretical foundation (Session 3, 2026-04-20)

The improvement bound gives:

$$\mathbb{E}[J(\theta_{t+1})] - J(\theta_t) \geq D_t \cdot \alpha \left[r_t - \frac{\alpha L}{2}\right]$$

Expected improvement > 0 iff α < 2r_t/L. The key quantity is the **signal fraction**:

$$r_t = \frac{\|g\|^2}{\|g\|^2 + \sigma^2/n_\text{eff}} \in (0,1], \quad \alpha^*(t) = \frac{r_t}{L}$$

$r_t$ is the minimal sufficient statistic for the safe-zone condition. Individual values of ||g||² and σ²/n_eff don't matter, only their ratio.

**Estimator**: split batch into two independent halves A, B under the same θ_t:

$$\hat{r}_t = \frac{\hat{g}_A^\top \hat{g}_B}{\frac{\|\hat{g}_A\|^2 + \|\hat{g}_B\|^2}{2}}$$

Only requirement: same θ_t. Prompts can differ entirely.

**Shape vs scale**: tracking r_t gives the shape of α*(t) but not the scale. α(t) = C·r_t where C = α(0)/r_0. The ratio α(t)/α*(t) = C·L is constant — initial errors never self-correct without a feedback signal.

---

## Proposed framework (Session 3, 2026-04-20) — FINALIZED

$$\alpha_t = c_t \cdot \hat{r}_t$$

- **r̂_t**: feedforward signal, theory-grounded, handles time-varying shape
- **c_t**: scale factor updated by feedback, should converge to 1/L

**Why**: directly estimating L is rejected — it's noisy (finite-difference estimate is dominated by gradient noise when n_eff is small), and L is not truly a global constant in RL (policy, sampling distribution, advantage distribution all change).

---

## Feedback signal for c_t: φ_t (RESOLVED, 2026-04-20)

**Signal definition** (using held-out half-batch B):

- p_t := α_t ĝ_B^T ĝ_A  (predicted gain: first-order, uses half-batch B direction)
- a_t := L_B(θ_{t+1}) - L_B(θ_t)  (actual gain on held-out B — no gradient leakage)
- φ_t := a_t / (p_t + ε)  (realization ratio)

**Why φ_t**: measures "how much of the predicted improvement was realized". Unbiased because B is held out from the update gradient.

**Derivation of φ* = 1/2** (4 steps, fully rigorous):
1. From improvement bound: f(α) = α||g||²(1 - αL/(2r_t))
2. First-order approximation: φ_t ≈ 1 - α_t L/(2r_t)
3. Substitute α_t = c_t·r_t: φ_t ≈ 1 - c_t·L/2  ← r_t cancels exactly
4. At c_t = 1/L: φ* = 1/2

**Critical property**: φ* = 1/2 does not depend on r_t. Scale controller is decoupled from shape signal.

**Controller update rule**:
c_{t+1} = c_t · exp(η_c(φ̄_t - 1/2))

where φ̄_t is an EMA of recent φ_t values.

**Three practical protections**:
1. Skip update when |p_t| too small (estimate unreliable)
2. Clip or EMA-smooth φ_t before feeding into controller
3. Hard bounds on c_t: c_t ∈ [c_min, c_max]

**Why φ* must not depend on r_t**: if it did, the scale controller would be reacting to shape variation, defeating the two-timescale decomposition. The cancellation of r_t in step 3 is exact and necessary for the design to work.

**~~Clipping fraction as feedback~~**: ruled out — clipping reflects total α_t magnitude which is already modulated by r̂_t; cannot isolate c_t error.

---

## 4.21 诊断：真实运行形态 vs 设计意图（2026-04-21）

### 真实运行形态

4.21 实验（sigfrac_lr1e-5 / sigfrac_lr1e-6，两轮，共 300 步）显示：

```
实际：warmup + handoff + sign-gated r-side scaling + dead c-side
设计：full two-timescale controller with three-condition validity guard
```

**c_t 状态**：两轮实验 c_t 只有两个唯一值——warmup 期初始值，以及 handoff 计算值（= α_base / r_boot = α_base / 0.05）。handoff 之后 300 步全程冻结。

**根因**：p_min = 1e-8 是绝对阈值，实际 p_t = α_t × g_dot ≈ 1e-6 × 7e-6 = 7e-12，比 p_min 小 4 个量级。

### 关键概念澄清

问题不是"量纲错误"，而是**缺乏归一化的绝对阈值，不具备尺度鲁棒性**。p_t 的量级随 LR、梯度范数、模型维度而变，固定的 p_min 与这些量不共变。结果：c_t controller 被人为绑回初始 LR，破坏 factorization 的理论含义（c → 1/L 与 base LR 无关）。

**φ_t 本身是无量纲的**（a_t 和 p_t 同量级，比值 ≈ O(1)），数值上在 float64 精度范围内完全稳定。clip 到 [-2, 2] 已防止幅度爆炸。因此 p_min 的唯一合理角色——防止 float 数值不稳定——在当前实现中并不需要。

### c_t guard 设计原则（已确立）

**已删除**：`abs(p_t) > p_min`（绝对阈值，破坏 scale-invariance）

**保留**：`g_dot > 0`（方向 guard，无量纲，scale-free）

**待实验验证**：仅保留 g_dot > 0 是否足够，或是否需要追加相对化 guard。

若需要额外 guard，优先候选是**余弦相似度 guard**：

$$\frac{\langle \hat{g}_C, \hat{g}_{\mathrm{upd}} \rangle}{\|\hat{g}_C\| \|\hat{g}_{\mathrm{upd}}\|} > \tau$$

优点：无量纲、天然 scale-invariant、语义清晰（"calibration 方向与 update 方向足够对齐"）、论文可解释性强。次优候选：相对历史尺度 `|p_t| / EMA(|p|) > τ_rel`（自归一化但解释性略弱）。暂不考虑 φ_t 方差 guard（过复杂，引入新调参层）。

### 当前可支持的 claim（4.21 数据）

1. split-batch alignment 的**符号**有一定信息量（~52–64% 步骤 g_dot > 0）
2. sign guard 可作为保守 gate，避免部分不可靠更新
3. slow calibration branch（c_t 控制器）在 p_min 删除前从未真正工作
4. validity guard 条件 2（d_t > 1e-30）和条件 3（g_rms > τ·ema）在当前量级下实际无效

### 下一步（4.21 写，已进入 4.22 实验）

跑第一轮无 p_min 实验，重点观察：
- `actor/c_t`：是否从 handoff 初值开始真实移动
- `actor/phi_t`：分布是否大量打满 clip [-2,2]
- `actor/phi_bar`：是否向 0.5 收敛
- 不同 base LR 的 c_t 是否向相近区间靠拢

---

## 4.22 实验诊断与实验方向（2026-04-22）

### 4.22 实验结果（sigfrac_lr1e-5 ~139步 / sigfrac_lr1e-6 ~95步）

**c_t 行为（无 p_min 后）**：
- c_t 有 11 个唯一值（比 4.21 的 2 个多），说明 p_min 删除后 c_t 开始更新
- 但更新方向**持续向下**：phi_bar 从 0.45 单调降至 0.157（lr1e-5）
- 根因：**phi_t ≡ 0 bug**（见 engineering_impl.md），导致 EMA 公式 phi_bar_{t+1} = 0.9*phi_bar_t + 0.1*0 = 0.9^n × phi_bar_0 → 0

**r-side 行为**：
- r̂_t 噪声大：38% 步骤 g_dot < 0
- EMA 后典型值 r̄_t ≈ 0.016–0.020，远小于 r_boot=0.05（warmup 期 EMA）
- 这导致 handoff 时 c_T = α_base / r_boot = α_base / 0.05（因 r̄_warm < r_boot，max 取 r_boot）

**handoff 机制有效**：
- lr1e-5: c_T = 1e-5/0.05 = 2e-4 → 典型 α_t = 2e-4 × 0.016 ≈ 3.2e-6（接近已知最优 3e-6）
- Handoff 校准到了正确量级，但 phi_t=0 bug 使 c_t 随后持续被驱向下

**性能**：lr1e-5 在 bug 存在的情况下仍达 AIME24=0.300（step 120），仍在上升。

### 核心问题定位：两个自由度不宜同时验证

- c_t controller 是否正确 → 现在还无法判断（phi_t 一直是 0 或接近 0）
- r̂_t estimator 是否有独立信息量 → 还没有干净对照

**解决思路**：先定住 c，再单独审问 r。

### 两阶段实验设计（2026-04-22 确定）

**Phase 1：找 c_fixed（eta_c=0，handoff 自动定 c）**

设 `eta_c=0`，冻住 c_t。handoff 仍运行，c_T = α_base/r_boot = α_base/0.05。扫三个 α_base：

| α_base | c_fixed | 典型 α_t (×r_typical=0.018) |
|--------|---------|---------------------------|
| 7.5e-6 | 1.5e-4  | ~2.7e-6 |
| 1.0e-5 | 2.0e-4  | ~3.6e-6 |
| 1.25e-5 | 2.5e-4 | ~4.5e-6 |

注意：c_fixed ≠ 最优固定 LR（3e-6），因 α_t = c_fixed × r_t，r_t ≈ 0.018，所以 c_fixed ≈ 3e-6/0.018 ≈ 1.7e-4。

脚本已创建：`new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr{7.5e-6,1e-5,1.25e-5}.sh`

**Phase 2：测 r_t 独立信息量（固定 c_fixed 后）**

Version A — sign-gate：α_t = α_target if g_dot > 0, γ×α_target if g_dot ≤ 0（γ = 0/0.2/0.5）
Version B — continuous r-shaping：α_t = c_fixed × r_ctrl（等价于 α_target × clip(r_ctrl/r_ref, m_min, m_max)）

判定：Version A 有收益 → r_t 至少有方向筛选价值；Version B > A → 连续值有额外信息。

**对比基线**：固定 LR = 3e-6（已知最优）。

---

## 4.23 理论诊断：r̂_t 的 statistical feasibility problem

### 核心结论（三层）

**第一层**：slow scale（c_t / handoff）是主导因素，已被 Phase 1 确认。

**第二层**：fast reliability 幅度（r̂_t 的连续值）在小 r_t regime 下难以稳定估计，原因叠加：
1. **分子 SNR 低**：E[ĝ_A1^T ĝ_A2] = ||g||²，但噪声量级为 tr(Σ)/n_eff，比值 = (1-r_t)/r_t。当 r_t=0.02 时噪声幅度是信号 49 倍。
2. **ratio estimator instability**：分母 (||ĝ_A1||² + ||ĝ_A2||²)/2 本身也是随机变量，与分子共享同一批 noisy gradients。小信号 regime 下分母的随机波动造成比值的虚高/虚低，而非单纯分子噪声。
3. **group structure 破坏独立性**：A1/A2 按 prompt-group 切割，group 内 8 条 response 高度相关，有效样本数更接近 group 数而非 response 数，进一步恶化 SNR。
4. **L 是 moving target**：理论要求 c_t → 1/L，但 RL 中 effective local L 随 policy/advantage/sampling distribution 漂移，c_t 在追移动靶。
5. **PPO clipping**：当 clip active token 比例非小时，L-smooth surrogate 对真实更新行为的解释力显著下降。

**第三层**：因此，当前 setting 下更合理的 fast control 形式，可能不是 continuous multiplier，而是提取该信号中统计上最可信的粗粒度特征——alignment sign。

### Sign-gate 的统计定位（重要）

sign-gate 不是 continuous r-shaping 的"降级版"，而可能是当前 regime 下**统计上最合适的信号提取方式**：
- continuous r-shaping 隐含要求：符号可靠 + 幅度可靠 + 分母尺度可靠
- sign-gate 只要求：sign(ĝ_A1^T ĝ_A2) 整体上有判别力

当前数据：g_dot > 0 比例 = 55–66%（三实验），即 5–16 个百分点超出随机基准。信号存在但微弱。

**设计含义**：5–16% 的方向优势不足以支持单步硬门控（误判率高）。sign 信号需要低通平滑/多步确认/hysteresis 等积累机制，将微弱优势积累到统计显著性水平后再作用于 LR。这是理论上必要的，不是可选的工程谨慎。

### 论文叙事升级

旧叙事："我们提出 continuous adaptive LR"
新叙事：
1. 理论分解出 α_t = c_t · r̂_t（slow scale × fast reliability）
2. Phase 1 确认 slow scale 可由 handoff 自动校准，且是主导因素
3. 理论分析表明 fast reliability 幅度在小-r_t regime 下存在根本性的 ratio estimator instability
4. 从该信号中识别出统计上最可信的分量：alignment sign
5. sign-gate 是对"可靠信息"的最大化提取，而非退而求其次

### Phase 2 三角对照（✅ 完成，2026-04-24 分析）

三组 expected mean alpha 完全对齐（均≈2.97e-6），300步全部完成：

| 组 | alpha 设计 | 实测 mean alpha | 状态 |
|---|---|---|---|
| M | 固定 2.97e-6（sign_gate_gamma=1.0） | 2.97e-6 | ✅ 300步 |
| A | g_dot>0: 3.71e-6 / g_dot≤0: 1.855e-6（gamma=0.5） | 2.89e-6 | ✅ 300步 |
| B | c_fixed=2.5e-4, continuous r-shaping | 2.97e-6 | ✅ 300步 |

### Phase 2 核心结论（2026-04-24）

**B 是整体最强**：5/5 benchmark 超过 M；4/5 benchmark final 最优

| Benchmark | M final | A final | B final |
|-----------|---------|---------|---------|
| AIME24    | 0.2938  | 0.3229  | **0.3271** |
| AIME2025  | 0.2271  | **0.2646** | 0.2396 |
| OLYMPIAD  | 0.4888  | 0.4850  | **0.5056** |
| GPQA      | 0.3663  | 0.3419  | **0.3725** |
| MINERVA   | 0.2794  | 0.2737  | **0.2900** |

**关键修正（不能再说 "A ≈ B，所以 sign 足够"）**：
- A 在 OLYMPIAD/GPQA/MINERVA 上**低于 M**（GPQA: A=0.3419 < M=0.3663），sign-gate 刹车对宽 benchmark 过于粗糙
- B 在宽 benchmark 上的优势说明 continuous magnitude 不是纯噪声，在这类任务有独立价值
- A 的价值体现在 AIME 类：peak-to-final drop 最小（-0.008 vs M 的 -0.044），在 AIME2025 final 最优

**Entropy 异常**：B 的 entropy mean 显著低于 M/A（0.362 vs 0.450），但性能反而最好。低 entropy + continuous r-shaping 的关系待进一步研究。

**KL/grad norm 三组完全对齐**（KL=0.0004，gradnorm≈0.039）——性能差异不来自整体更新强度。

**不能声明的内容**：
- φ̄_t=0.5 **不是** c_t controller 收敛证据（本轮 c-side 关闭，eta_c=0）
- P(g_dot>0) 的统计显著性不能用 iid 检验（时序相关）

**Sign-gate 动力学（2026-04-24 初步分析）**：

| 阶段 | M P(g_dot>0) | A P(g_dot>0) |
|------|-------------|-------------|
| warmup (1-20) | 0.750 | **0.900** |
| early (21-100) | 0.475 | 0.557 |
| mid (101-200) | 0.520 | 0.600 |
| late (201-300) | 0.530 | **0.510** |

A 的 late-period 对齐率降到 0.510（接近随机），说明训练后期 sign 信号显著减弱——与 AIME2025 late 回落一致。A 的游程分析：正游程 mean=2.40（max=9），负游程 mean=1.75（median=1），说明信号微弱、以短游程为主。

完整分析报告：`exp_data/4.24/analysis_report.md`

**均值匹配推导**（A）：假设 p(g_dot>0)≈0.60，alpha+ = 2.97e-6 / (p + (1-p)×0.5) = 2.97e-6 / 0.80 = 3.7125e-6。B 的 mean 来自实测 actor/alpha_t。

**sign-gate 实现原理**（transformer_impl.py）：
- warmup 期间：`_sign_gate_r_ref is None` → sign-gate 不活跃
- handoff 时：`_sign_gate_r_ref = max(r_ema, r_boot)`；若 `alpha_plus` 存在则覆盖 c_T
- post-handoff：g_dot>0 → `r_ctrl = r_ref`；g_dot≤0 → `r_ctrl = gamma × r_ref`
- M（gamma=1.0）：两分支相同 → 常数 LR

**新增监控指标**：`actor/g_dot_positive`（每步 0/1），用于核对实际 p(g_dot>0) 是否接近 0.60，进而核对 A 的实际 mean alpha。

---

## Sign-Gate 深度分析（2026-04-24）

### Late-period P(g_dot>0) 下降的机制

A 的 late-period 对齐率从 0.600 跌至 0.510，低于 M 的 0.530。

**最可能的根因**：低熵 regime 下 split-batch 方向估计噪声放大。A 比 M 更快到达低熵状态（entropy final: A=0.334 < M=0.390），在低熵 regime 中少数极端样本的梯度贡献更大，两半批次更容易出现方向分歧，sign 信号 SNR 下降。这是结构性的，不是可修复的 bug。

其他机制（自然收敛预测 P→升高，与观测矛盾；sign-gate 反馈机制理论上可能，但证据弱）。

### Late-period 信号衰弱与 AIME2025 的关系

A 在 AIME2025 final=0.2646（peak=final，drop=0）。这是"巧合性稳定"：
- late-period P(g_dot≤0)↑ → 刹车频率↑ → effective LR 进一步降低（A late mean=2.801e-6 vs M的2.970e-6）
- 这个自动降速恰好对 AIME 的 late degradation 有保护效果，但并非设计上的正确性

### GPQA: A < M 的机制

GPQA: A=0.3419 < M=0.3663。gamma=0.5 的刹车在 GPQA 对应步骤过于保守，抑制了有效更新。这是最干净的反例，KL/gradnorm 均值三组完全一致，排除了整体更新强度的解释。

### 游程分析的含义

正游程 mean=2.40（max=9），负游程 mean=1.75（median=1）。

**关键发现**：超过 50% 的负对齐步骤只持续 1 步就反转（负游程 median=1）。当前 sign-gate 每步独立判断，对这些单步负对齐触发 γ=0.5 刹车——但下一步就会翻正，造成过度刹车。

**设计含义（按优先级）**：
1. **Hysteresis（k=2）**：连续 k 步 g_dot≤0 才刹车，过滤单步负游程。实现简单，对 GPQA 过刹车有直接改善。
2. **EMA g_dot gate**：用 EMA(g_dot) sign 替代当步 sign，天然 hysteresis 效果。
3. 如果要走这个方向，本质上已接近 continuous r-shaping（B 的路线），不如直接用 B。

### 三个核心结论

1. **sign 信号在当前 regime 是弱信号，AIME stabilization 是任务特异性的**：GPQA 上 A 的净效果为负（A < M），不能声明"sign-gate 是通用改进"。
2. **late-period sign SNR 衰退是结构性的**：随训练推进策略集中、熵降低，split-batch 方向估计 SNR 系统性下降，与 γ 选择无关。
3. **continuous magnitude 不是纯噪声**：GPQA B > M > A，说明幅度信息在宽任务上有实际价值，sign-only 假设已被推翻。

### Seed 重复实验（2026-04-24 启动）

为量化单次运行的方差，补充 seed=0 和 seed=1 的 M/A/B 三组（共 6 个新实验）：

| 脚本 | seed |
|------|------|
| `sync_matched_alpha_2.97e-6_seed0.sh` | 0 |
| `sync_matched_alpha_2.97e-6_seed1.sh` | 1 |
| `sync_sign_gate_gamma0.5_seed0.sh` | 0 |
| `sync_sign_gate_gamma0.5_seed1.sh` | 1 |
| `sync_sigfrac_cfixed_lr1.25e-5_seed0.sh` | 0 |
| `sync_sigfrac_cfixed_lr1.25e-5_seed1.sh` | 1 |

原始 seed=42（3组）+ 6组 = 9次运行，3 seed × 3 condition。分析脚本（analyze.py）需更新以支持 mean ± std 汇总。

---

## 4.27 Baseline Controls: Mean Alpha, Jitter, and Decay Confounds

### 4.27 C310 / S baseline 分析结论

4.27 用户上传了 4.26 启动的 6 组 baseline 结果，分析报告：

- `exp_data/4.27/analysis_report.md`

条件：

- **C310**: constant LR = `3.10e-6`，对齐 B 的实测 post-handoff mean alpha。
- **S**: alpha-shuffled control，保留 B 的 alpha 分布，打乱 alpha 与 step/alignment 的对应。
- 对照参照：4.26 的 **B** continuous r-shaping。

核心结论：

1. **B 的收益不能归因于 mean alpha 更高**：
   - B step21-300 mean alpha ≈ `3.0953e-6`
   - C310 step21-300 alpha 严格为 `3.1000e-6`
   - avg5 final: B=`0.3443`, C310=`0.3322`
   - paired avg5 diff B-C310=`+0.0121 ± 0.0067`，3/3 seed 为正

2. **B 的收益不能归因于普通 LR jitter**：
   - S step22-300 与 shuffled replay 序列逐点一致
   - avg5 final: S=`0.3324`，基本等于 C310=`0.3322`
   - paired avg5 diff B-S=`+0.0120 ± 0.0125`，2/3 seed 为正
   - OLYMPIAD 上 B-S 为 3/3 seed 正

3. **必须披露的机制细节**：
   - alpha replay 日志存在 1-step offset：`REPLAY_START_STEP=21` 时，日志 step22-300 完全 replay；日志 step21 是 replay 前的 signal-fraction alpha。
   - 这个 offset 不推翻 shuffled control，但论文/报告中不能写成 step21-300 完美 replay。

可声明：

- Adaptive gain allocation matters beyond average LR scale.
- The marginal distribution of alpha is insufficient; coupling alpha to the training state/alignment signal is needed to reproduce B's overall gains.

不可声明：

- B wins every benchmark（AIME2025/MINERVA 上 C310 与 B 持平或略高）。
- shuffled replay 完美覆盖 step21-300（只能说 logged step22-300 exact replay）。

### Remaining confound: state-independent decay

4.27 分析后，剩余最关键质疑变为：

> B 是否只是学到了一个普通 state-independent early/mid/late decay schedule，而不是依赖 step-level alignment-conditioned allocation？

C310 和 S 已经排除：

- 不是 mean alpha 更高；
- 不是无序 LR jitter。

但它们没有排除：

- 一个手工设计的 monotone/piecewise decay schedule 也能达到 B。

因此下一步最关键 baseline 是 **D3: piecewise stage-matched decay**。

### D3 design: strongest hand-crafted decay baseline

D3 直接复制 B 的阶段级 alpha 形状，但去掉 per-step alignment coupling。

使用 4.26 B 三 seed pooled 阶段均值：

| phase | steps | alpha |
|---|---:|---:|
| early | 21-100 | `3.286384633e-6` |
| mid | 101-200 | `3.370729357e-6` |
| late | 201-300 | `2.667058466e-6` |
| weighted post mean | 21-300 | `3.0953198319e-6` |

注意：B 不是严格 monotone decay；mid 略高于 early。因此 D3 比简单 linear/cosine 更强，因为它复制了 B 的粗粒度非单调形状。

D3 的判定逻辑：

- 若 **B > D3**：coarse hand-crafted decay 不足以复现 B，step-level alignment-conditioned allocation 有独立价值。
- 若 **D3 ≈ B**：B 的主要收益可能是自动形成了一个有效 coarse schedule，claim 需要降级为“自动发现 schedule”，而不是“细粒度 step-level allocation 必要”。
- 若 **D3 > B**：当前 B 不是最终最优方法，alignment signal 仍有诊断价值，但实现需结合/改进 decay prior。

### 4.28 D3 results and interpretation update

用户上传了 D3 三 seed 结果到 `exp_data/4.28/`。

5-task core avg final:

| family | runs | best core | final core | final - best |
|---|---:|---:|---:|---:|
| B: `sigfrac_cfixed_lr1.25e-5` | 3 | `0.3443 @300` | `0.3443` | `0.0000` |
| D3: `piecewise_stage_matched_decay_d3` | 3 | `0.3372 @270` | `0.3360` | `-0.0011` |
| alpha shuffled | 3 | `0.3405 @290` | `0.3324` | `-0.0081` |
| matched alpha 3.10e-6 | 3 | `0.3328 @280` | `0.3322` | `-0.0005` |
| matched alpha 2.97e-6 | 3 | `0.3346 @260` | `0.3302` | `-0.0044` |

D3 轨迹稳定：core avg 从 `0.2248` 提升到 `0.3360`，peak `0.3372@270`，末尾基本不掉。`actor/alpha_t` sanity-check 显示 step21-100 / 101-200 / 201-300 精确匹配 replay schedule。

Interpretation:

1. D3 是好事，不是否定 signal-fraction。它强力支持主轴：**update scale / support matching** 是收益来源。
2. C310 / matched-alpha 不能达到 D3，说明平均 alpha 不够；时间结构很重要。
3. Shuffled alpha 不能达到 B，说明无序 jitter 不够；alpha 与训练阶段/状态的耦合很重要。
4. B 仍高于 D3 `+0.0083` core avg，说明 state-dependent signal-fraction 保留了粗 stage schedule 之外的剩余价值。

Updated claim boundary:

- 不再强写“fine-grained step-level allocation strictly necessary”。
- 更稳的论文表述：

> Signal-fraction primarily discovers and tracks a support-matched update-scale schedule. A strong stage-matched open-loop schedule recovers much of the gain, but still underperforms the adaptive controller, indicating residual value from state-dependent allocation.

## 7B transfer plan（2026-04-28）

Next model-scale test: DeepSeek-R1-Distill-Qwen-7B.

First step is to re-sweep `c_fixed`; do not copy 1.5B's `c_fixed=2.5e-4` directly. Reasons:

- `c_fixed` approximates `1/L`; model scale changes effective curvature.
- 7B can change gradient RMS and A1/A2 alignment distribution.
- The cleanest migration test is one degree of freedom: update scale.

Resource choice:

- Use **16 GPUs first**, not 32 GPUs.
- Current per-job cap is about `128 GPU hours`; 32 GPUs gives only ~4h wall-clock, while 16 GPUs gives ~8h and is better for checkpoint/resume.

First 7B sync c-fixed sweep:

| BASE_LR | implied c_fixed | role |
|---:|---:|---|
| `5e-6` | `1.0e-4` | conservative low |
| `7.5e-6` | `1.5e-4` | middle |
| `1e-5` | `2.0e-4` | high but below 1.5B B |

Use `eta_c=0.0`, `calib_frac=0.0`, `trainer.balance_batch=False`, and inspect dynamics before final scores.
