---
name: Algorithm Design — Current Framework and Open Questions
description: 6.12 Direction pivot — from adaptive LR to RL-aware optimizer; pos/neg gradient decomposition; Adam preconditioning analysis
type: project
---

## 2026-06-12 Direction Pivot — RL-Aware Optimizer Design

### 为什么不再做 adaptive LR

从 signal-fraction (α_t = c_t · r̂_t) 到 signal-quality gate 到 entropy gate，我们一直在调**标量** LR。但 RL 的核心问题在**方向**：

1. **r̂_t 本身噪声太大**：single-step g_dot_positive ≈ 55%（接近 coin flip），temporal aggregation (W10) 最终也不稳定
2. **adaptive LR 的上限是"自动发现 schedule"**：D3 piecewise decay 恢复了 B 大部分收益，说明 state-dependent allocation 的剩余价值有限
3. **LR 不改变 gradient 方向**：Adam update = α · m/(√v + ε)，α 只缩放 magnitude，不改变 D·m 的方向

真正的问题是：**Adam 的 coordinate-wise preconditioning 是否扭转了 gradient 中的 positive consolidation 信号**。

### 当前研究问题

> 在 RLVR 中，standard Adam 的 per-coordinate preconditioning 是否系统性地削弱 positive-advantage gradient 相对于 negative-advantage gradient？如果是，能否设计一个利用 RL gradient 结构的优化器来改善学习效率？

### |A⁻| 大 ≠ negative gradient update 占主导

之前过粗地说"negative advantage 占主导"。严格区分三个层面：

**1. 标量 advantage 一阶总量平衡（GRPO 结构保证）**

p = 正确率，K responses/prompt：
- Σ A⁺ = pK · √((1-p)/p) = K√(p(1-p))
- Σ |A⁻| = (1-p)K · √(p/(1-p)) = K√(p(1-p))
- **平衡**

**2. 二阶 advantage energy 偏向少数类**

- Σ(A⁺)² = pK · (1-p)/p = K(1-p)
- Σ(A⁻)² = (1-p)K · p/(1-p) = Kp
- p>0.5 时：Σ(A⁻)² > Σ(A⁺)²

但这只说明 per-sample squared energy 偏向 negative，不说明**净梯度方向**一定偏 negative。

**3. 真正的 update 看向量和**

u⁺ = Σ_{A>0} A·s,  u⁻ = Σ_{A<0} A·s

决定 update 的是 ‖u⁺‖, ‖u⁻‖, cos(u⁺, u⁻)，不是 ΣA²。

即使 ΣA²⁻ > ΣA²⁺，也可能：
- 正样本方向更一致 → ‖u⁺‖ 更大
- 负样本方向分散、互相抵消 → ‖u⁻‖ 不大
- 正负方向不一定相反

**→ 必须实验测量。这就是 P0 pos/neg gradient decomposition 的目的。**

### P0 → P1 → P2 路线图

| Phase | 问题 | 方法 | 所需数据 |
|---|---|---|---|
| **P0** | g⁺ vs g⁻ 谁主导？方向关系？ | lm_head exact gradient decomposition | 每 5 步采样 64 responses |
| **P1** | Adam D_t 是否扭转 g⁺ 方向？ | cos(m, g⁺) vs cos(Dm, g⁺) | P0 数据 + Adam state |
| **P2** | 设计 RL-aware optimizer | 基于 P0/P1 诊断结论 | — |

### P0 判断矩阵

| ‖g⁺‖ vs ‖g⁻‖ | cos(g⁺,g⁻) | 解读 | 下一步 |
|---|---|---|---|
| ‖g⁺‖ >> ‖g⁻‖ | ≈ -1 | 正负近似反向，正占主导 | 检查 Adam D 是否压 g⁺ (P1) |
| ‖g⁺‖ ≈ ‖g⁻‖ | ≈ -1 | 正负平衡、近似反向，净 update 很小 | 检查 ‖g_total‖ 是否过小 |
| ‖g⁺‖ ≈ ‖g⁻‖ | ≈ 0 | 正负近似正交 | 可独立调节两方向 |
| ‖g⁻‖ >> ‖g⁺‖ | any | negative 确实主导 | 需要 positive auxiliary |

### 与之前 Adam direction hypothesis (6.8b) 的关系

6.8b 提出的 cos(m, g⁺) vs cos(Dm, g⁺) 诊断框架**保留**，但现在是 P1 而非 P0。理由：要先知道 g⁺ 和 g⁻ 的 baseline 关系（P0），才能有意义地检查 Adam 对它们的差异化影响（P1）。

### 降级的旧方向

| 方向 | 状态 | 理由 |
|---|---|---|
| Signal-fraction LR (α_t = c_t · r̂_t) | **已关闭** | r̂_t 噪声根本性问题；LR 不改变方向 |
| Signal-quality gate (reward_std / entropy) | **已关闭** | 只防 collapse，不改善学习 |
| A²Q reweighting | **已关闭** | high-H 是有效信号不是 outlier |
| Variance decomposition (N×K) | **保留为参考** | 结论（prompt/rollout noise 对半）仍有效 |

## 2026-06-10 Exact lm_head Gradient Norm — A²Q Proxy 替代

### A²Q 的问题

A²Q 用 H = A²·Q 作为 per-response gradient energy proxy，其中 Q = Σ_t ||∇log π(y_t)||²。这假设：

```
||Σ_t J_t||² = Σ_t ||J_t||²
```

即不同 token 位置的 score function 在参数空间正交。但所有 token 共享参数、相邻 token hidden state 高度相关，cross terms Σ_{t≠s}⟨J_t, J_s⟩ 可以很大甚至主导。

### 精确公式

对 lm_head 层 W (V×d)，per-response 梯度 ∂S_b/∂W = Σ_t δ_t h_tᵀ（其中 δ_t = e_{y_t} - π_t）。

```
||∂S_b/∂W||²_F = Σ_{t,s} K_δ[t,s] · K_h[t,s]
               = trace(K_δ ⊙ K_h) 的完整版（A²Q 只用了 trace of diag）
```

K_δ 的对角线恢复 q_per_token，off-diagonal 就是 A²Q 丢掉的 cross terms。

### cross_term_ratio 的含义

定义 cross_term_ratio = ||∂S/∂W||² / Q：
- = 1 → cross terms 精确为零，A²Q 是完美 proxy
- > 1 → cross terms 正贡献（token 间梯度正相关，信号叠加）
- < 1 → cross terms 负贡献（token 间梯度反相关，部分抵消）

这个量直接回答 "A²Q 有多离谱"。

### 对后续方向的影响

| cross_term_ratio 结果 | 含义 | 下一步 |
|---|---|---|
| ≈ 1（stable across training） | A²Q 其实还行，正交近似在 lm_head 层成立 | A²Q 可继续用；问题在别处 |
| >> 1 或 << 1 | cross terms 重要，A²Q 系统性偏估 | 用 exact norm 替代 A²Q 做所有分解 |
| 随训练从 ≈1 漂移 | 训练改变了 token 间相关结构 | 动态 A²Q 校正 or 全面切换 exact |

### 为什么只算 lm_head 层

- lm_head 是最大的单层参数矩阵 (hidden_dim × vocab_size)
- 唯一直接看到 token-level loss signal 的层
- 有解析公式，不需要 per-sample backward
- 如果 lm_head 层 cross terms 不重要，其他层更不可能重要（因为梯度被 chain rule 进一步混合）
- 如果 lm_head 层 cross terms 很大，可以后续用 random projection 验证全模型

## 2026-06-08 Per-Problem Learning Analysis + Adam Preconditioning Direction Hypothesis

### Per-Problem Structure Inference (from 76 runs aggregate data)

**方法**：从 (mean@16, std@16, best@N, worst@N, maj@N, overlong) 统计推断 30 题的 per-problem 正确率分布。

**核心发现**（高置信度）：

1. **增益高度集中**：std@16 在 96.6% 的提升 run 中增大；best@16 提升是 worst@16 的 5.5×
2. **一致性是最大信号**：Δmaj@16 = 2.0× Δmean@16 (SNR=4.24, 最稳定指标)；~3 道题跨过 p=0.5 门槛
3. **约 11/30 题始终不动** (p≈0)，RL 对它们完全无效
4. **行为变化先于能力变化**：overlong 84%→51%，先于 accuracy 提升约 10-12 steps
5. **存在题型偏向**：AIME2025 提升仅为 AIME2024 的 60% (corr=0.46)

**定量分布推断** (Beta 拟合, 采样噪声修正后)：

| p_j 区间 | Base (30题) | Final (30题) | Δ |
|---|---|---|---|
| p < 0.05 | ~9 题 | ~2 题 | -6 |
| 0.05-0.30 | ~10 题 | ~8 题 | -2 |
| 0.30-0.50 | ~5 题 | ~8 题 | +3 |
| 0.50-0.80 | ~3 题 | ~6 题 | +3 |
| p > 0.80 | ~0 题 | ~1 题 | +0 |

**4 类模型估计** (mean@16 + best@16 联合约束)：
- ~4 道题从 "完全不会" 进入可探测范围
- ~5 道题从 "边界" 进入 "基本掌握" (p≈0.65)
- 仍有 ~11 道题完全未突破
- **没有** p=0→p=0.8 的突破性学习

**与 mixed-prompt/high-H 机制的一致性**：
- 提升集中在 0.1<p<0.5 = 训练时的 mixed prompts (0<success_rate<1)
- 完全不会的题 = all-wrong prompts, advantage≡0, 无梯度贡献
- 一致性提升 > 能力提升 ↔ hard-negative signal 主要压低错误模式

### Adam Preconditioning Direction Hypothesis (revised 6.8b)

**核心问题**（不是 LR 问题）：

Adam 的 update 是 Δθ = -α · m_t / (√v_t + ε)。我们研究的不是 α（LR），而是：

```
u_t = D_t · m_t,    D_t = diag(1 / (√v_t + ε))
```

u_t 是 preconditioned gradient — 模型真正执行的 update **方向**。

**命题**：Adam 的坐标级 preconditioning 将 update 方向从 positive consolidation 方向旋转开，使 boundary problems 停在 p≈0.5-0.7。

**关键区分**：两个竞争解释（必须先诊断 numerator 再看 denominator）

| 路径 | 含义 | 表征 |
|---|---|---|
| A: numerator 问题 | m_t 本身不对齐 positive consolidation | cos(m, g⁺_boundary) ≤ 0 |
| B: denominator 方向扭转 | m_t 有正向信号，但 D_t 把 u_t 拉离 g⁺ | cos(Dm, g⁺) < cos(m, g⁺) |

如果路径 A 成立 → Adam 改法无意义，应转向 positive auxiliary / sampling。
只有路径 B 成立 → denominator 干预有价值。

**注意**：uniform v*=0.1 不能区分方向 vs 步长 — 若所有坐标统一缩放 v，u 方向不变，等价于 LR ×√10。只有 coordinate-wise 非均匀 v 结构才能改变方向。

### Gradient Preconditioning Diagnostic Framework

**定义**：
- g⁺_boundary = Σ_{A>0, boundary prompts} A·s （positive consolidation gradient）
- g⁻_boundary = Σ_{A<0, boundary prompts} A·s （negative correction gradient）
- m_t = Adam first moment（EMA of gradients）
- D_t = diag(1/(√v_t + ε))
- u_t = D_t · m_t （preconditioned update direction）

**核心诊断指标**（全部归一化，不涉及 LR）：

| 指标 | 公式 | 含义 |
|---|---|---|
| cos(m, g⁺) | ⟨m, g⁺⟩ / (‖m‖‖g⁺‖) | Numerator 是否对齐 positive consolidation |
| cos(Dm, g⁺) | ⟨Dm, g⁺⟩ / (‖Dm‖‖g⁺‖) | Preconditioned update 是否对齐 |
| Δ_dir⁺ | cos(Dm, g⁺) - cos(m, g⁺) | **方向扭转量**：<0 说明 D 削弱 positive |
| cos(m, g⁻) | 同理 | Numerator 对 negative 的对齐 |
| cos(Dm, g⁻) | 同理 | Preconditioned 对 negative 的对齐 |
| κ(g⁺) | ‖D·g⁺‖ / ‖g⁺‖ | Positive 方向的 preconditioning strength |
| κ(g⁻) | ‖D·g⁻‖ / ‖g⁻‖ | Negative 方向的 preconditioning strength |
| d̄(g⁺) | Σ_ℓ(c_ℓ² · √v_ℓ) / Σ_ℓ(c_ℓ²) | Positive 方向遭遇的平均 denominator |
| d̄(g⁻) | 同理 | Negative 方向遭遇的平均 denominator |

**判断矩阵**：

| cos(m, g⁺) | Δ_dir⁺ | κ(g⁺) vs κ(g⁻) | 结论 |
|---|---|---|---|
| ≤ 0 | — | — | 路径 A：numerator 本身无 positive signal |
| > 0 | < 0 | κ(g⁺) < κ(g⁻) | **路径 B confirmed**：D 削弱 positive，保留 negative |
| > 0 | ≈ 0 | κ(g⁺) ≈ κ(g⁻) | D 对方向无差异影响 → 问题在别处 |
| > 0 | > 0 | κ(g⁺) > κ(g⁻) | D 实际帮助 positive → 假设否定 |

**Per-problem 分桶**（用 per-problem eval 结果选取）：

| 分桶 | 条件 | 用途 |
|---|---|---|
| Boundary (核心) | p ∈ [0.4, 0.7] | 检测 g⁺ 是否被 D 压弱 |
| Consolidated (对照) | p > 0.8 | 对照组 — positive 应已巩固 |
| Regressed (单题) | Problem 20 (退化) | 区分 numerator 反向 vs D 扭转 |
| Dead (负样本) | p ≈ 0 | g⁺ 应为 ~0，验证 sanity |

### Counterfactual Analysis (same m, modified D)

不做 uniform v-scale（等价于 LR 变化）。做结构性 counterfactual：

| Counterfactual | 操作 | 测试 |
|---|---|---|
| Flatten D | D_flat = (1/mean(√v)) · I | 方向退化为 m 方向 — 消除所有坐标差异 |
| Normalize D | D̃ = D / mean(D) | 去除全局 scale，只看方向改变 |
| Cap D | √v_ℓ ← min(√v_ℓ, q95(√v)) | 测试 "少数极大坐标压制 positive" |
| Channel-split | D⁺ = D restricted to g⁺-dominant coords | 看 positive 方向坐标的 v 是否系统性偏大 |

比较每个 counterfactual 下 cos(D'·m, g⁺) 是否恢复。

### Revised Experiment Priorities

| Priority | 内容 | 性质 |
|---|---|---|
| **P0 (最高)** | Gradient preconditioning diagnostic | 纯诊断，不跑训练；同一 checkpoint + 同一 batch |
| P1 | Counterfactual D analysis | 纯计算，对比 flatten/cap/normalize |
| P2 | Continuation (仅当 P0 确认路径 B) | 训练实验：cap-D 或 channel Adam |
| ~~P0 旧~~ | ~~v*=0.1 continuation~~ | ~~降级：等价于 LR change，不能区分方向~~ |

**P0 诊断实验步骤**：
1. 加载 step 200 或 250 checkpoint (含 Adam state m_t, v_t)
2. 固定一个训练 batch（含 boundary prompts）
3. Forward + backward 得到 per-response token-level gradients
4. 按 advantage sign × problem bucket 聚合得到 g⁺_boundary, g⁻_boundary
5. 读取 m_t, v_t → 计算 D_t → 计算所有指标
6. 对多个 batch 取 mean±std 确认稳定性

**成功标准**：
- 路径 B 确认：cos(m, g⁺) > 0.05 且 Δ_dir⁺ < -0.02 且 κ(g⁺)/κ(g⁻) < 0.85
- 路径 A 确认：cos(m, g⁺) ≤ 0 across multiple batches → 转向 positive auxiliary

### 不同算法组的学习模式差异

| Group | Δmaj/Δmean | Δbest/Δworst | 解读 |
|---|---|---|---|
| a2q | 1.75x | 13.8x | Reweighting 让提升极度集中在 top 题 |
| lr1e-5 | 2.84x | 3.0x | 高 LR 少数题跨门槛但整体退化 |
| lr3.1e-6 | 1.86x | 5.3x | 标准稳定训练 |
| lr_decay | 2.31x | 2.5x | 衰减让提升相对均匀 |
| signal_quality | 2.10x | 5.3x | SQ gate 无特殊影响 |
| stage2_NK | 1.75x | 4.8x | N/K 配置不改变学习模式 |

所有组都展现 Δmaj/Δmean > 1.7× → "边界题稳定化"是普遍现象。

---

## 2026-06-07 Top-H Attribution Analysis — Key Finding

**结论：high-H rollout 是有效学习信号，不是应该被压的 outlier。这直接解释了为什么 A²Q clipping 没有提升性能。**

### 三个核心发现（6 runs 一致，step 50-300 稳定）

1. **High-H 主要由 A² 驱动，不是 Q/length artifact**
   - Top-5% H 的 A² 是 global 的 5-9x，Q 只有 1.5-2.1x
   - Top-5% response 平均长度 4500-5000 vs global 2600-2800（仅 1.7x），不是极端长文
   - 结论：high-H ≈ high reward contrast，是 GRPO 学习信号的核心

2. **High-H 绝大部分来自 mixed (informative) prompts**
   - Step 100+ 后，top-5% H 中 mixed prompt 占 67-92%（远超 global 的 53-65%）
   - H_share_mixed 从 step 10 的 30-40% 升至 step 100+ 的 70-96%
   - All-correct prompts 的 H_share ≡ 0（A² ≡ 0 because all rewards equal → advantage = 0）
   - 结论：high-H 恰好集中在最有学习价值的 prompt 上

3. **High-H 以负 advantage 为主（~65-75% negative）**
   - 含义：GRPO 通过 top-H rollout 主要在**惩罚错误 response**（降低概率）
   - 这不是 noise — 这是 reward contrast 的自然结构：mixed prompt 中错误 response 的 advantage magnitude 通常更大
   - 压掉这些 rollout = 削弱对错误 response 的惩罚力度

### Vanilla vs Rollout-norm 的关键差异

| 指标 | Vanilla | Rollout-norm |
|---|---|---|
| top5 A²/global | 7.8-8.9x | 4.9-5.2x |
| top5 H share | 0.53-0.57 | 0.31-0.34 |
| neff_h_ratio | 0.11-0.12 | 0.23-0.27 |

Rollout-norm 确实降低了 concentration（neff 翻倍），但也降低了 A² ratio — 把最有区分力的 rollout 压平了。

### 方向修正

**不应继续做 A²Q clipping / reweighting**。High-H 不是 outlier，而是：
- 来自 mixed (informative) prompts
- 由 reward contrast (A²) 驱动
- 主要方向是惩罚错误 response

**更有价值的方向**：
- 增加 mixed prompt 比例（prompt selection / difficulty-aware sampling）
- Q-only normalization（如果要处理 score-energy artifact，不应连带压 A²）
- 条件 reweighting：只在 H high ∧ Q high（Q 主导的 outlier）时压

### 分析产物

- `exp_data/a2q_topH_attribution_analysis.py` — 离线分析脚本
- `exp_data/a2q_topH_attribution_results.txt` — 完整结果
- `verl/trainer/ppo/a2q_reweighting.py` — 新增 15 个在线 attribution metrics

---

## 2026-06-03 Day Summary

**完成**: 综合报告修订 + 认知修正 + 文档全面更新 + 实验脚本确认就绪

**关键决策**: 主实验从 LR=3.1e-6 stability story 转向 LR=1e-6 gradient composition story。

**实验状态**: 15 个 LR=1e-6 脚本已就绪（`new_experiments/a2q_reweighting/sync_a2q_lr1e-6_*.sh`），分三批优先级启动。核心对比 = vanilla vs rollout-only normalized (6 runs)。

**待办**: 启动第一批实验 → 等结果 → 分析 final avg5 / AUC / concentration metrics。

---

## 2026-06-03 综合报告修正与方向确认

### 研究问题重新定义

**旧问题**（已降级）：How to prevent collapse at high LR (1e-5, 5e-6)?

**新问题**（当前主线）：
> In a stable low-learning-rate RLVR regime, can we improve final performance by improving the composition of the policy-gradient estimator, rather than by changing the learning-rate schedule?

标准 GRPO: `ĝ_i = (1/K) Σ_k A_{i,k} s_{i,k}`
Rollout-robust normalized: `g̃_i = (1/K) Σ_k (w̃_{i,k}/w̄) A_{i,k} s_{i,k}`

### 5 条认知修正

1. **主问题不是稳定性，是低 LR 下的有效学习** — 实验应在 LR=1e-6
2. **A²Q 收益不是"防 collapse"，是"改 gradient composition"** — normalized 版本关键
3. **Prompt-level 不能简单按 E_i clip** — 核心是 informativeness (I_i = 4r̄(1-r̄))，不是 high-energy
4. **Rollout-level correction 最干净** — H_{i,k} 直接对应 per-rollout gradient leverage
5. **Entropy 是 diagnostic，不是主指标** — 低 LR 下 entropy 低不等于 failure

### 当前证据状态

- A²Q hierarchical @ 3.1e-6: final mean 0.3309 vs A1 0.3271 (Δ=+0.0038) — **弱正信号，未证明**
- "hierarchical > rollout-only > vanilla" 排序：**未被严格证明**（incomplete seeds + code-path confounds）
- "A²Q causes entropy collapse"：**因果未确认**（A2Q vanilla 不是 clean run）

### 实验 A：LR=1e-6 主性能实验

| 方法 | 说明 | 优先级 |
|---|---|---|
| Vanilla | baseline | 核心 |
| Rollout-only normalized | 主方法（mean(w)=1, 只改 composition） | 核心 |
| Rollout-only unnormalized | effective LR 下降对照 | 核心 |
| Hierarchical normalized | prompt-level 帮忙还是伤害 | 次要 |
| Prompt-only normalized | E_i clip 是否有害 | 次要 |

**核心对比**: vanilla vs rollout-only normalized

**解释力分析表**:

| 结果 | 解释 |
|---|---|
| rollout-only normalized > vanilla | A²Q 改善 gradient composition ✓ |
| rollout-only unnormalized > vanilla | 可能是 filtering 也可能是 effective LR 变化 |
| unnormalized < normalized | 低 LR 不能再降 update scale |
| hierarchical < rollout-only | prompt-level weighting 压掉有用 signal |
| prompt-only < vanilla | prompt E_i clipping 目标不对 |

分析指标：final avg5, AUC, last-5 eval mean, H concentration before/after, n_eff, update norm

---

## 2026-06-03 Stage 1 中期结论（保留，context-adjusted）

### 核心发现

1. **Rollout-only > Hierarchical > Vanilla @ step 200**
   - rollout_only 3-seed mean: 0.3366 (vs vanilla 0.3263, +0.0103, 三 seed 全正)
   - hierarchical 3-seed mean: 0.3286 (仅 +0.0023, 不显著)
   - ⚠️ 注意：最终 300-step 结果中 hierarchical 更好（vanilla 未完成），这些排序尚不确定

2. **Prompt-level control 可能有害**
   - High E_i prompts 不一定是 noise，可能是 informative prompts
   - ⚠️ 修正：问题不是 "high-energy = outlier"，而是 informativeness vs all-correct/all-wrong

3. **当前 clip 力度太弱（weight_mean=0.988）**
   - 在 3.1e-6 下几乎无实质影响
   - 在 1e-6 下可能需要更强 clip 才能看到 composition 差异

4. **Gini 实现符号 bug（已修复）**

### 理论支撑（保留）

- Rollout-level high H 更像 estimator noise
- Prompt-level high E_i 可能是 useful task signal
- Downweight prompt 引入 bias 风险更大

### Bug fixes (2026-06-03)

- Gini sign fix: `(n+1-2*ranks)` for descending sorted data
- 新增 `normalize` 参数：`w_apply = w / mean(w)` 保持 update magnitude
- 新增 energy-weighted 诊断指标

---

## 2026-06-02 方向转折：A²Q-Guided Hierarchical Gradient Reweighting

### 核心变化

从 adaptive LR scheduling (α_t = c_t · r̂_t) 转向 gradient estimator 改进 (ĝ → g̃)。
干预点从 Adam 之后移到 Adam 之前。

### 方法

标准 GRPO 梯度：ĝ = (1/NK) Σ A_{i,k} s_{i,k}
重加权后：g̃ = (1/NK) Σ w_i w_{i,k} A_{i,k} s_{i,k}

其中：
- H_{i,k} = A²_{i,k} Q_{i,k}（gradient energy proxy）
- E_i = Σ_k H_{i,k}（prompt energy）
- w_{i,k} = min(1, sqrt(τ_r / (H_{i,k} + ε)))（rollout-level soft clip）
- w_i = min(1, sqrt(τ_p / (E_i + ε)))（prompt-level soft clip）
- τ_r, τ_p 用 global percentile + EMA

### 实现状态

- `verl/trainer/ppo/a2q_reweighting.py`：核心模块
- `verl/trainer/config/algorithm.py`：6 个新配置字段（a2q_reweight_*）
- `verl/trainer/ppo/ray_trainer.py`：在 compute_advantage 之后、_update_actor 之前调用
- 方案 A：从 batch 中的 sum_pi_squared + old_log_probs 计算 Q，不改 actor 代码
- 前提：实验脚本须设 `actor.calculate_sum_pi_squared=True`

### 与之前工作的关系

- 之前的 A²Q noise decomposition 理论框架保留，成为新方法的基础
- Signal-fraction LR 路线暂停（r̂_t 噪声、c_t 追移动靶等困难被绕过）
- 5.19 normalization 校正（B vs W/K）直接成为理论基础

### 实验计划

- Stage 0: instrumentation validation（safe LR 3.1e-6，确认 metrics 正确）
- Stage 1: fixed-LR 4-way ablation（vanilla / rollout-only / prompt-only / hierarchical）
- Stage 2: mildly aggressive LR (5e-6) 测试
- Stage 3: threshold sensitivity
- Stage 4: N/K robustness

详细设计：`exp_data/README_A2Q_Hierarchical_Reweighting.md`

---

## 2026-06-01 AQH Closed-Loop Results & Next Direction

### 核心结论

A4 signal-quality LR (reward_std gate, base 1e-5) **未达目标**：
- 两个 seed 都完成 300 步，但 best@90-120 → final 0.27-0.28，持续下降
- 比 A2 catastrophic seed42 (0.10) 安全，但比 A2 lucky seed0 (0.327) 差

### 失败机制：self-masking feedback + entropy collapse

1. Gate 只降 LR 10-15%（有效 LR 8.6-9.0e-6，远超安全区 3.1e-6）
2. 轻微降 LR 保住了 reward diversity → reward_std 没大幅下降 → gate 不知道要更强刹车
3. 真正的 failure 走 policy-space 通道：entropy 0.45→0.08 → per-token q 77% collapse → score-gradient 消失 → 锁死在窄模式

### 两类 failure channel

| Type | 名称 | 指标 | A4 状态 |
|---|---|---|---|
| I | Reward-side signal degradation | frac_informative, reward_std, A², CV²(A²) | 稳定（gate 有效保护） |
| II | Policy-side score-energy collapse | entropy, per-token q, Q_response | **崩溃**（gate 完全未检测） |

**结论**：只用 Type I 指标的 gate 会漏掉 Type II collapse。

### A²/Q/H 结构稳定

- CV²(A²) >> CV²(Q)，A²-only ~50% H variance，Q-only ~3-5%，interaction ~40-45%
- Q between_frac > 92%（强 prompt-level 属性）
- 以上在 A1/A2/A4 所有 run 间一致，不是新 pathology

### Offline replay 确认 entropy gate 有效

| Gate | seed0 first<5e-6 | seed0 first≤3.1e-6 | seed42 first<5e-6 | seed42 first≤3.1e-6 |
|---|---|---|---|---|
| Entropy | step 58 | step 92 | step 66 | step 108 |
| Per-token Q | step 68 | step 102 | step 77 | step 132 |

Entropy gate 更早触发（step 58-66），与 A4 validation peak (step 90-120) 对齐。

### 下一步方法

```
q_entropy = clip(EMA(entropy) / entropy_ref, 0.31, 1.0)
q_total = min(q_info, q_entropy)
lr = base_lr * q_total
```

优先配置：
- **A4c**（最高优先）：base_lr=5e-6, reward + entropy gate
- A4d：base_lr=1e-5, reward + entropy gate
- 同时重跑 A1 (constant 3.1e-6) 和 A3 (cosine decay) 作为 baseline

详细报告：`exp_data/aqh_closedloop_report.md`

## 2026-05-20 Signal-Quality-Aware LR Scaling

### 方法定位

**不是** optimal LR estimator，**而是** Signal-Quality-Aware LR Scaling。

```
α_t = α_base · q_t,    q_t ∈ [q_min, 1]
q_min = α_safe / α_base = 3.1e-6 / 1e-5 ≈ 0.3
```

含义：前期用 aggressive LR 学得快，后期 signal quality 下降时自动回退到 known-safe LR。

### 离线验证结论

1. **q_info 对 high-LR degradation 有预测力**：lr1e-5 三个 seed 全部在 validation drop 前/同时 q 下降
2. **q_info 是 risk indicator，不是 deterministic predictor**：seed42 q 暴跌到 0.30，但 val 只掉 -0.005
3. **stable LR 下 q 自然下降但不危险**：lr3.1e-6 下 reward informativeness 随训练降低（模型变好 → 更多 all-correct），但因 LR 低不造成伤害
4. **支持我们的定位**：真正的风险不是 q_t↓ 本身，而是 α_eff = α_base × q_t 是否仍然过高

### Sensitivity 分析结论

| I_t 候选 | high-LR 刹车力度 | stable-LR 误报 | 判定 |
|---|---|---|---|
| **reward_std (mean)** | **-0.28** | **q_late=0.82** | ★ 最佳 |
| frac_informative | -0.025 | q_late=1.00 | 太弱，几乎不刹车 |
| bernoulli_var p(1-p) | **+0.02** | q_late=1.00 | **完全失败** — 无法区分模型变好和 collapse |
| reward_var_mean | -0.20 | q_late=0.64 | 可用但 stable LR 下过度刹车 |

β_I 在 0.7–0.95 间差异很小。β=0.95 false-alarm 最低。第一版用 β=0.9（更保守）。

**bernoulli_var 失败原因**：用的是全局 success_rate/mean 的 p(1-p)，p=0.5 时最大，模型变好或 collapse 都让 p 偏离 0.5 → 无法区分方向。

### seed1 是 info-only gate 的薄弱点

seed1: q 只从 0.94 降到 0.88，但 val drop 达 -0.0746。可能原因：
- seed1 退化不是 reward informativeness 驱动
- 可能是 concentration / ratio-tail / KL risk
- 如果 Group C 不修复 seed1，Group D (concentration gate) 有必要

### 分布式一致性修复

`_compute_signal_quality_indicators()` 重写：
- 每个 rank 算 local sums → `all_reduce(SUM)` → global mean
- 所有 rank 得到相同 q_t
- **不再用 median**（无法 all_reduce）；`reward_std_median` 配置实际计算 `mean_i[sqrt(Var_k(R_i))]`
- 始终用 uid 做 prompt group 聚合，不假设 batch 排列

### 额外 logging（不参与控制，用于离线比较）

每步记录 `actor/sq_alt_reward_var_mean`, `actor/sq_alt_bernoulli_var`, `actor/sq_alt_frac_informative`。
如果 reward_std 对某 seed 不敏感，可以离线判断换哪个，不用重跑。

### Scheduler timing

`q_t` 从 step t 的 batch 计算，应用于 step t+1 的 LR。
流程：`update_policy()` → metrics → `update_signal_quality(I_t)` → `scheduler.step()`

### 实验状态

**Group A/B**: 已有 Stage 1 数据（lr3.1e-6 和 lr1e-5，各 3 seeds）

**Group C** (待启动): info-only gate

```bash
SQ_BASE_LR=1e-5 SQ_TAG=sq_info SEED=42 NCPUS=32 bash sync_signal_quality_lr.sh
SQ_BASE_LR=1e-5 SQ_TAG=sq_info SEED=1  NCPUS=32 bash sync_signal_quality_lr.sh
SQ_BASE_LR=1e-5 SQ_TAG=sq_info SEED=0  NCPUS=32 bash sync_signal_quality_lr.sh
```

**Group D** (C 之后): info + concentration gate

```bash
SQ_BASE_LR=1e-5 SQ_TAG=sq_info_conc SQ_USE_CONC=true SEED={42,1,0} NCPUS=32 bash sync_signal_quality_lr.sh
```

### 分析顺序

1. 先看 `actor/sq_alpha_t` / `actor/sq_q_info` 动力学
2. 确认 late α 在 3–6e-6 范围
3. 再看 validation: peak-to-final drop, seed variance
4. 比较 `actor/sq_alt_*` 替代指标

### 预期

- seed0: 最可能改善（offline q drop 到 0.50，val drop -0.085）
- seed42: 可能无显著提升（原 drop 只 -0.005）
- seed1: 关键测试 — 如果 C 修不了 seed1，D 有必要

---

## 2026-05-19 Stage 2 N×K 分析完成 + Normalization 校正 + 二维 Noise 框架

### Stage 2 run status

9 runs 中 6 个成功（n256k4 仅 seed42，n128k8 两 seed，n64k16 全 3 seed），3 个失败/缺失（外部 kill）。
全部 9 runs 正在用更新后代码重跑（新增 factor decomposition + npz 表）。

### 关键发现 1：a2q normalization 校正反转了原结论

原 `a2q_between_frac` / `a2q_within_frac` 在 response level 做 law of total variance：
- `between_var` = Var_x(μ_i)，其中 μ_i = (1/K)Σ_k z_{i,k} — 已做 1/K prompt-level 平均
- `within_var` = E_x[Var_k(z_{i,k})] — response-level raw 方差

但 batch gradient variance 是：
```
Var(ĝ) = (1/N) · between_var + (1/NK) · within_var
```
within 端多一个 1/K 缩减因子。原代码直接 between/(between+within) 漏了这个。

校正后：

| Config | K | raw within_frac | corrected between_frac |
|---|---:|---:|---:|
| n256k4 | 4 | 0.67 | **0.67** (prompt noise 主导) |
| n128k8 | 8 | 0.87 | **0.56** (大致 50/50) |
| n64k16 | 16 | 0.94 | **0.50** (精确 50/50) |

**结论：baseline K=8 下 prompt-sampling noise 和 rollout noise 大致对半开，不是原以为的 "within 占 87%"。**

### 关键发现 2：K 越大 → 更多 informative prompts → 更好 validation

frac_informative 随 K 单调增加：K=4 → 0.40, K=8 → 0.57, K=16 → 0.66。
n64k16 final avg5 = 0.3369 vs baseline n128k8 = 0.3265（+0.0104）。

### 关键发现 3：r_hat 不受 N/K 影响

r_hat ≈ 0.01-0.02 across all configs, g_dot_positive ≈ 50%。总 batch size 不变 SNR 不变。

### 二维 Noise 框架（5.19 确立）

不能把四种 noise source 做四项加和。正确层级：

**维度 A（sampling axis）**：h = A²Q 的方差发生在哪个采样层面
- V_between_batch = Var_x(μ_i) / N  — prompt sampling noise
- V_within_batch = E_x[Var_k(h)] / (NK)  — rollout sampling noise

**维度 B（factor axis）**：h = A²Q 的方差由哪个因子驱动
- CV²(A²) vs CV²(Q) vs Corr(A², Q)
- counterfactual: Var(A²·Q̄) + Var(Ā²·Q) + interaction = Var(A²Q)

二维组合：每个 sampling axis 内部分解 A²/Q/interaction 贡献。

### 新增 logging（5.19 实现）

113 个 scalar metrics + per-response/per-prompt npz 表（每 5 步）。详见 engineering_impl.md。

核心新增：
- 三套 A²/Q/H between/within + batch-level corrected fractions
- CV²(A²), CV²(Q), CV²(H), Corr(A²,Q), interaction_ratio
- counterfactual variance attribution
- success rate histogram (5 bins)
- per-response/per-prompt npz 离线重算表

### Stage 3 方向（修订）

校正后两种 noise source 大致对半：

| Finding | Algorithm direction |
|---|---|
| Between-prompt noise dominates at low K | Prompt diversity / stratification |
| Within-prompt noise dominates at high K | Increase K / advantage estimator |
| **Both ~equal at baseline K=8** | **Need combined approach** |
| frac_informative increases with K | K is direct lever for reward signal quality |

分析报告：`exp_data/stage2_nk_noise_analysis.md`
分析脚本：`exp_data/stage2_nk_noise_analysis.py`

---

## 2026-05-16 方向转折：Variance Source Decomposition 优先于新 estimator 设计

### 核心判断

所有 r_t estimator 尝试（split-batch, EMA window, ratio_of_sums, logit-space M₂, parameter-space momentum）
都有根本性问题。在继续设计新 estimator 之前，需要先回答：

> RL 中的 gradient noise 到底来自哪一层 expectation 的有限样本估计？

### Parameter-space momentum estimator 的问题

用 A_t=EMA(||ĝ_t||²) 和 B_t=EMA(||m_t^Adam||²) 解出 r_t 的方案有 non-stationarity bias：
- 推导假设 g_t 在 momentum 时间窗口内近似不变
- 实际 E[||m_t||²] = ||ḡ_t||² + κ N_t，其中 ḡ_t 是过去梯度的加权平均
- 信号衰减时 ||ḡ_t||² > ||g_t||² → r_t 被系统性高估 → LR 偏高
- 这恰好在最危险的 late-stage regime 出错

### Variance decomposition 诊断设计

将 noise 分成四层：

1. **Prompt sampling noise** — between-prompt variance of gradient contributions
2. **Rollout sampling noise** — within-prompt, between-response variance
3. **Reward/advantage signal sparsity** — fraction of prompts with mixed success/failure
4. **Score-function / trajectory-length variance** — logit-space energy per token

用 law of total variance：

Var(ĝ) = (1/N)[Var_x(μ(x)) + E_x[Var(U|x)]]

### 已实现的诊断 metrics

在 `verl/trainer/ppo/metric_utils.py` 中添加 `compute_variance_decomposition_metrics()`，
在 `ray_trainer.py` 中每步调用。新增 metrics：

**Prompt-level reward informativeness:**
- `noise_decomp/reward_std/mean` — 每组 reward std 的均值
- `noise_decomp/reward_n_unique/mean` — 每组不同 reward 值数量
- `noise_decomp/frac_informative` — Pr(0 < p_i < 1)，有效学习信号的 prompt 比例
- `noise_decomp/success_rate/mean` — 平均成功率

**Advantage energy:**
- `noise_decomp/adv_energy/mean` — E_i[(1/K)Σ_k Â²_{i,k}]
- `noise_decomp/adv_scalar/abs_mean` — |Â| 平均值

**Score energy (需要 sum_pi_squared):**
- `noise_decomp/score_energy_q/mean` — per-token q 均值
- `noise_decomp/q_response/mean` — per-response Q 均值
- `noise_decomp/a2q/mean` — Â² × Q 均值（advantage-weighted score energy）

**Between/within-prompt variance decomposition:**
- `noise_decomp/a2q_between_prompt_var` — Var_x(H_i) where H_i = mean(Â²Q | prompt i)
- `noise_decomp/a2q_within_prompt_var` — E_x[Var(Â²Q | prompt x)]
- `noise_decomp/a2q_between_frac` / `a2q_within_frac` — 相对占比

### 诊断结果将引导算法方向

| 发现 | 对应算法方向 |
|---|---|
| 有效 reward prompts 变少 | reward-informativeness-aware LR decay |
| score energy 后期很大 | M₂ 路线做 noise diagnostic，不做 r_t estimator |
| between-prompt variance 主导 | batch construction / prompt stratification |
| within-prompt rollout variance 主导 | 增加 K / 调 temperature / 改 advantage estimator |

### 文件变更

- `verl/trainer/ppo/metric_utils.py` — 新增 `compute_variance_decomposition_metrics()`
- `verl/trainer/ppo/ray_trainer.py` — import 并调用新函数

### 下一步

在现有 B-current 配置下跑一轮 diagnostic run，收集 300 步的 noise_decomp/* metrics，
按 early/mid/late 分析各层 noise 的演变趋势。不启动新 estimator 实验。

---

## 2026-05-15 ratio_of_sums W10 rerun: underperforms B-current and replace_ema

Bug 17 修复后重跑 3 seed（seed42 失败为空文件）。机制确认正常工作（r_window ∈ [0.014, 0.037]，
window count 稳定 10，r_window_enabled=1 全程），但**性能低于 B-current 和 W10 replace_ema**。

### Results

| method | seeds | mean best | mean final | mean drop |
|---|---:|---:|---:|---:|
| B-current | 3 | 0.3466 | **0.3440** | -0.0023 |
| W10 replace_ema | 3 | 0.3513 | 0.3413 | -0.0101 |
| **ratio_of_sums W10** | 3 | 0.3393 | 0.3345 | -0.0048 |

ratio_of_sums 比 B 低 0.0095，比 replace_ema 低 0.0068。

### Why it failed to help

1. **r_hat_raw ≈ 0 post-warmup**: pooling 10 steps of cross-power/auto-power
   did not rescue the numerator from noise. Phase means are O(1e-4).
2. **g_dot_positive ≈ 50%**: no improvement over other methods.
3. **alpha_t scale too high**: post-warmup mean ~4.6e-6 vs B's ~3.1e-6.
   ratio_of_sums produces systematically higher r_window → inflated LR.
4. PPO stability was fine (KL ~3.8e-4, ratio_p95 ~1.03), so the issue is
   controller quality, not training instability.

### Decision

ratio_of_sums W10 方向关闭。replace_ema 仍是 temporal aggregation family 中的
最佳变体。详细分析：`exp_data/5.15_ratio_of_sums_w10_rerun.md`。

## 2026-05-14 ratio_of_sums W10: implementation bug invalidates all runs (RESOLVED)

Bug 17: call site passed `r_window_num=denom` and `r_window_den=g_dot`, inverting
the fraction. All four original seed runs were destroyed. Fixed by single swap.
Reruns completed 2026-05-15 — see results above.



The 1.5B multi-seed follow-up changed the controller priority:

```text
W10 windowed continuous-r is the most promising current variant, but it has not
yet solved the noise/stability problem.
```

### B-current reference is already multi-seed

B-current has three historical complete seeds:

| run | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|
| B seed42 old | 0.3470 | 0.3470 | +0.0000 |
| B seed0 old | 0.3458 | 0.3419 | -0.0040 |
| B seed1 old | 0.3471 | 0.3442 | -0.0029 |
| **mean** | **0.3466** | **0.3440** | **-0.0023** |

There is also a new B-current diagnostic rerun with final avg5 `0.3385`.
Do not claim that B-current lacks seeds. The fair caveat is only that old B
seeds and latest W10 seeds were not all launched under exactly the same logging
batch.

### W10 evidence

Current W10 results:

| run | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|
| W10 seed0 new | 0.3456 | 0.3419 | -0.0037 |
| W10 seed1 new | 0.3624 | 0.3600 | -0.0024 |
| W10 seed42 old | 0.3460 | 0.3219 | -0.0241 |
| **mean** | **0.3513** | **0.3413** | **-0.0101** |

Interpretation:

- W10 has the highest recent peak and the strongest new-seed result.
- New seed0/seed1 W10 mean final is `0.3510`, higher than slow EMA, alpha rate
  limit, matched constant, and the newest B rerun.
- Including old seed42, W10 final mean falls below B-current because of a large
  late drop.
- Therefore W10 supports the **temporal aggregation direction**, not a final
  controller claim.

Recommended advisor-facing statement:

> Single-step r is too noisy. Window aggregation, especially W10, is the most
> promising way we have found to extract useful low-frequency structure, but the
> result is still seed-sensitive and needs stability confirmation.

### Comparison to other low-frequency variants

Recent completed seed0/seed1 readout:

| group | seeds | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|---:|
| W10 | 2 | 0.3540 | 0.3510 | -0.0030 |
| slow EMA 0.95 | 2 | 0.3408 | 0.3344 | -0.0064 |
| alpharlim0.05 | 1 complete | 0.3405 | 0.3366 | -0.0039 |
| matched constant 3.10e-6 | 1 | 0.3407 | 0.3320 | -0.0087 |

Design implication:

- Slow EMA alone is not enough; it changes the effective LR scale and remains
  seed-sensitive.
- Alpha rate limit is a useful safety/stability knob, not the main source of
  performance.
- W10 is currently the only new controller variant worth targeted confirmation.

### Next experiments

Do not broaden the sweep. Run targeted W10 confirmation:

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

Interpretation after these runs:

- If W10 seed42 rerun and seed2 are both close to seed0/seed1, W10 becomes the
  main controller candidate.
- If one repeats the old seed42 late drop, the method should be framed as
  increasing peak/opportunity rather than solving final stability.
- If W10 remains unstable, shift back toward base schedule + small residual
  adaptive signal instead of pure r-side controller.

## 2026-05-05 multi-seed controller follow-up

The latest single-seed low-frequency controller results are too close to support
a strong design conclusion. Current interpretation:

```text
the r_t noise problem is not solved;
single-seed differences between B / slow EMA / alpha-rate-limit / windowing are
not enough to distinguish controller quality from seed noise.
```

Therefore the current experiment priority is **multi-seed confirmation**, not
adding more variants.

Runs launched:

| group | seeds launched | reason |
|---|---:|---|
| `alpharlim0.05` | 0, 1 | seed42 was stable and close to B |
| `slowema_ret0.95` | 0, 1 | seed42 reached high peak but did not preserve it |
| `windowr_w10` | 0, 1 | seed42 tested stronger low-pass control but had weak final |

New scripts:

```bash
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed0.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpharlim0.05_seed1.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_seed0.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_seed1.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed0.sh
new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed1.sh
```

Decision rule:

- If a variant improves multi-seed final avg5 and reduces peak-to-final drop
  versus B, treat it as a real controller improvement.
- If only best/peak improves but final does not, describe it as increased
  exploration rather than solved reliability control.
- If mean differences are small relative to seed variance, the advisor-facing
  conclusion should be that simple low-pass variants are not distinguishable
  yet.

## 2026-05-05 diagnostic runs launched

After adding PPO ratio-tail, clip high/low, and train-batch dispersion metrics,
diagnostic runs were launched to collect those new signals:

| run | role |
|---|---|
| current B (`sync_sigfrac_cfixed_lr1.25e-5.sh`) | main adaptive baseline with new diagnostics |
| matched constant (`sync_matched_alpha_3.10e-6.sh`) | same-scale constant LR comparison |

These are not new controller experiments. They are intended to answer whether
late-stage degradation is better explained by safety/tail metrics than by
`KL mean`, `ratio_std`, or single-step `r_hat` sign.

Analysis target:

```text
metric_t -> future late-stage avg5 drop / train-score drift / response-length
drift, especially after stage demeaning.
```

## 2026-05-05 r_hat x pg_loss diagnostic pivot

Offline analysis on six local 1.5B controller JSONL files tested whether
`r_hat` becomes more informative when combined with PPO learning pressure
(`pg_loss`). High/low was defined by the median within each run-stage.

Artifacts:

- `paper/figures/rhat_pg_loss_opportunity_diagnostic.png`
- `paper/analysis/rhat_pg_loss_opportunity_diagnostic.md`

Main finding:

```text
high r_hat + high pg_loss has the strongest future avg5 / train-score gain when
all stages are pooled, but this pattern does not hold cleanly in late stage.
```

Design implication:

- `r_hat` alone should not be treated as a strong controller signal.
- `pg_loss` looks like a stronger opportunity signal.
- `r_hat` may help qualify whether that learning pressure is usable.
- Any new controller should avoid the rule "increase LR whenever both are high"
  in late stage.

Most promising next controller family:

```text
base schedule (D3/WSD) + small early/mid opportunity residual
```

Late-stage behavior should be controlled by safety/tail diagnostics rather than
`pg_loss` alone. New tail diagnostics have been added for future runs:
`ratio_p95`, `ratio_p99`, ratio threshold fractions, high/low PPO clip fractions,
and score/reward/advantage dispersion metrics.

## 2026-05-05 low-frequency ablation readout

Local result files checked under `deepseek1.5b_lr/`:

- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.jsonl`

No local JSONL/log/checkpoint was found for the true 3A windowed continuous-r
runs:

- `*_windowr_w5*`
- `*_windowr_w10*`

Therefore the current readout is **not** a 3A window readout. It is a readout of
slow EMA and alpha-rate-limit ablations.

### 5-task avg results

| run | max step | best avg5 | final avg5 | final - best |
|---|---:|---:|---:|---:|
| `alpharlim0.05` | 300 | `0.3428 @290` | `0.3403 @300` | `-0.0025` |
| `slowema_ret0.95` | 300 | `0.3483 @250` | `0.3407 @300` | `-0.0075` |
| `slowema_ret0.95_alpharlim0.05` | 300 | `0.3365 @270` | `0.3324 @300` | `-0.0041` |
| `slowema_ret0.98` | 279 | `0.3378 @240` | `0.3240 @270` | `-0.0138` |

Previous B reference remains roughly:

```text
B current: best/final core avg ≈ 0.3443 @300
```

### Interpretation

The experiments are close enough that they do **not** establish a clear win over
B. Current conclusion:

```text
noise is not solved;
simple slow EMA / alpha-rate limiting only partially changes the controller and
does not provide a significant performance improvement.
```

More detailed reading:

- `slowema_ret0.95` reaches the highest observed peak (`0.3483`), which is a
  useful sign that larger effective alpha / slower smoothing can push the model
  into a better mid-training region. However it does not hold the gain and drops
  to `0.3407`, so it is not a clean improvement.
- `alpharlim0.05` is the most stable among these runs (`final-best=-0.0025`) and
  suppresses alpha jitter, but its final score remains close to B and not better.
- `slowema_ret0.98` is too slow and underperforms.
- `slowema_ret0.95 + alpharlim0.05` also underperforms, suggesting that stacking
  two low-pass mechanisms can over-damp useful adaptivity.

### Updated design implication

Do not continue blindly sweeping EMA/rate-limit. The evidence supports a more
careful claim:

> low-frequency control is plausible, but naive low-pass filtering does not yet
> extract a clearly better reliability signal.

The next clean test remains the actual 3A windowed controller (`W=5`, `W=10`),
with explicit verification that `actor/r_window_enabled=1` appears in logs. If
W5/W10 also fail to separate from B, then continuous `r_obs` should be treated as
too noisy for a strong controller, and the method should shift toward coarse
schedule discovery / weak state-proxy claims.

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
