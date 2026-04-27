# Phase 2 Triangle Comparison — Experiment Analysis Report

**Date**: 2026-04-24
**Model**: DeepSeek-R1-Distill-Qwen-1.5B, 8 GPU
**Benchmark**: AIME24, AIME2025, OLYMPIAD, GPQA, MINERVA (primary); entropy, KL, grad norm, response length (training dynamics)
**Training steps**: 300

---

## 1. Experimental Design

三组实验的 mean α_t 对齐（≈ 2.97e-6），性能差异只能归因于信号利用方式，而非整体 LR 强度：

| 组 | 名称 | LR 策略 | α_t 设计 |
|---|---|---|---|
| **M** | `matched_alpha_2.97e-6` | 固定 | α = 2.97e-6（全程常数） |
| **A** | `sign_gate_meanmatch_gamma0.5` | Sign-gate | g_dot>0: α⁺=3.71e-6；g_dot≤0: γ·α⁺=1.855e-6 |
| **B** | `sigfrac_cfixed_lr1.25e-5` | Continuous r-shaping | α_t = c_fixed × r_ctrl，c_fixed=2.5e-4 |

**实验目的**：在 mean α_t 匹配的条件下，测试 alignment-based LR modulation 是否带来超出平均 LR scale 的独立收益，以及 continuous magnitude 相比 sign-only 是否有额外价值。

---

## 2. Summary Statistics

| Metric | M (Fixed) | A (Sign-gate) | B (Continuous) |
|--------|-----------|---------------|----------------|
| **AIME24 peak** | 0.3375 | 0.3312 | **0.3417** |
| **AIME24 final** | 0.2938 | 0.3229 | **0.3271** |
| **AIME2025 peak** | 0.2500 | 0.2646 | **0.2687** |
| **AIME2025 final** | 0.2271 | **0.2646** | 0.2396 |
| **OLYMPIAD peak** | 0.5006 | 0.4931 | **0.5056** |
| **OLYMPIAD final** | 0.4888 | 0.4850 | **0.5056** |
| **GPQA peak** | 0.3663 | 0.3481 | **0.3725** |
| **GPQA final** | 0.3663 | 0.3419 | **0.3725** |
| **MINERVA peak** | 0.2819 | 0.2838 | **0.2900** |
| **MINERVA final** | 0.2794 | 0.2737 | **0.2900** |
| α_t mean (>20) | 2.97e-6 | 2.89e-6 | 2.97e-6 |
| α_t std (>20) | ≈ 0 | 9.23e-7 | 7.74e-7 |
| entropy mean | 0.4501 | 0.4461 | **0.3617** |
| entropy final | 0.3895 | 0.3338 | 0.3743 |
| KL mean | 0.0004 | 0.0004 | 0.0004 |
| grad norm mean | 0.0388 | 0.0393 | 0.0396 |
| resp len final | 2714 | 2601 | **2917** |
| P(g_dot>0) mean | 52.7% | **57.9%** | — |

---

## 3. Performance Analysis

### 3.1 Full Benchmark Final Results

| Benchmark | M | A | B | B vs M | B vs A |
|-----------|---|---|---|--------|--------|
| AIME24    | 0.2938 | 0.3229 | **0.3271** | +0.033 | +0.004 |
| AIME2025  | 0.2271 | **0.2646** | 0.2396 | +0.013 | -0.025 |
| OLYMPIAD  | 0.4888 | 0.4850 | **0.5056** | +0.017 | +0.021 |
| GPQA      | 0.3663 | 0.3419 | **0.3725** | +0.006 | +0.031 |
| MINERVA   | 0.2794 | 0.2737 | **0.2900** | +0.011 | +0.016 |

**B 在 5/5 个 benchmark 上均超过 M，在 4/5 个 benchmark 上 final 最优**（AIME2025 例外，A 最优）。

**A 的表现分化明显**：
- AIME24/AIME2025：A 超过 M，是有效的 AIME 稳定器
- OLYMPIAD/GPQA/MINERVA：A 低于 M（GPQA 差距 -0.024，MINERVA 差距 -0.006）

因此不能说 "A ≈ B"。更准确的描述是：

> **B 是当前整体最强版本**；A 是 AIME 类任务的更优选择，但在宽 benchmark 上不如 M 和 B。

### 3.2 Peak vs. Final Divergence（后期退化对比）

M 的 late degradation 在 AIME 类任务上最显著：

```
AIME24:   M peak→final  0.3375 → 0.2938  drop = -0.044
          A peak→final  0.3312 → 0.3229  drop = -0.008
          B peak→final  0.3417 → 0.3271  drop = -0.015

AIME2025: M drop = -0.023, A drop = 0.000, B drop = -0.029
```

M 的问题不是学不动——M 的早期峰值不低（AIME24 峰值甚至高于 A），而是**早期能冲上去，后期守不住**。GPQA/MINERVA 上 M 的回落较小（GPQA drop=0），说明 late degradation 主要发生在精确推理类任务，对宽 benchmark 影响较弱，但 B 仍能在终点维持更高水平。

### 3.3 Continuous Magnitude 的价值

之前担心 continuous r̂_t 幅度是纯噪声、sign 已足够。但 GPQA 数据直接挑战了这个假设：

```
GPQA final:  M=0.3663, A=0.3419, B=0.3725
```

A 不但没超过 M，反而**低于 M 0.024**；B 则最高（超 M +0.006）。sign-gate 的"粗粒度刹车"在 GPQA 上过于保守，损失了有效更新；continuous r-shaping 的细粒度调制在这里更合适。

> **Sign captures a robust coarse reliability cue and strongly reduces AIME late degradation, but continuous magnitude retains additional useful information for broader benchmarks.**

---

## 4. Training Dynamics Analysis

### 4.1 Entropy

```
entropy mean:  M=0.450, A=0.446, B=0.362
entropy final: M=0.390, A=0.334, B=0.374
```

B 的整体熵显著低于 M 和 A（mean 低约 0.08）。B 的低熵不是退化信号——B 在 GPQA/OLYMPIAD/MINERVA 上性能最高。可能的解释：continuous r-shaping 在低信噪比步骤自动缩小步长，使策略更新更集中，但并未压缩策略的整体表达能力。

entropy final 排序 A < B < M，但 AIME2025 final 排序 A > B > M——说明 entropy 与性能的关系依赖于任务特性，不是单调关系。

### 4.2 KL Divergence

三组 KL mean 均为 0.0004，无差异。这说明三组没有明显的整体更新强度差异，性能差异更可能来自 LR 在不同 alignment 状态下的时序分配方式，而非整体 policy shift 速度的不同。

注意：mean 相同不代表尾部完全一致，理想情况下还应对比 KL p95/max 和 clip ratio，但当前数据已排除均值层面的解释。

### 4.3 Grad Norm

三组 grad norm mean 非常接近（0.039 ± 0.0004），无实质差别，与 KL 结论一致。

### 4.4 Response Length

```
final response length: M=2714, A=2601, B=2917
```

A 最短，与其 entropy 最低一致；B 最长，与其鼓励高信噪比步骤全速推进的机制一致。三组均相对训练初期（~6000 tokens）大幅缩短，但缩短幅度相近，说明这一趋势是 GRPO on-policy 训练本身的特性，与 LR 策略关系不大。

### 4.5 P(g_dot > 0)

| 组 | P(g_dot>0) | 高于随机基准 |
|---|---|---|
| M | 52.7% | +2.7pp |
| A | 57.9% | +7.9pp |

A 的轨迹下观测到更高的方向对齐率，提示 sign-gate 可能维持了更自洽的学习 regime。注意这些 step 在时间上是相关的，不宜按 iid Bernoulli 套用显著性检验，结论仅作描述性陈述。

---

## 5. LR & Signal Dynamics

### 5.1 α_t 均值对齐验证

| 组 | 设计 mean | 实测 mean (>20) | α_t std |
|---|---|---|---|
| M | 2.97e-6 | 2.97e-6 | ≈0 |
| A | ≈2.97e-6 | 2.89e-6 | 9.23e-7 |
| B | 2.97e-6 | 2.97e-6 | 7.74e-7 |

A 实测均值略低（-2.7%），因实际 P(g_dot>0)=57.9% 略高于设计假设 60%。三组均值基本对齐，但 temporal distribution 存在差异（A/B 有非零 std，M 完全固定）。

### 5.2 r̂_t 信号

r_hat_raw 噪声大（ratio estimator instability，理论预期），EMA 平滑后典型值 0.015–0.025，三组曲线几乎重叠。说明 r̂_t 的计算结果本身不解释三组之间的性能差异，差异来自如何利用这个信号（固定 vs sign-gate vs continuous）。

### 5.3 c_t / φ̄_t

本轮实验中 c-side calibration branch 关闭（eta_c=0），c_t 固定在 handoff 值不更新，φ̄_t=0.5 是初始值而非收敛结果，不能作为 c_t controller 工作的证据。c_t controller 验证留待 Phase 3。

---

## 6. Core Conclusions

### 结论 1：Alignment-based modulation 有独立收益

在 mean α_t 基本匹配的条件下，B 在 5/5 个 benchmark 上均超过固定 LR M。KL 和 grad norm 均值无差异，说明这不能由整体更新强度解释。**平均 LR 量级无法解释主要差异**，split-batch alignment signal 提供了超出平均 LR scale 的有效信息。

### 结论 2：Continuous-r 是当前整体最强版本

B 在 4/5 个 benchmark 上 final 最优，在 OLYMPIAD/GPQA/MINERVA 上明显优于 A（GPQA：+0.031）。Continuous magnitude 并非纯噪声，在宽任务上有 sign-only 所不具备的细粒度调制价值。

### 结论 3：Sign-gate 是更保守的 AIME 稳定器

A 在 AIME24/AIME2025 上显著降低 peak-to-final drop（AIME24 drop -0.008 vs M 的 -0.044），并在 AIME2025 final 最优。对精确数学推理类任务，sign-gate 的粗粒度刹车有更强的 late-stage stabilization 效果。但这种保守性在 GPQA 上造成损失（A < M），不具有跨任务普适性。

### 结论 4：固定 LR 的主要问题是后期守不住

M 的早期峰值不低（AIME24 峰值 0.3375，略高于 A），但后期回落最严重（AIME24 drop -0.044）。问题不是初始学习能力，而是训练状态变化后缺少风险调节，导致 step 持续处于低信噪比区间仍全速推进。

### 结论 5：任务类型影响最优控制形式

AIME 类精确推理任务更偏向 sign-gate（更保守、稳定性强）；OLYMPIAD/GPQA/MINERVA 等宽 benchmark 更偏向 continuous-r（细粒度调制更合适）。这提示 LR 控制的最优形式可能存在任务依赖性。

---

## 7. 论文叙事支撑

| 理论声明 | 实验证据 | 注意事项 |
|---------|---------|---------|
| 固定 LR 在 on-policy RL 后期退化 | M: AIME24 drop=-0.044；5/5 benchmark 均低于 B final | 退化在 AIME 类最强，GPQA 较弱 |
| Alignment signal 有独立信息量 | B 在 5/5 benchmark 超过 mean-matched M | 均值对齐，但 temporal distribution 不同 |
| Continuous magnitude 有额外价值 | B GPQA/OLYMPIAD/MINERVA 均超 A | 样本量有限，需多次运行确认 |
| Sign-gate 对 AIME 有 stabilization 效果 | A AIME24 drop=-0.008 vs M -0.044 | 对宽 benchmark 有负效果 |
| A trajectory 有更高方向对齐率 | A P(g_dot>0)=57.9% vs M 52.7% | 时序相关，非 iid，不做显著性检验 |

---

## 8. Open Questions

1. **A 在宽 benchmark 上低于 M 的机制**：sign-gate 的 γ=0.5 刹车力度对 GPQA 是否过强？是否存在 γ 值使得 A 在宽 benchmark 上也能超过 M？

2. **B 的低 entropy 来源**：是 continuous r-shaping 本身的效果，还是更长 response 分布的副作用？需要控制 response length 再观察。

3. **单次运行的局限性**：A vs B 在某些 benchmark 上差距较小（AIME24：0.004），多次运行才能确认是否统计显著。

4. **Phase 3**：本轮 c_t controller 关闭，Phase 3 应验证完整 eta_c > 0 的 c_t 动态是否能进一步优化宽 benchmark 性能。

---

## 9. Figures

- `figures/1_full_benchmark.png` — 5 个 benchmark + reward 完整性能曲线
- `figures/2_training_dynamics.png` — entropy、KL、grad norm (pre/post-clip)、response length、reward
- `figures/3_lr_dynamics.png` — α_t 轨迹/分布、r̂_t、c_t、φ̄_t
- `figures/4_alignment_signal.png` — P(g_dot>0) 时间序列 + 分布
- `figures/summary_stats.csv` — 所有指标数值

---

*分析脚本：`exp_data/4.24/analyze.py`*
