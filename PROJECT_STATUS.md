# Project Status: Learning-Update Mismatch in Self-Generated Data Loops

**Last updated**: 2026-04-26
**Model**: DeepSeek-R1-Distill-Qwen-1.5B | **Framework**: verl | **Algorithm**: PPO/GRPO

---

## Today's Progress (2026-04-26)

### 1. 9-run 多 seed 结果完成分析（`exp_data/4.26/analysis_report.md`）

已完成 3 condition × 3 seed（M/A/B × 42/0/1）汇总分析，报告见：

- `exp_data/4.26/analysis_report.md`

**当前最稳结论（仅基于数据）**：

1. **B（continuous r-shaping）是当前 overall 最强条件（near-mean-match 前提下）**：5 benchmark 综合均值与胜场均领先。  
2. **A（sign-gate）是 task-specific stabilizer**：AIME 上稳定收益，但在 GPQA/MINERVA/OLYMPIAD 上稳定不优于 M。  
3. **sign-only 不能解释 B 的收益**：`B-A` 在 OLYMPIAD 上稳定为正，GPQA/AIME 也偏正。  
4. **结论边界**：B 与 M 不是严格完美 mean-match（step21-300: B≈3.095e-6, M=2.97e-6），需保留 residual confound 声明。

### 2. Baseline 1 完成：constant LR = 3.10e-6（对齐 B 的 mean alpha）

目的：回答 “B 是否仅仅因为 mean alpha 更高”。

新增脚本：

- `new_experiments/signal_fraction_lr/sync_matched_alpha_3.10e-6.sh`
- `new_experiments/signal_fraction_lr/sync_matched_alpha_3.10e-6_seed0.sh`
- `new_experiments/signal_fraction_lr/sync_matched_alpha_3.10e-6_seed1.sh`

实现要点：

- 仍走 `signal_fraction` 路径 + `sign_gate_gamma=1.0`，保持与 M/A/B 同代码路径，避免“换 scheduler 路径”混杂。
- 主脚本已参数化（`SEED` / `EXP_NAME`），便于扩展 seed。

### 3. Baseline 2 完成：alpha shuffled control（保分布，打乱与对齐信号对应）

目的：回答 “B 的收益来自 alignment-conditioned modulation，还是普通 LR jitter”。

#### 3.1 代码能力已实现（scheduler 层）

新增 signal-fraction 参数（`optimizer.py` + `fsdp_workers.py` 透传 + `transformer_impl.py` 实现）：

- `signal_fraction_alpha_replay_path`
- `signal_fraction_alpha_replay_shuffle`
- `signal_fraction_alpha_replay_seed`
- `signal_fraction_alpha_replay_start_step`
- `signal_fraction_alpha_replay_end_step`

支持从 `json/csv/txt` 读取 alpha 序列，并在指定 step 窗口内 replay/shuffle 覆盖 alpha。

新增监控：

- `actor/alpha_replay_enabled`
- `actor/alpha_replay_applied`

#### 3.2 运行脚本已就绪（3 seeds）

- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpha_shuffled.sh`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpha_shuffled_seed42.sh`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpha_shuffled_seed0.sh`
- `new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5_alpha_shuffled_seed1.sh`

replay 源文件（已生成）：

- `exp_data/4.26/alpha_replay/b_alpha_seed42_step21_300.json`
- `exp_data/4.26/alpha_replay/b_alpha_seed0_step21_300.json`
- `exp_data/4.26/alpha_replay/b_alpha_seed1_step21_300.json`

### 4. 新一轮 6 组对照实验已启动（2026-04-26）

按 3 seed 运行两类新 baseline，共 6 组：

1. Baseline 1（constant 3.10e-6）：seed42/0/1
2. Baseline 2（alpha shuffled）：seed42/0/1

这些实验将直接用于消除/量化当前最关键 confound（B 的 mean alpha 略高）并验证“对齐条件调制”是否为主要增益来源。

---

## Today's Progress (2026-04-24)

### 1. Phase 2 三角对照实验全部完成（300步）

M / A / B 三组均完成，数据已下载至 `exp_data/4.24/`，分析报告：`exp_data/4.24/analysis_report.md`。

**核心结论（完整版本，含用户 review 后的修正）**：

**结论 1：Alignment-based modulation 有独立收益**
在 mean α_t 基本对齐的条件下（M=2.97e-6, A=2.89e-6, B=2.97e-6），B 在 5/5 benchmark 上均超过 M；KL mean 和 grad norm mean 三组完全相同。平均 LR 量级无法解释主要差异。

**结论 2：B（Continuous r-shaping）是整体最强**
B 在 4/5 benchmark final 最优；尤其 GPQA/OLYMPIAD/MINERVA 上明显优于 A（GPQA：+0.031）。Continuous magnitude 在宽任务有 sign-only 之外的额外价值。

**结论 3：A（Sign-gate）是 AIME 稳定器，但不是全局最优**
- A AIME24 peak-to-final drop = -0.008（vs M 的 -0.044），AIME2025 final 最优
- A 在 OLYMPIAD/GPQA/MINERVA 上**低于 M**（GPQA: A=0.3419 < M=0.3663）
- 不能再说 "A ≈ B，所以 sign 足够"

**结论 4：固定 LR 后期守不住**
M 的问题不是初始学习能力不足（AIME24 峰值 0.3375），而是后期回落最严重（drop=-0.044）。

**结论 5：任务类型影响最优控制形式**
AIME 类：A 更好（保守稳定）；OLYMPIAD/GPQA/MINERVA：B 更好（细粒度调制）。

**不能声明的内容（用户明确纠正）**：
- φ̄_t=0.5 不能作为 c_t controller 收敛证据（本轮 c-side 关闭，eta_c=0）
- P(g_dot>0) 不能做 iid 显著性检验（时序相关）
- 不能说 "A/B 在所有 benchmark 上均超 M"（A 在宽 benchmark 上不如 M）

**关键 training dynamics 指标**：

| 指标 | M | A | B |
|-----|---|---|---|
| entropy mean | 0.450 | 0.446 | **0.362** |
| KL mean | 0.0004 | 0.0004 | 0.0004 |
| grad norm mean | 0.039 | 0.039 | 0.040 |
| resp len final | 2714 | 2601 | **2917** |
| P(g_dot>0) mean | 52.7% | 57.9% | — |

### 2. Sign-gate 深度分析（2026-04-24 完成）

P(g_dot>0) 分阶段统计：

| 阶段 | M | A |
|------|---|---|
| warmup (1-20) | 0.750 | **0.900** |
| early (21-100) | 0.475 | 0.557 |
| mid (101-200) | 0.520 | 0.600 |
| late (201-300) | 0.530 | **0.510** |

**核心结论（4.24 确定）**：

1. **Late-period P(g_dot>0) 下降的根因**：A 比 M 更快达到低熵 regime（entropy final: A=0.334 < M=0.390），低熵下 split-batch 方向估计 SNR 系统性下降。这是结构性问题，与 γ 无关。A 的 late-period 对齐率交叉低于 M（0.510 vs 0.530），signal 判别力在训练后期自然衰弱。

2. **A 的 AIME2025 稳定性是巧合性优势**：late-period P(g_dot≤0)↑ → 刹车频率↑ → A 的 effective LR 自动降低（late mean=2.801e-6 vs M 的2.970e-6），对 AIME 后期过更新恰好有保护作用，但这不是设计上的正确性。

3. **GPQA: A < M 的机制**：gamma=0.5 刹车力度在 GPQA 对应步骤过于保守。这是最干净的"sign-gate 有害"证据：KL/gradnorm 三组均值完全对齐，差异只能来自信号利用方式。

4. **游程分析含义**：负游程 median=1，>50% 的负对齐步骤在下一步就翻正，当前每步独立 gate 对这类"一步噪声"过度刹车。**Hysteresis（k=2）** 是最简单有效的改进候选：连续 k 步 g_dot≤0 才触发刹车，可过滤大部分单步负游程。但这个方向本质上已接近 continuous r-shaping（B），不如直接用 B。

5. **Continuous magnitude 不是纯噪声的最强证据**：GPQA final B=0.3725 > M=0.3663 > A=0.3419，幅度信息在宽任务有 sign-only 之外的实际价值，"sign 足够"的假设已被推翻。

### 3. Seed 重复实验（2026-04-24 已启动）

Phase 2 结论基于单次运行，部分差距较小（A vs B on AIME24: 0.004），需要 seed 重复量化方差。

已创建并启动 6 个新脚本（seed=0 和 seed=1，各跑 M/A/B 三组）：

```
new_experiments/signal_fraction_lr/
  sync_matched_alpha_2.97e-6_seed0.sh     ← M, seed=0
  sync_matched_alpha_2.97e-6_seed1.sh     ← M, seed=1
  sync_sign_gate_gamma0.5_seed0.sh        ← A, seed=0
  sync_sign_gate_gamma0.5_seed1.sh        ← A, seed=1
  sync_sigfrac_cfixed_lr1.25e-5_seed0.sh  ← B, seed=0
  sync_sigfrac_cfixed_lr1.25e-5_seed1.sh  ← B, seed=1
```

原始 seed=42（3组）+ 新 6组 = **9次运行，3 seed × 3 condition**。可报告 mean ± std，支撑论文声明。

### 3. GitHub 仓库上线

https://github.com/065536/Asycn_verl.git
包含：核心代码修改、实验脚本、论文草稿、项目文档。排除：exp_data/（CSV）、verl/ckpts/（1.5T checkpoints）。

### 4. Checkpoint 清理

verl/ckpts/DeepSeek1.5B/ 清理：1.5T → 477G（保留 18 个 sigfrac/Phase 2 实验目录）。

### 5. 分析脚本与报告

- `exp_data/4.24/analyze.py`：生成所有图表
- `exp_data/4.24/analysis_report.md`：完整分析报告（含用户 review 后的修正版结论）
- `exp_data/4.24/figures/`：5 张图（benchmark、training dynamics、LR dynamics、alignment signal）

---

## What To Do Next

### 第一优先级：等待 4.26 新 6 组 baseline 对照结果（已启动）

跑完后按以下顺序分析：

1. **先做机制 sanity-check（必须先于分数）**  
   - Baseline 1: `alpha_t` 是否严格常数 3.10e-6（step11 后）  
   - Baseline 2: `actor/alpha_replay_applied` 是否在 step21-300 全为 1  
   - Baseline 2: `alpha_t` 在 step21-300 的 multiset 是否与对应 B 源序列一致（仅顺序被打乱）
2. **再做结果对照（按 seed 配对差分）**  
   - B vs Constant-3.10e-6：验证 B 的优势是否超出“更高平均 LR”解释  
   - B vs Alpha-Shuffled：验证收益是否来自 alignment-conditioned modulation，而非普通 LR 波动
3. 更新 `exp_data/4.26/analysis_report.md`，写入两组 baseline 对照结论与可声明边界

#### 对照 1（Constant LR = B mean alpha）的判定目标

要回答的问题：

- **B 赢是不是仅仅因为 mean alpha 更大？**

期望主比较：

- `B` vs `Constant(3.10e-6)`（或等于 B 实测 mean alpha 的常数 LR）

理想支持证据：

- `B > Constant(B-mean)`，尤其在：
  - `final avg_5bench`
  - `AIME24 / OLYMPIAD final`
  - `peak-to-final drop`（B 更小）
  - late-stage stability（B 更稳）

若成立，可支持叙事：

- **adaptive gain allocation matters, not just average LR scale**
- 关键不是平均 LR，而是 LR 在不同训练状态下的分配（early 更积极、late 自动降速）。

#### 对照 2（Random/Shuffled alpha）的判定目标

要回答的问题：

- **B 赢是不是仅仅因为 LR 有波动（jitter），而不是因为与 alignment 对应？**

实验定义：

- 保留 B 的 alpha 分布（均值/方差/范围），仅打乱其与真实 step 状态（alignment）的对应关系。

期望主比较：

- `B` vs `Alpha-Shuffled`

理想支持证据：

- `B > Shuffled`（final 更高、drop 更小、late-stage 更稳）
- Shuffled 无法复现 B 在 AIME/OLYMPIAD 上的 late-stage stability

若成立，可支持叙事：

- 收益来自 **alignment signal → LR allocation**，不是普通 LR jitter。

#### 理想综合结果与结论强度

最理想排序：

- `B > Constant(B-mean) > M`，且 `B > Alpha-Shuffled`

这时可给出强结论：

- B 的提升既不是“平均 LR 更大”，也不是“LR 波动本身”，而是 alignment-conditioned continuous modulation 的因果收益。

#### 非理想结果的降级解释（预定义）

1. `Constant(B-mean) ≈ B`：
   - B 优势与 mean LR scale 仍耦合，当前证据不足以完全解耦 conditioning 贡献。
2. `Shuffled ≈ B`：
   - dynamic LR schedule 有效，但当前证据不足以证明 split-batch alignment 是因果信号。
3. `B < Constant` 或 `B < Shuffled`：
   - 当前 B 主结论不稳，需收缩 claim 并重新审查 r_t 信号因果价值。

#### 结果出来后的优先指标顺序（固定模板）

1. `final avg_5bench`（整体效果）
2. `AIME24 peak/final/drop`（late degradation）
3. `OLYMPIAD final`（当前最稳的宽任务观察点）
4. `alpha_t` 分段统计（early/mid/late）
5. late-stage `entropy/KL/grad norm` 曲线与分位数（避免只看均值）

### 第二优先级：论文更新

根据 4.24 + 4.26 结论更新 main.tex：
- Section 3–5：加入 slow scale（handoff）+ sign-gate + continuous-r 的实验叙事
- 修正旧叙事（"sign 是最优"→ 任务依赖；continuous magnitude 有额外价值；near-mean-match 限制）
- 填写 Section 7（Empirical Story）和 Section 8（Interventions）

### 第三优先级：Phase 3 设计

在完整 c_t controller（eta_c > 0）下重复 Phase 2，验证动态 scale 是否进一步提升宽 benchmark 性能。

### 第四优先级：外部比较

- DAPO Clip-Higher（脚本已有）
- AER-style adaptive entropy-coefficient baseline（未实现）

---



## Theory Revision (2026-04-16) — RESOLVED 2026-04-19 → see Theory Finalization below

Discussion revealed that the current theoretical framework has fundamental logical flaws. The entire chain needs to be rebuilt.

### Flaw 1: "Moving target" premise is wrong

The paper frames RL instability as a "moving target" problem. This is incorrect.

$J(\pi) = \mathbb{E}_{\tau \sim \pi}[R(\tau)]$ in policy space is a **fixed objective** — the reward function doesn't change, the environment doesn't change, the optimal policy doesn't change. There is no moving target. The landscape in policy space is stationary.

The "moving" appearance comes from parameterizing with $\theta$ and needing on-policy samples, but this is a sampling/estimation problem, not a non-stationary optimization problem. Consequences of this error:
- It does not motivate using RL-specific optimizers over Adam (Adam works fine on fixed objectives)
- It does not motivate entropy-adaptive LR (there are standard algorithms for moving targets — online learning, FTRL — and nobody uses them here)
- It makes the entire framing look like it's solving the wrong problem

### Flaw 2: "Parameter-data coupling → support shrinks" is not justified

Even if we accepted parameter-data coupling as the starting point, the next step — "as entropy decreases, effective gradient support shrinks" — is not logically derived. It conflates two different things:
- **Sample diversity** (decreases with entropy — this is true)
- **Validity range of the surrogate objective** (a property of the objective landscape, not the samples)

The paper jumps from "low entropy → low rollout diversity → small support" without establishing why rollout diversity determines the surrogate's validity range. This step was never rigorously argued.

### The logical chain is broken at every link

> moving target → parameter-data coupling → support shrinks → learning-update mismatch

Each arrow has a gap. The framework is not just imprecisely stated — it is logically incorrect at multiple points.

### What the correct starting point appears to be

Two paths exist and their connection needs to be established:

**Path 1** (what the first-principles derivation leads to): Gradient estimation requires sampling from $\pi_\theta$. With finite samples ($n=8$ rollouts), gradient estimate quality is limited. This is fundamentally a **sampling/estimation quality** problem. Should point toward sampling-related solutions.

**Path 2** (what entropy-adaptive LR actually does): Reduces LR as entropy decreases. This manages the **exploration/exploitation** balance — large updates during exploration (high entropy, diverse rollouts), small updates during exploitation (low entropy, peaked policy).

### Connection between Path 1 and Path 2 (2026-04-16)

The gradient estimate $\hat{g} = \frac{1}{n}\sum_{i=1}^n A_i \nabla_\theta \log \pi_\theta(\tau_i)$ is informative when two conditions hold:

1. **Sample diversity**: the $\tau_i$ are diverse enough to cover different regions of policy space
2. **Reward reliability**: the $r_i$ correctly rank response quality so $A_i$ are meaningful

Both determine the **information content** of the gradient estimate — how much the current rollouts tell us about which direction to improve the policy.

**In this project's specific setting** (GRPO + math + binary correctness reward):
- Reward is deterministic and verifiable — condition 2 is always satisfied (constant)
- Therefore gradient quality reduces to a single free variable: **sample diversity = entropy**

$$\text{gradient quality} \equiv \text{sample diversity} \equiv \text{entropy}$$

**In policy gradient methods**, there is no independent exploration mechanism (unlike ε-greedy or UCB in tabular RL). Exploration happens entirely through the entropy of the policy:

$$\text{exploration level} \equiv \text{sample diversity} \equiv \text{entropy}$$

**Conclusion**: In this setting, Path 1 and Path 2 are not two separate problems — they are **two descriptions of the same quantity**. Gradient estimation quality and exploration level both reduce to entropy. The two paths converge.

**The unified logical chain:**

> Gradient estimation requires finite sampling from $\pi_\theta$  
> → Estimate quality depends on sample diversity × reward reliability  
> → In this setting, reward is reliable (constant), so quality ≡ sample diversity ≡ entropy  
> → In policy gradient, exploration level ≡ entropy (no separate exploration mechanism)  
> → Therefore: gradient quality ≡ exploration level ≡ entropy  
> → LR should scale with gradient estimation quality  
> → **Entropy-adaptive LR**

This chain is grounded at every step. Does not require "moving target," "parameter-data coupling," or "effective gradient support."

**Still open**: Why does lower gradient quality justify smaller LR? The connection "LR ∝ gradient quality" needs to be argued explicitly — this is the next piece to establish.

**Next step**: Rebuild the theoretical framework from this chain before writing more of the paper.

> **→ RESOLVED 2026-04-19.** The missing link is derived explicitly in `paper/theory_derivation.tex` using the per-step improvement bound for $L$-smooth objectives. Key result: $\alpha^*(t) \propto n_{\mathrm{eff}}(t)$ (linear, not square-root). See Theory Finalization (2026-04-19) section for the full chain.

---

## Core Thesis (updated 2026-04-19)

**In on-policy RL, the policy generates the data used for its own next update. As training progresses, the policy concentrates onto high-reward trajectories, reducing the effective sample size $n_{\mathrm{eff}}(t) \approx \min(n, \exp(H(\pi_t)))$. A rigorous per-step improvement bound shows that the optimal learning rate $\alpha^*(t) \propto n_{\mathrm{eff}}(t)$ (linear). Fixed-LR training holds $\alpha$ constant while $\alpha^*(t)$ decreases; eventually $\alpha > \alpha^*(t)$, making expected per-step improvement negative. The path through policy space during training—not the terminal entropy—determines final performance.**

### The formal chain (paper/theory_derivation.tex)

1. **Estimator MSE**: $\mathbb{E}[\|\hat{g} - g\|^2] = \mathrm{tr}(\Sigma_t)/n$ (exact, from i.i.d. vector samples)
2. **Effective sample size**: $n_{\mathrm{eff}}(t) = \mathbb{E}[K_n] \approx \min(n, \exp(H(\pi_t)))$ — distinct trajectories, i.e., the perplexity
3. **Per-step improvement bound** ($L$-smooth objective): $\mathbb{E}[J(\theta_{t+1})] - J(\theta_t) \geq \alpha\|g\|^2 - \frac{\alpha^2 L}{2}(\|g\|^2 + \sigma^2/n_{\mathrm{eff}})$
4. **Optimal LR**: $\alpha^*(t) = \|g_t\|^2 / (L(\|g_t\|^2 + \sigma_t^2/n_{\mathrm{eff}}))$ — strictly increasing in $n_{\mathrm{eff}}$
5. **Linear scaling** (noise-dominated regime + slowly-varying SNR assumption): $\alpha^*(t) \propto n_{\mathrm{eff}}(t)$
6. **Mismatch**: $\alpha > \alpha^*(t) \Rightarrow \mathbb{E}[J(\theta_{t+1})] < J(\theta_t)$ — expected regression
7. **Path determines outcome**: same terminal entropy across all stable experiments; AIME@300 spans 0.017–0.325; difference is entirely which mode was reached before $n_{\mathrm{eff}} \approx 1$

**Note on earlier language**: "Effective gradient support" (used in earlier drafts) was a useful intuitive concept but is replaced here by the precisely defined $n_{\mathrm{eff}}(t)$. "Support shrinks" = $n_{\mathrm{eff}}$ decreases. "Learning-update mismatch" = $\alpha > \alpha^*(t)$.

### Critical refinement (2026-04-14, from data analysis of all 17 experiments):

**Entropy decline is the goal, not the disease.** A well-trained model should produce peaked token distributions (low entropy) on each context — that is what "good model" means. All successful and failed sync experiments converge to similar final entropy (~0.07–0.10 at step 300). The difference between AIME@300 = 0.325 (best) and 0.017 (worst) is not how much entropy was depleted, but **the quality of the path through policy space during depletion**.

**The core problem is path quality, not terminal state.** Experiments with LR=1e-5 held constant show a characteristic "rise-then-fall" AIME pattern: accuracy peaks around step 50–70 then declines as entropy drops below ~0.15. At that point, LR=1e-5 exceeds the support and updates push the policy into bad regions. Experiments where effective LR ≤ 3e-6 in the low-entropy regime (sync_lr3e-6, cosine_floor) show monotonically improving or stable accuracy — every update stays within support.

**Entropy bonus is structurally wrong** — it fights the training objective itself. Since the goal is a peaked distribution, any force that resists entropy decline opposes what training is supposed to achieve. No fixed β can work: too small = no effect, too large = pushes policy toward uniform (opposite of goal). The β sweep (0.001/0.01/0.1) confirms this is not a tuning problem but a conceptual error.

**The right intervention modulates update scale, not entropy itself.** Entropy-adaptive LR does not try to keep entropy high; it reduces LR as entropy (support proxy) decreases, ensuring each update stays within the shrinking support. cosine_floor works for the same reason (crude gain scheduling) — it's the only LR=1e-5-starting experiment without late-stage accuracy decline.

Key concepts:
- **Effective sample size $n_{\mathrm{eff}}(t)$** — the expected number of distinct trajectories among $n$ rollouts; $n_{\mathrm{eff}} \approx \min(n, \exp(H(\pi_t)))$. Replaces the intuitive "effective gradient support." Decreases as the policy concentrates; this is necessary and expected for a well-trained model.
- **Optimal LR $\alpha^*(t) \propto n_{\mathrm{eff}}(t)$** — derived from the per-step improvement bound for $L$-smooth objectives. Monotonically decreasing. Fixed LR will eventually exceed it.
- **Learning-update mismatch** — $\alpha > \alpha^*(t)$: the update scale exceeds the information content of current samples. Once triggered, expected per-step improvement is negative.
- **Path quality** — which local optimum was reached before $n_{\mathrm{eff}} \approx 1$. Identical terminal entropy, radically different outcomes (20× AIME spread).
- **KL monitors motion, not $n_{\mathrm{eff}}$** — KL measures per-step policy drift on sampled actions. Once the policy is narrow, samples miss the recovery directions. Reads "calm" while $n_{\mathrm{eff}} \approx 1$ and updates are noise-dominated.
- **Entropy as proxy for $n_{\mathrm{eff}}$** — $\exp(H) \approx n_{\mathrm{eff}}$ is a convenient observable. Low entropy is the *desired end state*; the problem is the ratio $\alpha/\alpha^*(t)$, not entropy itself.
- **Entropy-adaptive LR = gain scheduling** — LR(t) = LR₀ · H(t)/H(0). Tracks $\alpha^*(t) \propto n_{\mathrm{eff}} \approx \exp(H)$ as a proxy; maintains $\alpha \leq \alpha^*(t)$ through training.

The paper is primarily a **theoretical/diagnostic contribution** — a unified framework that reinterprets existing interventions (KL, clipping, entropy bonus, staleness control) through the lens of matching update scale to effective gradient support in a self-generated data loop.

---

## Unified View of Sync/Async Failure

Both synchronous and asynchronous training fail for the same root cause: **the update exceeds the effective gradient support**.

| | Synchronous | Asynchronous |
|---|---|---|
| Feedback topology | Immediate: update uses current-policy rollouts | Delayed + stale: update uses past-policy rollouts |
| Mismatch trigger | Single update too large relative to current support | Same, plus drift between behavior policy and current policy |
| Failure signature | Fast entropy decay → self-reinforcing collapse | Delayed failure — support may be gone while signal still reflects older state |
| Stability margin | Larger | Smaller (delay tightens the boundary) |
| Root mechanism | **Same: learning-update mismatch** | **Same: learning-update mismatch** |

The distinction is topology and stability margin, not different diseases. Async can look more robust early because delay decorrelates symptoms from causes — but support may already be depleted while optimization still reacts to an older policy state.

---

## Reinterpretation of Interventions as Mismatch Control

| Intervention | Mismatch-framework view | Limitation |
|---|---|---|
| Entropy bonus (fixed β) | Open-loop damping — resists support shrinkage | **Structurally wrong**: opposes the training objective itself (peaked distributions = low entropy). β sweep (0.001/0.01/0.1) confirms no correct β exists — too small = no effect, too large = pushes toward uniform. Not a tuning problem. |
| PPO clipping | Actuator limit — caps per-step update scale | Fixed limit regardless of current support; conservative enough for late training → restrictive early. clip0.1 delayed but worsened collapse. |
| Cosine LR decay (with floor) | Crude open-loop gain scheduling | Best open-loop method (AIME 0.273). Works because LR reduction tracks support decline *roughly*. But schedule is fixed and problem-independent — can't match optimal fixed LR (0.325). |
| Staleness control | Reduces support gap (behavior policy ↔ current policy) | Widens margin but doesn't address the update-to-support ratio itself |
| Advantage-variance entropy bonus | Directs exploration where uncertainty is highest | Addresses "where to explore" but not "how aggressively to update." LR still 1e-5 → same late-stage collapse as baseline. Wrong axis. |
| **Entropy-adaptive LR** | **Gain scheduling** — couples update scale to remaining support (via entropy proxy) | State-dependent; directly addresses the mismatch ratio. Does not fight the goal (low entropy). **Not yet experimentally validated.** |

---

## Complete Experiment Inventory

Local CSV data available in `exp_data/` for Baseline + Phase 1 batch 1 (12 experiments). All others are SwanLab-only until downloaded.

### Baseline (4) — data: `exp_data/4.5/`

| SwanLab 描述 | experiment_name | Mode | LR | Outcome |
|---|---|---|---|---|
| Baseline sync LR=1e-6 | `deepseek1.5b_sync_lr1e-6` | Sync | 1e-6 | Stable, best final accuracy |
| Baseline sync LR=1e-5 | `deepseek1.5b_sync_lr1e-5` | Sync | 1e-5 | Type I: entropy collapse |
| Baseline async LR=1e-6 | `deepseek1.5b_fa_partial_lr1e-6` | Async | 1e-6 | Stable, competitive |
| Baseline async LR=1e-5 | `deepseek1.5b_fa_partial_lr1e-5` | Async | 1e-5 | Type II: gradient truncation |

### Baseline extension — abandoned

`deepseek1.5b_sync_8gpu_base_lr1e-6_500steps` ran twice but both runs were truncated. Not usable; discard.

### Phase 1 batch 1 (8) — data: `exp_data/4.7/`

| SwanLab 描述 | experiment_name | Script |
|---|---|---|
| Phase1 固定熵奖励 β=0.01，sync | `deepseek1.5b_sync_8gpu_ent0.01_lr1e-5` | `entropy_regularization/sync_entropy_bonus_lr1e-5.sh` |
| Phase1 固定熵奖励 β=0.01，async | `deepseek1.5b_fa_partial_8gpu_ent0.01_lr1e-5` | `entropy_regularization/async_partial_entropy_bonus_lr1e-5.sh` |
| Phase1 LR sweep，sync，LR=5e-6 | `deepseek1.5b_sync_8gpu_lr5e-6` | `lr_sweep/sync_lr5e-6.sh` |
| Phase1 LR sweep，async，LR=5e-6 | `deepseek1.5b_fa_partial_8gpu_lr5e-6` | `lr_sweep/async_partial_lr5e-6.sh` |
| Phase1 cosine衰减，sync | `deepseek1.5b_sync_8gpu_cosine_lr1e-5` | `lr_schedule/sync_cosine_decay_lr1e-5.sh` |
| Phase1 cosine衰减，async | `deepseek1.5b_fa_partial_8gpu_cosine_lr1e-5` | `lr_schedule/async_partial_cosine_decay_lr1e-5.sh` |
| Phase1 clip=0.1，sync | `deepseek1.5b_sync_8gpu_clip0.1_lr1e-5` | `adaptive_clip/sync_adaptive_clip_lr1e-5.sh` |
| Phase1 LR sweep，sync，LR=3e-6 | `deepseek1.5b_sync_8gpu_lr3e-6` | `lr_sweep/sync_lr3e-6.sh` |

### Phase 1 batch 2 (4) — data: SwanLab only

| SwanLab 描述 | experiment_name | Script |
|---|---|---|
| 固定熵奖励 β=0.001，sync，LR=1e-5 | `deepseek1.5b_sync_8gpu_ent0.001_lr1e-5` | `entropy_regularization/sync_entropy_bonus_lr1e-5_beta0.001.sh` |
| 固定熵奖励 β=0.1，sync，LR=1e-5 | `deepseek1.5b_sync_8gpu_ent0.1_lr1e-5` | `entropy_regularization/sync_entropy_bonus_lr1e-5_beta0.1.sh` |
| 联合开环 cosine+β=0.01 (ran twice, keep one) | `deepseek1.5b_sync_8gpu_cosine_ent0.01_lr1e-5` | `combined_openloop/sync_cosine_entropy_bonus_lr1e-5.sh` |
| Cosine衰减+floor=1e-6，sync，LR=1e-5 | `deepseek1.5b_sync_8gpu_cosine_floor_lr1e-5` | `lr_schedule/sync_cosine_decay_lr1e-5_floor1e-6.sh` |

### Phase 2 — data: `exp_data/4.14/` (advvar), `exp_data/4.15/` (entadapt)

| SwanLab 描述 | experiment_name | Script | Status |
|---|---|---|---|
| 优势方差加权熵奖励，sync，LR=1e-5 | `deepseek1.5b_sync_8gpu_advvar_ent_lr1e-5` | `advantage_variance/sync_advvar_entropy_lr1e-5.sh` | ✅ Done |
| 熵自适应LR，初始模式，sync，LR=1e-5 | `deepseek1.5b_sync_8gpu_entadapt_initial_lr1e-5` | `entropy_adaptive_lr/sync_entropy_adaptive_lr1e-5_initial.sh` | ✅ Done |
| 熵自适应LR，EMA模式，sync，LR=1e-5 | `deepseek1.5b_sync_8gpu_entadapt_lr1e-5` | `entropy_adaptive_lr/sync_entropy_adaptive_lr1e-5.sh` | ✅ Done |
| — | `deepseek1.5b_sync_8gpu_combined_ours_lr1e-5` | `combined_ours/sync_adaptive_lr_advvar_lr1e-5.sh` | ❌ Failed (bugs, not re-run) |
| — | `deepseek1.5b_fa_partial_8gpu_combined_ours_lr1e-5` | `combined_ours/async_partial_adaptive_lr_advvar_lr1e-5.sh` | ❌ Failed (bugs, not re-run) |

### Signal-Fraction 实验（2026-04-21 起）

| 实验名 | 脚本 | 步数 | 状态 | 关键结果 |
|--------|------|------|------|---------|
| `deepseek1.5b_sync_8gpu_sigfrac_lr1e-5` | `sync_signal_fraction_lr1e-5.sh` | ~139 | ✅ 完成 | AIME24=0.300@120步，phi_t≡0 bug |
| `deepseek1.5b_sync_8gpu_sigfrac_lr1e-6` | `sync_signal_fraction_lr1e-6.sh` | ~95 | ✅ 完成（未跑完） | phi_t≡0 bug，性能落后 lr1e-5 |

**Phase 1 cfixed 实验**：

| 脚本 | alpha_base | c_fixed | 状态 |
|------|-----------|---------|------|
| `sync_sigfrac_cfixed_lr7.5e-6.sh` | 7.5e-6 | 1.5e-4 | 🟡 待启动 |
| `sync_sigfrac_cfixed_lr1e-5.sh` | 1.0e-5 | 2.0e-4 | 🟡 待启动 |
| `sync_sigfrac_cfixed_lr1.25e-5.sh` | 1.25e-5 | 2.5e-4 | ✅ 完成 300步（B 组基线，实测 mean alpha=2.97e-6） |

**Phase 2 三角对照（scripts 已就绪，4.23 Bug 16 修复后可启动）**：

| 组 | 脚本 | 设计 | 状态 |
|----|------|------|------|
| M | `sync_matched_alpha_2.97e-6.sh` | fixed 2.97e-6（gamma=1.0） | 🟡 待启动 |
| A | `sync_sign_gate_gamma0.5.sh` | sign-gate gamma=0.5, alpha_plus=3.71e-6, mean≈2.97e-6 | 🟡 待启动 |
| B | `sync_sigfrac_cfixed_lr1.25e-5.sh` | continuous r-shaping, mean=2.97e-6（实测） | ✅ 已完成，直接复用 |

| Phase | Status | Local CSV |
|-------|--------|-----------|
| Baseline (4) | ✅ Done | ✅ `exp_data/4.5/` |
| Baseline extension | ❌ Truncated, discarded | — |
| Phase 1 batch 1 (8) | ✅ Done | ✅ `exp_data/4.7/` |
| Phase 1 batch 2 (4) | ✅ Done | ✅ `exp_data/4.14/` |
| Phase 2 — advvar | ✅ Done | ✅ `exp_data/4.14/` |
| Phase 2 — entadapt_initial | ✅ Done | ✅ `exp_data/4.15/` |
| Phase 2 — entadapt_ema | ✅ Done | ✅ `exp_data/4.15/` |
| Phase 2 — combined_ours ×2 | ❌ Failed (not re-run) | — |
| Signal-Fraction — B (cfixed_lr1.25e-5) | ✅ Done | SwanLab only |
| **Signal-Fraction — M, A** | **🟡 待启动** | |

---

## Phase 1 Batch 1 Findings (2026-04-07)

Full analysis: `exp_data/4.7/ANALYSIS.md`

### Conclusion 1: There exists an optimal intermediate LR (confirms gain threshold)

LR sweep (sync): 1e-6 → 3e-6 → 5e-6 → 1e-5

| LR | AIME@300 | Status |
|---|---|---|
| 1e-6 | 0.281 | Stable, slow — gain too conservative |
| **3e-6** | **0.325** | **Optimal — gain in sweet spot** |
| 5e-6 | 0.294 | Stable, slightly past peak |
| 1e-5 | 0.183 | Budget exhausted → collapse |

sync_lr3e-6 outperforms sync_lr1e-6 by **16% relative** on AIME. By step 300, entropy at 3e-6 (0.073) ≈ entropy at 1e-6 (0.078) — gain didn't matter at the end. The advantage is that higher gain spent the budget faster during the early high-entropy phase, reaching better policy regions before the budget ran out. This is exactly what the entropy-budget model predicts: there is a Goldilocks gain regime where the budget is spent productively before support collapses.

**Paper implication**: This motivates gain scheduling — use high gain when budget is plentiful (early), reduce gain as budget depletes (late). Entropy-adaptive LR does this automatically.

### Conclusion 2: Open-loop interventions fail to manage the budget

#### Fixed entropy bonus (β=0.01, sync) — catastrophic overspend in a different direction
- Entropy exploded from 0.68 → 11.85 nats (≈ ln(151936), near-uniform over vocabulary) by step 30
- The entropy bonus injects artificial "budget" but simultaneously increases consumption through the loss coupling → positive feedback → runaway
- Fixed β is open-loop: it doesn't know the current budget level or consumption rate
- PPO KL spiked then collapsed to ~0 (policy stopped changing because it was already maximally random) — **KL blindness confirmed**

#### Fixed entropy bonus (β=0.01, async) — dampened but still mediocre
- Async staleness naturally dampened the feedback loop; no explosion
- But AIME 0.231, well below optimal LR (0.325) — uniform bonus preserves entropy indiscriminately, not where correction matters most

#### Tighter clip (clip=0.1 vs 0.2, sync, LR=1e-5) — slower drain, worse collapse
- Steps 0–80: appeared healthy, AIME rose to ~0.29 — **false sense of security**
- Steps 80–300: gradual then accelerating collapse; final AIME 0.017, worst in cohort
- Tighter clip slows the per-step drain but doesn't change the budget or the consumption direction. When entropy reaches the critical threshold, the remaining gradient information is doubly limited (narrow support + clipped actuation) → worse collapse than without clip
- **Analogy**: actuator limit on a runaway system — slows the crash but doesn't fix the root cause

#### Cosine LR decay — mediocre sync, catastrophic async
- Sync cosine: AIME peaked at 0.281 (step 60) then declined to 0.223. After LR dropped below 1e-6, gain became so low that learning effectively stopped — but by then the budget had not been productively used
- Async cosine: **complete failure** (AIME 0.021). As gain → 0, the policy cannot respond to new rollout data; stale rollouts pile up; correction diverges from current policy state. Open-loop gain reduction defeats the closed-loop nature of RL

### Conclusion 3: KL is systematically blind to collapse severity

In all collapsing experiments, PPO KL was small or declining *during* active collapse:
- sync_clip0.1: KL peaked at ~0.002 at step 127, then declined to ~0.0002 as accuracy catastrophically dropped
- sync_ent0.01: KL spiked early then ~0.0001 — policy stopped changing (maximally random)
- sync_lr1e-5: KL ~0.0004 throughout — unremarkable, while entropy was draining

KL measures per-step motion on sampled support. Once the policy is already narrow, the samples miss the regions where recovery could originate. KL therefore reads "calm" while the system is functionally broken. **Entropy is the right diagnostic.**

### Code verification (2026-04-07)
- **Entropy bonus sign**: correct. `policy_loss -= entropy_coeff * entropy_agg` → gradient descent maximizes entropy as intended. The explosion is real, not a sign bug.
- **Entropy units**: nats (natural log). Max entropy = ln(vocab_size) = ln(151936) ≈ 11.93 nats. Observed saturation at 11.85–11.92 is consistent with near-uniform distribution.
- **Temperature scaling**: entropy is computed from temperature-scaled logits in both rmpad and non-rmpad paths — consistent with log_prob computation.
- **Logged metric = loss metric**: `actor/entropy` and the entropy used in the loss are the same `entropy_agg` variable — no calibration discrepancy.

---

## Experiment Plan

### Phase 1: Validate framework with existing interventions

**Goal**: Show that open-loop methods are insufficient, motivating closed-loop approach.

| Category | Scripts | Purpose |
|----------|---------|---------|
| `entropy_regularization/` | 4 (sync × β=0.001/0.01/0.1, async × β=0.01) | Test open-loop damping + β sweep |
| `lr_sweep/` | 4 (sync/async × lr3e-6/5e-6) | Locate critical LR (gain margin) |
| `lr_schedule/` | 5 (sync/async × cosine/warmup, sync × cosine w/ floor) | Test open-loop gain scheduling |
| `adaptive_clip/` | 2 (sync/async × clip0.1) | Test open-loop actuator limiting |
| `combined_openloop/` | 1 (sync × cosine + entropy bonus) | Strongest open-loop combination |

**First batch (8, ✅ done):**
1. `entropy_regularization/sync_entropy_bonus_lr1e-5.sh` — β=0.01
2. `entropy_regularization/async_partial_entropy_bonus_lr1e-5.sh` — β=0.01
3. `lr_sweep/sync_lr5e-6.sh`
4. `lr_sweep/async_partial_lr5e-6.sh`
5. `lr_schedule/sync_cosine_decay_lr1e-5.sh`
6. `lr_schedule/async_partial_cosine_decay_lr1e-5.sh`
7. `adaptive_clip/sync_adaptive_clip_lr1e-5.sh`
8. `lr_sweep/sync_lr3e-6.sh`

**Second batch (4 new ★):**
5. ★ `entropy_regularization/sync_entropy_bonus_lr1e-5_beta0.001.sh` — **✅ done**
6. ★ `entropy_regularization/sync_entropy_bonus_lr1e-5_beta0.1.sh` — **✅ done**
7. ★ `combined_openloop/sync_cosine_entropy_bonus_lr1e-5.sh` — **✅ done**
8. ★ `lr_schedule/sync_cosine_decay_lr1e-5_floor1e-6.sh` — **✅ done**

> ★ = new script to create. Dropped: `entropy_regularization` at lr1e-6 (baseline already stable, does not advance argument).

**Design rationale for additions:**
- **β sweep**: prevents reviewer objection "you just didn't tune entropy bonus well enough"
- **Combined open-loop**: strongest possible open-loop baseline to beat, makes Phase 2 comparison convincing
- **Cosine w/ floor**: standard cosine decays to ~0, which is unfairly aggressive; floor=1e-6 is a fairer test of scheduling

### Phase 2: Our method — **entadapt_initial validated ✅**

**Primary intervention**: Entropy-Adaptive LR (gain scheduling) — LR ∝ H(t)/H(0)
**Secondary experiment**: Advantage-Variance-Weighted Entropy Bonus

| Script | Method | Status |
|--------|--------|--------|
| `entropy_adaptive_lr/sync_entropy_adaptive_lr1e-5_initial.sh` | Entropy-Adaptive LR, **initial mode** (H_t/H_0) | ✅ **Done — AIME@300 = 0.296, no collapse** |
| `entropy_adaptive_lr/sync_entropy_adaptive_lr1e-5.sh` | Entropy-Adaptive LR, EMA mode (H_t/H_ema) | ✅ Done — AIME@300 = 0.163, collapse as predicted |
| `entropy_adaptive_lr/async_partial_entropy_adaptive_lr1e-5.sh` | Entropy-Adaptive LR only (async) | ~~deferred~~ |
| ★ `advantage_variance/sync_advvar_entropy_lr1e-5.sh` | Advantage-Variance Entropy Bonus only | ✅ Done — AIME 0.158 (wrong axis) |
| ★ `advantage_variance/async_partial_advvar_entropy_lr1e-5.sh` | Advantage-Variance Entropy Bonus only | ~~deferred~~ |
| ★ `combined_ours/sync_adaptive_lr_advvar_lr1e-5.sh` | Combined (both components) | ❌ Failed (not re-run) |
| ★ `combined_ours/async_partial_adaptive_lr_advvar_lr1e-5.sh` | Combined (both components) | ❌ Failed (not re-run) |

### Phase 3: Related-Work Reproduction Plan (paper-critical)

Goal: reproduce the external methods that are explicitly discussed in `paper/main.tex` Related Work, so the core claims are supported by direct algorithm-level comparisons rather than only internal ablations.

| Priority | Paper / Method | Script(s) | Why this reproduction is required | Status |
|---|---|---|---|---|
| **Must** | **Fixed entropy bonus** (A3C-style standard regularization baseline) | existing β sweep scripts in `entropy_regularization/` | Supports key claim: fixed entropy bonus is open-loop damping, can slow collapse, but is structurally misaligned with the final low-entropy objective | **✅ Done** (β=0.001/0.01/0.1 + prior baselines) |
| **Must** | **DAPO Clip-Higher**~\cite{yu2025dapo} | `external_baselines/sync_dapo_cliphigher_lr1e-5.sh` | Represents a distinct stabilization handle (surrogate/clipping design) rather than LR/support tracking; central external comparator for the clipping argument | **🟡 Ready to run** |
| **Must** | **DAPO Clip-Higher + Ours** | `external_baselines/sync_dapo_cliphigher_entadapt_initial_lr1e-5.sh` | Tests complementarity: Clip-Higher addresses support expansion pressure; entropy-adaptive LR addresses update-scale/support matching | **🟡 Ready to run** |
| **Must** | **AER-style adaptive entropy coefficient**~\cite{zhang2025aer} | `entropy_regularization/sync_adaptive_entropy_coeff_lr1e-5.sh` (planned simplified reproduction) | Most direct "adaptive bonus vs adaptive LR" comparison; needed to justify why control handle should be LR, not entropy coefficient | **🔴 Not implemented yet** (framework code currently reverted) |
| **Should** | **Beyond Precision length-triggered LR scheduling**~\cite{zhang2026beyondprecision} | planned in `lr_schedule/` | Most relevant concurrent LR-scheduling line; needed for a hard comparison of trigger signal (`length` symptom vs `entropy` support proxy) | **🔴 Not implemented yet** |
| **Should (async focus)** | **VCPO (ESS-based LR scaling)**~\cite{huang2026vcpo} | async-only planned script in `external_baselines/` | Strong async-specific baseline; required to substantiate "same mismatch, different topology (sync vs async)" against a purpose-built async method | **🔴 Not implemented yet** |

Execution notes:
- Submission-minimum external set = **DAPO Clip-Higher + DAPO Clip-Higher + Ours + AER-style baseline**.
- Stronger camera-ready set adds **Beyond Precision scheduler** and **VCPO (async)**.
- P2-style exploratory runs (e.g., higher initial LR stress tests) are useful but secondary to the above reproductions.

### Phase 2 code changes (✅ done 2026-04-07, bugs fixed 2026-04-08–09):

**Entropy-Adaptive LR (✅ implemented):**
- `verl/workers/config/optimizer.py` — added `entropy_adaptive` scheduler type + `entropy_adaptive_min_ratio`
- `verl/workers/engine/fsdp/transformer_impl.py` — `EntropyAdaptiveLRScheduler` class + `update_entropy_for_lr()` method
- `verl/workers/engine_workers.py` — feeds `actor/entropy` from metrics into scheduler before each lr step

**Advantage-Variance Entropy Bonus (✅ implemented):**
- `verl/workers/actor/dp_actor.py` — `entropy_bonus_mode=advantage_variance` weights entropy by per-prompt advantage variance
- `verl/workers/config/actor.py` — added `entropy_bonus_mode` config field

---

## Bug Fixes (2026-04-08)

All Phase 2 experiments failed on launch. Root causes found and fixed:

### Bug 1: Wrong `verl` package imported (3 sync scripts)
Scripts did `cd $VERL_ROOT` (inside the package dir), so Python imported the installed `verl_origin` instead of the modified local code. Added `cd ${VERL_ROOT}/..` and `export PYTHONPATH="${VERL_ROOT}/..:${PYTHONPATH:-}"` to:
- `new_experiments/entropy_adaptive_lr/sync_entropy_adaptive_lr1e-5.sh`
- `new_experiments/combined_ours/sync_adaptive_lr_advvar_lr1e-5.sh`
- `new_experiments/advantage_variance/sync_advvar_entropy_lr1e-5.sh`

### Bug 2: New config fields missing from Hydra YAML (all 4 Phase 2 scripts)
`entropy_bonus_mode`, `lr_scheduler_type`, and `entropy_adaptive_min_ratio` were added to Python dataclasses but not to `verl/trainer/config/actor/actor.yaml`. Hydra struct validation failed before Python dataclass was ever reached. Added all three fields to YAML.

### Bug 3: `+` prefix conflict after YAML update (3 scripts)
After YAML was updated with `entropy_adaptive_min_ratio`, the `+` prefix in scripts (used to append new keys) conflicted with the now-existing key. Removed `+` prefix from `entropy_adaptive_min_ratio` override in all 3 scripts that used it.

### Bug 4: `fsdp_workers.py` missing `entropy_adaptive` scheduler branch
`_build_model_optimizer` only handled `"constant"` and `"cosine"`. All Phase 2 experiments using `lr_scheduler_type=entropy_adaptive` raised `NotImplementedError`. Added `entropy_adaptive` branch that instantiates `EntropyAdaptiveLRScheduler`.

### Bug 5: `fsdp_workers.py` never called `update_entropy()` on scheduler
Even after the scheduler was instantiated, `update_actor` never fed entropy back to it. The scheduler always used `entropy_ratio=1.0` (fallback), making adaptive LR completely non-functional. Added entropy update call before `lr_scheduler.step()`.

### Bug 6: `actor/entropy` metric is a list, not a scalar
`append_to_dict` appends one value per micro-batch, so `metrics["actor/entropy"]` is a list. `float(list)` raised `TypeError`. Added list→mean conversion before passing to `update_entropy()`.

### Bug 7: `calculate_entropy` never triggered for `sync_entropy_adaptive_lr1e-5`
This experiment uses `entropy_coeff=0` (no entropy bonus, only adaptive LR). The `calculate_entropy` flag was False (all three conditions False), so entropy was never computed and the scheduler never received updates. Fixed the third condition from dead code `self.config.get("entropy_adaptive_lr", False)` to `self.config.optim.get("lr_scheduler_type", "constant") == "entropy_adaptive"`.

### Bug 8: Advantage-variance grouping broken by dynamic batching (design flaw)
`prepare_dynamic_batch` internally calls `rearrange_micro_batches` which **re-sorts sequences by token length**, destroying the `[p1r1…p1r8, p2r1…p2r8, …]` prompt grouping. The original code computed variance inside the micro-batch loop, where `batch_size % rollout_n == 0` almost never held (typical micro-batch = 4–5 sequences, not divisible by 8). Result: advantage-variance weighting never fired; experiment ran as plain uniform entropy bonus.

**Fix**: Pre-compute `adv_var_weight` at mini-batch level (32 sequences, grouping intact) **before** calling `prepare_dynamic_batch`. Store in `mini_batch.batch["adv_var_weight"]`. The rearrange step then correctly slices this tensor alongside all others, so each micro-batch gets the right per-response weights regardless of how sequences are reordered.

### Bug 9: Advantage-variance weighting uses GRPO-normalized advantages — variance signal destroyed (2026-04-09)

`adv_var_weight` was computed from `advantages`, but GRPO normalizes advantages as `A_i = (r_i − mean_r) / std_r`. After normalization, `var(A_i) = var(r_i) / std_r² ≈ (n−1)/n` for **every** group with any reward diversity — the value is mathematically constant at `7/8 = 0.875` for `n=8`. After max-normalization all weights collapse to `1.0`. The `advantage_variance` mode was silently equivalent to `uniform` entropy bonus: the method had zero effect on directing exploration.

**Fix**: Use `token_level_rewards` (raw rewards, before GRPO std-normalization) for the variance computation. For binary rewards `var(r_i) = p*(1−p)`, correctly ranking groups: 4/8 correct → `var=0.25` (max), 1/8 or 7/8 correct → `var=0.109`, all correct/wrong → `var=0`. The fix falls back to `advantages` if `token_level_rewards` is not in the batch.

- File changed: `verl/workers/actor/dp_actor.py` (line ~601): `_raw = mini_batch.batch.get("token_level_rewards", mini_batch.batch["advantages"])`

### Bug 10: `token_level_rewards` not in `select_keys` — Bug 9 fix was non-functional (2026-04-09)

`update_policy` calls `data.select(batch_keys=select_keys, ...)` before splitting into mini-batches. `token_level_rewards` was not in `select_keys`, so it was filtered out. The `.get("token_level_rewards", advantages)` fallback always triggered — Bug 9 had zero effect. All `advantage_variance` experiments (including re-launched ones after Bug 9) were still using normalized advantages.

**Fix**: Added conditional `select_keys.append("token_level_rewards")` if the key exists in `data.batch`, matching the pattern used for `rollout_is_weights` and `rollout_log_probs`. File: `verl/workers/actor/dp_actor.py` (line ~566).

### Bug 11: `actor/entropy` inhomogeneous list crash in `reduce_metrics` (2026-04-09)

`update_policy` uses dynamic batching (`prepare_dynamic_batch`). `append_to_dict` appends one float per micro-batch for every metric (e.g. `actor/entropy`), so each DP worker returns a Python list of length = number of micro-batches on that rank. `DataProto.concat` merges all 8 workers' metrics via `list_of_dict_to_dict_of_list`, producing `actor/entropy = [[s1,s2], [s1',s2',s3'], ...]` — a nested list. Because different ranks can have different micro-batch counts (see Bug 12), the nested list is inhomogeneous. The trainer's `reduce_metrics` then calls `np.mean(val)` on this structure → `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.` Crashed `sync_advvar_entropy_lr1e-5` at step 1; would eventually crash all experiments once micro-batch count divergence occurs.

**Fix**: Added `from verl.utils.metric import reduce_metrics` import and changed `return metrics` → `return reduce_metrics(metrics)` at the end of `update_policy`. Each worker now pre-reduces its list metrics to per-worker scalars before returning. After 8-worker concat the trainer sees `[s0, s1, …, s7]` (8 scalars) → `np.mean` succeeds regardless of whether workers had different micro-batch counts.

- File changed: `verl/workers/actor/dp_actor.py` (end of `update_policy`, ~line 797)

### Bug 12: NCCL deadlock from unsynced micro-batch counts in `rearrange_micro_batches` (2026-04-09)

`rearrange_micro_batches` computes `num_micro_batches = ceildiv(total_seqlen, max_token_len)` independently on each DP rank. Because sequence length distributions differ across ranks (stochastic batching), ranks can arrive at different micro-batch counts. If rank A gets 2 micro-batches and rank B gets 3, the FSDP forward/backward loop executes a different number of AllGather / AllReduce collectives on each rank. Ranks block waiting for each other on mismatched collective operations → NCCL deadlock → watchdog fires after 600 s.

The intended guard was `same_micro_num_in_dp=True` (default), which adds a `dist.all_reduce(..., MAX)` so all ranks agree on `max(counts)`. However the condition was `if dist.is_initialized() and same_micro_num_in_dp and dp_group is not None:`. Since `prepare_dynamic_batch` in `update_policy` calls `rearrange_micro_batches` without passing `dp_group` (defaults to `None`), the sync was silently skipped — the flag was a no-op.

Crashed `sync_entropy_adaptive_lr1e-5` at step ~20 and `sync_adaptive_lr_advvar_lr1e-5` similarly. Phase 1 batch 2 has the same latent bug but hasn't triggered it in the first ~51 steps (probabilistic: depends on sequence length distribution divergence across workers in a given step).

**Fix**: Removed `and dp_group is not None` from the condition. `dist.all_reduce(..., group=None)` uses PyTorch's default world group automatically, synchronizing all 8 ranks to `max(micro_batch_counts)` without needing an explicit group handle.

- File changed: `verl/utils/seqlen_balancing.py` (line 402)

---

## Project File Structure

```
verl/
├── training_scripts/              -- 4 baseline scripts (already run)
├── exp_data/                      -- CSVs with experiment results
│   ├── 4.5/                       -- baseline
│   ├── 4.7/                       -- Phase 1 batch 1
│   ├── 4.14/                      -- Phase 1 batch 2 + advvar
│   ├── 4.15/                      -- entadapt_initial + entadapt_ema
│   └── 4.22/                      -- sigfrac_lr1e-5 / lr1e-6 (p_min removed)
│       └── analysis.md            -- 4.22 实验分析报告
├── new_experiments/
│   ├── README.md
│   ├── entropy_regularization/    -- 4 scripts (β=0.001/0.01/0.1 sweep + async)
│   ├── lr_schedule/               -- 5 scripts (cosine/warmup + cosine w/ floor)
│   ├── adaptive_clip/             -- 2 scripts (open-loop actuator limiting)
│   ├── lr_sweep/                  -- 4 scripts (locate gain margin)
│   ├── combined_openloop/         -- 1 script (cosine + entropy bonus)
│   ├── entropy_adaptive_lr/       -- 2 scripts (closed-loop LR, Phase 2)
│   ├── advantage_variance/        -- 2 scripts (adv-var entropy bonus, Phase 2)
│   ├── combined_ours/             -- 2 scripts (both components, Phase 2)
│   └── signal_fraction_lr/        -- signal-fraction adaptive LR scripts
│       ├── sync_signal_fraction_lr1e-5.sh     -- 4.22 实验（已跑）
│       ├── sync_signal_fraction_lr1e-6.sh     -- 4.22 实验（已跑）
│       ├── sync_sigfrac_cfixed_lr7.5e-6.sh    -- Phase 1（待启动）
│       ├── sync_sigfrac_cfixed_lr1e-5.sh      -- Phase 1（待启动）
│       └── sync_sigfrac_cfixed_lr1.25e-5.sh   -- Phase 1（待启动）
├── paper/
│   ├── main.tex                   -- LaTeX paper (main manuscript)
│   ├── theory_derivation.tex      -- Self-contained derivation document (2026-04-19)
│   ├── discussion_log.md          -- Theory discussion log (sessions 1+)
│   └── references.bib             -- Bibliography
└── PROJECT_STATUS.md              -- THIS FILE
```

---

## Paper Structure (reframed 2026-04-13)

**Title**: "Learning-Update Mismatch in Self-Generated Data Loops: A Unified View of Learning-Rate Instability in Synchronous and Asynchronous LLM RL"

| Section | Content | Status |
|---------|---------|--------|
| Abstract | Parameter–data coupling; effective gradient support; learning-update mismatch; sync/async unified; KL inadequacy; entropy-adaptive LR as gain scheduling | Done |
| 1. Introduction | Structural difference from SL; support concept; support shrinks; mismatch definition; sync/async unification; 4 contributions | Done |
| 2. Background | PPO/GRPO; sync vs async training | Done |
| 3. Parameter–Data Coupling: Why RL Is a Closed-Loop System | Open vs closed loop; two coupling channels (data channel + ratio channel); why small steps still work | Done |
| 4. Support, Shrinkage, and Learning-Update Mismatch | Effective gradient support definition; support shrinks as policy sharpens; mismatch as self-reinforcing instability; one mismatch two topologies (sync direct / async amplified by delay) | Done |
| 5. From Abstract Support to Observable Control | KL measures motion not support; entropy as practical proxy; LR as control handle; entropy dynamics + critical gain threshold; entropy-adaptive LR = gain scheduling | Done |
| 6. Experimental Setup | 4 testable predictions from framework | Done (needs exact numbers) |
| 7. Empirical Story | Baseline LR effect; entropy as early warning; KL vs entropy; async delayed failure | **TODO: waiting for results** |
| 8. Interventions Reinterpreted as Mismatch Control | Entropy bonus (open-loop damping); clipping (actuator limit); staleness control; entropy-adaptive LR (gain scheduling) | Theory done; **TODO: results** |
| 9. Related Work | RL for LLMs; A3C entropy collapse; TRPO/PPO motion-vs-support; SAC comparison; natural gradient; deep PG instability; async RL; control theory framing; positioning | Done |
| 10. Discussion | Motion vs remaining support; support not a fixed constraint; async confuses diagnosis; entropy as proxy not core concept; implications for method design | Done |
| 11. Conclusion | Mismatch as core problem; sync/async unified; KL inadequate; entropy-adaptive LR as gain scheduling; RL stability as matching problem | Done |

---

## Phase 2 Findings (2026-04-15)

Full analysis: `exp_data/4.15/ANALYSIS.md`

### Conclusion 1: Entropy-adaptive LR (initial mode) validated — first successful run

`entadapt_initial` (H_t/H_0): AIME@300 = **0.296**, peak = 0.315 @ step 250, **stable monotonic improvement with no collapse**. This is the primary Phase 2 result.

| Experiment | AIME@300 | Pattern |
|---|---|---|
| sync_lr3e-6 (best fixed LR) | 0.325 | Stable |
| sync_lr5e-6 | 0.294 | Stable |
| **entadapt_initial** | **0.296** | **Stable — beats cosine_floor** |
| cosine_floor (best open-loop) | 0.273 | Stable |
| sync_lr1e-6 | 0.281 | Stable, slow |
| sync_lr1e-5 (baseline) | 0.183 | Collapse |
| entadapt_ema | 0.163 | Collapse |
| advvar_ent | 0.158 | Collapse |

`entadapt_initial` is +0.023 above the best prior open-loop method (cosine_floor) and only 0.029 below the empirically optimal fixed LR (sync_lr3e-6).

### Conclusion 2: EMA mode fails exactly as predicted — and for precisely the predicted reason

`entadapt_ema` (H_t/H_ema): AIME@300 = **0.163**, rises to 0.313 @ step 100 then collapses — same failure mode as baseline sync_lr1e-5, delayed ~100 steps.

**Why, confirmed by actual LR data**: because EMA tracks H_t closely, H_t/H_ema ≈ 1 throughout — LR never actually decreases. Worse: as entropy stabilizes at low values (step 200+), H_t/H_ema occasionally exceeds 1, so LR is actually amplified back toward 1e-5. The EMA mode hits its *maximum* LR (1e-5) at step 244 — the most fragile point in training.

Actual LR at key steps:

| Step | entadapt_initial LR | entadapt_ema LR | ratio |
|---|---|---|---|
| 10 | 8.20e-6 | 8.20e-6 | 1.00× |
| 50 | 4.84e-6 | 7.40e-6 | 0.65× |
| 100 | 2.62e-6 | 7.42e-6 | **0.35×** |
| 150 | 1.51e-6 | 8.09e-6 | 0.19× |
| 244 | ~1.3e-6 | **1.00e-5** | — |
| 300 | 1.24e-6 | 9.32e-6 | 0.13× |

Steps with LR > 5e-6: **init = 44**, **ema = 294** (out of 300).

At step 100, when H < 0.15 (the critical threshold below which fixed LR=1e-5 collapses), `entadapt_initial` has already reduced LR to 2.62e-6 — right in the Goldilocks zone (empirically optimal: 3e-6). `entadapt_ema` is still at 7.42e-6. This is where learning quality diverges.

The comparison proves that the *specific reduction proportional to cumulative entropy decline* is what matters. The EMA mode is also "adaptive" but adapts to the wrong reference, producing no meaningful LR reduction.

### Conclusion 3: KL blindness confirmed in active collapse

During `entadapt_ema` collapse (step 100–300, AIME drops 48% relative), PPO KL reads 0.0001–0.0004 throughout — indistinguishable from the stable `entadapt_initial`. Entropy crossed the critical threshold at step 96, 100+ steps before the accuracy drop became obvious. Entropy is the right early warning signal; KL is structurally blind.

### Conclusion 4: Path quality thesis — terminal entropy ≈ same, outcomes radically different

Both experiments end at entropy ~0.07–0.08. AIME@300 differs by 82% relative (0.296 vs 0.163). Multi-benchmark collapse for EMA: AIME, AIME2025, MINERVA, OLYMPIAD all end *below their starting values* at step 300. The path to low entropy, not the terminal state, determines the outcome.

### Conclusion 5: EMA LR increase explains the delayed-then-accelerated collapse pattern

The EMA collapse follows a specific trajectory: stable AIME 0–100 (LR at ~7–8e-6, model still learning), plateau 100–210 (entropy entering critical zone), sharp collapse 210–300 (entropy nearly depleted, LR creeping back to 1e-5). This is not a smooth gradual failure — the LR increase in late stages (8.09e-6 → 9.88e-6 → 9.32e-6) adds a self-reinforcing mechanism on top of the standard mismatch pattern.

---

## Algorithm Redesign (2026-04-19)

Full discussion: `paper/discussion_log.md` Session 2 | Notes: `memory/algorithm_design.md`

### 当前算法 entadapt_initial 的三个根本缺陷

**缺陷1 — α₀ 依赖（"照着答案来设计"）**

公式 $\alpha(t) = \alpha_0 \times H(t)/H(0)$ 只是把 α₀ 按 entropy 比例缩放。当前 α₀=1e-5 之所以有效，是因为我们提前知道 3e-6 是经验最优值，而 $H(t)/H(0)$ 恰好把 α 压到这个范围。换模型、换任务，α₀ 需要重新调——和直接调 LR 没有本质区别。

**缺陷2 — 使用了错误的 proxy**

理论要求的衰减比例是 $n_{\mathrm{eff}}(t)/n_{\mathrm{eff}}(0) = \exp(H(t))/\exp(H(0))$，当前方法用的是 $H(t)/H(0)$（线性 vs 指数）。在 $H(t)=0.07, H(0)=0.68$ 时，正确比例 ≈ 0.54，当前比例 ≈ 0.10。两者相差 5×。当前方法"碰巧"有效是因为早期 $\exp(H)$ 严重低估实际 $n_{\mathrm{eff}}$（测量值 ≈ 6–8，理论值 ≈ 2），部分抵消了误差。

**缺陷3 — 不跨设置泛化**

$\alpha^*(t)$ 的绝对量级由 $L$、$\|g\|^2$、$\sigma^2$ 决定，这些量随模型大小、任务、batch size 而变。固定 α_max 相当于隐式假设这些量不变。

### 四个候选新算法

| 方案 | 公式 | 超参 | 优点 | 缺点 |
|---|---|---|---|---|
| **A — 信号占比** | $\alpha(t) = \alpha_{\max} \times r_t$，$r_t = \|\hat{g}\|^2/(\|\hat{g}\|^2 + \hat{\sigma}^2/n)$ | α_max | 直接来自 improvement bound | 需要 per-sample 梯度方差；仍有 α_max |
| **B — 直接测 n_eff** | $\alpha(t) = \alpha_{\max} \times \bar{K}_t / n$（$\bar{K}_t$ = 平均 unique response 数） | α_max | 最直接；不需要梯度计算 | 离散噪声（需 EMA）；仍有 α_max |
| **C — 估计 $\hat{L}_t$（无 LR 超参）** | $\hat{L}_t = \|\hat{g}_t - \hat{g}_{t-1}\| / (\alpha_{t-1}\|\hat{g}_{t-1}\|)$，然后代入 $\alpha^*(t)$ 公式 | 无 LR 超参 | 完全 data-driven；最符合理论 | 需存上一步梯度；早期噪声大 |
| **D — 梯度范数归一化** | $\alpha(t) = C / \|\hat{g}_t\| \times \hat{n}_{\mathrm{eff}}(t)/n$ | $C$（无量纲 $O(1)$） | 跨设置泛化；类似 Adam 的归一化动机 | $C$ 仍是超参，但量级稳定得多 |

### 关键设计问题（待决定）

1. Option C（估计 $\hat{L}_t$）vs Option D（梯度范数归一化）——先做哪个？
2. $\hat{L}_t$ 在实践中是否足够稳定？
3. "LR 太小"是**效率问题**，不是 mismatch 问题（$\alpha < \alpha^*(t)$ 期望 improvement 仍为正）。新算法的目标是防止 $\alpha > \alpha^*(t)$，不是双向优化。
4. 公平比较：新算法 vs sync_lr3e-6，需用**相同运行次数**（不允许对 baseline 做 LR sweep）。

> **→ RESOLVED 2026-04-20.** 选定 Option A（Signal-Fraction）。见下节。

---

## Signal-Fraction Algorithm (2026-04-20)

**最终算法**：$\alpha_t = c_t \cdot \hat{r}_t$，两时间尺度分解：

- **$\hat{r}_t$（快，每步）**：split-batch 估计 signal fraction，负责 shape
- **$c_t$（慢，每 K=5 步）**：scale factor，由 held-out realization ratio $\phi_t$ 驱动，targeting $\phi^* = 1/2$

**完整理论链**（`paper/theory_derivation.tex`）：

$$r_t = \frac{\|g\|^2}{\|g\|^2 + \sigma^2/n_{\mathrm{eff}}} \in (0,1],\quad \alpha^*(t) = \frac{r_t}{L}$$

**$\hat{r}_t$ 估计器**（同一 $\theta_t$ 下 split batch）：

$$\hat{r}_t = \frac{\hat{g}_{A1}^\top \hat{g}_{A2}}{(\|\hat{g}_{A1}\|^2 + \|\hat{g}_{A2}\|^2)/2}$$

**$c_t$ 控制律**（$c_{t+1} = c_t \cdot \exp(\eta_c(\bar{\phi}_t - 1/2))$）：

- $p_t = \alpha_t \cdot \hat{g}_C^\top \hat{g}_{\mathrm{upd}}$（预测增益）
- $a_t = L_C(\theta_t) - L_C(\theta_{t+1})$（held-out C 上的实际增益）
- $\phi_t = a_t / p_t$，$\phi^* = 1/2$ 来自 improvement bound 推导，且 $r_t$ 在 $\phi^*$ 表达式中精确抵消

**实现状态（2026-04-20）**：代码完成，4 文件已修改：

| 文件 | 变更 |
|------|------|
| `verl/workers/config/optimizer.py` | 新增 8 个 signal_fraction 字段 |
| `verl/workers/engine/fsdp/transformer_impl.py` | `SignalFractionLRScheduler` 类 |
| `verl/workers/fsdp_workers.py` | scheduler builder + metrics 提取 |
| `verl/workers/actor/dp_actor.py` | `_update_policy_signal_fraction` 方法 |

启用：`actor_rollout_ref.actor.optim.lr_scheduler_type: signal_fraction`

---

## Bug 1：r̂_t 在首步撞 r_min（2026-04-20 实验发现）

**配置**：train_batch_size=128，rollout_n=8，8 GPU fsdp_size=8（1 DP rank）→ actor 看到全部 128 样本 = 16 groups

**现象**：step 11（warmup 结束的第一步）α_t = c_0 × clamp(r̂_raw, 0.01, 1.0) = 1e-5 × 0.01 = **1e-7**，LR 一步塌陷。此后 r̂ 在 [0.01, 0.07] 振荡。

**根因（两层）**：

1. **Hard handoff**：warmup 结束后直接从"完全不用 r̂_t"切到"完全由 r̂_t 决定"，产生 discontinuity
2. **c_0 初始化错位**：$c_0 = \alpha_{\mathrm{base}}$（1e-5），但 $c_t$ 的语义是 $1/L$。若 warmup 期间典型 $\bar{r} \approx 0.04$，正确初值应为 $c_T = \alpha_{\mathrm{base}} / 0.04 \approx 2.5 \times 10^{-4}$

**为什么 c_t 无法自救**：fast-drop 后 $\alpha_t$ 很小 → $p_t = \alpha_t \cdot g_{\mathrm{dot}} \approx 0$ → calibration 被跳过或信噪比极低 → c_t 无法感知"LR 太小"。

**φ\*=1/2 不需要修改**：使用 EMA proxy $\bar{r}_t$ 后，quasi-static 时 $\bar{r}_t \approx r_t$，$\phi^* = 1/2$ 仍然对应 $c_t \to 1/L$。改 $\phi^*$ 会把 shape 信号重新掺进 scale controller，破坏两时间尺度分工。

**完整修复设计**：见 `memory/bug1_fix_design.md`，包含：
- warmup 期对称 EMA 收集 $\bar{r}_{\mathrm{warm}}$（算但不控）
- handoff 时 $c_T = \alpha_{\mathrm{base}} / \max(\bar{r}_{\mathrm{warm}}, r_{\mathrm{boot}})$（scale 对齐）
- M=10 步线性插值过渡
- asymmetric EMA（$\beta_\downarrow=0.5, \beta_\uparrow=0.1$）+ fast-drop 触发 + cooldown 状态机（freeze $c_t$，继续更新 $\bar{r}_t$）
- validity guard 三条件：dot>0，$d_t>d_{\min}^{\mathrm{abs}}$，$g_{\mathrm{rms},t} > \tau_{\mathrm{rms}} \cdot \bar{g}_{\mathrm{rms},t}$（$\bar{g}_{\mathrm{rms}}$ 与 validity 解耦，独立更新）

**实现状态**：设计完成，**已实现并通过代码审查**（2026-04-20）。见 `memory/bug1_fix_design.md`。

---



`paper/theory_derivation.tex` created — a fully self-contained derivation document with no logical gaps.

### The resolved chain

The "Path 1 and Path 2" and "why lower gradient quality → smaller LR" open questions from the 2026-04-16 revision are now closed:

| Step | Statement | Derivation |
|------|-----------|------------|
| 1 | $\mathbb{E}[\|\hat{g}-g\|^2] = \mathrm{tr}(\Sigma_t)/n$ | Expand $\|\hat{g}-g\|^2$, i.i.d. cross terms vanish, $\mathbb{E}[\|X-\mu\|^2] = \mathrm{tr}(\Sigma)$ |
| 2 | $n_{\mathrm{eff}} = \mathbb{E}[K_n] \approx \min(n, \exp(H))$ | Indicator expectation → union; exact for uniform; perplexity = effective support size |
| 3 | $\mathbb{E}[J(\theta_{t+1})] - J(\theta_t) \geq \alpha\|g\|^2 - \frac{\alpha^2 L}{2}(\|g\|^2 + \sigma^2/n_{\mathrm{eff}})$ | $L$-smooth descent lemma → substitute update → expand $\mathbb{E}[\|\hat{g}\|^2]$ |
| 4 | $\alpha^*(t) = \|g\|^2 / (L(\|g\|^2 + \sigma^2/n_{\mathrm{eff}}))$ | Differentiate improvement bound over $\alpha$, set = 0 |
| 5 | $\alpha^*(t) \propto n_{\mathrm{eff}}(t)$ | Noise-dominated regime ($\sigma^2/n_{\mathrm{eff}} \gg \|g\|^2$) + Assumption 1 ($\sigma^2/\|g\|^2 \approx C$) |

### Key correction from earlier draft

An earlier (discarded) derivation based on the condition $\|g\| \geq \|\hat{g} - g\|$ (SNR ≥ 1) yielded $\alpha^* \propto \sqrt{n_{\mathrm{eff}}}$. This was wrong: that condition constrains $\|g\| / (\sigma/\sqrt{n})$, from which $\alpha$ cancels — it does not bound $\alpha$. The correct bound via the improvement approach gives the **linear** relationship $\alpha^* \propto n_{\mathrm{eff}}$.

### Paper implications

- Abstract, RQ box, Open Questions: all updated to say $\alpha^* \propto n_{\mathrm{eff}}$ (linear), not $\sqrt{n_{\mathrm{eff}}}$
- Self-reinforcing collapse is now explicitly a **Remark** (empirical observation), not a load-bearing logical step
- The comparison with supervised learning (SL has $n_{\mathrm{eff}} = n = \mathrm{const}$, so $\alpha^*$ is constant) makes the distinction clean

---

---

## Today's Progress (2026-04-23)

### 1. 理论诊断：r̂_t 的 statistical feasibility problem

**核心结论（三层）**：

**第一层**：slow scale（c_t / handoff）是主导因素，已被 B 组数据确认。

**第二层**：fast reliability 幅度（r̂_t 的连续值）在小 r_t regime 下难以稳定估计，原因叠加：
1. **分子 SNR 低**：E[ĝ_A1^T ĝ_A2] = ||g||²，噪声与信号比 = (1-r_t)/r_t。当 r_t=0.02 时比值 = 49，分子期望被 49 倍量级噪声淹没。
2. **ratio estimator 不稳定**：分母 (||ĝ_A1||² + ||ĝ_A2||²)/2 本身也是随机变量，小信号 regime 下分母随机波动放大比值噪声（虚高/虚低）。
3. **group structure 破坏独立性**：A1/A2 按 prompt-group 切割，within-group 8 条 response 高度相关，有效样本数更接近 group 数，进一步压低 SNR。
4. **L 是移动靶**：理论链要求 c_t → 1/L，但 RL 中 effective local L 随 policy/advantage/sampling distribution 漂移，c_t controller 追的是移动靶。
5. **PPO clipping 违反 L-smooth 假设**：在 policy 快速变化时 clip 比例上升，bound 失效程度最高，恰好是 r̂_t 最被依赖的时刻。

**第三层**：sign-gate 不是 continuous r-shaping 的"降级版"，而是当前 regime 下**统计上最合适的信号提取方式**：continuous r-shaping 隐含要求符号可靠 + 幅度可靠 + 分母尺度可靠；sign-gate 只要求 sign(ĝ_A1^T ĝ_A2) 整体有判别力。

### 2. 论文叙事升级

旧叙事："我们提出 continuous adaptive LR"  
新叙事：
1. 理论分解出 α_t = c_t · r̂_t（slow scale × fast reliability）
2. Phase 1/B 确认 slow scale 可由 handoff 自动校准，且是主导因素
3. 理论分析表明 fast reliability 幅度在小-r_t regime 存在根本性 ratio estimator instability
4. 从该信号中识别统计上最可信的分量：alignment sign
5. sign-gate 是对"可靠信息"的最大化提取，而非退而求其次

### 3. Phase 2 三角对照确定

三组 expected mean alpha 完全对齐（≈2.97e-6），性能差异可归因于信号利用方式：

| 组 | alpha 设计 | 期望 mean alpha | 回答的问题 |
|---|---|---|---|
| M | fixed 2.97e-6（sign_gate_gamma=1.0） | 2.97e-6 | scale 选对是否足够？r-side 是否有净贡献？ |
| A | g_dot>0: 3.71e-6 / g_dot≤0: 1.855e-6（gamma=0.5） | ≈2.97e-6 | 统计最稳的信号分量（符号）是否带来独立收益？ |
| B | c_fixed=2.5e-4, r_min=0.01 | 2.97e-6（实测） | 幅度信息是否有额外价值，还是只是把噪声传进控制器？ |

**均值匹配推导**：alpha+ = target / (p + (1-p)×γ) = 2.97e-6 / (0.60 + 0.40×0.5) = 2.97e-6 / 0.80 = 3.71e-6

### 4. Bug 16：sign_gate 参数漏加 FSDPOptimizerConfig（2026-04-23 修复）

**发现**：M / A 两脚本启动即崩溃 —— `TypeError: FSDPOptimizerConfig.__init__() got an unexpected keyword argument 'signal_fraction_sign_gate_gamma'`

**根因**：`signal_fraction_sign_gate_gamma` 和 `signal_fraction_sign_gate_alpha_plus` 已在 `transformer_impl.py`（调度器构造函数）和 `fsdp_workers.py`（读取）中实现，但漏加到 `FSDPOptimizerConfig` 数据类。Hydra 在 `omega_conf_to_dataclass` 时把所有字段作为 kwargs 传给 `__init__`，遇到未知字段报错。

**修复**：在 `verl/workers/config/optimizer.py` 的 `FSDPOptimizerConfig` 中（`signal_fraction_handoff_steps` 之后、`__post_init__` 之前）添加：
```python
signal_fraction_sign_gate_gamma: Optional[float] = None
signal_fraction_sign_gate_alpha_plus: Optional[float] = None
```

**状态**：已修复（2026-04-23）。M / A 脚本可以重新启动。

---

## Today's Progress (2026-04-22)

### 1. 4.22 实验数据分析（sigfrac_lr1e-5 ~139步 / sigfrac_lr1e-6 ~95步，p_min 删除后首轮）

**关键指标**：

| 指标 | lr1e-5 | lr1e-6 |
|------|--------|--------|
| r̂_raw 均值 | 0.0069 | 0.0198 |
| r̂_raw 负值比例 | 38% | ~35% |
| r̄_t（EMA）均值 | 0.0163 | 0.0205 |
| c_t handoff 值 | ~2e-4 | ~2e-5 |
| alpha_t 均值 | **3.0e-6** | 4.1e-7 |
| phi_t | **0.0 全程** | **0.0 全程** |
| phi_bar 轨迹 | 0.45 → 0.157 | 0.50 → 0.266 |

**Handoff 有效**：c_T = α_base/r_boot = 1e-5/0.05 = 2e-4，使 alpha_t 初始值 ≈ 3.2e-6，接近已知最优 3e-6。

**性能（mean@16）**：lr1e-5 在 AIME24 step 120 = 0.300（仍在上升），全面领先 lr1e-6。

全文分析见 `exp_data/4.22/analysis.md`。

### 2. phi_t ≡ 0 bug 诊断与修复（Bug 15）

phi_t 全程为 0，根因：4.22 第一轮修复（解决 phi_t≈1e5）将两个 loss 函数都改为 ratio=1，导致 a_t = L_C_old - L_C_new ≡ 0 恒成立。

**修复**：`_backward_split` 保存 θ_t 的 log_probs（`collect_log_probs=True`），传给 `_forward_loss_nograd`（`old_log_probs_list`），使 ratio = π_{θ_{t+1}}/π_{θ_t}，L_C_new 真正依赖 θ_{t+1}。详见 Bug 15 记录。

### 3. 两阶段实验方向确定

核心结论：不要同时验证"c_t 学得对"和"r_t 有信息量"，先找到最优 c_fixed，再单独测 r_t。

**Phase 1（eta_c=0，找 c_fixed）**：

| α_base | c_fixed | 典型 α_t |
|--------|---------|---------|
| 7.5e-6 | 1.5e-4  | ~2.7e-6 |
| 1.0e-5 | 2.0e-4  | ~3.6e-6 |
| 1.25e-5 | 2.5e-4 | ~4.5e-6 |

**Phase 2（固定 c_fixed，测 r_t 独立信息量）**：Version A（sign-gate）vs Version B（continuous r-shaping）vs 固定 LR=3e-6 基线。

### 4. Phase 1 脚本创建 + calibration 关闭

三个脚本（eta_c=0.0，calib_frac=0.0）：
- [sync_sigfrac_cfixed_lr7.5e-6.sh](new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr7.5e-6.sh)
- [sync_sigfrac_cfixed_lr1e-5.sh](new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1e-5.sh)
- [sync_sigfrac_cfixed_lr1.25e-5.sh](new_experiments/signal_fraction_lr/sync_sigfrac_cfixed_lr1.25e-5.sh)

**calib_frac=0.0 的代码修复**：[dp_actor.py:574](verl/workers/actor/dp_actor.py#L574) 将 `max(1, round(...))` 改为 `max(0, round(...))`，使 `calib_frac=0` 时 `calib_groups=0`，`data_C=None`，C-split 和整个 calibration branch 被完全跳过。全部 batch 分给 A1/A2，与 fixed LR baseline 的 batch 利用率对等。

**Phase 1 的准确描述**：post-handoff fixed-c（前 10 步 warmup 后，handoff 设 c_T=lr/r_boot，之后冻结）；非全程 fixed-c。

### 5. 两条分析原则确立（今日新增）

**开跑前置检查**：step 1–10 alpha_t 走 warmup；step 11 c_t 跳至 handoff 值；`is_calibration_step` 全为 0。不符合预期则停跑排查，不依赖最终分数判断配置是否正确。

**先看动力学，再看最终分数**：先确认 effective LR 区间按预期分开，r_hat 分布独立于 c_fixed，再比较 val 曲线。"更保守"和"更稳定"是两件不同的事。

---

## Today's Progress (2026-04-21)

### 1. 4.21 实验数据分析（sigfrac_lr1e-5 / sigfrac_lr1e-6，Bug 1 修复后首轮）

**主要发现：**
- Bug 1（step 11 LR 塌陷）已确认修复：handoff 过渡平滑，无跳变
- r-side 行为：sign guard (~52–64% 步骤 g_dot > 0) 在起作用；r̂_raw 均值 ≈ 0.002，标准差 ≈ 0.024，SNR ≈ 0.08（噪声主导）
- **c_t 完全死亡**：两轮实验 c_t 全程只有两个值（初始值 + handoff 值），300 步无任何更新
- validity guard 条件 2、3 实际无效；g_rms_ema 的 tau_rms=0.05 条件从不触发

### 2. c_t 死亡根因：p_min 绝对阈值破坏 scale-invariance

**因果链：**
- 实际 p_t = α_t × g_dot ≈ 1e-6 × 7e-6 ≈ 7e-12
- p_min = 1e-8（固定绝对阈值）
- 差距 4 个量级 → 300/300 步 calibration 全部跳过

**概念澄清**（用户明确纠正）：
- 错误表述："p_min 是量纲错误"
- 正确表述：**"p_min 是缺乏归一化的绝对阈值，不具备尺度鲁棒性"**
- 含义：带量纲的量可以做阈值化，但阈值必须随系统尺度共变。p_min 不随 LR/梯度范数/维度变化，在不同 base LR 实验之间不可迁移，从而把 c_t controller 绑死在初始 LR 上，破坏了 c → 1/L 的 factorization 含义。

**另一个过头的结论**（也被纠正）：
- 错误结论："唯一合理的 guard 就是 g_dot > 0"
- 正确结论："绝对 p_min 应删，但 final algorithm 是否只需要 sign guard 还要实验确认"
- φ_t 数值上稳定不代表统计上可靠；a_t 和 p_t 都有噪声，比值可能是随机方向的 noisy controller

### 3. 算法真实运行形态的重新定性

```
当前实际：warmup + handoff + sign-gated r-side scaling + dead c-side
设计意图：full two-timescale controller with three-condition validity guard
```

这个区分很关键：当前实验结果支持的 claim 要收缩到：
1. split-batch alignment 的符号有一定信息量
2. 可作为保守 gate 避免部分不可靠更新
3. slow calibration branch 还未真正工作
4. 额外两个 validity 条件在当前量级下无贡献

### 4. c_t guard 重设计 + 代码修改

**两步计划：**
- Step 1（已实施）：删除 p_min，只保留 `g_dot > 0`，跑一轮验证 c_t 是否真正激活
- Step 2（待定）：若 c_t 乱，追加相对化 guard；若正常，维持 sign-only guard

**代码变更（已完成）**：
- `verl/workers/actor/dp_actor.py`：删 p_min 读取，`abs(p_t) > p_min and g_dot > 0` → `g_dot > 0`
- `verl/workers/engine/fsdp/transformer_impl.py`：删 `__init__` 的 p_min 参数和 `self.p_min`；`update_ct` 删除绝对阈值检查
- `verl/workers/fsdp_workers.py`：删 `p_min=` 传参
- 两个实验脚本：删 `signal_fraction_p_min=1e-8` 行

若未来需要追加 guard，**余弦相似度**为首选候选：
`⟨g_C, g_upd⟩ / (‖g_C‖‖g_upd‖) > τ`（无量纲、scale-invariant、语义清晰，论文可解释性强）

### 5. 参数复杂度问题（讨论结论）

- 当前脚本有 13 个 signal_fraction 参数，但真正 free 的只有 ~3–4 个（lr、eta_c、calib_freq、calib_frac）
- 其余要么是理论默认值（phi_ema_beta、r_boot、c_min/c_max），要么已是死代码（p_min 已删、d_min_abs、tau_rms）
- **清理参数的时机**：先把组件行为搞清楚再清理，不能在没数据支撑的情况下提前删除未测试的组件
- 参数多不是根本问题，关键是有无清晰 narrative 区分 free 参数和 fixed 参数

---

## Bug 记录（续）

### Bug 13：p_min 绝对阈值破坏 scale-invariance（2026-04-21 诊断 + 修复）

**发现**：4.21 实验数据（sigfrac_lr1e-5 / lr1e-6）显示 c_t 全程冻结。

**根因**：`signal_fraction_p_min = 1e-8` 作为绝对阈值，与实际 p_t ≈ 7e-12 差 4 个量级。p_min 被同时读入 dp_actor.py（外层 gate）和传给 SignalFractionLRScheduler（update_ct 内层 gate）。

**修复**：完全删除 p_min 参数，只保留 `g_dot > 0` 作为 c_t 更新条件。

**状态**：已修复（2026-04-21）。

---

### Bug 14：phi_t ≈ 1e5（4.22 第一轮修复）

**发现**：4.22 实验前诊断——`_forward_loss_nograd` 使用 rollout policy 的 `old_log_probs`（ratio = π_{θ_{t+1}}/π_rollout），而 `_backward_split` 使用 ratio=1（自指 detach）。两个不同基准，差值放大 5 个数量级。

**修复**：`_forward_loss_nograd` 改为 `old_log_prob = log_prob.detach()`（ratio=1，与 _backward_split 一致）。

**状态**：已修复（2026-04-22 第一轮），但此修复引入 Bug 15。

---

### Bug 15：phi_t ≡ 0（4.22 第二轮修复）

**发现**：4.22 实验（~139步）全程 phi_t = 0.0，phi_bar 从 0.45 单调降至 0.157，c_t 持续被错误驱向下。

**根因**：Bug 14 修复后，`_backward_split`（在 θ_t）和 `_forward_loss_nograd`（在 θ_{t+1}）都使用自指 log_probs（ratio=1）：

- L_C_old = -mean(A_C)（ratio=1 at θ_t，A_C 是固定的）
- L_C_new = -mean(A_C)（ratio=1 at θ_{t+1}，A_C 同样固定）

两者恒等，a_t = L_C_old - L_C_new ≡ 0，phi_t ≡ 0。c_t 因此被持续驱向下（phi_bar < 0.5 → exp(η_c·(phi_bar-0.5)) < 1）。

**修复**：`_backward_split` 新增 `collect_log_probs=True` 参数，在 C 的 backward 时保存 θ_t 的 log_probs；`_forward_loss_nograd` 新增 `old_log_probs_list` 参数，使 ratio = π_{θ_{t+1}}/π_{θ_t}，L_C_new 真正依赖 θ_{t+1}。

**状态**：已修复（2026-04-22）。

---



1. **Theoretical framework formalized** in `paper/theory_derivation.tex`.
   - Every derivation step written out explicitly (no "by standard result" shortcuts).
   - Full MSE derivation (4 steps: center → expand → eliminate cross terms → trace identity).
   - Full $n_{\mathrm{eff}} \approx \min(n, \exp(H))$ derivation (indicator expectation + two-regime analysis for uniform distribution).
   - Full improvement bound derivation (L-smooth descent lemma → expectation → assemble → differentiate → solve for $\alpha^*$).
   - Full linear-scaling simplification ($\sigma^2/n_{\mathrm{eff}} \gg \|g\|^2$ + Assumption 1 → cancel $\|g\|^2$).
2. **Fixed stale $\sqrt{n_{\mathrm{eff}}}$ references** in abstract, RQ box, and Open Questions item 2 — all now say $\propto n_{\mathrm{eff}}$ (linear).
3. **Theory Revision section** marked resolved; Core Thesis updated to use $n_{\mathrm{eff}}$ language throughout.

---



1. **entadapt_initial and entadapt_ema results analyzed** — see `exp_data/4.15/ANALYSIS.md`.
2. **Primary Phase 2 result confirmed**: `entadapt_initial` achieves AIME@300 = 0.296, stable monotonic, no collapse. First successful experimental validation of entropy-adaptive LR.
3. **EMA failure mechanism confirmed by actual LR data**: LR stays near 1e-5 the entire run; peaks at 1e-5 @ step 244. Steps with LR > 5e-6: 294/300 for EMA vs 44/300 for init.
4. **Key insight: EMA LR counter-amplifies in late training** — when entropy plateaus at low values, H_t/H_ema occasionally ≥ 1, pushing LR back to the ceiling. The EMA mode is not just ineffective; it is actively destabilizing in the low-entropy regime.
5. **Comparison between init and EMA modes is a clean controlled experiment**: every hyperparameter identical except H_ref computation. The 82% relative AIME gap (0.296 vs 0.163) is entirely attributable to whether LR actually decreases as entropy declines.
6. **Multi-benchmark results**: EMA ends below starting accuracy on AIME, AIME2025, MINERVA, OLYMPIAD. Init shows +8–19% absolute improvement on all math benchmarks.

---

## Today's Progress (2026-04-14)

1. **SwanLab data downloaded** for 5 new experiments (Phase 1 batch 2 × 4 + advvar_ent). CSV data now in `exp_data/4.14/`.
2. **Comprehensive data analysis completed** — see `exp_data/4.14/ANALYSIS.md` for full report.
3. **Critical insight: entropy decline is the goal, not the disease.**
   - All non-exploding sync experiments converge to entropy ~0.07–0.10 at step 300, regardless of LR or intervention.
   - AIME@300 ranges from 0.017 to 0.325 despite similar final entropy → **the difference is path quality, not terminal state**.
   - LR=1e-5 experiments show "rise-then-fall" AIME pattern (peak ~step 50–70, then decline when entropy < 0.15). LR ≤ 3e-6 experiments show stable/rising accuracy.
   - This reframes the thesis: the goal is not to prevent entropy decline, but to ensure each update respects the current support level during the inevitable decline.
4. **β sweep definitively kills "tune harder" objection**: β=0.001 (too weak, AIME 0.146), β=0.01 (entropy explosion), β=0.1 (immediate catastrophe, AIME=0). Entropy bonus is structurally wrong — it opposes the training objective.
5. **cosine_floor is best open-loop method** (AIME 0.273) — works because it's crude gain scheduling. Only LR=1e-5-starting experiment without late-stage accuracy decline.
6. **advvar_ent (Phase 2) no better than baseline** (AIME 0.158 vs 0.183) — addresses "where to explore" not "how aggressively to update." Wrong axis.
7. **Core thesis updated** to reflect path-quality framing.
8. **Experiment inventory fully clarified** — corrected earlier misidentification:
   - Phase 1 batch 2: β=0.001, β=0.1, cosine+floor, combined_openloop — all ✅ done
   - `deepseek1.5b_sync_8gpu_cosine_ent0.01_lr1e-5` (combined_openloop) ran **twice** with two different SwanLab descriptions
   - `deepseek1.5b_sync_8gpu_base_lr1e-6_500steps` (baseline extension, sync) ran **twice** — previously documented as "never ran"
   - Phase 2: only `sync_advvar_entropy` ✅ done; `entadapt`, `combined_ours` sync/async all ❌ failed

---

## Today's Progress (2026-04-13)

1. **Paper framing fundamentally reworked** — from "entropy-budget control" to "learning-update mismatch":
   - **New core concept**: *Effective gradient support* — the range of policy change that current rollout data can reliably inform. Replaces "entropy budget" as the governing abstraction.
   - **New instability definition**: *Learning-update mismatch* — the update scale exceeds current support. Replaces "entropy overspending".
   - **Entropy demoted from core concept to proxy** — entropy is now explicitly a practical *observable proxy* for support, not the thing we fundamentally care about. "We do not want to keep entropy high forever."
   - **New structural argument**: *Parameter–data coupling* → *support shrinkage* → *mismatch* as a three-step logical chain. Section 3 now establishes the closed-loop structure; Section 4 defines support and mismatch; Section 5 bridges to observable control.
   - **KL critique sharpened**: "measures motion, not remaining support" — framed as a *support mismatch in the monitor itself*, not just estimator blindness.
   - **Intervention reframing**: from "loop stabilization" to "mismatch control" — each intervention is analyzed by whether it addresses the update-to-support ratio.
   - **Related work significantly expanded**: detailed comparison with SAC auto-tuning (different control objective + handle), natural gradient (indirect geometry-based adaptation), trust-region methods (control motion, not track support).

2. **Title changed**: "Learning-Update Mismatch in Self-Generated Data Loops: A Unified View of Learning-Rate Instability in Synchronous and Asynchronous LLM RL"

3. **4 contributions reframed**:
   1. Reformulation of RL instability as learning-update mismatch
   2. Effective gradient support as the governing concept
   3. Unified account of sync/async failure
   4. Entropy as proxy, gain scheduling as control

---

## What To Do Next

### 第一优先级：启动 Phase 2 M / A 实验（Bug 16 已修复）

脚本已就绪，可直接重新启动：
- `new_experiments/signal_fraction_lr/sync_matched_alpha_2.97e-6.sh`（M，gamma=1.0）
- `new_experiments/signal_fraction_lr/sync_sign_gate_gamma0.5.sh`（A，gamma=0.5）

**开跑前置检查（step 1–20）**：

| 步数区间 | 预期行为 |
|---------|---------|
| step 1–10 | alpha_t 走 warmup（0 → 2.97e-6），`c_t` ≈ 2.97e-6，`in_handoff=0` |
| step 11 | handoff 触发：M: `c_t` 跳至 2.97e-6/r_boot；A: 跳至 3.71e-6/r_boot |
| step 11–20 | M: alpha_t ≡ 2.97e-6；A: alpha_t 插值从 2.97e-6 逐步到 3.71e-6 或 1.855e-6 |
| 全程 | `actor/is_calibration_step` 应全为 0（calib_frac=0.0） |

**开跑后分析顺序（先动力学，再分数）**：
1. M 的 alpha_t 是否全程≡2.97e-6（验证 sign_gate_gamma=1.0 的常数效果）
2. A 的 `actor/g_dot_positive` 均值是否接近 0.60；A 的 `actor/alpha_t` post-handoff mean 是否接近 2.97e-6
3. 三组 val AIME24 曲线对比

**判定标准**：
- M > B → slow scale 是主导，r-side 连续 shaping 有害（引入噪声）
- A > M → sign 有独立信息量，即使在 mean-matched 下也有收益
- B > A → 幅度信息确有价值，ratio estimator 在当前 batch size 下仍可用

### 第二优先级：Phase 1 cfixed 补充（可选）

若需要更完整的 c_fixed sweep（用于论文佐证），两个尚未启动的脚本：
- `sync_sigfrac_cfixed_lr7.5e-6.sh`（c_fixed=1.5e-4，典型 alpha≈2.7e-6）
- `sync_sigfrac_cfixed_lr1e-5.sh`（c_fixed=2.0e-4，典型 alpha≈3.6e-6）

### 第三优先级：理论 → 论文（等 Phase 2 结果后）

- 更新 `paper/theory_derivation.tex`：加入 signal-fraction 估计器、c_t 控制律、φ*=1/2 推导、ratio estimator instability 分析
- 重写 main.tex Sections 3–5：替换 entadapt 旧语言，加入新叙事（slow scale + sign-gate）
- 填写 Sections 6–8 placeholder

### 第四优先级：外部比较

- DAPO Clip-Higher（脚本已有）
- AER-style adaptive entropy-coefficient baseline（未实现，最关键外部比较）
- Beyond Precision / VCPO（camera-ready 级别）

---

## If Entropy-Adaptive LR Fails: Contingency and Proxy Improvement Directions

Entropy as a support proxy is **coarse but directionally correct**. The 10× range (0.68 → 0.07) is large enough that even a rough proxy should capture the "support is high" vs "support is depleted" distinction. If the experiments fail despite correct direction/magnitude, the likely causes are implementation details, not fundamental proxy failure. Ordered by likelihood:

### Likely causes (fixable without changing proxy)

1. **min_ratio too high/low** — current floor is 0.1 (LR minimum = 1e-6). If the optimal terminal LR is different, this needs tuning. Easy to sweep.
2. **Warmup interaction** — first 10 steps are warmup (LR: 0 → 1e-5). Entropy is still near H(0) during warmup, so the adaptive ratio ≈ 1 and doesn't interfere. But the transition from warmup to adaptive phase might create a discontinuity. Check LR trajectory.
3. **Entropy lag** — we use step t's entropy to set step t+1's LR (one-step delay). If entropy drops sharply in one step, the LR adjustment is one step late. Unlikely to matter for smooth decline, but could matter if there are entropy spikes.

### Deeper cause: entropy is too coarse as a proxy

Current entropy = **mean per-token Shannon entropy over all response tokens**. Two structural limitations:

1. **Spatial uniformity** — averages over all token positions equally. Most tokens are template/formatting ("Therefore", "\\frac{", etc.) that become deterministic early. The "important" tokens (key reasoning steps) may still have high entropy, but they're drowned out by the mean. The proxy may signal "support is gone" when support at critical decision points is still adequate, or vice versa.

2. **Information compression** — collapses a 151936-dim distribution to a scalar. Two very different distributions can have the same entropy but very different support characteristics.

### Possible proxy improvements (if needed)

Listed from simplest to most complex:

1. **Response-level entropy variance** — instead of mean entropy across tokens, look at variance of entropy across different responses for the same prompt. High variance = model is uncertain about some responses but not others = structured support remains. Low variance = uniformly certain or uniformly uncertain = less informative support.

2. **Advantage-weighted entropy** — compute entropy only on tokens where advantage is non-zero or above a threshold. This focuses on "decision-critical" tokens where the model's uncertainty actually affects learning, ignoring template tokens that contribute noise to the mean.

3. **Rollout diversity** — measure diversity directly at the response level (e.g., unique response ratio, pairwise similarity between n=8 responses per prompt) rather than at the token distribution level. This is the most direct measure of "does the policy still generate diverse enough data for learning," but is more expensive to compute and harder to differentiate through.

4. **Entropy quantiles** — instead of mean entropy, track the 10th/50th/90th percentile of per-token entropy. The 10th percentile tells you about the "most certain" tokens (template), the 90th tells you about "most uncertain" tokens (key decisions). Adaptive LR could be driven by the high quantile rather than the mean.

These are all **refinements of the proxy**, not changes to the framework. The core logic (match update scale to support level) remains the same regardless of which proxy is used.

---

## Technical Notes

- All scripts: verl framework, single node, 8 GPUs
- Sync: 8 GPUs shared, `verl.trainer.main_ppo`
- Async: 4 rollout + 4 training, `verl.experimental.fully_async_policy.fully_async_main`
- Async `require_batches=4` → 4×32×8 = 1024 rollouts/step (matches sync)
- Conda env: `verl2`
- Entropy computed in `verl/workers/actor/dp_actor.py` (~line 653), logged as `actor/entropy`
- LR scheduler built in `verl/workers/engine/fsdp/transformer_impl.py` (~line 413)
- LR scheduler stepped in `verl/workers/engine_workers.py` (~line 370)
- Advantage variance available in GRPO loss computation (group of n=8 responses per prompt)
