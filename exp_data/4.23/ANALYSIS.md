# Phase 1 cfixed 实验分析报告

**日期**：2026-04-23
**实验**：`sigfrac_cfixed` × 3 configs（lr1.25e-5 / lr1e-5 / lr7.5e-6），各跑 300 步
**目的**：找到最优 c_fixed（eta_c=0，冻住 c_t），为 Phase 2 测 r_t 独立信息量做准备

---

## 一、前置检查

### 1.1 is_calibration_step 异常（Logging Bug，已修复）

实验日志中 `is_calibration_step` 每 5 步出现一次非零值（step 1, 6, 11, ...），与 `calib_frac=0.0` 的预期（全为 0）不符。

**诊断**：纯 logging bug，不影响计算。`is_calibration` 由 `step_count % calib_freq` 独立决定，与 `calib_frac` 无关；C-split 和 c_t 更新均受 `data_C is not None` guard 保护——`calib_frac=0.0` 时 `calib_groups=0`，`data_C=None`，calibration branch 全程未运行。`phi_bar` 全程精确 = 0.500 确认此点。

**已修复**：[dp_actor.py:835](../workers/actor/dp_actor.py#L835) 改为 `float(is_calibration and data_C is not None)`。

**本轮实验设定确认**：这组实验是干净的 post-handoff fixed-c、无 calibration 的 r-only 设定。

### 1.2 in_handoff

| 步数区间 | in_handoff | 预期 |
|---------|-----------|------|
| step 1–11 | 0 | 0 ✓ |
| step 12–20 | 1 | 1 ✓（handoff_steps=10）|
| step 21+ | 0 | 0 ✓ |

三个实验完全一致，handoff 触发时机正确。

### 1.3 c_t 冻结确认

| 实验 | warmup c_t | handoff c_T（实际） | 期望 c_T=lr/r_boot | 偏差原因 |
|------|-----------|-------------------|--------------------|---------|
| lr1.25e-5 | 1.25e-5 | **2.500e-4** | 2.500e-4 | 精确（r_warm=r_boot=0.05）|
| lr1e-5 | 1.0e-5 | **1.855e-4** | 2.000e-4 | r_warm=0.054 > r_boot → c_T=lr/r_warm |
| lr7.5e-6 | 7.5e-6 | **1.500e-4** | 1.500e-4 | 精确 |

三个实验 c_t 全程只有两个唯一值（warmup值 / handoff值），eta_c=0 冻结正常。

lr1e-5 的 c_T 偏低（1.855e-4 而非 2.000e-4）：warmup 期 r_warm_eff=0.054 略超过 r_boot=0.05，handoff 公式取 `lr / max(r_warm, r_boot)` 导致分母更大，使该实验的 effective scale 偏向 lr7.5e-6 一侧。

---

## 二、alpha_t 有效学习率

### 2.1 warmup 阶段（step 1–11）

三个实验均为线性增长（0 → lr），step 11 到达 lr 本身：
- lr1.25e-5：step1=1.25e-6 → step11=1.25e-5
- lr1e-5：step1=1.0e-6 → step11=1.0e-5
- lr7.5e-6：step1=7.5e-7 → step11=7.5e-6

### 2.2 post-handoff 阶段（step 21+）

| 实验 | mean | median | p25 | p75 | max |
|------|------|--------|-----|-----|-----|
| lr1.25e-5 | 2.97e-6 | **2.50e-6** | 2.50e-6 | 3.19e-6 | 5.48e-6 |
| lr1e-5 | 2.66e-6 | 2.54e-6 | 1.91e-6 | 3.16e-6 | 4.27e-6 |
| lr7.5e-6 | 2.05e-6 | **1.97e-6** | 1.50e-6 | 2.48e-6 | 3.58e-6 |

三个实验的 alpha_t 区间有分离（lr1.25e-5 > lr1e-5 > lr7.5e-6），趋势符合预期，但间距被压缩：

- **预期**：alpha_t ≈ c_T × r_typical(0.018) → 2.7e-6 / 3.6e-6 / 4.5e-6，比例 1.0 : 1.33 : 1.67
- **实际**：2.05e-6 / 2.66e-6 / 2.97e-6，比例 1.0 : 1.30 : 1.45

压缩原因见下节。

---

## 三、r-side 行为分析（关键发现）

### 3.1 r_hat_raw 原始估计

| 实验 | 总步数 | neg 比例 | pos 均值 | [0, 0.01) | [0.01, 0.02) | [0.02, 0.05) | [0.05, 1) |
|------|-------|---------|---------|----------|------------|------------|---------|
| lr1.25e-5 | 59 | **41%** | 0.022 | 9 | 10 | 12 | 4 |
| lr1e-5 | 59 | **34%** | 0.028 | 5 | 14 | 15 | 5 |
| lr7.5e-6 | 55 | **45%** | 0.026 | 5 | 8 | 14 | 3 |

SNR 极低：neg 比例 34–45%，pos 均值约 0.02–0.03，超过 90% 的步骤 r_hat_raw < 0.05。三个实验的 r_hat_raw 分布几乎相同——符合理论预期（r̂_t 是 θ_t 处的梯度统计量，与 c_fixed 无关）。

### 3.2 r_ctrl 与 r_min 的关系（核心发现）

| 实验 | 贴 r_min(=0.01) 比例 | r_ctrl 均值 | r_ctrl 最大值 |
|------|-------------------|-----------|------------|
| lr1.25e-5 | **53%** | 0.0119 | 0.0219 |
| lr1e-5 | 21% | 0.0143 | 0.0230 |
| lr7.5e-6 | 29% | 0.0136 | 0.0238 |

r_ctrl 全程在 [r_min, 0.024] 的极窄区间内。lr1.25e-5 超过一半时间 r_ctrl = r_min = 0.01，alpha_t 中位数精确等于 c_T × r_min = 2.5e-4 × 0.01 = 2.5e-6。

当前三组实验更接近**以不同 fixed scale 为主、叠加有限 r-side 调制**的 regime，而非真正的 continuous r-shaping regime：

- lr1.25e-5：超过一半步骤 alpha_t 由 c_T × r_min 决定（固定值），其余步骤有一定连续调制（max 5.48e-6）
- lr1e-5 / lr7.5e-6：贴底比例较低（21% / 29%），p25–p75 跨度有一定宽度，r-side 调制作用更明显，但仍非主导

**r-side 的连续 shape modulation 被显著压缩，尚未成为主导因素。** 差异主要来自 c_T 的不同（即 base LR 的不同）导致的 scale 差异。

这也解释了 alpha_t 区间被压缩的根本原因：r_ctrl 很少到达较高值，使三组在 LR 量级上更接近彼此，而非理论预期的 2.7e-6 / 3.6e-6 / 4.5e-6 均匀分布。

### 3.3 g_A1_dot_A2 split-batch alignment

| 实验 | g_dot > 0 比例 | 均值 |
|------|-------------|------|
| lr1.25e-5 | 59% | 1.4e-5 |
| lr1e-5 | 66% | 4.0e-5 |
| lr7.5e-6 | 55% | 1.7e-5 |

g_dot > 0 比例 55–66%，优于随机的 50%，说明 split-batch alignment 有一定方向信号量，但 SNR 低，不足以单独主导训练行为。

---

## 四、训练动力学

### 4.1 Entropy 衰减

| 实验 | step 1 | step 10 | step 20 | step 50 | 趋势 |
|------|--------|---------|---------|---------|------|
| lr1.25e-5 | 0.678 | 0.651 | **0.431** | **0.411** | step 11–20 急速下降，此后稳定振荡 |
| lr1e-5 | 0.677 | 0.585 | 0.468 | 0.464 | 平缓下降后稳定 |
| lr7.5e-6 | 0.675 | 0.650 | 0.553 | 0.460 | 更平缓 |

lr1.25e-5 在 step 11–20（handoff 插值期，alpha_t 接近 base lr）出现 entropy 急速下降（0.625 → 0.410）。handoff 完成后长期 effective alpha 被压回 2.5–3e-6，训练趋于稳定。

**注意**：handoff + r-side 将长期 effective alpha 压回了较安全区间，从而避免了高 base LR 持续作用下的进一步失稳；但 handoff 过渡期本身仍然伴随一次明显的 entropy 快速消耗，并非完全无代价。

lr1.25e-5 在 step 50 时 entropy（0.411）明显低于 lr1e-5 和 lr7.5e-6（0.464 / 0.460），说明它在前50步消耗 budget 更快，学习也更积极。

### 4.2 pg_loss

| 实验 | step 1 | step 50 | 降幅 |
|------|--------|---------|------|
| lr1.25e-5 | 0.1448 | 0.0592 | -59% |
| lr1e-5 | 0.1327 | 0.0472 | -64% |
| lr7.5e-6 | 0.1430 | 0.0610 | -57% |

三者降幅相近，均从 0.13–0.14 降至约 0.05–0.06，无异常。

### 4.3 Response Length

| 实验 | step 1 | step 50 | step 100 | step 200 | step 300 | 趋势 |
|------|--------|---------|---------|---------|---------|------|
| lr1.25e-5 | 6194 | 2368 | 2535 | 2743 | **2918** | 持续回升 |
| lr1e-5 | 6158 | 2079 | 2339 | 2642 | 2650 | 回升后平台 |
| lr7.5e-6 | 6210 | 2258 | 2418 | 2650 | 2583 | 平台后微降 |

三个实验在 step 50 附近均快速收缩（policy 快速集中），之后缓慢回升。lr1.25e-5 的 response_len 持续增长至 2918 且仍有上升趋势，说明其 policy 探索仍然活跃；lr7.5e-6 在 step 300 附近轻微回落，与其 gpqa 不稳定迹象一致。

---

## 五、val 性能

### 5.1 长程数据（OLYMPIAD / gpqa / MINERVA，每步评估，300步完整）

| 实验 | OLYMPIAD@300 | gpqa@300 | MINERVA@300 | 趋势 |
|------|------------|---------|------------|------|
| **lr1.25e-5** | **0.5056** | **0.3725** | **0.2900** | 仍在上升 |
| lr1e-5 | 0.4881 | 0.3337 | 0.2712 | 平台期 |
| lr7.5e-6 | 0.4863 | 0.3038 | 0.2600 | 轻微下降 |

lr1.25e-5 全面领先，且三个 benchmark 在 step 300 时均仍在上升，尚未到平台期。

lr7.5e-6 在 gpqa 上出现明显波动（step200=0.301 → step250=0.328 → step300=0.304），有不稳定迹象，与 response_len 微降一致。

### 5.2 短程数据（AIME / AIME2025，每10步评估，~50步）

| 实验 | AIME@50 | AIME2025@50 |
|------|---------|------------|
| lr7.5e-6 | **0.2604** | **0.2208** |
| lr1.25e-5 | 0.2479 | 0.2062 |
| lr1e-5 | 0.2417 | 0.2104 |

短程 AIME 上 lr7.5e-6 略领先，排序与长程相反。AIME 只有 50 步数据，峰值均在 step 20–30 出现后即下降（lr1e-5 峰值 0.2875@step30）。

**关于短程 vs 长程排序的解读**：如果目标是选用于 Phase 2 的通用 c_fixed，应优先参考长程、完整的数据；但短程 AIME 的反向排序提示存在明显的 task dependence——较高 effective scale（lr1.25e-5）在 AIME 类短期数学任务上可能更快过冲，这一现象不应被忽略。

### 5.3 各实验长程曲线峰值时机

| 实验 | OLYMPIAD 峰 | gpqa 峰 | MINERVA 峰 |
|------|-----------|--------|-----------|
| lr1.25e-5 | 0.5056 @ step300 | 0.3725 @ step300 | 0.2900 @ step300 |
| lr1e-5 | 0.4913 @ step290 | 0.3431 @ step170 | 0.2781 @ step160 |
| lr7.5e-6 | 0.4975 @ step260 | 0.3275 @ step250 | 0.2794 @ step290 |

lr1.25e-5 的所有峰值均在 step300，说明仍有提升空间。lr1e-5 在 gpqa/MINERVA 上在 step 160–170 已出现峰值后回落，lr7.5e-6 的峰值时机较晚但曲线趋于平坦。

---

## 六、综合解读

### 6.1 当前结果最主要反映 effective scale 差异

性能排序（lr1.25e-5 > lr1e-5 > lr7.5e-6）与 post-handoff mean alpha_t（2.97e-6 > 2.66e-6 > 2.05e-6）高度一致，表明**长期性能排序首先由 effective step-size scale 决定**：c_fixed=2.5e-4 对应的 post-handoff alpha 落在约 2.5–3e-6 的区间，恰好接近此前 LR sweep 实验确认的 Goldilocks 区间（最优固定 LR = 3e-6），因此表现最好。

**当前结果尚不能证明 r-side 连续值本身有主要贡献**。

### 6.2 r_min 抑制了 continuous r-shaping 的体现

r_min=0.01 相对于 r_hat 的实际分布偏高：
- r_hat_raw 正值均值约 0.02–0.03，约 25% 的正值低于 r_min
- lr1.25e-5 有 53% 步骤 r_ctrl 贴底，r-side 的信息量被 clip 在 r_min 处

r_min 实际上成了有效 LR 的主要 floor，而不是 r_hat 的自然下界。**若要真正测试 r_t 的连续 shape modulation，需要下调 r_min 以释放更多连续调制空间；但下调到何种量级，还需要在"保留 shape information"与"避免噪声直通"之间做小网格验证，不宜直接指定目标值。**

### 6.3 r_hat 是弱信号，但不是纯噪声

neg 比例 34–45%（随机基准 = 50%），g_dot > 0 比例 55–66%，说明 split-batch alignment 有一定方向性信息，但 SNR 很低。降低 r_min 后，噪声对 alpha_t 的扰动会增大，需要实验确认这是否会造成不稳定。

---

## 七、结论与后续

### Phase 1 结论

**最优 c_fixed = 2.500e-4（对应 lr1.25e-5）**，alpha_t 中位数 ≈ 2.5e-6，在所有长程 benchmark 上全面领先且 step 300 仍在上升。

一句话总结：Phase 1 已经基本找到当前最合适的 fixed slow scale，c_fixed ≈ 2.5e-4；r-side 不是纯噪声，但目前还是弱信号，其连续值的独立贡献尚未被当前结果充分识别，这与 r_min 设置偏高有关。

### Phase 2 实验设计

以 c_fixed=2.5e-4 为基准，进行以下三组严格对照：

| 实验 | 描述 | 目的 |
|------|------|------|
| **M — matched constant-alpha baseline** | 固定 LR = c_fixed × r_mean ≈ 2.97e-6（与本轮 lr1.25e-5 的 mean alpha 精确匹配） | 回答：lr1.25e-5 这组的收益有多少其实只是 scale 选对了；将 r-side 的净贡献从 scale 效应中分离 |
| **A — sign-gate** | alpha_t = alpha_target (g_dot>0), γ × alpha_target (g_dot≤0)，γ ∈ {0, 0.2, 0.5} | r_t 的符号信息是否有独立价值 |
| **B — continuous r-shaping** | alpha_t = c_fixed × r_ctrl，r_min 下调后 | r_t 连续值是否优于简单符号门控 |

**判定逻辑**：
- Version A > M → r_t 符号有方向筛选价值，超越单纯 scale 匹配
- Version B > A → r_t 连续值提供额外信息，值得使用
- Version B ≈ M → r_t 调制仅相当于 conservative LR scaling，无独立信息

**关于 r_min**：Phase 2 开始前先做小网格（例如 r_min ∈ {0.01, 0.005, 0.002}），确认降低 r_min 后训练不会因 r_hat 噪声放大而失稳，再选定 Phase 2 的 r_min 值。
