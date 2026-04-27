# 4.22 实验分析报告

## 实验设置

**两组实验**：sigfrac_lr1e-5 / sigfrac_lr1e-6，均为 signal_fraction 调度器，同步训练，8 GPU，DeepSeek-R1-Distill-Qwen-1.5B，GRPO。

| 参数 | 值 |
|------|-----|
| warmup_steps | 10 |
| calib_freq | 5 |
| calib_frac | 0.25 |
| eta_c | 0.1 |
| r_boot | 0.05 |
| phi_ema_beta | 0.9 |
| c_min / c_max | 1e-8 / 1e-2 |

**步数**：lr1e-5 跑到 ~139 步，lr1e-6 跑到 ~95 步（未完成）。

---

## 关键指标数据

### r̂_t（信号分数估计）

| 指标 | lr1e-5 | lr1e-6 |
|------|--------|--------|
| r_hat_raw 均值 | 0.0069 | 0.0198 |
| r_hat_raw 负值比例 | **38%** | ~35% |
| r_hat（EMA）均值 | 0.0163 | 0.0205 |
| r_hat 范围 | [0.006, 0.050] | [0.007, 0.050] |

r̂_t 信号噪声大，约三分之一步骤 g_dot < 0（A1、A2 梯度方向相反）。EMA 后典型值约 **0.016–0.020**，远小于 r_boot=0.05。

### c_t 与 α_t

| 指标 | lr1e-5 | lr1e-6 |
|------|--------|--------|
| c_t 初始值（warmup 期） | ~1e-5 | ~1e-6 |
| c_t handoff 后值 | **~2e-4** | **~2e-5** |
| c_t handoff 计算 | α_base / max(r̄_warm, r_boot) = 1e-5/0.05 = 2e-4 | 1e-6/0.05 = 2e-5 |
| alpha_t 均值 | **3.0e-6** | 4.1e-7 |
| alpha_t 范围 | [1e-6, 1e-5] | [1e-7, 1e-6] |

Handoff 机制正确运作：c_T = α_base / r_boot = 2e-4，使得 α_t = c_T × r̂_t ≈ 2e-4 × 0.016 ≈ **3.2e-6**，接近已知最优固定 LR（3e-6）。

### phi_t / phi_bar（校准信号）—— 存在 bug

| 指标 | lr1e-5 | lr1e-6 |
|------|--------|--------|
| phi_t | **0.0 全程** | **0.0 全程** |
| phi_bar 初始 | 0.45 | 0.50 |
| phi_bar 末尾 | **0.157** | **0.266** |
| c_t 变化方向 | **持续下降** | **持续下降** |

**Bug 确认**：phi_t ≡ 0 为代码 bug，非正常信号。详见下节。c_t 因此被错误地持续驱动向下（phi_bar < 0.5 → exp(η_c × (phi_bar−0.5)) < 1）。

### 梯度范数

| 指标 | lr1e-5 | lr1e-6 |
|------|--------|--------|
| g_rms 均值 | 9.97e-7 | 8.09e-7 |
| g_rms 范围 | [6.8e-7, 1.4e-6] | [6.7e-7, 1.0e-6] |
| g_rms_ema 均值 | 9.62e-7 | 7.94e-7 |

两组实验梯度范数量级相同，说明模型状态类似。

---

## Validation 性能（mean@16）

步数：lr1e-5 到 step 120，lr1e-6 到 step 80。

| Benchmark | lr1e-5 last | lr1e-5 max | lr1e-6 last | lr1e-6 max |
|-----------|------------|-----------|------------|-----------|
| AIME24 | **0.300** | **0.300** | 0.240 | 0.240 |
| AIME25 | **0.233** | **0.233** | 0.183 | 0.202 |
| MINERVA | **0.255** | **0.268** | 0.236 | 0.236 |
| OLYMPIAD | **0.490** | **0.490** | 0.391 | 0.398 |
| GPQA | **0.316** | **0.333** | 0.159 | 0.164 |

lr1e-5 全面领先，且仍呈上升趋势（AIME24 step 120 = 0.300，未见顶）。lr1e-6 在 80 步后停止记录，性能明显落后。

**AIME24 逐步数据（lr1e-5）**：

| step | lr1e-5 | lr1e-6 |
|------|--------|--------|
| 0 | 0.210 | 0.196 |
| 20 | 0.250 | 0.196 |
| 50 | 0.273 | 0.223 |
| 80 | 0.260 | 0.240 |
| 100 | 0.275 | — |
| 120 | **0.300** | — |

---

## Bug 记录

### Bug 1（本次实验发现）：phi_t ≡ 0

**现象**：phi_t 全程为 0，phi_bar 单调从 0.5 降至 0.157，c_t 持续下降。

**根因**：4.22 第一轮修复（解决 phi_t ≈ 1e5 的问题）将 `_forward_loss_nograd` 改为 `old_log_prob = log_prob.detach()`，导致：

- L_C_old = -mean(A_C)（ratio=1 at θ_t）
- L_C_new = -mean(A_C)（ratio=1 at θ_{t+1}，同样自指）

两者恒等，a_t = L_C_old - L_C_new ≡ 0，phi_t ≡ 0。

**修复（4.22 当天）**：`_backward_split` 在 C 的 backward 时保存 θ_t 的 log_probs（`collect_log_probs=True`），传给 `_forward_loss_nograd` 作为 old reference。此后 ratio = π_{θ_{t+1}} / π_{θ_t}，L_C_new 真正依赖 θ_{t+1}，a_t ≠ 0。

### 参考：4.21 的 Bug（已修复，不影响本次实验）

p_min 绝对阈值导致 c_t 全程冻结，已于 4.21 删除。本次实验中 c_t 能够更新（11 个唯一值）。

---

## 结论

1. **Handoff 机制有效**：c_T 自动校准使 alpha_t 初始值接近最优 LR（3.2e-6 vs 3e-6）。

2. **phi_t controller 在本次实验中失效**：由于 phi_t ≡ 0 bug，c_t 被错误地持续驱动向下，但因 alpha_t 初始值恰当，前 100 步内影响有限。

3. **r̂_t 信号本身是弱信号**：38% 步骤 g_dot < 0，EMA 后 r̂_t ≈ 0.016–0.020，信噪比低。alpha_t 主要靠 handoff 初始化的 c_t 维持量级，r̂_t 提供形状调制。

4. **lr1e-5 明显优于 lr1e-6**，且仍在上升，尚未达到性能峰值。

---

## 下一步实验方向

核心逻辑：不要同时证明"c_t 学得对"和"r_t 定义得对"，这太难。应先定住总体尺度 c，再单独审问 r_t 是否有独立信息量。

注意：c_fixed ≠ 最优固定 LR（3e-6）。因为 α_t = c_fixed × r_t，而 r_t 典型值 ≈ 0.016，直接设 c_fixed = 3e-6 会使 α_t ≈ 6e-8，远低于合理范围。正确关系：

```
c_fixed = α_target / r_typical ≈ 3e-6 / 0.018 ≈ 1.7e-4
```

---

### Phase 1：找 c_fixed

**方法**：设 `eta_c=0`（冻住 c_t），通过 handoff 自动计算 `c_T = α_base / r_boot = α_base / 0.05`。扫三个 α_base 值：

| α_base | c_fixed (= α_base/0.05) | 典型 α_t (× r_typical=0.018) |
|--------|------------------------|------------------------------|
| 7.5e-6 | 1.5e-4 | ~2.7e-6 |
| 1.0e-5 | 2.0e-4 | ~3.6e-6 |
| 1.25e-5 | 2.5e-4 | ~4.5e-6 |

覆盖已知合理的固定 LR 区间（3e-6 附近），选出 val 性能最好的 c_fixed。零代码改动，只改脚本参数。

---

### Phase 2：测 r_t 独立信息量

固定 Phase 1 找到的最优 c_fixed，对比两个版本：

**Version A — sign-gate**

```
α_t = α_target       if g_dot > 0
α_t = γ · α_target   if g_dot ≤ 0     (γ = 0 / 0.2 / 0.5)
```

测的问题：r_t 的价值是否只在于"方向筛选"？

**Version B — continuous r-shaping**

```
α_t = c_fixed × r_ctrl
```

等价于归一化形式 `α_t = α_target × clip(r_ctrl / r_ref, m_min, m_max)`，其中 r_ref = r_boot = 0.05（handoff 时自然对齐）。

测的问题：r_t 的连续大小是否包含 sign 之外的额外信息？

---

### 判定标准

| 实验结果 | 结论 |
|---------|------|
| Version A 有收益 | r_t 至少有"方向筛选"价值 |
| Version B > Version A | r_t 连续值有额外信息 |
| 两者均无收益 | r_t estimator 本身不够好，应重构，而非折腾 c-side |

对比基线：固定 LR = 3e-6（已知最优）。
