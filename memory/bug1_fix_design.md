---
name: Bug 1 Fix Design — r-side State Machine
description: Full design of the warmup handoff fix + asymmetric EMA + fast-drop + validity guard for signal-fraction LR; 设计 + 实现均已完成 2026-04-20
type: project
---

## Bug 1 现象

Step 11（warmup 结束的第一步）：r̂_raw ≤ r_min → clamp → α_t = c_0 × r_min = 1e-5 × 0.01 = 1e-7，LR 一步塌陷。此后 r̂ 在 [0.01, 0.07] 振荡。

## 根因（两层）

**层 1：hard handoff**：warmup 结束后，从"完全不用 r̂_t"瞬间切到"完全由 r̂_t 决定"，产生 discontinuity。

**层 2：c_0 初始化错位**：c_0 = base_lr（1e-5），但 c_t 语义是 1/L，不是 LR。若 warmup 期间典型 r̄ ≈ 0.04，正确的 c_T 应为 α_base / 0.04 ≈ 2.5e-4，而非 1e-5。

## φ*=1/2 是否要改

**不改。** 用 EMA proxy r̄_t 代替 r_t 后，φ_t ≈ 1 - c_t·L/2·(r_t/r̄_t)，quasi-static 时 r̄_t ≈ r_t，φ*=1/2 仍然对应 c_t → 1/L。改 φ* 会把 shape 信号重新掺进 scale controller，破坏两时间尺度分工。fast-drop 机制负责 lag 问题，φ* 不动。

---

## 最终 r-side 状态机设计

### Warmup 阶段（step 1 到 T_w）

- 继续走原 warmup LR，r̂_t 不控制 LR
- 每步计算 raw r̂_t；过 validity guard（只需前两条：dot>0 且 d_t>d_min_abs）
- Valid 时：用**对称 EMA**（β_sym=0.3）更新 r̄_warm
- Invalid 时：hold r̄_warm
- ḡ_rms：所有 d_t>d_min_abs 的步骤都更新（β_g=0.05），与 validity 解耦
- 初始化：r̄_0_warm = r_boot；ḡ_rms 在第一个 d_t>d_min_abs 步直接初始化为 g_rms,1

### Handoff（step T_w）

```
c_Tw = α_base / max(r̄_Tw_warm, r_boot)
r̄_Tw_post = r̄_Tw_warm            ← EMA 连续，不重置
```

接下来 M=10 步做插值过渡：

```
α_t = (1 - λ_t) · α_base + λ_t · (c_t · r̄_t)
λ_t = (t - T_w) / M，线性递增到 1
```

插值期间 r̄_t 切换到 asymmetric EMA（β↓=0.5, β↑=0.1）正常更新。

### Post-handoff 正常状态

**Validity guard（三条件）：**
1. dot(ĝ_A1, ĝ_A2) > 0
2. d_t > d_min_abs
3. g_rms,t > τ_rms · ḡ_rms,t

其中 d_t = (||ĝ_A1||² + ||ĝ_A2||²) / 2，g_rms,t = sqrt(||ĝ_upd||² / n_params_global)

**ḡ_rms 更新规则（独立于 validity）：**
- 只要 d_t > d_min_abs，就更新：ḡ_rms,t = (1-β_g)·ḡ_rms,t-1 + β_g·g_rms,t

**r̄_t 更新（asymmetric EMA，只在 valid 时）：**
```
若 r_t_obs < r̄_{t-1}：r̄_t = (1-β↓)·r̄_{t-1} + β↓·r_t_obs   ← 下行快跟
若 r_t_obs ≥ r̄_{t-1}：r̄_t = (1-β↑)·r̄_{t-1} + β↑·r_t_obs   ← 上行慢跟
若 invalid：hold r̄_t
```

**控制量：**
```
r_t_ctrl = max(min(r̄_t, r_t_obs_valid), r_min_ctrl)
α_t = c_t × r_t_ctrl
```

**c_t：** 低频 calibration（每 K=5 步），φ*=1/2，正常更新。

### Fast-drop 触发

条件：r_t_obs valid **且** r_t_obs < ρ·r̄_{t-1}（ρ=0.7）

触发时：
- r_t_ctrl = max(r_t_obs, r_min_ctrl)（跳过 min(r̄_t, ...)，直接用 raw obs）
- 进入 cooldown，counter = H
- **冻结 c_t**

### Cooldown 状态

- 继续更新 r̄_t（asymmetric EMA，**不冻结**）
- 冻结 c_t：c_{t+1} = c_t
- calibration step 跳过（或仅记录 φ_t 做 log，不更新 c_t）
- 若再次触发 fast-drop：counter reset to H（延长 freeze）
- 退出条件：counter 归零（无需额外规则；r̄_t 追上新低位后触发条件自然消失）

**Cooldown counter 递减规则**：每个 training step 在 `update_r_and_set_lr` 末尾无条件递减一次（不依赖 calibration 或信号质量）。这样 `cooldown_steps=H` 精确对应 H 个 training step。

---

## 参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| r_boot | 0.05 | warmup EMA 初值 / c_T 分母 floor |
| β_sym | 0.3 | warmup 对称 EMA |
| β↓ | 0.5 | post-handoff 下行追踪 |
| β↑ | 0.1 | post-handoff 上行平滑 |
| β_g | 0.05 | ḡ_rms 背景 baseline EMA（慢，抗扰动）|
| d_min_abs | 1e-30 | 绝对数值 floor（防 underflow/NaN）|
| τ_rms | 0.05 | 相对 RMS 阈值 |
| ρ | 0.7 | fast-drop 触发阈值 |
| H | 5 | cooldown 步数（training step，不是 calibration step）|
| r_min_ctrl | 0.01 | 控制量 safety floor |
| M | 10 | handoff 插值步数 |

---

## 实现状态：**已完成**（2026-04-20）

**Why**: Bug 1 实验结果（step 11 LR 塌陷）暴露了旧设计的两个根因。

**How to apply**: 修改了 4 个文件，见 engineering_impl.md。旧的 `r_min` clamp 废弃，改为 r_min_ctrl 语义（EMA 之后的 safety floor）。

---

## 代码审查发现并修复的 bug（2026-04-20）

### Bug A（重要）：cooldown counter 永远不归零

**问题**：旧设计把 `_cooldown_counter -= 1` 放在 `update_ct` 里，而 `update_ct` 只在满足 `abs(p_t) > p_min AND g_dot > 0` 时调用。fast-drop 期间梯度方向可能混乱，导致该条件一直失败，counter 永远不递减，c_t 被永久冻结。

**修复**：把递减逻辑移到 `update_r_and_set_lr` 末尾（每 training step 无条件执行一次）。`update_ct` 只做检查（不递减）。位置：`transformer_impl.py:443-445`。

### Bug B（轻微，metric 精度）：n_params 是 shard-local 的

**问题**：FSDP 下每卡只持有参数的 1/D 分片，`sum(p.numel())` 给出的是 shard-local count，导致 `g_rms_t` 偏大 sqrt(D) 倍。validity check 是 ratio 比较（g_rms_t / g_rms_ema），比例不变，**算法正确性不受影响**，只有 `actor/g_rms` 绝对值偏大。

**修复**：第一步对 n_params 做 all_reduce(SUM) 得全局值，缓存到 `self._sf_n_params_global` 避免重复通信。位置：`dp_actor.py:744-752`。

### Bug C（代码整洁）：r_min 读了但不用

**问题**：旧代码用 `r_min` 做 clamp，新设计 clamp 逻辑已移入 scheduler。变量读出后无引用。

**修复**：删除 `r_min = self.config.optim.get("signal_fraction_r_min", 0.01)` 这一行。位置：`dp_actor.py:562`（已删除）。

### Bug D（重要）：calib_groups 在 n_groups < 2 时变为负数

**问题**：`calib_groups = min(calib_groups, n_groups - 2)` 当 `n_groups=1` 时 = -1，导致 `update_groups=2` 超出实际 group 数，data_A2 成为空切片（backward 在空 batch 上），ĝ_A2=0，ĝ_upd = ĝ_A1/2，等效浪费一半梯度。不报错，大规模训练（多卡 + 小 batch 时）会静默触发。

**修复**：`calib_groups = max(0, min(calib_groups, n_groups - 2))`，加 `max(0,...)` 保证非负。位置：`dp_actor.py:576`。

