# 4.26 多 Seed 三角对照实验分析报告（9 runs）

## 1. 实验集合与分析目标

本轮分析对象为 `exp_data/4.26` 下的 9 组实验：

- 3 个条件：M / A / B
  - **M**: `matched_alpha_2.97e-6`（固定 LR 基线）
  - **A**: `sign_gate_meanmatch_gamma0.5`（sign gate）
  - **B**: `sigfrac_cfixed_lr1.25e-5`（continuous r-shaping）
- 3 个 seed：`42 / 0 / 1`

分析遵循预先约定顺序：

1. 先做**机制/动力学校验**（alpha 轨迹、gate 是否真生效、训练动力学是否对齐）  
2. 再看**最终性能**（final / peak / drop，mean±std，seed 配对差分）

---

## 2. 数据清单与可用指标

目录结构包含：

- `val-core/`：AIME, AIME2025, GPQA, MINERVA, OLYMPIAD
- `actor/`：`alpha_t`, `entropy`, `ppo_kl_mean`, `grad_norm`, `r_hat`, `r_ctrl`, `c_t`, `g_dot_positive` 等
- `critic/`：score/reward/advantage/return
- `response-len/`：平均 response length

关键用于判定的指标：

- 机制层：`actor/alpha_t`, `actor/g_dot_positive`, `actor/r_hat`, `actor/r_ctrl`, `actor/c_t`
- 动力学层：`actor/entropy`, `actor/ppo_kl_mean`, `actor/grad_norm`, `response_length/mean`
- 结果层：`val-core/*/acc/mean@16`

---

## 3. 机制校验（先于分数）

## 3.1 M 组（固定 LR）是否真是常数基线

是。step11 后 `alpha_t` 严格常数：

- `alpha_step10 = 2.673e-6`
- `alpha_step11 = 2.97e-6`
- `alpha_mean(step11-300) = 2.97e-6`
- `alpha_std(step11-300) ≈ 0`

说明 warmup/handoff 与常数段行为正常。

## 3.2 A 组（sign-gate）是否按 two-level gate 执行

是。日志显示两档学习率准确为：

- `alpha+ = 3.71e-6`
- `alpha- = 1.855e-6`

并且用 `P(g_dot>0)` 反推 mean alpha 与实测几乎完全一致（step21-300）：

- seed0: `p=0.5214`，预测 `2.82225e-6`，实测 `2.82265e-6`
- seed1: `p=0.5714`，预测 `2.91500e-6`，实测 `2.91540e-6`
- seed42: `p=0.5560`，预测 `2.88630e-6`，实测 `2.88670e-6`

结论：A 组 gate 机制本身没有跑偏。

## 3.3 B 组（continuous）是否与 M 完全 mean-match

**不完全匹配**。在稳定区间 `step21-300`：

- M: `2.9700e-6 ± 0`
- A: `2.8749e-6 ± 4.75e-8`
- B: `3.0953e-6 ± 1.56e-7`

即 B 比 M 高约 `+4.2%`，A 比 M 低约 `-3.2%`。  
这会成为后续结果解释的残余混杂因素（需在论文中显式说明）。

---

## 4. 训练动力学（机制后再看）

## 4.1 全程均值（step1-300）

| 条件 | entropy mean | KL mean | grad norm mean | response len final |
|---|---:|---:|---:|---:|
| M | 0.4687 ± 0.0179 | 4.005e-4 ± 1.27e-5 | 0.03932 ± 0.00050 | 2671 ± 250 |
| A | 0.4416 ± 0.0110 | 4.003e-4 ± 6.00e-6 | 0.03966 ± 0.00040 | 2660 ± 52 |
| B | 0.4210 ± 0.0610 | 3.906e-4 ± 1.53e-5 | 0.04003 ± 0.00076 | 2800 ± 102 |

观察：

- KL/grad norm 三组量级接近，说明不是“总体更新强度失控”造成的差异。
- B 的平均 entropy 更低，但性能并未劣化，延续了此前“低 entropy 不必然坏”的现象。
- 需要保留谨慎项：KL/grad norm 的**均值接近**不等于完全排除强度差异；若要进一步排除该路径，建议补看 `p95/max` 和 late-stage 分布。

## 4.2 分阶段 `P(g_dot>0)`（对齐信号强度）

| 条件 | warmup(1-20) | early(21-100) | mid(101-200) | late(201-300) |
|---|---:|---:|---:|---:|
| M | 0.783 | 0.517 | 0.503 | 0.517 |
| A | 0.800 | 0.552 | 0.577 | 0.520 |
| B | 0.800 | 0.563 | 0.545 | 0.545 |

要点：

- A 在 early/mid 高于 M，但 late 下降到 `0.520`，接近随机边界。
- 这与先前“sign 信号后期衰弱”的机制判断一致。

---

## 5. 核心结果：5 个 benchmark 的 9-run 统计

## 5.1 Final（mean ± std，3 seeds）

| Benchmark | M | A | B |
|---|---:|---:|---:|
| AIME | 0.2889 ± 0.0043 | 0.3153 ± 0.0084 | **0.3278 ± 0.0052** |
| AIME2025 | 0.2354 ± 0.0163 | 0.2424 ± 0.0202 | **0.2479 ± 0.0072** |
| GPQA | 0.3506 ± 0.0159 | 0.3377 ± 0.0062 | **0.3565 ± 0.0166** |
| MINERVA | **0.2833 ± 0.0036** | 0.2779 ± 0.0048 | 0.2819 ± 0.0110 |
| OLYMPIAD | 0.4927 ± 0.0063 | 0.4867 ± 0.0019 | **0.5077 ± 0.0018** |

## 5.2 Peak 与 Peak-to-Final Drop（mean ± std）

### AIME

- M: peak `0.3229 ± 0.0127`, drop `-0.0340 ± 0.0084`
- A: peak `0.3375 ± 0.0127`, drop `-0.0222 ± 0.0136`
- B: peak `0.3431 ± 0.0084`, drop `-0.0153 ± 0.0136`

### AIME2025

- M: peak `0.2597 ± 0.0105`, drop `-0.0243 ± 0.0209`
- A: peak `0.2625 ± 0.0021`, drop `-0.0201 ± 0.0189`
- B: peak `0.2674 ± 0.0024`, drop `-0.0194 ± 0.0087`

### GPQA

- M: peak `0.3552 ± 0.0098`, drop `-0.0046 ± 0.0074`
- A: peak `0.3513 ± 0.0125`, drop `-0.0135 ± 0.0183`
- B: peak `0.3650 ± 0.0065`, drop `-0.0085 ± 0.0122`

### MINERVA

- M: peak `0.2877 ± 0.0080`, drop `-0.0044 ± 0.0056`
- A: peak `0.2850 ± 0.0013`, drop `-0.0071 ± 0.0045`
- B: peak `0.2892 ± 0.0020`, drop `-0.0073 ± 0.0121`

### OLYMPIAD

- M: peak `0.5023 ± 0.0024`, drop `-0.0096 ± 0.0075`
- A: peak `0.4923 ± 0.0032`, drop `-0.0056 ± 0.0029`
- B: peak `0.5123 ± 0.0058`, drop `-0.0046 ± 0.0040`

---

## 6. Seed 配对差分（最关键的稳健性证据）

下面是每个 benchmark 的 seed 配对 `final` 差分（先算每 seed，再求均值）：

## 6.1 A - M

- AIME: `+0.0264 ± 0.0048`（3/3 seed 为正）
- AIME2025: `+0.0069 ± 0.0337`（不稳定）
- GPQA: `-0.0129 ± 0.0105`（3/3 seed 为负）
- MINERVA: `-0.0054 ± 0.0022`（3/3 seed 为负）
- OLYMPIAD: `-0.0060 ± 0.0045`（3/3 seed 为负）

## 6.2 B - M

- AIME: `+0.0389 ± 0.0064`（3/3 正）
- AIME2025: `+0.0125 ± 0.0146`（偏正）
- GPQA: `+0.0058 ± 0.0175`（混合）
- MINERVA: `-0.0015 ± 0.0129`（近 0，混合）
- OLYMPIAD: `+0.0150 ± 0.0056`（3/3 正）

## 6.3 B - A

- AIME: `+0.0125 ± 0.0072`
- AIME2025: `+0.0056 ± 0.0272`（不稳定）
- GPQA: `+0.0187 ± 0.0174`
- MINERVA: `+0.0040 ± 0.0119`
- OLYMPIAD: `+0.0210 ± 0.0013`（非常稳）

---

## 7. 跨任务综合指标

## 7.1 每个 run 的 5-benchmark final 平均

- A:
  - seed42: 0.3376
  - seed0: 0.3283
  - seed1: 0.3301
- B:
  - seed42: 0.3470
  - seed0: 0.3419
  - seed1: 0.3442
- M:
  - seed42: 0.3310
  - seed0: 0.3333
  - seed1: 0.3262

条件均值（mean ± std）：

- **B: 0.3443 ± 0.0025**
- A: 0.3320 ± 0.0050
- M: 0.3302 ± 0.0036

## 7.2 胜场统计（15 个单元 = 5 benchmark × 3 seeds）

- B: **10.5**
- M: 3.5
- A: 1.0

（并列按分数平分计数）

---

## 8. 结论（可声明 vs 不可声明）

## 8.1 当前数据可支持的声明

1. **B（continuous r-shaping）是当前 overall 最强条件（near-mean-match 前提下）**：跨任务平均最优，且胜场明显领先。  
2. **A（sign-gate）应定位为 task-specific stabilizer，而非 overall controller**：AIME 上稳定有利，但在 GPQA/OLYMPIAD/MINERVA 上相对 M 稳定不占优。  
3. **sign-only 不足以解释 B 的收益**：`B-A` 在 OLYMPIAD 上稳定为正，在 GPQA/AIME 也偏正，说明 continuous 幅度信息有独立价值。  
4. **机制执行正确且可复核**：A 的 mean alpha 与 `P(g_dot>0)` 预测几乎完全一致，M 为严格常数，B 的 r-shaping 动态正常。

## 8.2 必须保留的限制（避免过度声明）

1. 本轮并非严格完美 mean-match（稳定区间 B 均值 alpha 略高、A 略低于 M）。  
2. AIME2025 上 A/B 与 M 的优势方差较大，不能作强结论。  
3. `P(g_dot>0)` 的统计显著性不能按 iid 处理（时序相关）。  
4. 不能把本轮结果解释为 c-side 收敛证据（当前实验主线仍是 cfixed 对照框架）。
5. 不应把 B 的收益“完全归因于 r 信号本身”：当前更准确表述是“在相近 mean LR 且 KL/grad norm 均值接近时，B 整体最好”。

---

## 9. 对后续实验与写作的直接建议

1. 若要消除 reviewer 对 mean-match 的质疑，建议补两类对照其一（或同时做）：  
   - 更严格匹配：让 B 的 step21-300 mean alpha 对齐到 `2.97e-6`（当前约 `3.10e-6`）；  
   - 补充常数基线：新增 `constant LR = 3.10e-6`，与当前 B 的实际 mean alpha 对齐。  
2. 论文叙事建议采用“任务依赖控制形式”：
   - AIME 类：sign 的保守门控可改善后期稳定性；
   - 宽任务（尤其 OLYMPIAD/GPQA）：continuous 幅度信息提供额外收益。  
3. 报告结果时优先使用“seed 配对差分 + mean±std”，避免只给单 seed 曲线。  
4. 若要进一步排除“强度差异解释”，补充 KL/grad norm 的 `p95/max` 与 late-window（如 step200-300）统计。

