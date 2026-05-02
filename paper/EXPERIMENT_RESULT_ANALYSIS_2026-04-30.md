# DeepSeek-1.5B Signal-Fraction 实验结果分析（截至 2026-04-30）

## 1. 分析目标与范围

本文档基于：

- `paper/PROJECT_STATUS.md`
- `exp_data/4.14/ANALYSIS.md`
- `exp_data/4.23/ANALYSIS.md`
- `memory/engineering_impl.md`（4.24–4.28 对照结论）

目标是建立一条完整证据链：

`弱单步信号 → 低频聚合 → 更稳定的 LR trajectory`。

---

## 2. 当前主线（4.30 checkpoint）

4.30 的问题定义不是“单步 sign 是否准确”，而是：

- 单步 `g_A1^T g_A2` SNR 偏低（`g_dot_positive` 约 55%）；
- 如何从 noisy split 信号中提取低频成分，得到更稳且不欠更新的 alpha trajectory。

4.30 变体（当前重点）：

- slow EMA: `ret0.95`, `ret0.98`
- alpha-rate-limit: `alpharlim0.05`
- windowed-r: `W5`, `W10`

---

## 3. 历史证据链（4.14 → 4.23 → 4.24–4.28）

## 3.1 4.14：open-loop 干预边界

4.14 的关键结论应精确表述为：

1. 固定 entropy bonus 不稳健（β sweep 失败）；
2. open-loop `cosine_floor` 是较强 baseline，但仍弱于 `sync_lr3e-6`；
3. `advvar_ent` 改变了探索分配，但未解决后期 update/support 失配。

因此 4.14 **不是**“所有 open-loop schedule 都无效”的证明，而是说明纯 open-loop 路径上限有限。

## 3.2 4.23：cfixed 的一阶结论

4.23 更准确的结论是：

- slow scale（effective alpha）是一阶主导因素；
- r 连续值独立贡献尚未被直接证明；
- `r_min=0.01` 使 `r_ctrl` 贴底比例偏高，压缩了连续调制可观测空间。

## 3.3 4.24–4.28：fixed-c continuous-r 因果对照（关键补链）

这段是 4.30 的直接前提，建议固定写入：

1. **B-current 是主方法候选**
   - fixed-c + continuous-r 在多 seed 对照中整体最强。
2. **C310 排除 mean-alpha confound**
   - constant LR 对齐 B 的 mean alpha 后仍弱于 B，说明 B 不是“仅靠平均 LR 更高”。
3. **S（shuffled alpha）排除随机抖动 confound**
   - 保留 alpha 分布但打乱时序后退化，说明“alpha 与状态对齐”有贡献。
4. **D3（stage-matched decay）解释大头但不解释全部**
   - D3 接近但低于 B，说明 coarse stage trajectory 很重要，但 stage 内 state-dependent modulation 仍有增量价值。

> 结论：4.30 研究“如何更稳地提取低频状态信号”是顺理成章，而不是跳步。

---

## 4. 4.30 结果模板（增强版）

## 4.1 性能指标（强调 late stability）

每个 benchmark 固定输出：

- step0
- best
- best_step
- final
- final-best（drop）
- final-step0（net gain）

core 指标固定输出：

- final core avg
- best core avg
- best_step
- core drop = final-best

## 4.2 alpha 轨迹（证明“低频化发生”）

- alpha mean / std / p95-p5
- 分阶段均值：
  - early: step 21–100
  - mid: step 101–200
  - late: step 201–300
- 高频变化：
  - mean |Δα_t|
  - p95 |Δα_t|
  - max |Δα_t|

推荐表：

| method | alpha mean | alpha std | alpha early | alpha mid | alpha late | mean |Δα| | p95 |Δα| |
|---|---:|---:|---:|---:|---:|---:|---:|

## 4.3 r 信号指标（看 window 在聚合什么）

基础：

- `r_hat_raw` mean
- `r_hat_raw` positive mean
- `r_hat_raw` negative ratio
- `g_dot_positive`
- valid ratio / invalid ratio / valid count

window 专属（W5/W10 必须）：

- `r_window` mean/std/IQR
- `r_window` early/mid/late mean
- window effective size（平均 valid 样本数）
- empty-window fallback ratio
- window update frequency

建议新增日志（若代码尚未打点）：

- `actor/r_window`
- `actor/r_window_size`
- `actor/r_window_empty`
- `actor/r_window_valid_ratio`

## 4.4 floor/clip 相关

- `r_ctrl at r_min` ratio（overall + early/mid/late）
- alpha at floor ratio（若有 alpha floor）
- upper-clip ratio（若有）

> 关键点：W10 若“稳定但多数时间贴底”，可能退化为保守 floor schedule，而非有效低频控制。

## 4.5 rate-limit 专属（alpharlim 必须）

- rate-limit triggered ratio
- upward-clipped ratio
- downward-clipped ratio
- mean raw alpha vs mean limited alpha
- mean absolute clipping amount

建议日志：`alpha_raw`, `alpha_limited`。

## 4.6 稳定性 tail 指标

- KL: mean / p95 / max / late mean / late p95
- grad norm: mean / p95 / max / late mean / late p95
- PPO ratio/clip: ratio mean/std, clipfrac, approx_kl（若有）

## 4.7 状态变量

- entropy: early/mid/late/final + drop rate
- response_len: early/mid/late/final + std

---

## 5. 4.30 固定对照口径

每个新配置至少与下列方法同表：

- B-current
- D3
- C310
- S（shuffled）
- M（matched constant-alpha）

主表建议：

| method | final core | best core | drop | alpha mean | alpha late | mean |Δα| | KL p95 | entropy final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

---

## 6. 4.30 判读矩阵（细化版）

1. **W5/W10 > B-current**
   - 支持 low-frequency controller；
   - 若 W5>W10：中等窗口最好；若 W10>W5：更低频趋势更重要。

2. **W5 ≈ B，W10 < B**
   - 当前 B 带宽接近最优；过平滑导致欠响应。

3. **W5/W10 ≈ D3**
   - 控制器退化为 coarse schedule，stage 内信号利用不足。

4. **W5/W10 < C310 或 S**
   - window 机制可能失败：检查 valid 稀疏、empty fallback、贴底比例、alpha late 过低。

5. **rate-limit 稳定性提升但 final 下降**
   - 说明抑制高频抖动有效，但阈值过紧；优先放宽 limit，不要直接否定 rate-limit。

---

## 7. 决策标准（避免“看起来都差不多”）

- 若 `final core > B-current` 且 `drop 更小`：进入多 seed 扩展。
- 若 `final core ≈ B-current` 且 `|Δα|`、方差显著更小：可作为默认稳健版本候选。
- 若 `W5/W10 < B-current` 但 `> D3`：低频化有价值但过平滑，回调窗口/EMA。
- 若 `W5/W10 ≤ D3`：当前 window signal 未超越 coarse schedule，不建议主推。

---

## 8. 总结

截至 2026-04-30，最稳妥的叙事是：

1. 4.14 说明固定 bonus 与纯 open-loop 的边界；
2. 4.23 说明 slow scale 是一阶因素，r 连续值贡献未证实；
3. 4.24–4.28（B/C310/S/D3）提供关键因果补链，证明 B 的收益不等同于 mean scale 或随机抖动；
4. 因此 4.30 聚焦 low-frequency controller 是正确下一步；
5. 下一阶段重点不是“再看一个 final 分数”，而是用增强指标体系确认：
   - 是否真的降了控制带宽；
   - 是否减少 late degradation；
   - 是否保留了 stage 内 state-dependent 增量价值。

