# DeepSeek-1.5B Signal-Fraction 实验结果分析（截至 2026-04-30）

## 0. 执行摘要（先给结论）

基于 4.14、4.23、4.24–4.28 与 4.30 的连续证据，当前最稳妥结论是：

1. **单步 split alignment 是弱信号（`g_dot_positive` 约 55%）**，不适合做硬门控，但可能作为低频状态代理。  
2. **B-current（fixed-c + continuous-r）收益真实存在**，且不能被「均值 LR 更高」或「随机抖动」完全解释。  
3. 4.30 的主问题不是再证“r 有没有信息”，而是：**如何在弱信号下提取低频成分，降低后期退化（late degradation）**。  
4. 评估重点必须从“单个 final 分数”升级为“**性能 + 控制轨迹 + 稳定性 tail**”三位一体；否则无法区分「有效自适应」与「退化成保守 open-loop」。

---

## 1. 分析目标与证据来源

### 1.1 本文档回答的问题

- Q1：B-current 的收益是否只是 scale 差异？
- Q2：4.30 的 low-frequency controller（slow EMA / window-r / alpha-rate-limit）是否真正改善“控制质量”？
- Q3：若某方法 final 接近，如何判断是否值得进入多 seed 扩展？

### 1.2 证据来源（按时间）

- `exp_data/4.14/ANALYSIS.md`：open-loop 干预边界（β sweep、cosine/floor、advvar）。
- `exp_data/4.23/ANALYSIS.md`：c-fixed / continuous-r 第一轮结论与 confound 提示。
- `memory/engineering_impl.md`：4.24–4.28 B/C310/S/D3 对照逻辑、4.30 工程落地细节。
- `paper/PROJECT_STATUS.md`：4.30 checkpoint 的统一问题定义与当前推荐实验路径。

---

## 2. 历史结论重建：为什么 4.30 是“自然下一步”

## 2.1 4.14：open-loop 有上限，不是“全无效”

4.14 的精确定性应为：

- 固定 entropy bonus 不稳健（β 小无效、β 大爆炸）。
- `cosine_floor` 是当时最强 open-loop，但仍弱于 `sync_lr3e-6`。
- `advvar_ent` 调整了探索分配，但未解决后期 mismatch。

**因此结论不是 open-loop 完全无效，而是其上限受限**：它可缓解部分阶段问题，但无法可靠处理阶段内状态变化。

## 2.2 4.23：先看到 scale 主导，再识别 confound

4.23 清晰显示：

- effective alpha 的慢尺度（slow scale）是一阶因素；
- continuous-r 的独立贡献仍未被直接识别；
- `r_min` 偏高导致 `r_ctrl` 贴底频繁，压缩可观测的连续调制空间。

这一步的价值在于：把“方法看起来有效”拆成了“scale 贡献 vs 状态调制贡献”。

## 2.3 4.24–4.28：因果补链（B/C310/S/D3）

这组对照是 4.30 的逻辑根基：

- **B-current（fixed-c + continuous-r）**：当前主候选。  
- **C310（matched constant LR）**：对齐 mean alpha 后仍弱于 B，排除纯 mean-scale 解释。  
- **S（shuffled alpha）**：保留分布但打乱时序后退化，证明“alpha 与状态时序对齐”有信息。  
- **D3（stage-matched decay）**：接近 B 但通常略低，说明 coarse stage 解释了大头，但 stage 内 state-dependent modulation 仍有增量价值。

**结论**：4.30 研究 low-frequency controller 不是跳步，而是沿已验证因果链继续前进。

---

## 3. 4.30 的正确问题定义（必须统一口径）

4.30 不应再表述为“单步 sign 分类准确率提升多少”，而应统一为：

> 在单步弱信号条件下，是否能通过低频聚合构造更稳健的 alpha 轨迹，
> 从而减少 late degradation，且不牺牲最终性能。

当前候选变体：

- **Slow EMA**：`ret0.95`, `ret0.98`
- **Alpha rate limit**：`alpharlim0.05`
- **Windowed-r**：`W5`, `W10`

这三类分别作用于：

- 信号平滑（降低瞬时噪声）
- 执行器约束（抑制过快控制变化）
- 观测聚合（提高低频成分占比）

---

## 4. 指标体系（详细版）：必须记录什么，为什么记录

## 4.1 性能层：判断“好不好”

每个 benchmark 统一输出：

- `step0`, `best`, `best_step`, `final`
- `drop = final - best`
- `net_gain = final - step0`

core 聚合（建议 AIME/OLYMPIAD/GPQA/MINERVA 统一口径）：

- `core_final_avg`, `core_best_avg`, `core_best_step`, `core_drop`

**解释重点**：`core_drop` 比 `best` 更能反映后期控制质量。

## 4.2 控制层：判断“怎么做到的”

alpha 轨迹：

- `alpha_mean`, `alpha_std`, `alpha_p95_p5`
- 分段均值：`alpha_early(21–100)`, `alpha_mid(101–200)`, `alpha_late(201–300)`
- 高频变化：`mean|Δalpha|`, `p95|Δalpha|`, `max|Δalpha|`

r 信号：

- `r_hat_raw_mean`, `r_hat_raw_pos_mean`, `r_hat_raw_neg_ratio`
- `g_dot_positive`
- `valid_ratio`, `invalid_ratio`, `valid_count`

Window 专属（W5/W10）：

- `r_window_mean/std/IQR`
- `r_window_early/mid/late`
- `r_window_effective_size`
- `r_window_empty_ratio`
- `r_window_update_freq`

Rate-limit 专属：

- `rate_limit_trigger_ratio`
- `upward_clip_ratio`, `downward_clip_ratio`
- `alpha_raw_mean`, `alpha_limited_mean`
- `mean_abs_clip`

## 4.3 稳定性层：判断“代价与风险”

- KL：`mean/p95/max/late_mean/late_p95`
- grad norm：`mean/p95/max/late_mean/late_p95`
- PPO 相关：`ratio_mean/std`, `clipfrac`, `approx_kl`
- 状态变量：`entropy early/mid/late/final`, `response_len early/mid/late/final/std`

**关键判断**：如果 final 接近但某方法 KL/grad tail 明显更低，优先该方法做默认候选。

---

## 5. 对照设计与主表模板（可直接复用）

## 5.1 最小对照集合（每个新方法必须同表）

- B-current
- D3
- C310
- S（shuffled）
- M（matched mean-alpha constant）

## 5.2 主表（建议）

| method | core_final | core_best | core_drop | alpha_mean | alpha_late | mean\|Δalpha\| | KL_p95_late | entropy_final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## 5.3 补充表（window/rate-limit）

| method | g_dot_pos | r_valid_ratio | r_floor_ratio | r_window_empty | limiter_trigger | mean_abs_clip |
|---|---:|---:|---:|---:|---:|---:|

---

## 6. 判读矩阵（实验结果出来后按此落结论）

1. **W5/W10 > B-current 且 drop 更小**  
   → 低频聚合有效，进入多 seed。  

2. **W5 ≈ B，W10 < B**  
   → 适度平滑有益，过平滑欠响应；优先 W5。  

3. **W5/W10 ≈ D3**  
   → 退化为 coarse schedule，stage 内信息利用不足。  

4. **W5/W10 < C310 或 S**  
   → 控制失败，优先排查 `valid稀疏 / empty fallback / floor贴底 / late alpha过低`。  

5. **alpharlim 稳定性更好但 final 下降**  
   → 方向正确但阈值过紧；先放宽 limit 再否定机制。  

---

## 7. 决策门槛（Go / No-Go）

为避免“看起来差不多”，建议固定门槛：

- **Go-1（强通过）**：`core_final > B` 且 `core_drop` 更小。  
- **Go-2（稳健通过）**：`core_final ≈ B`（差异在噪声内）但 `mean|Δalpha|` 与 KL/grad tail 显著更低。  
- **Conditional**：`B > new > D3`，说明有信号但过平滑，进入小网格调参。  
- **No-Go**：`new ≤ D3` 或显著低于 C310/S，暂停主线推进。

---

## 8. 下一轮执行清单（可直接给实验同学）

1. 跑完：`W5`, `W10`, `ret0.95`, `ret0.98`, `alpharlim0.05`（必要时加 `ret0.95+alpharlim0.05`）。
2. 每个 run 导出统一 CSV：性能层 / 控制层 / 稳定性层全字段。
3. 先做单 seed 判读矩阵，再决定多 seed 扩展（建议优先 top-2）。
4. 若 window 方法退化，先调 `W` 和 floor，再考虑复杂控制器（不要直接跳到多参数混合策略）。

---

## 9. 总结（一句话版本）

截至 2026-04-30，主结论不是“找到了更准的单步信号”，而是：

> 已确认 B-current 的收益包含真实状态对齐成分；
> 4.30 的关键是把弱单步信号变成可用低频控制，
> 并用“性能 + 轨迹 + tail 稳定性”三层指标证明其增益可复现、可解释、可扩展。