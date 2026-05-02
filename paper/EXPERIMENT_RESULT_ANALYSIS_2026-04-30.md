# DeepSeek-1.5B Signal-Fraction 实验结果分析（截至 2026-04-30）

## 1. 分析目标与范围

本文档基于 `paper/PROJECT_STATUS.md` 与 `exp_data` 历史分析，整理当前 1.5B 主线实验结论，并给出下一步实验建议。

重点回答三个问题：

1. 4.30 低频控制实验（slow EMA / alpha-rate-limit / window-r）的研究目标是什么；
2. 与 4.14、4.23 等历史结果相比，当前证据链是否一致；
3. 下一轮应如何最小化实验成本、最大化信息增益。

---

## 2. 当前主线（4.30 checkpoint）

`PROJECT_STATUS.md` 明确指出，主线从“逐步 sign 判别”转为“低频时域聚合”：

- 单步 `g_A1^T g_A2` 信号偏弱，`g_dot_positive` 约 55%；
- 目标变为让 noisy alignment 成为“状态代理”，用来塑造更稳的 LR 轨迹，而非逐步精确估计最优 LR。

对应新增实验族：

- slow EMA: `...slowema_ret0.95(.sh)` / `...slowema_ret0.98(.sh)`
- alpha-rate-limit: `...alpharlim0.05(.sh)`
- windowed-r: `...windowr_w5(.sh)` / `...windowr_w10(.sh)`

并明确推荐优先跑 W5/W10 用于验证“低频聚合是否提升控制器有效性”。

---

## 3. 历史结果回顾（用于对照）

## 3.1 4.14（Phase1 Batch2 + advvar）

关键结论：

- 固定熵奖励 β 不存在稳定甜点：
  - β=0.1 灾难性崩溃（AIME@300=0）
  - β=0.001 不足以防后期退化
- `cosine_floor` 是 open-loop 最佳，但仍明显低于 `sync_lr3e-6`
- `advvar_ent` 早期提升，后期仍出现与 `lr1e-5` 类似的慢性退化

代表数据（AIME@300）：

- `sync_lr3e-6`: 0.325（历史最优）
- `cosine_floor`: 0.273
- `sync_lr1e-5`: 0.183
- `advvar_ent`: 0.158
- `ent0.1`: 0.000

这批最重要洞察：

- 多组实验终态 entropy 接近，但准确率差距很大；
- 问题核心在 update/support 的“路径匹配”，而非单纯终态 entropy 高低。

## 3.2 4.23（Phase1 cfixed）

目标：在 `eta_c=0` 下做 c_fixed 对照，剥离 c_t 学习因素。

关键结论：

- 最优 c_fixed 对应 `lr1.25e-5` 组（等价长期 alpha 落在 ~2.5e-6~3e-6 区间）；
- 当前收益主要像 effective scale 差异，尚不能证明 r 连续值是主导贡献；
- `r_min=0.01` 导致 `r_ctrl` 贴底比例高（例如 lr1.25e-5 约 53%），连续调制空间被压缩。

代表数据：

- post-handoff mean alpha:
  - lr1.25e-5: 2.97e-6
  - lr1e-5: 2.66e-6
  - lr7.5e-6: 2.05e-6
- g_dot>0 比例约 55–66%，有方向信息但 SNR 低

---

## 4. 4.30 实验与历史证据的一致性判断

根据 `PROJECT_STATUS.md` 的 4.30 目标定义，当前研究问题是：

> 在弱单步信号条件下，低频控制是否能在不引入强阶段先验的前提下，提升稳定性并保持/提升最终精度。

这与 4.14/4.23 的证据完全一致：

1. 4.14 已证明“固定 open-loop 熵/调度”不够稳健；
2. 4.23 已证明“仅靠 scale 命中”不足以解释全部价值，需进一步分离低频控制的独立增益；
3. 因此 4.30 选择 slow EMA / rate-limit / window-r 是逻辑上最小、最干净的下一步。

---

## 5. 4.30 结果判读模板（用于落地）

对每个 4.30 配置（slowema95、slowema98、alpharlim、slowema95+alpharlim、W5、W10），建议统一输出以下关键数据：

1. `val-core`：final、best@step、final-best
2. `actor/alpha_t`：mean、std、p95-p5
3. `actor/r_hat`：mean、IQR
4. `actor/r_ctrl`：贴 `r_min` 比例
5. `actor/alpha_rate_limited`：触发比例（若该指标存在）

按 `PROJECT_STATUS.md` 判读矩阵解释：

- W5/W10 > B-current：说明低频聚合确实抑制了单步噪声过响应
- W5 ≈ B、W10 < B：中等带宽最优，过平滑欠响应
- W10 ≈ D3：窗口过长，退化为粗阶段调度

---

## 6. 下一步实验建议（最小成本）

### 建议 A（优先）

先完成 W5/W10 与 B-current 的同口径对比（同 seed、同 300 step），用上节 5 个指标判断是否“稳中有升”。

### 建议 B（若 W5 有收益）

在 W5 上叠加 `alpharlim0.05` 做二阶稳定化验证：

- 若性能持平且方差下降，则作为默认控制器；
- 若性能下降，说明限速过强，缩小 rate-limit。

### 建议 C（若 W10 接近 D3）

停止继续增大窗口，回到 W5/slowema 方向；避免控制器退化成阶段性 schedule。

---

## 7. 结论

截至 2026-04-30 的证据链支持以下结论：

1. 主问题不是“有没有信号”，而是“弱信号在高频控制下会被噪声放大”；
2. `lr1.25e-5 cfixed` 给出的有效 alpha 区间是当前可行锚点；
3. 4.30 低频控制实验是必要且正确的下一步；
4. 需要用 W5/W10 与 B-current 的同口径关键指标对比，来确认低频化是否带来真实增益，而不只是更保守。

