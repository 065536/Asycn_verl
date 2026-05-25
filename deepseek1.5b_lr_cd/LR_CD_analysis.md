# LR Coordinate Descent 调度分析（AIME24）

## 1) 你看到“decay 不明显”的核心原因

### A. 评测点太稀疏：每个 run 只有 step=300 一个验证点
从 jsonl 可见，AIME 指标只在 step=300 出现（`test_freq=300`），因此每条曲线在 AIME 上只有一个点，很难判断“先大后小”调度在中前期是否有收益，统计噪声会淹没 scheduler 差异。

### B. 线性 vs 余弦在你的区间内差异本来就小
当前配置是 `1e-5 -> 3.1e-6`（`min_lr_ratio=0.31`）或 `5e-6 -> 3.1e-6`（`min_lr_ratio=0.62`）。这两类都在 300 step 内收敛到大约 `3.1e-6`，终点接近，且总 step 不长时，linear/cosine 形状差异只体现在中间少量 step。

### C. seed 方差大，足以覆盖 scheduler 差异
以 `linear_1e-5_to_3.1e-6` 为例，两条结果分别是 29.17% 和 9.17%，波动极大；而 `cosine_1e-5_to_3.1e-6` 两条在 22.92% / 24.38%。在这种 seed 方差下，很难得出“linear/cosine 谁显著更好”的结论。

## 2) 现有结果汇总（final@300）

- linear 1e-5 -> 3.1e-6: `[0.2917, 0.0917]`，均值 **0.1917**，std **0.1000**。
- cosine 1e-5 -> 3.1e-6: `[0.2292, 0.2438]`，均值 **0.2365**，std **0.0073**。
- cosine 5e-6 -> 3.1e-6: `[0.3292, 0.2917, 0.3063]`，均值 **0.3090**，std **0.0154**。

这说明“起始 LR 更低（5e-6）+ 到 3.1e-6 的小幅衰减”目前比“1e-5 大幅衰减”更稳、更好；但由于样本数仍少，建议继续加 seed。

## 3) 为什么你之前会看到 36%，现在只有 31% 左右

最常见解释不是 scheduler 本身，而是“实验统计口径差异 + seed + 长度惩罚耦合”：

1. **best-of-k / maj@k 口径是否一致**：AIME 在日志里有 `mean@16`、`best@16`、`maj@16`，不同口径差异很大。
2. **长度相关奖励变化会强烈影响结果**：你的日志里有 `overlong` 系列指标，长度裁剪/惩罚与 LR 同时变化时会造成“归因错位”。
3. **评测频率低造成“幸运点”现象**：若只看 step=300，很容易把偶然高点当成 scheduler 提升。

## 4) 建议的下一轮实验（能回答“scheduler 是否有效”）

1. **提高评测频率**：把 `test_freq` 从 300 改到 50 或 100，至少能看到曲线形状，而不是单点。
2. **固定其它变量，仅比较 scheduler**：同一组 seed（建议 ≥5）跑 linear/cosine/constant。
3. **把“起始 LR”与“调度形状”拆开做二因素实验**：
   - 起始 LR: {1e-5, 7e-6, 5e-6}
   - scheduler: {constant, linear->3.1e-6, cosine->3.1e-6}
4. **增加 warmup（例如 3%~5%）**：当前是 `lr_warmup_steps_ratio=0.0`，在 RL 场景可能导致前期 update 过激，掩盖后期 decay 作用。
5. **统一主指标**：建议以 `val-core/AIME/acc/maj@16` 作为主报表，同时附 `mean@16` 和 `best@16`，避免口径漂移。

## 5) 一个直接可执行的判断标准

若你最终关心的是 AIME24 主线上线效果，可以用：
- 同 seed 集合下，比较 `maj@16` 的均值差；
- 用 bootstrap 置信区间或至少报告均值±标准差；
- 只有当“均值提升 > 1.5~2.0 个百分点且跨 seed 稳定”时，才认为 scheduler 真正有效。


## 6) 回答你的追问：之前 batch split 实验里“长度惩罚显著”吗？

结论：**在你当前这批可见日志里，不能下“显著”结论，更像是有影响但不是主导因素。**

- 在 `deepseek1.5b_lr_cd` 里，`val-aux/AIME/overlong/mean@16` 与 `val-core/AIME/acc/mean@16` 没有稳定单调关系：
  - 例如 `cosine_5e-6_to_3.1e-6_seed42`，`acc=0.3292` 同时 `overlong=0.4771`（并不低）；
  - 但 `linear_1e-5_to_3.1e-6_seed1`，`acc=0.2917` 时 `overlong=0.2167`（明显更低）。
  这说明“更少 overlong”不必然带来更高 AIME。
- 在 `deepseek1.5b_clip_sweep` 这批 jsonl 的最终记录中，很多 run 没有写出 `val-aux/AIME/overlong/*` 字段，因此无法与 lr_cd 做严格横向比较。
- 更稳妥的判断方式是：同一数据/seed 下固定 LR 策略，只扫描长度惩罚系数（或 overlong reward 设计），再看 `maj@16` 的均值差与置信区间。

所以你说“之前 batch split 有长度惩罚显著”这件事，**目前证据不足以确认为显著主因**；更像是它与 seed、起始 LR、采样分布共同耦合。
