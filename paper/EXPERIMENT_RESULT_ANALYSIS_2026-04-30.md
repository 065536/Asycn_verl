# DeepSeek-1.5B Signal-Fraction 4.30 总报告（Scale-Confound 修订版）

版本：2026-05-02

## 1) 一句话结论

4.30 阶段的关键不是“证明低频控制已经更好”，而是重新确认：
split-batch alignment 是弱但有信息的低频状态代理。当前结果中，`alpharlim0.05`
最干净地体现了稳定性收益；slow-EMA 结果存在显著 alpha-scale confound，需 scale-matched rerun 后再做因果解释。

## 2) 本次分析对象（4 个 run）

- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.95_alpharlim0.05.jsonl`
- `deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_slowema_ret0.98.jsonl`

核心指标口径：

`core(step)=mean(val-core/{AIME,AIME2025,Idavidrein/gpqa,MINERVA,OLYMPIAD_BENCH}/acc/mean@16)`

## 3) 4.30 组内对比

| Run | core_init | core_best@step | core_final | net_gain | drop(final-best) |
|---|---:|---:|---:|---:|---:|
| slowema_ret0.95 | 0.230333 | 0.348250@250 | **0.340750** | +0.110417 | -0.007500 |
| alpharlim0.05 | 0.224833 | 0.342833@290 | 0.340333 | **+0.115500** | **-0.002500** |
| slowema_ret0.95 + alpharlim0.05 | 0.223500 | 0.336458@270 | 0.332375 | +0.108875 | -0.004083 |
| slowema_ret0.98 | 0.222417 | 0.337833@240 | 0.324042 | +0.101625 | -0.013792 |

## 4) 最关键修正：Scale-Confound Check

历史参考：B-current typical alpha 约 `3.1e-6`（项目既有口径）。

| Run | alpha_mean | alpha_last | 相对 B-current alpha_mean | 解释 |
|---|---:|---:|---:|---|
| slowema_ret0.95 | 6.17e-6 | 4.87e-6 | ~2.0x | 严重 scale confound |
| alpharlim0.05 | 3.42e-6 | 2.50e-6 | ~1.1x | 轻度偏高，可接受但需标注 |
| slowema_ret0.95 + alpharlim0.05 | 6.29e-6 | 2.64e-6 | ~2.0x | 严重 scale confound |
| slowema_ret0.98 | 5.45e-6 | 4.52e-6 | ~1.8x | 严重 scale confound |

因此：

- slowema_ret0.95 虽然 final 在 4.30 四组中最高，但不能直接归因为“slow EMA 更好”。
- alpharlim0.05 是目前最干净的稳定性信号。

## 5) 关键判断修订

1. **alpharlim0.05 是当前最可信的 stability-improvement candidate。**
   - near-best final（0.340333）
   - 最小 late regression（-0.0025）
   - `mean|Δalpha|` 显著更低（1.55e-7）
   - rate limit 触发率 57%，说明控制带宽确实被改变

2. **slowema_ret0.95 的收益目前仍被 scale confound 覆盖。**
   - 其 mean alpha 接近 B-current 的 2 倍，不能直接得出“低频平滑带来收益”的因果结论。

3. **combined 方案不应描述为“更保守”。**
   - `alpha_mean` 反而最高（6.29e-6）。
   - 更合理解释：slow EMA 与 rate limit 交互导致 alpha 轨迹失真（早期偏高，后期修正受限）。

4. **ret0.98 的问题更像“响应太慢”，不是简单“更保守”。**
   - `alpha_last` 仍偏高（4.52e-6），且 late regression 更大。
   - 符合“过慢响应错过 late-stage 风险信号”的机制预期。

## 6) 当前报告缺失的基线上下文

仅看 4.30 四组不足以回答“控制器是否真正改进”，还必须并列历史基线：

- B-current（主方法）
- D3（coarse schedule）
- C310（mean-alpha matched constant）
- S shuffled（alpha distribution shuffled）
- M 2.97e-6（fixed LR）

没有这些对照时，只能做组内比较，不能做完整机制归因。

## 7) 下一步优先级（修订）

1. **第一优先级：** `alpharlim0.05` 多 seed。
2. **第二优先级：** `slowema_ret0.95` 先做 scale-matched rerun（调低 c_fixed，使 mean alpha 回到约 `3.1e-6` 到 `3.4e-6`），再考虑多 seed。
3. **第三优先级：** 继续 W5/W10（windowed-r）；当前报告尚未回答 temporal aggregation 是否有效。

## 8) 可直接引用的结论文本

**English**  
`alpharlim0.05` gives the cleanest stability signal in 4.30: near-best final core with the smallest late regression and substantially reduced alpha high-frequency variation.  
`slowema_ret0.95` reaches the highest final core among the four variants, but its mean alpha is substantially larger than B-current / alpharlim, so the gain cannot yet be attributed solely to low-frequency smoothing.  

**中文**  
4.30 阶段最干净的稳定性正信号来自 `alpharlim0.05`：在保持较高 final core 的同时，late regression 最小且 alpha 高频抖动显著下降。  
`slowema_ret0.95` 虽然在四组内 final 最高，但其 mean alpha 显著偏大，当前仍存在 scale confound，不能直接归因为低频控制收益。

## 9) 与历史基线（4.27/4.28）的统一解释

为了避免把 4.30 的局部现象误读成“方法最终结论”，应将其放入已完成的 C310 / S / D3 / B 证据链中：

### 9.1 历史基线定量锚点（来自 4.27/4.28 已完成分析）

> 注：下表用于“历史锚点”而非同批次严格 apples-to-apples 对照；4.30 主体仍是组内 4-run 比较。

| Family / Baseline | 典型设置 | core_final（5-task avg） | 备注 |
|---|---|---:|---|
| B-current | `sigfrac_cfixed_lr1.25e-5` | **0.3443** | 4.26/4.28 主方法参考 |
| D3 | stage-matched piecewise decay | 0.3360 | 强 open-loop schedule baseline |
| C310 | constant `3.10e-6` | 0.3322 | mean-alpha matched constant |
| S | shuffled alpha replay | 0.3324 | 保留 alpha 分布、打乱时序耦合 |
| M | matched alpha `2.97e-6` | 0.3302 | fixed-LR 对照 |

结合 4.30 的 `alpharlim0.05 core_final=0.3403` 与 `slowema_ret0.95 core_final=0.3408`，当前最稳妥解读是：

- 4.30 的两个较优点位处在 **D3 之上、B-current 附近**；
- 但 slowema 点位仍受 scale confound，不可直接当作“机制优于 B”的证据；
- `alpharlim0.05` 因 scale 更接近 B-current，当前可作为更干净的稳定性候选。

1. **C310（matched mean alpha）与 S（shuffled alpha）已排除两个常见混杂：**
   - B 的收益不是来自更高 mean alpha；
   - B 的收益不是来自无序 jitter。

2. **D3（stage-matched open-loop decay）说明 coarse schedule 可以恢复大部分收益。**
   - 这支持“support-matched update scale”是主轴；
   - 也说明不应把全部收益归因为 step-level 精细控制。

3. **B 仍高于 D3，保留 residual state-dependent value 的证据。**
   - 但该 residual 的主要载体更可能是低频状态分配，而不是单步 sign 决策。

4. **4.30 的价值定位：**
   - 不是重做 B vs D3 结论；
   - 而是在 B-family 内继续回答“低频化后，controller 的稳定性与可解释性是否提升”。

因此，4.30 与 4.27/4.28 不冲突，而是对同一主线的下一层验证：  
`mean-scale` confound 与 `schedule` confound 已基本清理后，继续清理 `alpha bandwidth` confound。

## 10) Claim Boundary（建议写入论文/汇报）

### 当前可以声明（supported）

- split alignment 在单步层面是弱信号，但在低频聚合后可作为有信息的状态代理。
- `alpharlim0.05` 在 4.30 组内给出最干净的稳定性改进信号（最小 late regression + 较高 final）。
- `slowema_ret0.95` 当前存在明显 scale confound，不能单独支持“低频平滑本身带来收益”的因果结论。
- 现阶段更稳妥叙事应是：  
  **signal-fraction 主要在发现/跟踪 support-matched update-scale；低频控制用于提高该过程的稳健性。**

### 当前不能声明（unsupported）

- “slow EMA 优于 alpharlim0.05（因果上）”；
- “combined 方案更保守因此更稳定”；
- “ret0.98 只是更保守所以差”；
- “4.30 已经证明 temporal aggregation 显著优于 B-current”。

## 11) 下一轮实验的判定门槛（避免重复无效算力）

### A. `alpharlim0.05` 多 seed（Priority 1）

最低门槛（建议）：

- 3 seed 下 `core_final` 平均不低于 B-current；
- `final-best` 持续优于 B-current（late regression 更小）；
- `mean|Δalpha|` 稳定低于 B-current；
- 不出现显著 entropy/kl 异常漂移。

若满足，`alpharlim` 可进入“默认低频稳态控制候选”。

### B. `slowema_ret0.95` scale-matched rerun（Priority 2）

先把 mean alpha 调回 B-current 区间（约 `3.1e-6 ~ 3.4e-6`）再比较。  
判定重点：

- 若 scale-matched 后仍优于 B/alpharlim，才可支持“低频平滑有独立增益”；
- 若优势消失，当前 slow EMA 收益主要来自 scale 而非 smoothing 机制。

### C. W5/W10 windowed-r（Priority 3）

解释矩阵沿用 `PROJECT_STATUS.md`：

- W5/W10 > B：当前 B 对单步噪声反应过强，temporal aggregation 有效；
- W5 ~= B, W10 < B：中等低通有效，过强低通损失适应性；
- W5/W10 < B：现有 B 平滑已足够或 window 过阻尼；
- W10 ~= D3：长窗近似 coarse schedule，state-dependent residual 变弱。

## 12) 分析流程约束（方法学）

结合项目历史 bug 与协作规范，后续报告建议固定如下顺序：

1. **先机制再分数**：先看 `alpha_t/r_hat/r_ctrl/g_dot_positive`，再看 val-core。
2. **先核对执行再解释机制**：特别是 replay/window/limiter 配置后，先确认日志行为符合预期。
3. **一次只比较一个自由度**：避免把 scale、bandwidth、gate 变化混在一个结论里。
4. **区分“更稳定”与“更保守”**：如果 mean alpha 显著变小，应优先视为保守化，而非直接归因于控制质量提升。

## 13) 面向后续写作的标准英文段落（可复用）

`The 4.30 batch should be interpreted as a bandwidth-control ablation within the signal-fraction family, not as a standalone proof of causal superiority for any single variant.`

`Among the four runs, alpharlim0.05 provides the cleanest stability evidence (near-best final core with the smallest late regression) under a relatively matched alpha scale, while slowema_ret0.95 remains scale-confounded and requires a scale-matched rerun before causal attribution.`

`Together with prior C310/S/D3 controls, the current evidence supports a conservative claim: signal-fraction's primary benefit is support-matched update-scale tracking, and low-frequency control is a mechanism to improve robustness of that allocation under noisy split alignment.`