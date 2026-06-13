# Project Status — 2026-06-03 (Revised)

## 今日进展（2026-06-03）

1. **完成综合实验报告修订** — 重新定义研究问题，从 high-LR stability 转向 low-LR gradient composition
2. **确立 5 条认知修正** — 主问题、A²Q 价值定位、prompt-level 问题本质、rollout-level 最干净、entropy 降级为 diagnostic
3. **实验优先级重排** — P1: LR=1e-6 normalized 主实验；P2: 机制日志；P3: prompt informativeness；P4: 暂缓 A4c
4. **实验脚本已就绪** — 15 个 LR=1e-6 wrapper 脚本确认可用，待启动
5. **更新所有文档** — `experiment_report_20260603.md`, `project_status.md`, `memory/MEMORY.md`, `memory/algorithm_design.md`

### 下一步动作

启动实验 A（按优先级分批）：

| 批次 | 内容 | Runs | 状态 |
|---|---|---|---|
| 第一批（核心） | vanilla × 3 + rollout-only-norm × 3 | 6 | **待启动** |
| 第二批（诊断） | rollout-only-unnorm × 3 | 3 | 待启动 |
| 第三批（次要） | hierarchical-norm × 3 + prompt-only-norm × 3 | 6 | 等第一批结果 |

---

## 当前方向

**A²Q-Guided Robust Rollout Aggregation for RLVR — Gradient Composition at Low LR**

核心研究问题（已修正）：

> In a stable low-learning-rate RLVR regime, can we improve final performance by improving the composition of the policy-gradient estimator, rather than by changing the learning-rate schedule?

主公式：
```
ĝ_i = (1/K) Σ_k A_{i,k} s_{i,k}         (standard GRPO)
g̃_i = (1/K) Σ_k (w̃_{i,k}/w̄) A_{i,k} s_{i,k}  (rollout-only normalized)
w̃_{i,k} = min(1, τ_r / (A²_{i,k} Q_{i,k} + ε))
```

关键点：不是防 collapse（低 LR 下不会崩），而是**改善 gradient composition 提升 sample efficiency**。

## 当前认知（2026-06-03 修正后）

1. **3.1e-6 是稳定 baseline，但不是最干净研究点。** 目标是 1e-6 下的 gradient composition 提升。
2. **A²Q hierarchical 在 3.1e-6 有弱正信号（+0.0038），但不能证明方法成立。** Vanilla 对照不完整，code-path 差异未排除。
3. **Rollout-level A²Q correction 是最干净的机制假设。** 直接作用于 per-rollout energy proxy H_{i,k}。
4. **Prompt-level 不能简单按 E_i clip。** 核心问题是 informativeness (all-correct/all-wrong vs mixed)，不是 high-energy outlier。
5. **Entropy 是 behavior diagnostic，不是主指标。** 在低 LR 下不驱动 validation 下降。

## 实验优先级

### Priority 1: LR=1e-6 Normalized 主实验（NEW — 最高优先）

| 方法 | 目的 |
|---|---|
| Vanilla | baseline |
| Rollout-only normalized | **主 claim** — gradient composition 改善 |
| Rollout-only unnormalized | 区分 composition vs effective LR |
| Hierarchical normalized | prompt-level 帮忙还是伤害？ |
| Prompt-only normalized | E_i clip 是否有害？ |

核心对比：**vanilla vs rollout-only normalized**

### Priority 2: 机制日志补充

- top5 share before/after
- n_eff before/after
- ΣH'/ΣH, mean(w̃), max(w̃)
- 验证 normalized 后 w̃>1 的分布转移行为

### Priority 3: Prompt informativeness 分析

- frac_mixed, frac_all_correct, frac_all_wrong
- mixed_H_share, clipped_mixed_frac
- 决定 prompt-level 方法是否应改为 informativeness-based

### Priority 4: 暂缓 A4c

Bug 18 (entropy gate) 记录但不阻塞主线。非当前方向。

## Stage 1 (LR=3.1e-6) 进度

| 方法 | seed0 | seed1 | seed42 |
|---|---|---|---|
| Vanilla | 229/300 | 229/300 | 230/300 |
| Rollout-only | 219/300 | 219/300 | **300/300 ✓** |
| Prompt-only | 218/300 | 233/300 | 未产出 |
| Hierarchical | **300/300 ✓** | **300/300 ✓** | **300/300 ✓** |

Hierarchical 3-seed 完整结果：final mean = 0.3309, drop = -0.0007

## 代码变更（2026-06-03）

| 文件 | 变更 |
|---|---|
| `verl/trainer/ppo/a2q_reweighting.py` | Gini sign fix；`normalize` 参数；energy-weighted 诊断 |
| `verl/trainer/config/algorithm.py` | `a2q_reweight_normalize: bool = True` |
| `verl/trainer/ppo/ray_trainer.py` | 传递 normalize 参数 |
| `new_experiments/a2q_reweighting/sync_a2q_stage1.sh` | 支持 `A2Q_NORMALIZE` 环境变量 |
| `new_experiments/a2q_reweighting/sync_a2q_lr1e-6_*.sh` | LR=1e-6 实验 A 脚本（15 个） |
| `exp_data/experiment_report_20260603.md` | 综合报告（已修订） |

## 之前路线状态

| 路线 | 状态 | 说明 |
|---|---|---|
| Signal-fraction LR (α = c·r̂) | 暂停 | r̂_t 噪声大，c_t 追移动靶 |
| Signal-quality LR (A4 reward gate) | 失败 | self-masking feedback + entropy collapse |
| Entropy-ratio gate (A4c) | 暂缓 | Bug 18 未修；非当前方向 blocker |
| A²Q hierarchical @ 3.1e-6 | 弱正信号 | +0.0038 vs A1，但对照不完整 |
| Higher-LR stability story | 降级 | 主线转向 low-LR composition |

## 关键文件索引

| 文件 | 内容 |
|---|---|
| `exp_data/experiment_report_20260603.md` | 综合实验报告（修订版） |
| `exp_data/a2q_reweighting_stage1_interim_report.md` | Stage 1 中期报告 |
| `exp_data/aqh_closedloop_report.md` | AQH closed-loop 报告 |
| `exp_data/README_A2Q_Hierarchical_Reweighting.md` | 设计文档 |
| `memory/algorithm_design.md` | 算法设计全史 |
| `memory/engineering_impl.md` | 工程实现细节 |
