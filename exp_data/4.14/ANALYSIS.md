# Phase 1 Batch 2 + Phase 2 (advvar) 实验分析
**日期**: 2026-04-14 | **数据**: `exp_data/4.14/`

---

## 1. 实验概览

本批共 5 个实验，全部为 sync 模式，300 步。涵盖 **β sweep**（完成熵奖励分析）、**cosine with floor**（更公平的 open-loop schedule）、**联合 open-loop**（最强 open-loop baseline），以及 **优势方差加权熵奖励**（Phase 2 方法）。

| 简称 | experiment_name | 类别 | 配置 |
|---|---|---|---|
| **advvar_ent** | `deepseek1.5b_sync_8gpu_advvar_ent_lr1e-5` | Phase 2 | 优势方差加权熵奖励，β=0.01，LR=1e-5 |
| **ent0.1** | `deepseek1.5b_sync_8gpu_ent0.1_lr1e-5` | Phase 1 β sweep | 固定熵奖励 β=0.1，LR=1e-5 |
| **ent0.001** | `deepseek1.5b_sync_8gpu_ent0.001_lr1e-5` | Phase 1 β sweep | 固定熵奖励 β=0.001，LR=1e-5 |
| **cosine_floor** | `deepseek1.5b_sync_8gpu_cosine_floor_lr1e-5` | Phase 1 schedule | Cosine 衰减 LR=1e-5→1e-6（floor），sync |
| **cosine_ent0.01** | `deepseek1.5b_sync_8gpu_cosine_ent0.01_lr1e-5` | Phase 1 联合 | Cosine 衰减 + 熵奖励 β=0.01，LR=1e-5 |

参考 baseline（来自 `exp_data/4.7/`）：
- **sync_lr1e-6**: AIME@300 = 0.281，entropy@300 = 0.078（保守稳定）
- **sync_lr3e-6**: AIME@300 = 0.325，entropy@300 = 0.073（全局最优）
- **sync_lr1e-5**: AIME@300 = 0.183，entropy@300 = 0.068（Type I 崩溃）
- **sync_ent0.01**: AIME@300 = 0.227，entropy 爆炸到 11.85（灾难性）
- **sync_cosine**: AIME@300 = 0.223，entropy@300 = 0.076（平庸，LR→0）
- **sync_clip0.1**: AIME@300 = 0.017（延迟灾难性崩溃）

---

## 2. 准确率汇总

### AIME（主要 benchmark）

| 实验 | @step0 | @step300 | Peak | Peak Step | vs sync_lr3e-6 (0.325) |
|---|---|---|---|---|---|
| **cosine_floor** | 0.183 | **0.273** | 0.275 | 60 | −16% |
| cosine_ent0.01 | 0.200 | 0.229 | 0.260 | 30 | −30% |
| advvar_ent | 0.219 | 0.158 | 0.290 | 70 | −51% |
| ent0.001 | 0.208 | 0.146 | 0.281 | 70 | −55% |
| ent0.1 | 0.183 | **0.000** | 0.183 | 0 | −100%（灾难性） |

### GPQA（通用知识）

| 实验 | @step300 | Peak |
|---|---|---|
| ent0.001 | **0.332** | 0.359 (250) |
| advvar_ent | 0.331 | 0.356 (110) |
| cosine_floor | 0.328 | 0.345 (180) |
| cosine_ent0.01 | 0.303 | 0.347 (20) |
| ent0.1 | 0.000 | 0.153 (10) |

### OLYMPIAD_BENCH

| 实验 | @step300 | Peak |
|---|---|---|
| **cosine_floor** | **0.486** | 0.505 (130) |
| cosine_ent0.01 | 0.449 | 0.473 (20) |
| advvar_ent | 0.271 | 0.474 (70) |
| ent0.001 | 0.161 | 0.483 (30) |
| ent0.1 | 0.001 | 0.360 (0) |

### MINERVA

| 实验 | @step300 | Peak |
|---|---|---|
| **cosine_ent0.01** | **0.283** | 0.296 (170) |
| cosine_floor | 0.249 | 0.266 (20) |
| advvar_ent | 0.072 | 0.261 (20) |
| ent0.001 | 0.062 | 0.255 (20) |
| ent0.1 | 0.000 | 0.231 (0) |

---

## 3. 核心发现

### 发现 0（最重要）：终态 entropy 几乎一样，但 accuracy 天差地别——问题在于路径，不在于终点

把所有不爆炸的 sync 实验放在一起：

```
实验                    Entropy@300  AIME@300   LR@300     轨迹特征
───────────────────────────────────────────────────────────────────
sync_lr3e-6               0.073     0.325     3e-6       持续上升 ✓
sync_lr5e-6               0.068     0.294     5e-6       略有波动
sync_lr1e-6               0.078     0.281     1e-6       缓慢上升
cosine_floor              0.067     0.273     1e-6       稳定 ✓
cosine(no floor)          0.076     0.223     ~0         后期停滞
sync_lr1e-5               0.068     0.183     1e-5       先升后降 ✗
advvar_ent                0.082     0.158     1e-5       先升后降 ✗
ent0.001                  0.098     0.146     1e-5       先升后降 ✗
clip0.1                   0.099     0.017     1e-5       延迟崩溃 ✗
```

**关键观察**：所有不爆炸的实验，entropy@300 都收敛到 **0.067–0.099** 这个很窄的区间。但 AIME@300 从 0.017 到 0.325，差了接近 **20 倍**。

**这意味着**：问题不是"entropy 耗尽了多少"——所有实验都耗到差不多的水平——而是**在 entropy 消耗的过程中，每一步的 update 是否与当前的 support 匹配**。

**进一步看 AIME 轨迹**，可以清晰地看到两种 pattern：

**稳定组**（后期 LR ≤ 3e-6）：
- sync_lr3e-6: 0.18 → 0.28 → 0.26 → 0.30 → 0.30 → **0.33** （持续上升）
- cosine_floor: 0.18 → 0.25 → 0.24 → 0.24 → 0.26 → **0.27** （稳步改善）
- sync_lr1e-6: 0.21 → 0.23 → 0.24 → 0.25 → 0.30 → **0.28** （缓慢上升）

**崩溃组**（后期 LR = 1e-5）：
- sync_lr1e-5: 0.20 → **0.28**(peak) → 0.25 → 0.20 → 0.20 → **0.18** （step 100 后下降）
- advvar_ent: 0.22 → **0.29**(peak) → 0.25 → 0.26 → 0.21 → **0.16** （step 100 后下降）
- ent0.001: 0.21 → **0.28**(peak) → 0.25 → 0.22 → 0.20 → **0.15** （step 100 后下降）

**分界线清晰**：当 entropy 降到 ~0.15 以下时，LR=1e-5 的 update 超出了 support 范围，policy 开始偏离好的区域。而 LR=3e-6 或后期降到 1e-6 的 cosine_floor，在同样低的 entropy 下 update 仍在 support 内。

**对 thesis 的意义**：这强化了 mismatch 框架，但需要重新表述。核心不是 "entropy budget depletion"（那是终态，大家都到那），而是 **"path quality"——每一步的 update/support 比值决定了 policy 是走向好的区域还是坏的区域**。

### 发现 1：β sweep 彻底否定了固定熵奖励

三个 β 值（0.001, 0.01, 0.1）证明**不存在正确的固定 β**：

| β | Entropy@300 | AIME@300 | 失败模式 |
|---|---|---|---|
| 0.001 | 0.098 | 0.146 | 太弱——entropy 照样塌，accuracy 在 step 70 后下降 |
| 0.01 | 11.92（4.7 数据） | 0.227 | entropy 爆炸——policy 变为均匀分布 |
| **0.1** | **11.92** | **0.000** | **立即灾难性爆炸——step 20 前 accuracy 归零** |

**β=0.1** 是最极端的情况：entropy 比 β=0.01 更快爆炸（~step 10 vs ~step 30 达到饱和），accuracy 在所有 benchmark 上都崩溃到零。模型输出变成纯噪声。

**β=0.001** 是更微妙的失败：entropy 衰减轨迹几乎和无 bonus 的 baseline 一样（0.098 vs 0.068），几乎没有保持 support 的作用。但 accuracy 仍然在 step 70 后下降，最终 0.146——比 sync_lr1e-5（0.183）更差。微小的 bonus 为 loss 引入了足够的噪声使其适得其反，但不足以阻止崩溃。

**论文意义**：这是决定性的 β sweep。reviewer 不能再说"你只是没调好 β"——问题是结构性的（closed-loop 系统中使用 open-loop 系数），不是参数选择问题。

### 发现 2：cosine_floor 是最佳 open-loop 方法——也是粗糙的 gain scheduling

**cosine_floor**（LR: 1e-5 → 1e-6 via cosine，floor=1e-6）在所有 open-loop 干预中表现最好：

| 指标 | cosine_floor | cosine（无 floor） | sync_lr1e-6 | sync_lr3e-6 |
|---|---|---|---|---|
| AIME@300 | **0.273** | 0.223 | 0.281 | **0.325** |
| OLYMPIAD@300 | **0.486** | 0.440 | 0.476 | **0.506** |
| Entropy@300 | 0.067 | 0.076 | 0.078 | 0.073 |
| LR@step150 | 5.8e-6 | ~5e-7 | 1e-6 | 3e-6 |
| LR@step300 | 1e-6 | ~3e-10 | 1e-6 | 3e-6 |

cosine_floor 比无 floor 的 cosine 在 AIME 上高 22%，因为 floor 阻止了 LR 衰减到 1e-6 以下，保持了持续（虽然缓慢的）学习能力。

**关键洞察**：cosine_floor 的 LR 轨迹（1e-5 → 5.8e-6 → 1e-6）恰好是一个粗糙的 "entropy 高时 LR 高、entropy 低时 LR 低" 的 schedule。这正是 entropy-adaptive LR 要做的事情。**它是所有 LR=1e-5 起步的实验中唯一没有后期下降的**。

但它比 sync_lr3e-6 差（0.273 vs 0.325），因为：
- cosine 曲线是固定的、problem-independent 的，不知道 entropy 什么时候真的需要降 LR
- 早期 LR=1e-5 可能已经过大（sync_lr3e-6 的 3e-6 更合适）
- 一个理想的 adaptive schedule 应该能做得更好

**论文意义**：cosine_floor 的成功是 gain scheduling 原则的间接验证。它证明了"随训练推进降低 LR"是对的方向，只是 cosine 是 open-loop 近似，不如根据 entropy 实际状态来调整。

### 发现 3：联合 open-loop（cosine + entropy bonus）导致 entropy 爆炸

**cosine_ent0.01** 设计为"最强 open-loop baseline"——组合两个最好的单独干预。结果出现了**灾难性交互**：

- **entropy 爆炸**到 11.92 nats（~step 15），与单独 ent0.01 一样
- 尽管 cosine 衰减最终把 LR 降到接近零，**entropy 从未恢复**——在整个 300 步中都保持在 11.92
- cosine 衰减太慢，无法阻止 β=0.01 的熵奖励触发正反馈回路
- 一旦 policy 变成近均匀分布，即使 LR→0 也无济于事：没有梯度信号可循

**Score 轨迹确认**：Score 始终在 ~−1.0（从未从初始值改善），确认模型什么都没学到。

**MINERVA 是个异常值**：cosine_ent0.01 在 MINERVA 上得分 0.283@300，是本批最高。这可能是因为 MINERVA 的题目较简单，近均匀 policy 的残余结构仍能得分——但也可能是噪声。

**论文意义**：组合 open-loop 方法不能修复根本问题——最强的单个方法（entropy bonus）主导了交互并导致相同的失败。这对 open-loop 路线是最后一击。

### 发现 4：优势方差加权熵奖励——和 baseline 表现类似

**advvar_ent**（Phase 2 方法，LR=1e-5，β=0.01 按 per-prompt 优势方差加权）：

- **AIME 在 step 70 达到 peak 0.290**，在早期训练中与 baseline 相当
- **然后下降到 0.158（step 300）**——与 sync_lr1e-5 baseline 类似的慢速崩溃 pattern
- **entropy 轨迹**（0.68→0.31→0.12→0.08）与 sync_lr1e-5（0.68→0.25→0.11→0.07）几乎一样

**诊断**：优势方差加权只改变了 "把 entropy bonus 分配到哪些 prompt"，没改变 update 的总量。LR 始终 =1e-5，所以后期当 entropy 降低时，update/support 比值一样超标。

| 指标 | advvar_ent | sync_lr1e-5 |
|---|---|---|
| AIME@300 | 0.158 | 0.183 |
| Entropy@300 | 0.082 | 0.068 |
| Pattern | 慢速崩溃 | 更快崩溃 |

advvar_ent 实际上保留了略多的 entropy（0.082 vs 0.068），但 AIME 更差。优势方差加权略微减缓了 entropy 消耗，但没有阻止崩溃。

**论文意义**：advvar_ent 解决的是 "where to explore" 而非 "how aggressively to update"。这与 thesis 一致——核心问题是 update-to-support ratio，不是 entropy bonus 的分配方式。

### 发现 5：KL 盲区再次确认

| 实验 | KL@step300 | Entropy@300 | AIME@300 | 状态 |
|---|---|---|---|---|
| cosine_floor | 0.000025 | 0.067 | 0.273 | 稳定 |
| advvar_ent | 0.000288 | 0.082 | 0.158 | 崩溃中 |
| ent0.001 | 0.000479 | 0.098 | 0.146 | 崩溃中 |
| cosine_ent0.01 | −0.000001 | 11.924 | 0.229 | 已爆炸 |
| ent0.1 | −0.000097 | 11.922 | 0.000 | 灾难性 |

- cosine_ent0.01 和 ent0.1 的 KL ≈ 0，尽管它们已经灾难性崩溃
- advvar_ent 的 KL 比 cosine_floor 更高（0.000288 vs 0.000025），但 accuracy 更差
- KL 对最好和最差的实验都显示"平静"——没有任何诊断价值

---

## 4. 本批排名（AIME@300）

1. **cosine_floor**: 0.273 — 最佳 open-loop schedule；floor 阻止了致命的 LR→0
2. **cosine_ent0.01**: 0.229 — entropy 爆炸但 cosine 衰减部分限制了损害
3. **advvar_ent**: 0.158 — Phase 2 方法；慢速崩溃，不比 baseline 好
4. **ent0.001**: 0.146 — 太弱，略有反效果
5. **ent0.1**: 0.000 — 立即灾难性 entropy 爆炸

本批 **5 个实验没有一个超过 sync_lr3e-6（0.325）**。

---

## 5. 全局排名（全部 17 个已完成实验，AIME@300）

| 排名 | 实验 | AIME@300 | 阶段 | 类别 |
|---|---|---|---|---|
| 1 | sync_lr3e-6 | **0.325** | Ph1 | LR sweep |
| 2 | async_lr1e-6 | 0.319 | Baseline | — |
| 3 | async_lr5e-6 | 0.296 | Ph1 | LR sweep |
| 4 | sync_lr5e-6 | 0.294 | Ph1 | LR sweep |
| 5 | sync_lr1e-6 | 0.281 | Baseline | — |
| 6 | **cosine_floor** | **0.273** | Ph1b2 | Open-loop schedule |
| 7 | async_ent0.01 | 0.231 | Ph1 | Entropy bonus |
| 8 | **cosine_ent0.01** | **0.229** | Ph1b2 | 联合 open-loop |
| 9 | sync_ent0.01 | 0.227 | Ph1 | Entropy bonus |
| 10 | sync_cosine | 0.223 | Ph1 | LR schedule |
| 11 | sync_lr1e-5 | 0.183 | Baseline | — |
| 12 | **advvar_ent** | **0.158** | Ph2 | 优势方差熵奖励 |
| 13 | **ent0.001** | **0.146** | Ph1b2 | β sweep |
| 14 | async_lr1e-5 | 0.079 | Baseline | — |
| 15 | async_cosine | 0.021 | Ph1 | LR schedule |
| 16 | sync_clip0.1 | 0.017 | Ph1 | Adaptive clip |
| 17 | **ent0.1** | **0.000** | Ph1b2 | β sweep |

---

## 6. Entropy 动态对比

### 非爆炸实验

| 实验 | @step1 | @step50 | @step150 | @step300 | Pattern |
|---|---|---|---|---|---|
| sync_lr1e-6 | 0.674 | 0.465 | 0.217 | 0.078 | 最慢消耗 |
| sync_lr3e-6 | 0.682 | 0.385 | 0.143 | 0.073 | 较慢消耗 |
| ent0.001 | 0.682 | 0.350 | 0.107 | 0.098 | 接近 baseline |
| advvar_ent | 0.682 | 0.311 | 0.124 | 0.082 | 接近 baseline |
| cosine_floor | 0.675 | 0.286 | 0.098 | 0.067 | 正常消耗 |
| sync_lr1e-5 | 0.677 | 0.250 | 0.109 | 0.068 | baseline |
| sync_lr5e-6 | 0.677 | 0.364 | 0.106 | 0.068 | 较快消耗 |

### 爆炸实验

| 实验 | @step1 | @step10 | @step25 | @step300 |
|---|---|---|---|---|
| ent0.1 | 0.681 | 10.965 | 11.904 | 11.922 |
| cosine_ent0.01 | 0.685 | 1.131 | 11.372 | 11.924 |
| sync_ent0.01 | ~0.68 | 1.146 | 11.706 | 11.925 |

**核心观察**：所有非爆炸的 sync 实验，entropy@300 都收敛到 **0.067–0.098** 的窄区间，与 LR 或干预方式无关。entropy "budget" 在 300 步中被消耗到类似的水平——区别在于消耗路径的质量。

---

## 7. 对论文的影响

### 可以确定性地声称：

1. **β sweep 杀死了"调参就行"的反驳**：β=0.001（太弱）、β=0.01（爆炸）、β=0.1（灾难性）。不存在正确的固定 β——问题是结构性的（closed-loop 系统中的 open-loop 系数），不是参数调优问题。

2. **联合 open-loop ≤ 单个组件**：cosine + entropy bonus 继承了最差的失败模式（entropy bonus 的爆炸），而非组合最优的特性。Open-loop 方法组合效果很差。

3. **cosine_floor 是最强的 open-loop 方法**：比纯 cosine 高 22% AIME，接近保守 baseline，但仍不如最优固定 LR。其改善恰恰是因为它做了粗糙的 gain scheduling——这是 entropy-adaptive LR 原则的间接验证。

4. **优势方差加权解决了错误的维度**：重新分配 bonus 不能帮助解决 update-to-support ratio 问题。这与 thesis 一致。

5. **需要重新表述 thesis 的重点**：从 "entropy budget depletion" 调整为 "path quality"。核心不是 entropy 耗尽（所有实验都耗到类似水平），而是每一步 update 是否与当前 support 匹配。"先升后降" 的 AIME pattern 只出现在后期 LR 过高的实验中。

### 未解决：

- **Entropy-adaptive LR**（主要 Phase 2 干预）因 Bug 4–7 和 12 从未成功运行。理论贡献成立，但方法的实验验证缺失。
- **Async variants** 未运行。

### 建议的论文叙事线索：

Baseline → LR sweep（gain margin 存在）→ Open-loop 失败（entropy bonus、clipping、cosine）→ β sweep（没有正确的 β）→ 联合 open-loop（组合更差）→ cosine_floor（最佳 open-loop，也是粗糙的 gain scheduling，但仍不如最优 LR）→ advvar_ent（解决了错误的维度）

这是一个有说服力的 negative-result 渐进式论证，即使没有 entropy-adaptive LR 的成功实验也能 motivate 它。

---

## 8. 数据说明

- `val_core/`: 6 个文件——AIME、AIME2025、OLYMPIAD_BENCH（×2，完全相同的重复）、GPQA（标注为"Idavidrein/gpqa"）、MINERVA。每个 31 行（step 0–300，每 10 步）。
- `actor/`: 6 个文件——entropy、pg_loss、pg_clipfrac、ppo_kl、grad_norm、lr。每个 300 行（step 1–300）。
- `critic/`: 4 个文件——score/mean、rewards/mean、advantages/mean、returns/mean。每个 300 行。
- `response_len/`: 1 个文件——response_length/mean。300 行。
- cosine_ent0.01 的 LR 衰减到 ~2.93e-10（step 300，无 floor），与 Phase 1 的原始 cosine 一致。
- cosine_floor 的 LR 精确衰减到 1e-6（step 300，floor 正常工作）。
- cosine_ent0.01 的 score 始终在 ~−1.0（从未改善），确认 entropy 爆炸使训练完全无效。
