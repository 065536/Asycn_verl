# Gradient Diagnostics Checklist

> **核心原则**：不再用 A²Q proxy。所有 gradient 量直接算 ∇_θ L_c。
>
> **统一术语**：
> - **RGE_{i,k}** (Rollout Gradient Energy) = ‖∇_θ ℓ_{i,k}‖² = ‖g_{i,k}‖²
>   这是 per-rollout loss 对参数的梯度平方范数，之前 A²Q 试图近似的量。
> - **旧 proxy**: A²Q = A²_{i,k} · Σ_t q_t（logit-space，假设 token 间正交）→ 已废弃
> - **ℓ_{i,k}** = -A_{i,k} Σ_t log π_θ(a_{i,k,t} | x_i, a_{i,k,<t})（vanilla REINFORCE）
>   实际用 PPO clipped loss 时，直接用代码里的 loss_per_token。
>
> 记号：N prompts × K responses × T tokens。
>
> Group masked loss:
>   L_c = (1/D) Σ_{i,k,t} 1[(i,k)∈c] m_{i,k,t} ℓ_{i,k,t}
> D = 训练原始 denominator（如 total valid tokens），保证 g_total = Σ_c g_c

## 0. Group 定义

**按 advantage sign:**
- G⁺ = {r : A_r > 0}
- G⁻ = {r : A_r < 0}

**按 prompt success rate (p_i = (1/K) Σ_k r_{i,k}):**
- all_wrong: p_i = 0
- low_mixed: 0 < p_i < 0.5
- high_mixed: 0.5 ≤ p_i < 1
- all_correct: p_i = 1

**按 sign × success rate (4 核心 group):**
- G⁺_low = {A>0, 0<p<0.5}
- G⁻_low = {A<0, 0<p<0.5}
- G⁺_high = {A>0, 0.5≤p<1}
- G⁻_high = {A<0, 0.5≤p<1}

**Union groups（用于 cosine from norms trick）:**
- pos_neg = G⁺ ∪ G⁻
- high_pos_neg = G⁺_high ∪ G⁻_high
- low_pos_neg = G⁺_low ∪ G⁻_low

---

## 1. build_group_masks 实现

输入 rewards [N,K] + advantages [N,K]，输出 dict of bool masks [N,K]。

```python
def build_group_masks(rewards, advantages, eps=1e-8):
    p = rewards.float().mean(dim=1)  # [N]
    adv = advantages.detach()
    pos = adv > eps
    neg = adv < -eps
    all_wrong = p <= eps
    all_correct = p >= 1.0 - eps
    low_mixed = (p > eps) & (p < 0.5)
    high_mixed = (p >= 0.5) & (p < 1.0 - eps)
    # expand prompt-level → rollout-level [N,K]
    groups = {
        "pos": pos, "neg": neg,
        "low_pos": low_mixed[:,None].expand_as(adv) & pos,
        "low_neg": low_mixed[:,None].expand_as(adv) & neg,
        "high_pos": high_mixed[:,None].expand_as(adv) & pos,
        "high_neg": high_mixed[:,None].expand_as(adv) & neg,
        "mixed": (low_mixed | high_mixed)[:,None].expand_as(adv),
        "all_wrong": all_wrong[:,None].expand_as(adv),
        "all_correct": all_correct[:,None].expand_as(adv),
        "pos_neg": pos | neg,
        "high_pos_neg": high_mixed[:,None].expand_as(adv) & (pos | neg),
        "low_pos_neg": low_mixed[:,None].expand_as(adv) & (pos | neg),
    }
    return groups, {"p": p, "all_wrong": all_wrong, "all_correct": all_correct,
                    "low_mixed": low_mixed, "high_mixed": high_mixed}
```

Status: ☐

---

## Type 1: Group Composition — Advantage Sign

每步记录，纯标量，零开销。

| Metric | Status |
|---|---|
| `group_comp/num_pos`, `num_neg`, `frac_pos`, `frac_neg` | ✅ metric_utils.py |
| `group_comp/mean_A_pos`, `mean_A_neg` | ✅ metric_utils.py |
| `group_comp/mean_abs_A_pos`, `mean_abs_A_neg` | ✅ metric_utils.py |
| `group_comp/sum_A_pos`, `sum_A_neg` | ✅ metric_utils.py |
| `group_comp/sum_abs_A_pos`, `sum_abs_A_neg` | ✅ metric_utils.py |
| `group_comp/sum_A2_pos`, `sum_A2_neg` | ✅ metric_utils.py |

核心对比：Σ A⁺ vs -Σ A⁻（一阶平衡？）、Σ A²⁺ vs Σ A²⁻（二阶偏向？）

---

## Type 2: Group Composition — Prompt Success Rate

每步记录，纯标量，零开销。

| Metric | Status |
|---|---|
| `group_comp/num_prompts_{all_wrong,low_mixed,high_mixed,all_correct}` | ✅ metric_utils.py |
| `group_comp/num_rollouts_{all_wrong,low_mixed,high_mixed,all_correct}` | ✅ metric_utils.py |
| `group_comp/mean_A2_{low_mixed,high_mixed}` | ✅ metric_utils.py |
| `group_comp/sum_A2_{low_mixed,high_mixed}` | ✅ metric_utils.py |

---

## Type 3-6: Gradient Geometry（norm, cosine, cancellation, projection）

### Cosine from norms trick（核心技巧：不存 full gradient）

已知 ‖g_a‖², ‖g_b‖², ‖g_{a∪b}‖²（即 ‖g_a + g_b‖²），则：

```
⟨g_a, g_b⟩ = (‖g_{a∪b}‖² - ‖g_a‖² - ‖g_b‖²) / 2
cos(g_a, g_b) = ⟨g_a, g_b⟩ / (‖g_a‖·‖g_b‖ + ε)
cancel(a,b) = 1 - ‖g_{a∪b}‖ / (‖g_a‖ + ‖g_b‖ + ε)
```

当前实现存 4 个 atomic V×d matrices（high_pos, high_neg, low_pos, low_neg），
从中计算所有 pairwise inner products → 所有 composite group norms → 所有 cosine/cancellation。

**显存注意**：每个 V×d 矩阵约 933MB (V=151936, D=1536, float32)。
当前同时存 4 个 ≈ 3.7GB。若需更多 group，应改为逐 group 计算 norm_sq 后释放。

### 需要记录的 derived metrics

| Category | Metric | 推导方式 | Status |
|---|---|---|---|
| **Norm** | `lm_grad/norm_{pos,neg,high_pos,high_neg,low_pos,low_neg,pos_neg,...}` | 直接 | ✅ dp_actor.py |
| **Norm ratio** | `lm_grad/pos_vs_neg/norm_ratio` = ‖g⁺‖/‖g⁻‖ | 直接 | ✅ dp_actor.py |
| **Norm ratio** | `lm_grad/high_pos_vs_neg/norm_ratio` | 直接 | ✅ dp_actor.py |
| **Cosine** | `lm_grad/pos_vs_neg/cos` | from inner products | ✅ dp_actor.py |
| **Cosine** | `lm_grad/high_pos_vs_neg/cos` | from inner products | ✅ dp_actor.py |
| **Cosine** | `lm_grad/low_pos_vs_neg/cos` | from inner products | ✅ dp_actor.py |
| **Cancel** | `lm_grad/pos_vs_neg/cancel` | from inner products | ✅ dp_actor.py |
| **Cancel** | `lm_grad/high_pos_vs_neg/cancel` | from inner products | ✅ dp_actor.py |
| **Cancel** | `lm_grad/low_pos_vs_neg/cancel` | from inner products | ✅ dp_actor.py |
| **Projection** | `lm_grad/cos_pos_total`, `cos_neg_total` | 当前用 pos_neg 近似 | ✅ dp_actor.py |

**Projection TODO**：当前用 pos_neg 近似 g_total。更严谨应显式算 g_total（包含
all_wrong/all_correct 的贡献）。如果以后加 KL/entropy/length penalty 到 loss，
必须改为显式 g_total。

---

## Type 7: Per-Sample Gradient Strength

| Metric | 公式 | Status |
|---|---|---|
| `lm_grad/norm_pos_batch` | g_c with D=subsample_denom | 即 Type 3 的 norm |
| `lm_grad/norm_pos_mean` | g_c with D_c = group valid tokens | ✅ dp_actor.py |
| `lm_grad/norm_neg_batch` | 同上 | 即 Type 3 |
| `lm_grad/norm_neg_mean` | g_c with D_c = group valid tokens | ✅ dp_actor.py |

group_mean 定义：D_c = Σ_{r∈c,t} m_{r,t}（group 内 valid token 数），即 token-level mean。
处理 empty group：D_c = max(token_count, 1)。

---

## ~~Type 8: 已删除~~

旧 A²Q proxy 已确认不合理（logit-space、假设 token 正交），无需再做对比验证。

---

## Implementation: Route A (output lm_head projection, analytic)

### 覆盖范围说明

Route A 计算的是 **output lm_head projection** 对 W_{lm,out} 的梯度。
如果模型使用 tied embeddings (W_lm = W_embed)，本计算不包含 input embedding
lookup 路径的梯度贡献。论文中须说明。

### w_token 提取

对 PPO clipped loss，w_t = ∂ℓ_t / ∂ log p_{θ,t}。
**可导变量是 current log_prob**（selected_logprobs），old_log_prob 是 detached constant。

```python
def extract_w_token(loss_fn, selected_logprobs, old_log_probs,
                    advantages, response_mask, config):
    lp = selected_logprobs.detach().clone().requires_grad_(True)
    loss_per_token, _ = loss_fn(
        log_prob=lp,
        old_log_prob=old_log_probs.detach(),
        advantages=advantages.detach(),
        response_mask=response_mask,
        config=config,
    )
    global_denom = response_mask.sum().clamp_min(1.0)
    loss = (loss_per_token * response_mask).sum() / global_denom
    w_token = torch.autograd.grad(loss, lp, retain_graph=False)[0].detach()
    return w_token
```

**注意**：w_token 必须在**当前 policy 的 selected_logprobs** 处求导，不是在 ratio=1 处。
如果训练时 current policy 已偏离 old policy，ratio≠1 会影响 clipping 分支，从而影响 w_t。
当前实现（vanilla REINFORCE form）等价于在 ratio=1 处求导（w_t = -A_{i,k}），
这是一个 rollout-time diagnostic 近似，不是精确的 training gradient weight。

### lm_head group gradient

G_c^{out-head} = Σ_{t∈c} w_t (e_{a_t} - p_t) h_tᵀ

```python
def compute_lm_head_group_grad(hidden_states, logits, labels, w_token,
                                response_mask, group_mask, chunk_size=128):
    token_mask = group_mask[:,:,None].bool() & response_mask.bool()
    h = hidden_states[token_mask].float()   # [S, D]
    z = logits[token_mask].float()          # [S, V]
    a = labels[token_mask].long()           # [S]
    w = w_token[token_mask].float()         # [S]

    V, D = z.shape[-1], h.shape[-1]
    grad_W = torch.zeros((V, D), dtype=torch.float32, device=h.device)

    for start in range(0, h.shape[0], chunk_size):
        end = min(start + chunk_size, h.shape[0])
        p_c = torch.softmax(z[start:end], dim=-1)
        wh = w[start:end, None] * h[start:end]     # [C, D]
        grad_W.index_add_(0, a[start:end], wh)      # + w·e_a·hᵀ
        grad_W.add_(-(p_c.T @ wh))                  # - w·p·hᵀ
    return grad_W
```

**显存**：一个 grad_W ≈ V×D×4 bytes。V=151936, D=1536 → ~933MB。
应逐 group 计算 norm_sq 后立即释放，不要同时缓存多个 grad_W。
当前实现用 4 atomic accumulators (~3.7GB) 以支持 pairwise inner products。

### Denominator 与子采样

当前实现子采样 64 responses（按 prompt group 抽样）。
denominator 用 **subsample 内的 valid token 数**（D_sub），
使得 norm/cosine/cancellation 反映子样本内的 group geometry，不依赖 full-batch size。

如果需要和训练 loss 的真实 gradient scale 对齐，应改用 D_global 或不做子采样。

### Distributed 注意事项

**FSDP (model/parameter parallel)**：不同 rank 持有不同参数 shard。
对 lm_head Route A，grad_W 通常是完整的（lm_head 一般不做 column shard）。
如果 FSDP shard 了 lm_head，需要 all_reduce grad_W tensor 后再算 norm。

**Data Parallel**：不同 DP rank 处理不同数据子集。
标量 all_reduce(norm_sq) 给出 Σ_r ‖g^(r)‖²，**不是** ‖Σ_r g^(r)‖²。
两者差别 = 2 × Σ_{r≠s} ⟨g^(r), g^(s)⟩（cross-rank inner products）。
如果要真实全局 gradient geometry，必须先 all_reduce gradient tensor 再算 norm：
```python
torch.distributed.all_reduce(grad_W, op=torch.distributed.ReduceOp.SUM)
norm_sq = (grad_W * grad_W).sum()
```
当前实现是 **per-DP-rank local diagnostic**。

### Route A 在线 metrics 前缀

`lm_grad/norm_{pos,neg,...}`, `lm_grad/pos_vs_neg/cos`, `lm_grad/pos_vs_neg/cancel` 等。

---

## Implementation: Route B (masked backward, full model) — 暂缓

### masked group loss

```python
def masked_group_loss(loss_per_token, response_mask, group_mask, global_denom):
    token_mask = group_mask[:,:,None].float() * response_mask.float()
    return (loss_per_token * token_mask).sum() / global_denom
```

### gradient norm via autograd

```python
def grad_norm_sq_from_loss(loss, model, retain_graph=True):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                 create_graph=False, allow_unused=True)
    norm_sq = sum((g.detach().float()**2).sum() for g in grads if g is not None)
    # NOTE: this all_reduce is only correct for model-parallel parameter shards.
    # For data parallel, it gives Σ_r ‖g^(r)‖², NOT ‖Σ_r g^(r)‖².
    # For exact global gradient geometry, all-reduce gradient tensors before norms.
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.SUM)
    return norm_sq
```

对 9 个 needed groups 各调一次 → 9 次 backward-style autograd.grad。

### Route B metrics 前缀

`full_grad/norm_{pos,neg,...}`, `full_grad/pos_vs_neg/cos`, `full_grad/pos_vs_neg/cancel` 等。

---

## Execution Strategy

**Step 1** ✅: Type 1-2（每步，零开销）+ Route A（lm_head，每 N 步）。在线跑训练收集数据。

**Step 2** (暂缓): 在少量 checkpoint 跑 Route B（全模型），对比 Route A 趋势是否一致。

### 频率

- Type 1-2: **每步**
- Route A (Type 3-7): 每 5 步，子采样 64 responses
- Route B (Type 3-7): checkpoint-only，离线脚本
