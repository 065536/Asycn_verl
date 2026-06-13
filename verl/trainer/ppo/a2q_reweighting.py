"""
A²Q-Guided Robust Rollout Aggregation for RLVR.

Within each prompt group, uses H = A²Q as a gradient-energy proxy to
soft-clip outlier rollouts before aggregation:

    A_new_{i,k} = w_{i,k} * A_{i,k}

Normalized mode (default): w is rescaled so that the token-weighted
mean equals 1, controlling for first-order effective loss scale.
This makes gains attributable to gradient composition rather than
scalar LR reduction.

Prompt-level reweighting is available as an ablation but not
recommended as the main method (may suppress informative prompts).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class A2QReweightState:
    tau_r_ema: Optional[float] = None
    tau_p_ema: Optional[float] = None
    step: int = 0


def compute_q_response(
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-response score energy Q_{i,k} = sum_t q_t.

    q_per_token ≈ ||∇log π(y_t)||² = 1 - 2π_t + Σ_v π(v)²
    Q_{i,k} = Σ_t q_t * mask_t

    Returns: shape (batch_size,)
    """
    pi_t = torch.exp(old_log_probs)
    q_per_token = (1.0 - 2.0 * pi_t + sum_pi_squared).clamp(min=0.0)
    q_response = (q_per_token * response_mask).sum(dim=-1)
    return q_response


def compute_response_a2(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-response A²_{i,k}.

    For GRPO the advantage is constant across tokens within a response,
    so A² is simply the squared value at any valid token position.

    Returns: shape (batch_size,)
    """
    mask_sum = response_mask.sum(dim=-1).clamp(min=1.0)
    adv_mean = (advantages * response_mask).sum(dim=-1) / mask_sum
    return adv_mean ** 2


def compute_h_and_e(
    a2: torch.Tensor,
    q_response: torch.Tensor,
    prompt_indices: torch.Tensor,
    n_prompts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rollout energy H and prompt energy E.

    H_{i,k} = A²_{i,k} * Q_{i,k}       shape (batch_size,)
    E_i = Σ_k H_{i,k}                    shape (n_prompts,)

    Args:
        a2: per-response A², shape (batch_size,)
        q_response: per-response Q, shape (batch_size,)
        prompt_indices: int tensor mapping each response to its prompt index,
                        shape (batch_size,), values in [0, n_prompts)
        n_prompts: total number of prompts

    Returns: (H, E)
    """
    h = a2 * q_response
    e = torch.zeros(n_prompts, dtype=h.dtype, device=h.device)
    e.scatter_add_(0, prompt_indices, h)
    return h, e


def compute_thresholds(
    h: torch.Tensor,
    e: torch.Tensor,
    percentile_r: float,
    percentile_p: float,
) -> tuple[float, float]:
    """Compute batch-level percentile thresholds for H and E."""
    tau_r = torch.quantile(h.float(), percentile_r / 100.0).item()
    tau_p = torch.quantile(e.float(), percentile_p / 100.0).item()
    return tau_r, tau_p


def compute_weights(
    h: torch.Tensor,
    e: torch.Tensor,
    prompt_indices: torch.Tensor,
    tau_r: float,
    tau_p: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rollout-level and prompt-level soft-clipping weights.

    w_{i,k} = min(1, sqrt(τ_r / (H_{i,k} + ε)))
    w_i     = min(1, sqrt(τ_p / (E_i + ε)))

    Returns: (w_rollout, w_prompt) both shape (batch_size,)
             w_prompt is broadcast from per-prompt to per-response.
    """
    w_rollout = torch.clamp(torch.sqrt(tau_r / (h + eps)), max=1.0)
    e_per_response = e[prompt_indices]
    w_prompt = torch.clamp(torch.sqrt(tau_p / (e_per_response + eps)), max=1.0)
    return w_rollout, w_prompt


def build_prompt_indices(uids, uid_to_idx=None):
    """Map uid list to integer prompt indices.

    Args:
        uids: list/array of prompt uids, length = batch_size.
               Each uid appears K times (once per rollout).
        uid_to_idx: optional pre-built mapping.

    Returns:
        prompt_indices: LongTensor shape (batch_size,)
        n_prompts: int
        uid_to_idx: dict
    """
    if uid_to_idx is None:
        uid_to_idx = {}
        for uid in uids:
            uid_key = uid if isinstance(uid, str) else str(uid)
            if uid_key not in uid_to_idx:
                uid_to_idx[uid_key] = len(uid_to_idx)

    indices = []
    for uid in uids:
        uid_key = uid if isinstance(uid, str) else str(uid)
        indices.append(uid_to_idx[uid_key])

    return torch.tensor(indices, dtype=torch.long), len(uid_to_idx), uid_to_idx


def apply_a2q_reweighting(
    batch,
    state: A2QReweightState,
    enabled: bool = False,
    mode: str = "hierarchical",
    percentile_r: float = 95.0,
    percentile_p: float = 90.0,
    ema_beta: float = 0.9,
    warmup_steps: int = 10,
    normalize: bool = True,
    eps: float = 1e-8,
):
    """Apply A²Q hierarchical gradient reweighting to the batch advantages.

    This function modifies batch.batch["advantages"] in-place and returns
    reweighting metrics.

    Args:
        batch: DataProto with batch.batch containing "advantages",
               "response_mask", "old_log_probs", "sum_pi_squared",
               and batch.non_tensor_batch containing "uid".
        state: mutable EMA state, persisted across steps on the trainer.
        enabled: master switch.
        mode: "hierarchical" | "prompt_only" | "rollout_only"
        percentile_r: rollout-level threshold percentile (e.g. 95).
        percentile_p: prompt-level threshold percentile (e.g. 90).
        ema_beta: EMA smoothing for thresholds.
        warmup_steps: steps before reweighting activates.
        normalize: if True, rescale weights so mean(w)=1, preserving
                   effective update magnitude (only gradient composition changes).
        eps: numerical stability constant.

    Returns:
        dict of metrics (always computed for diagnostics, even if not enabled).
    """
    metrics = {}
    state.step += 1

    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"]
    device = advantages.device

    has_score_energy = (
        "sum_pi_squared" in batch.batch and "old_log_probs" in batch.batch
    )

    if not has_score_energy:
        metrics["a2q_reweight/skipped"] = 1.0
        return metrics

    old_log_probs = batch.batch["old_log_probs"]
    sum_pi_squared = batch.batch["sum_pi_squared"]
    uids = batch.non_tensor_batch["uid"]

    q_response = compute_q_response(old_log_probs, sum_pi_squared, response_mask)
    a2 = compute_response_a2(advantages, response_mask)

    prompt_indices, n_prompts, _ = build_prompt_indices(uids)
    prompt_indices = prompt_indices.to(device)

    h, e = compute_h_and_e(a2, q_response, prompt_indices, n_prompts)

    tau_r_batch, tau_p_batch = compute_thresholds(h, e, percentile_r, percentile_p)

    if state.tau_r_ema is None:
        state.tau_r_ema = tau_r_batch
        state.tau_p_ema = tau_p_batch
    else:
        state.tau_r_ema = ema_beta * state.tau_r_ema + (1 - ema_beta) * tau_r_batch
        state.tau_p_ema = ema_beta * state.tau_p_ema + (1 - ema_beta) * tau_p_batch

    tau_r = state.tau_r_ema
    tau_p = state.tau_p_ema

    w_rollout, w_prompt = compute_weights(h, e, prompt_indices, tau_r, tau_p, eps)

    if mode == "hierarchical":
        w_combined = w_rollout * w_prompt
    elif mode == "rollout_only":
        w_combined = w_rollout
    elif mode == "prompt_only":
        w_combined = w_prompt
    elif mode == "vanilla":
        w_combined = torch.ones_like(w_rollout)
    else:
        raise ValueError(f"Unknown a2q_reweight_mode: {mode}")

    h_total = h.sum().item()
    h_sq_total = (h ** 2).sum().item()
    neff_rollout = (h_total ** 2) / (h_sq_total + 1e-30)

    e_total = e.sum().item()
    e_sq_total = (e ** 2).sum().item()
    neff_prompt = (e_total ** 2) / (e_sq_total + 1e-30)

    h_sorted = torch.sort(h, descending=True).values
    h_cumsum = h_sorted.cumsum(0)
    h_sum = h_sorted.sum().clamp(min=1e-30)

    e_sorted = torch.sort(e, descending=True).values
    e_cumsum = e_sorted.cumsum(0)
    e_sum = e_sorted.sum().clamp(min=1e-30)

    bs = float(h.numel())
    n_p = float(n_prompts)

    def _top_share(cumsum, total, count, top_n):
        idx = min(top_n, int(count)) - 1
        if idx < 0:
            return 0.0
        return (cumsum[idx] / total).item()

    def _gini(sorted_desc, total, count):
        if count < 2 or total < 1e-30:
            return 0.0
        n = int(count)
        ranks = torch.arange(1, n + 1, dtype=sorted_desc.dtype, device=sorted_desc.device)
        return (((n + 1 - 2 * ranks) * sorted_desc[:n]).sum() / (n * total)).item()

    metrics["a2q_reweight/h_mean"] = h_total / max(bs, 1.0)
    metrics["a2q_reweight/h_max"] = h.max().item()
    metrics["a2q_reweight/h_rollout_top1_share"] = _top_share(h_cumsum, h_sum, bs, 1)
    metrics["a2q_reweight/h_rollout_top5_share"] = _top_share(h_cumsum, h_sum, bs, max(1, int(bs * 0.05)))
    metrics["a2q_reweight/h_rollout_top10_share"] = _top_share(h_cumsum, h_sum, bs, max(1, int(bs * 0.10)))
    metrics["a2q_reweight/h_rollout_neff"] = neff_rollout
    metrics["a2q_reweight/h_rollout_neff_ratio"] = neff_rollout / max(bs, 1.0)
    metrics["a2q_reweight/h_rollout_gini"] = _gini(h_sorted, h_sum, bs)

    metrics["a2q_reweight/e_mean"] = e_total / max(n_p, 1.0)
    metrics["a2q_reweight/e_max"] = e.max().item()
    metrics["a2q_reweight/h_prompt_top1_share"] = _top_share(e_cumsum, e_sum, n_p, 1)
    metrics["a2q_reweight/h_prompt_top5_share"] = _top_share(e_cumsum, e_sum, n_p, max(1, int(n_p * 0.05)))
    metrics["a2q_reweight/h_prompt_top10_share"] = _top_share(e_cumsum, e_sum, n_p, max(1, int(n_p * 0.10)))
    metrics["a2q_reweight/h_prompt_neff"] = neff_prompt
    metrics["a2q_reweight/h_prompt_neff_ratio"] = neff_prompt / max(n_p, 1.0)
    metrics["a2q_reweight/h_prompt_gini"] = _gini(e_sorted, e_sum, n_p)

    metrics["a2q_reweight/tau_r"] = tau_r
    metrics["a2q_reweight/tau_p"] = tau_p
    metrics["a2q_reweight/tau_r_batch"] = tau_r_batch
    metrics["a2q_reweight/tau_p_batch"] = tau_p_batch
    metrics["a2q_reweight/rollout_weight_mean"] = w_rollout.mean().item()
    metrics["a2q_reweight/rollout_weight_min"] = w_rollout.min().item()
    metrics["a2q_reweight/rollout_clipped_frac"] = (w_rollout < 1.0).float().mean().item()
    metrics["a2q_reweight/prompt_weight_mean"] = w_prompt.mean().item()
    metrics["a2q_reweight/prompt_weight_min"] = w_prompt.min().item()
    metrics["a2q_reweight/prompt_clipped_frac"] = (w_prompt < 1.0).float().mean().item()
    metrics["a2q_reweight/combined_weight_mean"] = w_combined.mean().item()
    metrics["a2q_reweight/skipped"] = 0.0
    metrics["a2q_reweight/active"] = 0.0

    w2_h = w_combined ** 2 * h
    w2_h_sum = w2_h.sum().clamp(min=1e-30)
    metrics["a2q_reweight/energy_weighted_w2_mean"] = (w2_h_sum / h_sum).item()
    metrics["a2q_reweight/h_sum_ratio"] = (w2_h_sum / h_sum).item()

    top5_n = max(1, int(bs * 0.05))
    h_top5_idx = torch.topk(h, top5_n).indices
    metrics["a2q_reweight/h_top5_share_before"] = (h[h_top5_idx].sum() / h_sum).item()
    metrics["a2q_reweight/h_top5_share_after"] = (w2_h[h_top5_idx].sum() / w2_h_sum).item()

    w2_h_sq = (w2_h ** 2).sum().clamp(min=1e-30)
    h_sq_for_neff = (h ** 2).sum().clamp(min=1e-30)
    metrics["a2q_reweight/neff_before"] = (h_sum ** 2 / h_sq_for_neff).item()
    metrics["a2q_reweight/neff_after"] = (w2_h_sum ** 2 / w2_h_sq).item()

    metrics["a2q_reweight/energy_weighted_w_mean"] = ((w_combined * h).sum() / h_sum).item()

    if n_prompts > 0:
        e_after = torch.zeros(n_prompts, device=h.device, dtype=h.dtype)
        e_after.scatter_add_(0, prompt_indices, w2_h)
        e_after_sum = e_after.sum().clamp(min=1e-30)
        top1_p_idx = e.argmax()
        metrics["a2q_reweight/prompt_top1_share_before"] = (e[top1_p_idx] / e_sum).item()
        metrics["a2q_reweight/prompt_top1_share_after"] = (e_after[top1_p_idx] / e_after_sum).item()
        e_after_sq = (e_after ** 2).sum().clamp(min=1e-30)
        metrics["a2q_reweight/neff_prompt_before"] = (e_sum ** 2 / (e ** 2).sum().clamp(min=1e-30)).item()
        metrics["a2q_reweight/neff_prompt_after"] = (e_after_sum ** 2 / e_after_sq).item()

    # === Top-H attribution: what are high-H rollouts made of? ===
    mask_sum_attr = response_mask.sum(dim=-1).clamp(min=1.0)
    adv_mean_signed = (advantages * response_mask).sum(dim=-1) / mask_sum_attr
    resp_length = mask_sum_attr

    top5_n_attr = max(1, int(bs * 0.05))
    top5_idx_attr = torch.topk(h, top5_n_attr).indices

    # Q1: Factor attribution — is top-H driven by A² or Q?
    metrics["a2q_reweight/top5_A2_mean"] = a2[top5_idx_attr].mean().item()
    metrics["a2q_reweight/top5_Q_mean"] = q_response[top5_idx_attr].mean().item()
    metrics["a2q_reweight/top5_H_mean"] = h[top5_idx_attr].mean().item()
    metrics["a2q_reweight/top5_length_mean"] = resp_length[top5_idx_attr].mean().item()

    # Q3: Advantage sign — are top-H rollouts being reinforced or suppressed?
    top5_adv = adv_mean_signed[top5_idx_attr]
    metrics["a2q_reweight/top5_positive_adv_frac"] = (top5_adv > 0).float().mean().item()
    metrics["a2q_reweight/top5_negative_adv_frac"] = (top5_adv < 0).float().mean().item()
    metrics["a2q_reweight/top5_adv_mean"] = top5_adv.mean().item()

    # Q2: Prompt type — do top-H rollouts come from informative (mixed) prompts?
    if "token_level_scores" in batch.batch:
        resp_reward = (batch.batch["token_level_scores"] * response_mask).sum(dim=-1)
        resp_success = (resp_reward > 0).float()

        prompt_succ_sum = torch.zeros(n_prompts, device=device, dtype=torch.float32)
        prompt_cnt = torch.zeros(n_prompts, device=device, dtype=torch.float32)
        prompt_succ_sum.scatter_add_(0, prompt_indices, resp_success)
        prompt_cnt.scatter_add_(0, prompt_indices, torch.ones(int(bs), device=device, dtype=torch.float32))
        prompt_sr = prompt_succ_sum / prompt_cnt.clamp(min=1.0)

        all_correct_mask = (prompt_sr == 1.0)
        all_wrong_mask = (prompt_sr == 0.0)
        mixed_mask = ~all_correct_mask & ~all_wrong_mask

        metrics["a2q_reweight/frac_all_correct"] = all_correct_mask.float().mean().item()
        metrics["a2q_reweight/frac_all_wrong"] = all_wrong_mask.float().mean().item()
        metrics["a2q_reweight/frac_mixed"] = mixed_mask.float().mean().item()

        prompt_type_vec = torch.zeros(n_prompts, dtype=torch.long, device=device)
        prompt_type_vec[all_wrong_mask] = 1
        prompt_type_vec[mixed_mask] = 2
        resp_prompt_type = prompt_type_vec[prompt_indices]

        h_total_safe = h.sum().clamp(min=1e-30)
        metrics["a2q_reweight/H_share_all_correct"] = h[resp_prompt_type == 0].sum().item() / h_total_safe.item()
        metrics["a2q_reweight/H_share_all_wrong"] = h[resp_prompt_type == 1].sum().item() / h_total_safe.item()
        metrics["a2q_reweight/H_share_mixed"] = h[resp_prompt_type == 2].sum().item() / h_total_safe.item()

        top5_prompt_type = resp_prompt_type[top5_idx_attr]
        metrics["a2q_reweight/top5_mixed_prompt_frac"] = (top5_prompt_type == 2).float().mean().item()
        metrics["a2q_reweight/top5_allcorrect_prompt_frac"] = (top5_prompt_type == 0).float().mean().item()
        metrics["a2q_reweight/top5_allwrong_prompt_frac"] = (top5_prompt_type == 1).float().mean().item()
        metrics["a2q_reweight/top5_reward_mean"] = resp_reward[top5_idx_attr].mean().item()

    if enabled and state.step > warmup_steps:
        w_raw = w_combined.detach()
        if normalize:
            token_counts = response_mask.sum(dim=-1).clamp(min=1.0)  # (batch_size,)
            w_token_weighted_sum = (w_raw * token_counts).sum()
            total_tokens = token_counts.sum().clamp(min=1.0)
            w_mean_token = (w_token_weighted_sum / total_tokens).clamp(min=eps)
            w_apply = w_raw / w_mean_token
            metrics["a2q_reweight/normalize_factor"] = w_mean_token.item()
            metrics["a2q_reweight/normalize_mode"] = 1.0  # token-weighted
        else:
            w_apply = w_raw
            metrics["a2q_reweight/normalize_factor"] = 1.0
            metrics["a2q_reweight/normalize_mode"] = 0.0

        metrics["a2q_reweight/w_apply_mean"] = w_apply.mean().item()
        metrics["a2q_reweight/w_apply_min"] = w_apply.min().item()
        metrics["a2q_reweight/w_apply_max"] = w_apply.max().item()
        metrics["a2q_reweight/w_apply_std"] = w_apply.std().item()
        metrics["a2q_reweight/w_raw_mean"] = w_raw.mean().item()

        w_token = w_apply.unsqueeze(-1)
        adv_before_abs_sum = (advantages * response_mask).abs().sum().clamp(min=1e-30)
        batch.batch["advantages"] = advantages * w_token
        adv_after_abs_sum = (advantages * w_token * response_mask).abs().sum()
        metrics["a2q_reweight/active"] = 1.0
        metrics["a2q_reweight/effective_adv_scale"] = (adv_after_abs_sum / adv_before_abs_sum).item()

        adv_rms_before = ((advantages ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1.0)).sqrt()
        adv_rms_after = (((advantages * w_token) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1.0)).sqrt()
        metrics["a2q_reweight/adv_rms_before"] = adv_rms_before.item()
        metrics["a2q_reweight/adv_rms_after"] = adv_rms_after.item()

    return metrics
