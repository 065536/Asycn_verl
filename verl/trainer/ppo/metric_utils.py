# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _tensor_std(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return torch.std(tensor.float(), unbiased=False).detach().item()


def _tensor_abs_mean(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return torch.mean(torch.abs(tensor.float())).detach().item()


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        "critic/score/std": _tensor_std(non_aborted_sequence_score),
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        "critic/rewards/std": _tensor_std(non_aborted_sequence_reward),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        "critic/advantages/std": _tensor_std(valid_adv),
        "critic/advantages/abs_mean": _tensor_abs_mean(valid_adv),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        "critic/returns/std": _tensor_std(valid_returns),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    if "overlong_reward" in batch.non_tensor_batch:
        import numpy as np

        overlong_rewards = batch.non_tensor_batch["overlong_reward"]
        metrics["reward/overlong_reward_mean"] = float(np.mean(overlong_rewards))
        metrics["reward/overlong_reward_min"] = float(np.min(overlong_rewards))
    if "overlong" in batch.non_tensor_batch:
        import numpy as np

        metrics["reward/overlong_rate"] = float(np.mean(batch.non_tensor_batch["overlong"]))

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    """
    Compute variance proxy metrics using the simplified expected squared norm approach.

    This metric provides a computationally efficient way to monitor gradient variance
    during training. It works for any advantage estimator as long as sum_pi_squared
    is available from the actor.

    Theory:
    - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
    - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
    - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

    Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
    """
    metrics = {}

    # Check if we have the necessary data (sum_pi_squared is required for W-score)
    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    # Get response mask to only consider valid tokens
    response_mask = batch.batch["response_mask"]

    # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
    # IS weights are computed centrally in ray_trainer.py to avoid duplication
    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        # Extract pre-computed IS weights from batch (already computed in trainer)
        rollout_is_weights = batch.batch["rollout_is_weights"]

        # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

        # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

    # Get scalar advantages (mean over timesteps)
    advantages = batch.batch["advantages"]
    # Compute mean advantage per trajectory using masked_mean
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

    # Compute W values (sum over timesteps)
    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)

    # ====== COMPUTE VARIANCE PROXIES ======
    # Variance proxy should match the actual gradient computation:
    # - If IS weights were computed/applied: use them in variance proxy calculation
    # - Otherwise: compute on-policy variance proxy

    # ====== PROXY 1: Signal Strength ||ḡ||² ======
    # The squared norm of the mean gradient (provided from training loop)
    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    # ====== PROXY 2: Total Power E[||ĝ_τ||²] ======
    # Measures the average of squared gradient norms (Signal + Noise)
    if rollout_is_weights is not None:
        # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
        # Recover original W (before IS correction was applied in line 657)
        # Clamp to avoid division by zero when IS weights are zero
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_original = torch.clamp(w_original, min=0.0)
        # Proxy 2 for off-policy: E[ρ̄² × A² × W]
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        # On-policy Proxy 2: E[A² × W]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    # ====== PROXY 3: Pure Noise - Variance of Mean Vector ======
    # Requires ||ḡ||² from actual batch gradient
    # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            # Ensure non-negative (can be negative due to numerical errors)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    # Decompose into components for analysis
    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            # Proxy 1: Signal Strength ||ḡ||²
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            # Proxy 2: Total Power E[||ĝ_τ||²]
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            # Proxy 3: Pure Noise - Variance of Mean Vector
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            # Component metrics for debugging
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def compute_variance_decomposition_metrics(
    batch: DataProto,
) -> dict[str, float]:
    """Compute variance source decomposition diagnostics for RL gradient noise analysis.

    Returns scalar metrics dict (logged every step).
    Call ``save_noise_decomp_tables`` separately for the heavy per-response / per-prompt tables.
    """
    metrics: dict[str, float] = {}

    required_keys = {"token_level_scores", "advantages", "response_mask"}
    if not required_keys.issubset(batch.batch.keys()):
        return metrics
    if "uid" not in batch.non_tensor_batch:
        return metrics

    with torch.no_grad():
        scores_per_token = batch.batch["token_level_scores"]
        response_mask = batch.batch["response_mask"]
        advantages = batch.batch["advantages"]
        uids = batch.non_tensor_batch["uid"]

        reward_scalar = scores_per_token.sum(dim=-1)
        resp_lengths = response_mask.sum(dim=-1).float()

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        bsz = reward_scalar.shape[0]
        for idx in range(bsz):
            uid_to_indices[uids[idx]].append(idx)

        n_prompts = len(uid_to_indices)
        if n_prompts == 0:
            return metrics

        adv_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

        has_score_energy = "sum_pi_squared" in batch.batch and "old_log_probs" in batch.batch
        if has_score_energy:
            pi_t = torch.exp(batch.batch["old_log_probs"])
            sum_pi_sq = batch.batch["sum_pi_squared"]
            q_per_token = (1.0 - 2.0 * pi_t + sum_pi_sq).clamp(min=0.0)
            q_response = (q_per_token * response_mask).sum(dim=-1)

        N = n_prompts
        K = bsz // N if N > 0 else 1
        metrics["noise_decomp/n_prompts"] = float(N)
        metrics["noise_decomp/n_responses_per_prompt"] = float(K)

        # ── per-prompt reward structure ──
        prompt_ids_ordered = []
        prompt_reward_mean = []
        prompt_reward_std = []
        prompt_success_rate = []
        prompt_n_unique = []
        prompt_is_informative = []

        adv_energy_per_prompt = []
        prompt_mean_a2q = []

        for uid, indices in uid_to_indices.items():
            prompt_ids_ordered.append(uid)
            indices_t = torch.tensor(indices, device=reward_scalar.device)
            rewards_group = reward_scalar[indices_t]
            advs_group = adv_scalar[indices_t]
            k = len(indices)

            r_mean = rewards_group.mean().item()
            r_std = rewards_group.std(unbiased=False).item() if k > 1 else 0.0
            p_i = (rewards_group == 1.0).float().mean().item()
            n_uniq = int(rewards_group.unique().numel())

            prompt_reward_mean.append(r_mean)
            prompt_reward_std.append(r_std)
            prompt_success_rate.append(p_i)
            prompt_n_unique.append(n_uniq)
            prompt_is_informative.append(1.0 if 0.0 < p_i < 1.0 else 0.0)

            adv_energy_per_prompt.append((advs_group ** 2).mean().item())

            if has_score_energy:
                q_group = q_response[indices_t]
                a2q_group = (advs_group ** 2) * q_group
                prompt_mean_a2q.append(a2q_group.mean().item())

        prompt_reward_std_arr = np.array(prompt_reward_std)
        prompt_success_arr = np.array(prompt_success_rate)

        # ── P0 reward informativeness stats ──
        metrics["noise_decomp/reward_std/mean"] = float(prompt_reward_std_arr.mean())
        metrics["noise_decomp/reward_std/median"] = float(np.median(prompt_reward_std_arr))
        metrics["noise_decomp/reward_std/p25"] = float(np.percentile(prompt_reward_std_arr, 25))
        metrics["noise_decomp/reward_std/p75"] = float(np.percentile(prompt_reward_std_arr, 75))
        metrics["noise_decomp/reward_n_unique/mean"] = float(np.mean(prompt_n_unique))
        metrics["noise_decomp/success_rate/mean"] = float(prompt_success_arr.mean())
        metrics["noise_decomp/success_rate/std"] = float(prompt_success_arr.std())
        metrics["noise_decomp/frac_informative"] = float(np.mean(prompt_is_informative))
        metrics["noise_decomp/frac_all_correct"] = float((prompt_success_arr == 1.0).mean())
        metrics["noise_decomp/frac_all_wrong"] = float((prompt_success_arr == 0.0).mean())

        metrics["noise_decomp/adv_energy/mean"] = float(np.mean(adv_energy_per_prompt))
        metrics["noise_decomp/adv_energy/std"] = float(np.std(adv_energy_per_prompt))
        metrics["noise_decomp/adv_scalar/abs_mean"] = float(adv_scalar.abs().mean().item())

        # ── success rate histogram bins ──
        metrics["noise_decomp/success_rate_bin_0"] = float((prompt_success_arr == 0.0).mean())
        metrics["noise_decomp/success_rate_bin_0_25"] = float(
            ((prompt_success_arr > 0.0) & (prompt_success_arr <= 0.25)).mean()
        )
        metrics["noise_decomp/success_rate_bin_25_75"] = float(
            ((prompt_success_arr > 0.25) & (prompt_success_arr < 0.75)).mean()
        )
        metrics["noise_decomp/success_rate_bin_75_1"] = float(
            ((prompt_success_arr >= 0.75) & (prompt_success_arr < 1.0)).mean()
        )
        metrics["noise_decomp/success_rate_bin_1"] = float((prompt_success_arr == 1.0).mean())

        # ── Type 1: Group composition — advantage sign ──
        eps_adv = 1e-8
        pos_mask_resp = adv_scalar > eps_adv
        neg_mask_resp = adv_scalar < -eps_adv
        n_pos = int(pos_mask_resp.sum().item())
        n_neg = int(neg_mask_resp.sum().item())
        n_total = int(bsz)
        adv_pos = adv_scalar[pos_mask_resp]
        adv_neg = adv_scalar[neg_mask_resp]

        metrics["group_comp/num_pos"] = float(n_pos)
        metrics["group_comp/num_neg"] = float(n_neg)
        metrics["group_comp/frac_pos"] = float(n_pos) / max(n_total, 1)
        metrics["group_comp/frac_neg"] = float(n_neg) / max(n_total, 1)
        if n_pos > 0:
            metrics["group_comp/mean_A_pos"] = float(adv_pos.mean().item())
            metrics["group_comp/mean_abs_A_pos"] = float(adv_pos.abs().mean().item())
            metrics["group_comp/sum_A_pos"] = float(adv_pos.sum().item())
            metrics["group_comp/sum_abs_A_pos"] = float(adv_pos.abs().sum().item())
            metrics["group_comp/sum_A2_pos"] = float((adv_pos ** 2).sum().item())
        if n_neg > 0:
            metrics["group_comp/mean_A_neg"] = float(adv_neg.mean().item())
            metrics["group_comp/mean_abs_A_neg"] = float(adv_neg.abs().mean().item())
            metrics["group_comp/sum_A_neg"] = float(adv_neg.sum().item())
            metrics["group_comp/sum_abs_A_neg"] = float(adv_neg.abs().sum().item())
            metrics["group_comp/sum_A2_neg"] = float((adv_neg ** 2).sum().item())

        # ── Type 2: Group composition — prompt success rate (0.5 boundary) ──
        is_all_wrong = prompt_success_arr == 0.0
        is_all_correct = prompt_success_arr == 1.0
        is_low_mixed = (prompt_success_arr > 0.0) & (prompt_success_arr < 0.5)
        is_high_mixed = (prompt_success_arr >= 0.5) & (prompt_success_arr < 1.0)

        n_aw = int(is_all_wrong.sum())
        n_lm = int(is_low_mixed.sum())
        n_hm = int(is_high_mixed.sum())
        n_ac = int(is_all_correct.sum())

        metrics["group_comp/num_prompts_all_wrong"] = float(n_aw)
        metrics["group_comp/num_prompts_low_mixed"] = float(n_lm)
        metrics["group_comp/num_prompts_high_mixed"] = float(n_hm)
        metrics["group_comp/num_prompts_all_correct"] = float(n_ac)
        metrics["group_comp/num_rollouts_all_wrong"] = float(n_aw * K)
        metrics["group_comp/num_rollouts_low_mixed"] = float(n_lm * K)
        metrics["group_comp/num_rollouts_high_mixed"] = float(n_hm * K)
        metrics["group_comp/num_rollouts_all_correct"] = float(n_ac * K)

        adv_sq_per_resp = adv_scalar ** 2
        for bucket_name, bucket_mask in [("low_mixed", is_low_mixed), ("high_mixed", is_high_mixed)]:
            bucket_indices = []
            for idx, (uid, indices) in enumerate(uid_to_indices.items()):
                if bucket_mask[idx]:
                    bucket_indices.extend(indices)
            if bucket_indices:
                bucket_idx_t = torch.tensor(bucket_indices, device=adv_scalar.device)
                a2_bucket = adv_sq_per_resp[bucket_idx_t]
                metrics[f"group_comp/mean_A2_{bucket_name}"] = float(a2_bucket.mean().item())
                metrics[f"group_comp/sum_A2_{bucket_name}"] = float(a2_bucket.sum().item())

        # ── Type 1b: sign × success rate composition (4 atomic groups) ──
        for bucket_name, bucket_mask in [("low", is_low_mixed), ("high", is_high_mixed)]:
            bucket_indices = []
            for idx, (uid, indices) in enumerate(uid_to_indices.items()):
                if bucket_mask[idx]:
                    bucket_indices.extend(indices)
            if not bucket_indices:
                continue
            bucket_idx_t = torch.tensor(bucket_indices, device=adv_scalar.device)
            adv_bucket = adv_scalar[bucket_idx_t]
            resp_mask_bucket = response_mask[bucket_idx_t]
            tok_per_resp = resp_mask_bucket.sum(dim=-1)

            for sign_name, sign_mask_val in [("pos", adv_bucket > eps_adv), ("neg", adv_bucket < -eps_adv)]:
                group_name = f"{bucket_name}_{sign_name}"
                n_group = int(sign_mask_val.sum().item())
                metrics[f"group_comp/num_{group_name}"] = float(n_group)
                metrics[f"group_comp/frac_{group_name}"] = float(n_group) / max(n_total, 1)
                if n_group > 0:
                    adv_group = adv_bucket[sign_mask_val]
                    tok_group = tok_per_resp[sign_mask_val]
                    metrics[f"group_comp/token_count_{group_name}"] = float(tok_group.sum().item())
                    metrics[f"group_comp/sum_A_{group_name}"] = float(adv_group.sum().item())
                    metrics[f"group_comp/sum_A2_{group_name}"] = float((adv_group ** 2).sum().item())

        # ── Advantage-based between/within (unconditional, runs without score energy) ──
        if n_prompts > 1:
            prompt_adv_means = []
            within_adv_vars = []
            for uid, indices in uid_to_indices.items():
                indices_t = torch.tensor(indices, device=adv_scalar.device)
                advs_group = adv_scalar[indices_t]
                prompt_adv_means.append(advs_group.mean().item())
                if len(indices) > 1:
                    within_adv_vars.append(advs_group.var(unbiased=False).item())

            prompt_means_arr = np.array(prompt_adv_means)
            adv_between_var = float(np.var(prompt_means_arr))
            adv_within_var = float(np.mean(within_adv_vars)) if within_adv_vars else 0.0
            adv_total_var = adv_between_var + adv_within_var

            metrics["noise_decomp/adv_between_prompt_var"] = adv_between_var
            metrics["noise_decomp/adv_within_prompt_var"] = adv_within_var
            metrics["noise_decomp/adv_total_var"] = adv_total_var
            if adv_total_var > 0:
                metrics["noise_decomp/adv_between_frac"] = adv_between_var / adv_total_var
                metrics["noise_decomp/adv_within_frac"] = adv_within_var / adv_total_var

            metrics["noise_decomp/prompt_adv_mean/std"] = float(prompt_means_arr.std())
            metrics["noise_decomp/prompt_adv_mean/abs_mean"] = float(np.abs(prompt_means_arr).mean())
            metrics["noise_decomp/prompt_adv_mean/max"] = float(np.abs(prompt_means_arr).max())

        # ── Score-energy-dependent metrics ──
        if has_score_energy:
            q_mean_per_token = (q_per_token * response_mask).sum() / response_mask.sum()
            metrics["noise_decomp/score_energy_q/mean"] = float(q_mean_per_token.item())
            metrics["noise_decomp/q_response/mean"] = float(q_response.mean().item())
            metrics["noise_decomp/q_response/std"] = float(q_response.std(unbiased=False).item())

            a_sq_all = adv_scalar ** 2
            q_all = q_response
            h_all = a_sq_all * q_all

            # ── H pos/neg split (denominator pressure diagnostics) ──
            pos_mask = adv_scalar > 0
            neg_mask = adv_scalar < 0
            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())
            h_total_sum = float(h_all.sum().item())

            metrics["grpo/n_pos_responses"] = float(n_pos)
            metrics["grpo/n_neg_responses"] = float(n_neg)

            if h_total_sum > 1e-12:
                metrics["grpo/H_pos_frac"] = float(h_all[pos_mask].sum().item()) / h_total_sum if n_pos > 0 else 0.0
                metrics["grpo/H_neg_frac"] = float(h_all[neg_mask].sum().item()) / h_total_sum if n_neg > 0 else 0.0
            if n_pos > 0:
                metrics["grpo/H_pos_mean"] = float(h_all[pos_mask].mean().item())
            if n_neg > 0:
                metrics["grpo/H_neg_mean"] = float(h_all[neg_mask].mean().item())

            # Mixed-prompt H split: only prompts with 0 < success_rate < 1
            mixed_pos_h = []
            mixed_neg_h = []
            n_mixed_prompts = 0
            for uid, indices in uid_to_indices.items():
                indices_t = torch.tensor(indices, device=adv_scalar.device)
                advs_g = adv_scalar[indices_t]
                has_pos = (advs_g > 0).any().item()
                has_neg = (advs_g < 0).any().item()
                if has_pos and has_neg:
                    n_mixed_prompts += 1
                    h_g = h_all[indices_t]
                    mixed_pos_h.append(h_g[advs_g > 0])
                    mixed_neg_h.append(h_g[advs_g < 0])

            metrics["grpo/n_mixed_prompts"] = float(n_mixed_prompts)
            metrics["grpo/frac_mixed_prompts"] = float(n_mixed_prompts) / n_prompts if n_prompts > 0 else 0.0
            if mixed_pos_h:
                cat_pos = torch.cat(mixed_pos_h)
                cat_neg = torch.cat(mixed_neg_h)
                metrics["grpo/H_pos_mixed"] = float(cat_pos.mean().item())
                metrics["grpo/H_neg_mixed"] = float(cat_neg.mean().item())
                metrics["grpo/H_pos_mixed_sum"] = float(cat_pos.sum().item())
                metrics["grpo/H_neg_mixed_sum"] = float(cat_neg.sum().item())
                mixed_total = float(cat_pos.sum().item()) + float(cat_neg.sum().item())
                if mixed_total > 1e-12:
                    metrics["grpo/H_neg_mixed_frac"] = float(cat_neg.sum().item()) / mixed_total

            a2q_all_t = torch.tensor(
                [v for pm in prompt_mean_a2q for v in [pm]], dtype=torch.float64
            ) if prompt_mean_a2q else torch.zeros(0, dtype=torch.float64)

            eps = 1e-12

            # ── means / std / CV² for A², Q, H ──
            a_sq_f64 = a_sq_all.double()
            q_f64 = q_all.double()
            h_f64 = h_all.double()

            a2_mean_v = float(a_sq_f64.mean().item())
            q_mean_v = float(q_f64.mean().item())
            h_mean_v = float(h_f64.mean().item())
            a2_std_v = float(a_sq_f64.std(unbiased=False).item())
            q_std_v = float(q_f64.std(unbiased=False).item())
            h_std_v = float(h_f64.std(unbiased=False).item())
            a2_var_v = a2_std_v ** 2
            q_var_v = q_std_v ** 2
            h_var_v = h_std_v ** 2

            metrics["noise_decomp/mean_a2"] = a2_mean_v
            metrics["noise_decomp/mean_q"] = q_mean_v
            metrics["noise_decomp/mean_h"] = h_mean_v
            metrics["noise_decomp/std_a2"] = a2_std_v
            metrics["noise_decomp/std_q"] = q_std_v
            metrics["noise_decomp/std_h"] = h_std_v
            metrics["noise_decomp/cv2_a2"] = a2_var_v / (a2_mean_v ** 2 + eps)
            metrics["noise_decomp/cv2_q"] = q_var_v / (q_mean_v ** 2 + eps)
            metrics["noise_decomp/cv2_h"] = h_var_v / (h_mean_v ** 2 + eps)

            # ── Corr(A², Q) and interaction ratio ──
            cov_a2_q = float(((a_sq_f64 - a2_mean_v) * (q_f64 - q_mean_v)).mean().item())
            corr_a2_q = cov_a2_q / (a2_std_v * q_std_v + eps)
            metrics["noise_decomp/corr_a2_q"] = corr_a2_q
            interaction_ratio = h_mean_v / (a2_mean_v * q_mean_v + eps)
            metrics["noise_decomp/interaction_ratio_a2q"] = interaction_ratio

            # ── Response length diagnostics ──
            resp_len_f64 = resp_lengths.double()
            resp_len_mean = float(resp_len_f64.mean().item())
            resp_len_std = float(resp_len_f64.std(unbiased=False).item())
            metrics["noise_decomp/resp_length/mean"] = resp_len_mean
            metrics["noise_decomp/resp_length/std"] = resp_len_std

            q_over_t = q_all.float() / (resp_lengths + 1.0)
            metrics["noise_decomp/q_over_t/mean"] = float(q_over_t.mean().item())
            metrics["noise_decomp/q_over_t/std"] = float(q_over_t.std(unbiased=False).item())

            corr_a2_t = float(
                ((a_sq_f64 - a2_mean_v) * (resp_len_f64 - resp_len_mean)).mean().item()
                / (a2_std_v * resp_len_std + eps)
            )
            corr_q_t = float(
                ((q_f64 - q_mean_v) * (resp_len_f64 - resp_len_mean)).mean().item()
                / (q_std_v * resp_len_std + eps)
            )
            metrics["noise_decomp/corr_a2_t"] = corr_a2_t
            metrics["noise_decomp/corr_q_t"] = corr_q_t

            # ── Three-way between/within decomposition for A², Q, H ──
            # Also do counterfactual: h_cfact_a = A²·Q̄, h_cfact_q = Ā²·Q
            h_cfact_a = a_sq_all * q_mean_v
            h_cfact_q = a2_mean_v * q_all

            decomp_vars = {
                "a2": a_sq_all, "q": q_all, "h": h_all,
                "h_cfact_a": h_cfact_a, "h_cfact_q": h_cfact_q,
            }

            for dname, d_tensor in decomp_vars.items():
                prompt_means_d = []
                within_vars_d = []
                for uid, indices in uid_to_indices.items():
                    idx_t = torch.tensor(indices, device=d_tensor.device)
                    grp = d_tensor[idx_t]
                    prompt_means_d.append(grp.mean().item())
                    if len(indices) > 1:
                        within_vars_d.append(grp.var(unbiased=False).item())

                if len(prompt_means_d) > 1:
                    btwn = float(np.var(prompt_means_d))
                    wthn = float(np.mean(within_vars_d)) if within_vars_d else 0.0
                    wthn_corr = wthn / K if K > 0 else 0.0

                    batch_btwn = btwn / N if N > 0 else 0.0
                    batch_wthn = wthn / (N * K) if (N * K) > 0 else 0.0
                    batch_total = batch_btwn + batch_wthn

                    pfx = f"noise_decomp/factor_{dname}"
                    metrics[f"{pfx}/between_var"] = btwn
                    metrics[f"{pfx}/within_var_raw"] = wthn
                    metrics[f"{pfx}/within_var_corrected"] = wthn_corr
                    metrics[f"{pfx}/batch_between"] = batch_btwn
                    metrics[f"{pfx}/batch_within"] = batch_wthn
                    metrics[f"{pfx}/batch_total"] = batch_total
                    if batch_total > 0:
                        metrics[f"{pfx}/batch_between_frac"] = batch_btwn / batch_total
                        metrics[f"{pfx}/batch_within_frac"] = batch_wthn / batch_total
                    raw_total = btwn + wthn
                    if raw_total > 0:
                        metrics[f"{pfx}/raw_between_frac"] = btwn / raw_total
                        metrics[f"{pfx}/raw_within_frac"] = wthn / raw_total

            # ── Counterfactual variance attribution ──
            var_h_a_only = float(h_cfact_a.double().var(unbiased=False).item())
            var_h_q_only = float(h_cfact_q.double().var(unbiased=False).item())
            var_interaction = h_var_v - var_h_a_only - var_h_q_only

            metrics["noise_decomp/cfact_var_h"] = h_var_v
            metrics["noise_decomp/cfact_var_a_only"] = var_h_a_only
            metrics["noise_decomp/cfact_var_q_only"] = var_h_q_only
            metrics["noise_decomp/cfact_var_interaction"] = var_interaction
            if h_var_v > 0:
                metrics["noise_decomp/cfact_frac_a_only"] = var_h_a_only / h_var_v
                metrics["noise_decomp/cfact_frac_q_only"] = var_h_q_only / h_var_v
                metrics["noise_decomp/cfact_frac_interaction"] = var_interaction / h_var_v

            # ── Exact lm_head gradient norm metrics (replaces A²Q proxy) ──
            has_exact_grad = "lm_head_grad_norm_sq" in batch.batch
            if has_exact_grad:
                grad_norm_sq = batch.batch["lm_head_grad_norm_sq"].double()
                sampled_mask = grad_norm_sq >= 0
                n_sampled = int(sampled_mask.sum().item())
                metrics["noise_decomp/exact_grad/n_sampled"] = float(n_sampled)

                if n_sampled > 0:
                    gn_sampled = grad_norm_sq[sampled_mask]
                    a2_sampled = a_sq_f64[sampled_mask]
                    h_exact_sampled = a2_sampled * gn_sampled

                    metrics["noise_decomp/exact_grad/mean_gnorm_sq"] = float(gn_sampled.mean().item())
                    metrics["noise_decomp/exact_grad/std_gnorm_sq"] = float(gn_sampled.std(unbiased=False).item())
                    metrics["noise_decomp/exact_grad/mean_h_exact"] = float(h_exact_sampled.mean().item())

                    q_sampled = q_f64[sampled_mask]
                    q_safe = q_sampled.clamp(min=1e-12)
                    cross_ratio = gn_sampled / q_safe
                    metrics["noise_decomp/exact_grad/cross_term_ratio_mean"] = float(cross_ratio.mean().item())
                    metrics["noise_decomp/exact_grad/cross_term_ratio_std"] = float(cross_ratio.std(unbiased=False).item())

                    corr_gn_q = float(
                        ((gn_sampled - gn_sampled.mean()) * (q_sampled - q_sampled.mean())).mean().item()
                        / (gn_sampled.std(unbiased=False).item() * q_sampled.std(unbiased=False).item() + eps)
                    )
                    metrics["noise_decomp/exact_grad/corr_gnorm_q"] = corr_gn_q

                    if n_prompts > 1:
                        prompt_means_gn = []
                        within_vars_gn = []
                        prompt_means_hex = []
                        within_vars_hex = []
                        for uid, indices in uid_to_indices.items():
                            idx_t = torch.tensor(indices, device=grad_norm_sq.device)
                            mask_g = sampled_mask[idx_t]
                            if mask_g.sum() == 0:
                                continue
                            gn_grp = grad_norm_sq[idx_t][mask_g]
                            hex_grp = (a_sq_f64[idx_t][mask_g]) * gn_grp
                            prompt_means_gn.append(gn_grp.mean().item())
                            prompt_means_hex.append(hex_grp.mean().item())
                            if mask_g.sum() > 1:
                                within_vars_gn.append(gn_grp.var(unbiased=False).item())
                                within_vars_hex.append(hex_grp.var(unbiased=False).item())

                        for suffix, pm_list, wv_list in [
                            ("gnorm", prompt_means_gn, within_vars_gn),
                            ("h_exact", prompt_means_hex, within_vars_hex),
                        ]:
                            if len(pm_list) > 1:
                                btwn = float(np.var(pm_list))
                                wthn = float(np.mean(wv_list)) if wv_list else 0.0
                                total_v = btwn + wthn
                                pfx = f"noise_decomp/exact_grad/{suffix}"
                                metrics[f"{pfx}/between_var"] = btwn
                                metrics[f"{pfx}/within_var"] = wthn
                                if total_v > 0:
                                    metrics[f"{pfx}/between_frac"] = btwn / total_v
                                    metrics[f"{pfx}/within_frac"] = wthn / total_v

            # ── Legacy a2q metrics (backward compat) ──
            if len(prompt_mean_a2q) > 1:
                h_arr = np.array(prompt_mean_a2q)
                between_var = float(np.var(h_arr))
                within_vars_legacy = []
                for uid, indices in uid_to_indices.items():
                    if len(indices) > 1:
                        idx_t = torch.tensor(indices, device=reward_scalar.device)
                        a2q_g = (adv_scalar[idx_t] ** 2) * q_response[idx_t]
                        within_vars_legacy.append(a2q_g.var(unbiased=False).item())
                within_var = float(np.mean(within_vars_legacy)) if within_vars_legacy else 0.0
                metrics["noise_decomp/a2q_between_prompt_var"] = between_var
                metrics["noise_decomp/a2q_within_prompt_var"] = within_var
                total = between_var + within_var
                if total > 0:
                    metrics["noise_decomp/a2q_between_frac"] = between_var / total
                    metrics["noise_decomp/a2q_within_frac"] = within_var / total

            # ── prompt_h stats ──
            if n_prompts > 1 and prompt_mean_a2q:
                prompt_h = np.array(prompt_mean_a2q)
                metrics["noise_decomp/prompt_h/mean"] = float(prompt_h.mean())
                metrics["noise_decomp/prompt_h/std"] = float(prompt_h.std())
                metrics["noise_decomp/prompt_h/max"] = float(prompt_h.max())
                metrics["noise_decomp/prompt_h/p90"] = float(np.percentile(prompt_h, 90))

    return metrics


def save_noise_decomp_tables(
    batch: DataProto,
    global_step: int,
    save_dir: str,
) -> None:
    """Save per-response and per-prompt tables as compressed npz.

    Lightweight enough to call every N steps (e.g. every 5-10 steps).
    Files: ``{save_dir}/noise_tables_step{global_step}.npz``

    Per-response arrays (shape [bsz]):
        prompt_idx, reward, advantage, advantage_sq, response_length,
        score_energy_Q (if available), Q_per_token, H_a2q

    Per-prompt arrays (shape [n_prompts]):
        prompt_id, reward_mean, reward_std, success_rate, num_unique_rewards,
        is_informative, a2_mean, q_mean, h_mean, a2_var, q_var, h_var
    """
    required_keys = {"token_level_scores", "advantages", "response_mask"}
    if not required_keys.issubset(batch.batch.keys()):
        return
    if "uid" not in batch.non_tensor_batch:
        return

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        scores_per_token = batch.batch["token_level_scores"]
        response_mask = batch.batch["response_mask"]
        advantages = batch.batch["advantages"]
        uids = batch.non_tensor_batch["uid"]

        reward_scalar = scores_per_token.sum(dim=-1).cpu().numpy().astype(np.float32)
        resp_lengths = response_mask.sum(dim=-1).cpu().numpy().astype(np.float32)
        adv_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1).cpu().numpy().astype(np.float32)
        adv_sq = (adv_scalar ** 2).astype(np.float32)

        bsz = reward_scalar.shape[0]

        has_score_energy = "sum_pi_squared" in batch.batch and "old_log_probs" in batch.batch

        if has_score_energy:
            pi_t = torch.exp(batch.batch["old_log_probs"])
            sum_pi_sq = batch.batch["sum_pi_squared"]
            q_per_token_t = (1.0 - 2.0 * pi_t + sum_pi_sq).clamp(min=0.0)
            q_response_np = (q_per_token_t * response_mask).sum(dim=-1).cpu().numpy().astype(np.float32)
            q_per_tok_np = q_response_np / (resp_lengths + 1.0)
            h_a2q = adv_sq * q_response_np
        else:
            q_response_np = np.full(bsz, np.nan, dtype=np.float32)
            q_per_tok_np = np.full(bsz, np.nan, dtype=np.float32)
            h_a2q = np.full(bsz, np.nan, dtype=np.float32)

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(bsz):
            uid_to_indices[uids[idx]].append(idx)

        uid_list = list(uid_to_indices.keys())
        uid_to_int = {u: i for i, u in enumerate(uid_list)}
        prompt_idx_per_response = np.array([uid_to_int[uids[i]] for i in range(bsz)], dtype=np.int32)

        n_prompts = len(uid_list)
        p_reward_mean = np.empty(n_prompts, dtype=np.float32)
        p_reward_std = np.empty(n_prompts, dtype=np.float32)
        p_success_rate = np.empty(n_prompts, dtype=np.float32)
        p_n_unique = np.empty(n_prompts, dtype=np.int32)
        p_is_informative = np.empty(n_prompts, dtype=np.float32)
        p_a2_mean = np.empty(n_prompts, dtype=np.float32)
        p_q_mean = np.empty(n_prompts, dtype=np.float32)
        p_h_mean = np.empty(n_prompts, dtype=np.float32)
        p_a2_var = np.empty(n_prompts, dtype=np.float32)
        p_q_var = np.empty(n_prompts, dtype=np.float32)
        p_h_var = np.empty(n_prompts, dtype=np.float32)

        for pi, uid in enumerate(uid_list):
            idxs = uid_to_indices[uid]
            rw = reward_scalar[idxs]
            a2 = adv_sq[idxs]
            q_arr = q_response_np[idxs]
            h_arr = h_a2q[idxs]

            p_reward_mean[pi] = rw.mean()
            p_reward_std[pi] = rw.std() if len(idxs) > 1 else 0.0
            p_success = (rw == 1.0).mean()
            p_success_rate[pi] = p_success
            p_n_unique[pi] = len(np.unique(rw))
            p_is_informative[pi] = 1.0 if 0.0 < p_success < 1.0 else 0.0
            p_a2_mean[pi] = a2.mean()
            p_q_mean[pi] = q_arr.mean()
            p_h_mean[pi] = h_arr.mean()
            p_a2_var[pi] = a2.var() if len(idxs) > 1 else 0.0
            p_q_var[pi] = q_arr.var() if len(idxs) > 1 else 0.0
            p_h_var[pi] = h_arr.var() if len(idxs) > 1 else 0.0

        fname = os.path.join(save_dir, f"noise_tables_step{global_step}.npz")
        np.savez_compressed(
            fname,
            global_step=np.int32(global_step),
            N=np.int32(n_prompts),
            K=np.int32(bsz // n_prompts) if n_prompts > 0 else np.int32(0),
            resp_prompt_idx=prompt_idx_per_response,
            resp_reward=reward_scalar,
            resp_advantage=adv_scalar,
            resp_advantage_sq=adv_sq,
            resp_score_energy_Q=q_response_np,
            resp_Q_per_token=q_per_tok_np,
            resp_response_length=resp_lengths,
            resp_H_a2q=h_a2q,
            prompt_id=np.array(uid_list, dtype=object),
            prompt_reward_mean=p_reward_mean,
            prompt_reward_std=p_reward_std,
            prompt_success_rate=p_success_rate,
            prompt_num_unique_rewards=p_n_unique,
            prompt_is_informative=p_is_informative,
            prompt_a2_mean=p_a2_mean,
            prompt_q_mean=p_q_mean,
            prompt_h_mean=p_h_mean,
            prompt_a2_var=p_a2_var,
            prompt_q_var=p_q_var,
            prompt_h_var=p_h_var,
        )


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    # generate bootstrap indices, shape: (n_bootstrap, subset_size)
    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    # pre-allocate result array, shape: (n_fns, n_bootstrap)
    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    # compute metric results for each bootstrap sample
    for fn_idx, reduce_fn in enumerate(reduce_fns):
        # bootstrap sample and compute metric
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    # compute mean and std for each metric function
    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    # 2. cache ns list
    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    # 3. cache metric results
    data_src2uid2var2metric = {}

    # 4. flatten loop
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        # create uid dict
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                # skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                # compute mean and std
                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    # cache ns list
                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    # compute best/worst metrics
                    for n in ns:
                        # compute best/worst metrics
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        # compute maj metrics
                        if has_pred:
                            # create vote_data
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            # compute maj metrics
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val
