# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def _build_topk_rollout_actor_prob_metrics(
    responses: torch.Tensor,
    actor_probs: torch.Tensor,
    rollout_probs: torch.Tensor,
    response_mask_bool: torch.Tensor,
    tokenizer=None,
    topk: int = 10,
) -> dict:
    """
    Build scalar metrics for top-k rollout/actor probability diffs.

    Returned keys are numeric-only so all logging backends can ingest them safely.
    """
    # Align response token IDs with probability tensors.
    # Some pipelines may keep longer response tensors than log-prob tensors.
    if responses.shape != actor_probs.shape:
        if responses.size(0) == actor_probs.size(0) and responses.size(1) >= actor_probs.size(1):
            responses = responses[:, -actor_probs.size(1) :]
        else:
            logger.warning(
                "responses and prob tensors shape mismatch: responses=%s actor_probs=%s, skip top-k token metrics",
                tuple(responses.shape),
                tuple(actor_probs.shape),
            )
            return {}

    full_diff = torch.abs(actor_probs - rollout_probs)
    # Fill non-response positions with -1 to exclude them from top-k selection.
    masked_diff = torch.where(response_mask_bool, full_diff, torch.full_like(full_diff, -1.0))
    flat_diff = masked_diff.flatten()

    valid_count = int(response_mask_bool.sum().item())
    if valid_count <= 0:
        return {}

    k = min(topk, valid_count)
    top_vals, top_indices = torch.topk(flat_diff, k=k, largest=True, sorted=True)

    flat_actor_probs = actor_probs.flatten()
    flat_rollout_probs = rollout_probs.flatten()
    flat_tokens = responses.flatten().to(torch.long)

    def _decode_token(token_id: int) -> str:
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            return ""

    top_metrics = {}
    for rank in range(k):
        idx = top_indices[rank]
        token_id = int(flat_tokens[idx].item())
        top_metrics[f"training/rollout_probs_diff_top{rank + 1}_token_text"] = _decode_token(token_id)
        top_metrics[f"training/rollout_probs_diff_top{rank + 1}_rollout_prob"] = float(flat_rollout_probs[idx].item())
        top_metrics[f"training/rollout_probs_diff_top{rank + 1}_trainer_prob"] = float(flat_actor_probs[idx].item())
        top_metrics[f"training/rollout_probs_diff_top{rank + 1}_abs_diff"] = float(top_vals[rank].item())

    return top_metrics


def calculate_debug_metrics(data: DataProto, tokenizer=None) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }

    # Percentiles of probs_diff: distinguish fat-tail vs uniform drift
    if rollout_probs_diff.numel() > 0:
        diff_f = rollout_probs_diff.float()
        metrics["training/rollout_probs_diff_p90"] = torch.quantile(diff_f, 0.90).item()
        metrics["training/rollout_probs_diff_p99"] = torch.quantile(diff_f, 0.99).item()

    # Response length percentiles: track tail of length distribution
    response_lengths = response_mask.sum(dim=-1).float()
    if response_lengths.numel() > 0:
        metrics["training/response_len_p90"] = torch.quantile(response_lengths, 0.90).item()
        metrics["training/response_len_p99"] = torch.quantile(response_lengths, 0.99).item()

    # EOS token statistics: detect EOS collapse before it hits val acc
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_id = tokenizer.eos_token_id
        resp = responses if responses.shape[1] == response_mask.shape[1] else responses[:, -response_mask.shape[1]:]
        if resp.shape == response_mask.shape:
            eos_mask = (resp == eos_id) & response_mask_bool
            # fraction of sequences that actually generated EOS (vs truncated)
            metrics["training/eos_rate"] = eos_mask.any(dim=-1).float().mean().item()
            # mean log-prob assigned to EOS tokens by the training policy
            if eos_mask.any() and actor_old_log_probs.shape == resp.shape:
                metrics["training/eos_log_prob_mean"] = actor_old_log_probs[eos_mask].float().mean().item()

    # Add top-k token-level details for largest rollout-vs-trainer probability gaps.
    # These are logged as scalar metrics for SwanLab compatibility.
    metrics.update(
        _build_topk_rollout_actor_prob_metrics(
            responses=responses,
            actor_probs=actor_probs,
            rollout_probs=rollout_probs,
            response_mask_bool=response_mask_bool,
            tokenizer=tokenizer,
            topk=10,
        )
    )
    return metrics
