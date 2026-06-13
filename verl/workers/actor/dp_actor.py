# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        entropy_in_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            entropy_in_loss: Whether entropy will be used in the loss (needs grad).
                When False (e.g., entropy only for LR scheduling/logging), entropy is
                computed with torch.no_grad() before the inplace log_probs op, saving
                memory and reducing computation graph complexity.
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    # Compute entropy BEFORE inplace log_probs to preserve original logits_rmpad.
                    # When entropy is not in the loss (pure metric/LR scheduling), use no_grad
                    # to avoid retaining the logits computation graph, which reduces memory
                    # pressure and avoids extra FSDP all-gathers during backward.
                    if calculate_entropy:
                        if entropy_in_loss:
                            # Entropy contributes to loss: need gradients, disable inplace
                            inplace_backward = False
                            entropy_rmpad = (
                                self.compute_entropy_from_logits(logits_rmpad)
                                if not self.config.entropy_checkpointing
                                else torch.utils.checkpoint.checkpoint(
                                    self.compute_entropy_from_logits, logits_rmpad
                                )
                            )
                        else:
                            # Entropy is metric/LR-scheduler only: no grad needed
                            with torch.no_grad():
                                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad.detach())
                    # sum_pi_squared also needs the original logits; disable inplace if needed
                    if calculate_sum_pi_squared:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm_pre_clip = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
            # Measure post-clip norm without modifying gradients.
            grad_norm_post_clip = self.actor_module.clip_grad_norm_(max_norm=float("inf"))
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm_pre_clip = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
            # Measure post-clip norm without modifying gradients.
            grad_norm_post_clip = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=float("inf"))
        else:
            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
            # Measure post-clip norm without modifying gradients.
            grad_norm_post_clip = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=float("inf"))

        if isinstance(grad_norm_pre_clip, DTensor):
            grad_norm_pre_clip = grad_norm_pre_clip.full_tensor()
        if isinstance(grad_norm_post_clip, DTensor):
            grad_norm_post_clip = grad_norm_post_clip.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm_pre_clip):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm_pre_clip}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm_pre_clip, grad_norm_post_clip

    def _compute_adam_diagnostics(self):
        """Compute Adam optimizer state diagnostics after optimizer.step().

        Reads exp_avg (m) and exp_avg_sq (v) from the optimizer state to
        compute effective step-size metrics.  The raw gradient g_t is still
        available in p.grad at this point (AdamW reads but does not zero it).

        Metrics returned (all scalars, all-reduced across FSDP shards):
          adam/v_rms          — RMS of sqrt(v) across all params (denominator scale)
          adam/v_mean         — mean of v across all params
          adam/m_rms          — RMS of m across all params (numerator scale)
          adam/update_norm    — ||m / (sqrt(v) + eps)||  (preconditioned update norm)
          adam/raw_grad_norm  — ||g_t||  (post-clip gradient used for the step)
          adam/update_to_grad_ratio — update_norm / (raw_grad_norm + 1e-12)
          adam/eta_eff        — <u, g> / (||g||^2 + 1e-12)  (effective LR on g direction)
          adam/cos_update_grad — cos(u, g)
          adam/v_max          — max element of v (tracks extreme denominator)
        """
        device = None
        sum_v = torch.zeros(1, dtype=torch.float64)
        sum_v_sqrt = torch.zeros(1, dtype=torch.float64)
        sum_m_sq = torch.zeros(1, dtype=torch.float64)
        sum_u_sq = torch.zeros(1, dtype=torch.float64)
        sum_g_sq = torch.zeros(1, dtype=torch.float64)
        sum_ug = torch.zeros(1, dtype=torch.float64)
        max_v = torch.zeros(1, dtype=torch.float64)
        n_elements = torch.zeros(1, dtype=torch.float64)
        has_state = False

        eps = 1e-8
        for pg in self.actor_optimizer.param_groups:
            eps = pg.get("eps", 1e-8)
            break

        for pg in self.actor_optimizer.param_groups:
            for p in pg["params"]:
                if p not in self.actor_optimizer.state:
                    continue
                state = self.actor_optimizer.state[p]
                if "exp_avg_sq" not in state:
                    continue
                has_state = True
                v = state["exp_avg_sq"]
                m = state["exp_avg"]
                if device is None:
                    device = v.device
                    sum_v = sum_v.to(device)
                    sum_v_sqrt = sum_v_sqrt.to(device)
                    sum_m_sq = sum_m_sq.to(device)
                    sum_u_sq = sum_u_sq.to(device)
                    sum_g_sq = sum_g_sq.to(device)
                    sum_ug = sum_ug.to(device)
                    max_v = max_v.to(device)
                    n_elements = n_elements.to(device)

                v_flat = v.detach().to(torch.float64).flatten()
                m_flat = m.detach().to(torch.float64).flatten()
                denom = v_flat.sqrt() + eps
                u_flat = m_flat / denom

                sum_v[0] += v_flat.sum()
                sum_v_sqrt[0] += denom.sum()
                sum_m_sq[0] += (m_flat * m_flat).sum()
                sum_u_sq[0] += (u_flat * u_flat).sum()
                cur_max = v_flat.max()
                if cur_max > max_v[0]:
                    max_v[0] = cur_max
                n_elements[0] += v_flat.numel()

                g = p.grad
                if g is not None:
                    g_flat = g.detach().to(torch.float64).flatten()
                    sum_g_sq[0] += (g_flat * g_flat).sum()
                    sum_ug[0] += (u_flat * g_flat).sum()

        if not has_state:
            return {}

        if torch.distributed.is_initialized():
            for t in [sum_v, sum_v_sqrt, sum_m_sq, sum_u_sq, sum_g_sq, sum_ug, n_elements]:
                torch.distributed.all_reduce(t)
            torch.distributed.all_reduce(max_v, op=torch.distributed.ReduceOp.MAX)

        n = n_elements[0].item()
        if n == 0:
            return {}

        v_mean = (sum_v[0] / n).item()
        v_rms = ((sum_v[0] / n).sqrt()).item()
        m_rms = ((sum_m_sq[0] / n).sqrt()).item()
        update_norm = (sum_u_sq[0].sqrt()).item()
        raw_grad_norm = (sum_g_sq[0].sqrt()).item()
        dot_ug = sum_ug[0].item()

        update_to_grad_ratio = update_norm / (raw_grad_norm + 1e-12)
        eta_eff = dot_ug / (sum_g_sq[0].item() + 1e-12)
        cos_ug = dot_ug / (update_norm * raw_grad_norm + 1e-12)

        return {
            "adam/v_rms": v_rms,
            "adam/v_mean": v_mean,
            "adam/v_max": max_v[0].item(),
            "adam/m_rms": m_rms,
            "adam/update_norm": update_norm,
            "adam/raw_grad_norm": raw_grad_norm,
            "adam/update_to_grad_ratio": update_to_grad_ratio,
            "adam/eta_eff": eta_eff,
            "adam/cos_update_grad": cos_ug,
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        calculate_lm_head_grad_norm = self.config.get("calculate_lm_head_grad_norm", False)
        if calculate_lm_head_grad_norm and "response_mask" in data.batch:
            select_keys.append("response_mask")
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch and k not in select_keys]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared

        calculate_lm_head_grad_norm = self.config.get("calculate_lm_head_grad_norm", False)
        if calculate_lm_head_grad_norm and not self.use_fused_kernels:
            grad_norm_freq = self.config.get("lm_head_grad_norm_freq", 5)
            step_counter = data.meta_info.get("global_step", 0)
            if step_counter % grad_norm_freq == 0:
                grad_norms, sampled_indices = self.compute_lm_head_grad_norms(
                    data,
                    max_responses=self.config.get("lm_head_grad_norm_max_responses", 64),
                )
                batch_size = log_probs.shape[0]
                full_norms = torch.full(
                    (batch_size,), -1.0, dtype=torch.float32, device=log_probs.device
                )
                if grad_norms.numel() > 0:
                    full_norms[sampled_indices] = grad_norms.float().to(full_norms.device)
                outputs["lm_head_grad_norm_sq"] = full_norms

        return outputs

    @torch.no_grad()
    def compute_lm_head_grad_norms(self, data: DataProto, max_responses: int = 64):
        """Compute per-response exact lm_head gradient norms for a subsample.

        Runs a separate forward pass with output_hidden_states=True on a small
        subsample to obtain pre-lm_head hidden states and logits, then computes
        ||∂S_b/∂W_lm_head||²_F per response using the analytical Gram-matrix
        formula (no orthogonality assumption).

        Only works when use_fused_kernels=False (standard HF forward returns logits).
        Returns empty tensors if fused kernels are active.

        Args:
            data: DataProto with input_ids, attention_mask, position_ids, responses,
                  response_mask.  meta_info must contain 'temperature'.
            max_responses: max number of responses to process.

        Returns:
            (grad_norm_sq, indices): grad_norm_sq is (n,) float64, indices is (n,) long
            indicating which batch rows were sampled.
        """
        empty = (torch.tensor([], dtype=torch.float64), torch.tensor([], dtype=torch.long))
        if self.use_fused_kernels:
            return empty

        from verl.trainer.ppo.lm_head_grad_norm import compute_response_lm_head_grad_norms

        self.actor_module.eval()
        device = get_device_id()
        temperature = data.meta_info["temperature"]

        batch_size = data.batch["input_ids"].shape[0]
        num_to_process = min(max_responses, batch_size)
        indices = torch.randperm(batch_size)[:num_to_process].sort().values
        response_length = data.batch["responses"].shape[-1]

        all_norms = []
        micro_batch_size = 2

        for mb_start in range(0, num_to_process, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, num_to_process)
            mb_indices = indices[mb_start:mb_end]

            input_ids = data.batch["input_ids"][mb_indices].to(device)
            attention_mask = data.batch["attention_mask"][mb_indices].to(device)
            position_ids = data.batch["position_ids"][mb_indices].to(device)
            responses = data.batch["responses"][mb_indices].to(device)
            response_mask = data.batch["response_mask"][mb_indices].to(device)

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )

            last_hidden = output.hidden_states[-1]
            logits_full = output.logits

            hs_resp = last_hidden[:, -response_length - 1 : -1, :]
            logits_resp = logits_full[:, -response_length - 1 : -1, :]

            norms = compute_response_lm_head_grad_norms(
                hidden_states=hs_resp,
                logits=logits_resp,
                response_token_ids=responses,
                response_mask=response_mask,
                temperature=temperature,
                token_chunk_size=512,
            )
            all_norms.append(norms)

        result = torch.cat(all_norms, dim=0)
        return result, indices

    @torch.no_grad()
    def compute_pos_neg_grad_decomposition(self, data: DataProto, max_responses: int = 64):
        """Positive/negative advantage lm_head gradient decomposition.

        Computes g+ = Σ_{A>0} A·G_b and g- = Σ_{A<0} A·G_b where G_b is the
        per-response exact lm_head score-function gradient, then reports norms,
        cosine similarity, and per-success-rate-bucket metrics.

        Subsamples at the prompt-group level so bucketing by success rate is valid.
        Must be called when advantages are available (i.e. during update_policy).

        Args:
            data: DataProto with input_ids, attention_mask, position_ids,
                  responses, response_mask, advantages.
            max_responses: max number of responses to process.

        Returns:
            Dict of scalar metrics under the ``grad_decomp/`` prefix.
        """
        if self.use_fused_kernels:
            return {}

        from verl.trainer.ppo.lm_head_grad_norm import compute_single_response_lm_head_grad

        self.actor_module.eval()
        device = get_device_id()
        temperature = data.meta_info["temperature"]
        rollout_n = self.config.rollout_n

        batch_size = data.batch["input_ids"].shape[0]
        response_length = data.batch["responses"].shape[-1]

        advantages_raw = data.batch["advantages"]
        response_mask_all = data.batch["response_mask"]
        adv_per_resp = (advantages_raw * response_mask_all).sum(-1) / response_mask_all.sum(-1).clamp(min=1)

        n_prompts = batch_size // rollout_n
        max_groups = max(1, max_responses // rollout_n)
        n_sample_groups = min(max_groups, n_prompts)
        group_perm = torch.randperm(n_prompts)[:n_sample_groups]

        indices = []
        for g_idx in group_perm:
            start = g_idx.item() * rollout_n
            indices.extend(range(start, start + rollout_n))
        indices = torch.tensor(indices, dtype=torch.long)

        adv_grouped = adv_per_resp.reshape(n_prompts, rollout_n)
        success_rate_all = (adv_grouped > 0).float().mean(dim=1)

        prompt_bucket = torch.full((n_prompts,), -1, dtype=torch.long)
        mixed = (success_rate_all > 0) & (success_rate_all < 1)
        prompt_bucket[mixed & (success_rate_all < 0.5)] = 0
        prompt_bucket[mixed & (success_rate_all >= 0.5)] = 1

        g_accum = {}
        counts = {}
        token_counts = {}
        for bk in [0, 1]:
            g_accum[bk] = {"pos": None, "neg": None}
            counts[bk] = {"pos": 0, "neg": 0}
            token_counts[bk] = {"pos": 0, "neg": 0}
        n_zero = 0

        has_old_log_probs = "old_log_probs" in data.batch
        advantages_token = data.batch["advantages"]

        group_names = ["low_pos", "low_neg", "high_pos", "high_neg", "pos", "neg"]
        w_abs_sums = {g: 0.0 for g in group_names}
        w_sq_sums = {g: 0.0 for g in group_names}
        w_token_counts = {g: 0 for g in group_names}
        ratio_collections = {g: [] for g in group_names}
        nonzero_w_tokens = {g: 0 for g in group_names}

        # ── Orthogonality control pre-assignments ──────────────────────────────────
        n_sel = len(indices)
        selected_adv_ctrl = adv_per_resp[indices]
        n_rand_splits = 3

        # Control 1: K independent random split labels (each response → side 0 or 1)
        rand_labels = [torch.randint(0, 2, (n_sel,)) for _ in range(n_rand_splits)]

        # Control 2: same-sign half-split labels (assign each pos/neg resp to half 0 or 1)
        pos_ctrl_idx = (selected_adv_ctrl > 1e-8).nonzero(as_tuple=True)[0]
        neg_ctrl_idx = (selected_adv_ctrl < -1e-8).nonzero(as_tuple=True)[0]
        same_sign_half = torch.full((n_sel,), -1, dtype=torch.long)
        if len(pos_ctrl_idx) >= 2:
            same_sign_half[pos_ctrl_idx] = torch.randperm(len(pos_ctrl_idx)) % 2
        if len(neg_ctrl_idx) >= 2:
            same_sign_half[neg_ctrl_idx] = torch.randperm(len(neg_ctrl_idx)) % 2

        # Control 3: prompt-level (up to 5 mixed prompts from the sampled set)
        sampled_mixed_pids = [g.item() for g in group_perm if mixed[g].item()]
        n_ctrl_prompts = min(5, len(sampled_mixed_pids))
        ctrl_prompt_ids = set(sampled_mixed_pids[:n_ctrl_prompts])

        # Accumulators for controls
        g_rand = [[None, None] for _ in range(n_rand_splits)]
        g_pos_half = [None, None]
        g_neg_half = [None, None]
        g_prompt_ctrl = {pid: {"pos": None, "neg": None} for pid in ctrl_prompt_ids}
        # ────────────────────────────────────────────────────────────────────────────

        micro_batch_size = 2
        for mb_start in range(0, len(indices), micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, len(indices))
            mb_idx = indices[mb_start:mb_end]

            input_ids = data.batch["input_ids"][mb_idx].to(device)
            attention_mask = data.batch["attention_mask"][mb_idx].to(device)
            position_ids = data.batch["position_ids"][mb_idx].to(device)
            responses = data.batch["responses"][mb_idx].to(device)
            response_mask = data.batch["response_mask"][mb_idx].to(device)
            mb_adv = adv_per_resp[mb_idx]

            if has_old_log_probs:
                mb_old_lp = data.batch["old_log_probs"][mb_idx].to(device)
            mb_adv_token = advantages_token[mb_idx].to(device)

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )

            last_hidden = output.hidden_states[-1]
            logits_full = output.logits
            hs_resp = last_hidden[:, -response_length - 1 : -1, :]
            logits_resp = logits_full[:, -response_length - 1 : -1, :]

            if has_old_log_probs:
                log_probs_resp = logprobs_from_logits(
                    logits_full[:, -response_length - 1 : -1, :], responses
                )

            for local_i in range(mb_end - mb_start):
                mask_i = response_mask[local_i].bool()
                T_valid = mask_i.sum().item()
                if T_valid == 0:
                    continue

                h_i = hs_resp[local_i][mask_i].float()
                logits_i = logits_resp[local_i][mask_i].float() / temperature
                y_i = responses[local_i][mask_i]

                G_i = compute_single_response_lm_head_grad(h_i, logits_i, y_i, token_chunk_size=256)

                A_i = mb_adv[local_i].item()
                global_resp_idx = mb_idx[local_i].item()
                prompt_id = global_resp_idx // rollout_n
                bucket = prompt_bucket[prompt_id].item()

                if abs(A_i) < 1e-8:
                    n_zero += 1
                    del G_i
                    continue

                sign_key = "pos" if A_i > 0 else "neg"
                bucket_sign = f"{'low' if bucket == 0 else 'high'}_{sign_key}" if bucket >= 0 else None

                if bucket >= 0:
                    if g_accum[bucket][sign_key] is None:
                        g_accum[bucket][sign_key] = torch.zeros_like(G_i)
                    g_accum[bucket][sign_key].add_(G_i, alpha=A_i)
                    counts[bucket][sign_key] += 1
                    token_counts[bucket][sign_key] += T_valid

                # ── Control accumulations (all non-zero-adv responses) ────────────
                sel_idx = mb_start + local_i

                # 4.1 Random split: K independent splits
                for _k in range(n_rand_splits):
                    _side = rand_labels[_k][sel_idx].item()
                    if g_rand[_k][_side] is None:
                        g_rand[_k][_side] = torch.zeros_like(G_i)
                    g_rand[_k][_side].add_(G_i, alpha=A_i)

                # 4.2 Same-sign half-split
                _half = same_sign_half[sel_idx].item()
                if _half >= 0:
                    _half_buf = g_pos_half if sign_key == "pos" else g_neg_half
                    if _half_buf[_half] is None:
                        _half_buf[_half] = torch.zeros_like(G_i)
                    _half_buf[_half].add_(G_i, alpha=A_i)

                # 4.3 Prompt-level
                if prompt_id in ctrl_prompt_ids:
                    _gd = g_prompt_ctrl[prompt_id]
                    if _gd[sign_key] is None:
                        _gd[sign_key] = torch.zeros_like(G_i)
                    _gd[sign_key].add_(G_i, alpha=A_i)
                # ─────────────────────────────────────────────────────────────────

                del G_i

                adv_tok = mb_adv_token[local_i][mask_i]
                w_tok = -adv_tok.float()
                abs_w = w_tok.abs()
                w_abs_sum = abs_w.sum().item()
                w_sq_sum = (w_tok * w_tok).sum().item()
                nonzero_count = (abs_w > 1e-8).sum().item()

                w_abs_sums[sign_key] += w_abs_sum
                w_sq_sums[sign_key] += w_sq_sum
                w_token_counts[sign_key] += T_valid
                nonzero_w_tokens[sign_key] += nonzero_count
                if bucket_sign is not None:
                    w_abs_sums[bucket_sign] += w_abs_sum
                    w_sq_sums[bucket_sign] += w_sq_sum
                    w_token_counts[bucket_sign] += T_valid
                    nonzero_w_tokens[bucket_sign] += nonzero_count

                if has_old_log_probs:
                    lp_cur = log_probs_resp[local_i][mask_i].detach().float()
                    lp_old = mb_old_lp[local_i][mask_i].float()
                    ratio_tok = torch.exp(torch.clamp(lp_cur - lp_old, -20.0, 20.0))
                    ratio_collections[sign_key].append(ratio_tok.cpu())
                    if bucket_sign is not None:
                        ratio_collections[bucket_sign].append(ratio_tok.cpu())

        self.actor_module.train()

        metrics = {}

        g_lp = g_accum[0]["pos"]
        g_ln = g_accum[0]["neg"]
        g_hp = g_accum[1]["pos"]
        g_hn = g_accum[1]["neg"]
        atoms = {"low_pos": g_lp, "low_neg": g_ln, "high_pos": g_hp, "high_neg": g_hn}

        def _norm_sq(g):
            return (g * g).sum().item() if g is not None else 0.0

        def _dot(ga, gb):
            if ga is not None and gb is not None:
                return (ga * gb).sum().item()
            return 0.0

        nsq = {k: _norm_sq(v) for k, v in atoms.items()}

        dots = {}
        atom_keys = list(atoms.keys())
        for i in range(len(atom_keys)):
            for j in range(i + 1, len(atom_keys)):
                ka, kb = atom_keys[i], atom_keys[j]
                dots[(ka, kb)] = _dot(atoms[ka], atoms[kb])

        def _get_dot(ka, kb):
            if ka == kb:
                return nsq[ka]
            return dots.get((ka, kb), dots.get((kb, ka), 0.0))

        def _composite_norm_sq(members):
            total = sum(nsq.get(m, 0.0) for m in members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    total += 2.0 * _get_dot(members[i], members[j])
            return total

        def _composite_dot(members_a, members_b):
            total = 0.0
            for ma in members_a:
                for mb in members_b:
                    total += _get_dot(ma, mb)
            return total

        composites = {
            "pos": ["low_pos", "high_pos"],
            "neg": ["low_neg", "high_neg"],
            "pos_neg": ["low_pos", "high_pos", "low_neg", "high_neg"],
            "high_pos_neg": ["high_pos", "high_neg"],
            "low_pos_neg": ["low_pos", "low_neg"],
        }
        comp_nsq = {k: _composite_norm_sq(v) for k, v in composites.items()}

        all_nsq = {**nsq, **comp_nsq}
        eps = 1e-30

        for name, val in all_nsq.items():
            metrics[f"lm_grad/norm_{name}"] = val ** 0.5

        def _cos_cancel(name_a, name_b, members_a, members_b, prefix):
            nsq_a = all_nsq.get(name_a, _composite_norm_sq(members_a) if members_a else 0.0)
            nsq_b = all_nsq.get(name_b, _composite_norm_sq(members_b) if members_b else 0.0)
            dot_ab = _composite_dot(members_a, members_b)
            nsq_ab = nsq_a + nsq_b + 2.0 * dot_ab
            norm_a = nsq_a ** 0.5
            norm_b = nsq_b ** 0.5
            norm_ab = nsq_ab ** 0.5

            cos_val = dot_ab / max(norm_a * norm_b, eps)
            cancel_val = 1.0 - norm_ab / max(norm_a + norm_b, eps)

            metrics[f"{prefix}/cos"] = cos_val
            metrics[f"{prefix}/cancel"] = cancel_val
            metrics[f"{prefix}/norm_a"] = norm_a
            metrics[f"{prefix}/norm_b"] = norm_b
            metrics[f"{prefix}/norm_sum"] = norm_ab
            if norm_b > 1.0 and norm_a > 1.0:
                metrics[f"{prefix}/norm_ratio"] = norm_a / norm_b
            else:
                metrics[f"{prefix}/norm_ratio"] = float("nan")

        _cos_cancel("pos", "neg", composites["pos"], composites["neg"], "lm_grad/pos_vs_neg")
        _cos_cancel("high_pos", "high_neg", ["high_pos"], ["high_neg"], "lm_grad/high_pos_vs_neg")
        _cos_cancel("low_pos", "low_neg", ["low_pos"], ["low_neg"], "lm_grad/low_pos_vs_neg")

        composites["high_mixed"] = ["high_pos", "high_neg"]
        composites["low_mixed"] = ["low_pos", "low_neg"]
        all_nsq["high_mixed"] = _composite_norm_sq(composites["high_mixed"])
        all_nsq["low_mixed"] = _composite_norm_sq(composites["low_mixed"])
        metrics["lm_grad/norm_high_mixed"] = all_nsq["high_mixed"] ** 0.5
        metrics["lm_grad/norm_low_mixed"] = all_nsq["low_mixed"] ** 0.5

        _cos_cancel("high_mixed", "low_mixed",
                     composites["high_mixed"], composites["low_mixed"],
                     "lm_grad/high_mixed_vs_low_mixed")

        # ── Orthogonality controls ───────────────────────────────────────────────
        def _ctrl_cos(ga, gb):
            if ga is None or gb is None:
                return float("nan")
            nsq_a = _norm_sq(ga)
            nsq_b = _norm_sq(gb)
            if nsq_a < eps or nsq_b < eps:
                return float("nan")
            return _dot(ga, gb) / (nsq_a ** 0.5 * nsq_b ** 0.5)

        # 4.1 Random split baseline (K independent splits → mean / std / p05 / p95 / z-score)
        rand_cos_vals = [_ctrl_cos(g_rand[k][0], g_rand[k][1]) for k in range(n_rand_splits)]
        for k, v in enumerate(rand_cos_vals):
            metrics[f"lm_grad/ctrl/random_split_cos_{k}"] = v
        valid_rand = [v for v in rand_cos_vals if v == v]
        if valid_rand:
            mu_rand = sum(valid_rand) / len(valid_rand)
            metrics["lm_grad/ctrl/random_split_cos_mean"] = mu_rand
            if len(valid_rand) >= 2:
                std_rand = (sum((v - mu_rand) ** 2 for v in valid_rand) / len(valid_rand)) ** 0.5
                sorted_rand = sorted(valid_rand)
                metrics["lm_grad/ctrl/random_split_cos_std"] = std_rand
                metrics["lm_grad/ctrl/random_split_cos_p05"] = sorted_rand[max(0, int(0.05 * len(sorted_rand)))]
                metrics["lm_grad/ctrl/random_split_cos_p95"] = sorted_rand[min(len(sorted_rand) - 1, int(0.95 * len(sorted_rand)))]
                pn_cos = metrics.get("lm_grad/pos_vs_neg/cos", float("nan"))
                metrics["lm_grad/ctrl/zscore_posneg_vs_random"] = (
                    (pn_cos - mu_rand) / (std_rand + eps) if pn_cos == pn_cos else float("nan")
                )
            else:
                for key in ("std", "p05", "p95"):
                    metrics[f"lm_grad/ctrl/random_split_cos_{key}"] = float("nan")
                metrics["lm_grad/ctrl/zscore_posneg_vs_random"] = float("nan")
        else:
            for key in ("mean", "std", "p05", "p95"):
                metrics[f"lm_grad/ctrl/random_split_cos_{key}"] = float("nan")
            metrics["lm_grad/ctrl/zscore_posneg_vs_random"] = float("nan")

        # 4.2 Same-sign half-split: within-pos and within-neg coherence
        metrics["lm_grad/ctrl/pos_half_split_cos"] = _ctrl_cos(g_pos_half[0], g_pos_half[1])
        metrics["lm_grad/ctrl/neg_half_split_cos"] = _ctrl_cos(g_neg_half[0], g_neg_half[1])

        # 4.3 Prompt-level pos/neg cosine (mean ± std over sampled mixed prompts)
        prompt_ctrl_cos = []
        for _gd in g_prompt_ctrl.values():
            c = _ctrl_cos(_gd["pos"], _gd["neg"])
            if c == c:
                prompt_ctrl_cos.append(c)
        if prompt_ctrl_cos:
            mu_p = sum(prompt_ctrl_cos) / len(prompt_ctrl_cos)
            metrics["lm_grad/ctrl/prompt_level_cos_mean"] = mu_p
            metrics["lm_grad/ctrl/prompt_level_n"] = float(len(prompt_ctrl_cos))
            metrics["lm_grad/ctrl/prompt_level_cos_std"] = (
                (sum((c - mu_p) ** 2 for c in prompt_ctrl_cos) / len(prompt_ctrl_cos)) ** 0.5
                if len(prompt_ctrl_cos) >= 2 else float("nan")
            )
        else:
            metrics["lm_grad/ctrl/prompt_level_cos_mean"] = float("nan")
            metrics["lm_grad/ctrl/prompt_level_n"] = 0.0
            metrics["lm_grad/ctrl/prompt_level_cos_std"] = float("nan")

        # Cleanup ctrl accumulators
        for k in range(n_rand_splits):
            for side in [0, 1]:
                if g_rand[k][side] is not None:
                    del g_rand[k][side]
        for _buf in [g_pos_half, g_neg_half]:
            for side in [0, 1]:
                if _buf[side] is not None:
                    del _buf[side]
        for _gd in g_prompt_ctrl.values():
            for sk in ("pos", "neg"):
                if _gd[sk] is not None:
                    del _gd[sk]
        # ────────────────────────────────────────────────────────────────────────────

        atom_pairs = [
            ("high_pos", "high_neg"), ("low_pos", "low_neg"),
            ("high_pos", "low_pos"), ("high_neg", "low_neg"),
            ("high_pos", "low_neg"), ("high_neg", "low_pos"),
        ]
        for ka, kb in atom_pairs:
            dot_val = _get_dot(ka, kb)
            na = all_nsq.get(ka, 0.0) ** 0.5
            nb = all_nsq.get(kb, 0.0) ** 0.5
            cos_val = dot_val / max(na * nb, eps)
            metrics[f"lm_grad/cos_{ka}_{kb}"] = cos_val

        norm_total = all_nsq["pos_neg"] ** 0.5
        norm_pos = all_nsq["pos"] ** 0.5
        norm_neg = all_nsq["neg"] ** 0.5

        dot_pos_total = _composite_dot(composites["pos"], composites["pos_neg"])
        dot_neg_total = _composite_dot(composites["neg"], composites["pos_neg"])
        metrics["lm_grad/cos_pos_posneg"] = dot_pos_total / max(norm_pos * norm_total, eps)
        metrics["lm_grad/cos_neg_posneg"] = dot_neg_total / max(norm_neg * norm_total, eps)

        nsq_total = all_nsq["pos_neg"]
        proj_groups = {
            "pos": composites["pos"], "neg": composites["neg"],
            "high_pos": ["high_pos"], "high_neg": ["high_neg"],
            "low_pos": ["low_pos"], "low_neg": ["low_neg"],
            "high_mixed": composites["high_mixed"],
            "low_mixed": composites["low_mixed"],
        }
        for pg_name, pg_members in proj_groups.items():
            dot_gc_total = _composite_dot(pg_members, composites["pos_neg"])
            proj_coeff = dot_gc_total / max(nsq_total, eps)
            metrics[f"lm_grad/proj_{pg_name}_on_posneg"] = proj_coeff

        n_pos_global = counts[0]["pos"] + counts[1]["pos"]
        n_neg_global = counts[0]["neg"] + counts[1]["neg"]
        tok_pos_global = token_counts[0]["pos"] + token_counts[1]["pos"]
        tok_neg_global = token_counts[0]["neg"] + token_counts[1]["neg"]

        if tok_pos_global > 0 and tok_neg_global > 0:
            scale_pos = tok_pos_global + tok_neg_global
            mean_factor_pos = scale_pos / tok_pos_global
            mean_factor_neg = scale_pos / tok_neg_global
            metrics["lm_grad/norm_pos_mean"] = norm_pos * mean_factor_pos
            metrics["lm_grad/norm_neg_mean"] = norm_neg * mean_factor_neg

        metrics["lm_grad/n_pos"] = n_pos_global
        metrics["lm_grad/n_neg"] = n_neg_global
        metrics["lm_grad/n_zero"] = n_zero
        metrics["lm_grad/n_prompts_sampled"] = n_sample_groups
        n_mixed = mixed[group_perm].sum().item()
        metrics["lm_grad/n_prompts_mixed"] = n_mixed
        metrics["lm_grad/frac_uninformative"] = 1.0 - n_mixed / max(n_sample_groups, 1)

        for bk_name, bk_id in [("low", 0), ("high", 1)]:
            metrics[f"lm_grad/n_{bk_name}_pos"] = counts[bk_id]["pos"]
            metrics[f"lm_grad/n_{bk_name}_neg"] = counts[bk_id]["neg"]
            metrics[f"lm_grad/tok_{bk_name}_pos"] = token_counts[bk_id]["pos"]
            metrics[f"lm_grad/tok_{bk_name}_neg"] = token_counts[bk_id]["neg"]

        n_empty = sum(1 for k, v in nsq.items() if atoms[k] is None)
        metrics["lm_grad/sanity/num_empty_groups"] = n_empty

        rel_err_pn = abs(all_nsq["pos_neg"] - _composite_norm_sq(composites["pos_neg"])) / max(all_nsq["pos_neg"], eps)
        metrics["lm_grad/sanity/rel_err_pos_neg_union"] = rel_err_pn

        has_nan = any(v != v for v in metrics.values() if isinstance(v, float))
        metrics["lm_grad/sanity/has_nan"] = float(has_nan)

        clip_ratio_val = getattr(self.config, "clip_ratio", 0.2)
        for gname in group_names:
            tc = w_token_counts[gname]
            if tc > 0:
                metrics[f"w_stats/mean_abs_w_{gname}"] = w_abs_sums[gname] / tc
                metrics[f"w_stats/sum_w2_{gname}"] = w_sq_sums[gname]
                metrics[f"clip_stats/nonzero_w_frac_{gname}"] = nonzero_w_tokens[gname] / tc
            else:
                metrics[f"w_stats/mean_abs_w_{gname}"] = float("nan")
                metrics[f"w_stats/sum_w2_{gname}"] = 0.0
                metrics[f"clip_stats/nonzero_w_frac_{gname}"] = float("nan")

            ratios = ratio_collections[gname]
            if ratios:
                all_ratios = torch.cat(ratios)
                if all_ratios.numel() > 0:
                    sorted_r, _ = all_ratios.sort()
                    idx95 = min(int(0.95 * len(sorted_r)), len(sorted_r) - 1)
                    metrics[f"ratio_stats/p95_ratio_{gname}"] = sorted_r[idx95].item()
                    metrics[f"ratio_stats/mean_ratio_{gname}"] = all_ratios.mean().item()
                    clipped_frac = ((all_ratios > 1 + clip_ratio_val) | (all_ratios < 1 - clip_ratio_val)).float().mean().item()
                    metrics[f"ratio_stats/clip_frac_{gname}"] = clipped_frac
                else:
                    metrics[f"ratio_stats/p95_ratio_{gname}"] = float("nan")
                    metrics[f"ratio_stats/mean_ratio_{gname}"] = float("nan")
                    metrics[f"ratio_stats/clip_frac_{gname}"] = float("nan")
            else:
                metrics[f"ratio_stats/p95_ratio_{gname}"] = float("nan")
                metrics[f"ratio_stats/mean_ratio_{gname}"] = float("nan")
                metrics[f"ratio_stats/clip_frac_{gname}"] = float("nan")

        for bk in g_accum:
            for sign_key in list(g_accum[bk].keys()):
                v = g_accum[bk][sign_key]
                if v is not None:
                    del v
                g_accum[bk][sign_key] = None

        return metrics

    def compute_full_model_grad_decomposition(self, data: DataProto, max_responses: int = 64):
        """Route B: Full-model masked backward for gradient geometry verification.

        Does multiple backward passes with group-masked loss to compute
        full-model gradient norms per group. Uses the cosine-from-norms trick:
            cos(g_a, g_b) = (||g_{a+b}||^2 - ||g_a||^2 - ||g_b||^2) / (2*||g_a||*||g_b||)

        Groups computed: pos, neg, pos_neg, high_pos, high_neg, low_pos, low_neg,
                         high_pos_neg, low_pos_neg

        Returns:
            Dict of metrics under ``full_grad/`` prefix.
        """
        if self.use_fused_kernels:
            return {}

        self.actor_module.train()
        device = get_device_id()
        rollout_n = self.config.rollout_n

        batch_size = data.batch["input_ids"].shape[0]
        response_length = data.batch["responses"].shape[-1]

        advantages_raw = data.batch["advantages"]
        response_mask_all = data.batch["response_mask"]
        adv_per_resp = (advantages_raw * response_mask_all).sum(-1) / response_mask_all.sum(-1).clamp(min=1)

        n_prompts = batch_size // rollout_n
        max_groups = max(1, max_responses // rollout_n)
        n_sample_groups = min(max_groups, n_prompts)
        group_perm = torch.randperm(n_prompts)[:n_sample_groups]

        indices = []
        for g_idx in group_perm:
            start = g_idx.item() * rollout_n
            indices.extend(range(start, start + rollout_n))
        indices = torch.tensor(indices, dtype=torch.long)

        adv_grouped = adv_per_resp.reshape(n_prompts, rollout_n)
        success_rate_all = (adv_grouped > 0).float().mean(dim=1)
        prompt_bucket = torch.full((n_prompts,), -1, dtype=torch.long)
        mixed = (success_rate_all > 0) & (success_rate_all < 1)
        prompt_bucket[mixed & (success_rate_all < 0.5)] = 0
        prompt_bucket[mixed & (success_rate_all >= 0.5)] = 1

        resp_adv = adv_per_resp[indices]
        resp_prompt_id = torch.arange(len(indices)) // rollout_n
        resp_bucket = prompt_bucket[group_perm][resp_prompt_id]

        group_masks = {}
        pos_mask = resp_adv > 1e-8
        neg_mask = resp_adv < -1e-8
        low_mask = resp_bucket == 0
        high_mask = resp_bucket == 1
        mixed_mask = low_mask | high_mask

        group_masks["pos"] = pos_mask & mixed_mask
        group_masks["neg"] = neg_mask & mixed_mask
        group_masks["pos_neg"] = (pos_mask | neg_mask) & mixed_mask
        group_masks["high_pos"] = pos_mask & high_mask
        group_masks["high_neg"] = neg_mask & high_mask
        group_masks["high_pos_neg"] = (pos_mask | neg_mask) & high_mask
        group_masks["low_pos"] = pos_mask & low_mask
        group_masks["low_neg"] = neg_mask & low_mask
        group_masks["low_pos_neg"] = (pos_mask | neg_mask) & low_mask

        input_ids_sub = data.batch["input_ids"][indices].to(device)
        attention_mask_sub = data.batch["attention_mask"][indices].to(device)
        position_ids_sub = data.batch["position_ids"][indices].to(device)
        responses_sub = data.batch["responses"][indices].to(device)
        response_mask_sub = data.batch["response_mask"][indices].to(device)
        advantages_sub = data.batch["advantages"][indices].to(device)

        def _compute_group_grad_norm_sq(group_name):
            """Forward + backward for a group, return sum of squared gradients."""
            mask = group_masks[group_name]
            if mask.sum().item() == 0:
                return 0.0

            self.actor_module.zero_grad()

            resp_mask_group = torch.zeros_like(response_mask_sub)
            for i in range(len(mask)):
                if mask[i]:
                    resp_mask_group[i] = response_mask_sub[i]

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                output = self.actor_module(
                    input_ids=input_ids_sub,
                    attention_mask=attention_mask_sub,
                    position_ids=position_ids_sub,
                    use_cache=False,
                )

            logits_resp = output.logits[:, -response_length - 1:-1, :]
            log_probs = torch.log_softmax(logits_resp.float(), dim=-1)
            selected_lp = torch.gather(log_probs, 2, responses_sub.unsqueeze(-1)).squeeze(-1)

            token_mask = resp_mask_group.float()
            loss_per_token = -selected_lp * advantages_sub
            denom = token_mask.sum().clamp(min=1.0)
            loss = (loss_per_token * token_mask).sum() / denom

            loss.backward()

            norm_sq = 0.0
            for param in self.actor_module.parameters():
                if param.grad is not None:
                    norm_sq += (param.grad.detach().float() ** 2).sum().item()

            self.actor_module.zero_grad()
            return norm_sq

        nsq = {}
        needed_groups = ["pos", "neg", "pos_neg", "high_pos", "high_neg",
                         "high_pos_neg", "low_pos", "low_neg", "low_pos_neg"]

        for gname in needed_groups:
            nsq[gname] = _compute_group_grad_norm_sq(gname)

        self.actor_module.train()

        metrics = {}
        eps = 1e-30

        for gname in needed_groups:
            metrics[f"full_grad/norm_{gname}"] = nsq[gname] ** 0.5

        def _cos_from_norms(nsq_a, nsq_b, nsq_ab, prefix):
            norm_a = nsq_a ** 0.5
            norm_b = nsq_b ** 0.5
            dot_ab = (nsq_ab - nsq_a - nsq_b) / 2.0
            cos_val = dot_ab / max(norm_a * norm_b, eps)
            metrics[f"{prefix}/cos"] = cos_val
            if norm_a > 1.0 and norm_b > 1.0:
                metrics[f"{prefix}/norm_ratio"] = norm_a / norm_b

        _cos_from_norms(nsq["pos"], nsq["neg"], nsq["pos_neg"], "full_grad/pos_vs_neg")
        _cos_from_norms(nsq["high_pos"], nsq["high_neg"], nsq["high_pos_neg"], "full_grad/high_pos_vs_neg")
        _cos_from_norms(nsq["low_pos"], nsq["low_neg"], nsq["low_pos_neg"], "full_grad/low_pos_vs_neg")

        nsq_high_mixed = nsq["high_pos_neg"]
        nsq_low_mixed = nsq["low_pos_neg"]
        _cos_from_norms(nsq_high_mixed, nsq_low_mixed, nsq["pos_neg"], "full_grad/high_mixed_vs_low_mixed")

        norm_total_sq = nsq["pos_neg"]
        norm_total = norm_total_sq ** 0.5

        metrics["full_grad/n_groups"] = len(needed_groups)
        metrics["full_grad/n_responses"] = len(indices)

        return metrics

    def _update_policy_signal_fraction(self, data: DataProto, temperature: float, pad_token_id: int) -> dict:
        """Signal-fraction adaptive LR: two backward passes on A1/A2 halves.

        Every step:
          1. Split batch at GRPO-group level → A1 (50%) and A2 (50%)
          2. Backward on A1 → ĝ_A1; backward on A2 → ĝ_A2
          3. r̂_t = dot(ĝ_A1, ĝ_A2) / ((||ĝ_A1||² + ||ĝ_A2||²) / 2), clamped to [r_min, 1]
          4. ĝ_upd = (ĝ_A1 + ĝ_A2) / 2;  α_t = c_t · r̂_t (set on optimizer)
          5. Optimizer step

        Every K calibration steps (additionally):
          6. Hold out C portion (calib_frac of groups) from A1/A2
          7. Backward on C at θ_t → ĝ_C,  L_C(θ_t)
          8. p_t = α_t · dot(ĝ_C, ĝ_upd);  optimizer step
          9. Forward-only on C at θ_{t+1} → L_C(θ_{t+1})
          10. a_t = L_C(θ_t) - L_C(θ_{t+1});  φ_t = a_t / (p_t + ε)
          11. Update c_t via controller if dot(ĝ_C, ĝ_upd) > 0
        """
        self.actor_module.train()

        # --- Config ---
        rollout_n = self.config.rollout_n
        micro_bsz = self.config.ppo_micro_batch_size_per_gpu  # None when use_dynamic_bsz=True
        calib_freq = self.config.optim.get("signal_fraction_calib_freq", 5)
        calib_frac = self.config.optim.get("signal_fraction_calib_frac", 0.25)

        step_count = getattr(self, "_sf_step_count", 0)
        self._sf_step_count = step_count + 1
        is_calibration = (step_count % calib_freq == 0)

        # --- Group-level split ---
        local_bsz = data.batch["input_ids"].shape[0]
        n_groups = max(1, local_bsz // rollout_n)

        if is_calibration:
            calib_groups = max(0, round(n_groups * calib_frac))  # 0 when calib_frac=0 (disables calibration)
            calib_groups = max(0, min(calib_groups, n_groups - 2))  # at least 1 group each for A1, A2
        else:
            calib_groups = 0

        update_groups = n_groups - calib_groups
        a1_groups = update_groups // 2
        a2_groups = update_groups - a1_groups

        a1_size = a1_groups * rollout_n
        a2_size = a2_groups * rollout_n

        data_A1 = data[:a1_size]
        data_A2 = data[a1_size:a1_size + a2_size]
        if is_calibration and calib_groups > 0:
            data_C = data[a1_size + a2_size:]
        else:
            data_C = None

        eps = 1e-10

        policy_loss_fn = get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))
        loss_agg_mode = self.config.loss_agg_mode

        # ------------------------------------------------------------------ #
        # Helper: build micro-batch list and per-mb scale factors.
        # With use_dynamic_bsz: sort by seqlen, scale = mb_size / total_size
        #                        (matches standard update_policy path).
        # Without            : split by fixed micro_bsz, scale = 1 / n_mb.
        # ------------------------------------------------------------------ #
        def _make_mb_list(split_data):
            total_size = split_data.batch["input_ids"].shape[0]
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                mb_list, _ = prepare_dynamic_batch(split_data, max_token_len=max_token_len)
                scales = [mb.batch["response_mask"].shape[0] / max(1, total_size) for mb in mb_list]
            else:
                mb_list = split_data.split(micro_bsz) if micro_bsz is not None else [split_data]
                n_mb = max(1, len(mb_list))
                scales = [1.0 / n_mb] * len(mb_list)
            return mb_list, scales

        # ------------------------------------------------------------------ #
        # Helper: accumulate gradients over micro-batches of a data split.
        # Returns the average policy loss (for monitoring).
        # ------------------------------------------------------------------ #
        def _backward_split(split_data, collect_log_probs=False):
            mb_list, scales = _make_mb_list(split_data)
            total_pg_loss = 0.0
            saved_log_probs = [] if collect_log_probs else None
            for mb, scale in zip(mb_list, scales):
                mb = mb.to(get_device_id())
                model_inputs = {**mb.batch, **mb.non_tensor_batch, "pad_token_id": pad_token_id}
                response_mask = model_inputs["response_mask"]
                advantages = model_inputs["advantages"]

                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    entropy_in_loss=False,
                )
                log_prob = outputs["log_probs"]
                old_log_prob = log_prob.detach()
                if collect_log_probs:
                    saved_log_probs.append(old_log_prob)

                pg_loss, _ = policy_loss_fn(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    config=self.config,
                )
                (pg_loss * scale).backward()
                total_pg_loss += pg_loss.detach().item()
            if collect_log_probs:
                return total_pg_loss / len(mb_list), saved_log_probs
            return total_pg_loss / len(mb_list)

        # ------------------------------------------------------------------ #
        # Helper: forward-only (no grad), returns average policy loss scalar.
        # ------------------------------------------------------------------ #
        def _forward_loss_nograd(split_data, old_log_probs_list=None):
            mb_list, _ = _make_mb_list(split_data)
            total_pg_loss = 0.0
            with torch.no_grad():
                for i, mb in enumerate(mb_list):
                    mb = mb.to(get_device_id())
                    model_inputs = {**mb.batch, **mb.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        entropy_in_loss=False,
                    )
                    log_prob = outputs["log_probs"]
                    # Use θ_t log_probs (saved during C backward) as reference so that
                    # ratio = π_θ_{t+1} / π_θt, making L_C_new actually depend on θ_{t+1}.
                    # This ensures a_t = L_C_old - L_C_new ≈ p_t (first-order) and
                    # φ_t = a_t / p_t is a valid calibration signal.
                    if old_log_probs_list is not None:
                        old_log_prob = old_log_probs_list[i].to(log_prob.device)
                    else:
                        old_log_prob = log_prob.detach()

                    pg_loss, _ = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )
                    total_pg_loss += pg_loss.detach().item()
            return total_pg_loss / len(mb_list)

        # ---- helper: list of current .grad tensors for all trainable params ----
        def _get_grads():
            return [
                p.grad.data.clone() if p.grad is not None else p.data.new_zeros(p.shape)
                for p in self.actor_module.parameters()
                if p.requires_grad
            ]

        params_with_grad = [p for p in self.actor_module.parameters() if p.requires_grad]

        # ================================================================== #
        # Pass 1: backward on A1
        # ================================================================== #
        self.actor_optimizer.zero_grad()
        pg_loss_A1 = _backward_split(data_A1)
        grad_A1 = _get_grads()

        # ================================================================== #
        # Pass 2: backward on A2
        # ================================================================== #
        self.actor_optimizer.zero_grad()
        pg_loss_A2 = _backward_split(data_A2)
        # p.grad now holds ĝ_A2 for each param

        # ================================================================== #
        # Compute r̂_t via local dot products then all_reduce
        # (works correctly for FSDP: shard-local dots are summed globally)
        # ================================================================== #
        local_stats = torch.zeros(3, device=get_device_id(), dtype=torch.float64)
        for g1, p in zip(grad_A1, params_with_grad):
            g2 = p.grad.data if p.grad is not None else p.data.new_zeros(p.shape)
            g1f = g1.to(torch.float64).flatten()
            g2f = g2.to(torch.float64).flatten()
            local_stats[0] += torch.dot(g1f, g2f)   # numerator: ĝ_A1 · ĝ_A2
            local_stats[1] += torch.dot(g1f, g1f)   # ||ĝ_A1||²
            local_stats[2] += torch.dot(g2f, g2f)   # ||ĝ_A2||²
        torch.distributed.all_reduce(local_stats)

        g_dot = local_stats[0].item()
        g_A1_norm_sq = local_stats[1].item()
        g_A2_norm_sq = local_stats[2].item()
        denom = (g_A1_norm_sq + g_A2_norm_sq) / 2.0
        r_hat_raw = g_dot / (denom + eps)

        # ================================================================== #
        # Build ĝ_upd = (ĝ_A1 + ĝ_A2) / 2  →  write into p.grad
        # Also accumulate ||ĝ_upd||² (shard-local) for g_rms_t
        # ================================================================== #
        upd_norm_sq_local = 0.0
        for g1, p in zip(grad_A1, params_with_grad):
            g2 = p.grad.data if p.grad is not None else p.data.new_zeros(p.shape)
            g_upd = (g1 + g2) / 2.0
            p.grad = g_upd
            upd_norm_sq_local += g_upd.to(torch.float64).norm().item() ** 2

        # All-reduce ||ĝ_upd||² across FSDP shards, then compute per-param RMS.
        # n_params is cached on self after the first step to avoid repeated all_reduce.
        upd_norm_sq_t = torch.tensor(upd_norm_sq_local, device=get_device_id(), dtype=torch.float64)
        torch.distributed.all_reduce(upd_norm_sq_t)
        if not hasattr(self, "_sf_n_params_global"):
            n_params_local = sum(p.numel() for p in params_with_grad)
            n_params_t = torch.tensor(n_params_local, device=get_device_id(), dtype=torch.int64)
            torch.distributed.all_reduce(n_params_t)
            self._sf_n_params_global = int(n_params_t.item())
        g_rms_t = float((upd_norm_sq_t.item() / max(self._sf_n_params_global, 1)) ** 0.5)

        # ================================================================== #
        # Set α_t via scheduler (handles warmup, handoff, EMA, fast-drop)
        # ================================================================== #
        sched = getattr(self, "actor_lr_scheduler", None)
        if sched is not None:
            alpha_t = sched.update_r_and_set_lr(
                g_dot,
                denom,
                g_rms_t,
                r_hat_raw,
                r_window_num=g_dot,
                r_window_den=denom,
            )
        else:
            alpha_t = self.actor_optimizer.param_groups[0]["lr"]

        # ================================================================== #
        # Calibration: compute φ_t and update c_t
        # ================================================================== #
        phi_t = None
        p_t_val = None
        a_t_val = None
        g_dot_C_upd = None

        if is_calibration and data_C is not None:
            # Save ĝ_upd before it is overwritten by the C backward pass
            g_upd = _get_grads()

            # Backward on C at θ_t; save log_probs for use as reference in forward pass
            self.actor_optimizer.zero_grad()
            L_C_old, log_probs_C_theta_t = _backward_split(data_C, collect_log_probs=True)
            grad_C = _get_grads()

            # p_t = α_t · dot(ĝ_C, ĝ_upd)  (all_reduce across FSDP shards)
            local_p = torch.zeros(1, device=get_device_id(), dtype=torch.float64)
            for gc, gu in zip(grad_C, g_upd):
                local_p[0] += torch.dot(
                    gc.to(torch.float64).flatten(),
                    gu.to(torch.float64).flatten(),
                )
            torch.distributed.all_reduce(local_p)
            g_dot_C_upd = local_p[0].item()
            p_t_val = alpha_t * g_dot_C_upd

            # Restore ĝ_upd into p.grad for optimizer step
            for gu, p in zip(g_upd, params_with_grad):
                p.grad = gu

        # ================================================================== #
        # Optimizer step  θ_t → θ_{t+1}
        # ================================================================== #
        grad_norm_pre_clip, grad_norm_post_clip = self._optimizer_step()
        adam_diag = self._compute_adam_diagnostics()

        # ================================================================== #
        # Calibration continued: forward on C at θ_{t+1}, compute φ_t
        # ================================================================== #
        if is_calibration and data_C is not None and p_t_val is not None:
            L_C_new = _forward_loss_nograd(data_C, old_log_probs_list=log_probs_C_theta_t)
            # Improvement in objective = decrease in loss
            a_t_val = L_C_old - L_C_new
            # φ_t = a_t / p_t  (guard: update direction must align with g_C)
            if g_dot_C_upd > 0:
                phi_t = a_t_val / (p_t_val + eps * (1.0 if p_t_val >= 0 else -1.0))
                if sched is not None:
                    sched._last_a_t = a_t_val
                    sched.update_ct(phi_t, p_t_val, g_dot_C_upd)

        self.actor_optimizer.zero_grad()

        # ================================================================== #
        # Metrics
        # ================================================================== #
        metrics = {
            "actor/pg_loss": (pg_loss_A1 + pg_loss_A2) / 2.0,
            "actor/g_A1_dot_A2": g_dot,
            "actor/g_dot_positive": float(g_dot > 0),
            "actor/g_norm_A1_sq": g_A1_norm_sq,
            "actor/g_norm_A2_sq": g_A2_norm_sq,
            "actor/g_rms": g_rms_t,
            "actor/grad_norm": grad_norm_pre_clip.detach().item(),
            "actor/grad_norm_pre_clip": grad_norm_pre_clip.detach().item(),
            "actor/grad_norm_post_clip": grad_norm_post_clip.detach().item(),
            "actor/is_calibration_step": float(is_calibration and data_C is not None),
        }
        if sched is not None:
            metrics.update(sched.get_signal_fraction_metrics())
        metrics.update(adam_diag)

        return reduce_metrics(metrics)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # Include token_level_rewards for advantage_variance entropy bonus (Bug 10 fix:
        # Bug 9 fix was non-functional because select() filtered this key before mini-batch
        # split, so mini_batch.batch.get("token_level_rewards") always fell back to advantages)
        if "token_level_rewards" in data.batch.keys():
            select_keys.append("token_level_rewards")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Positive/negative advantage gradient decomposition diagnostic
        decomp_metrics = {}
        if self.config.get("calculate_pos_neg_grad_decomp", False) and not self.use_fused_kernels:
            decomp_freq = self.config.get("pos_neg_grad_decomp_freq", 5)
            decomp_step = data.meta_info.get("global_step", 0)
            if decomp_step % decomp_freq == 0:
                decomp_metrics = self.compute_pos_neg_grad_decomposition(
                    data,
                    max_responses=self.config.get("pos_neg_grad_decomp_max_responses", 64),
                )

            full_grad_freq = self.config.get("full_model_grad_decomp_freq", 0)
            if full_grad_freq > 0 and decomp_step % full_grad_freq == 0:
                full_metrics = self.compute_full_model_grad_decomposition(
                    data,
                    max_responses=self.config.get("pos_neg_grad_decomp_max_responses", 64),
                )
                decomp_metrics.update(full_metrics)

        # Signal-fraction scheduler: redirect to dedicated implementation
        if self.config.optim.get("lr_scheduler_type", "constant") == "signal_fraction":
            sf_result = self._update_policy_signal_fraction(data, temperature=temperature, pad_token_id=pad_token_id)
            sf_result.update(decomp_metrics)
            return sf_result

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # Initialize _H0_entropy on first ever call (persists across training steps).
        if not hasattr(self, "_H0_entropy"):
            self._H0_entropy = None

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
            "actor/entropy_var_weight_mean": 0.0,
            "actor/entropy_var_weight_count": 0.0,
            "actor/entropy_bonus_suppressed": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # Pre-compute per-response advantage variance weights at mini-batch level.
                # Must happen before prepare_dynamic_batch: rearrange_micro_batches re-sorts
                # sequences by length, breaking prompt grouping. Computing here ensures all
                # rollout_n responses per prompt are together, then the weight tensor is
                # correctly indexed when split across micro-batches.
                if self.config.get("entropy_bonus_mode", "uniform") == "advantage_variance" and self.config.entropy_coeff != 0:
                    _rollout_n = self.config.rollout_n
                    _mb_size = mini_batch.batch["response_mask"].shape[0]
                    if _mb_size % _rollout_n == 0:
                        # Use raw token_level_rewards (before GRPO std-normalization) so that
                        # variance reflects genuine reward diversity within a prompt group.
                        # With normalized advantages A_i = (r_i - mean) / std, var(A_i) ≈ (n-1)/n
                        # for ALL groups with any reward diversity — normalization destroys the
                        # continuous signal we need. Raw reward variance = p*(1-p) for binary
                        # rewards, which correctly ranks groups by uncertainty.
                        _raw = mini_batch.batch.get("token_level_rewards", mini_batch.batch["advantages"])
                        _mask = mini_batch.batch["response_mask"]
                        _adv_per_resp = (_raw * _mask).sum(-1) / _mask.sum(-1).clamp(min=1)
                        _num_prompts = _mb_size // _rollout_n
                        _var_per_prompt = _adv_per_resp.view(_num_prompts, _rollout_n).var(dim=-1, unbiased=False)
                        _max_var = _var_per_prompt.max().clamp(min=1e-8)
                        _norm_var = _var_per_prompt / _max_var
                        mini_batch.batch["adv_var_weight"] = (
                            _norm_var.unsqueeze(1).expand(_num_prompts, _rollout_n).reshape(_mb_size).contiguous()
                        )

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = (
                        self.config.calculate_entropy
                        or (entropy_coeff != 0)
                        or self.config.optim.get("lr_scheduler_type", "constant") == "entropy_adaptive"
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy_in_loss = entropy_coeff != 0
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        entropy_in_loss=entropy_in_loss,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Compute rollout_corr/ diagnostics (log_ppl_diff, ppl_ratio, chi2_seq, etc.)
                    # for ALL loss modes including bypass_mode (FA async training).
                    # Note: ratio_mean/ppo_kl_mean are in pg_metrics, but rollout_corr/ metrics are NOT.
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        entropy_agg_val = entropy_agg.detach().item()
                        micro_batch_metrics["actor/entropy"] = entropy_agg_val
                        # Set H0 on first micro-batch ever seen (persists for run lifetime)
                        if self._H0_entropy is None:
                            self._H0_entropy = entropy_agg_val

                        if entropy_coeff != 0:
                            entropy_bonus_mode = self.config.get("entropy_bonus_mode", "uniform")
                            if entropy_bonus_mode == "advantage_variance":
                                adv_var_weight = model_inputs.get("adv_var_weight", None)
                                if adv_var_weight is not None:
                                    # Use pre-computed weights (correct prompt grouping guaranteed)
                                    entropy_per_resp = (entropy * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1)
                                    entropy_agg_weighted = (entropy_per_resp * adv_var_weight).mean()
                                    # Accumulate directly into metrics as scalars (not via
                                    # append_to_dict) to avoid inhomogeneous list lengths when
                                    # dynamic batching gives different micro-batch counts per GPU.
                                    metrics["actor/entropy_var_weight_mean"] += adv_var_weight.mean().detach().item()
                                    metrics["actor/entropy_var_weight_count"] += 1.0
                                    # Entropy ceiling: suppress bonus once entropy exceeds
                                    # entropy_max_ratio × H0. Prevents runaway entropy explosion.
                                    entropy_max_ratio = self.config.get("entropy_max_ratio", 1.5)
                                    if entropy_agg_val <= self._H0_entropy * entropy_max_ratio:
                                        policy_loss -= entropy_agg_weighted * entropy_coeff
                                    else:
                                        metrics["actor/entropy_bonus_suppressed"] += 1.0
                                else:
                                    # mini_batch_size not divisible by rollout_n: fall back to uniform
                                    policy_loss -= entropy_agg * entropy_coeff
                            else:
                                policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm_pre_clip, grad_norm_post_clip = self._optimizer_step()
                mini_batch_metrics = {
                    "actor/grad_norm": grad_norm_pre_clip.detach().item(),
                    "actor/grad_norm_pre_clip": grad_norm_pre_clip.detach().item(),
                    "actor/grad_norm_post_clip": grad_norm_post_clip.detach().item(),
                }
                adam_diag = self._compute_adam_diagnostics()
                mini_batch_metrics.update(adam_diag)
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        # Convert entropy_var_weight accumulator to mean; drop count key.
        if metrics["actor/entropy_var_weight_count"] > 0:
            metrics["actor/entropy_var_weight_mean"] /= metrics["actor/entropy_var_weight_count"]
        del metrics["actor/entropy_var_weight_count"]
        # Reduce list metrics to per-worker scalars before returning.
        # With dynamic batching, different DP ranks may have different numbers of
        # micro-batches. DataProto.concat merges per-worker metrics via
        # list_of_dict_to_dict_of_list, producing a list-of-lists with inhomogeneous
        # lengths → np.mean fails in the trainer's reduce_metrics call. Pre-reducing
        # here ensures each worker returns scalars, making the cross-worker aggregation
        # consistent regardless of micro-batch count differences.
        metrics.update(decomp_metrics)
        return reduce_metrics(metrics)
