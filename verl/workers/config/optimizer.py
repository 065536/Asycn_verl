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
import warnings
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = [
    "OptimizerConfig",
    "FSDPOptimizerConfig",
    "McoreOptimizerConfig",
    "build_optimizer",
    "VeOmniOptimizerConfig",
    "TorchtitanOptimizerConfig",
]


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
        zero_indexed_step (bool): Whether the LR schedule uses 0-indexed steps. If True (default),
            step counting starts at 0. If False, step counting starts at 1.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None
    zero_indexed_step: bool = True
    # Entropy-adaptive LR: minimum LR ratio floor (ρ in α_t = α_0 * max(H_t/H_ref, ρ))
    entropy_adaptive_min_ratio: float = 0.1
    # Entropy-adaptive LR: EMA smoothing factor for reference entropy (higher = smoother)
    entropy_adaptive_ema_beta: float = 0.95
    # Entropy-adaptive LR: reference mode — "ema" (H_t/H_ref_ema) or "initial" (H_t/H_0)
    entropy_adaptive_reference_mode: str = "ema"
    # Signal-fraction adaptive LR (α_t = c_t · r̂_t)
    # Controller step size for c_t update: c_{t+1} = c_t * exp(η_c * (φ̄_t - 0.5))
    signal_fraction_eta_c: float = 0.1
    # Hard bounds on scale factor c_t
    signal_fraction_c_min: float = 1e-8
    signal_fraction_c_max: float = 1e-2
    # How often (every K steps) to run the calibration step that updates c_t
    signal_fraction_calib_freq: int = 5
    # EMA smoothing factor for φ_t (realization ratio) before feeding into controller
    signal_fraction_phi_ema_beta: float = 0.9
    # Minimum |p_t| threshold: skip c_t update when predicted gain is too small
    signal_fraction_p_min: float = 1e-8
    # Lower bound for r_t_ctrl (safety floor after EMA/fast-drop, replaces raw clamp)
    signal_fraction_r_min: float = 0.01
    # Fraction of per-rank batch reserved as held-out C on calibration steps
    signal_fraction_calib_frac: float = 0.25

    # ------------------------------------------------------------------ #
    # Warmup handoff + r-side state machine (Bug 1 fix)
    # ------------------------------------------------------------------ #
    # Bootstrap floor: warmup EMA initial value and denominator floor for c_T init
    signal_fraction_r_boot: float = 0.05
    # Symmetric EMA beta used during warmup to estimate typical r̄
    signal_fraction_r_ema_beta_sym: float = 0.3
    # Asymmetric EMA betas for post-handoff control (down = fast, up = slow)
    signal_fraction_r_ema_beta_down: float = 0.5
    signal_fraction_r_ema_beta_up: float = 0.1
    # Background gradient-scale EMA (unconditional, only gated by d_min_abs)
    signal_fraction_g_rms_ema_beta: float = 0.05
    # Absolute numerical floor for denom (d_t = mean of ||ĝ_A1||² + ||ĝ_A2||²)
    signal_fraction_d_min_abs: float = 1e-30
    # Relative RMS threshold: step valid only if g_rms > tau_rms * g_rms_ema
    signal_fraction_tau_rms: float = 0.05
    # Fast-drop trigger: r_t_obs < rho * r̄_{t-1} → enter cooldown
    signal_fraction_fast_drop_rho: float = 0.7
    # Cooldown duration (steps) after fast-drop trigger; c_t frozen during cooldown
    signal_fraction_cooldown_steps: int = 5
    # Handoff interpolation steps (warmup LR → c_t * r̄_t)
    signal_fraction_handoff_steps: int = 10
    # Sign-gate mode: two-level LR gate based on alignment sign.
    # gamma=None → continuous r-shaping (default); gamma=1.0 → constant LR (M baseline);
    # gamma=0.5 → half speed on misaligned steps (A experiment).
    signal_fraction_sign_gate_gamma: Optional[float] = None
    # Optional: override post-handoff full-speed LR for mean-matching across experiments.
    # If set, c_T = alpha_plus / r_ref instead of alpha_base / r_ref.
    signal_fraction_sign_gate_alpha_plus: Optional[float] = None

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine", "entropy_adaptive", "signal_fraction"]
        assert self.entropy_adaptive_reference_mode in ["ema", "initial"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


@dataclass
class TorchtitanOptimizerConfig(OptimizerConfig):
    """Torchtitan optimizer configuration extending base OptimizerConfig.

    Args:
        name (str): Optimizer name; default is "AdamW".
        eps (float): Epsilon value for AdamW optimizer, default 1e-8.
        decay_type (str): Weight decay type: "linear", "sqrt", or "cosine".
        min_lr_factor (float): Minimum learning rate factor.
    """

    name: str = "AdamW"
    eps: float = 1e-8
    decay_type: str = "linear"
    min_lr_factor: float = 0.0


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"
    """
    import importlib

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = config.optimizer.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)
