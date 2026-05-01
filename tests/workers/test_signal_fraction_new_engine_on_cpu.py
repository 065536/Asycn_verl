import ast
from pathlib import Path

import torch

from verl.workers.engine.fsdp.transformer_impl import SignalFractionLRScheduler


def _make_scheduler(num_warmup_steps=2):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1e-5)
    return SignalFractionLRScheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        c_init=1e-5,
        eta_c=0.0,
        c_min=1e-8,
        c_max=1e-2,
        phi_ema_beta=0.9,
        r_min_ctrl=0.01,
        r_boot=0.05,
        r_ema_beta_sym=0.3,
        r_ema_beta_down=0.5,
        r_ema_beta_up=0.1,
        g_rms_ema_beta=0.05,
        d_min_abs=1e-30,
        tau_rms=0.05,
        fast_drop_rho=0.7,
        cooldown_steps=5,
        handoff_steps=10,
    )


def test_signal_fraction_scheduler_uses_current_step_without_internal_increment():
    scheduler = _make_scheduler(num_warmup_steps=2)

    scheduler.step()
    alpha_step_1 = scheduler.update_r_and_set_lr(g_dot=1.0, d_t=2.0, g_rms_t=1.0, r_hat_raw=0.5)

    assert scheduler.step_count == 1
    assert alpha_step_1 == scheduler.base_lrs[0] / 2

    scheduler.step()
    alpha_step_2 = scheduler.update_r_and_set_lr(g_dot=1.0, d_t=2.0, g_rms_t=1.0, r_hat_raw=0.5)

    assert scheduler.step_count == 2
    assert alpha_step_2 == scheduler.base_lrs[0]


def test_engine_worker_skips_outer_scheduler_step_for_signal_fraction():
    source = Path("verl/workers/engine_workers.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    train_batch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_batch"
    )

    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "signal_fraction_engine_step" for target in node.targets)
        for node in ast.walk(train_batch)
    )
    matching_if = [
        node
        for node in ast.walk(train_batch)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and len(node.test.values) == 2
    ]
    assert any(
        isinstance(node.test.values[0], ast.Name)
        and node.test.values[0].id == "update_lr_scheduler"
        and isinstance(node.test.values[1], ast.UnaryOp)
        and isinstance(node.test.values[1].op, ast.Not)
        and isinstance(node.test.values[1].operand, ast.Name)
        and node.test.values[1].operand.id == "signal_fraction_engine_step"
        for node in matching_if
    )
