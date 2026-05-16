#!/usr/bin/env python3
"""Comprehensive analysis of ratio_of_sums W10 runs (Bug 17 fix reruns)."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path("/data/250010176/codes/verl/deepseek1.5b_lr")

# === Data sources ===
ROS_FILES = {
    "RoS seed0": BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_ratio_of_sums_w10_seed0.jsonl",
    "RoS seed1": BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_ratio_of_sums_w10_seed1.jsonl",
    "RoS seed2": BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_ratio_of_sums_w10_seed2.jsonl",
}

REF_FILES = {
    "RE seed0":  BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed0.jsonl",
    "RE seed1":  BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed1.jsonl",
    "RE seed2":  BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed2.jsonl",
    "RE seed42": BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10_seed42_rerun.jsonl",
    "RE old42":  BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_windowr_w10.jsonl",
    "B-current": BASE / "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5.jsonl",
}

VAL_TASKS = [
    "val-core/AIME/acc/mean@16",
    "val-core/AIME2025/acc/mean@16",
    "val-core/Idavidrein/gpqa/acc/mean@16",
    "val-core/MINERVA/acc/mean@16",
    "val-core/OLYMPIAD_BENCH/acc/mean@16",
]

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def extract(records, key):
    steps, vals = [], []
    for rec in records:
        data = rec.get("data", rec)
        step = rec.get("step", data.get("step"))
        if key in data:
            steps.append(step)
            vals.append(data[key])
    return np.array(steps), np.array(vals)

def safe_extract(records, key):
    steps, vals = extract(records, key)
    if len(steps) == 0:
        return np.array([]), np.array([])
    return steps, vals

def avg5_at_step(records, step):
    for rec in records:
        data = rec.get("data", rec)
        s = rec.get("step", data.get("step"))
        if s == step:
            vals = []
            for task in VAL_TASKS:
                if task in data:
                    vals.append(data[task])
            if len(vals) == 5:
                return np.mean(vals)
    return None

def compute_avg5_series(records):
    steps, vals = [], []
    for rec in records:
        data = rec.get("data", rec)
        step = rec.get("step", data.get("step"))
        task_vals = []
        for task in VAL_TASKS:
            if task in data:
                task_vals.append(data[task])
        if len(task_vals) == 5:
            steps.append(step)
            vals.append(np.mean(task_vals))
    return np.array(steps), np.array(vals)

def fmt(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"

def fmt_e(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:.{digits}e}"

# ============================================================
# Load all data
# ============================================================
all_data = {}
for name, path in {**ROS_FILES, **REF_FILES}.items():
    if path.exists():
        all_data[name] = load_jsonl(path)
    else:
        print(f"WARNING: {path} not found")

print("=" * 80)
print("RATIO_OF_SUMS W10 — COMPREHENSIVE ANALYSIS (Bug 17 fix reruns)")
print("=" * 80)

# ============================================================
# 1. SCORE: 5-task avg validation
# ============================================================
print("\n" + "=" * 80)
print("1. VALIDATION SCORE (5-task core avg)")
print("=" * 80)

summary_rows = []
for name, records in all_data.items():
    vsteps, vvals = compute_avg5_series(records)
    if len(vsteps) == 0:
        continue
    best_idx = np.argmax(vvals)
    best_step = int(vsteps[best_idx])
    best_val = vvals[best_idx]
    final_val = vvals[-1]
    final_step = int(vsteps[-1])
    drop = final_val - best_val
    summary_rows.append((name, final_step, best_val, best_step, final_val, drop))

print(f"\n{'Run':<16} {'MaxStep':>7} {'Best avg5':>10} {'@step':>6} {'Final avg5':>11} {'Drop':>8}")
print("-" * 65)
for row in summary_rows:
    name, fs, bv, bs, fv, dr = row
    print(f"{name:<16} {fs:>7d} {bv:>10.4f} {bs:>6d} {fv:>11.4f} {dr:>+8.4f}")

# Group means
print("\n--- Group means ---")
for group_name, group_keys in [
    ("ratio_of_sums", ["RoS seed0", "RoS seed1", "RoS seed2"]),
    ("replace_ema (new)", ["RE seed0", "RE seed1", "RE seed2"]),
    ("replace_ema (new+42)", ["RE seed0", "RE seed1", "RE seed2", "RE seed42"]),
]:
    bests, finals, drops = [], [], []
    for k in group_keys:
        if k in all_data:
            vs, vv = compute_avg5_series(all_data[k])
            if len(vs) > 0:
                bi = np.argmax(vv)
                bests.append(vv[bi])
                finals.append(vv[-1])
                drops.append(vv[-1] - vv[bi])
    if bests:
        print(f"  {group_name}: best={np.mean(bests):.4f}±{np.std(bests):.4f}  "
              f"final={np.mean(finals):.4f}±{np.std(finals):.4f}  "
              f"drop={np.mean(drops):+.4f}±{np.std(drops):.4f}  (n={len(bests)})")

# Per-task breakdown at final step
print("\n--- Per-task final scores ---")
print(f"{'Run':<16}", end="")
short_names = ["AIME", "AIME25", "GPQA", "MINERVA", "OLYMP"]
for sn in short_names:
    print(f" {sn:>8}", end="")
print()
print("-" * 65)
for name, records in all_data.items():
    vsteps, _ = compute_avg5_series(records)
    if len(vsteps) == 0:
        continue
    final_step = int(vsteps[-1])
    print(f"{name:<16}", end="")
    for task in VAL_TASKS:
        v = avg5_at_step(records, final_step)  # dummy
        # extract per-task
        for rec in records:
            data = rec.get("data", rec)
            s = rec.get("step", data.get("step"))
            if s == final_step and task in data:
                print(f" {data[task]:>8.4f}", end="")
                break
        else:
            print(f" {'N/A':>8}", end="")
    print()

# ============================================================
# 2. LEARNING RATE (alpha_t) dynamics
# ============================================================
print("\n" + "=" * 80)
print("2. LEARNING RATE (alpha_t) DYNAMICS")
print("=" * 80)

for name in list(ROS_FILES.keys()) + ["B-current"]:
    if name not in all_data:
        continue
    records = all_data[name]
    steps, alpha = safe_extract(records, "actor/alpha_t")
    if len(steps) == 0:
        continue
    # warmup = steps 1-20, post-handoff = steps 21-300
    warmup_mask = (steps >= 1) & (steps <= 20)
    post_mask = steps >= 21
    early_mask = (steps >= 21) & (steps <= 100)
    mid_mask = (steps >= 101) & (steps <= 200)
    late_mask = (steps >= 201) & (steps <= 300)

    print(f"\n  {name}:")
    if warmup_mask.any():
        print(f"    warmup  (1-20):   mean={np.mean(alpha[warmup_mask]):.3e}  "
              f"range=[{np.min(alpha[warmup_mask]):.3e}, {np.max(alpha[warmup_mask]):.3e}]")
    if early_mask.any():
        print(f"    early  (21-100):  mean={np.mean(alpha[early_mask]):.3e}  "
              f"range=[{np.min(alpha[early_mask]):.3e}, {np.max(alpha[early_mask]):.3e}]")
    if mid_mask.any():
        print(f"    mid   (101-200):  mean={np.mean(alpha[mid_mask]):.3e}  "
              f"range=[{np.min(alpha[mid_mask]):.3e}, {np.max(alpha[mid_mask]):.3e}]")
    if late_mask.any():
        print(f"    late  (201-300):  mean={np.mean(alpha[late_mask]):.3e}  "
              f"range=[{np.min(alpha[late_mask]):.3e}, {np.max(alpha[late_mask]):.3e}]")
    if post_mask.any():
        print(f"    post-handoff:     mean={np.mean(alpha[post_mask]):.3e}  "
              f"std={np.std(alpha[post_mask]):.3e}")

# ============================================================
# 3. R_WINDOW / R_HAT / intermediate quantities
# ============================================================
print("\n" + "=" * 80)
print("3. R-SIDE SIGNALS: r_window, r_hat, g_dot, norms")
print("=" * 80)

r_metrics = [
    ("actor/r_window", "r_window"),
    ("actor/r_window_count", "r_win_cnt"),
    ("actor/r_hat_raw", "r_hat_raw"),
    ("actor/r_hat", "r_hat(ema)"),
    ("actor/r_ctrl", "r_ctrl"),
    ("actor/g_A1_dot_A2", "g_A1·A2"),
    ("actor/g_norm_A1_sq", "||gA1||²"),
    ("actor/g_norm_A2_sq", "||gA2||²"),
    ("actor/g_dot_positive", "g_dot+"),
    ("actor/g_rms", "g_rms"),
]

for name in list(ROS_FILES.keys()):
    if name not in all_data:
        continue
    records = all_data[name]
    print(f"\n  {name}:")
    for metric_key, label in r_metrics:
        steps, vals = safe_extract(records, metric_key)
        if len(steps) == 0:
            continue
        post = (steps >= 21) & (steps <= 300)
        if not post.any():
            post = steps >= 1
        v = vals[post]
        if label == "g_dot+":
            print(f"    {label:<12}  post-handoff P(g_dot>0)={np.mean(v):.3f}")
        elif label == "r_win_cnt":
            print(f"    {label:<12}  post-handoff mean={np.mean(v):.1f}  "
                  f"min={np.min(v):.0f}  max={np.max(v):.0f}")
        else:
            print(f"    {label:<12}  post-handoff mean={np.mean(v):.4e}  "
                  f"std={np.std(v):.4e}  "
                  f"range=[{np.min(v):.4e}, {np.max(v):.4e}]")

# r_window sanity: should be in (0, 1] for correct implementation
print("\n--- r_window sanity check (must be in (0,1] if correct) ---")
for name in list(ROS_FILES.keys()):
    if name not in all_data:
        continue
    records = all_data[name]
    steps, rw = safe_extract(records, "actor/r_window")
    if len(rw) == 0:
        continue
    post = steps >= 21
    rw_post = rw[post]
    gt1 = np.sum(rw_post > 1.0)
    le0 = np.sum(rw_post <= 0.0)
    print(f"  {name}: r_window>1: {gt1}/{len(rw_post)}, r_window<=0: {le0}/{len(rw_post)}, "
          f"range=[{np.min(rw_post):.4e}, {np.max(rw_post):.4e}]  {'✓ OK' if gt1==0 and le0==0 else '✗ PROBLEM'}")

# ============================================================
# 4. RESPONSE LENGTH
# ============================================================
print("\n" + "=" * 80)
print("4. RESPONSE LENGTH")
print("=" * 80)

for name in list(ROS_FILES.keys()) + ["B-current"]:
    if name not in all_data:
        continue
    records = all_data[name]
    steps, rlen = safe_extract(records, "response_length/mean")
    if len(steps) == 0:
        continue
    early = (steps >= 1) & (steps <= 50)
    mid = (steps >= 101) & (steps <= 200)
    late = (steps >= 251) & (steps <= 300)
    parts = []
    for label, mask in [("early(1-50)", early), ("mid(101-200)", mid), ("late(251-300)", late)]:
        if mask.any():
            parts.append(f"{label}={np.mean(rlen[mask]):.0f}")
    print(f"  {name:<16} {', '.join(parts)}")

# Also check if response length collapses (sign of bug)
print("\n--- Response length collapse check ---")
for name in list(ROS_FILES.keys()):
    if name not in all_data:
        continue
    records = all_data[name]
    steps, rlen = safe_extract(records, "response_length/mean")
    late = steps >= 250
    if late.any():
        min_late = np.min(rlen[late])
        print(f"  {name}: min response_len (step>=250) = {min_late:.0f}  "
              f"{'✓ OK' if min_late > 50 else '✗ COLLAPSED'}")

# ============================================================
# 5. TRAINING SCORE (critic/score/mean) and entropy
# ============================================================
print("\n" + "=" * 80)
print("5. TRAINING SCORE & ENTROPY")
print("=" * 80)

for name in list(ROS_FILES.keys()) + ["B-current"]:
    if name not in all_data:
        continue
    records = all_data[name]
    steps_s, score = safe_extract(records, "critic/score/mean")
    steps_e, entropy = safe_extract(records, "actor/entropy")
    print(f"\n  {name}:")
    if len(steps_s) > 0:
        for label, lo, hi in [("early(1-50)", 1, 50), ("mid(101-200)", 101, 200), ("late(251-300)", 251, 300)]:
            mask = (steps_s >= lo) & (steps_s <= hi)
            if mask.any():
                print(f"    score  {label}: mean={np.mean(score[mask]):.4f}  std={np.std(score[mask]):.4f}")
    if len(steps_e) > 0:
        for label, lo, hi in [("early(1-50)", 1, 50), ("mid(101-200)", 101, 200), ("late(251-300)", 251, 300)]:
            mask = (steps_e >= lo) & (steps_e <= hi)
            if mask.any():
                print(f"    entropy {label}: mean={np.mean(entropy[mask]):.4f}")

# ============================================================
# 6. PPO DIAGNOSTICS
# ============================================================
print("\n" + "=" * 80)
print("6. PPO DIAGNOSTICS (KL, ratio tails, clip)")
print("=" * 80)

ppo_metrics = [
    ("actor/ppo_kl_mean", "kl_mean"),
    ("actor/ratio_std", "ratio_std"),
    ("actor/ratio_p95", "ratio_p95"),
    ("actor/ratio_frac_gt_1p2", "frac>1.2"),
    ("actor/pg_loss", "pg_loss"),
]

for name in list(ROS_FILES.keys()) + ["B-current"]:
    if name not in all_data:
        continue
    records = all_data[name]
    print(f"\n  {name}:")
    for metric_key, label in ppo_metrics:
        steps, vals = safe_extract(records, metric_key)
        if len(steps) == 0:
            continue
        post = steps >= 21
        if not post.any():
            post = steps >= 1
        v = vals[post]
        print(f"    {label:<12}  mean={np.mean(v):.4e}  std={np.std(v):.4e}  "
              f"max={np.max(v):.4e}")

# ============================================================
# 7. COMPARISON TABLE: ratio_of_sums vs replace_ema vs B-current
# ============================================================
print("\n" + "=" * 80)
print("7. HEAD-TO-HEAD: ratio_of_sums vs replace_ema vs B-current")
print("=" * 80)

comparison_groups = {
    "RoS W10 (n=3)": ["RoS seed0", "RoS seed1", "RoS seed2"],
    "RE W10 new (n=3)": ["RE seed0", "RE seed1", "RE seed2"],
    "RE W10 +42 (n=4)": ["RE seed0", "RE seed1", "RE seed2", "RE seed42"],
    "B-current (n=1)": ["B-current"],
}

print(f"\n{'Group':<22} {'best avg5':>10} {'final avg5':>11} {'drop':>8} "
      f"{'alpha_mean':>11} {'r_ctrl_mean':>12} {'resp_len_late':>14}")
print("-" * 95)

for group_name, keys in comparison_groups.items():
    bests, finals, drops, alphas, rctrls, rlens = [], [], [], [], [], []
    for k in keys:
        if k not in all_data:
            continue
        rec = all_data[k]
        vs, vv = compute_avg5_series(rec)
        if len(vs) > 0:
            bi = np.argmax(vv)
            bests.append(vv[bi])
            finals.append(vv[-1])
            drops.append(vv[-1] - vv[bi])
        sa, va = safe_extract(rec, "actor/alpha_t")
        post = sa >= 21
        if post.any():
            alphas.append(np.mean(va[post]))
        sr, vr = safe_extract(rec, "actor/r_ctrl")
        post = sr >= 21
        if post.any():
            rctrls.append(np.mean(vr[post]))
        sl, vl = safe_extract(rec, "response_length/mean")
        late = sl >= 251
        if late.any():
            rlens.append(np.mean(vl[late]))

    if bests:
        print(f"{group_name:<22} {np.mean(bests):>10.4f} {np.mean(finals):>11.4f} "
              f"{np.mean(drops):>+8.4f} {np.mean(alphas):>11.3e} "
              f"{np.mean(rctrls):>12.4e} "
              f"{np.mean(rlens) if rlens else float('nan'):>14.0f}")

# ============================================================
# 8. STAGE-BY-STAGE alpha_t comparison: RoS vs RE
# ============================================================
print("\n" + "=" * 80)
print("8. STAGE-BY-STAGE alpha_t: ratio_of_sums vs replace_ema")
print("=" * 80)

stage_defs = [("warmup(1-20)", 1, 20), ("early(21-100)", 21, 100),
              ("mid(101-200)", 101, 200), ("late(201-300)", 201, 300)]

for group_label, group_keys in [("RoS", list(ROS_FILES.keys())),
                                 ("RE new", ["RE seed0", "RE seed1", "RE seed2"])]:
    print(f"\n  {group_label}:")
    for stage_label, lo, hi in stage_defs:
        stage_alphas = []
        for k in group_keys:
            if k not in all_data:
                continue
            sa, va = safe_extract(all_data[k], "actor/alpha_t")
            mask = (sa >= lo) & (sa <= hi)
            if mask.any():
                stage_alphas.append(np.mean(va[mask]))
        if stage_alphas:
            print(f"    {stage_label:<18} mean={np.mean(stage_alphas):.4e}  "
                  f"seeds=[{', '.join(f'{v:.4e}' for v in stage_alphas)}]")

# ============================================================
# 9. VALIDATION CURVE PROGRESSION (every 50 steps)
# ============================================================
print("\n" + "=" * 80)
print("9. VALIDATION avg5 PROGRESSION (at val steps)")
print("=" * 80)

check_steps = [0, 50, 100, 150, 200, 250, 300]
print(f"{'Run':<16}", end="")
for s in check_steps:
    print(f" {'@'+str(s):>7}", end="")
print()
print("-" * 75)

for name in ["RoS seed0", "RoS seed1", "RoS seed2",
             "RE seed0", "RE seed1", "RE seed2", "RE seed42", "B-current"]:
    if name not in all_data:
        continue
    records = all_data[name]
    vsteps, vvals = compute_avg5_series(records)
    print(f"{name:<16}", end="")
    for s in check_steps:
        idx = np.where(vsteps == s)[0]
        if len(idx) > 0:
            print(f" {vvals[idx[0]]:>7.4f}", end="")
        else:
            # find closest
            diffs = np.abs(vsteps - s)
            ci = np.argmin(diffs)
            if diffs[ci] <= 5:
                print(f" {vvals[ci]:>7.4f}", end="")
            else:
                print(f" {'---':>7}", end="")
    print()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
