import json, os, sys
import numpy as np

SEEDS = [0, 1, 2, 42]
BASE = "deepseek1.5b_lr/deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_ratio_of_sums_w10_seed{}.jsonl"

VAL_BENCHMARKS = ["AIME", "AIME2025", "OLYMPIAD", "GPQA", "MINERVA"]
VAL_KEY_TPL = "val-aux/{}/score/mean@16"

ACTOR_KEYS = [
    "actor/alpha_t", "actor/r_hat", "actor/r_hat_raw", "actor/r_ctrl",
    "actor/r_window", "actor/r_window_count", "actor/r_window_enabled",
    "actor/r_window_invalid_value",
    "actor/g_dot_positive", "actor/g_A1_dot_A2",
    "actor/g_norm_A1_sq", "actor/g_norm_A2_sq",
    "actor/c_t", "actor/phi_bar",
    "actor/entropy", "actor/ppo_kl_mean", "actor/ppo_kl_max",
    "actor/grad_norm", "actor/pg_loss",
    "actor/ratio_mean", "actor/ratio_std", "actor/ratio_p95", "actor/ratio_p99",
    "actor/ratio_frac_gt_1p2", "actor/ratio_frac_lt_0p8",
    "actor/alpha_rate_limited",
]

CRITIC_KEYS = [
    "critic/score/mean", "critic/score/std",
    "critic/rewards/mean", "critic/rewards/std",
    "critic/advantages/mean", "critic/advantages/std", "critic/advantages/abs_mean",
]

RESP_KEYS = [
    "response_length/mean", "response_length/max", "response_length/min",
    "response_length/clip_ratio",
    "reward/overlong_rate",
]

def load_run(seed):
    path = BASE.format(seed)
    steps = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            steps[d["step"]] = d["data"]
    return steps

def get_val_series(run, benchmark):
    key = VAL_KEY_TPL.format(benchmark)
    vals = []
    for step in sorted(run.keys()):
        if key in run[step]:
            vals.append((step, run[step][key]))
    return vals

def get_series(run, key):
    vals = []
    for step in sorted(run.keys()):
        if key in run[step]:
            vals.append((step, run[step][key]))
    return vals

def avg5(run, step):
    total = 0
    count = 0
    for bm in VAL_BENCHMARKS:
        key = VAL_KEY_TPL.format(bm)
        if key in run.get(step, {}):
            total += run[step][key]
            count += 1
    return total / count if count == 5 else None

runs = {s: load_run(s) for s in SEEDS}

print("=" * 80)
print("RATIO-OF-SUMS W10: 4-SEED EXPERIMENT REPORT")
print("=" * 80)

# 1. Validation: 5-task avg
print("\n## 1. 5-Task Core Average (score/mean@16)")
print("-" * 70)
print(f"{'step':>5}", end="")
for s in SEEDS:
    print(f"  {'seed'+str(s):>10}", end="")
print(f"  {'mean':>10}  {'std':>10}")

val_steps = sorted([st for st in runs[SEEDS[0]] if avg5(runs[SEEDS[0]], st) is not None])
seed_best = {s: (0, 0.0) for s in SEEDS}
seed_final = {s: 0.0 for s in SEEDS}

for step in val_steps:
    vals = []
    print(f"{step:>5}", end="")
    for s in SEEDS:
        v = avg5(runs[s], step)
        if v is not None:
            vals.append(v)
            print(f"  {v:>10.4f}", end="")
            if v > seed_best[s][1]:
                seed_best[s] = (step, v)
            seed_final[s] = v
        else:
            print(f"  {'N/A':>10}", end="")
    if vals:
        print(f"  {np.mean(vals):>10.4f}  {np.std(vals):>10.4f}", end="")
    print()

print("\n### Summary: best / final / drop")
print(f"{'seed':>6} {'best_step':>10} {'best_avg5':>10} {'final_avg5':>10} {'drop':>10}")
all_best, all_final = [], []
for s in SEEDS:
    bst_step, bst_val = seed_best[s]
    fin_val = seed_final[s]
    drop = fin_val - bst_val
    all_best.append(bst_val)
    all_final.append(fin_val)
    print(f"{s:>6} {bst_step:>10} {bst_val:>10.4f} {fin_val:>10.4f} {drop:>10.4f}")
print(f"{'mean':>6} {'':>10} {np.mean(all_best):>10.4f} {np.mean(all_final):>10.4f} {np.mean(all_final)-np.mean(all_best):>10.4f}")
print(f"{'std':>6} {'':>10} {np.std(all_best):>10.4f} {np.std(all_final):>10.4f}")

# 2. Per-benchmark final
print("\n## 2. Per-Benchmark Final (step 300) score/mean@16")
print(f"{'benchmark':>12}", end="")
for s in SEEDS:
    print(f"  {'seed'+str(s):>10}", end="")
print(f"  {'mean':>10}  {'std':>10}")
for bm in VAL_BENCHMARKS:
    key = VAL_KEY_TPL.format(bm)
    print(f"{bm:>12}", end="")
    vals = []
    for s in SEEDS:
        v = runs[s].get(300, {}).get(key)
        if v is not None:
            vals.append(v)
            print(f"  {v:>10.4f}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    if vals:
        print(f"  {np.mean(vals):>10.4f}  {np.std(vals):>10.4f}", end="")
    print()

# 3. Per-benchmark best
print("\n## 3. Per-Benchmark Best score/mean@16")
print(f"{'benchmark':>12}", end="")
for s in SEEDS:
    print(f"  {'seed'+str(s):>10}", end="")
print(f"  {'mean':>10}")
for bm in VAL_BENCHMARKS:
    key = VAL_KEY_TPL.format(bm)
    print(f"{bm:>12}", end="")
    vals = []
    for s in SEEDS:
        best_v = 0
        for step in sorted(runs[s].keys()):
            v = runs[s][step].get(key)
            if v is not None and v > best_v:
                best_v = v
        vals.append(best_v)
        print(f"  {best_v:>10.4f}", end="")
    print(f"  {np.mean(vals):>10.4f}")

# 4. Training dynamics - stage summary
print("\n## 4. Training Dynamics: Stage-Level Summary")
stages = {"warmup(1-20)": (1,20), "early(21-100)": (21,100), "mid(101-200)": (101,200), "late(201-300)": (201,300)}
dyn_keys = ["actor/alpha_t", "actor/r_hat", "actor/r_hat_raw", "actor/r_ctrl",
            "actor/r_window", "actor/r_window_count",
            "actor/g_dot_positive", "actor/entropy",
            "actor/ppo_kl_mean", "actor/grad_norm", "actor/pg_loss",
            "actor/ratio_std"]

for dk in dyn_keys:
    print(f"\n### {dk}")
    print(f"{'stage':>18}", end="")
    for s in SEEDS:
        print(f"  {'seed'+str(s):>10}", end="")
    print(f"  {'mean':>10}")
    for sname, (s1, s2) in stages.items():
        print(f"{sname:>18}", end="")
        stage_means = []
        for s in SEEDS:
            vals = [runs[s][st][dk] for st in range(s1, s2+1) if st in runs[s] and dk in runs[s][st]]
            m = np.mean(vals) if vals else float('nan')
            stage_means.append(m)
            print(f"  {m:>10.6g}", end="")
        print(f"  {np.mean(stage_means):>10.6g}")

# 5. Response length and overlong
print("\n## 5. Response Length & Overlong Rate (stage means)")
for rk in RESP_KEYS:
    print(f"\n### {rk}")
    print(f"{'stage':>18}", end="")
    for s in SEEDS:
        print(f"  {'seed'+str(s):>10}", end="")
    print(f"  {'mean':>10}")
    for sname, (s1, s2) in stages.items():
        print(f"{sname:>18}", end="")
        stage_means = []
        for s in SEEDS:
            vals = [runs[s][st][rk] for st in range(s1, s2+1) if st in runs[s] and rk in runs[s][st]]
            m = np.mean(vals) if vals else float('nan')
            stage_means.append(m)
            print(f"  {m:>10.4g}", end="")
        print(f"  {np.mean(stage_means):>10.4g}")

# 6. Score (train) trajectory
print("\n## 6. Training Score (critic/score/mean) - stage means")
tk = "critic/score/mean"
print(f"{'stage':>18}", end="")
for s in SEEDS:
    print(f"  {'seed'+str(s):>10}", end="")
print(f"  {'mean':>10}")
for sname, (s1, s2) in stages.items():
    print(f"{sname:>18}", end="")
    stage_means = []
    for s in SEEDS:
        vals = [runs[s][st][tk] for st in range(s1, s2+1) if st in runs[s] and tk in runs[s][st]]
        m = np.mean(vals) if vals else float('nan')
        stage_means.append(m)
        print(f"  {m:>10.4f}", end="")
    print(f"  {np.mean(stage_means):>10.4f}")

# 7. r_window diagnostics
print("\n## 7. r_window Diagnostics")
for s in SEEDS:
    series = get_series(runs[s], "actor/r_window")
    rw_vals = [v for _, v in series if v != 0.05]  # exclude boot value
    invalid_series = get_series(runs[s], "actor/r_window_invalid_value")
    inv_vals = [v for _, v in invalid_series]
    n_invalid = sum(1 for v in inv_vals if v >= 0)
    print(f"seed{s}: r_window range [{min(rw_vals):.6f}, {max(rw_vals):.6f}], "
          f"mean={np.mean(rw_vals):.6f}, invalid_steps={n_invalid}")

print("\n## 8. g_dot_positive rate by stage")
print(f"{'stage':>18}", end="")
for s in SEEDS:
    print(f"  {'seed'+str(s):>10}", end="")
print(f"  {'mean':>10}")
for sname, (s1, s2) in stages.items():
    print(f"{sname:>18}", end="")
    stage_means = []
    for s in SEEDS:
        vals = [runs[s][st]["actor/g_dot_positive"] for st in range(s1, s2+1) if st in runs[s] and "actor/g_dot_positive" in runs[s][st]]
        m = np.mean(vals) if vals else float('nan')
        stage_means.append(m)
        print(f"  {m:>10.4f}", end="")
    print(f"  {np.mean(stage_means):>10.4f}")
