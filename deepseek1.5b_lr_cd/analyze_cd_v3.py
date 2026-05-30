import json, os, statistics

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
files = sorted([f for f in os.listdir(base) if f.endswith('.jsonl')])

CORE_TASKS = [
    'val-core/AIME/acc/mean@16',
    'val-core/AIME2025/acc/mean@16',
    'val-core/Idavidrein/gpqa/acc/mean@16',
    'val-core/MINERVA/acc/mean@16',
    'val-core/OLYMPIAD_BENCH/acc/mean@16',
]

print("=" * 110)
print("FILE COMPLETENESS")
print("=" * 110)

all_results = {}
for fn in files:
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    short = fn.replace('deepseek1.5b_sync_8gpu_cd_', '').replace('.jsonl', '')
    
    val_data = []
    lr_data = []
    max_step = 0
    
    for line in lines:
        rec = json.loads(line)
        d = rec.get('data', rec)
        gs = rec.get('step', d.get('global_step', -1))
        
        if gs > max_step:
            max_step = gs
        
        lr = d.get('actor/lr', None)
        if lr is not None:
            lr_data.append((gs, lr))
        
        vals = []
        for k in CORE_TASKS:
            if k in d:
                vals.append(d[k])
        if len(vals) == len(CORE_TASKS):
            avg5 = sum(vals) / len(vals)
            val_data.append((gs, avg5, vals))
    
    complete = "COMPLETE" if max_step >= 300 else f"TRUNC@{max_step}"
    n_val = len(val_data)
    vr = f"[{min(v[0] for v in val_data)}-{max(v[0] for v in val_data)}]" if val_data else "none"
    print(f"  {short:<45s}  lines={len(lines):4d}  max={max_step:4d}  n_val={n_val:3d}  range={vr:<12s}  {complete}")
    
    if val_data:
        best_avg5 = max(v[1] for v in val_data)
        best_step = [v[0] for v in val_data if v[1] == best_avg5][0]
        final_step, final_avg5, final_vals = val_data[-1]
        drop = final_avg5 - best_avg5
        all_results[short] = {
            'best_avg5': best_avg5, 'best_step': best_step,
            'final_avg5': final_avg5, 'final_step': final_step,
            'drop': drop, 'n_val': n_val,
            'val_data': val_data, 'lr_data': lr_data,
            'max_step': max_step, 'final_vals': final_vals,
        }

print("\n" + "=" * 110)
print("VALIDATION SUMMARY (5-task core avg)")
print("=" * 110)
print(f"  {'Config':<45s} {'best':>8s} {'@step':>6s} {'final':>8s} {'@step':>6s} {'drop':>8s} {'n_val':>6s}")
print("  " + "-" * 85)

families = {}
for short, r in sorted(all_results.items()):
    family = short.rsplit('_seed', 1)[0]
    if family not in families:
        families[family] = []
    families[family].append((short, r))
    print(f"  {short:<45s} {r['best_avg5']:8.4f} {r['best_step']:6d} {r['final_avg5']:8.4f} {r['final_step']:6d} {r['drop']:8.4f} {r['n_val']:6d}")

print("\n" + "=" * 110)
print("GROUP MEANS")
print("=" * 110)
for family, items in sorted(families.items()):
    bests = [r['best_avg5'] for _, r in items]
    finals = [r['final_avg5'] for _, r in items]
    drops = [r['drop'] for _, r in items]
    n = len(items)
    mean_best = sum(bests) / n
    mean_final = sum(finals) / n
    mean_drop = sum(drops) / n
    std_final = statistics.stdev(finals) if n > 1 else float('nan')
    seeds = [s.rsplit('_seed', 1)[1] for s, _ in items]
    max_steps = [r['max_step'] for _, r in items]
    print(f"  {family}")
    print(f"    seeds={seeds}  max_steps={max_steps}")
    print(f"    mean_best={mean_best:.4f}  mean_final={mean_final:.4f} +/- {std_final:.4f}  mean_drop={mean_drop:.4f}")

# Per-task at final
print("\n" + "=" * 110)
print("PER-TASK FINAL")
print("=" * 110)
task_short = ['AIME', 'AIME25', 'GPQA', 'MINERV', 'OLYMP']
header = f"  {'Config':<45s}" + "".join(f"{t:>10s}" for t in task_short) + f"{'avg5':>10s}"
print(header)
print("  " + "-" * (45 + 10*len(task_short) + 10))
for short, r in sorted(all_results.items()):
    line = f"  {short:<45s}"
    for v in r['final_vals']:
        line += f"{v:10.4f}"
    line += f"{r['final_avg5']:10.4f}"
    print(line)

# LR trajectory
print("\n" + "=" * 110)
print("LR TRAJECTORY")
print("=" * 110)
for short, r in sorted(all_results.items()):
    lr_data = r['lr_data']
    if not lr_data:
        continue
    lr_dict = {gs: lr for gs, lr in lr_data}
    sample_steps = [1, 5, 10, 50, 100, 150, 200, 250, 300]
    vals = []
    for s in sample_steps:
        if s in lr_dict:
            vals.append(f"{lr_dict[s]:.2e}")
        else:
            vals.append("   -   ")
    print(f"  {short:<45s}  " + "  ".join(f"s{s}={v}" for s, v in zip(sample_steps, vals)))

# Validation trajectory for complete runs
print("\n" + "=" * 110)
print("VALIDATION TRAJECTORY (every 10 steps)")
print("=" * 110)
for short, r in sorted(all_results.items()):
    vd = r['val_data']
    print(f"\n  {short}:")
    for gs, avg5, _ in vd:
        bar = "*" * int(avg5 * 100)
        print(f"    step={gs:4d}  avg5={avg5:.4f}  {bar}")

# Reference constant LR
print("\n" + "=" * 110)
print("REFERENCE: constant LR (from deepseek1.5b_lr/)")
print("=" * 110)
ref_base = '/data/250010176/codes/verl/deepseek1.5b_lr'
for pat in ['diag_constant_lr1e-5', 'diag_constant_lr3.1e-6']:
    ref_files = sorted([f for f in os.listdir(ref_base) if pat in f and f.endswith('.jsonl')])
    for fn in ref_files:
        path = os.path.join(ref_base, fn)
        lines = open(path).readlines()
        short = fn.replace('deepseek1.5b_sync_8gpu_', '').replace('.jsonl', '')
        max_step = 0
        val_data = []
        for line in lines:
            rec = json.loads(line)
            d = rec.get('data', rec)
            gs = rec.get('step', d.get('global_step', -1))
            if gs > max_step:
                max_step = gs
            vals = []
            for k in CORE_TASKS:
                if k in d:
                    vals.append(d[k])
            if len(vals) == len(CORE_TASKS):
                avg5 = sum(vals) / len(vals)
                val_data.append((gs, avg5))
        if val_data:
            best = max(v[1] for v in val_data)
            best_s = [v[0] for v in val_data if v[1] == best][0]
            final = val_data[-1][1]
            final_s = val_data[-1][0]
            drop = final - best
            print(f"  {short:<45s}  best={best:.4f}@{best_s}  final={final:.4f}@{final_s}  drop={drop:.4f}")
