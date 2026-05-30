import json, os, sys
import statistics

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
files = sorted([f for f in os.listdir(base) if f.endswith('.jsonl')])

# 5-task core avg keys
CORE_TASKS = [
    'val-core/AIME/acc/mean@16',
    'val-core/AIME2025/acc/mean@16',
    'val-core/Idavidrein/gpqa/acc/mean@16',
    'val-core/MINERVA/acc/mean@16',
    'val-core/OLYMPIAD_BENCH/acc/mean@16',
]

print("=" * 100)
print("PART 1: File completeness")
print("=" * 100)

for fn in files:
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    n = len(lines)
    max_step = 0
    val_steps = []
    for line in lines:
        d = json.loads(line)
        gs = d.get('global_step', d.get('step', -1))
        if gs > max_step:
            max_step = gs
        has_val = any(k.startswith('val-core/') for k in d)
        if has_val:
            val_steps.append(gs)

    short = fn.replace('deepseek1.5b_sync_8gpu_cd_', '').replace('.jsonl', '')
    vs_str = f"[{min(val_steps)},{max(val_steps)}]" if val_steps else "[?,?]"
    print(f"  {short:45s}  lines={n:4d}  max_step={max_step:4d}  n_val={len(val_steps):3d}  val_range={vs_str}")

print()
print("=" * 100)
print("PART 2: Validation trajectories (5-task core avg)")
print("=" * 100)

results = {}
for fn in files:
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    short = fn.replace('deepseek1.5b_sync_8gpu_cd_', '').replace('.jsonl', '')
    
    val_data = []
    for line in lines:
        d = json.loads(line)
        gs = d.get('global_step', d.get('step', -1))
        vals = []
        for k in CORE_TASKS:
            if k in d:
                vals.append(d[k])
        if len(vals) == 5:
            avg5 = sum(vals) / 5
            val_data.append((gs, avg5, vals))
    
    if val_data:
        print(f"\n  {short}:")
        best_avg5 = -1
        best_step = -1
        for gs, avg5, _ in val_data:
            if avg5 > best_avg5:
                best_avg5 = avg5
                best_step = gs
            marker = " <-- best" if gs == best_step and gs == val_data[-1][0] else (" <-- best" if avg5 == best_avg5 else "")
            print(f"    step={gs:4d}  avg5={avg5:.4f}{marker}")
        
        final_step, final_avg5, _ = val_data[-1]
        drop = final_avg5 - best_avg5
        results[short] = {
            'best_avg5': best_avg5, 'best_step': best_step,
            'final_avg5': final_avg5, 'final_step': final_step,
            'drop': drop, 'n_val': len(val_data)
        }

print()
print("=" * 100)
print("PART 3: Summary table")
print("=" * 100)
print(f"  {'Config':<45s} {'best_avg5':>10s} {'@step':>6s} {'final_avg5':>11s} {'@step':>6s} {'drop':>8s} {'n_val':>6s}")
print("  " + "-" * 95)

# Group by config family
families = {}
for short, r in sorted(results.items()):
    # extract family: everything before _seed
    family = short.rsplit('_seed', 1)[0]
    if family not in families:
        families[family] = []
    families[family].append((short, r))
    print(f"  {short:<45s} {r['best_avg5']:10.4f} {r['best_step']:6d} {r['final_avg5']:11.4f} {r['final_step']:6d} {r['drop']:8.4f} {r['n_val']:6d}")

print()
print("=" * 100)
print("PART 4: Group means")
print("=" * 100)
for family, items in sorted(families.items()):
    bests = [r['best_avg5'] for _, r in items]
    finals = [r['final_avg5'] for _, r in items]
    drops = [r['drop'] for _, r in items]
    n = len(items)
    mean_best = sum(bests) / n
    mean_final = sum(finals) / n
    mean_drop = sum(drops) / n
    std_final = statistics.stdev(finals) if n > 1 else 0
    final_steps = [r['final_step'] for _, r in items]
    print(f"  {family:<45s}  n={n}  mean_best={mean_best:.4f}  mean_final={mean_final:.4f} +/- {std_final:.4f}  mean_drop={mean_drop:.4f}  final_steps={final_steps}")

# Part 5: Check LR trajectory from actor/alpha_t or actor/lr
print()
print("=" * 100)
print("PART 5: LR trajectory sample (first file with actor/lr)")
print("=" * 100)
for fn in files:
    path = os.path.join(base, fn)
    short = fn.replace('deepseek1.5b_sync_8gpu_cd_', '').replace('.jsonl', '')
    lr_data = []
    for line in open(path):
        d = json.loads(line)
        gs = d.get('global_step', d.get('step', -1))
        lr = d.get('actor/lr', d.get('actor/alpha_t', None))
        if lr is not None:
            lr_data.append((gs, lr))
    if lr_data:
        print(f"\n  {short}: {len(lr_data)} LR points")
        # show first 5, middle 3, last 5
        show = lr_data[:5] + [('...', '...')] + lr_data[len(lr_data)//2-1:len(lr_data)//2+2] + [('...', '...')] + lr_data[-5:]
        for gs, lr in show:
            if gs == '...':
                print(f"    ...")
            else:
                print(f"    step={gs:4d}  lr={lr:.6e}")
