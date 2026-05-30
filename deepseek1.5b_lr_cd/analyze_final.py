import json, os, statistics

CORE_TASKS = [
    'val-core/AIME/acc/mean@16',
    'val-core/AIME2025/acc/mean@16',
    'val-core/Idavidrein/gpqa/acc/mean@16',
    'val-core/MINERVA/acc/mean@16',
    'val-core/OLYMPIAD_BENCH/acc/mean@16',
]

def extract_results(path):
    lines = open(path).readlines()
    max_step = 0
    val_data = []
    lr_data = []
    for line in lines:
        rec = json.loads(line)
        d = rec.get('data', rec)
        gs = rec.get('step', d.get('global_step', -1))
        if gs > max_step:
            max_step = gs
        lr = d.get('actor/lr', None)
        if lr is not None:
            lr_data.append((gs, lr))
        vals = [d.get(k) for k in CORE_TASKS]
        if all(v is not None for v in vals):
            avg5 = sum(vals) / len(vals)
            val_data.append((gs, avg5, vals))
    return max_step, val_data, lr_data, len(lines)

# ============================================
# Gather all decay runs
# ============================================
base_cd = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
base_lr = '/data/250010176/codes/verl/deepseek1.5b_lr'

all_runs = []

# CD runs
for fn in sorted(os.listdir(base_cd)):
    if not fn.endswith('.jsonl'):
        continue
    path = os.path.join(base_cd, fn)
    short = fn.replace('deepseek1.5b_sync_8gpu_cd_', '').replace('.jsonl', '')
    max_step, val_data, lr_data, n_lines = extract_results(path)
    all_runs.append((short, max_step, val_data, lr_data, n_lines, 'decay'))

# Reference constant LR runs
for pat in ['diag_constant_lr1e-5', 'diag_constant_lr3.1e-6']:
    for fn in sorted(os.listdir(base_lr)):
        if pat in fn and fn.endswith('.jsonl'):
            path = os.path.join(base_lr, fn)
            short = fn.replace('deepseek1.5b_sync_8gpu_', '').replace('.jsonl', '')
            max_step, val_data, lr_data, n_lines = extract_results(path)
            all_runs.append((short, max_step, val_data, lr_data, n_lines, 'ref'))

# ============================================
# Print all results
# ============================================
print("=" * 120)
print("COMPLETE RESULTS TABLE")
print("=" * 120)
print(f"  {'Config':<55s} {'max':>5s} {'lines':>6s} {'n_val':>6s} {'best':>8s} {'@':>4s} {'final':>8s} {'@':>5s} {'drop':>8s} {'type':>6s}")
print("  " + "-" * 115)

group_data = {}
for short, max_step, val_data, lr_data, n_lines, rtype in all_runs:
    if not val_data:
        print(f"  {short:<55s} {max_step:5d} {n_lines:6d}      0      -    -      -     -        -   {rtype:>6s}")
        continue
    
    best = max(v[1] for v in val_data)
    best_s = [v[0] for v in val_data if v[1] == best][0]
    final_s, final, final_vals = val_data[-1]
    drop = final - best
    
    print(f"  {short:<55s} {max_step:5d} {n_lines:6d} {len(val_data):6d} {best:8.4f} {best_s:4d} {final:8.4f} {final_s:5d} {drop:8.4f} {rtype:>6s}")
    
    # Group by family
    family = short.rsplit('_seed', 1)[0]
    if family not in group_data:
        group_data[family] = []
    group_data[family].append({
        'seed': short.rsplit('_seed', 1)[1],
        'best': best, 'best_step': best_s,
        'final': final, 'final_step': final_s,
        'drop': drop, 'max_step': max_step,
        'final_vals': final_vals,
    })

print("\n" + "=" * 120)
print("GROUP MEANS (sorted by mean_final)")
print("=" * 120)
print(f"  {'Family':<55s} {'n':>3s} {'mean_best':>10s} {'mean_final':>11s} {'std_final':>10s} {'mean_drop':>10s} {'status':>12s}")
print("  " + "-" * 115)

group_list = []
for family, items in sorted(group_data.items()):
    n = len(items)
    mean_best = sum(r['best'] for r in items) / n
    mean_final = sum(r['final'] for r in items) / n
    std_final = statistics.stdev([r['final'] for r in items]) if n > 1 else 0
    mean_drop = sum(r['drop'] for r in items) / n
    status = "COMPLETE" if all(r['max_step'] >= 300 for r in items) else "PARTIAL"
    seeds = [r['seed'] for r in items]
    group_list.append((family, n, mean_best, mean_final, std_final, mean_drop, status, seeds))

for family, n, mb, mf, sf, md, st, seeds in sorted(group_list, key=lambda x: -x[3]):
    print(f"  {family:<55s} {n:3d} {mb:10.4f} {mf:11.4f} {sf:10.4f} {md:10.4f} {st:>12s}")

# Val trajectory for linear runs
print("\n" + "=" * 120)
print("VALIDATION TRAJECTORY: linear_1e-5_to_3.1e-6 (all 3 seeds)")
print("=" * 120)
for short, max_step, val_data, lr_data, n_lines, rtype in all_runs:
    if 'linear_1e-5' in short and val_data:
        print(f"\n  {short}:")
        for gs, avg5, _ in val_data:
            bar = "#" * int(avg5 * 100)
            print(f"    step={gs:4d}  avg5={avg5:.4f}  {bar}")

# Cosine 1e-5 truncated trajectory
print("\n" + "=" * 120)
print("VALIDATION TRAJECTORY: cosine_1e-5_to_3.1e-6 (TRUNCATED @169)")
print("=" * 120)
for short, max_step, val_data, lr_data, n_lines, rtype in all_runs:
    if 'cosine_1e-5' in short and val_data:
        print(f"\n  {short}:")
        for gs, avg5, _ in val_data:
            bar = "#" * int(avg5 * 100)
            print(f"    step={gs:4d}  avg5={avg5:.4f}  {bar}")
