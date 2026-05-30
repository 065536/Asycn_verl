import json, os, statistics

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
files = sorted([f for f in os.listdir(base) if f.endswith('.jsonl')])

CORE_TASKS = [
    'val-aux/AIME/acc/mean@16/mean',
    'val-aux/AIME2025/acc/mean@16/mean',
    'val-aux/Idavidrein/gpqa/acc/mean@16/mean',
    'val-aux/MINERVA/acc/mean@16/mean',
    'val-aux/OLYMPIAD_BENCH/acc/mean@16/mean',
]

# First: discover the correct val key names from one file
sample_path = os.path.join(base, files[-1])
sample_lines = open(sample_path).readlines()
for line in sample_lines:
    rec = json.loads(line)
    if 'data' in rec:
        d = rec['data']
    else:
        d = rec
    gs = rec.get('step', d.get('global_step', -1))
    aime_keys = [k for k in d.keys() if 'AIME' in k and 'mean@16' in k and k.endswith('/mean')]
    if aime_keys:
        print("Discovered val keys with mean@16/mean:")
        all_val = [k for k in d.keys() if 'mean@16' in k and k.endswith('/mean') and ('AIME' in k or 'MINERVA' in k or 'OLYMPIAD' in k or 'gpqa' in k)]
        for k in sorted(all_val):
            print(f"  {k}: {d[k]}")
        CORE_TASKS = sorted(all_val)
        break

print(f"\nUsing {len(CORE_TASKS)} core task keys")

# Now parse all files
print("\n" + "=" * 110)
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
        if 'data' in rec:
            d = rec['data']
            gs = rec.get('step', -1)
        else:
            d = rec
            gs = d.get('global_step', d.get('step', -1))
        
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
    
    complete = "COMPLETE" if max_step >= 300 else f"TRUNCATED (max_step={max_step})"
    n_val = len(val_data)
    print(f"  {short:<45s}  lines={len(lines):4d}  max_step={max_step:4d}  n_val={n_val:3d}  {complete}")
    
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

# Summary table
print("\n" + "=" * 110)
print("VALIDATION SUMMARY (5-task core avg)")
print("=" * 110)
print(f"  {'Config':<45s} {'best':>8s} {'@step':>6s} {'final':>8s} {'@step':>6s} {'drop':>8s} {'n_val':>6s} {'status':>10s}")
print("  " + "-" * 100)

families = {}
for short, r in sorted(all_results.items()):
    family = short.rsplit('_seed', 1)[0]
    if family not in families:
        families[family] = []
    families[family].append((short, r))
    status = "OK" if r['max_step'] >= 300 else "TRUNC"
    print(f"  {short:<45s} {r['best_avg5']:8.4f} {r['best_step']:6d} {r['final_avg5']:8.4f} {r['final_step']:6d} {r['drop']:8.4f} {r['n_val']:6d} {status:>10s}")

# Group means
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
    std_final = statistics.stdev(finals) if n > 1 else 0
    final_steps = set(r['final_step'] for _, r in items)
    seeds = [s.rsplit('_seed', 1)[1] for s, _ in items]
    print(f"  {family}")
    print(f"    seeds={seeds}  mean_best={mean_best:.4f}  mean_final={mean_final:.4f} +/- {std_final:.4f}  mean_drop={mean_drop:.4f}  final_steps={final_steps}")

# Per-task breakdown at final step
print("\n" + "=" * 110)
print("PER-TASK FINAL VALUES")
print("=" * 110)
task_short = [k.split('/')[1] for k in CORE_TASKS]
header = f"  {'Config':<45s}" + "".join(f"{t:>12s}" for t in task_short) + f"{'avg5':>10s}"
print(header)
print("  " + "-" * (45 + 12*len(task_short) + 10))
for short, r in sorted(all_results.items()):
    line = f"  {short:<45s}"
    for v in r['final_vals']:
        line += f"{v:12.4f}"
    line += f"{r['final_avg5']:10.4f}"
    print(line)

# LR trajectory for each config
print("\n" + "=" * 110)
print("LR TRAJECTORY (sampled)")
print("=" * 110)
for short, r in sorted(all_results.items()):
    lr_data = r['lr_data']
    if not lr_data:
        continue
    print(f"\n  {short}: {len(lr_data)} LR points")
    # sample at steps 1,5,10,20,50,100,150,200,250,300
    sample_steps = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    lr_dict = {gs: lr for gs, lr in lr_data}
    for s in sample_steps:
        if s in lr_dict:
            print(f"    step={s:4d}  lr={lr_dict[s]:.6e}")
        elif s <= max(gs for gs, _ in lr_data):
            # find closest
            closest = min(lr_data, key=lambda x: abs(x[0] - s))
            print(f"    step~{closest[0]:4d}  lr={closest[1]:.6e}")

# Reference: constant LR runs from deepseek1.5b_lr
print("\n" + "=" * 110)
print("REFERENCE: constant LR runs (from deepseek1.5b_lr/)")
print("=" * 110)
ref_base = '/data/250010176/codes/verl/deepseek1.5b_lr'
ref_patterns = ['diag_constant_lr1e-5', 'diag_constant_lr3.1e-6']
for pat in ref_patterns:
    ref_files = sorted([f for f in os.listdir(ref_base) if pat in f and f.endswith('.jsonl')])
    for fn in ref_files:
        path = os.path.join(ref_base, fn)
        lines = open(path).readlines()
        short = fn.replace('deepseek1.5b_sync_8gpu_', '').replace('.jsonl', '')
        max_step = 0
        val_data = []
        for line in lines:
            rec = json.loads(line)
            if 'data' in rec:
                d = rec['data']
                gs = rec.get('step', -1)
            else:
                d = rec
                gs = d.get('global_step', d.get('step', -1))
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
            print(f"  {short:<45s}  best={best:.4f}@{best_s}  final={final:.4f}@{final_s}  drop={drop:.4f}  n_val={len(val_data)}")
