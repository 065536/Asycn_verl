import json, os

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
CORE_5 = [
    'val-core/AIME/acc/mean@16',
    'val-core/AIME2025/acc/mean@16',
    'val-core/Idavidrein/gpqa/acc/mean@16',
    'val-core/MINERVA/acc/mean@16',
    'val-core/OLYMPIAD_BENCH/acc/mean@16',
]

print("=== cosine 5e-6→3.1e-6 (step 300 only) ===")
for seed in ['0', '1', '42']:
    fn = f'deepseek1.5b_sync_8gpu_cd_cosine_5e-6_to_3.1e-6_seed{seed}.jsonl'
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    rec = json.loads(lines[-1])
    d = rec.get('data', rec)
    gs = rec.get('step', -1)
    vals = [d.get(k) for k in CORE_5]
    if all(v is not None for v in vals):
        avg5 = sum(vals) / 5
        print(f"  seed{seed} step={gs}: AIME={vals[0]:.4f} AIME25={vals[1]:.4f} GPQA={vals[2]:.4f} MINERVA={vals[3]:.4f} OLYMP={vals[4]:.4f}  avg5={avg5:.4f}")
    else:
        # Maybe different key format
        print(f"  seed{seed}: missing some keys. Available val-core keys:")
        vc = [k for k in d.keys() if k.startswith('val-core')]
        for k in sorted(vc):
            print(f"    {k} = {d[k]}")

print("\n=== linear 1e-5→3.1e-6 (step 300) ===")
for seed in ['0', '1', '42']:
    fn = f'deepseek1.5b_sync_8gpu_cd_linear_1e-5_to_3.1e-6_seed{seed}.jsonl'
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    rec = json.loads(lines[-1])
    d = rec.get('data', rec)
    gs = rec.get('step', -1)
    vals = [d.get(k) for k in CORE_5]
    avg5 = sum(v for v in vals if v is not None) / sum(1 for v in vals if v is not None) if any(v is not None for v in vals) else 0
    print(f"  seed{seed} step={gs}: AIME={vals[0]:.4f} AIME25={vals[1]:.4f} GPQA={vals[2]:.4f} MINERVA={vals[3]:.4f} OLYMP={vals[4]:.4f}  avg5={avg5:.4f}")

print("\n=== cosine 1e-5→3.1e-6 (TRUNCATED, last val step) ===")
for seed in ['0', '1', '42']:
    fn = f'deepseek1.5b_sync_8gpu_cd_cosine_1e-5_to_3.1e-6_seed{seed}.jsonl'
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    # Find last line with val
    for line in reversed(lines):
        rec = json.loads(line)
        d = rec.get('data', rec)
        gs = rec.get('step', -1)
        vals = [d.get(k) for k in CORE_5]
        if all(v is not None for v in vals):
            avg5 = sum(vals) / 5
            print(f"  seed{seed} step={gs}: AIME={vals[0]:.4f} AIME25={vals[1]:.4f} GPQA={vals[2]:.4f} MINERVA={vals[3]:.4f} OLYMP={vals[4]:.4f}  avg5={avg5:.4f}")
            break

print("\n=== Reference: constant LR ===")
ref_base = '/data/250010176/codes/verl/deepseek1.5b_lr'
for tag in ['diag_constant_lr3.1e-6', 'diag_constant_lr1e-5']:
    for seed in ['0', '1', '42']:
        fn = f'deepseek1.5b_sync_8gpu_{tag}_seed{seed}.jsonl'
        path = os.path.join(ref_base, fn)
        if not os.path.exists(path):
            continue
        lines = open(path).readlines()
        # Find best and last val
        best_avg5 = -1
        best_step = -1
        last_avg5 = -1
        last_step = -1
        for line in lines:
            rec = json.loads(line)
            d = rec.get('data', rec)
            gs = rec.get('step', -1)
            vals = [d.get(k) for k in CORE_5]
            if all(v is not None for v in vals):
                avg5 = sum(vals) / 5
                if avg5 > best_avg5:
                    best_avg5 = avg5
                    best_step = gs
                last_avg5 = avg5
                last_step = gs
        short = f"{tag}_seed{seed}"
        print(f"  {short:<40s}  best={best_avg5:.4f}@{best_step}  final={last_avg5:.4f}@{last_step}  drop={last_avg5-best_avg5:.4f}")
