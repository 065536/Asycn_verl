import json, os

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'

# Check cosine_5e-6 runs - do they have val at step 300?
for seed in ['0', '1', '42']:
    fn = f'deepseek1.5b_sync_8gpu_cd_cosine_5e-6_to_3.1e-6_seed{seed}.jsonl'
    path = os.path.join(base, fn)
    lines = open(path).readlines()
    
    # count val lines
    n_val = 0
    val_steps = []
    for line in lines:
        rec = json.loads(line)
        d = rec.get('data', rec)
        gs = rec.get('step', d.get('global_step', -1))
        has_val = any('val-core' in k for k in d.keys())
        if has_val:
            n_val += 1
            val_steps.append(gs)
            # Extract values
            vals = {k: d[k] for k in d.keys() if 'val-core' in k}
            if gs == 300 or gs == max(int(lines[-1].split('"step":')[0] or '0') for _ in [1]):
                avg5 = sum(vals.values()) / len(vals) if vals else 0
                print(f"  seed{seed} step {gs}: avg5={avg5:.4f}, vals={vals}")
    
    print(f"cosine_5e-6_seed{seed}: {len(lines)} lines, n_val={n_val}, val_steps={val_steps}")

# Also check: are these the OLD runs from 5/24 (no test_freq=10)?
# Check config dates
print("\n--- Config file dates ---")
for fn in sorted(os.listdir(base)):
    if '5e-6' in fn and fn.endswith('.json'):
        path = os.path.join(base, fn)
        mtime = os.path.getmtime(path)
        import time
        print(f"  {fn}: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}")
