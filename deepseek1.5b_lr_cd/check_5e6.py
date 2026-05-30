import json, os

# Check cosine_5e-6 - why n_val=0?
path = '/data/250010176/codes/verl/deepseek1.5b_lr_cd/deepseek1.5b_sync_8gpu_cd_cosine_5e-6_to_3.1e-6_seed42.jsonl'
lines = open(path).readlines()
print(f"cosine_5e-6_seed42: {len(lines)} lines")

# Check some lines
for i in [0, 10, 50, -1]:
    rec = json.loads(lines[i])
    d = rec.get('data', rec)
    gs = rec.get('step', d.get('global_step', -1))
    has_val = any('val-core' in k for k in d.keys())
    lr = d.get('actor/lr', None)
    n_keys = len(d.keys())
    print(f"  line {i}: step={gs}, n_keys={n_keys}, has_val={has_val}, lr={lr}")

# Check reference constant LR data format
ref_path = '/data/250010176/codes/verl/deepseek1.5b_lr/deepseek1.5b_sync_8gpu_diag_constant_lr3.1e-6_seed42.jsonl'
if os.path.exists(ref_path):
    rlines = open(ref_path).readlines()
    print(f"\nref lr3.1e-6_seed42: {len(rlines)} lines")
    rec = json.loads(rlines[0])
    d = rec.get('data', rec)
    gs = rec.get('step', d.get('global_step', -1))
    has_val = any('val-core' in k for k in d.keys())
    print(f"  first line: step={gs}, has_val={has_val}, top_keys={list(rec.keys())[:5]}")
    
    # Check step 10
    for line in rlines[:15]:
        rec = json.loads(line)
        d = rec.get('data', rec)
        gs = rec.get('step', d.get('global_step', -1))
        if gs == 10:
            has_val = any('val-core' in k for k in d.keys())
            vals = {k: d[k] for k in d.keys() if 'val-core' in k and 'mean@16' in k}
            print(f"  step 10: has_val={has_val}, n_val_keys={len(vals)}")
            for k in sorted(vals)[:5]:
                print(f"    {k} = {vals[k]}")
            break
