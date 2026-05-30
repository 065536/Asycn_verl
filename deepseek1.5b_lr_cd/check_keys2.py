import json, os

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
path = os.path.join(base, 'deepseek1.5b_sync_8gpu_cd_linear_1e-5_to_3.1e-6_seed42.jsonl')
lines = open(path).readlines()

# Find val keys by checking a step that's a multiple of 10
for line in lines:
    d = json.loads(line)
    gs = d.get('global_step', d.get('step', -1))
    if gs == 0:
        val_keys = [k for k in sorted(d.keys()) if 'AIME' in k or 'MINERVA' in k or 'OLYMPIAD' in k or 'gpqa' in k]
        print(f"Step {gs}: val keys found = {len(val_keys)}")
        for k in val_keys:
            print(f"  {k}: {d[k]}")
        break

# Now check step 10
for line in lines:
    d = json.loads(line)
    gs = d.get('global_step', d.get('step', -1))
    if gs == 10:
        val_keys = [k for k in sorted(d.keys()) if 'AIME' in k or 'MINERVA' in k or 'OLYMPIAD' in k or 'gpqa' in k]
        lr_keys = [k for k in sorted(d.keys()) if 'lr' in k.lower()]
        print(f"\nStep {gs}: val keys found = {len(val_keys)}")
        for k in val_keys[:10]:
            print(f"  {k}: {d[k]}")
        print(f"\nStep {gs}: lr keys found = {len(lr_keys)}")
        for k in lr_keys:
            print(f"  {k}: {d[k]}")
        break

# Check what the data dict key looks like
d = json.loads(lines[0])
print(f"\nStep 0 all keys (first 5 chars): {[k[:30] for k in sorted(d.keys())][:50]}")
