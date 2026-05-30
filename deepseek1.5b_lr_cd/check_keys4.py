import json, os

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
path = os.path.join(base, 'deepseek1.5b_sync_8gpu_cd_linear_1e-5_to_3.1e-6_seed42.jsonl')
lines = open(path).readlines()

# Step 10 should have val data
rec = json.loads(lines[10])
d = rec['data']
gs = rec['step']

# Show all val keys containing mean@16
print(f"Step {gs}:")
mean16_keys = sorted([k for k in d.keys() if 'mean@16' in k])
print(f"  Keys with 'mean@16': {len(mean16_keys)}")
for k in mean16_keys:
    print(f"    {k} = {d[k]}")

# Also check for acc/mean
acc_mean_keys = sorted([k for k in d.keys() if '/acc/' in k and 'mean' in k])[:20]
print(f"\n  Keys with '/acc/' and 'mean' ({len(acc_mean_keys)} shown):")
for k in acc_mean_keys:
    print(f"    {k} = {d[k]}")
