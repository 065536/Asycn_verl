import json

path = '/data/250010176/codes/verl/deepseek1.5b_lr_cd/deepseek1.5b_sync_8gpu_cd_linear_1e-5_to_3.1e-6_seed42.jsonl'
lines = open(path).readlines()

# Check last line keys
d = json.loads(lines[-1])
keys = sorted(d.keys())
print(f"Last line (step={d.get('global_step', d.get('step', '?'))}), {len(keys)} keys:")
for k in keys:
    print(f"  {k}: {d[k]}")

print("\n\n--- Check line with step 0 (first val) ---")
d0 = json.loads(lines[0])
keys0 = sorted(d0.keys())
print(f"First line (step={d0.get('global_step', d0.get('step', '?'))}), {len(keys0)} keys:")
for k in keys0[:30]:
    print(f"  {k}: {d0[k]}")
if len(keys0) > 30:
    print(f"  ... ({len(keys0)-30} more)")

# Check a line near step 10 (first val point with test_freq=10)
for i, line in enumerate(lines[:15]):
    d = json.loads(line)
    gs = d.get('global_step', d.get('step', -1))
    if gs == 10:
        print(f"\n\n--- Line at step 10 ---")
        for k in sorted(d.keys()):
            if 'val' in k.lower() or 'AIME' in k or 'acc' in k.lower() or 'lr' in k.lower():
                print(f"  {k}: {d[k]}")
        break
