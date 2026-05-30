import json, os

base = '/data/250010176/codes/verl/deepseek1.5b_lr_cd'
path = os.path.join(base, 'deepseek1.5b_sync_8gpu_cd_linear_1e-5_to_3.1e-6_seed42.jsonl')
lines = open(path).readlines()

# Check structure of first few lines
for i in [0, 1, 10, -1]:
    d = json.loads(lines[i])
    print(f"Line {i}: top-level keys = {list(d.keys())}")
    if 'data' in d:
        inner = d['data']
        if isinstance(inner, dict):
            inner_keys = sorted(inner.keys())
            val_keys = [k for k in inner_keys if 'AIME' in k or 'MINERVA' in k or 'OLYMPIAD' in k or 'gpqa' in k]
            lr_keys = [k for k in inner_keys if 'lr' in k.lower() or 'alpha' in k.lower()]
            print(f"  data has {len(inner_keys)} keys, step={d.get('step', '?')}")
            print(f"  val keys ({len(val_keys)}): {val_keys[:8]}")
            print(f"  lr keys ({len(lr_keys)}): {lr_keys[:8]}")
            if val_keys:
                for k in val_keys[:5]:
                    print(f"    {k} = {inner[k]}")
    print()
