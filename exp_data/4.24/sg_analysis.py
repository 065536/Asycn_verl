import pandas as pd, numpy as np, glob

BASE = '/data/250010176/codes/verl/exp_data/4.24'
EXPS = {'M': 'deepseek1.5b_sync_8gpu_matched_alpha_2.97e-6',
        'A': 'deepseek1.5b_sync_8gpu_sign_gate_meanmatch_gamma0.5',
        'B': 'deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5'}

def load(cat, substr):
    for f in sorted(glob.glob(f'{BASE}/{cat}/*.csv')):
        if substr in open(f).readline():
            df = pd.read_csv(f)
            ren = {'step':'step'}
            for k,exp in EXPS.items():
                for col in df.columns:
                    if exp in col: ren[col]=k
            cols = ['step']+[k for k in EXPS if k in ren.values()]
            return df.rename(columns=ren)[cols]

gdot  = load('actor','g_dot_positive')
alpha = load('actor','alpha_t')
aime  = load('val-core','AIME/acc/mean@16')
aime25= load('val-core','AIME2025/acc/mean@16')
gpqa  = load('val-core','gpqa/acc/mean@16')
olymp = load('val-core','OLYMPIAD_BENCH/acc/mean@16')
ent   = load('actor','actor/entropy')
kl    = load('actor','ppo_kl_mean')
rlen  = load('response_len','response_length/mean')

# 1. A 正负两档 alpha 实测
merged = alpha[['step','A']].merge(gdot[['step','A']].rename(columns={'A':'gdot'}),on='step')
merged = merged[merged.step>20]
pos = merged[merged.gdot==1.0]['A']
neg = merged[merged.gdot==0.0]['A']
print('=== 1. A: alpha_t by g_dot sign (post-handoff) ===')
print(f'  g_dot>0  n={len(pos):3d}  mean={pos.mean():.3e}  std={pos.std():.3e}')
print(f'  g_dot<=0 n={len(neg):3d}  mean={neg.mean():.3e}  std={neg.std():.3e}')
print(f'  ratio hi/lo = {pos.mean()/neg.mean():.3f}  (design = 2.0)')
print(f'  P(g_dot>0) post-20 = {len(pos)/(len(pos)+len(neg)):.3f}')

# 2. g_dot smoothed 趋势与 AIME 曲线对照
print()
print('=== 2. g_dot_smooth(20) at each AIME val step ===')
g_sm = gdot['A'].rolling(20,min_periods=5).mean().reset_index(drop=True)
for _, row in aime.iterrows():
    s = int(row['step'])
    if s < 20: continue
    idx_list = gdot.index[gdot.step==s].tolist()
    if idx_list:
        print(f'  step={s:3d}  AIME_A={row["A"]:.4f}  gdot_sm={g_sm.iloc[idx_list[0]]:.3f}')

# 3. Phase-wise: alpha, entropy, KL, g_dot
print()
print('=== 3. Phase-wise comparison (A vs M) ===')
for lbl,lo,hi in [('early 21-100',21,100),('mid 101-200',101,200),('late 201-300',201,300)]:
    print(f'\n  [{lbl}]')
    for k in ['M','A']:
        a = alpha[(alpha.step>=lo)&(alpha.step<=hi)][k].dropna().mean()
        e = ent[(ent.step>=lo)&(ent.step<=hi)][k].dropna().mean()
        k2 = kl[(kl.step>=lo)&(kl.step<=hi)][k].dropna().mean()
        r = rlen[(rlen.step>=lo)&(rlen.step<=hi)][k].dropna().mean()
        print(f'    {k}: alpha={a:.3e}  entropy={e:.4f}  KL={k2:.5f}  resplen={r:.0f}')
    if 'A' in gdot.columns:
        g = gdot[(gdot.step>=lo)&(gdot.step<=hi)]['A'].dropna().mean()
        gm = gdot[(gdot.step>=lo)&(gdot.step<=hi)]['M'].dropna().mean() if 'M' in gdot.columns else float('nan')
        print(f'    A g_dot>0={g:.3f}   M g_dot>0={gm:.3f}')

# 4. 对 AIME vs GPQA：A 的相对表现
print()
print('=== 4. A relative to M: final performance ===')
benchmarks = [
    ('AIME24',   aime),
    ('AIME2025', aime25),
    ('GPQA',     gpqa),
    ('OLYMPIAD', olymp),
]
for name, df in benchmarks:
    if df is None: continue
    mM = df['M'].iloc[-1] if 'M' in df.columns else float('nan')
    mA = df['A'].iloc[-1] if 'A' in df.columns else float('nan')
    mB = df['B'].iloc[-1] if 'B' in df.columns else float('nan')
    print(f'  {name:10s}  M={mM:.4f}  A={mA:.4f}  delta_A={mA-mM:+.4f}  B={mB:.4f}  delta_B={mB-mM:+.4f}')

# 5. 游程统计
print()
print('=== 5. Run-length analysis (A g_dot sequence) ===')
vals = gdot['A'].dropna().values
runs_p, runs_n = [], []
cur, clen = vals[0], 1
for v in vals[1:]:
    if v == cur: clen += 1
    else:
        (runs_p if cur==1 else runs_n).append(clen)
        cur, clen = v, 1
(runs_p if cur==1 else runs_n).append(clen)
print(f'  Positive runs: n={len(runs_p)}, mean={np.mean(runs_p):.2f}, median={np.median(runs_p):.0f}, max={max(runs_p)}')
print(f'  Negative runs: n={len(runs_n)}, mean={np.mean(runs_n):.2f}, median={np.median(runs_n):.0f}, max={max(runs_n)}')
long_pos = sum(1 for r in runs_p if r>=3)
long_neg = sum(1 for r in runs_n if r>=3)
print(f'  Long pos runs (>=3): {long_pos} ({long_pos/len(runs_p)*100:.0f}%)')
print(f'  Long neg runs (>=3): {long_neg} ({long_neg/len(runs_n)*100:.0f}%)')

# 6. late-stage g_dot collapse check
print()
print('=== 6. g_dot_positive 分段均值（细粒度）===')
for lo in range(1,301,20):
    hi = min(lo+19, 300)
    seg_a = gdot[(gdot.step>=lo)&(gdot.step<=hi)]['A'].dropna()
    seg_m = gdot[(gdot.step>=lo)&(gdot.step<=hi)]['M'].dropna() if 'M' in gdot.columns else pd.Series([])
    if len(seg_a):
        print(f'  step {lo:3d}-{hi:3d}:  A={seg_a.mean():.3f}  M={seg_m.mean():.3f}')
