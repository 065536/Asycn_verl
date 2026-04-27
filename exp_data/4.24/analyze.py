"""
Phase 2 Triangle Comparison Analysis
M: fixed LR 2.97e-6 (matched_alpha_2.97e-6)
A: sign-gate gamma=0.5 (sign_gate_meanmatch_gamma0.5)
B: continuous r-shaping (sigfrac_cfixed_lr1.25e-5)
All three have matched mean alpha ~2.97e-6.
"""

import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/data/250010176/codes/verl/exp_data/4.24")
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

EXPS = {
    "M": "deepseek1.5b_sync_8gpu_matched_alpha_2.97e-6",
    "A": "deepseek1.5b_sync_8gpu_sign_gate_meanmatch_gamma0.5",
    "B": "deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5",
}
COLORS = {"M": "#4C72B0", "A": "#DD8452", "B": "#55A868"}
NAMES  = {"M": "M: Fixed LR (2.97e-6)", "A": "A: Sign-gate (γ=0.5)", "B": "B: Continuous r-shaping"}

def load_metric(category, metric_substr):
    for f in sorted(glob.glob(str(BASE / category / "*.csv"))):
        if metric_substr in open(f).readline():
            df = pd.read_csv(f)
            renamed = {"step": "step"}
            for k, exp in EXPS.items():
                for col in df.columns:
                    if exp in col:
                        renamed[col] = k
            cols = ["step"] + [k for k in EXPS if k in renamed.values()]
            return df.rename(columns=renamed)[cols]
    return None

def smooth(s, w=5):
    return s.rolling(window=w, min_periods=1, center=True).mean()

def plot_grid(panels, fname, title, figsize=(15, 9), smooth_w=1):
    """panels: list of (df, title, ylabel, [axhline_val])"""
    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, panel in enumerate(panels):
        df, ptitle, ylabel = panel[:3]
        hline = panel[3] if len(panel) > 3 else None
        ax = axes[i]
        if df is not None:
            for k in EXPS:
                if k in df.columns:
                    y = smooth(df[k], smooth_w) if smooth_w > 1 else df[k]
                    ax.plot(df["step"], y, color=COLORS[k], label=NAMES[k], linewidth=1.8)
        if hline is not None:
            ax.axhline(hline, color='gray', linestyle='--', alpha=0.6)
        ax.set_title(ptitle, fontsize=10, fontweight='bold')
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

# ── Load all metrics ───────────────────────────────────────────────────────────
df_aime     = load_metric("val-core", "AIME/acc/mean@16")
df_aime25   = load_metric("val-core", "AIME2025/acc/mean@16")
df_gpqa     = load_metric("val-core", "gpqa/acc/mean@16")
df_minerva  = load_metric("val-core", "MINERVA/acc/mean@16")
df_olympiad = load_metric("val-core", "OLYMPIAD_BENCH/acc/mean@16")
df_entropy  = load_metric("actor", "actor/entropy")
df_kl       = load_metric("actor", "ppo_kl_mean")
df_gradnorm = load_metric("actor", "actor/grad_norm_step")
df_gradpre  = load_metric("actor", "grad_norm_pre_clip")
df_resplen  = load_metric("response_len", "response_length/mean")
df_alpha    = load_metric("actor", "alpha_t")
df_rhat     = load_metric("actor", "r_hat_step")
df_rraw     = load_metric("actor", "r_hat_raw")
df_rctrl    = load_metric("actor", "r_ctrl")
df_ct       = load_metric("actor", "c_t")
df_phibar   = load_metric("actor", "phi_bar")
df_gdot     = load_metric("actor", "g_dot_positive")
df_reward   = load_metric("critic", "rewards/mean")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Full Benchmark (5 tasks)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
val_specs = [
    (df_aime,     "AIME24",        "acc/mean@16"),
    (df_aime25,   "AIME2025",      "acc/mean@16"),
    (df_olympiad, "OLYMPIAD BENCH","acc/mean@16"),
    (df_gpqa,     "GPQA",          "acc/mean@16"),
    (df_minerva,  "MINERVA",       "acc/mean@16"),
    (df_reward,   "Training Reward","reward"),
]
for i, (df, title, ylabel) in enumerate(val_specs):
    ax = axes[i]
    if df is not None:
        for k in EXPS:
            if k in df.columns:
                y = smooth(df[k], 3) if title == "Training Reward" else df[k]
                ax.plot(df["step"], y, 'o-' if title != "Training Reward" else '-',
                        color=COLORS[k], label=NAMES[k], linewidth=1.8, markersize=3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Step", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
axes[5].set_visible(True)
plt.suptitle("Phase 2 Triangle Comparison — Full Benchmark Suite", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / "1_full_benchmark.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 1_full_benchmark.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Training Dynamics (entropy, KL, grad norm, response len)
# ═══════════════════════════════════════════════════════════════════════════════
plot_grid([
    (df_entropy,  "Entropy",                    "entropy",      None),
    (df_kl,       "PPO KL (approx)",            "KL",           None),
    (df_gradnorm, "Grad Norm (post-clip)",       "‖∇‖",          None),
    (df_gradpre,  "Grad Norm (pre-clip)",        "‖∇‖ pre-clip", None),
    (df_resplen,  "Response Length",             "tokens",       None),
    (df_reward,   "Training Reward",             "reward",       None),
], "2_training_dynamics.png", "Phase 2 — Training Dynamics", smooth_w=5)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — LR Dynamics
# ═══════════════════════════════════════════════════════════════════════════════
# alpha_t distribution violin
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

ax = axes[0]
if df_alpha is not None:
    for k in EXPS:
        if k in df_alpha.columns:
            ax.plot(df_alpha["step"], df_alpha[k], color=COLORS[k], label=NAMES[k], linewidth=1.5)
ax.axhline(2.97e-6, color='gray', linestyle='--', alpha=0.5, label='target 2.97e-6')
ax.set_title("α_t trajectory", fontsize=10, fontweight='bold')
ax.set_xlabel("Step"); ax.set_ylabel("LR"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

ax = axes[1]
if df_alpha is not None:
    data, labels = [], []
    for k in EXPS:
        if k in df_alpha.columns:
            data.append(df_alpha[df_alpha["step"] > 20][k].dropna().values)
            labels.append(k)
    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True)
    for pc, k in zip(parts['bodies'], labels):
        pc.set_facecolor(COLORS[k]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([NAMES[k] for k in labels], fontsize=7)
ax.set_title("α_t Distribution (step > 20)", fontsize=10, fontweight='bold')
ax.set_ylabel("LR"); ax.grid(alpha=0.3)

ax = axes[2]
if df_rhat is not None:
    for k in EXPS:
        if k in df_rhat.columns:
            ax.plot(df_rhat["step"], df_rhat[k], color=COLORS[k], label=NAMES[k], linewidth=1.5)
ax.set_title("r̂_t (EMA)", fontsize=10, fontweight='bold')
ax.set_xlabel("Step"); ax.set_ylabel("r̄_t"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

ax = axes[3]
if df_rraw is not None:
    for k in EXPS:
        if k in df_rraw.columns:
            ax.plot(df_rraw["step"], smooth(df_rraw[k], 10), color=COLORS[k], label=NAMES[k], linewidth=1.5)
ax.set_title("r̂_t raw (smoothed w=10)", fontsize=10, fontweight='bold')
ax.set_xlabel("Step"); ax.set_ylabel("r̂_t raw"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

ax = axes[4]
if df_ct is not None:
    for k in EXPS:
        if k in df_ct.columns:
            ax.plot(df_ct["step"], df_ct[k], color=COLORS[k], label=NAMES[k], linewidth=1.5)
ax.set_title("c_t (scale factor)", fontsize=10, fontweight='bold')
ax.set_xlabel("Step"); ax.set_ylabel("c_t"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

ax = axes[5]
if df_phibar is not None:
    for k in EXPS:
        if k in df_phibar.columns:
            ax.plot(df_phibar["step"], df_phibar[k], color=COLORS[k], label=NAMES[k], linewidth=1.8)
ax.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='φ* = 0.5')
ax.set_title("φ̄_t (realization ratio EMA)", fontsize=10, fontweight='bold')
ax.set_xlabel("Step"); ax.set_ylabel("φ̄_t"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

plt.suptitle("Phase 2 — LR & r-signal Dynamics", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / "3_lr_dynamics.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 3_lr_dynamics.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Sign alignment
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
if df_gdot is not None:
    for k in ["M", "A"]:
        if k in df_gdot.columns:
            ax.plot(df_gdot["step"], smooth(df_gdot[k], 20), color=COLORS[k], label=NAMES[k], linewidth=1.8)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='random baseline')
ax.set_title("P(g_dot > 0) — smoothed w=20", fontsize=10, fontweight='bold')
ax.set_xlabel("Step"); ax.set_ylabel("Fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

ax = axes[1]
if df_gdot is not None:
    for k in ["M", "A"]:
        if k in df_gdot.columns:
            vals = df_gdot[k].dropna()
            ax.hist(vals, bins=20, color=COLORS[k], alpha=0.6, label=NAMES[k], density=True)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
ax.set_title("P(g_dot > 0) Distribution", fontsize=10, fontweight='bold')
ax.set_xlabel("Fraction"); ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("Phase 2 — Gradient Alignment Signal", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / "4_alignment_signal.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 4_alignment_signal.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════
val_dfs = {"AIME24": df_aime, "AIME2025": df_aime25,
           "GPQA": df_gpqa, "MINERVA": df_minerva, "OLYMPIAD": df_olympiad}

stats = {k: {} for k in EXPS}
for label, df in val_dfs.items():
    for k in EXPS:
        if df is not None and k in df.columns:
            stats[k][f"{label}_peak"]  = df[k].max()
            stats[k][f"{label}_final"] = df[k].iloc[-1]

for k in EXPS:
    if df_alpha is not None and k in df_alpha.columns:
        post = df_alpha[df_alpha["step"] > 20][k].dropna()
        stats[k]["alpha_mean"] = post.mean()
        stats[k]["alpha_std"]  = post.std()
    if df_entropy is not None and k in df_entropy.columns:
        stats[k]["entropy_final"] = df_entropy[k].dropna().iloc[-1]
        stats[k]["entropy_mean"]  = df_entropy[k].dropna().mean()
    if df_kl is not None and k in df_kl.columns:
        stats[k]["kl_mean"]  = df_kl[k].dropna().mean()
        stats[k]["kl_final"] = df_kl[k].dropna().iloc[-1]
    if df_gradnorm is not None and k in df_gradnorm.columns:
        stats[k]["gradnorm_mean"] = df_gradnorm[k].dropna().mean()
    if df_resplen is not None and k in df_resplen.columns:
        stats[k]["resplen_final"] = df_resplen[k].dropna().iloc[-1]
    if df_gdot is not None and k in df_gdot.columns:
        stats[k]["gdot_pos_mean"] = df_gdot[k].dropna().mean()
    if df_phibar is not None and k in df_phibar.columns:
        stats[k]["phi_bar_final"] = df_phibar[k].dropna().iloc[-1]

print("\n=== Summary Statistics ===")
rows = []
metrics = [
    ("AIME24_peak",    "AIME24 peak"),
    ("AIME24_final",   "AIME24 final"),
    ("AIME2025_peak",  "AIME2025 peak"),
    ("AIME2025_final", "AIME2025 final"),
    ("OLYMPIAD_peak",  "OLYMPIAD peak"),
    ("OLYMPIAD_final", "OLYMPIAD final"),
    ("GPQA_peak",      "GPQA peak"),
    ("GPQA_final",     "GPQA final"),
    ("MINERVA_peak",   "MINERVA peak"),
    ("MINERVA_final",  "MINERVA final"),
    ("alpha_mean",     "α_t mean (>20)"),
    ("alpha_std",      "α_t std (>20)"),
    ("entropy_mean",   "entropy mean"),
    ("entropy_final",  "entropy final"),
    ("kl_mean",        "KL mean"),
    ("kl_final",       "KL final"),
    ("gradnorm_mean",  "grad norm mean"),
    ("resplen_final",  "resp len final"),
    ("gdot_pos_mean",  "P(g_dot>0) mean"),
    ("phi_bar_final",  "φ̄_t final"),
]
for key, label in metrics:
    row = {"metric": label}
    print(f"{label:<22}", end="")
    for k in EXPS:
        v = stats[k].get(key, float('nan'))
        row[k] = v
        fmt = f"{v:>10.2e}" if "alpha" in key else f"{v:>10.4f}"
        print(fmt, end="")
    print()
    rows.append(row)

pd.DataFrame(rows).to_csv(OUT / "summary_stats.csv", index=False)
print(f"\nAll figures saved to: {OUT}")
