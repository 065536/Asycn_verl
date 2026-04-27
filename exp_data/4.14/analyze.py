#!/usr/bin/env python3
"""Analyze Phase 1 batch 2 + Phase 2 advvar experiment data."""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Short names for readability
SHORT = {
    "deepseek1.5b_sync_8gpu_advvar_ent_lr1e-5": "advvar_ent",
    "deepseek1.5b_sync_8gpu_ent0.1_lr1e-5": "ent0.1",
    "deepseek1.5b_sync_8gpu_ent0.001_lr1e-5": "ent0.001",
    "deepseek1.5b_sync_8gpu_cosine_floor_lr1e-5": "cosine_floor",
    "deepseek1.5b_sync_8gpu_cosine_ent0.01_lr1e-5": "cosine_ent0.01",
}

def load_csv_dir(subdir):
    """Load all CSVs in a subdirectory, return dict of metric_name -> DataFrame."""
    path = os.path.join(DATA_DIR, subdir)
    result = {}
    for f in sorted(os.listdir(path)):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(path, f))
        # Extract metric name from first non-step column
        cols = [c for c in df.columns if c != 'step']
        if not cols:
            continue
        # Get metric name: everything after the experiment name prefix
        sample_col = cols[0]
        # e.g. "deepseek1.5b_sync_8gpu_advvar_ent_lr1e-5-val-core/AIME/acc/mean@16_step"
        # metric = "val-core/AIME/acc/mean@16_step"
        for exp_name in SHORT:
            if sample_col.startswith(exp_name):
                metric = sample_col[len(exp_name)+1:]  # skip the '-'
                break
        else:
            metric = sample_col

        # Rename columns to short names
        renamed = {'step': 'step'}
        for c in cols:
            for exp_name, short in SHORT.items():
                if c.startswith(exp_name):
                    renamed[c] = short
                    break
        df = df.rename(columns=renamed)
        result[metric] = df
    return result

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def analyze_val_core(data):
    """Analyze validation accuracy data."""
    print_section("VALIDATION ACCURACY")

    benchmarks = {}
    for metric, df in data.items():
        # Extract benchmark name
        # e.g. "val-core/AIME/acc/mean@16_step" -> "AIME"
        parts = metric.split('/')
        if len(parts) >= 2:
            bench = parts[1]
        else:
            bench = metric
        benchmarks[bench] = df

    exps = [c for c in list(data.values())[0].columns if c != 'step']

    for bench, df in sorted(benchmarks.items()):
        print(f"\n--- {bench} ---")
        print(f"{'Experiment':<20} {'@step0':>8} {'@step300':>8} {'Peak':>8} {'PeakStep':>8} {'Trend':>12}")
        print("-" * 70)
        for exp in exps:
            vals = df[['step', exp]].dropna()
            if vals.empty:
                continue
            v0 = vals[exp].iloc[0]
            v_last = vals[exp].iloc[-1]
            peak_idx = vals[exp].idxmax()
            peak_val = vals[exp].max()
            peak_step = vals.loc[peak_idx, 'step']

            # Trend: compare last 5 values
            last5 = vals[exp].iloc[-5:]
            if len(last5) > 1:
                slope = np.polyfit(range(len(last5)), last5.values, 1)[0]
                if slope > 0.005:
                    trend = "Rising"
                elif slope < -0.005:
                    trend = "Declining"
                else:
                    trend = "Stable"
            else:
                trend = "N/A"

            print(f"{exp:<20} {v0:>8.4f} {v_last:>8.4f} {peak_val:>8.4f} {int(peak_step):>8} {trend:>12}")

def analyze_actor(data):
    """Analyze actor metrics."""
    print_section("ACTOR METRICS")

    exps = None
    for metric, df in data.items():
        cols = [c for c in df.columns if c != 'step']
        if exps is None:
            exps = cols

        # Key metrics to analyze in detail
        if 'entropy' in metric:
            print(f"\n--- Entropy ---")
            print(f"{'Experiment':<20} {'@step1':>8} {'@step50':>8} {'@step150':>8} {'@step300':>8} {'Min':>8} {'Max':>8}")
            print("-" * 78)
            for exp in exps:
                vals = df[['step', exp]].dropna()
                s1 = vals[vals['step'] == 1][exp].values
                s50 = vals[vals['step'] == 50][exp].values
                s150 = vals[vals['step'] == 150][exp].values
                s300 = vals[vals['step'] == 300][exp].values
                s1 = s1[0] if len(s1) > 0 else float('nan')
                s50 = s50[0] if len(s50) > 0 else float('nan')
                s150 = s150[0] if len(s150) > 0 else float('nan')
                s300 = s300[0] if len(s300) > 0 else float('nan')
                print(f"{exp:<20} {s1:>8.4f} {s50:>8.4f} {s150:>8.4f} {s300:>8.4f} {vals[exp].min():>8.4f} {vals[exp].max():>8.4f}")

        elif 'ppo_kl' in metric:
            print(f"\n--- PPO KL ---")
            print(f"{'Experiment':<20} {'Mean':>10} {'Max':>10} {'@step300':>10}")
            print("-" * 55)
            for exp in exps:
                vals = df[['step', exp]].dropna()
                s300 = vals[vals['step'] == 300][exp].values
                s300 = s300[0] if len(s300) > 0 else float('nan')
                print(f"{exp:<20} {vals[exp].mean():>10.6f} {vals[exp].max():>10.6f} {s300:>10.6f}")

        elif 'grad_norm' in metric:
            print(f"\n--- Gradient Norm ---")
            print(f"{'Experiment':<20} {'Mean':>10} {'Max':>10} {'@step300':>10}")
            print("-" * 55)
            for exp in exps:
                vals = df[['step', exp]].dropna()
                s300 = vals[vals['step'] == 300][exp].values
                s300 = s300[0] if len(s300) > 0 else float('nan')
                print(f"{exp:<20} {vals[exp].mean():>10.6f} {vals[exp].max():>10.6f} {s300:>10.6f}")

        elif 'lr_step' in metric:
            print(f"\n--- Learning Rate ---")
            print(f"{'Experiment':<20} {'@step10':>12} {'@step50':>12} {'@step150':>12} {'@step300':>12}")
            print("-" * 70)
            for exp in exps:
                vals = df[['step', exp]].dropna()
                def get_at(s):
                    v = vals[vals['step'] == s][exp].values
                    return v[0] if len(v) > 0 else float('nan')
                print(f"{exp:<20} {get_at(10):>12.2e} {get_at(50):>12.2e} {get_at(150):>12.2e} {get_at(300):>12.2e}")

def analyze_critic(data):
    """Analyze critic/reward metrics."""
    print_section("CRITIC / REWARD METRICS")

    for metric, df in data.items():
        cols = [c for c in df.columns if c != 'step']

        if 'score/mean' in metric:
            print(f"\n--- Score (Mean) ---")
            print(f"{'Experiment':<20} {'@step1':>10} {'@step50':>10} {'@step150':>10} {'@step300':>10} {'Max':>10} {'MaxStep':>8}")
            print("-" * 78)
            for exp in cols:
                vals = df[['step', exp]].dropna()
                short = None
                for long_name, s in SHORT.items():
                    if long_name in exp:
                        short = s
                        break
                if short is None:
                    short = exp[:20]
                def get_at(s):
                    v = vals[vals['step'] == s][exp].values
                    return v[0] if len(v) > 0 else float('nan')
                max_idx = vals[exp].idxmax()
                max_val = vals[exp].max()
                max_step = vals.loc[max_idx, 'step']
                print(f"{short:<20} {get_at(1):>10.4f} {get_at(50):>10.4f} {get_at(150):>10.4f} {get_at(300):>10.4f} {max_val:>10.4f} {int(max_step):>8}")

def analyze_response_len(data):
    """Analyze response length."""
    print_section("RESPONSE LENGTH")

    for metric, df in data.items():
        cols = [c for c in df.columns if c != 'step']
        print(f"\n--- {metric} ---")
        print(f"{'Experiment':<20} {'@step1':>8} {'@step50':>8} {'@step150':>8} {'@step300':>8} {'Min':>8} {'Max':>8}")
        print("-" * 75)
        for exp in cols:
            vals = df[['step', exp]].dropna()
            short = None
            for long_name, s in SHORT.items():
                if long_name in exp:
                    short = s
                    break
            if short is None:
                short = exp[:20]
            def get_at(s):
                v = vals[vals['step'] == s][exp].values
                return v[0] if len(v) > 0 else float('nan')
            print(f"{short:<20} {get_at(1):>8.0f} {get_at(50):>8.0f} {get_at(150):>8.0f} {get_at(300):>8.0f} {vals[exp].min():>8.0f} {vals[exp].max():>8.0f}")


if __name__ == '__main__':
    print("=" * 80)
    print("  Phase 1 Batch 2 + Phase 2 (advvar) Experiment Analysis")
    print("  Data: exp_data/4.14/")
    print("=" * 80)

    val_core = load_csv_dir('val_core')
    actor = load_csv_dir('actor')
    critic = load_csv_dir('critic')
    response_len = load_csv_dir('response_len')

    analyze_val_core(val_core)
    analyze_actor(actor)
    analyze_critic(critic)
    analyze_response_len(response_len)
