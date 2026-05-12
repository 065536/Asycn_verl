# r_hat x pg_loss diagnostic

Data: six local 1.5B signal-fraction controller JSONL files under `deepseek1.5b_lr/`. High/low is defined by median split within each run-stage, so the comparison removes most early/mid/late level confounding.

## +30 step validation avg5 delta

| bucket | high r/high PG | high r/low PG | low r/high PG | low r/low PG |
|---|---:|---:|---:|---:|
| all | +0.0102 (n=40) | +0.0038 (n=45) | +0.0040 (n=38) | +0.0031 (n=36) |
| early | +0.0210 (n=17) | +0.0087 (n=14) | +0.0139 (n=7) | +0.0035 (n=16) |
| mid | +0.0052 (n=11) | +0.0036 (n=18) | +0.0023 (n=18) | +0.0023 (n=13) |
| late | -0.0006 (n=12) | -0.0013 (n=13) | +0.0008 (n=13) | +0.0035 (n=7) |

## +30 step train-score delta

| bucket | high r/high PG | high r/low PG | low r/high PG | low r/low PG |
|---|---:|---:|---:|---:|
| all | +0.1369 (n=449) | +0.0451 (n=389) | +0.0186 (n=373) | +0.0416 (n=388) |
| early | +0.3259 (n=185) | +0.0822 (n=119) | +0.0639 (n=115) | +0.0670 (n=175) |
| mid | +0.0178 (n=134) | +0.0322 (n=167) | -0.0085 (n=166) | +0.0154 (n=133) |
| late | -0.0094 (n=130) | +0.0231 (n=103) | +0.0109 (n=92) | +0.0293 (n=80) |

## Interpretation

- Across all stages, high `r_hat` plus high `pg_loss` has the largest future validation and train-score gain. This supports an opportunity-signal interpretation: alignment is more useful when PPO still has learning pressure.

- In late stage, high `r_hat` plus high `pg_loss` no longer dominates. This argues against a simple rule that always increases LR when both signals are high.

- The safer next controller shape is a base schedule plus a small early/mid residual opportunity multiplier, while late-stage control should rely on safety/tail diagnostics rather than `pg_loss` alone.

