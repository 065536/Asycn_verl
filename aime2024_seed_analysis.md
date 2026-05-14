# AIME2024 Four-Seed Quick Analysis

Experiment group: `ratio_of_sums_w10`, seeds `0/1/2/42`.

## Final outcome

All 4 seeds end with **AIME2024 `acc@16 = 0.0`** at the final validation.

## Per-seed summary

- seed 0: initial `0.0662` -> final `0.0000`, wall time `2:09:33`, step300 global_seqlen mean `18,765`
- seed 1: initial `0.0501` -> final `0.0000`, wall time `6:42:53`, step300 global_seqlen mean `95,500`
- seed 2: initial `0.0761` -> final `0.0000`, wall time `2:28:57`, step300 global_seqlen mean `20,366`
- seed 42: initial `0.0393` -> final `0.0000`, wall time `3:53:14`, step300 global_seqlen mean `18,949`

## Interpretation

- This is not just "short run time"; all seeds show a bad end state on AIME2024.
- 3 seeds (`0/2/42`) also show low token volume by step300, consistent with training-quality collapse behavior.
- seed 1 ran longer and kept more tokens, but still ended at `0.0`, so the current setting is unstable for this benchmark.
