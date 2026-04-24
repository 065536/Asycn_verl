---
name: Collaboration Feedback
description: User's preferences about how to work together, from explicit corrections and confirmations
type: feedback
---

---
name: Collaboration Feedback
description: User's preferences about how to work together, from explicit corrections and confirmations
type: feedback
---

**Demand rigorous derivations, no logical gaps.**
Every non-trivial step must be proved or cited. "By standard result" is not acceptable unless the result is actually trivial. User will catch gaps and ask for them to be filled.
**Why**: User caught multiple flaws (SNR argument, scalar vs vector MSE, wrong formula for α*).
**How to apply**: Before claiming any result, check that each step is either (a) a definition, (b) a standard algebra step, or (c) an explicit citation with equation number.

**Do not reverse-engineer theory from the solution.**
The framework must be derived from the problem statement, not from "how do we justify the existing method."
**Why**: User explicitly flagged entadapt as "照着答案来设计".
**How to apply**: When proposing an algorithm, derive it from α*(t) formula. When writing theory, start from the objective, not from "what justifies LR decay."

**When proposing algorithms, flag non-generalizing assumptions immediately.**
If a method requires a hyperparameter that encodes problem-specific knowledge, say so upfront rather than presenting it as a clean solution.
**Why**: User caught that α_max in all current proposals is still a problem-specific fixed value.
**How to apply**: Always state what each hyperparameter represents and what problem-specific knowledge it encodes.

**Communicate in Chinese during discussion sessions.**
**Why**: All discussion messages from the user are in Chinese.
**How to apply**: Respond in Chinese for conceptual/discussion messages; code and math can stay in English/LaTeX.

**Always verify alpha_t logs match expectations before analyzing results.**
After code/config changes, confirm alpha_t trajectory in first 15–20 steps before reading val scores. The project has repeatedly had "logic looks right but execution is wrong" bugs (Bug 13: c_t frozen all 300 steps; Bug 14/15: phi_t wrong; warmup/handoff discontinuity).
**Why**: Multiple prior experiments wasted compute because the actual LR being applied was not what was intended.
**How to apply**: For any new experiment config, define expected alpha_t behavior per phase (warmup / handoff / post-handoff) and explicitly check logs match before proceeding.

**Look at dynamics first, final scores second.**
When analyzing multi-seed or multi-config experiments, first confirm the mechanics work as intended (alpha_t range, r_hat distribution, c_t stability), then ask whether the signal is stable vs just conservative.
**Why**: "More conservative" and "more stable" are different outcomes; a method that just reduces LR uniformly would also look stable but provides no insight.
**How to apply**: For Phase 1/2 cfixed experiments, check (1) effective LR range separates as expected across configs, (2) r-side shape is independent of c_fixed, (3) then compare val curves.

**Experimental design: do not test two free variables simultaneously.**
When c_t and r̂_t are both unknown, fix c first (eta_c=0 sweep), then test r independently.
**Why**: User explicitly insisted on this separation: "先找到 c_t 再说 r_t".
**How to apply**: In any new ablation, reduce to one free variable per experiment series.
