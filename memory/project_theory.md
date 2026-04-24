---
name: Project Theory — n_eff Framework and r_t Signal Fraction
description: Full theoretical framework as of 2026-04-20; formalized in paper/theory_derivation.tex and paper/main_modified.tex Section 4
type: project
---

The formal chain (all steps derived in paper/theory_derivation.tex and paper/main_modified.tex):

1. **MSE**: E[||ĝ-g||²] = tr(Σ)/n — from i.i.d. vector sample mean; cross terms vanish by independence
2. **n_eff**: n_eff(t) = E[K_n] ≈ min(n, exp(H(π_t))) — distinct trajectories = perplexity
3. **Improvement bound** (L-smooth): E[J(θ_{t+1})] - J(θ_t) ≥ D_t · α[r_t - αL/2]
4. **Signal fraction**: r_t = ||g||² / (||g||² + σ²/n_eff) ∈ (0,1] — safe zone condition is α < 2r_t/L
5. **Optimal LR**: α*(t) = r_t/L — monotone decreasing as n_eff decreases
6. **Shape vs scale**: r_t determines shape of α*(t); L (≈ constant) determines absolute scale
7. **Mismatch**: fixed LR eventually exceeds 2α*(t) → expected-negative updates

**Key correction from Session 2**: earlier wrong derivation via SNR ≥ 1 gave α* ∝ √n_eff. Correct result via improvement bound is α* ∝ r_t ∝ n_eff (linear, in noise-dominated regime).

**Key addition from Session 3 (2026-04-20)**:
- r_t is estimable via split-batch estimator: r̂_t = (ĝ_A^T ĝ_B) / ((||ĝ_A||²+||ĝ_B||²)/2)
- Only requirement: same θ_t for both halves (prompts can differ)
- r_t solves the shape problem; scale 1/L requires separate feedback

**Key addition (Session 3, later): c_t feedback signal RESOLVED**:
- φ_t = a_t/(p_t+ε) where p_t = α_t ĝ_B^T ĝ_A (predicted gain), a_t = L_B(θ_{t+1})-L_B(θ_t) (actual gain on held-out B)
- φ* = 1/2 derived from improvement bound: φ_t ≈ 1 - c_t·L/2 when α_t = c_t·r_t; at c_t=1/L → φ*=1/2
- r_t cancels exactly in the φ_t expression — scale controller is decoupled from shape signal
- Controller: c_{t+1} = c_t · exp(η_c(φ̄_t - 1/2))

**Why**: The research question is why fixed LR causes collapse in on-policy RL and how to fix it principally.

**How to apply**: Any new algorithm proposal must be evaluated against this chain. α_t = c_t · r̂_t is the finalized framework, with c_t driven by φ_t targeting φ* = 1/2.
