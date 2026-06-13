---
name: Project Theory — From Signal Fraction LR to RL-Aware Optimizer
description: Theoretical framework evolution; original r_t/LR theory retained as historical context; current focus on RL gradient structure and optimizer design
type: project
---

## Current Research Question (2026-06-12)

> Standard Adam treats all gradient coordinates uniformly. RL policy gradients have specific structure (positive/negative advantage decomposition, prompt-level heterogeneity) that Adam does not exploit. Can we improve RLVR training by designing an optimizer that leverages this structure?

**Key shift**: the problem is not the learning rate scalar α, but the gradient **direction** after Adam's preconditioning D_t·m_t.

### Why LR is not the answer

1. α_t = c_t · r̂_t framework is theoretically sound but practically blocked: r̂_t estimation is fundamentally noisy in small-r_t regime (SNR ≈ 1/r_t - 1 ≈ 49× at r_t=0.02)
2. Best adaptive LR variants (W10 windowed, signal-quality gate) provide marginal improvement over well-tuned constant LR or simple decay (D3)
3. LR only scales magnitude; it cannot change the direction of D_t·m_t

### What we know about RL gradient structure

1. **GRPO advantage structure**: within a prompt group, Σ A⁺ = Σ|A⁻| (first moment balanced), but Σ(A⁻)² > Σ(A⁺)² for p>0.5 (second moment favors minority class)
2. **Reward sparsity**: ~40-60% of prompt groups are uninformative (all-correct or all-wrong), contributing zero gradient. Learning concentrates on mixed prompts.
3. **Learning pattern**: GRPO mainly stabilizes near-boundary problems (p: 0.1-0.3 → 0.5-0.7), not breakthrough learning. Consistency improvement > capability improvement.
4. **Cross-term importance**: A²Q proxy (assuming orthogonal per-token gradients) likely misestimates per-response gradient energy; exact lm_head norms being measured

### Open questions (ordered by priority)

1. **P0**: Do g⁺ and g⁻ have similar magnitudes? What is their angular relationship? → pos/neg gradient decomposition (implemented)
2. **P1**: Does Adam's D_t rotate the combined update away from g⁺ direction? → cos(m,g⁺) vs cos(Dm,g⁺) diagnostic
3. **P2**: If D_t hurts, what coordinate-level structure causes it? → channel analysis of v_t vs g⁺/g⁻ alignment

---

## Historical: Signal Fraction Framework (2026-04-20, status: closed)

The formal chain (retained as theoretical foundation, no longer actively pursued for LR control):

1. **MSE**: E[||ĝ-g||²] = tr(Σ)/n — from i.i.d. vector sample mean; cross terms vanish by independence
2. **n_eff**: n_eff(t) = E[K_n] ≈ min(n, exp(H(π_t))) — distinct trajectories = perplexity
3. **Improvement bound** (L-smooth): E[J(θ_{t+1})] - J(θ_t) ≥ D_t · α[r_t - αL/2]
4. **Signal fraction**: r_t = ||g||² / (||g||² + σ²/n_eff) ∈ (0,1] — safe zone condition is α < 2r_t/L
5. **Optimal LR**: α*(t) = r_t/L — monotone decreasing as n_eff decreases
6. **Shape vs scale**: r_t determines shape of α*(t); L (≈ constant) determines absolute scale
7. **Mismatch**: fixed LR eventually exceeds 2α*(t) → expected-negative updates

**Key correction from Session 2**: earlier wrong derivation via SNR ≥ 1 gave α* ∝ √n_eff. Correct result via improvement bound is α* ∝ r_t ∝ n_eff (linear, in noise-dominated regime).

**Why closed**: r̂_t estimation too noisy; adaptive LR provides marginal gains over simple schedules; the real problem is gradient direction, not step size.
