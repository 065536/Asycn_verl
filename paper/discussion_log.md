# Theory Discussion Log

**Date**: 2026-04-19
**Goal**: Build the paper's logical framework from first principles, without gaps.

---

## Session 1

### Starting point (original framing — later abandoned)

We started trying to explain: why does RL training of LLMs exhibit sharp LR sensitivity?

**Empirical anchor**:

| LR | AIME@300 | Pattern |
|----|----------|---------|
| 1e-6 | 0.281 | Stable, slow |
| 3e-6 | 0.325 | Optimal |
| 5e-6 | 0.294 | Slightly past peak |
| 1e-5 | 0.183 | Rise-then-collapse |

---

### Step 1: The objective

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

Fixed objective. Reward function doesn't change. Optimal policy $\pi^*$ doesn't change.
Both old and new frameworks agree on this.

---

### Step 2: The update rule

$$\theta_{t+1} = \theta_t + \alpha \cdot \hat{g}_t, \qquad \hat{g}_t = \frac{1}{n}\sum_{i=1}^n A_i \nabla_\theta \log \pi_\theta(\tau_i), \quad \tau_i \sim \pi_{\theta_t}$$

$\hat{g}_t$ is a finite-sample estimate ($n=8$ in GRPO), not the true gradient.

---

### Step 3: What makes RL structurally different from SL?

The key structural difference: in RL, the sampling distribution is determined by the current policy. Every update changes the policy, which changes the data distribution for the next update.

In SL: data is fixed → gradient estimate is an unbiased estimate of the true gradient for any θ.
In RL: gradient estimate is unbiased only at the current θ. Once θ changes, the old samples are stale.

This is not just "non-stationarity" — it is self-referential: the optimizer modifies the very source of its gradient information.

---

### Step 4: Attempted logical chain (following main.tex — identified as flawed)

Attempted chain:
```
policy sharpens → samples less diverse → gradient reliable range shrinks → LR should shrink
```

**Why this chain is broken:**

- "Sample diversity" (affects gradient estimate variance) ≠ "surrogate validity range" (affected by policy concentration itself, not the samples drawn from it). These are different phenomena conflated under "support."
- Even if we accepted that the reliable range shrinks, "LR ∝ H(t)/H(0)" is asserted, not derived.
- The chain only motivates decreasing LR, never increasing it.

**Conclusion**: main.tex's logical chain has gaps at every step. We should not rebuild from it.

---

### Step 5: The research question shifts

**Critical realization** (raised by user): Why are we trying to fix "LR too large"? LR=3e-6 already works — just use it. The deeper problem is:

> **The best result across ALL experiments is only ~32.5% on AIME. What is fundamentally limiting performance? How do we push past that?**

LR selection is one possible tool, not the fundamental goal.

Key observation: all stable experiments end at similar entropy (~0.07–0.10), but AIME@300 varies from 0.017 to 0.325. **Same terminal state, radically different performance.** The path through policy space during training determines the outcome, not the terminal entropy.

---

### Step 6: Two threads identified (from main_modified.tex and user)

The user identified two distinct threads that currently appear disconnected:

**Thread 1 (Sampling/Estimation problem)**
From policy gradient: J(θ) = E_{τ~π_θ}[R(τ)] can only be approximated via samples.
Update reliability depends on the quality of those samples.
RL is fundamentally "optimization under evidence constraints."
→ As training progresses, the effective information content of each batch may change.
→ Fixed update rules ignore this change.

**Thread 2 (Exploration-Exploitation problem)**
As the policy sharpens, it stops exploring alternative directions.
Some directions' value is only discoverable through repeated visitation.
Premature convergence cuts off these directions — "exploration contraction."
→ The policy may converge to a suboptimal mode because it stopped exploring before finding better strategies.

---

## Open questions (Session 1 — all resolved in Session 2)

1. Are these two threads the same problem seen from different angles, or genuinely independent problems?
2. Which thread (or which combination) explains why performance is capped at ~32.5%?
3. How do we connect "sampling has a problem" to "exploration-exploitation" into a unified causal chain?
4. What does a rigorous, gap-free logical chain look like starting from these two threads?

---

## Key methodological note

**Do not force-fit the analysis toward the existing solution (entropy-adaptive LR).**
The research question is: what is fundamentally limiting RL training performance for LLMs, and how can we improve it? LR scheduling is at most one tool. The framework should be derived from the problem, not retrofitted to justify a pre-existing method.

---

## Session 2

**Date**: 2026-04-19
**Goal**: Formalize the theoretical framework; identify problems with the current algorithm; design a better one.

---

### Step 1: The two threads converge — unified via n_eff

Session 1 left open: are "sampling/estimation quality" and "exploration-exploitation" the same problem?

**Resolution**: In our setting (GRPO + binary correctness reward), reward is deterministic and verifiable, so reward reliability is constant. Both threads reduce to the same scalar: **how many distinct trajectories does the current batch contain?** This is $n_{\mathrm{eff}}(t)$.

- Thread 1 (estimation): gradient estimate quality ∝ n_eff
- Thread 2 (exploration): exploration level ∝ entropy ∝ n_eff (in policy gradient, no separate exploration mechanism)

They are the same quantity viewed from two angles. The unified framing: **on-policy RL is optimization under a shrinking evidence constraint.**

---

### Step 2: The formal derivation chain (→ paper/theory_derivation.tex)

Every step derived explicitly, no gaps:

**1. MSE of gradient estimator (vector case)**

Let $X_i = A_i \nabla \log \pi_\theta(\tau_i)$, i.i.d. from $\pi_{\theta_t}$.

$$\mathbb{E}[\|\hat{g} - g\|^2] = \frac{\mathrm{tr}(\Sigma_t)}{n}$$

Derivation: expand $\|\hat{g}-g\|^2$ as double sum over $i,j$; cross terms $(i \neq j)$ vanish by independence + zero mean; diagonal terms give $(1/n)\,\mathbb{E}[\|X - g\|^2] = (1/n)\,\mathrm{tr}(\Sigma)$.

**2. Effective sample size**

$$n_{\mathrm{eff}}(t) = \mathbb{E}[K_n] \approx \min(n,\, \exp(H(\pi_t)))$$

Derivation: $\mathbb{E}[K_n] = \sum_\tau [1-(1-\pi(\tau))^n]$ by indicator linearity. For uniform distribution over $M$ trajectories: two-regime analysis gives $\mathbb{E}[K_n] \approx n$ when $n \ll M = \exp(H)$, and $\approx M$ when $n \gg M$.

**3. Per-step improvement bound** ($L$-smooth objective)

$$\mathbb{E}[J(\theta_{t+1})] - J(\theta_t) \geq \alpha \|g\|^2 - \frac{\alpha^2 L}{2}\!\left(\|g\|^2 + \frac{\sigma_t^2}{n_{\mathrm{eff}}}\right)$$

Derivation: $L$-smooth descent lemma → substitute $\theta' = \theta + \alpha\hat{g}$ → take expectation → use $\mathbb{E}[\hat{g}^\top g] = \|g\|^2$ (unbiasedness) and $\mathbb{E}[\|\hat{g}\|^2] = \|g\|^2 + \sigma^2/n_{\mathrm{eff}}$ (bias-variance decomposition).

**4. Optimal LR**

Differentiate the bound over $\alpha$, set = 0:

$$\alpha^*(t) = \frac{\|g_t\|^2}{L\!\left(\|g_t\|^2 + \sigma_t^2/n_{\mathrm{eff}}(t)\right)}$$

Monotonically increasing in $n_{\mathrm{eff}}$: $d\alpha^*/dn_{\mathrm{eff}} > 0$ (verified by explicit differentiation).

**5. Linear scaling** (noise-dominated regime + Assumption 1: $\sigma^2/\|g\|^2 \approx C$)

$$\boxed{\alpha^*(t) \propto n_{\mathrm{eff}}(t)}$$

Note: an earlier wrong derivation via SNR condition $\|g\| \geq \|\hat{g}-g\|$ gave $\alpha^* \propto \sqrt{n_{\mathrm{eff}}}$. This was incorrect: the SNR condition does not constrain $\alpha$ (it cancels). The correct result from the improvement bound is **linear**.

**6. Mismatch and path quality**

$\alpha > \alpha^*(t) \Rightarrow \mathbb{E}[J(\theta_{t+1})] < J(\theta_t)$ — expected regression. Terminal entropy is similar across all stable experiments (~0.07–0.10); AIME@300 spans 20×. Path before $n_{\mathrm{eff}} \approx 1$ determines which mode is reached.

Full document: `paper/theory_derivation.tex`

---

### Step 3: Critique of the current algorithm (entadapt_initial)

Current formula: $\alpha(t) = \alpha_0 \times H(t)/H(0)$

**Flaw 1 — α₀ dependency ("照着答案来设计")**

The formula only works because α₀ = 1e-5 was set knowing that 3e-6 is empirically optimal. The H(t)/H(0) ratio happens to land α in the right range at the right time. If α₀ = 1e-6 (too small), the algorithm makes it smaller; it has no mechanism to detect or correct a wrong starting LR.

**Flaw 2 — Wrong proxy**

Theory says: ratio should be $n_{\mathrm{eff}}(t)/n_{\mathrm{eff}}(0) = \exp(H(t))/\exp(H(0))$. Current method uses $H(t)/H(0)$ (linear in $H$, not exponential). These differ significantly:
- At $H(t)=0.07$, $H(0)=0.68$: correct ratio $\approx 0.54$, current ratio $\approx 0.10$. Current method is ~5× more aggressive than theory justifies.
- The current method "works" empirically because exp(H) underestimates actual n_eff early on (measured $n_{\mathrm{eff}}(0) \approx 6$–8, not 2), partially compensating. But this is accidental.

**Flaw 3 — Not LR-scale independent**

A fixed α_max does not generalize across different models, tasks, or batch sizes. $\alpha^*(t)$ depends on $L$, $\|g\|^2$, $\sigma^2$ — all of which change with the experimental setting.

---

### Step 4: What a better algorithm looks like

**Core requirement**: estimate $\alpha^*(t)$ from the training data itself, without requiring the user to pre-set an LR that is already in the right range.

**Option A — Signal fraction (directly from improvement bound)**

Define:
$$r_t = \frac{\|\hat{g}_t\|^2}{\|\hat{g}_t\|^2 + \hat{\sigma}_t^2 / n}, \qquad \alpha(t) = \alpha_{\max} \times r_t$$

$r_t \in [0,1]$: signal fraction of the gradient estimate. Large when batch is diverse (high signal), small when batch is repetitive (noise dominates). Directly derived from the improvement bound. Still requires $\alpha_{\max}$, but the ratio $r_t$ auto-corrects if $\alpha_{\max}$ is too large.

Cost: requires per-sample gradient variance $\hat{\sigma}_t^2$ (memory/compute overhead in FSDP).

**Option B — Direct n_eff measurement**

$$\alpha(t) = \alpha_{\max} \times \frac{\bar{K}_t}{n}$$

$\bar{K}_t$ = average unique responses per prompt in the batch (directly = $\hat{n}_{\mathrm{eff}}(t)$). No gradient computation needed. Discrete and noisy; apply exponential moving average. Still has $\alpha_{\max}$ problem.

**Option C — Estimate L from consecutive gradient differences (no α_max)**

$$\hat{L}_t = \frac{\|\hat{g}_t - \hat{g}_{t-1}\|}{\alpha_{t-1} \cdot \|\hat{g}_{t-1}\|}, \qquad \alpha(t) = \frac{\|\hat{g}_t\|^2}{\hat{L}_t\!\left(\|\hat{g}_t\|^2 + \hat{\sigma}_t^2/\hat{n}_{\mathrm{eff}}(t)\right)}$$

Fully data-driven: no LR hyperparameter. $\hat{L}_t$ is a running estimate of the local smoothness constant. Noisy in early steps; needs smoothing. Most theoretically complete; requires storing the previous gradient.

**Option D — Gradient-norm normalized (dimensionless constant)**

$$\alpha(t) = \frac{C}{\|\hat{g}_t\|} \times \frac{\hat{n}_{\mathrm{eff}}(t)}{n}$$

$C$ is a dimensionless $O(1)$ constant (vs current $O(10^{-5})$ LR). Since numerator and denominator both scale with gradient magnitude, $C$ is much more transferable across models and tasks. Analogous to Adam's normalization by gradient moments, but here motivated by $n_{\mathrm{eff}}$.

---

### Open questions (Session 2)

1. **Which option to implement first?** Option C (estimate $\hat{L}_t$) is theoretically cleanest; Option D (gradient-norm normalized) is simplest and most transferable.

2. **Is $\hat{L}_t$ estimation feasible and stable in practice?** Consecutive gradient differences are noisy at early steps. Need to decide on smoothing window.

3. **Does "LR too small" fall within the mismatch framework?** From theory: $\alpha < \alpha^*(t)$ gives positive but suboptimal improvement — not a mismatch, just an efficiency problem. The new algorithm should primarily focus on preventing $\alpha > \alpha^*(t)$; fixing "too small" is a separate problem.

4. **Fair comparison**: the new algorithm should be benchmarked against sync_lr3e-6 with the same number of experimental runs (not after a LR sweep). This is the key test: does the algorithm find good LR without pre-tuning?
