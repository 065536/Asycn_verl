genggeng# Calibrated PG-Moment Signal Fraction for Adaptive LR in RL

> **STATUS (2026-05-16)**: 此方案和后续的 parameter-space momentum 方案均已暂停。
> Logit-space 方案需要 λ_M calibration，metric mismatch 难解决。
> Parameter-space momentum 方案有 non-stationarity bias（late-stage r_t 高估）。
> 当前优先级转向 variance source decomposition 诊断。见 `algorithm_design.md`。

## 0. Document Purpose

This document summarizes a new adaptive learning-rate design for policy-gradient-based RL training, especially GRPO/PPO-style LLM RL. The goal is to replace the noisy batch-split estimator of the signal fraction \(r_t\) with a more stable, RL-aware estimator.

The key idea is:

\[
r_t \approx \frac{\text{stable policy-gradient signal power}}{\text{stable policy-gradient signal power} + \text{estimated policy-gradient sampling noise power}}.
\]

Instead of estimating \(\|g_t\|^2\) through noisy half-batch gradient dot products, we estimate the signal using a momentum-smoothed gradient tracker, and estimate the noise using an advantage-weighted policy-score second moment.

---

## 1. Background: Best LR from \(L\)-Smoothness

Consider maximizing an RL objective:

\[
J(\theta) = \mathbb{E}_{x \sim \mathcal D,\ y \sim \pi_\theta(\cdot|x)}[R(x,y)].
\]

Assume \(J\) is locally \(L\)-smooth around \(\theta_t\). For an update

\[
\theta_{t+1} = \theta_t + \alpha \hat g_t,
\]

we have the lower bound

\[
J(\theta_t + \alpha \hat g_t)
\ge
J(\theta_t) + \alpha g_t^\top \hat g_t
- \frac{L_t}{2}\alpha^2 \|\hat g_t\|^2,
\]

where

\[
g_t = \nabla J(\theta_t).
\]

If \(\hat g_t\) is an unbiased stochastic gradient,

\[
\mathbb{E}[\hat g_t] = g_t,
\]

and

\[
\mathbb{E}\|\hat g_t\|^2
=
\|g_t\|^2 + \frac{\sigma_t^2}{n_{\mathrm{eff},t}},
\]

then the expected lower-bound improvement is

\[
\mathbb{E}[\Delta J_{lb}(\alpha)]
=
\alpha \|g_t\|^2
-
\frac{L_t}{2}\alpha^2
\left(
\|g_t\|^2 + \frac{\sigma_t^2}{n_{\mathrm{eff},t}}
\right).
\]

Maximizing this quadratic gives

\[
\alpha_t^*
=
\frac{1}{L_t}
\cdot
\frac{\|g_t\|^2}{\|g_t\|^2 + \sigma_t^2/n_{\mathrm{eff},t}}.
\]

Define the signal fraction

\[
r_t
=
\frac{\|g_t\|^2}{\|g_t\|^2 + \sigma_t^2/n_{\mathrm{eff},t}}.
\]

Then

\[
\alpha_t^* = \frac{r_t}{L_t}.
\]

Thus the adaptive LR problem can be decomposed into two parts:

1. Estimate a slow curvature scale \(c_t \approx 1/L_t\).
2. Estimate a fast signal fraction \(r_t\).

This document focuses on the second part: estimating \(r_t\) robustly in RL.

---

## 2. Policy-Gradient Form of the Signal Fraction

For policy gradient, a single trajectory or response-level gradient contribution is

\[
z = \hat A(x,y) \nabla_\theta \log \pi_\theta(y|x),
\]

where \(\hat A\) is the estimated advantage.

The true policy gradient is

\[
g_t = \mathbb{E}[z].
\]

The second moment is

\[
M_{2,t}^{PG,\theta}
=
\mathbb{E}\|z\|^2
=
\mathbb{E}\left[
\hat A^2
\left\|
\nabla_\theta \log \pi_{\theta_t}(y|x)
\right\|^2
\right].
\]

If the batch contains \(n_{\mathrm{eff},t}\) effective independent units, then the noise of the batch-mean policy gradient is approximately

\[
N_t^\theta
\approx
\frac{M_{2,t}^{PG,\theta} - \|g_t\|^2}{n_{\mathrm{eff},t}}.
\]

When \(n_{\mathrm{eff},t}\) is large, this is often approximated as

\[
N_t^\theta
\approx
\frac{M_{2,t}^{PG,\theta}}{n_{\mathrm{eff},t}}.
\]

Therefore the ideal RL signal fraction is

\[
r_t
=
\frac{\|g_t\|^2}
{\|g_t\|^2 + N_t^\theta}.
\]

The implementation problem is that neither \(\|g_t\|^2\) nor \(M_{2,t}^{PG,\theta}\) is directly available at low cost.

---

## 3. Problem with the Previous Batch-Split Estimator

The previous estimator splits a batch into two halves \(A\) and \(B\):

\[
\hat g_A = g_t + \epsilon_A,
\qquad
\hat g_B = g_t + \epsilon_B.
\]

Then it estimates the signal fraction as

\[
\hat r_t^{split}
=
\frac{\hat g_A^\top \hat g_B}
{\frac{1}{2}(\|\hat g_A\|^2 + \|\hat g_B\|^2)}.
\]

The numerator is

\[
\hat g_A^\top \hat g_B
=
\|g_t\|^2
+ g_t^\top \epsilon_A
+ g_t^\top \epsilon_B
+ \epsilon_A^\top \epsilon_B.
\]

In late-stage RL training, the true signal \(\|g_t\|^2\) can become small. Then the estimator is dominated by the noise term

\[
\epsilon_A^\top \epsilon_B.
\]

This creates several issues:

1. The estimated numerator is often negative.
2. \(r_t\) has high step-to-step variance.
3. LR control becomes unstable.
4. The estimator ignores RL-specific structure such as advantage variance, prompt groups, response length, score-function norm, and clipping masks.

Therefore, we need a more stable and RL-aware estimator.

---

## 4. New Method: Calibrated PG-Moment Signal Fraction

### 4.1 High-Level Idea

We estimate

\[
r_t = \frac{S_t}{S_t + N_t^\theta},
\]

where

\[
S_t \approx \|g_t\|^2
\]

is a stable signal-power estimate, and

\[
N_t^\theta \approx \frac{\sigma_t^2}{n_{\mathrm{eff},t}}
\]

is a policy-gradient noise-power estimate in parameter space.

The proposed estimator is

\[
\boxed{
r_t^{new}
=
\frac{S_t^\theta}
{S_t^\theta + N_t^\theta + \epsilon_{den}}
}
\]

with

\[
S_t^\theta = \mathrm{EMA}(\|m_t\|^2),
\]

and

\[
N_t^\theta
=
\mathrm{EMA}\left(
\lambda_M
\frac{M_{2,t}^{PG,logit}}{n_{\mathrm{eff},t}}
\right).
\]

The final LR is

\[
\boxed{
\alpha_t = c_t r_t^{new}
}
\]

where \(c_t\) is a slow scale approximating \(1/L_t\). In the first implementation, \(c_t\) can be fixed.

---

## 5. Signal Estimate: Momentum-Smoothed Gradient Power

### 5.1 Motivation

Instead of estimating \(\|g_t\|^2\) through noisy split-gradient dot products, we use a smoothed gradient tracker.

The simplest version is to reuse AdamW's first moment:

\[
m_t^{Adam}
=
\beta_1 m_{t-1}^{Adam} + (1-\beta_1)\hat g_t.
\]

Then define

\[
S_t^{raw} = \|m_t^{Adam}\|^2.
\]

To further reduce noise:

\[
\boxed{
S_t^\theta
=
\beta_r S_{t-1}^\theta
+ (1-\beta_r)\|m_t^{Adam}\|^2
}
\]

This is a cheap MARS-style idea: use momentum as a lower-variance signal tracker.

### 5.2 Relation to MARS

MARS constructs a variance-reduced gradient estimator of the form

\[
c_t
=
g(\theta_t;\xi_t)
+
\rho
\left[g(\theta_t;\xi_t)-g(\theta_{t-1};\xi_t)\right],
\]

then updates a momentum variable:

\[
m_t = \beta m_{t-1} + (1-\beta)c_t.
\]

The exact MARS correction requires evaluating gradients at both \(\theta_t\) and \(\theta_{t-1}\) on the same batch \(\xi_t\). This is expensive in large-model RL.

Therefore, the first version of our method does not use exact MARS. It borrows the main idea: use a momentum-smoothed gradient signal instead of a single noisy gradient dot product.

### 5.3 Recommended First Version

Use

\[
\boxed{
S_t^\theta = \mathrm{EMA}(\|m_t^{Adam}\|^2)
}
\]

where \(m_t^{Adam}\) is the optimizer's existing first-moment state.

Advantages:

1. No extra model forward.
2. No extra backward.
3. No extra parameter-sized buffer.
4. Only requires computing a global norm over Adam's `exp_avg` state.

---

## 6. Noise Estimate: Advantage-Weighted Logit Score Second Moment

### 6.1 True Parameter-Space Second Moment

The ideal policy-gradient second moment is

\[
M_{2,t}^{PG,\theta}
=
\mathbb{E}\left[
\hat A^2
\left\|
\nabla_\theta \log \pi_{\theta_t}(y|x)
\right\|^2
\right].
\]

This is in the same parameter space as \(\|m_t\|^2\), so it would be dimensionally consistent.

However, computing per-sample parameter-gradient norms is expensive for large LLMs.

### 6.2 Logit-Space Proxy

For a token with logits \(\ell\), probability vector

\[
p = \mathrm{softmax}(\ell),
\]

and sampled token \(a\), the logit-space score is

\[
\nabla_\ell \log p(a) = e_a - p.
\]

Its squared norm is

\[
\boxed{
\|e_a-p\|^2 = 1 - 2p(a) + \sum_v p(v)^2
}
\]

For a token \((i,k,\tau)\), define

\[
q_{i,k,\tau}
=
1
-2p_\theta(y_{i,k,\tau})
+
\sum_v p_\theta(v|h_{i,k,\tau})^2.
\]

Let \(c_{i,k,\tau}\) be the actual coefficient of this token in the actor loss:

\[
\mathcal L_{actor}
= -\sum_{i,k,\tau} c_{i,k,\tau}
\log \pi_\theta(y_{i,k,\tau}|h_{i,k,\tau}).
\]

Then define the logit-space second-moment proxy:

\[
\boxed{
M_{2,t}^{PG,logit}
=
\sum_{i,k,\tau}
c_{i,k,\tau}^2
q_{i,k,\tau}
}
\]

This estimates the advantage-weighted policy-score energy in logit space.

### 6.3 Importance of Using the Correct Coefficient \(c_{i,k,\tau}\)

The coefficient must match the actual gradient contribution used in the actor loss.

For a simplified PPO/GRPO loss:

\[
\mathcal L
=
-\frac{1}{Z}
\sum_{i,k,\tau}
mask_{i,k,\tau}\rho_{i,k,\tau}\hat A_{i,k}
\log \pi_\theta(y_{i,k,\tau}),
\]

we have

\[
c_{i,k,\tau}
=
\frac{mask_{i,k,\tau}\rho_{i,k,\tau}\hat A_{i,k}}{Z}.
\]

If clipping makes the policy-gradient branch inactive for a token, then

\[
c_{i,k,\tau}=0.
\]

This is important because \(M_2\) should estimate the second moment of the actual gradient entering backward, not an unclipped theoretical gradient.

---

## 7. The Metric Mismatch Problem and \(\lambda_M\)

### 7.1 Why \(\lambda_M\) Is Needed

The numerator

\[
S_t^\theta = \mathrm{EMA}(\|m_t\|^2)
\]

is a parameter-space quantity.

The proxy

\[
M_{2,t}^{PG,logit}
\]

is a logit-space quantity.

They cannot be directly added.

The true relation is

\[
\nabla_\theta \log \pi_\theta(y)
=
J_\theta^\top
\nabla_\ell \log \pi_\theta(y),
\]

where \(J_\theta\) is the Jacobian from parameters to logits.

Therefore,

\[
\left\|
\nabla_\theta \log \pi_\theta(y)
\right\|^2
=
\left(\nabla_\ell \log \pi_\theta(y)\right)^\top
J_\theta J_\theta^\top
\left(\nabla_\ell \log \pi_\theta(y)\right).
\]

Our logit proxy replaces this metric by a scalar approximation:

\[
J_\theta J_\theta^\top \approx \lambda_M I.
\]

Thus

\[
M_{2,t}^{PG,\theta}
\approx
\lambda_M M_{2,t}^{PG,logit}.
\]

So \(\lambda_M\) is a calibration factor that maps logit-space score energy to parameter-space gradient-noise scale.

It is not optional. Without it, the denominator is not dimensionally consistent.

### 7.2 Calibrated Noise Power

The calibrated noise estimate is

\[
\boxed{
N_t^\theta
=
\mathrm{EMA}\left(
\lambda_M
\frac{M_{2,t}^{PG,logit}}{n_{\mathrm{eff},t}}
\right)
}
\]

or, if \(n_{\mathrm{eff},t}\) is treated as constant and absorbed into \(\lambda_M\):

\[
\boxed{
N_t^\theta
=
\mathrm{EMA}\left(
\lambda'_M M_{2,t}^{PG,logit}
\right).
}
\]

### 7.3 How to Calibrate \(\lambda_M\)

The recommended first implementation uses warmup calibration.

During the first \(T_0\) steps, log both:

1. A parameter-space noise reference \(V_t^{block,\theta}\).
2. The logit-space noise proxy \(V_t^{logit}\).

The logit-space proxy is

\[
V_t^{logit}
=
\frac{M_{2,t}^{PG,logit}}{n_{\mathrm{eff},t}}.
\]

A block-level parameter-space noise reference can be computed by splitting the batch into \(M\) prompt-aware blocks:

\[
\hat g_1,\dots,\hat g_M.
\]

Let

\[
\bar g = \frac{1}{M}\sum_{m=1}^M \hat g_m,
\]

and

\[
B = \frac{1}{M}\sum_{m=1}^M \|\hat g_m\|^2.
\]

Then a block-level noise reference is

\[
\boxed{
V_t^{block,\theta}
=
\frac{B-\|\bar g\|^2}{M-1}.
}
\]

Then calibrate

\[
\boxed{
\lambda_M
=
\mathrm{median}_{t\le T_0}
\frac{V_t^{block,\theta}}{V_t^{logit}+\epsilon}
}
\]

using only valid steps.

After warmup, \(\lambda_M\) can be fixed.

A later version may update \(\lambda_M\) slowly:

\[
\lambda_{M,t}
=
\beta_\lambda \lambda_{M,t-1}
+ (1-\beta_\lambda)
\frac{V_t^{block,\theta}}{V_t^{logit}+\epsilon}.
\]

However, the first version should use fixed warmup calibration for stability.

---

## 8. Effective Sample Size \(n_{\mathrm{eff}}\)

### 8.1 Why It Appears

\(M_2^{PG}\) estimates single-sample or single-unit gradient energy. But the actual update uses a batch mean gradient.

If there are \(n\) independent units,

\[
\mathrm{Var}(\hat g)
=
\frac{1}{n}\mathrm{Var}(z).
\]

Therefore, noise power scales as

\[
\frac{M_2}{n}.
\]

In RL, the effective number of independent units is often smaller than the nominal number of responses.

### 8.2 Recommended First Version

For GRPO, the independent unit should be the prompt group, not an individual response.

If a batch has \(N_{prompt}\) prompts and \(K\) completions per prompt, then the nominal number of responses is \(N_{prompt}K\), but the effective unit is closer to \(N_{prompt}\).

Recommended first version:

\[
\boxed{
n_{\mathrm{eff},t} = N_{prompt}.
}
\]

If \(N_{prompt}\) is fixed across training, this factor can be absorbed into \(\lambda_M\).

### 8.3 Optional Reward-Based Effective Sample Size

A later version can use reward reliability.

For prompt \(i\), let

\[
s_i = \mathrm{std}(R_{i,1:K}).
\]

Define

\[
w_i
=
\frac{s_i^2}{s_i^2+\tau_R^2}.
\]

Then

\[
\boxed{
n_{\mathrm{eff},t}
=
\frac{(\sum_i w_i)^2}{\sum_i w_i^2+\epsilon}
}
\]

If many prompt groups have nearly identical rewards, \(w_i\) becomes small and \(n_{\mathrm{eff},t}\) decreases. This increases the estimated noise power and reduces LR.

This is theoretically meaningful, but it should be treated as a second-stage enhancement, not part of the first implementation.

---

## 9. Final Recommended First-Version Algorithm

### 9.1 Formula

Use

\[
\boxed{
S_t^\theta
=
\mathrm{EMA}(\|m_t^{Adam}\|^2)
}
\]

and

\[
\boxed{
N_t^\theta
=
\mathrm{EMA}\left(
\lambda_M
\frac{M_{2,t}^{PG,logit}}{N_{prompt}}
\right)
}
\]

Then

\[
\boxed{
r_t
=
\frac{S_t^\theta}{S_t^\theta + N_t^\theta + \epsilon_{den}}
}
\]

and

\[
\boxed{
\alpha_t = c_t r_t.
}
\]

If \(c_t\) is fixed:

\[
\boxed{
\alpha_t = c_{fixed} r_t.
}
\]

### 9.2 Practical Clipping

Use

\[
r_t \leftarrow \mathrm{clip}(r_t, r_{min}, r_{max}).
\]

A reasonable first setting is

\[
r_{min}=0.01,
\qquad
r_{max}=1.0.
\]

The lower bound prevents LR from becoming exactly zero due to estimator noise.

---

## 10. Implementation Procedure

At each update step:

### Step 1: Run the normal GRPO/PPO actor forward

Compute logits, log probabilities, ratios, advantages, masks, and the normal actor loss.

### Step 2: Compute the logit-space score energy

For each token:

\[
q_{i,k,\tau}
=
1
-2p_\theta(y_{i,k,\tau})
+\sum_v p_\theta(v|h_{i,k,\tau})^2.
\]

The stable computation is

\[
\sum_v p_v^2
=
\exp\left(
\log\sum_v e^{2\ell_v}
-2\log\sum_v e^{\ell_v}
\right).
\]

### Step 3: Compute \(M_{2,t}^{PG,logit}\)

Use the actual actor-loss coefficient \(c_{i,k,\tau}\):

\[
M_{2,t}^{PG,logit}
=
\sum_{i,k,\tau}
c_{i,k,\tau}^2 q_{i,k,\tau}.
\]

### Step 4: Run normal backward and optimizer step

No extra backward is required.

### Step 5: Read Adam first-moment norm

After AdamW updates its first-moment state, compute

\[
\|m_t^{Adam}\|^2.
\]

Under FSDP, each rank computes its local shard norm, then all-reduces the scalar sum.

### Step 6: Update EMAs

\[
S_t^\theta
=
\beta_r S_{t-1}^\theta
+(1-\beta_r)\|m_t^{Adam}\|^2.
\]

\[
N_t^\theta
=
\beta_r N_{t-1}^\theta
+(1-\beta_r)
\lambda_M
\frac{M_{2,t}^{PG,logit}}{N_{prompt}}.
\]

### Step 7: Compute \(r_t\)

\[
r_t
=
\frac{S_t^\theta}{S_t^\theta + N_t^\theta + \epsilon_{den}}.
\]

Then clip to a safe range.

### Step 8: Set LR

\[
\alpha_t=c_t r_t.
\]

This LR can be applied to the next optimizer step or to the current step depending on implementation convenience. For logging clarity, using it for the next update is often simpler.

---

## 11. Pseudocode

```python
# During actor forward
logits = actor_outputs.logits          # [B, T, V]
target_ids = batch["response_ids"]     # [B, T]
token_mask = batch["response_mask"]    # [B, T]

# coeff must match the actual actor-loss gradient coefficient
# Example: coeff = mask * ratio * advantage / normalization
coeff = actor_loss_coeff.detach()      # [B, T]

with torch.no_grad():
    logits_f = logits.float()

    logZ = torch.logsumexp(logits_f, dim=-1)  # [B, T]
    target_logits = logits_f.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    log_p_y = target_logits - logZ
    p_y = torch.exp(log_p_y)

    log_sum_p2 = torch.logsumexp(2.0 * logits_f, dim=-1) - 2.0 * logZ
    sum_p2 = torch.exp(log_sum_p2)

    q = 1.0 - 2.0 * p_y + sum_p2
    q = torch.clamp(q, min=0.0)

    m2_local = ((coeff ** 2) * q * token_mask).sum()

m2_pg_logit = all_reduce_sum(m2_local)
```

After optimizer moment update:

```python
m_norm_sq_local = torch.zeros((), device=device)

for group in optimizer.param_groups:
    for p in group["params"]:
        state = optimizer.state[p]
        if "exp_avg" in state:
            m = state["exp_avg"]
            m_norm_sq_local += (m.float() ** 2).sum()

m_norm_sq = all_reduce_sum(m_norm_sq_local)
```

Update the control statistics:

```python
noise_raw = m2_pg_logit / (num_prompts_global + eps)
noise_calibrated = lambda_M * noise_raw

S_ema = beta_r * S_ema + (1.0 - beta_r) * m_norm_sq
N_ema = beta_r * N_ema + (1.0 - beta_r) * noise_calibrated

r_t = S_ema / (S_ema + N_ema + eps_den)
r_t = torch.clamp(r_t, r_min, r_max)

lr_t = c_t * r_t
```

---

## 12. Computational Overhead

### 12.1 Recommended First Version

Using Adam first moment + logit-space \(M_2\):

1. Extra model forward: none.
2. Extra backward: none.
3. Extra parameter-sized buffer: none if reusing Adam's `exp_avg`.
4. Extra scalar all-reduce: a few scalar reductions.
5. Extra logits computation: one additional vocab-dimension reduction for \(\sum_v p_v^2\).

The main overhead is computing

\[
\log\sum_v e^{2\ell_v}
\]

for all valid response tokens.

For 1.5B experiments, this overhead should be acceptable.

For 7B/32B or very long responses, reduce overhead by:

1. Computing \(M_2\) every \(q\) steps, e.g. \(q=2,4,8\).
2. Estimating \(M_2\) on a token subset.
3. Estimating \(M_2\) on a prompt subset.
4. Reusing EMA between measurement steps.

### 12.2 Exact MARS Is Not Recommended Initially

Exact MARS requires computing

\[
g(\theta_{t-1};\xi_t)
\]

on the current rollout batch. This implies an extra old-policy backward and likely almost doubles actor update compute.

Therefore exact MARS should not be part of the first implementation.

### 12.3 Approximate MARS as a Later Enhancement

A cheaper approximate MARS variant is

\[
c_t=(1+\rho)\hat g_t - \rho \hat g_{t-1}.
\]

Then

\[
m_t=\beta m_{t-1}+(1-\beta)c_t.
\]

This requires saving \(\hat g_{t-1}\), a gradient-sized buffer. It does not need extra forward/backward, but it increases memory pressure.

This can be tested later if Adam first moment is insufficient.

---

## 13. Numerical Stabilization

The theoretical formula is

\[
r_t = \frac{S_t}{S_t+N_t}.
\]

The implementation uses

\[
r_t = \frac{S_t}{S_t+N_t+\epsilon_{den}}.
\]

\(\epsilon_{den}\) is not a theoretical noise term. It is only a numerical stabilizer to prevent division by zero or near-zero denominators.

A more explicit implementation is:

\[
r_t=
\begin{cases}
\dfrac{S_t}{S_t+N_t}, & S_t+N_t > \epsilon_{den},\\
r_{t-1}, & S_t+N_t \le \epsilon_{den}.
\end{cases}
\]

For the first implementation, the additive \(\epsilon_{den}\) version is simpler.

---

## 14. What This Method Is and Is Not

### 14.1 What It Is

It is a calibrated, RL-aware, low-overhead LR control signal:

\[
r_t
=
\frac{\text{momentum-smoothed gradient power}}
{\text{momentum-smoothed gradient power} + \text{calibrated policy-gradient noise power}}.
\]

It uses:

1. Adam/MARS-style momentum for the signal.
2. Advantage-weighted logit score second moment for the noise.
3. A calibration factor \(\lambda_M\) to correct the metric mismatch.
4. EMA smoothing to reduce step-level noise.

### 14.2 What It Is Not

It is not an exact unbiased estimator of the ideal theoretical \(r_t\).

The logit-space second moment is only a proxy for parameter-space policy-gradient noise.

Therefore \(\lambda_M\) is necessary, and the method must be validated empirically.

---

## 15. Recommended Experimental Plan

### Stage 1: Logging Only

Do not control LR yet. Log:

1. Existing split estimator:

\[
r_t^{split}
=
\frac{\hat g_A^\top \hat g_B}
{0.5(\|\hat g_A\|^2+\|\hat g_B\|^2)}.
\]

2. New signal estimate:

\[
S_t^\theta=\mathrm{EMA}(\|m_t^{Adam}\|^2).
\]

3. Raw logit noise proxy:

\[
V_t^{logit}=M_{2,t}^{PG,logit}/N_{prompt}.
\]

4. Calibrated noise:

\[
N_t^\theta=\lambda_M V_t^{logit}.
\]

5. New ratio:

\[
r_t^{new}=\frac{S_t^\theta}{S_t^\theta+N_t^\theta+\epsilon}.
\]

Compare stability against:

- KL spike;
- grad norm spike;
- clip fraction;
- entropy drop;
- reward plateau;
- validation degradation;
- late-stage collapse.

### Stage 2: Fixed-Scale LR Control

Use

\[
\alpha_t = c_{fixed} r_t^{new}.
\]

Compare against:

1. Constant LR baseline.
2. Split-based adaptive LR.
3. Sign-gate adaptive LR.
4. Existing continuous-\(r_t\) adaptive LR.

### Stage 3: Add Optional Enhancements

Possible enhancements:

1. Reward-based \(n_{\mathrm{eff}}\).
2. Slow EMA update for \(\lambda_M\).
3. Approximate MARS numerator.
4. Block-jackknife calibration refresh.
5. Preconditioned-space signal norm.

These should not be included in the first implementation unless the basic version shows promise.

---

## 16. Key Ablations

### Ablation A: Numerator Source

Compare:

1. Split numerator \(\hat g_A^\top \hat g_B\).
2. Full gradient norm \(\|\hat g_t\|^2\).
3. Adam first moment \(\|m_t^{Adam}\|^2\).
4. Approximate MARS moment.

### Ablation B: Denominator Source

Compare:

1. Split denominator.
2. Block-jackknife noise.
3. Logit-space \(M_2\) proxy.
4. Calibrated logit-space \(M_2\) proxy.

### Ablation C: Calibration

Compare:

1. No calibration.
2. Warmup median calibration.
3. Slow EMA calibration.
4. Fixed hand-tuned \(\lambda_M\).

### Ablation D: Effective Sample Size

Compare:

1. \(n_{\mathrm{eff}}=N_{prompt}\).
2. \(n_{\mathrm{eff}}=N_{prompt}K\).
3. Reward-based \(n_{\mathrm{eff}}\).
4. Absorb \(n_{\mathrm{eff}}\) into \(\lambda_M\).

---

## 17. Main Risks

### Risk 1: Metric mismatch remains too large

Even with \(\lambda_M\), logit-space score energy may not track parameter-space noise well.

Mitigation:

- Use warmup calibration.
- Validate correlation with block-level parameter-space noise.
- Use logit proxy only as a relative gate if direct calibration fails.

### Risk 2: Adam momentum lags behind fast RL changes

\(m_t^{Adam}\) may be too slow to track abrupt policy changes.

Mitigation:

- Use a smaller \(\beta_r\).
- Use custom controller EMA with different \(\beta_m\).
- Add KL-jump reset or decay.

### Risk 3: \(M_2\) computation overhead is nontrivial

Computing \(\sum_v p_v^2\) adds a vocab reduction.

Mitigation:

- Compute every few steps.
- Token subsampling.
- Prompt subsampling.
- Use EMA to smooth sparse measurements.

### Risk 4: Clipping and masks are implemented inconsistently

If \(c_{i,k,\tau}\) does not match the actual actor loss coefficient, \(M_2\) will estimate the wrong noise.

Mitigation:

- Reuse the exact per-token loss coefficient from the actor loss implementation.
- Log sanity checks: coefficient distribution, active token count, clip mask count.

---

## 18. Suggested Naming

Possible method names:

1. PG-Moment Signal Fraction.
2. Calibrated PG-Moment LR Scaling.
3. Variance-Aware Policy-Gradient LR Scaling.
4. Momentum-Calibrated PG Noise Scaling.

A concise working name:

\[
\boxed{\text{Calibrated PG-Moment LR Scaling}}
\]

---

## 19. Summary

The new algorithm replaces noisy batch-split signal estimation with a calibrated signal-vs-noise estimator.

Old estimator:

\[
r_t^{split}
=
\frac{\hat g_A^\top \hat g_B}
{0.5(\|\hat g_A\|^2+\|\hat g_B\|^2)}.
\]

New estimator:

\[
\boxed{
r_t^{new}
=
\frac{S_t^\theta}{S_t^\theta+N_t^\theta+\epsilon_{den}}
}
\]

with

\[
\boxed{
S_t^\theta=\mathrm{EMA}(\|m_t^{Adam}\|^2)
}
\]

and

\[
\boxed{
N_t^\theta
=
\mathrm{EMA}\left(
\lambda_M
\frac{M_{2,t}^{PG,logit}}{N_{prompt}}
\right).
}
\]

The policy-gradient logit second moment is

\[
\boxed{
M_{2,t}^{PG,logit}
=
\sum_{i,k,\tau}
c_{i,k,\tau}^2
\left[
1-2p_\theta(y_{i,k,\tau})
+
\sum_v p_\theta(v|h_{i,k,\tau})^2
\right].
}
\]

The final LR is

\[
\boxed{
\alpha_t=c_t r_t^{new}.
}
\]

The method is not an exact estimator of the ideal theoretical signal fraction. It is a calibrated, RL-aware, low-overhead proxy designed to be more stable than batch-split gradient dot products.

Its first implementation should:

1. Reuse Adam's first moment for the numerator.
2. Use logit-space score second moment for the denominator.
3. Calibrate \(\lambda_M\) during warmup.
4. Treat \(n_{\mathrm{eff}}=N_{prompt}\) initially.
5. Avoid exact MARS or per-sample parameter-gradient norms in the first version.
