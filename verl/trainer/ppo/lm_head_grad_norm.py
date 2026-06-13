"""Per-response exact lm_head gradient norm computation.

For REINFORCE / GRPO, the per-response policy gradient through the lm_head is:

    ∂S_b/∂W = Σ_t (e_{y_t} - π_t) h_t^T

where S_b = Σ_t log π(y_t | y_{<t}, x), W is lm_head weight (V×d),
h_t is the pre-lm_head hidden state, and π_t = softmax(W h_t / temperature).

The squared Frobenius norm (exact, no orthogonality assumption):

    ||∂S_b/∂W||²_F = Σ_{t,s} K_δ[t,s] · K_h[t,s]

where:
    K_h[t,s] = h_t · h_s        (hidden-space Gram matrix)
    K_δ[t,s] = δ_t · δ_s        (vocab-space Gram matrix of error vectors)
             = 1_{y_t=y_s} - π_s[y_t] - π_t[y_s] + π_t · π_s

This replaces the A²Q proxy which dropped the cross-terms (t≠s) in K_δ,
equivalent to assuming per-token score functions are orthogonal in param space.

The diagonal of K_δ recovers q_per_token = 1 - 2π_t[y_t] + Σ_v π_t[v]²,
so the A²Q proxy equals A² · trace(K_h ⊙ diag(K_δ)). The off-diagonal
captures inter-token score correlations from shared parameters.

Memory: peak (T_b, V) for softmax per response. For T=2000, V=100K in
float32 this is ~800MB. Use token_chunk_size to cap this if needed.
"""

import torch
import torch.nn.functional as F


def compute_response_lm_head_grad_norms(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    token_chunk_size: int = 0,
) -> torch.Tensor:
    """Compute per-response ||∂S_b/∂W_lm_head||²_F exactly.

    All inputs are in padded (batch, response_len, ...) format, response part only.

    Args:
        hidden_states: (batch, response_len, hidden_dim) pre-lm_head hidden states.
        logits: (batch, response_len, vocab_size) raw logits (before temperature).
        response_token_ids: (batch, response_len) token ids of the response.
        response_mask: (batch, response_len) binary mask, 1 for valid tokens.
        temperature: temperature used for log_prob computation.
        token_chunk_size: if > 0, process softmax in chunks to limit memory.
            0 means no chunking (full T×V softmax in memory).

    Returns:
        grad_norm_sq: (batch,) per-response ||∂S_b/∂W||²_F in float64.
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    grad_norm_sq = torch.zeros(batch_size, dtype=torch.float64, device=device)

    for b in range(batch_size):
        mask_b = response_mask[b].bool()
        T_b = mask_b.sum().item()
        if T_b == 0:
            continue

        h_b = hidden_states[b][mask_b].float()
        logits_b = logits[b][mask_b].float() / temperature
        y_b = response_token_ids[b][mask_b]

        K_h = h_b @ h_b.T

        if token_chunk_size <= 0 or T_b <= token_chunk_size:
            grad_norm_sq[b] = _grad_norm_sq_full(K_h, logits_b, y_b, device)
        else:
            grad_norm_sq[b] = _grad_norm_sq_chunked(
                K_h, logits_b, y_b, device, token_chunk_size
            )

    return grad_norm_sq


def _grad_norm_sq_full(
    K_h: torch.Tensor,
    logits_b: torch.Tensor,
    y_b: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Full (non-chunked) computation for one response."""
    T_b = logits_b.shape[0]

    pi_b = F.softmax(logits_b, dim=-1)

    K_pi = pi_b @ pi_b.T

    pi_cross_ts = pi_b[:, y_b]

    token_match = (y_b.unsqueeze(0) == y_b.unsqueeze(1)).float()

    K_delta = token_match - pi_cross_ts - pi_cross_ts.T + K_pi

    return (K_h * K_delta).sum().to(torch.float64)


def _grad_norm_sq_chunked(
    K_h: torch.Tensor,
    logits_b: torch.Tensor,
    y_b: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Memory-efficient chunked computation for one response.

    Avoids materializing the full (T, V) softmax matrix by processing
    token chunks. Peak softmax memory: (chunk_size, V) × 2.
    """
    T_b = logits_b.shape[0]
    result = torch.tensor(0.0, dtype=torch.float64, device=device)

    for ci in range(0, T_b, chunk_size):
        ci_end = min(ci + chunk_size, T_b)
        pi_ci = F.softmax(logits_b[ci:ci_end], dim=-1)
        y_ci = y_b[ci:ci_end]

        for cj in range(0, T_b, chunk_size):
            cj_end = min(cj + chunk_size, T_b)
            pi_cj = F.softmax(logits_b[cj:cj_end], dim=-1)
            y_cj = y_b[cj:cj_end]

            K_pi_block = pi_ci @ pi_cj.T

            pi_cross_block = pi_cj[:, y_ci].T

            pi_cross_T_block = pi_ci[:, y_cj]

            token_match_block = (y_ci.unsqueeze(1) == y_cj.unsqueeze(0)).float()

            K_delta_block = token_match_block - pi_cross_block - pi_cross_T_block + K_pi_block

            K_h_block = K_h[ci:ci_end, cj:cj_end]
            result += (K_h_block * K_delta_block).sum().to(torch.float64)

    return result


def compute_single_response_lm_head_grad(
    h_b: torch.Tensor,
    logits_b: torch.Tensor,
    y_b: torch.Tensor,
    token_chunk_size: int = 256,
) -> torch.Tensor:
    """Compute per-response lm_head score-function gradient vector.

    Returns G_b = Σ_t (e_{y_t} - π_t) h_tᵀ as a (V, d) float32 matrix,
    where π_t = softmax(logits_b[t]).  The caller is responsible for
    temperature-scaling logits *before* passing them here.

    Chunked over tokens to cap peak softmax memory at (chunk, V).

    Args:
        h_b: (T_valid, d) float32 hidden states for unmasked tokens.
        logits_b: (T_valid, V) float32 logits (already divided by temperature).
        y_b: (T_valid,) long token ids.
        token_chunk_size: token-dimension chunk size.

    Returns:
        (V, d) float32 gradient matrix.
    """
    T_valid, V = logits_b.shape
    d = h_b.shape[1]
    device = h_b.device

    grad = torch.zeros(V, d, dtype=torch.float32, device=device)
    chunk = token_chunk_size if token_chunk_size > 0 else T_valid

    for start in range(0, T_valid, chunk):
        end = min(start + chunk, T_valid)
        pi = F.softmax(logits_b[start:end], dim=-1)  # (C, V)
        ones = torch.ones(end - start, 1, dtype=pi.dtype, device=device)
        pi.scatter_add_(1, y_b[start:end].unsqueeze(1), -ones)
        # pi is now (π - e_y); G_chunk = (e_y - π)ᵀ h = -(pi)ᵀ h
        grad.sub_(pi.T @ h_b[start:end])

    return grad
