"""P4-B — independent golden reference for GAE and PPO arithmetic.

Written deliberately WITHOUT importing any P4 loss/GAE code. The design forbids
"call the same function twice and call it an oracle", so this module is a
separate implementation from the P4 training path.

Everything here is plain scalar/tensor arithmetic on explicitly constructed
inputs, so a shared mistake in the P4 implementation cannot hide here.
"""
from __future__ import annotations

import math

import torch

#: Declared golden comparison criteria (P4 design §4.4). Never widen to pass.
GOLDEN_ATOL = 1e-5
GOLDEN_RTOL = 1e-4


# ---------------------------------------------------------------------------
# GAE (per seat decision chain)
# ---------------------------------------------------------------------------
def gae(pred_values, rewards, dones, last_value=0.0, gamma=1.0, lam=0.95):
    """Reference GAE over ONE seat's decision chain.

    delta_t = r_t + gamma * (1 - d_t) * V_{t+1} - V_t
    A_t     = delta_t + gamma * lam * (1 - d_t) * A_{t+1}
    R_t     = A_t + V_t

    For P4's completed-only path the terminal row has done=True and the final
    bootstrap is zero, so V_{T} is never consulted.
    """
    n = len(pred_values)
    values = [float(v) for v in pred_values]
    rewards = [float(r) for r in rewards]
    dones = [bool(d) for d in dones]

    adv = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = float(last_value)
            next_nonterminal = 0.0 if dones[t] else 1.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_nonterminal * next_value - values[t]
        running = delta + gamma * lam * next_nonterminal * running
        adv[t] = running

    returns = [adv[t] + values[t] for t in range(n)]
    return adv, returns


def normalize_advantages(advantages, mask=None):
    """Population-std normalization (unbiased=False) with a 1e-8 floor.

    Normalization is over VALID ACTOR ROWS ONLY, and is applied exactly once per
    iteration. Returns stay unnormalized.
    """
    vals = torch.as_tensor(advantages, dtype=torch.float32)
    if mask is None:
        sel = vals
    else:
        m = torch.as_tensor(mask, dtype=torch.bool)
        sel = vals[m]
    if sel.numel() == 0:
        return vals
    std = sel.std(unbiased=False).clamp_min(1e-8)
    return (vals - sel.mean()) / std


# ---------------------------------------------------------------------------
# PPO terms
# ---------------------------------------------------------------------------
def ppo_policy_loss(new_logprob, old_logprob, advantages, clip=0.2,
                    mask=None):
    """-mean( min(r*A, clip(r)*A) ) over actor rows.

    Uses the exact stored action's logprob; never resamples.
    """
    new_lp = torch.as_tensor(new_logprob, dtype=torch.float32)
    old_lp = torch.as_tensor(old_logprob, dtype=torch.float32)
    adv = torch.as_tensor(advantages, dtype=torch.float32)

    ratio = torch.exp(new_lp - old_lp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    per_row = -torch.minimum(unclipped, clipped)
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool)
        if m.sum() == 0:
            return torch.zeros((), dtype=torch.float32)
        return per_row[m].mean()
    return per_row.mean()


def value_loss(pred_values, returns, mask=None):
    """0.5 * MSE over ALL valid value rows (no value clipping)."""
    v = torch.as_tensor(pred_values, dtype=torch.float32)
    r = torch.as_tensor(returns, dtype=torch.float32)
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool)
        if m.sum() == 0:
            return torch.zeros((), dtype=torch.float32)
        v, r = v[m], r[m]
    return 0.5 * ((v - r) ** 2).mean()


def approx_kl(new_logprob, old_logprob, mask=None):
    """mean( (ratio - 1) - log(ratio) ) over actor rows."""
    new_lp = torch.as_tensor(new_logprob, dtype=torch.float32)
    old_lp = torch.as_tensor(old_logprob, dtype=torch.float32)
    log_ratio = new_lp - old_lp
    ratio = torch.exp(log_ratio)
    per_row = (ratio - 1.0) - log_ratio
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool)
        if m.sum() == 0:
            return torch.zeros((), dtype=torch.float32)
        return per_row[m].mean()
    return per_row.mean()


def clip_fraction(new_logprob, old_logprob, clip=0.2, mask=None):
    new_lp = torch.as_tensor(new_logprob, dtype=torch.float32)
    old_lp = torch.as_tensor(old_logprob, dtype=torch.float32)
    ratio = torch.exp(new_lp - old_lp)
    hit = (ratio - 1.0).abs() > clip
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool)
        if m.sum() == 0:
            return 0.0
        hit = hit[m]
    return float(hit.float().mean().item())


def normalized_entropy(logprob, mask, k):
    """Per-request entropy normalized by log(K); 0 when K < 2."""
    lp = torch.as_tensor(logprob, dtype=torch.float32)
    m = torch.as_tensor(mask, dtype=torch.bool)
    probs = torch.exp(lp)
    entropy = -(probs * lp)[m].sum()
    kk = float(k)
    if kk < 2.0:
        return torch.zeros((), dtype=torch.float32)
    return entropy / math.log(kk)


def uniform_kl(logprob, mask, k):
    """KL(U || pi) = -log K - mean log pi, normalized by log K; 0 when K < 2."""
    lp = torch.as_tensor(logprob, dtype=torch.float32)
    m = torch.as_tensor(mask, dtype=torch.bool)
    kk = float(k)
    if kk < 2.0:
        return torch.zeros((), dtype=torch.float32)
    mean_lp = lp[m].mean()
    return (-math.log(kk) - mean_lp) / math.log(kk)


def total_loss(policy_loss, value_loss_, entropy, uniform_kl_,
               value_coef=0.5, entropy_coef=0.01, kl_coef=0.001):
    """L = L_pi + 0.5*L_V - 0.01*H_norm + 0.001*KL_norm (design §5.2)."""
    return (policy_loss + value_coef * value_loss_
            - entropy_coef * entropy + kl_coef * uniform_kl_)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def compare(golden, actual, name="value"):
    """Return (abs, rel, passed) under the declared atol/rtol."""
    g = torch.as_tensor(golden, dtype=torch.float64)
    a = torch.as_tensor(actual, dtype=torch.float64)
    if g.shape != a.shape:
        return float("inf"), float("inf"), False
    diff = (g - a).abs()
    tol = GOLDEN_ATOL + GOLDEN_RTOL * g.abs()
    ok = bool((diff <= tol).all())
    abs_max = float(diff.max().item()) if diff.numel() else 0.0
    denom = g.abs().clamp_min(1e-12)
    rel_max = float((diff / denom).max().item()) if diff.numel() else 0.0
    return abs_max, rel_max, ok
