"""P4-C — conditional PPO-2 loss with exact row weighting.

Design §5.2/§5.3. The policy objective uses the JOINT logprob (sum of all valid
conditional branch logprobs) of the EXACT stored action; nothing is resampled.

Actor and value terms have separate denominators, and microbatch accumulation is
weighted so the accumulated gradient equals the full-minibatch gradient.
"""
from __future__ import annotations

import torch

DEFAULT_CLIP = 0.2
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_KL_COEF = 0.001
TARGET_APPROX_KL = 0.03


class LossTerms:
    """One microbatched loss evaluation, retaining each term separately."""

    __slots__ = ("total", "policy", "value", "entropy", "kl",
                 "actor_rows", "value_rows", "approx_kl", "clip_fraction")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def ppo_terms(new_joint_logprob, old_joint_logprob, advantages, returns,
              values, *, actor_mask, value_mask,
              entropy=None, kl=None, clip=DEFAULT_CLIP,
              value_coef=DEFAULT_VALUE_COEF,
              entropy_coef=DEFAULT_ENTROPY_COEF,
              kl_coef=DEFAULT_KL_COEF,
              actor_weight=1.0, value_weight=1.0):
    """Compute the PPO-2 objective with separate actor/value denominators.

    `actor_weight` / `value_weight` scale each term by its share of the FULL
    minibatch (exact row-weighted accumulation). A naive equal micro average
    would give each micro the same weight regardless of its row counts.
    """
    new_lp = torch.as_tensor(new_joint_logprob, dtype=torch.float32)
    old_lp = torch.as_tensor(old_joint_logprob, dtype=torch.float32)
    adv = torch.as_tensor(advantages, dtype=torch.float32)
    ret = torch.as_tensor(returns, dtype=torch.float32)
    val = torch.as_tensor(values, dtype=torch.float32)
    am = torch.as_tensor(actor_mask, dtype=torch.bool)
    vm = torch.as_tensor(value_mask, dtype=torch.bool)

    n_actor = int(am.sum().item())
    n_value = int(vm.sum().item())

    # ---- policy (actor rows only) ------------------------------------
    if n_actor:
        log_ratio = new_lp[am] - old_lp[am]
        ratio = torch.exp(log_ratio)
        a = adv[am]
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
        policy = -torch.minimum(unclipped, clipped).mean()
        approx_kl_value = float(((ratio - 1.0) - log_ratio).mean().item())
        clip_frac = float(((ratio - 1.0).abs() > clip).float().mean().item())
    else:
        policy = torch.zeros((), dtype=torch.float32, requires_grad=new_lp.requires_grad)
        approx_kl_value = 0.0
        clip_frac = 0.0

    # ---- value (all valid value rows, including K==1) ----------------
    if n_value:
        value_loss = 0.5 * ((val[vm] - ret[vm]) ** 2).mean()
    else:
        value_loss = torch.zeros((), dtype=torch.float32, requires_grad=val.requires_grad)

    # ---- regularizers (actor rows only; zero when K<2) ---------------
    if entropy is not None and n_actor:
        ent = torch.as_tensor(entropy, dtype=torch.float32)[am].mean()
    else:
        ent = torch.zeros((), dtype=torch.float32)
    if kl is not None and n_actor:
        klv = torch.as_tensor(kl, dtype=torch.float32)[am].mean()
    else:
        klv = torch.zeros((), dtype=torch.float32)

    total = (actor_weight * policy
             + value_coef * value_weight * value_loss
             - entropy_coef * actor_weight * ent
             + kl_coef * actor_weight * klv)

    return LossTerms(
        total=total, policy=policy, value=value_loss, entropy=ent, kl=klv,
        actor_rows=n_actor, value_rows=n_value,
        approx_kl=approx_kl_value, clip_fraction=clip_frac,
    )


def minibatch_slices(n_rows, minibatch_rows):
    """Yield index tensors covering every row exactly once.

    A trailing partial minibatch is included, never dropped.
    """
    for start in range(0, n_rows, minibatch_rows):
        yield torch.arange(start, min(start + minibatch_rows, n_rows))


def microbatch_weights(actor_counts, value_counts):
    """Per-micro scaling so accumulation equals the FULL minibatch objective.

    Each term is weighted by its share of the full minibatch's valid rows, so
    the summed gradient is the true full-batch gradient.
    """
    total_actor = max(1, sum(actor_counts))
    total_value = max(1, sum(value_counts))
    return (
        [c / total_actor for c in actor_counts],
        [c / total_value for c in value_counts],
    )


def learning_rate(committed_matches, *, start=1e-5, peak=3e-4,
                  warmup=250_000, horizon=100_000_000, final=3e-5):
    """Match-clock LR: linear warmup then cosine to `final`.

    The clock advances ONLY on committed natural training matches, never on rows,
    epochs or wall time.
    """
    import math

    m = max(0, int(committed_matches))
    if warmup > 0 and m < warmup:
        return start + (peak - start) * (m / warmup)
    span = max(1, horizon - warmup)
    progress = min(1.0, (m - warmup) / span)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))
