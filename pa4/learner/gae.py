"""P4-C — per-seat GAE over naturally completed decision chains.

Design §5.1. One chain per `(episode_key, seat)` through that seat's actual
accepted decisions. Wait/Finished are never timesteps. The terminal payout is
already attached to the last real decision by `ingest.build_cohort`.
"""
from __future__ import annotations

import torch


def seat_gae(values, rewards, dones, gamma=1.0, gae_lambda=0.95):
    """GAE for one completed seat chain. Final bootstrap is zero.

    delta_t = r_t + gamma*(1-d_t)*V_{t+1} - V_t
    A_t     = delta_t + gamma*lambda*(1-d_t)*A_{t+1}
    R_t     = A_t + V_t
    """
    n = len(values)
    if n == 0:
        raise ValueError("empty seat chain")
    if not (len(rewards) == n and len(dones) == n):
        raise ValueError("values/rewards/dones length mismatch")
    if not bool(dones[-1]):
        raise ValueError("completed seat chain must end with done=True")

    vals = torch.as_tensor(values, dtype=torch.float32)
    rew = torch.as_tensor(rewards, dtype=torch.float32)
    adv = torch.zeros(n, dtype=torch.float32)
    last = torch.zeros((), dtype=torch.float32)  # terminal bootstrap 0

    for t in range(n - 1, -1, -1):
        nxt = last if t == n - 1 else vals[t + 1]
        nonterminal = 0.0 if bool(dones[t]) else 1.0
        delta = rew[t] + gamma * nonterminal * nxt - vals[t]
        last = delta + gamma * gae_lambda * nonterminal * last
        adv[t] = last

    returns = adv + vals
    return adv, returns


def normalize_advantages(adv, actor_mask):
    """Normalize ONCE over valid actor rows; population std with 1e-8 floor.

    Returns remain unnormalized. If there are no actor rows the advantages are
    returned unchanged (they carry no actor loss weight anyway).
    """
    a = torch.as_tensor(adv, dtype=torch.float32)
    m = torch.as_tensor(actor_mask, dtype=torch.bool)
    sel = a[m]
    if sel.numel() == 0:
        return a.clone()
    std = sel.std(unbiased=False).clamp_min(1e-8)
    return (a - sel.mean()) / std


def gae_for_cohort(cohort, values_by_identity, gamma=1.0, gae_lambda=0.95):
    """Compute per-seat GAE for every trajectory in a cohort.

    `values_by_identity` maps SeatRow.identity() -> V_old from the FROZEN critic.

    Returns (rows, advantages, returns, actor_mask) aligned by row order, where
    rows is the flattened value-row list. Chains never cross seats or episodes.
    """
    rows, advs, rets = [], [], []

    for traj in cohort.trajectories:
        for seat, seat_rows in traj.seats.items():
            if not seat_rows:
                continue
            vals = []
            for r in seat_rows:
                key = r.identity()
                if key not in values_by_identity:
                    raise KeyError(f"missing V_old for {key}")
                vals.append(float(values_by_identity[key]))

            rewards = [float(r.reward) for r in seat_rows]
            dones = [bool(r.done) for r in seat_rows]
            adv, ret = seat_gae(vals, rewards, dones,
                                gamma=gamma, gae_lambda=gae_lambda)
            rows.extend(seat_rows)
            advs.append(adv)
            rets.append(ret)

    advantages = torch.cat(advs) if advs else torch.zeros(0)
    returns = torch.cat(rets) if rets else torch.zeros(0)
    actor_mask = torch.tensor([bool(r.actor_active) for r in rows], dtype=torch.bool)
    return rows, advantages, returns, actor_mask
