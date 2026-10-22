"""P4-C — trainable evaluation of STORED actions with gradients.

Design §4.1 B: score the same saved indices, masks, token refs and observation
with the current trainable model, preserving gradients. Nothing is resampled and
no native candidate is regenerated.

This is deliberately separate from `parity.py`, which uses `torch.no_grad()` and
the frozen snapshot: the learner path must build a graph through the conditional
scorer, including the reused GRU prefix.
"""
from __future__ import annotations

import numpy as np
import torch

from pa4.learner.parity import build_observation, _multi_level_table


def evaluate_stored_batch(model, rows, device="cpu"):
    """Differentiable joint logprob, entropy and reverse-KL for stored rows.

    Returns a dict of tensors aligned with `rows`:
        joint_logprob [n]
        entropy       [n]   (normalized per request; 0 when K<2)
        uniform_kl    [n]   (KL(U||pi) normalized; 0 when K<2)
        values        [n]   (observation-only critic, differentiable)

    The critic sees only the pre-action observation.
    """
    model.train()
    joints, ents, kls, values = [], [], [], []

    for row in rows:
        obs = build_observation(row, device=device)
        a2r = obs.active_to_roster()
        table = _multi_level_table_plain(row, a2r).to(device)
        selected = torch.tensor(
            [[int(b["selected_index"]) for b in row.branches]],
            dtype=torch.long, device=device,
        )

        encoded = model.encode(obs)
        evaluation = model.evaluate_encoded(encoded, table, selected=selected)

        joint = evaluation.logprob_selected[0, :len(row.branches)].sum()
        joints.append(joint)
        values.append(model.value(encoded).reshape(()))

        # Regularizers: PA3 exposes normalized entropy and reverse uniform KL
        # per branch; a request-level value is the mean over real branches.
        ent = evaluation.entropy_normalized[0, :len(row.branches)]
        kl = evaluation.uniform_kl_normalized[0, :len(row.branches)]
        ents.append(ent.mean())
        kls.append(kl.mean())

    return {
        "joint_logprob": torch.stack(joints),
        "entropy": torch.stack(ents),
        "uniform_kl": torch.stack(kls),
        "values": torch.stack(values),
    }


def _multi_level_table_plain(row, a2r):
    """Build the multi-level candidate table without touching a frozen scorer."""
    from pa4.actions.conditional_policy import build_branch_table
    from agent.types.observation import ObservationLayout

    layout = ObservationLayout()
    parts = []
    for branch in row.branches:
        parts.append(build_branch_table([list(branch["candidates"])], layout, a2r))
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    return type(first)(
        action_ids=torch.cat([t.action_ids for t in parts], dim=1),
        mask=torch.cat([t.mask for t in parts], dim=1),
        entity_token=torch.cat([t.entity_token for t in parts], dim=1),
        move_token=torch.cat([t.move_token for t in parts], dim=1),
        branch_valid=torch.cat([t.branch_valid for t in parts], dim=1),
        selected=torch.cat([t.selected for t in parts], dim=1),
    )


def values_only(model, rows, device="cpu"):
    """Observation-only critic values with gradients (all value rows)."""
    model.train()
    out = []
    for row in rows:
        obs = build_observation(row, device=device)
        encoded = model.encode(obs)
        out.append(model.value(encoded).reshape(()))
    return torch.stack(out)


def score_branch_table(model, rows, device="cpu"):
    """Return per-row (joint, entropy, kl, value) as plain floats.

    Used by the parity harness to compare a trainable path against the golden
    reference on identical inputs.
    """
    got = evaluate_stored_batch(model, rows, device=device)
    return {
        "joint_logprob": [float(x.detach().cpu()) for x in got["joint_logprob"]],
        "entropy": [float(x.detach().cpu()) for x in got["entropy"]],
        "uniform_kl": [float(x.detach().cpu()) for x in got["uniform_kl"]],
        "values": [float(x.detach().cpu()) for x in got["values"]],
    }
