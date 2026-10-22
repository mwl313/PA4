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

    Returns {joint_logprob, entropy, uniform_kl, values}, each [n], aligned with
    `rows`. The critic sees only the pre-action observation.

    The encoder runs ONCE for all rows (it is batch-capable), while each row
    keeps its own multi-level prefix table and saved index. Semantics are
    identical to scoring rows one at a time; only the encoder call is batched.
    """
    model.train()
    if not rows:
        z = torch.zeros(0)
        return {"joint_logprob": z, "entropy": z, "uniform_kl": z, "values": z}

    from agent.types.observation import ObservationBatch

    obs_list = [build_observation(row, device=device) for row in rows]
    batched = _stack_observations(obs_list, device)

    a2rs = [o.active_to_roster() for o in obs_list]

    encoded_all = model.encode(batched)
    values = model.value(encoded_all)

    joints, ents, kls = [], [], []
    for i, row in enumerate(rows):
        single = _slice_encoded(encoded_all, i)
        a2r = _table_roster(a2rs[i])
        table = _multi_level_table_plain(row, a2r).to(device)
        selected = torch.tensor(
            [[int(b["selected_index"]) for b in row.branches]],
            dtype=torch.long, device=device,
        )
        evaluation = model.evaluate_encoded(single, table, selected=selected)
        n = len(row.branches)
        joints.append(evaluation.logprob_selected[0, :n].sum())
        ents.append(evaluation.entropy_normalized[0, :n].mean())
        kls.append(evaluation.uniform_kl_normalized[0, :n].mean())

    return {
        "joint_logprob": torch.stack(joints),
        "entropy": torch.stack(ents),
        "uniform_kl": torch.stack(kls),
        "values": values,
    }


def _table_roster(a2r):
    """Roster map for ONE row.

    `active_to_roster()` returns a [2] pair per row. `build_branch_table` indexes
    it by row, so wrap this single row as a length-1 list rather than unwrapping
    it to a 2-vector (which PA3 reads as two rows and rejects).
    """
    if a2r is None:
        return None
    return [a2r[0]]


def _stack_observations(obs_list, device):
    """Concatenate several single-row ObservationBatch objects along batch."""
    from agent.types.observation import ObservationBatch
    from agent.types.ragged import FIELDS, RaggedBatch

    if len(obs_list) == 1:
        return obs_list[0]

    first = obs_list[0]
    ragged = {}
    for family in FIELDS:
        counts = torch.cat([o.ragged[family].counts for o in obs_list], dim=0)
        values_np = []
        for o in obs_list:
            values_np.append(o.ragged[family].values.cpu().numpy())
        import numpy as np

        vals = np.concatenate(values_np, axis=0)
        host = np.concatenate([o.ragged[family].host_counts for o in obs_list], axis=0)
        ragged[family] = RaggedBatch(
            torch.as_tensor(vals), counts, host,
        ).to(device)

    return ObservationBatch(
        token_mask=torch.cat([o.token_mask for o in obs_list], dim=0),
        categories=torch.cat([o.categories for o in obs_list], dim=0),
        category_known=torch.cat([o.category_known for o in obs_list], dim=0),
        floats=torch.cat([o.floats for o in obs_list], dim=0),
        float_known=torch.cat([o.float_known for o in obs_list], dim=0),
        flags=torch.cat([o.flags for o in obs_list], dim=0),
        flag_known=torch.cat([o.flag_known for o in obs_list], dim=0),
        role_ids=torch.cat([o.role_ids for o in obs_list], dim=0),
        side_ids=torch.cat([o.side_ids for o in obs_list], dim=0),
        ragged=ragged,
    )


def _slice_encoded(encoded, i):
    """One row of a batched EncodedState."""
    from agent.model.pa3_model import EncodedState

    return EncodedState(
        tokens=encoded.tokens[i:i + 1],
        global_repr=encoded.global_repr[i:i + 1],
        token_mask=encoded.token_mask[i:i + 1],
        layout=encoded.layout,
    )


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
