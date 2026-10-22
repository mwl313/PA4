"""P4-B — frozen-policy replay and independent numerical oracle.

Two deliberately separate paths (design §4.1):

A. **Behavior verification** — rebuild the exact stored `ragged.v2` observation
   and ordered candidate tables, then score the SAVED index under the frozen
   behavior weights. Never resamples. Requires <= 1e-5 absolute branch and joint
   delta against the stored FP32 logprobs.

B. **Learner evaluation** — score the same saved indices with the current
   trainable model with gradients enabled, plus the frozen critic for V_old.

The golden reference in `golden.py` is written independently of both.
"""
from __future__ import annotations

import numpy as np
import torch

from pa4.env import ragged_transport as _rt

#: Declared P4-B numeric criteria (design §4.4). Never widen these to pass.
MAX_BRANCH_LOGPROB_DELTA = 1e-5
MAX_JOINT_LOGPROB_DELTA = 1e-5
PROB_NORMALIZATION_TOL = 1e-6


class ParityError(RuntimeError):
    """A replay could not reproduce stored behavior within the declared bounds."""


def _ensure_pa3():
    _rt._ensure_pa3_on_path()


def build_observation(row, device="cpu"):
    """Rebuild a PA3 `ObservationBatch` from a stored seat row (ragged.v2).

    Raises if the row is not the repaired representation: replaying a
    nine-leaf-only record would silently zero the encoder's ragged input.
    """
    _ensure_pa3()
    from agent.types.observation import ObservationBatch

    _rt.assert_representation(row.observation_representation)

    def t(name, dtype):
        return torch.as_tensor(row.observation[name]).to(device=device).to(dtype)

    # Stored leaves are per-token/per-feature vectors, so add a batch axis.
    obs = ObservationBatch(
        token_mask=t("token_mask", torch.bool).unsqueeze(0),
        categories=t("categories", torch.long).unsqueeze(0),
        category_known=t("category_known", torch.bool).unsqueeze(0),
        floats=t("floats", torch.float32).unsqueeze(0),
        float_known=t("float_known", torch.bool).unsqueeze(0),
        flags=t("flags", torch.bool).unsqueeze(0),
        flag_known=t("flag_known", torch.bool).unsqueeze(0),
        role_ids=t("role_ids", torch.long).unsqueeze(0),
        side_ids=t("side_ids", torch.long).unsqueeze(0),
        ragged=_ragged_batch(row, device),
    )
    return obs


def _ragged_batch(row, device):
    """Wrap the stored ragged transport leaves as one batched view."""
    from agent.types.ragged import FIELDS, RaggedBatch

    out = {}
    for family, fields in FIELDS.items():
        counts = row.observation_ragged[f"ragged.{family}_counts"]
        values = row.observation_ragged[f"ragged.{family}_values"]
        counts_np = np.asarray(torch.as_tensor(counts).numpy(), dtype=np.int64)
        if counts_np.ndim == 1:
            counts_np = counts_np[None, :]
        packed = []
        for tok in range(counts_np.shape[-1]):
            k = int(counts_np[0, tok])
            if k:
                packed.append(torch.as_tensor(values)[tok, :k].numpy().astype(np.float32))
        flat = (np.concatenate(packed, axis=0) if packed
                else np.zeros((0, len(fields)), np.float32))
        out[family] = RaggedBatch.from_rows([flat], counts_np, len(fields)).to(device)
    return out


def replay_seat_row(scorer, row, device="cpu"):
    """Score a stored row under the frozen scorer WITHOUT sampling.

    Returns dict with per-branch deltas and the joint delta.
    """
    obs = build_observation(row, device=device)
    encoded = scorer.encode(obs)
    # Normal actions require the player-known active-to-roster map, exactly as
    # the live actor derives it at ENCODE time.
    a2r = obs.active_to_roster()

    # CRITICAL: score every level in ONE branch table. The conditional scorer is
    # prefix-dependent (a GRU carries hidden state across levels), so scoring
    # levels independently resets the prefix and produces growing drift
    # (measured 0 -> 7e-3 -> 9.9e-3 -> 2.5e-2 on a 4-level request).
    table = _multi_level_table(scorer, row, a2r).to(scorer.device)
    selected = torch.tensor(
        [[int(b["selected_index"]) for b in row.branches]],
        dtype=torch.long, device=scorer.device,
    )
    with torch.no_grad():
        evaluation = scorer.model.evaluate_encoded(encoded, table, selected=selected)

    branch_deltas = []
    joint_new = 0.0
    for level, branch in enumerate(row.branches):
        lp = float(evaluation.logprob_selected[0, level].item())
        stored = float(branch["branch_logprob_fp32"])
        branch_deltas.append(abs(lp - stored))
        joint_new += lp

    joint_delta = abs(joint_new - float(row.joint_logprob_fp32))
    return {
        "branch_deltas": branch_deltas,
        "max_branch_delta": max(branch_deltas) if branch_deltas else 0.0,
        "joint_delta": joint_delta,
        "joint_replayed": joint_new,
        "joint_stored": float(row.joint_logprob_fp32),
    }


def assert_within_bounds(result, where=""):
    if result["max_branch_delta"] > MAX_BRANCH_LOGPROB_DELTA:
        raise ParityError(
            f"{where}: branch logprob delta {result['max_branch_delta']:.3e} exceeds "
            f"{MAX_BRANCH_LOGPROB_DELTA:.0e}")
    if result["joint_delta"] > MAX_JOINT_LOGPROB_DELTA:
        raise ParityError(
            f"{where}: joint logprob delta {result['joint_delta']:.3e} exceeds "
            f"{MAX_JOINT_LOGPROB_DELTA:.0e}")


def frozen_values(model, rows, device="cpu"):
    """Observation-only critic values under the FROZEN snapshot.

    The critic sees only the pre-action observation — never the selected action,
    opponent hidden truth, or the terminal winner.
    """
    out = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            obs = build_observation(row, device=device)
            encoded = model.encode(obs)
            out.append(float(model.value(encoded).item()))
    return out

def _multi_level_table(scorer, row, a2r):
    """Stack one `build_branch_table` result per level into a single table.

    `build_branch_table` produces [batch, 1, capacity, ...]; concatenating on the
    branch axis yields [1, n_levels, capacity, ...] which is exactly the shape
    `evaluate_encoded` walks with prefix carry.
    """
    from pa4.actions.conditional_policy import build_branch_table

    parts = []
    for branch in row.branches:
        t = build_branch_table([list(branch["candidates"])], scorer.layout, a2r)
        parts.append(t)
    if len(parts) == 1:
        return parts[0]
    return type(parts[0])(
        action_ids=torch.cat([t.action_ids for t in parts], dim=1),
        mask=torch.cat([t.mask for t in parts], dim=1),
        entity_token=torch.cat([t.entity_token for t in parts], dim=1),
        move_token=torch.cat([t.move_token for t in parts], dim=1),
        branch_valid=torch.cat([t.branch_valid for t in parts], dim=1),
        selected=torch.cat([t.selected for t in parts], dim=1),
    )
