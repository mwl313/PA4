"""P1-B: Conditional (prefix-dependent) policy selection + replay.

Semantics implemented here (unchanged from PA3 at the pinned SHA):

  P(a_0:L-1 | s)       = prod_j P(a_j | s, a_<j, C_j(a_<j))
  log P(a_0:L-1 | s)   = sum_{j valid} log P(a_j | s, a_<j, C_j(a_<j))

Key invariants:
  * Legality is never invented here. Every candidate table comes from the Rust
    engine via NativeReference.candidates().
  * Illegal / padded candidates get exactly zero probability (PA3 masked
    log-softmax with FP32 `LOGPROB_FLOOR`).
  * The encoder runs ONCE per request; every branch reuses the encoded state and
    only advances the GRU prefix state.
  * Replay re-evaluates SAVED choices; it never resamples.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import torch

# Imported lazily so the module can be introspected without PA3 on sys.path.


def _pa3():
    from agent.model.config import PA3Config  # noqa: PLC0415
    from agent.model.pa3_model import PA3Model  # noqa: PLC0415
    from agent.types.actions import AtomicAction  # noqa: PLC0415
    from agent.types.observation import ObservationLayout  # noqa: PLC0415
    from agent.types.requests import ActionRef, BranchCandidatesBatch  # noqa: PLC0415

    return PA3Config, PA3Model, AtomicAction, ObservationLayout, ActionRef, BranchCandidatesBatch


CANDIDATE_CAPACITY = 64
MAX_BRANCHES = 4


@dataclass(frozen=True)
class BranchSnapshot:
    """Minimal sufficient evidence to recompute one branch's logprob exactly."""

    level: int
    branch_slot: int
    candidate_tuples: tuple[tuple[int, ...], ...]   # engine order, no truncation
    legal_mask: tuple[bool, ...]
    entity_token_refs: tuple[int, ...]
    move_token_refs: tuple[int, ...]
    selected_index: int
    branch_k: int
    branch_logprob_fp32: float

    def candidates_digest(self) -> str:
        """Stable identity of this branch's ordered candidate set."""
        h = hashlib.sha256()
        for row in self.candidate_tuples:
            h.update(bytes(int(v) & 0xFF for v in row))
        h.update(bytes(self.legal_mask))
        return h.hexdigest()


@dataclass(frozen=True)
class ActionReplayRecordV1:
    """Everything needed to recompute a chosen player action's joint logprob."""

    request_id: int
    policy_version: int
    policy_weight_fingerprint: str
    precision_profile: str
    schema_version: int
    contract_hash: str
    observation_digest: str
    branches: tuple[BranchSnapshot, ...]
    joint_logprob_fp32: float
    actor_active: bool
    request_kind: int

    @property
    def valid_branch_count(self) -> int:
        return sum(1 for b in self.branches if b.branch_k >= 1)

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "policy_version": self.policy_version,
            "policy_weight_fingerprint": self.policy_weight_fingerprint,
            "precision_profile": self.precision_profile,
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "observation_digest": self.observation_digest,
            "joint_logprob_fp32": self.joint_logprob_fp32,
            "actor_active": self.actor_active,
            "request_kind": self.request_kind,
            "branches": [
                {
                    "level": b.level,
                    "branch_slot": b.branch_slot,
                    "branch_k": b.branch_k,
                    "selected_index": b.selected_index,
                    "branch_logprob_fp32": b.branch_logprob_fp32,
                    "candidates_digest": b.candidates_digest(),
                }
                for b in self.branches
            ],
        }


def weight_fingerprint(model: torch.nn.Module) -> str:
    """Deterministic fingerprint of the exact weights under test (dtype/shape aware)."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def observation_digest(obs) -> str:
    """Content digest over the decoded observation, aware of dtype and shape."""
    h = hashlib.sha256()
    for attr in ("token_mask", "categories", "category_known", "floats",
                 "float_known", "flags", "flag_known", "role_ids", "side_ids"):
        t = getattr(obs, attr, None)
        if t is None:
            continue
        t = t.detach().cpu().contiguous()
        h.update(attr.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        if t.dtype == torch.bool:
            h.update(t.to(torch.uint8).numpy().tobytes())
        elif t.dtype.is_floating_point:
            h.update(t.to(torch.float32).numpy().tobytes())
        else:
            h.update(t.to(torch.int64).numpy().tobytes())
    return h.hexdigest()


def build_branch_table(candidate_groups: Sequence[Sequence[tuple[int, ...]]], layout,
                       active_to_roster=None):
    """Build BranchCandidatesBatch from real engine tuples for one level.

    Mirrors PA3's `_level_table`: capacity 64, explicit mask, and it FAILS
    LOUDLY on overflow instead of silently truncating candidates.

    `active_to_roster` is the player-known active->roster map from
    `ObservationBatch.active_to_roster()`. PA3 requires it to resolve token
    references for non-PICK actions (Normal/Replacement); P1 only ever exercised
    PICK actions, which do not need it.
    """
    _, _, AtomicAction, _, ActionRef, BranchCandidatesBatch = _pa3()

    batch = len(candidate_groups)
    capacity = CANDIDATE_CAPACITY
    action_ids = torch.zeros((batch, 1, capacity, 6), dtype=torch.long)
    mask = torch.zeros((batch, 1, capacity), dtype=torch.bool)
    entity = torch.zeros((batch, 1, capacity), dtype=torch.long)
    move = torch.full((batch, 1, capacity), -1, dtype=torch.long)

    for row_index, candidates in enumerate(candidate_groups):
        if not candidates:
            raise RuntimeError("request level without a legal candidate")
        if len(candidates) > capacity:
            raise RuntimeError(
                f"candidate capacity {capacity} exceeded ({len(candidates)}); "
                "refusing to truncate legal candidates (G1-E)"
            )
        for index, packed in enumerate(candidates):
            action = AtomicAction.from_tuple(packed)
            row_map = None if active_to_roster is None else active_to_roster[row_index]
            ref = ActionRef.resolve(layout, action, row_map)
            action_ids[row_index, 0, index] = torch.tensor(ref.tuple, dtype=torch.long)
            mask[row_index, 0, index] = True
            entity[row_index, 0, index] = int(ref.entity_token)
            move[row_index, 0, index] = int(ref.move_token)

    return BranchCandidatesBatch(
        action_ids=action_ids,
        mask=mask,
        entity_token=entity,
        move_token=move,
        branch_valid=torch.ones((batch, 1), dtype=torch.bool),
        selected=torch.full((batch, 1), -1, dtype=torch.long),
    )


class ConditionalPolicy:
    """Staged prefix sampler over real Rust candidates, with exact replay."""

    def __init__(self, model=None, seed: int = 20261006, device: str | torch.device = "cpu",
                 policy_version: int = 1):
        PA3Config, PA3Model, _, ObservationLayout, _, _ = _pa3()
        self.device = torch.device(device)
        self.policy_version = int(policy_version)
        if model is None:
            torch.manual_seed(seed)
            model = PA3Model(PA3Config(initialization="random", seed=seed))
        self.model = model.to(self.device).eval()
        self.layout = ObservationLayout()
        self.fingerprint = weight_fingerprint(self.model)

    # -- encoding -------------------------------------------------------
    def encode(self, observation):
        obs = observation.to(self.device)
        with torch.no_grad():
            return self.model.encode(obs)

    # -- staged selection ----------------------------------------------
    def select_branch(self, encoded, candidates: Sequence[tuple[int, ...]],
                      generator: torch.Generator | None = None, active_to_roster=None):
        """Sample ONE branch from real candidates; return (index, logprob, table)."""
        table = build_branch_table([candidates], self.layout, active_to_roster)
        table = type(table)(
            action_ids=table.action_ids.to(self.device),
            mask=table.mask.to(self.device),
            entity_token=table.entity_token.to(self.device),
            move_token=table.move_token.to(self.device),
            branch_valid=table.branch_valid.to(self.device),
            selected=table.selected.to(self.device),
        )
        with torch.no_grad():
            result = self.model.sample_levels(
                encoded, [table], temperature=1.0, generator=generator
            )
        idx = int(result.selected[0, 0].item())
        lp = float(result.logprob_selected[0, 0].item())
        k = int(result.branch_k[0, 0].item())
        return idx, lp, k, table

    def evaluate_branch(self, encoded, candidates: Sequence[tuple[int, ...]], index: int,
                        active_to_roster=None):
        """Score a SAVED index without sampling (the replay path)."""
        table = build_branch_table([candidates], self.layout, active_to_roster)
        table = type(table)(
            action_ids=table.action_ids.to(self.device),
            mask=table.mask.to(self.device),
            entity_token=table.entity_token.to(self.device),
            move_token=table.move_token.to(self.device),
            branch_valid=table.branch_valid.to(self.device),
            selected=torch.tensor([[index]], dtype=torch.long, device=self.device),
        )
        with torch.no_grad():
            evaluation = self.model.evaluate_encoded(encoded, table, selected=table.selected)
        lp = float(evaluation.logprob_selected[0, 0].item())
        return lp, evaluation

    # -- full request ---------------------------------------------------
    def complete_request(self, native, slot: int, generation: int, side: int,
                         request_id: int, observation,
                         generator: torch.Generator | None = None,
                         temperature: float = 1.0):
        """Walk every branch level, asking Rust for candidates after each pick.

        Returns (prefix_actions, record). `prefix_actions` is the ordered list of
        canonical 6-field tuples that forms the completed player action.
        """
        req = native.request(slot, generation, side)
        if not req.produces_actor_row:
            raise RuntimeError(f"request kind {req.kind_label} yields no actor row")

        encoded = self.encode(observation)
        obs_digest = observation_digest(observation)
        a2r = observation.active_to_roster()  # player-known map from PA3 layout

        prefix_actions: list[tuple[int, ...]] = []
        snapshots: list[BranchSnapshot] = []

        # Preview has 4 branch slots; Normal/Replacement have 2 (engine-provided).
        for level, branch_slot in enumerate(req.branch_slots):
            candidates = native.candidates(slot, generation, side, prefix_actions)
            idx, lp, k, table = self.select_branch(encoded, candidates, generator, a2r)
            chosen = candidates[idx]
            prefix_actions.append(chosen)

            snapshots.append(
                BranchSnapshot(
                    level=level,
                    branch_slot=int(branch_slot),
                    candidate_tuples=tuple(candidates),
                    legal_mask=tuple(bool(x) for x in table.mask[0, 0, :len(candidates)].tolist()),
                    entity_token_refs=tuple(int(v) for v in table.entity_token[0, 0, :len(candidates)].tolist()),
                    move_token_refs=tuple(int(v) for v in table.move_token[0, 0, :len(candidates)].tolist()),
                    selected_index=idx,
                    branch_k=k,
                    branch_logprob_fp32=lp,
                )
            )

        joint = float(sum(s.branch_logprob_fp32 for s in snapshots))
        actor_active = any(s.branch_k >= 2 for s in snapshots)

        record = ActionReplayRecordV1(
            request_id=request_id,
            policy_version=self.policy_version,
            policy_weight_fingerprint=self.fingerprint,
            precision_profile="fp32_reference",
            schema_version=int(self.layout.SCHEMA_VERSION) if hasattr(self.layout, "SCHEMA_VERSION") else 2,
            contract_hash=self.layout.CONTRACT_HASH if hasattr(self.layout, "CONTRACT_HASH") else "",
            observation_digest=obs_digest,
            branches=tuple(snapshots),
            joint_logprob_fp32=joint,
            actor_active=actor_active,
            request_kind=req.kind,
        )
        return prefix_actions, record

    # -- replay ---------------------------------------------------------
    def replay(self, record: ActionReplayRecordV1, observation):
        """Recompute the saved action's logprob exactly. Never resamples."""
        # Recompute live so in-place weight mutation is detected, not just a
        # stale fingerprint captured at construction time.
        if weight_fingerprint(self.model) != record.policy_weight_fingerprint:
            raise RuntimeError("replay refused: policy weights differ from the record")
        if record.policy_version != self.policy_version:
            raise RuntimeError("replay refused: policy_version mismatch")
        if observation_digest(observation) != record.observation_digest:
            raise RuntimeError("replay refused: observation content differs from the record")

        # Validate the stored branch metadata BEFORE scoring: an illegal index
        # or an inconsistent mask/branch_k must fail closed, not be rescored.
        for snap in record.branches:
            n = len(snap.candidate_tuples)
            if not (0 <= snap.selected_index < n):
                raise RuntimeError(
                    f"replay refused: selected_index {snap.selected_index} outside "
                    f"[0,{n}) at level {snap.level}"
                )
            if len(snap.legal_mask) != n:
                raise RuntimeError(
                    f"replay refused: legal_mask length {len(snap.legal_mask)} != {n}"
                )
            if not snap.legal_mask[snap.selected_index]:
                raise RuntimeError(
                    f"replay refused: selected index {snap.selected_index} is masked illegal"
                )
            if snap.branch_k != sum(snap.legal_mask):
                raise RuntimeError(
                    f"replay refused: branch_k {snap.branch_k} != mask sum "
                    f"{sum(snap.legal_mask)} at level {snap.level}"
                )

        encoded = self.encode(observation)
        a2r = observation.active_to_roster()
        recomputed: list[float] = []
        for snap in record.branches:
            lp, _ = self.evaluate_branch(encoded, snap.candidate_tuples,
                                        snap.selected_index, a2r)
            recomputed.append(lp)

        joint = float(sum(recomputed))
        deltas = [abs(a - b.branch_logprob_fp32) for a, b in zip(recomputed, record.branches)]
        return {
            "branch_logprobs": recomputed,
            "joint_logprob": joint,
            "max_abs_branch_delta": max(deltas) if deltas else 0.0,
            "abs_joint_delta": abs(joint - record.joint_logprob_fp32),
        }
