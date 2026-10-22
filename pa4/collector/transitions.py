"""P3-C: bounded, in-memory PA4ActorTransitionV1 records.

Design: PA4-P3-DESIGN-v1.0 §3.2. A record holds the *pre-action* per-seat
observation by value, the ordered candidate table for every prefix level (with
its legal mask and token refs), the selected index/logprob, the real P2 action
packet, the post-action per-seat view and the native outcome, so P4 can replay a
behaviour decision exactly.

Storage is deliberately bounded: a fixed-capacity deque that drops the oldest
record when full and counts the drop. There is no database, no serialization and
no unbounded retention.
"""

from __future__ import annotations

import hashlib
from collections import deque

from pa4.env import ragged_transport as _ragged

OBS_LEAVES = ("token_mask", "categories", "category_known", "floats", "float_known",
              "flags", "flag_known", "role_ids", "side_ids")

#: P3.5/ADR-009. Stamped on every record so P4 ingest can refuse pre-repair data.
OBSERVATION_REPRESENTATION = _ragged.OBSERVATION_REPRESENTATION

SCHEMA_VERSION = "pa4.actor_transition.v1"


class BoundedTransitionStore:
    """Fixed-capacity FIFO of transition records with explicit drop accounting."""

    def __init__(self, capacity: int = 256):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._q: deque = deque()
        self.stored = 0
        self.dropped = 0

    def put(self, record: dict) -> None:
        if len(self._q) >= self.capacity:
            self._q.popleft()
            self.dropped += 1
        self._q.append(record)
        self.stored += 1

    def drain(self) -> list[dict]:
        out = list(self._q)
        self._q.clear()
        return out

    def __len__(self) -> int:
        return len(self._q)

    def __iter__(self):
        return iter(list(self._q))

    def summary(self) -> dict:
        return {"capacity": self.capacity, "stored": self.stored,
                "dropped": self.dropped, "retained": len(self._q)}


def accepted_records(records: list[dict], revised_rows) -> list[dict]:
    """Behaviour records only for rows whose step was actually accepted.

    A revised step (`accepted=False`) did not apply the tentative choices, so
    those seats must not be emitted or counted as on-policy behaviour rows.
    """
    revised = set(revised_rows)
    return [r for r in records if r.get("row") not in revised]


def _clone(t):
    return t.detach().to("cpu").clone()


def _require_ragged(view: dict, where: str) -> dict:
    """Require the explicit repaired representation and its ragged payload."""
    representation = view.get("_observation_representation")
    _ragged.assert_representation(representation)
    row = view.get("_ragged_row")
    if not row:
        raise _ragged.ObservationContractMismatch(
            f"{where}: view claims {representation!r} but carries no ragged row")
    _ragged.validate_transport_row(row, where=where)
    return row


def observation_digest(view: dict) -> str:
    """Content digest over one decoded view, dtype/shape aware.

    Covers the nine fixed leaves AND all five ragged families (P3.5/ADR-009),
    so a missing or mutated ragged family changes the digest. A fixed9-only
    view is refused here rather than producing a digest of an incomplete
    representation.
    """
    _require_ragged(view, "observation_digest")
    h = hashlib.sha256()
    for name in OBS_LEAVES:
        t = view[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    row = view["_ragged_row"]
    for key in sorted(row):
        t = row[key].detach().to("cpu").contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _clone_ragged(view: dict) -> dict:
    """Detach+clone the ragged transport rows carried by a decoded view."""
    row = view.get("_ragged_row") or {}
    return {k: _clone(v) for k, v in row.items()}


def seat_record(seat: int, request_id: int, view: dict, branches: list[dict],
                actor_active: bool, joint_logprob_fp32: float,
                policy_version: int, policy_weight_fingerprint: str,
                request_kind: int, precision_profile: str = "fp32") -> dict:
    """One seat's accepted decision, including its pre-action observation."""
    _require_ragged(view, f"seat_record seat {seat}")
    return {
        "seat": int(seat),
        "request_id": int(request_id),
        "observation": {k: _clone(view[k]) for k in OBS_LEAVES},
        "observation_ragged": _clone_ragged(view),
        "observation_representation": OBSERVATION_REPRESENTATION,
        "observation_digest": observation_digest(view),
        "request_kind": int(request_kind),
        "precision_profile": precision_profile,
        "actor_active": bool(actor_active),
        "joint_logprob_fp32": float(joint_logprob_fp32),
        "policy_version": int(policy_version),
        "policy_weight_fingerprint": policy_weight_fingerprint,
        "branches": [
            {
                "level": int(b["level"]),
                "candidates": tuple(tuple(int(v) for v in row)
                                    for row in b["candidates"]),
                "legal_mask": [bool(x) for x in b["legal_mask"]],
                "entity_token": [int(x) for x in b["entity_token"]],
                "move_token": [int(x) for x in b["move_token"]],
                "selected_index": int(b["selected_index"]),
                "branch_k": int(b["branch_k"]),
                "branch_logprob_fp32": float(b["branch_logprob_fp32"]),
            }
            for b in branches
        ],
    }


def next_observation(seat: int, view: dict, valid: bool) -> dict:
    """Post-action per-seat P2-visible data, including the same ragged contract.

    `observation_valid=False` (Wait/Finished) still carries the previous state's
    leaves by the P2 contract; P4 must mask them, never bootstrap from them. The
    representation id is only stamped when the matching ragged payload is
    retained, so a v2 label can never describe a nine-leaf-only record.
    """
    _require_ragged(view, f"next_observation seat {seat}")
    return {
        "seat": int(seat),
        "observation_valid": bool(valid),
        "observation": {k: _clone(view[k]) for k in OBS_LEAVES},
        "observation_ragged": _clone_ragged(view),
        "observation_representation": OBSERVATION_REPRESENTATION,
    }


def transition(*, run_id: str, worker_id: int, native_slot: int,
               native_generation: int, episode_id: int, step_index: int,
               request_ids, policy_version: int, policy_weight_fingerprint: str,
               observation_contract_hash: str, action_packet: dict,
               next_observations: list[dict],
               reward, done: bool, terminated: bool, truncated: bool,
               match_completed: bool, winner: int, request_revised: bool,
               match_id, seat_records: list[dict]) -> dict:
    """Assemble one PA4ActorTransitionV1 record (plain, detached values)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "worker_id": int(worker_id),
        "native_slot": int(native_slot),
        "native_generation": int(native_generation),
        "episode_id": int(episode_id),
        "step_index": int(step_index),
        "request_ids": [int(x) for x in request_ids],
        "policy_version": int(policy_version),
        "policy_weight_fingerprint": policy_weight_fingerprint,
        "observation_contract_hash": observation_contract_hash,
        "next_observation": next_observations,
        "real_p2_action_packet": action_packet,
        "reward": [float(x) for x in reward],
        "done": bool(done),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "match_completed": bool(match_completed),
        "winner": int(winner),
        "request_revised": bool(request_revised),
        "match_id": list(match_id) if match_id is not None else None,
        "seat_records": seat_records,
    }
