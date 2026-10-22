"""P3-C: bounded, in-memory PA4ActorTransitionV1 records.

Design: PA4-P3-DESIGN-v1.0 §3.2. Records hold the *pre-action* per-seat
observation by value, the ordered candidate table for every prefix level, the
selected index/logprob, the real P2 action packet and the native outcome, so P4
can replay a behaviour decision exactly.

Storage is deliberately bounded: a fixed-capacity deque that drops the oldest
record when full and counts the drop. There is no database, no serialization and
no unbounded retention.
"""

from __future__ import annotations

from collections import deque

OBS_LEAVES = ("token_mask", "categories", "category_known", "floats", "float_known",
              "flags", "flag_known", "role_ids", "side_ids")


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


def seat_record(seat: int, request_id: int, view: dict, branches: list[dict],
                actor_active: bool, joint_logprob_fp32: float) -> dict:
    """One seat's decision, including its pre-action observation by value."""
    return {
        "seat": int(seat),
        "request_id": int(request_id),
        "observation": {k: _clone(view[k]) for k in OBS_LEAVES},
        "actor_active": bool(actor_active),
        "joint_logprob_fp32": float(joint_logprob_fp32),
        "branches": [
            {
                "level": int(b["level"]),
                "candidates": tuple(tuple(int(v) for v in row)
                                    for row in b["candidates"]),
                "selected_index": int(b["selected_index"]),
                "branch_k": int(b["branch_k"]),
                "branch_logprob_fp32": float(b["branch_logprob_fp32"]),
            }
            for b in branches
        ],
    }


def transition(*, run_id: str, worker_id: int, native_slot: int,
               native_generation: int, episode_id: int, step_index: int,
               request_ids, policy_version: int, policy_weight_fingerprint: str,
               observation_contract_hash: str, action_packet: dict,
               reward, done: bool, terminated: bool, truncated: bool,
               match_completed: bool, winner: int, request_revised: bool,
               match_id, seat_records: list[dict]) -> dict:
    """Assemble one PA4ActorTransitionV1 record (plain, detached values)."""
    return {
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
