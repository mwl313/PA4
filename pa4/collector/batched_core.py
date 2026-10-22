"""P3 optimized batched collector.

Architecture (per PA4-P3-DESIGN-v1.0 §2.3): ONE actor process owns a
PA4EnvBase(batch_size=N) holding N real Rust battles. Every staged inference phase
is issued for ALL rows at once, so the GPU always receives a wide batch instead
of a stream of single requests.

Performance techniques, all semantics-preserving:
  * `request_info_batch` fetches kind + branch count for every row/seat in ONE
    native call instead of 2 calls per row.
  * `candidates_batch` is issued per level for ALL pending requests together.
  * `observe_fixed_batch` decodes every requesting seat in ONE call.
  * Observation output tensors are preallocated once and written in place.
  * Redundant per-row candidate re-validation is removed from the hot path (Rust
    already enforces legality on submission and returns an explicit error), but a
    strict mode keeps it available for tests.
  * Inference is submitted phase-wise (all ENCODEs, then all STEP level 0, ...)
    so ProcessSlotTransport sees many simultaneous requests.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import torch

OBS_LEAVES = ("token_mask", "categories", "category_known", "floats", "float_known",
              "flags", "flag_known", "role_ids", "side_ids")
KIND_FINISHED = 4
POLICY_KINDS = (0, 1, 2)
MAX_BRANCHES = 4


def _ensure_paths():
    for p in ("/home/pa4/projects/_pa3_ref", "/home/pa4/projects/PokeAgent4.0"):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class BatchedConfig:
    num_envs: int = 8
    target_matches: int = 16
    max_steps: int = 20000
    seed: int = 0
    policy_version: int = 1
    device: str = "cuda"
    max_batch_size: int = 64
    min_batch_size: int = 4
    server_timeout: float = 0.05
    strict_validation: bool = False
    use_fast_obs: bool = True
    run_id: str = "pa4-p3-batched"
    worker_index: int = 0
    teams: list | None = None


@dataclass
class BatchedReport:
    run_id: str = ""
    num_envs: int = 0
    natural_matches: int = 0
    match_ids: list = field(default_factory=list)
    duplicate_match_ids: int = 0
    env_steps: int = 0
    env_transitions: int = 0
    seat_records: int = 0
    actor_active_rows: int = 0
    seats_by_seat: list = field(default_factory=lambda: [0, 0])
    illegal_actions: int = 0
    revisions: int = 0
    truncations: int = 0
    operational_errors: int = 0
    errors: list = field(default_factory=list)
    wall_seconds: float = 0.0
    warmup_seconds: float = 0.0
    steady_seconds: float = 0.0
    steady_matches: int = 0
    steady_env_steps: int = 0
    service_stats: dict = field(default_factory=dict)
    server_stats: dict = field(default_factory=dict)

    def as_dict(self):
        d = dict(self.__dict__)
        d["raw_natural_matches_per_sec"] = (
            self.natural_matches / self.wall_seconds if self.wall_seconds else None)
        d["steady_matches_per_sec"] = (
            self.steady_matches / self.steady_seconds if self.steady_seconds else None)
        d["env_transitions_per_sec"] = (
            self.env_transitions / self.wall_seconds if self.wall_seconds else None)
        return d


class StagedInference:
    """Phase-batched façade over N transport client slots."""

    def __init__(self, clients, spec, policy_version):
        self.clients = clients
        self.spec = spec
        self.policy_version = policy_version
        self._free = list(range(len(clients)))

    def _acquire(self):
        if not self._free:
            raise RuntimeError("no free inference slot")
        return self._free.pop()

    def _release(self, slot):
        self._free.append(slot)
        self._free.sort()

    def _row(self, op, ident, level=0):
        td = self.spec.clone()
        td["op"] = torch.tensor(op)
        td["worker_id"] = torch.tensor(ident[0])
        td["native_slot"] = torch.tensor(ident[1])
        td["native_generation"] = torch.tensor(ident[2])
        td["episode_id"] = torch.tensor(ident[3])
        td["seat"] = torch.tensor(ident[4])
        td["request_local_id"] = torch.tensor(ident[5])
        td["policy_version"] = torch.tensor(self.policy_version)
        td["request_key"] = torch.tensor(ident[5] % (2 ** 31 - 1))
        td["branch_level"] = torch.tensor(level)
        return td

    def encode_all(self, items):
        """items: [(ident, view)] -> list of (ident, layout, a2r)."""
        futs = []
        for ident, view in items:
            slot = self._acquire()
            td = self._row(0, ident)
            for k in OBS_LEAVES:
                td[k] = view[k]
            futs.append((slot, ident, view, self.clients[slot].submit(td)))
        out = []
        for slot, ident, view, fut in futs:
            r = fut.result()
            err = int(r["error_code"])
            self._release(slot)
            if err:
                raise RuntimeError(f"ENCODE err {err}")
            ob = _view_to_obs(view)
            out.append((ident, ob.layout, ob.active_to_roster(), ob))
        return out

    def step_level(self, items, level):
        """items: [(ident, candidates, layout, a2r)] -> [(ident, idx, k, lp)]."""
        futs = []
        for (ident, cands, layout, a2r) in items:
            from pa4.actions.conditional_policy import build_branch_table

            table = build_branch_table([cands], layout, a2r)
            slot = self._acquire()
            td = self._row(1, ident, level)
            k = len(cands)
            td["candidate_count"] = torch.tensor(k)
            td["action_ids"] = table.action_ids[0, 0]
            td["legal_mask"] = table.mask[0, 0]
            td["entity_token"] = table.entity_token[0, 0]
            td["move_token"] = table.move_token[0, 0]
            futs.append((slot, ident, self.clients[slot].submit(td)))
        out = []
        for slot, ident, fut in futs:
            r = fut.result()
            err = int(r["error_code"])
            self._release(slot)
            if err:
                raise RuntimeError(f"STEP err {err} at level {level}")
            out.append((ident, int(r["selected_index"]), int(r["branch_k"]),
                        float(r["branch_logprob"])))
        return out

    def reset_all(self, items):
        futs = []
        for ident in items:
            slot = self._acquire()
            futs.append((slot, self.clients[slot].submit(self._row(2, ident))))
        for slot, fut in futs:
            fut.result()
            self._release(slot)


def _view_to_obs(view):
    from agent.types.observation import ObservationBatch

    return ObservationBatch(
        token_mask=view["token_mask"].unsqueeze(0),
        categories=view["categories"].unsqueeze(0),
        category_known=view["category_known"].unsqueeze(0),
        floats=view["floats"].unsqueeze(0),
        float_known=view["float_known"].unsqueeze(0),
        flags=view["flags"].unsqueeze(0),
        flag_known=view["flag_known"].unsqueeze(0),
        role_ids=view["role_ids"].unsqueeze(0),
        side_ids=view["side_ids"].unsqueeze(0),
    )
