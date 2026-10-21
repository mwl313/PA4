"""P1-C: Real TorchRL InferenceServer carrying PA3's conditional scorer.

Design
------
The server owns the real PA3 `PA3Model` (9,889,755 params) plus a per-request
cache keyed by (slot, request_key). One fixed-layout TensorDict is used for every
call, with an `op` code selecting the action:

  op = 0  ENCODE : observation -> cached EncodedState
  op = 1  STEP   : score legal candidates at one branch level, sample, advance GRU
  op = 2  RESET  : drop the cached request state

Legality is never decided here: the client supplies the exact candidate rows the
Rust engine returned. The server only scores and masks them.

`ProcessSlotTransport` requires the request/response *slot* spec to be a single
unbatched row; the server stacks rows across slots into a leading batch dim.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

MAX_TOKENS = 96
CANDIDATE_CAPACITY = 64
MAX_BRANCHES = 4
ACTION_FIELDS = 6

OP_ENCODE = 0
OP_STEP = 1
OP_RESET = 2

ERR_OK = 0
ERR_NO_CACHE = 1
ERR_NO_LEGAL = 2
ERR_BAD_OP = 3


def _scalar(td: TensorDict, key: str) -> int:
    t = td[key]
    if t.dim() > 0:
        return int(t.reshape(-1)[0].item())
    return int(t.item())


def request_spec() -> TensorDict:
    """Representative single-slot request row (unbatched)."""
    return TensorDict(
        {
            "op": torch.zeros((), dtype=torch.long),
            "request_key": torch.zeros((), dtype=torch.long),
            "policy_version": torch.zeros((), dtype=torch.long),
            "branch_level": torch.zeros((), dtype=torch.long),
            "token_mask": torch.zeros(MAX_TOKENS, dtype=torch.bool),
            "categories": torch.zeros(MAX_TOKENS, 32, dtype=torch.long),
            "category_known": torch.zeros(MAX_TOKENS, 32, dtype=torch.bool),
            "floats": torch.zeros(MAX_TOKENS, 50, dtype=torch.float32),
            "float_known": torch.zeros(MAX_TOKENS, 50, dtype=torch.bool),
            "flags": torch.zeros(MAX_TOKENS, 40, dtype=torch.bool),
            "flag_known": torch.zeros(MAX_TOKENS, 40, dtype=torch.bool),
            "role_ids": torch.zeros(MAX_TOKENS, dtype=torch.long),
            "side_ids": torch.zeros(MAX_TOKENS, dtype=torch.long),
            "candidate_count": torch.zeros((), dtype=torch.long),
            "action_ids": torch.zeros(CANDIDATE_CAPACITY, ACTION_FIELDS, dtype=torch.long),
            "legal_mask": torch.zeros(CANDIDATE_CAPACITY, dtype=torch.bool),
            "entity_token": torch.zeros(CANDIDATE_CAPACITY, dtype=torch.long),
            "move_token": torch.full((CANDIDATE_CAPACITY,), -1, dtype=torch.long),
        },
        batch_size=[],
    )


def response_spec() -> TensorDict:
    return TensorDict(
        {
            "selected_index": torch.full((), -1, dtype=torch.long),
            "branch_logprob": torch.zeros((), dtype=torch.float32),
            "branch_k": torch.zeros((), dtype=torch.long),
            "request_key": torch.zeros((), dtype=torch.long),
            "branch_level": torch.zeros((), dtype=torch.long),
            "policy_version": torch.zeros((), dtype=torch.long),
            "cache_size": torch.zeros((), dtype=torch.long),
            "error_code": torch.zeros((), dtype=torch.long),
        },
        batch_size=[],
    )


@dataclass
class CacheEntry:
    encoded: object
    policy_version: int
    steps_done: int
    hidden: object = None


class ConditionalScorerService(nn.Module):
    """Owns the PA3 model and the per-request encoded/GRU cache."""

    def __init__(self, seed: int = 20261006, device: str = "cpu", policy_version: int = 1):
        super().__init__()
        self.device = torch.device(device)
        self.policy_version = int(policy_version)
        self._cache: dict[tuple[int, int], CacheEntry] = {}
        self._model = None
        self._seed = seed
        self.encode_calls = 0
        self.step_calls = 0

    def _ensure_model(self):
        if self._model is None:
            from agent.model.config import PA3Config
            from agent.model.pa3_model import PA3Model

            torch.manual_seed(self._seed)
            model = PA3Model(PA3Config(initialization="random", seed=self._seed))
            self._model = model.to(self.device).eval()
        return self._model

    def cache_size(self) -> int:
        return len(self._cache)

    def _observation_batch(self, td: TensorDict):
        """Build an ObservationBatch from the stacked slot batch.

        `ObservationBatch` validates *unbatched* per-view shapes ((96,), (96,32),
        ...). The transport hands us a stacked leading batch dim, so we rebuild
        the batch with an explicit batch_size instead of nesting dims.
        """
        from agent.types.observation import ObservationBatch, ObservationLayout

        n = td["token_mask"].shape[0]
        obs = ObservationBatch(
            token_mask=td["token_mask"].reshape(n, 96).to(torch.bool).to(self.device),
            categories=td["categories"].reshape(n, 96, 32).to(torch.long).to(self.device),
            category_known=td["category_known"].reshape(n, 96, 32).to(torch.bool).to(self.device),
            floats=td["floats"].reshape(n, 96, 50).to(torch.float32).to(self.device),
            float_known=td["float_known"].reshape(n, 96, 50).to(torch.bool).to(self.device),
            flags=td["flags"].reshape(n, 96, 40).to(torch.bool).to(self.device),
            flag_known=td["flag_known"].reshape(n, 96, 40).to(torch.bool).to(self.device),
            role_ids=td["role_ids"].reshape(n, 96).to(torch.long).to(self.device),
            side_ids=td["side_ids"].reshape(n, 96).to(torch.long).to(self.device),
        )
        return obs

    def _candidate_table(self, td: TensorDict):
        """Rebuild PA3's BranchCandidatesBatch (shape [B,1,64,...]) from a slot row."""
        from agent.types.requests import BranchCandidatesBatch

        def row(name):
            t = td[name]
            # Transport stacks slot rows: (64,6)->(B,64,6), (64,)->(B,64).
            # Ensure a leading batch dim, then no more.
            if t.dim() == 1:
                t = t.unsqueeze(0)
            elif t.dim() == 2 and t.shape[-1] == ACTION_FIELDS:
                t = t.unsqueeze(0)
            return t

        action_ids = row("action_ids").to(torch.long).unsqueeze(1)
        legal = row("legal_mask").to(torch.bool).unsqueeze(1)
        entity = row("entity_token").to(torch.long).unsqueeze(1)
        move = row("move_token").to(torch.long).unsqueeze(1)
        b = action_ids.shape[0]
        return BranchCandidatesBatch(
            action_ids=action_ids.to(self.device),
            mask=legal.to(self.device),
            entity_token=entity.to(self.device),
            move_token=move.to(self.device),
            branch_valid=torch.ones((b, 1), dtype=torch.bool, device=self.device),
            selected=torch.full((b, 1), -1, dtype=torch.long, device=self.device),
        )

    def forward(self, td: TensorDict) -> TensorDict:
        op = _scalar(td, "op")
        key = _scalar(td, "request_key")
        level = _scalar(td, "branch_level")
        pv = _scalar(td, "policy_version")

        out = {
            "selected_index": -1,
            "branch_logprob": 0.0,
            "branch_k": 0,
            "request_key": key,
            "branch_level": level,
            "policy_version": pv,
            "cache_size": len(self._cache),
            "error_code": ERR_OK,
        }

        def emit():
            return TensorDict(
                {
                    "selected_index": torch.tensor([out["selected_index"]], dtype=torch.long),
                    "branch_logprob": torch.tensor([out["branch_logprob"]], dtype=torch.float32),
                    "branch_k": torch.tensor([out["branch_k"]], dtype=torch.long),
                    "request_key": torch.tensor([out["request_key"]], dtype=torch.long),
                    "branch_level": torch.tensor([out["branch_level"]], dtype=torch.long),
                    "policy_version": torch.tensor([out["policy_version"]], dtype=torch.long),
                    "cache_size": torch.tensor([out["cache_size"]], dtype=torch.long),
                    "error_code": torch.tensor([out["error_code"]], dtype=torch.long),
                },
                batch_size=[1],
            )

        if op == OP_RESET:
            self._cache.pop((0, key), None)
            out["cache_size"] = len(self._cache)
            return emit()

        if op == OP_ENCODE:
            model = self._ensure_model()
            obs = self._observation_batch(td)
            with torch.no_grad():
                encoded = model.encode(obs)
            self._cache[(0, key)] = CacheEntry(encoded=encoded, policy_version=pv, steps_done=0)
            self.encode_calls += 1
            out["cache_size"] = len(self._cache)
            return emit()

        if op == OP_STEP:
            entry = self._cache.get((0, key))
            if entry is None or entry.policy_version != pv or entry.steps_done != level:
                out["error_code"] = ERR_NO_CACHE
                return emit()

            k = _scalar(td, "candidate_count")
            legal = td["legal_mask"]
            if legal.dim() == 2:
                legal = legal[0]
            if k <= 0 or not bool(legal[:k].any()):
                out["error_code"] = ERR_NO_LEGAL
                return emit()

            model = self._ensure_model()
            table = self._candidate_table(td)

            # The GRU prefix state MUST advance across levels. sample_levels()
            # restarts from Scorer.initial_state() each call, so a per-level
            # server call has to carry `hidden` itself (G1-B stale-hidden guard).
            if entry.hidden is None:
                entry.hidden = model.scorer.initial_state(
                    entry.encoded.tokens.shape[0],
                    device=entry.encoded.tokens.device,
                    dtype=entry.encoded.tokens.dtype
                    if entry.encoded.tokens.dtype.is_floating_point
                    else torch.float32,
                )

            with torch.no_grad():
                step = model.sample_level_step(
                    entry.encoded, entry.hidden, table, temperature=1.0, generator=None
                )
            entry.hidden = step.hidden
            entry.steps_done += 1
            self.step_calls += 1
            out["selected_index"] = int(step.pick[0].item())
            out["branch_logprob"] = float(step.logprob[0].item())
            out["branch_k"] = int(step.branch_k[0].item())
            out["cache_size"] = len(self._cache)
            return emit()

        out["error_code"] = ERR_BAD_OP
        return emit()


def ensure_pa3_on_path():
    ref = os.environ.get("PA4_PA3_REF", "/home/pa4/projects/_pa3_ref")
    if ref not in sys.path:
        sys.path.insert(0, ref)
