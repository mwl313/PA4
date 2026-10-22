"""P3: batched, collision-safe conditional scorer service.

Fixes the three real P1 limitations identified in the P3 design §4.1:

  1. `_scalar()` read only the first row of a stacked request.
  2. The cache key was `(0, request_key)`, which collides across workers,
     episodes and seats (P2 resets request_id to (1,1) per episode).
  3. The response carried a single selection.

Now every row is dispatched independently, the cache uses a full collision-free
identity, and compatible same-level STEP rows are scored in ONE model forward.

PA3's score mathematics is untouched: this is dispatch/batching only.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

MAX_TOKENS = 96
CATEGORY_COUNT = 32
FLOAT_COUNT = 50
FLAG_COUNT = 40
CANDIDATE_CAPACITY = 64
ACTION_FIELDS = 6

OP_ENCODE = 0
OP_STEP = 1
OP_RESET = 2

ERR_OK = 0
ERR_NO_CACHE = 1
ERR_NO_LEGAL = 2
ERR_BAD_OP = 3
ERR_OVERFLOW = 4
ERR_COUNT_MISMATCH = 5
ERR_STALE_IDENTITY = 6
ERR_MIXED_LEVEL = 7

ID_FIELDS = ("run_id", "worker_id", "native_slot", "native_generation",
             "episode_id", "seat", "request_local_id")

_FP_CACHE: dict[int, str] = {}


def model_fingerprint(seed: int = 20261006) -> str:
    """sha256 over the exact served weights, computed once per seed.

    Built on CPU so clients and tests can compute the same value without a GPU;
    the bytes are identical to the CUDA model because the weights are the same.
    """
    if seed in _FP_CACHE:
        return _FP_CACHE[seed]
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model

    torch.manual_seed(seed)
    model = PA3Model(PA3Config(initialization="random", seed=seed)).eval()
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    _FP_CACHE[seed] = h.hexdigest()
    return _FP_CACHE[seed]


def fingerprint_id(hex_digest: str) -> int:
    """Stable 63-bit id carried in the fixed int64 transport layout."""
    return int(hex_digest[:16], 16) & ((1 << 63) - 1)


def request_spec() -> TensorDict:
    """Unbatched single-slot request row (transport adds the batch dim)."""
    return TensorDict(
        {
            "op": torch.zeros((), dtype=torch.long),
            "request_key": torch.zeros((), dtype=torch.long),
            "policy_version": torch.zeros((), dtype=torch.long),
            "policy_fingerprint": torch.zeros((), dtype=torch.long),
            "branch_level": torch.zeros((), dtype=torch.long),
            # full collision-free identity
            "run_id": torch.zeros((), dtype=torch.long),
            "worker_id": torch.zeros((), dtype=torch.long),
            "native_slot": torch.zeros((), dtype=torch.long),
            "native_generation": torch.zeros((), dtype=torch.long),
            "episode_id": torch.zeros((), dtype=torch.long),
            "seat": torch.zeros((), dtype=torch.long),
            "request_local_id": torch.zeros((), dtype=torch.long),
            # observation
            "token_mask": torch.zeros(MAX_TOKENS, dtype=torch.bool),
            "categories": torch.zeros(MAX_TOKENS, CATEGORY_COUNT, dtype=torch.long),
            "category_known": torch.zeros(MAX_TOKENS, CATEGORY_COUNT, dtype=torch.bool),
            "floats": torch.zeros(MAX_TOKENS, FLOAT_COUNT, dtype=torch.float32),
            "float_known": torch.zeros(MAX_TOKENS, FLOAT_COUNT, dtype=torch.bool),
            "flags": torch.zeros(MAX_TOKENS, FLAG_COUNT, dtype=torch.bool),
            "flag_known": torch.zeros(MAX_TOKENS, FLAG_COUNT, dtype=torch.bool),
            "role_ids": torch.zeros(MAX_TOKENS, dtype=torch.long),
            "side_ids": torch.zeros(MAX_TOKENS, dtype=torch.long),
            # candidate table
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
            "policy_fingerprint": torch.zeros((), dtype=torch.long),
            "cache_size": torch.zeros((), dtype=torch.long),
            "error_code": torch.zeros((), dtype=torch.long),
        },
        batch_size=[],
    )


def _row_int(td: TensorDict, key: str, row: int) -> int:
    t = td[key]
    return int(t.reshape(-1)[row].item()) if t.dim() > 0 else int(t.item())


def identity_key(td: TensorDict, row: int) -> tuple:
    """Collision-free identity for one row.

    Includes run/worker/native slot/generation/episode/seat/request id AND the
    policy version, so a stale or cross-worker request can never alias.
    """
    return tuple(_row_int(td, f, row) for f in ID_FIELDS) + (
        _row_int(td, "policy_version", row),
    )


@dataclass
class CacheEntry:
    encoded: object
    policy_version: int
    steps_done: int
    hidden: object = None


class BatchedConditionalScorer(nn.Module):
    """Server model: full-row dispatch + real batched scoring on CUDA."""

    def __init__(self, seed: int = 20261006, device: str = "cpu", policy_version: int = 1):
        super().__init__()
        self.device = torch.device(device)
        self.policy_version = int(policy_version)
        self._cache: dict[tuple, CacheEntry] = {}
        self._model = None
        self._seed = seed
        # instrumentation (G3-B evidence)
        self.encode_calls = 0
        self.step_rows = 0
        self.reset_calls = 0
        self.forward_batch_sizes: list[int] = []
        self.model_forward_count = 0
        self.cache_high_water = 0
        # Provenance: which worker namespaces contributed to each forward.
        self.forward_run_ids: list[list[int]] = []
        self.fingerprint = model_fingerprint(self._seed)
        self.fingerprint_id = fingerprint_id(self.fingerprint)

    # -- model ------------------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            from agent.model.config import PA3Config
            from agent.model.pa3_model import PA3Model

            torch.manual_seed(self._seed)
            model = PA3Model(PA3Config(initialization="random", seed=self._seed))
            self._model = model.to(self.device).eval()
        return self._model

    @property
    def model_device(self) -> str:
        if self._model is None:
            return "not_built"
        return str(next(self._model.parameters()).device)

    def cache_size(self) -> int:
        return len(self._cache)

    # -- observation / candidate helpers ---------------------------------
    def _obs_batch(self, td: TensorDict, rows: list[int]):
        from agent.types.observation import ObservationBatch

        def col(name, dtype):
            t = td[name]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            elif t.dim() == 2 and name in ("token_mask", "role_ids", "side_ids"):
                t = t  # already [B, T]
            return t[rows].to(dtype).to(self.device)

        return ObservationBatch(
            token_mask=col("token_mask", torch.bool),
            categories=col("categories", torch.long),
            category_known=col("category_known", torch.bool),
            floats=col("floats", torch.float32),
            float_known=col("float_known", torch.bool),
            flags=col("flags", torch.bool),
            flag_known=col("flag_known", torch.bool),
            role_ids=col("role_ids", torch.long),
            side_ids=col("side_ids", torch.long),
        )

    def _table_for_row(self, td: TensorDict, row: int):
        from agent.types.requests import BranchCandidatesBatch

        def pick(name, dtype):
            t = td[name]
            if t.dim() == 1:
                return t.to(dtype).unsqueeze(0).to(self.device)
            return t[row].to(dtype).unsqueeze(0).to(self.device)

        action_ids = pick("action_ids", torch.long).unsqueeze(1)
        legal = pick("legal_mask", torch.bool).unsqueeze(1)
        entity = pick("entity_token", torch.long).unsqueeze(1)
        move = pick("move_token", torch.long).unsqueeze(1)
        return BranchCandidatesBatch(
            action_ids=action_ids,
            mask=legal,
            entity_token=entity,
            move_token=move,
            branch_valid=torch.ones((1, 1), dtype=torch.bool, device=self.device),
            selected=torch.full((1, 1), -1, dtype=torch.long, device=self.device),
        )

    # -- forward: dispatch every row, batch compatible STEPs -------------
    def forward(self, td: TensorDict) -> TensorDict:
        n_rows = td["op"].shape[0] if td["op"].dim() > 0 else 1
        results = [self._blank(td, r) for r in range(n_rows)]

        # Group rows by op so STEP rows at the same level can share a forward.
        by_op: dict[int, list[int]] = {}
        for r in range(n_rows):
            by_op.setdefault(_row_int(td, "op", r), []).append(r)

        for op, rows in by_op.items():
            if op == OP_RESET:
                self._do_reset(td, rows, results)
            elif op == OP_ENCODE:
                self._do_encode(td, rows, results)
            elif op == OP_STEP:
                self._do_step(td, rows, results)
            else:
                for r in rows:
                    results[r]["error_code"] = ERR_BAD_OP

        self.cache_high_water = max(self.cache_high_water, len(self._cache))
        return self._pack(results)

    def _blank(self, td, row) -> dict:
        return {
            "selected_index": -1,
            "branch_logprob": 0.0,
            "branch_k": 0,
            "request_key": _row_int(td, "request_key", row),
            "branch_level": _row_int(td, "branch_level", row),
            "policy_version": self.policy_version,
            "policy_fingerprint": self.fingerprint_id,
            "cache_size": len(self._cache),
            "error_code": ERR_OK,
        }

    def _do_reset(self, td, rows, results):
        for r in rows:
            self._cache.pop(identity_key(td, r), None)
            self.reset_calls += 1
            results[r]["cache_size"] = len(self._cache)

    def _version_ok(self, td, r) -> bool:
        if _row_int(td, "policy_version", r) != self.policy_version:
            return False
        fp = _row_int(td, "policy_fingerprint", r)
        return fp == 0 or fp == self.fingerprint_id

    def _do_encode(self, td, rows, results):
        model = self._ensure_model()
        keys = {r: identity_key(td, r) for r in rows}
        counts: dict[tuple, int] = {}
        for k in keys.values():
            counts[k] = counts.get(k, 0) + 1
        valid_rows = []
        for r in rows:
            key = keys[r]
            if key in self._cache or counts[key] > 1:
                # Either a genuine retry or two rows claiming the same identity
                # in one batch: both must fail closed, never overwrite.
                results[r]["error_code"] = ERR_STALE_IDENTITY
                continue
            if not self._version_ok(td, r):
                results[r]["error_code"] = ERR_STALE_IDENTITY
                continue
            valid_rows.append(r)
        if not valid_rows:
            return
        obs = self._obs_batch(td, valid_rows)
        with torch.no_grad():
            run_ids = sorted({_row_int(td, "run_id", r) for r in valid_rows})
            self.forward_run_ids.append(run_ids)
            encoded = model.encode(obs)
        self.encode_calls += len(valid_rows)
        for i, r in enumerate(valid_rows):
            key = identity_key(td, r)
            single = _slice_encoded(encoded, i)
            self._cache[key] = CacheEntry(
                encoded=single,
                policy_version=_row_int(td, "policy_version", r),
                steps_done=0,
            )
            results[r]["cache_size"] = len(self._cache)

    def _do_step(self, td, rows, results):
        model = self._ensure_model()
        ready = []
        keys = {r: identity_key(td, r) for r in rows}
        counts: dict[tuple, int] = {}
        for k in keys.values():
            counts[k] = counts.get(k, 0) + 1
        for r in rows:
            key = keys[r]
            entry = self._cache.get(key)
            level = _row_int(td, "branch_level", r)
            pv = _row_int(td, "policy_version", r)
            if counts[key] > 1:
                self._cache.pop(key, None)
                results[r]["error_code"] = ERR_STALE_IDENTITY
                continue
            if not self._version_ok(td, r):
                self._cache.pop(key, None)
                results[r]["error_code"] = ERR_STALE_IDENTITY
                continue
            if entry is None or entry.policy_version != pv or entry.steps_done != level:
                self._cache.pop(key, None)
                results[r]["error_code"] = ERR_NO_CACHE
                continue
            k = _row_int(td, "candidate_count", r)
            legal = td["legal_mask"]
            legal_row = legal[r] if legal.dim() > 1 else legal
            if k > CANDIDATE_CAPACITY:
                self._cache.pop(key, None)
                results[r]["error_code"] = ERR_OVERFLOW
                continue
            if k != int(legal_row.sum().item()):
                self._cache.pop(key, None)
                results[r]["error_code"] = ERR_COUNT_MISMATCH
                continue
            if k <= 0 or not bool(legal_row[:k].any()):
                self._cache.pop(key, None)
                results[r]["error_code"] = ERR_NO_LEGAL
                continue
            ready.append(r)

        if not ready:
            return

        # --- REAL batched forward: one model call for all compatible rows ---
        tables = [self._table_for_row(td, r) for r in ready]
        hidden_list = []
        for r in ready:
            key = identity_key(td, r)
            entry = self._cache[key]
            if entry.hidden is None:
                entry.hidden = model.scorer.initial_state(
                    1,
                    device=self.device,
                    dtype=torch.float32,
                )
            hidden_list.append(entry.hidden)

        # If all hidden states share a shape we can run a single batched pass.
        stacked_hidden = _try_stack(hidden_list)
        if stacked_hidden is not None and len(ready) > 1 and _same_level(td, ready):
            self.forward_run_ids.append(sorted({_row_int(td, "run_id", r) for r in ready}))
            out = self._batched_step(ready, td, tables, stacked_hidden)
        else:
            out = self._serial_step(ready, td, tables, hidden_list)

        for r, (idx, lp, k, hidden) in zip(ready, out):
            key = identity_key(td, r)
            entry = self._cache[key]
            entry.steps_done += 1
            entry.hidden = hidden
            results[r].update({
                "selected_index": idx,
                "branch_logprob": lp,
                "branch_k": k,
                "cache_size": len(self._cache),
            })

    def _batched_step(self, ready, td, tables, hidden):
        """One real model forward covering every ready row."""
        model = self._ensure_model()
        from agent.types.requests import BranchCandidatesBatch

        action_ids = torch.cat([t.action_ids for t in tables], dim=0)
        mask = torch.cat([t.mask for t in tables], dim=0)
        entity = torch.cat([t.entity_token for t in tables], dim=0)
        move = torch.cat([t.move_token for t in tables], dim=0)
        table = BranchCandidatesBatch(
            action_ids=action_ids, mask=mask, entity_token=entity, move_token=move,
            branch_valid=torch.ones((len(tables), 1), dtype=torch.bool, device=self.device),
            selected=torch.full((len(tables), 1), -1, dtype=torch.long, device=self.device),
        )
        enc = _stack_encoded([self._cache[identity_key(td, r)].encoded for r in ready])
        with torch.no_grad():
            # hidden is already [B, D] (see scorer.initial_state); do NOT add a dim.
            step = model.sample_level_step(enc, hidden, table, temperature=1.0)
        self.model_forward_count += 1
        self.forward_batch_sizes.append(len(ready))
        self.step_rows += len(ready)
        return [
            (int(step.pick[i].item()), float(step.logprob[i].item()),
             int(step.branch_k[i].item()), step.hidden[i:i + 1])
            for i in range(len(ready))
        ]

    def _serial_step(self, ready, td, tables, hidden_list):
        model = self._ensure_model()
        out = []
        for r, table, hidden in zip(ready, tables, hidden_list):
            enc = self._cache[identity_key(td, r)].encoded
            with torch.no_grad():
                step = model.sample_level_step(enc, hidden, table, temperature=1.0)
            self.model_forward_count += 1
            self.forward_batch_sizes.append(1)
            self.step_rows += 1
            out.append((int(step.pick[0].item()), float(step.logprob[0].item()),
                        int(step.branch_k[0].item()), step.hidden))
        return out

    def _pack(self, results) -> TensorDict:
        n = len(results)
        return TensorDict(
            {
                "selected_index": torch.tensor([r["selected_index"] for r in results], dtype=torch.long),
                "branch_logprob": torch.tensor([r["branch_logprob"] for r in results], dtype=torch.float32),
                "branch_k": torch.tensor([r["branch_k"] for r in results], dtype=torch.long),
                "request_key": torch.tensor([r["request_key"] for r in results], dtype=torch.long),
                "branch_level": torch.tensor([r["branch_level"] for r in results], dtype=torch.long),
                "policy_version": torch.tensor([r["policy_version"] for r in results], dtype=torch.long),
                "policy_fingerprint": torch.tensor(
                    [r["policy_fingerprint"] for r in results], dtype=torch.long),
                "cache_size": torch.tensor([r["cache_size"] for r in results], dtype=torch.long),
                "error_code": torch.tensor([r["error_code"] for r in results], dtype=torch.long),
            },
            batch_size=[n],
        )


def _same_level(td, rows) -> bool:
    levels = {_row_int(td, "branch_level", r) for r in rows}
    return len(levels) == 1


def _try_stack(hidden_list):
    try:
        return torch.cat(hidden_list, dim=0)
    except Exception:
        return None


def _slice_encoded(encoded, i):
    """Return a single-row view of a batched EncodedState."""
    from agent.model.pa3_model import EncodedState

    return EncodedState(
        tokens=encoded.tokens[i:i + 1],
        global_repr=encoded.global_repr[i:i + 1],
        token_mask=encoded.token_mask[i:i + 1],
        layout=encoded.layout,
    )


def _stack_encoded(encoded_list):
    from agent.model.pa3_model import EncodedState

    return EncodedState(
        tokens=torch.cat([e.tokens for e in encoded_list], dim=0),
        global_repr=torch.cat([e.global_repr for e in encoded_list], dim=0),
        token_mask=torch.cat([e.token_mask for e in encoded_list], dim=0),
        layout=encoded_list[0].layout,
    )


def ensure_pa3_on_path():
    ref = os.environ.get("PA4_PA3_REF", "/home/pa4/projects/_pa3_ref")
    if ref not in sys.path:
        sys.path.insert(0, ref)
