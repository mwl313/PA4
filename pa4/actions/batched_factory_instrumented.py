"""Module-level instrumented factory for P3 batching evidence (spawn-safe).

Must be importable by the child process, so the factory cannot be a closure.
Instrumentation is written to PA4_P3_STATS_PATH after every forward.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

from pa4.actions.batched_scorer import BatchedConditionalScorer


def _tensor_feed(h, t):
    t = t.detach().to("cpu").contiguous()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())


def _encoded_digest(encoded):
    h = hashlib.sha256()
    for name in ("tokens", "global_repr", "token_mask"):
        h.update(name.encode())
        _tensor_feed(h, getattr(encoded, name))
    return h.hexdigest()


def _ragged_batch_digest(ragged):
    """Digest the exact RaggedBatch objects that reach the encoder."""
    from pa4.env import ragged_transport as _rt

    h = hashlib.sha256()
    for family in _rt.family_names():
        section = ragged[family]
        h.update(f"ragged.{family}_counts".encode())
        _tensor_feed(h, section.counts)
        h.update(f"ragged.{family}_values".encode())
        _tensor_feed(h, section.values)
    return h.hexdigest()


class _Instrumented(BatchedConditionalScorer):
    def _do_encode(self, td, rows, results):
        super()._do_encode(td, rows, results)
        path = os.environ.get("PA4_P3_ENCODE_PROBE_PATH")
        if not path:
            return
        try:
            from pa4.actions.batched_scorer import _ragged_from_td, identity_key

            for r in rows:
                key = identity_key(td, r)
                entry = self._cache.get(key)
                if entry is None:
                    continue
                with open(path, "w") as fh:
                    json.dump({
                        "identity": list(key),
                        "encoded": _encoded_digest(entry.encoded),
                        "ragged": _ragged_batch_digest(
                            _ragged_from_td(td, [r], self.device)),
                    }, fh)
                break
        except Exception:
            pass

    def forward(self, td):
        out = super().forward(td)
        path = os.environ.get("PA4_P3_STATS_PATH")
        if path:
            try:
                with open(path, "w") as fh:
                    json.dump({
                        "step_forward_count": self.model_forward_count,
                        "step_forward_batch_sizes": list(self.forward_batch_sizes),
                        "encode_forward_count": self.encode_forward_count,
                        "encode_batch_sizes": list(self.encode_batch_sizes),
                        "total_model_forward_count": (self.model_forward_count
                                                      + self.encode_forward_count),
                        "step_rows": self.step_rows,
                        "encode_calls": self.encode_calls,
                        "reset_calls": self.reset_calls,
                        "cache_size_now": len(self._cache),
                        "cache_high_water": self.cache_high_water,
                        "model_device": self.model_device,
                        "policy_version": self.policy_version,
                        "policy_weight_fingerprint": self.fingerprint,
                        "forward_run_ids": list(self.forward_run_ids),
                        "max_worker_ids_per_forward": max(
                            (len(set(s)) for s in self.forward_run_ids), default=0),
                    }, fh)
            except Exception:
                pass
        return out


def build_instrumented_service():
    seed = int(os.environ.get("PA4_POLICY_SEED", "20261006"))
    device = os.environ.get("PA4_POLICY_DEVICE", "cpu")
    pv = int(os.environ.get("PA4_POLICY_VERSION", "1"))
    # P4-C: serve the immutable published snapshot when configured, so the
    # server's identity is derived from the loaded weights rather than the seed.
    snapshot_path = os.environ.get("PA4_POLICY_SNAPSHOT") or None
    return _Instrumented(seed=seed, device=device, policy_version=pv,
                         snapshot_path=snapshot_path)
