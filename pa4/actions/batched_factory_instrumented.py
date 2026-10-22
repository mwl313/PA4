"""Module-level instrumented factory for P3 batching evidence (spawn-safe).

Must be importable by the child process, so the factory cannot be a closure.
Instrumentation is written to PA4_P3_STATS_PATH after every forward.
"""

from __future__ import annotations

import json
import os

from pa4.actions.batched_scorer import BatchedConditionalScorer


class _Instrumented(BatchedConditionalScorer):
    def forward(self, td):
        out = super().forward(td)
        path = os.environ.get("PA4_P3_STATS_PATH")
        if path:
            try:
                with open(path, "w") as fh:
                    json.dump({
                        "model_forward_count": self.model_forward_count,
                        "forward_batch_sizes": list(self.forward_batch_sizes),
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
    return _Instrumented(seed=seed, device=device, policy_version=pv)
