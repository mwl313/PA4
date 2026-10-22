"""Picklable module-level factory for the batched scorer (spawn-safe)."""

from __future__ import annotations

import os

from pa4.actions.batched_scorer import BatchedConditionalScorer


def build_service():
    seed = int(os.environ.get("PA4_POLICY_SEED", "20261006"))
    device = os.environ.get("PA4_POLICY_DEVICE", "cpu")
    pv = int(os.environ.get("PA4_POLICY_VERSION", "1"))
    # P4-C: when a snapshot is configured, the served weights come from that
    # immutable artifact and the fingerprint is derived from the loaded tensors.
    # The P3 seed-only benchmark path remains available when this is unset.
    snapshot_path = os.environ.get("PA4_POLICY_SNAPSHOT") or None
    return BatchedConditionalScorer(seed=seed, device=device, policy_version=pv,
                                    snapshot_path=snapshot_path)
