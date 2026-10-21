"""Picklable module-level factory for the batched scorer (spawn-safe)."""

from __future__ import annotations

import os

from pa4.actions.batched_scorer import BatchedConditionalScorer


def build_service():
    seed = int(os.environ.get("PA4_POLICY_SEED", "20261006"))
    device = os.environ.get("PA4_POLICY_DEVICE", "cpu")
    pv = int(os.environ.get("PA4_POLICY_VERSION", "1"))
    return BatchedConditionalScorer(seed=seed, device=device, policy_version=pv)
