"""Module-level picklable factory for the ConditionalScorerService.

Process-spawned InferenceServer children must be able to unpickle the policy
factory, so it cannot be a closure. Configuration travels via env vars that are
read inside the child when the service is constructed.
"""

from __future__ import annotations

import os

from pa4.actions.server_probe import ConditionalScorerService


def build_service():
    """Picklable entry point used by TorchRL's process InferenceServer."""
    seed = int(os.environ.get("PA4_POLICY_SEED", "20261006"))
    device = os.environ.get("PA4_POLICY_DEVICE", "cpu")
    policy_version = int(os.environ.get("PA4_POLICY_VERSION", "1"))
    return ConditionalScorerService(seed=seed, device=device, policy_version=policy_version)
