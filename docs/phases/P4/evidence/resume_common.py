"""P4-D — shared helpers for the fresh-process recovery test.

Used by BOTH the parent (continuous baseline, interrupted iterations 1-2) and the
child (a genuinely new interpreter that resumes iterations 3-4 from disk).
"""
from __future__ import annotations

import json
import os
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import gae, ppo_loss
from pa4.learner.ingest import build_cohort
from pa4.learner.learner_eval import evaluate_stored_batch
from pa4.learner.parity import build_observation
from pa4.learner.snapshot import load_snapshot
from pa4.learner.update import PpoUpdater, UpdateConfig

SEED = 20261006
DEPTH = 4
ATOL = 1e-5          # predeclared in the P4 design
RTOL = 1e-4          # predeclared in the P4 design


def build_model(seed=SEED):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def encode(obj):
    """JSON encoder for tensors/arrays inside transition records."""
    if torch.is_tensor(obj):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj))


def build_cohort_from_file(path, fingerprint, policy_version):
    """Rebuild and VALIDATE a stored cohort against its own snapshot identity."""
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return build_cohort(records, expected_fingerprint=fingerprint,
                        expected_policy_version=policy_version)


def make_updater(model, lr):
    return PpoUpdater(model, UpdateConfig(), lr=lr)


def train_one(model, cohort, committed, iteration, optimizer=None, updater=None):
    """One deterministic PPO update on a PERSISTENT module.

    `model` is the single learner instance for the whole run: theta_k is loaded
    into it in place, so Adam's parameter-keyed moments stay valid and the step
    counter accumulates. Rebuilding either per iteration is the M1 defect this
    test exists to catch.
    """
    rows = cohort.value_rows()

    # V_old from the frozen copy of the CURRENT weights.
    frozen = build_model().to("cuda")
    frozen.load_state_dict(model.state_dict(), strict=True)
    frozen.eval()
    values = {}
    with torch.no_grad():
        for row in rows:
            obs = build_observation(row, device="cuda")
            values[row.identity()] = float(frozen.value(frozen.encode(obs)).item())
    del frozen

    order, adv, ret, actor_mask = gae.gae_for_cohort(cohort, values)
    value_mask = torch.ones(len(order), dtype=torch.bool)
    adv_norm = gae.normalize_advantages(adv, actor_mask)
    lr = ppo_loss.learning_rate(committed)

    if updater is None:
        updater = make_updater(model, lr)
        if optimizer is not None:
            updater.optimizer.load_state_dict(optimizer)
    updater.set_lr(lr)
    model.train()

    batch = {
        "rows": order,
        "old_lp": torch.tensor([r.joint_logprob_fp32 for r in order],
                               dtype=torch.float32, device="cuda"),
        "adv": adv_norm.cuda(),
        "ret": ret.cuda(),
        "actor_mask": actor_mask.cuda(),
        "value_mask": value_mask.cuda(),
        "generator": torch.Generator().manual_seed(SEED + iteration),
    }
    updater.run(batch, lambda sub: evaluate_stored_batch(model, sub, device="cuda"))
    return model, updater.optimizer.state_dict(), lr, updater


def run_range(cohorts, start, end, committed, model=None, optimizer_state=None,
              ledger=None, updater=None):
    """Run iterations [start..end] on ONE persistent model+optimizer."""
    if model is None:
        model = build_model().to("cuda")
    ledger = list(ledger or [])
    lr, steps = 0.0, 0

    if updater is None and optimizer_state is not None:
        updater = make_updater(model, ppo_loss.learning_rate(committed))
        updater.optimizer.load_state_dict(optimizer_state)

    for it in range(start, end + 1):
        cohort = cohorts[it]
        model, optimizer_state, lr, updater = train_one(
            model, cohort, committed, it, updater=updater)
        steps = updater.optimizer_steps
        committed += cohort.matches()
        ledger += [str(t.match_id) for t in cohort.trajectories]

    return {"model": model, "optimizer": optimizer_state, "lr": lr,
            "steps": steps, "committed": committed, "ledger": ledger,
            "updater": updater}


def check_rng_roundtrip(state):
    """Prove the checkpointed RNG actually resumes the saved stream.

    This is what makes the test fail if RNG restoration is removed.
    """
    import random

    import numpy as np

    from pa4.learner.checkpoint import _set_rng_state

    saved = state.rng_state
    # Draw from the saved state to learn the expected continuation.
    _set_rng_state(saved)
    expected_torch = torch.randn(4).tolist()
    expected_py = [random.random() for _ in range(3)]
    expected_np = np.random.rand(3).tolist()

    # Disturb everything, then restore from the checkpoint blob only.
    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)

    _set_rng_state(saved)
    got_torch = torch.randn(4).tolist()
    got_py = [random.random() for _ in range(3)]
    got_np = np.random.rand(3).tolist()

    return (got_torch == expected_torch and got_py == expected_py
            and np.allclose(got_np, expected_np))
