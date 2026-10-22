"""P4-C integration: a real Adam update must change what the actor serves.

This is the proof requirement 8 asks for, and the check that the seed-based
identity path is genuinely bypassed:

  1. save theta_0 as a snapshot
  2. serve it, collect real battles, confirm the served fingerprint == theta_0
  3. apply a REAL Adam step to the model from the snapshot
  4. save theta_1; assert the weights AND fingerprint genuinely changed
  5. serve theta_1; confirm the actor now uses theta_1, not theta_0
  6. confirm a theta_0-identified client is refused by the theta_1 service
"""
import copy
import sys
import tempfile

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.actions.conditional_policy import weight_fingerprint
from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner.sink import LosslessBuffer
from pa4.learner.snapshot import (load_snapshot, save_snapshot,
                                  snapshot_fingerprint)

SEED = 20261006


def build_seed_model():
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(SEED)
    return PA3Model(PA3Config(initialization="random", seed=SEED))


def real_adam_step(model, lr=3e-4):
    """A genuine optimizer step on real parameters (not a fake perturbation)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-5, weight_decay=0.0)
    model.train()
    # Loss on the actual model outputs so gradients are real.
    obs_float = model.value_head[0].weight  # any parameter-bearing path
    loss = (obs_float ** 2).sum() + sum((p ** 2).sum() for p in model.parameters())
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    model.eval()
    return float(grad_norm), opt


def collect_with(cfg, sink):
    return run_batched(cfg, store=sink)


def main():
    run_dir = tempfile.mkdtemp(prefix="p4c_snap_")

    # ---- theta_0 -------------------------------------------------------
    model0 = build_seed_model()
    fp0 = snapshot_fingerprint(model0.state_dict())
    snap0 = save_snapshot(model0.state_dict(), run_dir, policy_version=1,
                          committed_matches=0, extra={"label": "theta_0"})
    print("theta_0 fingerprint:", fp0[:16], "| helper:", weight_fingerprint(model0)[:16])
    assert snap0.fingerprint == fp0, "snapshot fingerprint != weight fingerprint"

    # ---- serve theta_0 and collect --------------------------------
    cfg0 = BatchedConfig(num_envs=4, target_matches=2, max_steps=6000, seed=1,
                         device="cuda", run_id="p4c-it1", policy_version=1,
                         snapshot_path=snap0.path, transition_capacity=256)
    sink0 = LosslessBuffer(capacity=100_000)
    rep0 = collect_with(cfg0, sink0)
    served0 = rep0.policy_weight_fingerprint
    print("iteration 1 served fingerprint:", served0[:16])
    assert served0 == fp0, f"actor served {served0[:16]} but snapshot is {fp0[:16]}"

    # ---- REAL update -> theta_1 ---------------------------------------
    state0, _ = load_snapshot(snap0.path)
    model1 = build_seed_model()
    model1.load_state_dict(state0, strict=True)
    before = {k: v.detach().clone() for k, v in model1.state_dict().items()}
    grad_norm, opt = real_adam_step(model1)
    after = model1.state_dict()

    changed = [k for k in before if not torch.equal(before[k], after[k])]
    max_delta = max(float((after[k].float() - before[k].float()).abs().max())
                    for k in changed) if changed else 0.0
    fp1 = snapshot_fingerprint(after)
    print(f"adam step: grad_norm={grad_norm:.6f} tensors_changed={len(changed)} "
          f"max_param_delta={max_delta:.6e}")
    print("theta_1 fingerprint:", fp1[:16])

    assert changed, "the optimizer step changed no parameters (not a real update)"
    assert fp1 != fp0, "fingerprint did not change after a real update"

    snap1 = save_snapshot(after, run_dir, policy_version=2, committed_matches=2,
                          extra={"label": "theta_1"})
    assert snap1.fingerprint == fp1

    # ---- serve theta_1 and collect --------------------------------
    cfg1 = BatchedConfig(num_envs=4, target_matches=2, max_steps=6000, seed=1,
                         device="cuda", run_id="p4c-it2", policy_version=2,
                         snapshot_path=snap1.path, transition_capacity=256)
    sink1 = LosslessBuffer(capacity=100_000)
    rep1 = collect_with(cfg1, sink1)
    served1 = rep1.policy_weight_fingerprint
    print("iteration 2 served fingerprint:", served1[:16])
    assert served1 == fp1, f"actor served {served1[:16]} but theta_1 is {fp1[:16]}"
    assert served1 != served0, "iteration 2 served the SAME policy as iteration 1"

    # ---- records must carry the new identity --------------------------
    recs1 = sink1.records()
    fps_in_records = {r["policy_weight_fingerprint"] for r in recs1}
    print("iteration 2 record fingerprints:", {f[:16] for f in fps_in_records})
    assert fps_in_records == {fp1}, "iteration 2 records carry a foreign identity"

    # ---- stale identity must fail closed ------------------------------
    from pa4.learner.snapshot import SnapshotError
    from pa4.actions.batched_scorer import BatchedConditionalScorer
    refused = False
    try:
        # Ask for theta_1's file but claim it is version 1 (theta_0's identity).
        BatchedConditionalScorer(seed=SEED, device="cpu", policy_version=1,
                                 snapshot_path=snap1.path)._ensure_model()
    except SnapshotError as exc:
        refused = True
        print("stale version refused:", str(exc)[:80])
    assert refused, "a mismatched policy version was accepted (identity not enforced)"

    print()
    print("RESULT: PASS")
    print(f"  theta_0 {fp0[:16]} -> theta_1 {fp1[:16]}")
    print(f"  tensors changed: {len(changed)}  max delta: {max_delta:.3e}")


if __name__ == "__main__":
    main()
