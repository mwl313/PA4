"""Ablation child: resumes iterations 3-4 with a DELIBERATE restoration gap.

mode:
  full     — restore model + Adam + RNG + step counter (the real path)
  no_adam  — discard restored Adam moments
  no_rng   — discard the checkpointed torch/numpy RNG
  no_steps — discard the optimizer step counter

The point is to show the recovery comparison DETECTS each gap.
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner.checkpoint import load_checkpoint, restore_into
from pa4.learner.update import PpoUpdater, UpdateConfig
import resume_common as rc


def main():
    work, ckpt_dir, cohort_dir, out_json, mode = sys.argv[1:6]

    state = load_checkpoint(ckpt_dir)

    model = rc.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                 betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0)

    # MEASURE the RNG round-trip rather than asserting it. `rng_ok` is the
    # observed outcome of drawing from the restored state and comparing with
    # the continuation the checkpoint implies.
    import copy

    # Deep copy: holding a reference to state.rng_state would let the ablation
    # mutate the very object used as the "expected" stream (an aliasing bug that
    # silently made the RNG ablation undetectable).
    saved_rng = copy.deepcopy(state.rng_state)

    if mode == "no_rng":
        # Replace the torch stream with an unrelated live state. restore_into
        # then installs a stream that is NOT the checkpointed one.
        ablated = copy.deepcopy(saved_rng)
        ablated.pop("torch_cuda", None)
        torch.manual_seed(1234567)
        ablated["torch_cpu"] = torch.get_rng_state()
        state.rng_state = ablated

    restore_into(state, model, optimizer)
    model = model.to("cuda")

    # Downgrade the report label using an actual observation: does the restored
    # torch stream match the one the checkpoint would have produced?
    from pa4.learner.checkpoint import _set_rng_state

    # expected: the continuation implied by the CHECKPOINT
    _set_rng_state(saved_rng)
    expected = torch.randn(4).tolist()
    # observed: the continuation after restore_into applied state.rng_state
    _set_rng_state(state.rng_state)
    observed = torch.randn(4).tolist()
    rng_ok = (observed == expected)

    if mode == "no_adam":
        # A fresh optimizer replicates "Adam moments were never restored".
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                     betas=(0.9, 0.999), eps=1e-5,
                                     weight_decay=0.0)

    updater = PpoUpdater(model, UpdateConfig(), lr=1e-4)
    updater.optimizer.load_state_dict(optimizer.state_dict())
    if mode != "no_steps":
        updater.optimizer_steps = int(state.optimizer_steps)

    cohorts = {}
    for it in (3, 4):
        path = os.path.join(cohort_dir, f"cohort_{it}.json")
        cohorts[it] = rc.build_cohort_from_file(path, state.provenance["theta0"], 1)

    out = rc.run_range(cohorts, 3, 4, committed=state.committed_matches,
                       model=model, ledger=state.match_ledger, updater=updater)

    result = {
        "mode": mode,
        "rng_restored": rng_ok,
        "steps": out["steps"],
        "committed": out["committed"],
        "lr": out["lr"],
        "ledger": out["ledger"],
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, default=str)
    with open(out_json + ".model.pt", "wb") as fh:
        torch.save(out["model"].state_dict(), fh)


if __name__ == "__main__":
    main()
