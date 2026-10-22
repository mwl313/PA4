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

    rng_ok = True
    save_rng = state.rng_state
    if mode == "no_rng":
        # Keep only python RNG; drop the torch/numpy streams the test asserts on.
        state.rng_state = {"python": save_rng["python"],
                           "torch_cpu": torch.get_rng_state(),
                           "numpy": save_rng["numpy"]}
        rng_ok = False

    restore_into(state, model, optimizer)
    model = model.to("cuda")

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
