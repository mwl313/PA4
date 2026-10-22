"""P4-D child: a GENUINELY NEW interpreter + CUDA context.

Launched by fresh_process_recovery.py via subprocess. It knows nothing about
the parent's Python state: it loads theta from the committed checkpoint on disk
and completes iterations 3-4.

If Adam restoration or RNG restoration were removed, this process would diverge
and the parent's comparison would fail.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner.checkpoint import load_checkpoint, restore_into
import resume_common as rc


def main():
    work, ckpt_dir, cohort_dir, out_json = sys.argv[1:5]

    pid = os.getpid()
    torch.cuda.init()
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"child pid={pid} device={dev}")

    state = load_checkpoint(ckpt_dir)
    print(f"child loaded checkpoint: iteration={state.iteration} "
          f"steps={state.optimizer_steps} matches={state.committed_matches}")

    # Restore model AND Adam into a real optimizer, then verify.
    model = rc.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                 betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0)
    restore_into(state, model, optimizer)
    model = model.to("cuda")

    adam_ids = sorted(optimizer.state_dict()["state"].keys())
    adam_nonzero = sum(
        1 for v in optimizer.state_dict()["state"].values()
        if "exp_avg" in v and float(v["exp_avg"].abs().sum()) > 0)
    print(f"child adam: {len(adam_ids)} params, {adam_nonzero} with nonzero exp_avg")

    # Prove the checkpointed RNG resumes the saved stream.
    rng_ok = rc.check_rng_roundtrip(state)
    print(f"child rng roundtrip ok={rng_ok}")

    # Rebuild iterations 3-4 from their stored records, bound to theta_0.
    cohorts = {}
    for it in (3, 4):
        path = os.path.join(cohort_dir, f"cohort_{it}.json")
        cohorts[it] = rc.build_cohort_from_file(path, state.provenance["theta0"],
                                                1)

    # Train on the SAME module the optimizer was restored into, so Adam's
    # parameter-keyed moments stay valid.
    from pa4.learner.update import PpoUpdater, UpdateConfig
    updater = PpoUpdater(model, UpdateConfig(), lr=1e-4)
    updater.optimizer.load_state_dict(optimizer.state_dict())
    # Continue the step ledger rather than restarting it at zero.
    updater.optimizer_steps = int(state.optimizer_steps)

    out = rc.run_range(cohorts, 3, 4, committed=state.committed_matches,
                       model=model,
                       ledger=state.match_ledger,
                       updater=updater)

    result = {
        "pid": pid,
        "device": dev,
        "rng_restored": rng_ok,
        "adam_params": len(adam_ids),
        "adam_nonzero": adam_nonzero,
        "steps": out["steps"],
        "committed": out["committed"],
        "lr": out["lr"],
        "ledger": out["ledger"],
        "cohort_match_ids": {str(it): [str(t.match_id) for t in cohorts[it].trajectories]
                             for it in (3, 4)},
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, default=str)
    with open(out_json + ".model.pt", "wb") as fh:
        torch.save(out["model"].state_dict(), fh)
    with open(out_json + ".adam.pt", "wb") as fh:
        torch.save(out["optimizer"], fh)
    print("child done")


if __name__ == "__main__":
    main()
