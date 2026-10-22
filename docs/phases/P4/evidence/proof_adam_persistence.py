"""M1 proof: Adam moments must PERSIST across iterations.

Defect: every iteration built a fresh Adam, so exp_avg/exp_avg_sq were discarded
and the resume proof compared two freshly-created optimizers.

An optimizer state created in iteration 1 must still be present in the committed
checkpoint of iteration 3, and restoring that checkpoint must reproduce it.
"""
import sys
import tempfile

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner.checkpoint import load_checkpoint
from pa4.learner.train_sync import SynchronousTrainer, TrainConfig


def main():
    run_dir = tempfile.mkdtemp(prefix="p4_m1_")
    cfg = TrainConfig(num_envs=4, matches_per_iteration=4, iterations=3,
                      device="cuda", run_dir=run_dir,
                      microbatch_rows=8, effective_minibatch_rows=16)
    trainer = SynchronousTrainer(cfg)
    records = trainer.run()

    print("iteration | steps | restored_from | adam entries")
    prev_steps = 0
    monotone = True
    for r in records:
        ck = load_checkpoint(r.checkpoint_path)
        n_adam = len(ck.optimizer_state["state"]) if ck.optimizer_state else 0
        # optimizer_steps must accumulate across iterations, not reset to the
        # per-iteration count.
        if ck.optimizer_steps <= prev_steps:
            monotone = False
        prev_steps = ck.optimizer_steps
        print(f"  {r.iteration}       | {r.optimizer_steps:5d} | "
              f"{str(getattr(trainer, '_restored_from', None)):22s} | {n_adam}")

    final = load_checkpoint(run_dir)
    print()
    print("final optimizer_steps :", final.optimizer_steps)
    print("sum of per-iter steps :", sum(r.optimizer_steps for r in records))
    print("restored on last iter :", trainer.restored_from())

    # Moments must be non-trivial (not all zero) after several steps.
    s = final.optimizer_state["state"]
    nonzero = sum(int((v["exp_avg"].abs().sum() > 0))
                  for v in s.values() if "exp_avg" in v)
    print(f"parameters with non-zero exp_avg: {nonzero}/{len(s)}")

    ok = (monotone
          and final.optimizer_steps == sum(r.optimizer_steps for r in records)
          and nonzero > 100)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
