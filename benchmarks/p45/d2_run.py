"""P4.5 instrumented diagnostic run (labelled: NOT a throughput claim).

Runs the real synchronous trainer for a bounded number of iterations with
PA4_P45_PROFILE=1, so every stage gets an exclusive timer. Throughput from this
run is reported only as an overhead ratio against the uninstrumented baseline.
"""
import json
import os
import sys
import tempfile
import time

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.learner.train_sync import SynchronousTrainer, TrainConfig


def main():
    iters = int(os.environ.get("P45_ITERS", "2"))
    matches = int(os.environ.get("P45_MATCHES", "16"))
    run_dir = os.environ.get("P45_RUN_DIR") or tempfile.mkdtemp(prefix="p45_prof_")
    os.makedirs(run_dir, exist_ok=True)
    prof_dir = os.path.join(run_dir, "profiles")
    os.makedirs(prof_dir, exist_ok=True)
    os.environ["PA4_P45_PROFILE"] = "1"
    os.environ["PA4_P45_OUT"] = prof_dir

    cfg = TrainConfig(num_envs=4, matches_per_iteration=matches,
                      iterations=iters, device="cuda", run_dir=run_dir,
                      microbatch_rows=8, effective_minibatch_rows=64,
                      ppo_epochs=2)

    t0 = time.perf_counter()
    trainer = SynchronousTrainer(cfg)
    recs = trainer.run()
    wall = time.perf_counter() - t0

    print(json.dumps({
        "run_dir": run_dir,
        "instrumented": True,
        "iterations": len(recs),
        "wall_seconds": round(wall, 3),
        "accepted_matches": sum(r.accepted_matches for r in recs),
        "accepted_matches_per_sec": round(
            sum(r.accepted_matches for r in recs) / wall, 4),
        "collection_s": round(sum(r.collection_seconds for r in recs), 3),
        "update_s": round(sum(r.update_seconds for r in recs), 3),
        "checkpoint_s": round(sum(r.checkpoint_seconds for r in recs), 3),
        "dropped": sum(r.dropped for r in recs),
        "optimizer_steps": sum(r.optimizer_steps for r in recs),
        "profiles": sorted(os.listdir(prof_dir)),
    }, indent=1))


if __name__ == "__main__":
    main()
