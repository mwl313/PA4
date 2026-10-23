"""D3: uninstrumented baseline trials (the only legitimate throughput source).

Runs the fixed P4-equivalent profile N times with profiling OFF, recording
per-stage times and variance. Compare against the instrumented run to quantify
profiler overhead separately from throughput.
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

import torch


def run_trial(idx, iters=4, matches=16):
    # Ensure instrumentation is OFF for this trial.
    os.environ.pop("PA4_P45_PROFILE", None)
    os.environ.pop("PA4_P45_OUT", None)
    from pa4.learner.train_sync import SynchronousTrainer, TrainConfig

    run_dir = tempfile.mkdtemp(prefix=f"p45_base{idx}_")
    cfg = TrainConfig(num_envs=4, matches_per_iteration=matches,
                      iterations=iters, device="cuda", run_dir=run_dir,
                      microbatch_rows=8, effective_minibatch_rows=64,
                      ppo_epochs=2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer = SynchronousTrainer(cfg)
    recs = trainer.run()
    wall = time.perf_counter() - t0

    accepted = sum(r.accepted_matches for r in recs)
    return {
        "trial": idx,
        "instrumented": False,
        "iterations": len(recs),
        "wall_s": round(wall, 3),
        "accepted_matches": accepted,
        "accepted_matches_per_sec": round(accepted / wall, 4),
        "collection_s": round(sum(r.collection_seconds for r in recs), 3),
        "update_s": round(sum(r.update_seconds for r in recs), 3),
        "checkpoint_s": round(sum(r.checkpoint_seconds for r in recs), 3),
        "dropped": sum(r.dropped for r in recs),
        "duplicates": sum(r.duplicates for r in recs),
        "optimizer_steps": sum(r.optimizer_steps for r in recs),
        "actor_rows": sum(r.actor_rows for r in recs),
        "value_rows": sum(r.value_rows for r in recs),
        "early_stopped": sum(1 for r in recs if r.early_stopped),
        "params_changed_min": min((r.params_changed for r in recs), default=0),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if torch.cuda.is_available() else None,
        "fingerprints_chain_ok": all(
            recs[i].theta_next_fingerprint == recs[i + 1].theta_k_fingerprint
            for i in range(len(recs) - 1)),
        "distinct_policy_versions": len({r.theta_k_fingerprint for r in recs}),
    }


def main():
    n = int(os.environ.get("P45_TRIALS", "3"))
    out = [run_trial(i + 1) for i in range(n)]

    rates = sorted(t["accepted_matches_per_sec"] for t in out)
    med = rates[len(rates) // 2]
    summary = {
        "trials": out,
        "median_matches_per_sec": round(med, 4),
        "min": rates[0],
        "max": rates[-1],
        "spread_pct": round((rates[-1] - rates[0]) / med * 100, 1),
        "all_zero_drops": all(t["dropped"] == 0 for t in out),
        "all_zero_duplicates": all(t["duplicates"] == 0 for t in out),
        "all_chain_ok": all(t["fingerprints_chain_ok"] for t in out),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
