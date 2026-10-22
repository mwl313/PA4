"""P4-E: the full 20-iteration synchronous training trial on real CUDA.

Profile per the approved development config: 4 envs, 16 distinct natural
completions per iteration, 8-row microbatches, 64-row effective minibatches,
2 PPO epochs, 20 iterations, eager FP32.

Records every field the design requires per iteration and writes
FULL_STACK_BASELINE.csv plus a JSON summary.
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

from pa4.learner.train_sync import SynchronousTrainer, TrainConfig

FIELDS = [
    "iteration", "policy_version", "theta_k_fingerprint", "theta_next_fingerprint",
    "checkpoint_path", "checkpoint_sha256",
    "generated_matches", "accepted_matches", "incomplete_episodes",
    "dropped", "duplicates", "transition_records",
    "actor_rows", "value_rows", "optimizer_steps",
    "cumulative_optimizer_steps", "epochs_run", "early_stopped",
    "lr", "approx_kl", "clip_fraction", "grad_norm",
    "loss_policy", "loss_value", "entropy", "uniform_kl",
    "committed_matches", "params_changed", "max_param_delta", "nan_inf",
    "collection_seconds", "update_seconds", "checkpoint_seconds", "wall_seconds",
]


def main():
    iterations = int(os.environ.get("P4_ITERATIONS", "20"))
    matches = int(os.environ.get("P4_MATCHES", "16"))
    run_dir = os.environ.get("P4_RUN_DIR") or tempfile.mkdtemp(prefix="p4e_run_")

    cfg = TrainConfig(
        num_envs=4, matches_per_iteration=matches, iterations=iterations,
        device="cuda", run_dir=run_dir,
        microbatch_rows=8, effective_minibatch_rows=64, ppo_epochs=2,
    )
    print(f"run_dir={run_dir}")
    print(f"iterations={iterations} matches_per_iteration={matches}")

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = SynchronousTrainer(cfg)
    records = trainer.run()
    total_wall = time.perf_counter() - t0

    peak_vram = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                 if torch.cuda.is_available() else 0.0)

    rows = []
    for r in records:
        d = {f: getattr(r, f) for f in FIELDS}
        rows.append(d)

    total_accepted = sum(r.accepted_matches for r in records)
    total_generated = sum(r.generated_matches for r in records)
    total_actor = sum(r.actor_rows for r in records)
    total_steps = sum(r.optimizer_steps for r in records)
    final_cumulative = records[-1].cumulative_optimizer_steps if records else 0
    total_row_passes = sum(r.actor_rows * max(1, r.epochs_run) for r in records)

    summary = {
        "iterations": len(records),
        "matches_per_iteration_target": matches,
        "total_wall_seconds": round(total_wall, 3),
        "generated_natural_matches": total_generated,
        "accepted_unique_matches": total_accepted,
        "policy_rows": total_actor,
        "optimizer_steps_per_iteration_sum": total_steps,
        "optimizer_steps_cumulative_final": final_cumulative,
        "peak_vram_mib": round(peak_vram, 1),
        "raw_completed_matches_per_sec": round(total_generated / total_wall, 4),
        "accepted_training_matches_per_sec": round(total_accepted / total_wall, 4),
        "policy_rows_per_sec": round(total_actor / total_wall, 3),
        "learner_row_passes_per_sec": round(total_row_passes / total_wall, 3),
        "optimizer_steps_per_sec": round(total_steps / total_wall, 4),
        "dropped_total": sum(r.dropped for r in records),
        "duplicates_total": sum(r.duplicates for r in records),
        "incomplete_total": sum(r.incomplete_episodes for r in records),
        "nan_inf_total": sum(r.nan_inf for r in records),
        "iterations_with_zero_steps": sum(1 for r in records if r.optimizer_steps == 0),
        "iterations_without_param_change": sum(1 for r in records if r.params_changed == 0),
        "early_stopped_iterations": sum(1 for r in records if r.early_stopped),
        "fingerprint_chain_ok": all(
            records[i].theta_next_fingerprint == records[i + 1].theta_k_fingerprint
            for i in range(len(records) - 1)),
        "distinct_policy_versions": len({r.theta_k_fingerprint for r in records}),
        "run_dir": run_dir,
    }

    print()
    print(json.dumps(summary, indent=1))

    with open("docs/phases/P4/evidence/full_run.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "iterations": rows}, fh, indent=1,
                  default=str)

    import csv
    with open("docs/phases/P4/FULL_STACK_BASELINE.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for d in rows:
            w.writerow({k: d[k] for k in FIELDS})

    ok = (len(records) == iterations
          and summary["dropped_total"] == 0
          and summary["duplicates_total"] == 0
          and summary["iterations_with_zero_steps"] == 0
          and summary["iterations_without_param_change"] == 0
          and summary["nan_inf_total"] == 0
          and summary["fingerprint_chain_ok"])
    print()
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
