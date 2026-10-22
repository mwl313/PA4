"""P4 smoke: two real synchronous iterations end to end."""
import json
import sys
import tempfile

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.learner.train_sync import SynchronousTrainer, TrainConfig


def main():
    cfg = TrainConfig(
        num_envs=4, matches_per_iteration=4, iterations=2,
        device="cuda", run_dir=tempfile.mkdtemp(prefix="p4_smoke_"),
    )
    trainer = SynchronousTrainer(cfg)
    records = trainer.run()
    for r in records:
        print(json.dumps({
            "iter": r.iteration,
            "theta_k": r.theta_k_fingerprint[:16],
            "theta_next": (r.theta_next_fingerprint or "")[:16],
            "generated": r.generated_matches,
            "accepted": r.accepted_matches,
            "incomplete": r.incomplete_episodes,
            "dropped": r.dropped,
            "actor_rows": r.actor_rows,
            "value_rows": r.value_rows,
            "steps": r.optimizer_steps,
            "epochs": r.epochs_run,
            "early_stop": r.early_stopped,
            "lr": r.lr,
            "approx_kl": round(r.approx_kl, 6),
            "grad_norm": round(r.grad_norm, 4),
            "params_changed": r.params_changed,
            "max_delta": f"{r.max_param_delta:.3e}",
            "wall_s": round(r.wall_seconds, 2),
            "errors": r.errors,
        }))
    ok = (len(records) == 2
          and all(r.dropped == 0 for r in records)
          and all(r.optimizer_steps > 0 for r in records)
          and all(r.params_changed > 0 for r in records)
          and records[0].theta_next_fingerprint == records[1].theta_k_fingerprint
          and records[0].theta_k_fingerprint != records[1].theta_k_fingerprint)
    print()
    print("chained identity:", records[0].theta_next_fingerprint
          == records[1].theta_k_fingerprint)
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
