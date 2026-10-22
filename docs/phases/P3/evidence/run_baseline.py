"""P3-D: regenerate the measured baseline with windows and stage timing.

Writes one JSON per configuration under docs/phases/P3/evidence/ and rewrites
PERFORMANCE_BASELINE.csv from the measured values. Every row records the commit
that produced it; there is no projection or extrapolation.

Usage:
    cd /home/pa4/projects/PokeAgent4.0
    uv run --frozen python docs/phases/P3/evidence/run_baseline.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)


def _repo_root(start):
    p = start
    while p != os.path.dirname(p):
        if os.path.exists(os.path.join(p, "pyproject.toml")):
            return p
        p = os.path.dirname(p)
    raise RuntimeError("repository root not found")


REPO = _repo_root(HERE)


def _sha():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short=7", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _save(name, payload):
    path = os.path.join(HERE, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return path


def main():
    sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
    sys.path.insert(0, REPO)
    from pa4.collector.batched_core import BatchedConfig
    from pa4.collector.multi_driver import run_multi
    from pa4.collector.run_batched import run_batched

    sha = _sha()
    rows = []

    singles = [(16, "bl_16_1", 32), (32, "bl_32_1", 32), (64, "bl_64_16", 32)]
    for num_envs, run_id, target in singles:
        cfg = BatchedConfig(
            num_envs=num_envs, target_matches=target, max_steps=20000,
            seed=1, run_id=run_id, min_batch_size=16, server_timeout=0.002,
            warmup_steps=8, transition_capacity=256,
        )
        rep = run_batched(cfg)
        d = rep.as_dict()
        _save(f"baseline_{run_id}", d)
        rows.append({
            "run_id": run_id, "git_sha": sha, "mode": "single",
            "worker_count": 1, "envs_per_worker": num_envs,
            "total_envs": num_envs, "setup_s": round(d["setup_seconds"], 2),
            "warmup_s": round(d["warmup_seconds"], 2),
            "steady_s": round(d["steady_seconds"], 2),
            "matches": d["natural_matches"],
            "matches_per_s": round(d["raw_natural_matches_per_sec"] or 0, 2),
            "steady_matches": d["steady_matches"],
            "steady_matches_per_s": (round(d["steady_matches_per_sec"], 2)
                                     if d["steady_matches_per_sec"] else None),
            "env_transitions": d["env_transitions"],
            "env_transitions_per_s": round(d["env_transitions_per_sec"] or 0, 1),
            "forward_batch_max": max((d["service_stats"].get("step_forward_batch_sizes") or [0])
                                     or [0]),
            "forward_batch_mean": (round(sum(d["service_stats"]["step_forward_batch_sizes"])
                                         / len(d["service_stats"]["forward_batch_sizes"]), 2)
                                   if d["service_stats"].get("step_forward_batch_sizes") else None),
            "transitions_stored": d["transitions_stored"],
            "transitions_dropped": d["transitions_dropped"],
            "duplicate_matches": d["duplicate_match_ids"],
            "operational_errors": len(d["errors"]),
            "stage_seconds": d["stage_seconds"],
        })

    multi = [(2, "multi2x16"), (4, "multi4x16")]
    for num_drivers, base_id in multi:
      for central in (True, False):
        run_id = base_id + ("_central" if central else "_perdrv")
        out = run_multi(num_drivers=num_drivers, num_envs=16, target=32, seed=1,
                        central=central, warmup=2)
        _save(f"baseline_{run_id}", out)
        rows.append({
            "run_id": run_id, "git_sha": sha,
            "mode": "central" if central else "per_driver",
            "worker_count": num_drivers, "envs_per_worker": 16,
            "total_envs": num_drivers * 16,
            "setup_s": round(out["setup_seconds"], 2),
            "warmup_s": round(out["warmup_seconds"], 2),
            "steady_s": round(out["steady_seconds_sum"], 2),
            "steady_s_max": round(out["steady_seconds_max"], 2),
            "window_note": out["steady_definition"],
            "matches": out["matches"],
            "matches_per_s": out["matches_per_s"],
            "steady_matches": out["steady_matches"],
            "steady_matches_per_s": out["steady_matches_per_s"],
            "env_transitions": out["transitions"],
            "env_transitions_per_s": out["transitions_per_s"],
            "forward_batch_max": (max((out["service_stats"].get("step_forward_batch_sizes") or [0])
                                      or [0]) if out.get("service_stats") else None),
            "forward_batch_mean": (round(sum(out["service_stats"]["step_forward_batch_sizes"])
                                         / len(out["service_stats"]["step_forward_batch_sizes"]), 2)
                                   if out.get("service_stats", {}).get("step_forward_batch_sizes")
                                   else None),
            "transitions_stored": out["transitions_stored"],
            "transitions_dropped": out["transitions_dropped"],
            "duplicate_matches": out["dupes"],
            "operational_errors": len(out["errors"]),
            "cross_worker_forwards": out["cross_worker_forwards"],
            "max_worker_ids_per_forward": out["max_worker_ids_per_forward"],
        })

    fields = ["run_id", "git_sha", "mode", "worker_count", "envs_per_worker",
              "total_envs", "setup_s", "warmup_s", "steady_s", "steady_s_max",
              "window_note", "matches",
              "matches_per_s", "steady_matches", "steady_matches_per_s",
              "env_transitions", "env_transitions_per_s", "forward_batch_max",
              "forward_batch_mean", "transitions_stored", "transitions_dropped",
              "duplicate_matches", "operational_errors", "cross_worker_forwards",
              "max_worker_ids_per_forward"]
    with open(os.path.join(P3, "PERFORMANCE_BASELINE.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
