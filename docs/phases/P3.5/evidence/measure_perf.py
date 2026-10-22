"""P3.5: remeasure actor-only throughput after the observation repair."""
import json
import sys
import time

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched


def measure(num_envs, target, seed, run_id):
    cfg = BatchedConfig(
        num_envs=num_envs,
        target_matches=target,
        max_steps=200000,
        seed=seed,
        device="cuda",
        run_id=run_id,
    )
    t0 = time.perf_counter()
    rep = run_batched(cfg)
    wall = time.perf_counter() - t0
    d = rep.as_dict()
    return {
        "num_envs": num_envs,
        "target_matches": target,
        "wall_seconds": round(wall, 3),
        "natural_matches": d["natural_matches"],
        "raw_matches_per_sec": round(d["natural_matches"] / wall, 3),
        "env_transitions": d["env_transitions"],
        "env_transitions_per_sec": round(d["env_transitions"] / wall, 2),
        "seat_records": d["seat_records"],
        "drops": getattr(rep, "transitions_dropped", None),
        "duplicate_match_ids": d["duplicate_match_ids"],
    }


def main():
    out = []
    for num_envs, target, seed, rid in (
        (16, 64, 1, "p35-16"),
        (32, 64, 1, "p35-32"),
    ):
        row = measure(num_envs, target, seed, rid)
        out.append(row)
        print(json.dumps(row))
    print()
    print("JSON:")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
