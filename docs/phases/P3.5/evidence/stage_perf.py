"""P3.5: record stage timing for the repaired collector.

This script is NOT an ON/OFF ablation. It records the repaired path only.
`payload_to_transport` runs inside `env.step`/`reset` (and therefore inside the
`env_step`/`reset` stages); the `views` stage only extracts leaves and runs the
identity gate. The end-to-end -26%/-23% headline came from a separate same-host
git-archive A/B, not from this script.
"""
import json
import sys
import time

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched


def run(label, num_envs, target, seed):
    cfg = BatchedConfig(num_envs=num_envs, target_matches=target,
                        max_steps=200000, seed=seed, device="cuda",
                        run_id=f"p35-{label}")
    t0 = time.perf_counter()
    rep = run_batched(cfg)
    wall = time.perf_counter() - t0
    d = rep.as_dict()
    stage = getattr(rep, "stage_seconds", None) or {}
    return {
        "label": label,
        "num_envs": num_envs,
        "wall_s": round(wall, 3),
        "matches_per_s": round(d["natural_matches"] / wall, 3),
        "matches": d["natural_matches"],
        "stages_ms": {k: round(v * 1000 / max(1, d["env_steps"]), 2)
                      for k, v in sorted(stage.items())},
    }


def main():
    rows = []
    rows.append(run("repaired", 16, 48, 1))
    print(json.dumps(rows[-1], indent=1))


if __name__ == "__main__":
    main()
