"""Scale test: how does throughput grow with N envs?"""

import json
import sys
import time

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched


def run(n, target, seed=1):
    cfg = BatchedConfig(num_envs=n, target_matches=target, max_steps=target * 400,
                        seed=seed, run_id=f"s{n}",
                        min_batch_size=1, server_timeout=0.0)
    t0 = time.perf_counter()
    rep = run_batched(cfg)
    w = time.perf_counter() - t0
    d = rep.as_dict()
    svc = d.get("service_stats") or {}
    sizes = svc.get("forward_batch_sizes") or []
    return {
        "n": n, "matches": d["natural_matches"], "wall_s": round(w, 2),
        "matches_per_s": round(d["natural_matches"] / w, 2) if w else None,
        "transitions": d["env_transitions"],
        "transitions_per_s": round(d["env_transitions"] / w, 1) if w else None,
        "fwd": svc.get("model_forward_count"),
        "bmean": round(sum(sizes) / len(sizes), 2) if sizes else None,
        "bmax": max(sizes) if sizes else None,
        "errors": len(d["errors"]), "dupes": d["duplicate_match_ids"],
    }


if __name__ == "__main__":
    for n, tgt in ((16, 32), (32, 64), (64, 64)):
        print(json.dumps(run(n, tgt)))
