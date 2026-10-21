"""Tune min_batch_size / timeout: latency vs batch width."""

import json
import sys
import time

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched


def run(n, mb, to, target, seed=1):
    cfg = BatchedConfig(num_envs=n, target_matches=target, max_steps=target * 400,
                        seed=seed, run_id=f"t{mb}_{int(to*1000)}",
                        min_batch_size=mb, server_timeout=to)
    t0 = time.perf_counter()
    rep = run_batched(cfg)
    w = time.perf_counter() - t0
    d = rep.as_dict()
    svc = d.get("service_stats") or {}
    sizes = svc.get("forward_batch_sizes") or []
    return {
        "n": n, "min_batch": mb, "timeout": to,
        "matches": d["natural_matches"],
        "wall_s": round(w, 2),
        "matches_per_s": round(d["natural_matches"] / w, 2) if w else None,
        "fwd": svc.get("model_forward_count"),
        "bmean": round(sum(sizes) / len(sizes), 2) if sizes else None,
        "bmax": max(sizes) if sizes else None,
        "errors": len(d["errors"]),
    }


if __name__ == "__main__":
    for (mb, to) in ((8, 0.02), (4, 0.005), (2, 0.002), (1, 0.0)):
        r = run(16, mb, to, 32)
        print(json.dumps(r))
