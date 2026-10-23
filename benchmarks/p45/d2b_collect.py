"""D2b: decompose collection using the EXISTING stage timers in run_batched.

run_batched already records request_info / views / encode / step_levels /
packet / env_step / accounting / reset. Those are a subset of the design's D2
collector list; this script runs a real bounded collection and reports them,
then notes which required sub-areas are still unmeasured (server startup,
snapshot load, per-request submit/queue/result split, transition materialisation).
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

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner.sink import LosslessBuffer
from pa4.learner.snapshot import save_snapshot


def build_model(seed=20261006):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    import torch
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def main():
    d = tempfile.mkdtemp(prefix="p45_coll_")
    base = build_model()
    snap = save_snapshot(base.state_dict(), d, policy_version=1, committed_matches=0)

    cfg = BatchedConfig(num_envs=4, target_matches=16, max_steps=20000, seed=1,
                        device="cuda", run_id="p45-coll", policy_version=1,
                        snapshot_path=snap.path, transition_capacity=500_000)

    sink = LosslessBuffer(capacity=500_000)
    t0 = time.perf_counter()
    rep = run_batched(cfg, store=sink)
    wall = time.perf_counter() - t0

    stage = getattr(rep, "stage_seconds", {}) or {}
    total_stage = sum(stage.values())
    # run_batched's own setup/teardown is whatever wall minus the accounted stages.
    out = {
        "wall_seconds": round(wall, 3),
        "natural_matches": rep.as_dict()["natural_matches"],
        "env_steps": rep.as_dict()["env_steps"],
        "seat_records": rep.as_dict()["seat_records"],
        "transition_records": len(sink.records()),
        "stages_ms": {k: round(v * 1000, 1) for k, v in sorted(stage.items())},
        "stage_total_ms": round(total_stage * 1000, 1),
        "unaccounted_ms": round((wall - total_stage) * 1000, 1),
        "stage_share_pct": {k: round(v / total_stage * 100, 1)
                            for k, v in sorted(stage.items())} if total_stage else {},
        "per_env_step_ms": {k: round(v * 1000 / max(1, rep.as_dict()["env_steps"]), 2)
                            for k, v in sorted(stage.items())},
        "server_timeout": cfg.server_timeout,
        "min_batch_size": cfg.min_batch_size,
        "unmeasured_by_existing_timers": [
            "inference service startup and snapshot load",
            "per-request submit vs queue vs result-wait split",
            "transition materialisation inside 'accounting'",
            "server shutdown",
            "per-forward JSON stats write (PA4_P3_STATS_PATH)",
        ],
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
