"""P3: multi-driver scaling — K independent driver processes, each with its own
InferenceServer and PA4EnvBase(N).

Rationale from measurement: a single driver is limited by TorchRL
ProcessSlotTransport round-trip overhead (~40 ms/forward) rather than GPU compute
(encode 7 ms, step 2.5 ms). Adding driver processes multiplies throughput without
touching the model, the transport or any correctness invariant.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time


def driver_main(idx, num_envs, target, seed, out_q, device):
    sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
    sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")
    os.environ["PA4_POLICY_DEVICE"] = device
    try:
        from pa4.collector.batched_core import BatchedConfig
        from pa4.collector.run_batched import run_batched

        cfg = BatchedConfig(
            num_envs=num_envs, target_matches=target,
            max_steps=max(4000, target * 400), seed=seed + idx * 101,
            run_id=f"drv{idx}", worker_index=idx,
            min_batch_size=1, server_timeout=0.0,
            device=device,
        )
        rep = run_batched(cfg)
        out_q.put(("ok", idx, rep.as_dict()))
    except Exception as exc:
        import traceback

        out_q.put(("err", idx, f"{exc!r}\n{traceback.format_exc()[-600:]}"))


def run_multi(num_drivers=4, num_envs=16, target=32, seed=1, device="cuda"):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    t0 = time.perf_counter()
    for i in range(num_drivers):
        p = ctx.Process(target=driver_main,
                        args=(i, num_envs, target, seed, q, device), daemon=False)
        p.start()
        procs.append(p)

    results, errors = [], []
    for _ in range(num_drivers):
        try:
            kind, idx, payload = q.get(timeout=1800)
        except Exception as exc:
            errors.append(f"timeout: {exc!r}")
            break
        if kind == "ok":
            results.append(payload)
        else:
            errors.append(f"driver{idx}: {payload}")

    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    wall = time.perf_counter() - t0
    matches = sum(r["natural_matches"] for r in results)
    trans = sum(r["env_transitions"] for r in results)
    return {
        "num_drivers": num_drivers,
        "num_envs_each": num_envs,
        "total_envs": num_drivers * num_envs,
        "matches": matches,
        "transitions": trans,
        "wall_s": round(wall, 2),
        "matches_per_s": round(matches / wall, 2) if wall else None,
        "transitions_per_s": round(trans / wall, 1) if wall else None,
        "dupes": sum(r["duplicate_match_ids"] for r in results),
        "errors": errors + [e for r in results for e in r["errors"]],
        "per_driver_mps": [round(r["natural_matches"] / max(r["wall_seconds"], 1e-9), 2)
                           for r in results],
    }


if __name__ == "__main__":
    for k in (1, 2, 4):
        print(json.dumps(run_multi(k, 16, 32)), flush=True)
