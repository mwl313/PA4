"""P3: multi-driver collection with ONE central inference server.

`central=True` (default) creates a single `InferenceServer` in the parent using
`transport="distributed"`. Its transport is picklable, and each driver process
creates its own actor-side clients from it (`transport.client()`), so requests
from different worker processes can be coalesced into one CUDA forward. The
result reports how many forwards genuinely mixed worker namespaces.

`central=False` keeps the older per-driver-server mode for comparison.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time


def _bootstrap():
    for p in ("/home/pa4/projects/_pa3_ref", "/home/pa4/projects/PokeAgent4.0"):
        sys.path.insert(0, p)


def driver_main(idx, num_envs, target, seed, out_q, device, transport=None,
                clients_per_driver=0, central_stats_path=None, warmup=2):
    _bootstrap()
    os.environ["PA4_POLICY_DEVICE"] = device
    try:
        from pa4.collector.batched_core import BatchedConfig
        from pa4.collector.run_batched import run_batched

        clients = None
        if transport is not None:
            clients = [transport.client() for _ in range(clients_per_driver)]
            # The parent owns stats collection in central mode; avoid clobbering.
            os.environ.pop("PA4_P3_STATS_PATH", None)

        cfg = BatchedConfig(
            num_envs=num_envs, target_matches=target,
            max_steps=max(4000, target * 400), seed=seed + idx * 101,
            run_id=f"drv{idx}", worker_index=idx,
            min_batch_size=1, server_timeout=0.0,
            device=device, warmup_steps=warmup,
        )
        rep = run_batched(cfg, clients=clients)
        out_q.put(("ok", idx, rep.as_dict()))
    except Exception as exc:
        import traceback

        out_q.put(("err", idx, f"{exc!r}\n{traceback.format_exc()[-600:]}"))


def _make_central_server(num_clients, device, stats_path):
    from torchrl.modules.inference_server import InferenceServer

    from pa4.actions.batched_factory_instrumented import build_instrumented_service
    from pa4.actions.batched_scorer import request_spec, response_spec

    os.environ["PA4_POLICY_DEVICE"] = device
    os.environ["PA4_P3_STATS_PATH"] = stats_path
    try:
        os.remove(stats_path)
    except OSError:
        pass
    srv = InferenceServer(
        model=None, policy_factory=build_instrumented_service,
        transport="distributed", service_backend="process",
        request_spec=request_spec(), response_spec=response_spec(),
        num_clients=num_clients,
        max_batch_size=64, min_batch_size=2, timeout=0.02,
        transport_options={"host": "127.0.0.1", "port": 0,
                           "channel_name": f"pa4p3-{os.getpid()}"},
    )
    srv.start()
    return srv


def run_multi(num_drivers=4, num_envs=16, target=32, seed=1, device="cuda",
              central=True, warmup=2):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    srv = None
    stats_path = f"/tmp/pa4_p3_central_{os.getpid()}.json"
    clients_per_driver = max(num_envs * 2, 2)
    t0 = time.perf_counter()

    if central:
        srv = _make_central_server(num_drivers * clients_per_driver, device, stats_path)
        transport = srv.transport
    else:
        transport = None

    for i in range(num_drivers):
        p = ctx.Process(
            target=driver_main,
            args=(i, num_envs, target, seed, q, device, transport,
                  clients_per_driver, stats_path, warmup),
            daemon=False,
        )
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

    server_stats = {}
    service_stats = {}
    if srv is not None:
        try:
            server_stats = dict(srv.stats())
        except Exception:
            pass
        try:
            srv.shutdown()
        except Exception:
            pass
        try:
            with open(stats_path) as fh:
                service_stats = json.load(fh)
        except Exception:
            pass

    wall = time.perf_counter() - t0
    matches = sum(r["natural_matches"] for r in results)
    trans = sum(r["env_transitions"] for r in results)
    run_sets = service_stats.get("forward_run_ids", []) or []
    cross = [s for s in run_sets if len(set(s)) > 1]
    steady_matches = sum(r.get("steady_matches", 0) for r in results)
    steady_seconds = max((r.get("steady_seconds", 0.0) for r in results), default=0.0)
    return {
        "num_drivers": num_drivers,
        "num_envs_each": num_envs,
        "total_envs": num_drivers * num_envs,
        "central": bool(central),
        "matches": matches,
        "transitions": trans,
        "wall_s": round(wall, 2),
        "matches_per_s": round(matches / wall, 2) if wall else None,
        "transitions_per_s": round(trans / wall, 1) if wall else None,
        "steady_matches": steady_matches,
        "steady_matches_per_s": (round(steady_matches / steady_seconds, 2)
                                 if steady_seconds else None),
        "dupes": sum(r["duplicate_match_ids"] for r in results),
        "errors": errors + [e for r in results for e in r["errors"]],
        "per_driver_mps": [round(r["natural_matches"] / max(r["wall_seconds"], 1e-9), 2)
                           for r in results],
        "transitions_stored": sum(r.get("transitions_stored", 0) for r in results),
        "transitions_dropped": sum(r.get("transitions_dropped", 0) for r in results),
        "cross_worker_forwards": len(cross),
        "max_worker_ids_per_forward": max((len(set(s)) for s in run_sets), default=0),
        "server_stats": server_stats,
        "service_stats": service_stats,
    }


if __name__ == "__main__":
    for k in (1, 2, 4):
        print(json.dumps(run_multi(k, 16, 32)), flush=True)
