"""P3-F orchestrator: concurrent parent servicing so worker requests coalesce.

Critical design point (P3-B/G3-B): if the parent handles one worker's request and
blocks until the reply, two workers can never be simultaneously "ready" and the
inference server therefore never batches (observed: 79 forwards, all B=1).

The parent must drain *all* pending worker messages each loop iteration, submit
every STEP to its transport client WITHOUT waiting, and only then collect the
results. Identical-level STEPs submitted in the same pass are what lets the
server coalesce them into one forward with B>=2.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass, field

import torch

from pa4.collector.collector_core import OBS_KEYS, _ensure_paths, worker_main


@dataclass
class CollectorConfig:
    num_workers: int = 2
    target_matches_per_worker: int = 8
    max_steps_per_worker: int = 4000
    seed: int = 0
    policy_version: int = 1
    device: str = "cuda"
    max_batch_size: int = 8
    min_batch_size: int = 2
    server_timeout: float = 0.5
    queue_max: int = 8192
    use_fast_obs: bool = True
    run_id: str = "pa4-p3"


@dataclass
class CollectorReport:
    run_id: str = ""
    num_workers: int = 0
    natural_matches: int = 0
    match_ids: list = field(default_factory=list)
    duplicate_match_ids: int = 0
    transitions: int = 0
    seat_records: int = 0
    actor_active_rows: int = 0
    errors: list = field(default_factory=list)
    revisions: int = 0
    truncations: int = 0
    illegal_actions: int = 0
    wall_seconds: float = 0.0
    per_worker_episodes: dict = field(default_factory=dict)
    server_stats: dict = field(default_factory=dict)
    service_stats: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(self.__dict__)


class _Bridge:
    def __init__(self, client, spec, policy_version):
        self.client = client
        self.spec = spec
        self.policy_version = policy_version

    def row(self, op, ident, level=0):
        td = self.spec.clone()
        td["op"] = torch.tensor(op)
        td["worker_id"] = torch.tensor(ident["worker_id"])
        td["native_slot"] = torch.tensor(ident["native_slot"])
        td["native_generation"] = torch.tensor(ident["native_generation"])
        td["episode_id"] = torch.tensor(ident["episode_id"])
        td["seat"] = torch.tensor(ident["seat"])
        td["request_local_id"] = torch.tensor(ident["request_local_id"])
        td["policy_version"] = torch.tensor(self.policy_version)
        td["request_key"] = torch.tensor(ident["request_local_id"] % (2**31 - 1))
        td["branch_level"] = torch.tensor(level)
        return td

    def submit_encode(self, ident, view):
        td = self.row(0, ident)
        for k in OBS_KEYS:
            td[k] = view[k]
        return self.client.submit(td)          # future, do not block

    def submit_step(self, ident, level, cands, layout, a2r):
        from pa4.actions.conditional_policy import build_branch_table

        table = build_branch_table([cands], layout, a2r)
        td = self.row(1, ident, level)
        k = len(cands)
        td["candidate_count"] = torch.tensor(k)
        td["action_ids"] = table.action_ids[0, 0]
        td["legal_mask"] = table.mask[0, 0]
        td["entity_token"] = table.entity_token[0, 0]
        td["move_token"] = table.move_token[0, 0]
        return self.client.submit(td)

    def submit_reset(self, ident):
        return self.client.submit(self.row(2, ident))


def run_collector(cfg: CollectorConfig) -> CollectorReport:
    _ensure_paths()
    from torchrl.modules.inference_server import InferenceServer

    from pa4.actions.batched_factory_instrumented import build_instrumented_service
    from pa4.actions.batched_scorer import request_spec, response_spec
    from agent.types.observation import ObservationBatch

    os.environ["PA4_POLICY_DEVICE"] = cfg.device
    stats_path = f"/tmp/pa4_p3_svc_{cfg.run_id}.json"
    os.environ["PA4_P3_STATS_PATH"] = stats_path
    try:
        os.remove(stats_path)
    except OSError:
        pass

    req_sp, res_sp = request_spec(), response_spec()
    srv = InferenceServer(
        model=None, policy_factory=build_instrumented_service,
        transport="process_slot", service_backend="process",
        request_spec=req_sp, response_spec=res_sp,
        num_clients=cfg.num_workers * 2,   # one slot per seat-request in flight
        max_batch_size=cfg.max_batch_size,
        min_batch_size=cfg.min_batch_size,
        timeout=cfg.server_timeout,
    )
    srv.start()

    ctx = mp.get_context("spawn")
    to_parent = ctx.Queue(maxsize=cfg.queue_max)
    report = CollectorReport(run_id=cfg.run_id, num_workers=cfg.num_workers)
    procs, pipes = [], {}
    t0 = time.perf_counter()

    def obs_of(view):
        return ObservationBatch(
            token_mask=view["token_mask"].unsqueeze(0),
            categories=view["categories"].unsqueeze(0),
            category_known=view["category_known"].unsqueeze(0),
            floats=view["floats"].unsqueeze(0),
            float_known=view["float_known"].unsqueeze(0),
            flags=view["flags"].unsqueeze(0),
            flag_known=view["flag_known"].unsqueeze(0),
            role_ids=view["role_ids"].unsqueeze(0),
            side_ids=view["side_ids"].unsqueeze(0),
        )

    try:
        for w in range(cfg.num_workers):
            q = ctx.Queue()
            p = ctx.Process(
                target=worker_main,
                args=(w, cfg.run_id, cfg.seed + w, cfg.policy_version,
                      cfg.max_steps_per_worker, to_parent, q,
                      cfg.target_matches_per_worker, cfg.use_fast_obs),
                daemon=True,
            )
            p.start()
            procs.append(p)
            pipes[w] = q

        # Each worker may have up to 2 in-flight seat requests (one per seat), so
        # allocate 2 clients per worker and hand each request a dedicated slot.
        clients = [srv.client() for _ in range(cfg.num_workers * 2)]
        bridges = [_Bridge(c, req_sp, cfg.policy_version) for c in clients]
        free_slots = list(range(len(clients)))
        slot_busy = set()
        cache = {}

        done, seen = set(), set()
        idle_deadline = time.perf_counter() + 600

        # Phase A: drain a whole batch of worker requests, submit all STEPs
        # without blocking, then collect. This is what enables B>=2.
        pending = []          # (wid, ident, level, cands, future)
        while len(done) < cfg.num_workers:
            if time.perf_counter() > idle_deadline:
                report.errors.append("collector idle timeout")
                break
            try:
                msg = to_parent.get(timeout=30)
            except Exception:
                continue
            kind = msg[0]

            if kind == "encode":
                _, wid, ident, view = msg
                try:
                    ob = obs_of(view)
                    cache[(wid, ident["request_local_id"])] = (
                        ob.layout, ob.active_to_roster())
                    slot = free_slots.pop()
                    bridges[slot].submit_encode(ident, view).result()
                    free_slots.append(slot)
                    free_slots.sort()
                    pipes[wid].put(("ok", 0))
                except Exception as exc:
                    pipes[wid].put(("error", repr(exc)))

            elif kind == "step":
                _, wid, ident, level, cands, view = msg
                batch = [(wid, ident, level, cands)]
                while True:
                    try:
                        nxt = to_parent.get_nowait()
                    except Exception:
                        break
                    if nxt[0] == "step":
                        batch.append((nxt[1], nxt[2], nxt[3], nxt[4]))
                    else:
                        to_parent.put(nxt)
                        break

                inflight = []
                for (w2, id2, lv2, c2) in batch:
                    try:
                        lay2, a22 = cache[(w2, id2["request_local_id"])]
                        slot = free_slots.pop()
                        fut = bridges[slot].submit_step(id2, lv2, c2, lay2, a22)
                        inflight.append((w2, slot, fut))
                    except Exception as exc:
                        pipes[w2].put(("error", repr(exc)))

                for w2, slot, fut in inflight:
                    try:
                        out = fut.result()
                        err = int(out["error_code"])
                        if err:
                            pipes[w2].put(("error", "step err " + str(err)))
                        else:
                            pipes[w2].put(("ok", int(out["selected_index"]),
                                           int(out["branch_k"]),
                                           float(out["branch_logprob"])))
                    except Exception as exc:
                        pipes[w2].put(("error", repr(exc)))
                    finally:
                        free_slots.append(slot)
                free_slots.sort()

            elif kind == "reset":
                _, wid, ident = msg
                try:
                    slot = free_slots.pop()
                    bridges[slot].submit_reset(ident).result()
                    free_slots.append(slot)
                    free_slots.sort()
                    cache.pop((wid, ident["request_local_id"]), None)
                    pipes[wid].put(("ok", 0))
                except Exception as exc:
                    pipes[wid].put(("error", repr(exc)))

            elif kind == "transition":
                rec = msg[1]
                report.transitions += 1
                report.seat_records += len(rec["seat_records"])
                report.actor_active_rows += sum(
                    1 for s in rec["seat_records"] if s["actor_active"])
                if rec["request_revised"]:
                    report.revisions += 1
                if rec["truncated"]:
                    report.truncations += 1
                if rec["match_id"] is not None:
                    mid = tuple(rec["match_id"])
                    if mid in seen:
                        report.duplicate_match_ids += 1
                    else:
                        seen.add(mid)
                        report.match_ids.append(list(mid))
                idle_deadline = time.perf_counter() + 600
            elif kind == "error":
                report.errors.append(f"worker{msg[1]}: {msg[2]}")
                done.add(msg[1])
            elif kind == "done":
                report.per_worker_episodes[str(msg[1])] = msg[2]
                done.add(msg[1])
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        try:
            report.server_stats = dict(srv.stats())
        except Exception:
            pass
        try:
            srv.shutdown()
        except Exception:
            pass
        report.wall_seconds = time.perf_counter() - t0

    report.natural_matches = len(report.match_ids)
    try:
        with open(stats_path) as fh:
            report.service_stats = json.load(fh)
    except Exception:
        pass
    return report
