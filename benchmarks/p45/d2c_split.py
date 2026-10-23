"""D2c: split ENCODE/STEP into submit vs result-wait, and measure pure GPU.

This is the decisive measurement for H3 (IPC/queue) versus GPU compute.

For each request we time:
  prep     — building the TensorDict payload (pure Python/tensor copies)
  submit   — clients[slot].submit(td) returning a future
  wait     — future.result() (the queue + server work + IPC return)

Then, using the SAME server, we time an in-process direct GPU encode of the same
observation to establish how long the actual kernel work takes. If wait >> gpu,
the difference is transport/queue, not compute.
"""
import json
import sys
import tempfile
import time

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torchrl.modules.inference_server import InferenceServer

from pa4.actions.batched_factory_instrumented import build_instrumented_service
from pa4.actions.batched_scorer import request_spec, response_spec
from pa4.collector.batched_core import BatchedConfig, StagedInference
from pa4.collector.run_batched import OBS_LEAVES
from pa4.env.native_env import PA4EnvBase
from pa4.learner.parity import build_observation
from pa4.learner.snapshot import save_snapshot


def build_model(seed=20261006):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def main():
    d = tempfile.mkdtemp(prefix="p45_split_")
    base = build_model()
    snap = save_snapshot(base.state_dict(), d, policy_version=1,
                         committed_matches=0)
    import os
    os.environ["PA4_POLICY_SNAPSHOT"] = snap.path
    os.environ["PA4_POLICY_VERSION"] = "1"

    n_clients = 8
    srv = InferenceServer(
        model=None, policy_factory=build_instrumented_service,
        transport="process_slot", service_backend="process",
        request_spec=request_spec(), response_spec=response_spec(),
        num_clients=n_clients, max_batch_size=64, min_batch_size=4, timeout=0.05,
    )
    srv.start()
    clients = srv.clients(n_clients)

    env = PA4EnvBase(batch_size=4, seed=1)
    env.reset()
    views = []
    for i in range(4):
        st = env._slots[i]
        v = env._observe_one(st, 0)
        views.append(v)

    inf = StagedInference(clients, request_spec(), policy_version=1,
                          run_id_int=0, fingerprint_id=0)
    # Resolve the real fingerprint so the server check passes.
    from pa4.actions.batched_scorer import fingerprint_id, model_fingerprint
    inf.fingerprint_id = fingerprint_id(model_fingerprint(20261006))

    # ---- time prep / submit / wait separately -------------------------
    prep_ns = submit_ns = wait_ns = 0
    rounds = 6
    for _ in range(rounds):
        futs = []
        for j, v in enumerate(views):
            t0 = time.perf_counter_ns()
            td = inf._row(0, (0, 0, 0, 0, j, 1000 + j))
            for k in OBS_LEAVES:
                td[k] = v[k]
            for k, val in (v.get("_ragged_row") or {}).items():
                td[k] = val
            prep_ns += time.perf_counter_ns() - t0

            slot = j % n_clients
            t1 = time.perf_counter_ns()
            fut = clients[slot].submit(td)
            submit_ns += time.perf_counter_ns() - t1
            futs.append(fut)
        for fut in futs:
            t2 = time.perf_counter_ns()
            fut.result(timeout=300)
            wait_ns += time.perf_counter_ns() - t2

    n_req = rounds * len(views)
    # ---- direct GPU encode of the same observations -------------------
    obs = build_observation_from(views)
    model = build_model().cuda().eval()
    with torch.no_grad():
        model.encode(obs)
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    reps = 20
    for _ in range(reps):
        with torch.no_grad():
            model.encode(obs)
    torch.cuda.synchronize()
    gpu_batch_ns = (time.perf_counter_ns() - t0) / reps

    srv.shutdown()

    out = {
        "requests": n_req,
        "batch_rows": len(views),
        "per_request_ms": {
            "prep": round(prep_ns / n_req / 1e6, 3),
            "submit": round(submit_ns / n_req / 1e6, 3),
            "wait": round(wait_ns / n_req / 1e6, 3),
        },
        "wait_to_prep_ratio": round((wait_ns / n_req) / (prep_ns / n_req), 2)
        if prep_ns else None,
        "direct_gpu_encode_batch_ms": round(gpu_batch_ns / 1e6, 3),
        "direct_gpu_encode_per_row_ms": round(gpu_batch_ns / 1e6 / len(views), 3),
        "note": ("prep+submit are CPU; wait covers queue dwell + server-side "
                 "decode + GPU + IPC return, so it is NOT pure GPU time"),
    }
    print(json.dumps(out, indent=1))


def build_observation_from(views):
    import numpy as np
    from agent.types.observation import ObservationBatch, ObservationLayout
    from pa4.learner.learner_eval import _stack_observations

    obs_list = []
    for v in views:
        layout = ObservationLayout()
        kw = {}
        for k in OBS_LEAVES:
            t = torch.as_tensor(v[k]).to("cuda")
            kw[k] = t
        obs_list.append(ObservationBatch(
            token_mask=kw["token_mask"].bool().unsqueeze(0),
            categories=kw["categories"].long().unsqueeze(0),
            category_known=kw["category_known"].bool().unsqueeze(0),
            floats=kw["floats"].float().unsqueeze(0),
            float_known=kw["float_known"].bool().unsqueeze(0),
            flags=kw["flags"].bool().unsqueeze(0),
            flag_known=kw["flag_known"].bool().unsqueeze(0),
            role_ids=kw["role_ids"].long().unsqueeze(0),
            side_ids=kw["side_ids"].long().unsqueeze(0),
            ragged=build_ragged(v),
        ).to("cuda"))
    from pa4.learner.learner_eval import _stack_observations as stack
    return stack(obs_list, "cuda")


def build_ragged(v):
    import numpy as np
    from agent.types.ragged import FIELDS, RaggedBatch
    row = v.get("_ragged_row") or {}
    out = {}
    for fam, fields in FIELDS.items():
        counts = np.asarray(torch.as_tensor(
            row[f"ragged.{fam}_counts"]).numpy(), dtype=np.int64)[None, :]
        vals = torch.as_tensor(row[f"ragged.{fam}_values"])
        packed = [vals[t, :int(counts[0, t])].numpy().astype(np.float32)
                  for t in range(counts.shape[-1]) if int(counts[0, t])]
        flat = np.concatenate(packed, 0) if packed else np.zeros((0, len(fields)),
                                                                np.float32)
        out[fam] = RaggedBatch.from_rows([flat], counts, len(fields)).to("cuda")
    return out


if __name__ == "__main__":
    main()
