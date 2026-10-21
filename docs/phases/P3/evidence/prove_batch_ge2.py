"""P3-B definitive proof: capture the service's own forward-batch histogram.

The transport stats show `batches: 3` for 4 requests, which is consistent with one
B=2 forward. This probe proves it from the service side by having the service
write its instrumentation to a file the parent can read.
"""

import json
import os
import sys

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch
from torchrl.modules.inference_server import InferenceServer

from pa4.env.native_env import PA4EnvBase
from pa4.actions.batched_scorer import request_spec, response_spec
from pa4.actions.batched_factory_instrumented import build_instrumented_service

OBS_KEYS = ("token_mask", "categories", "category_known", "floats", "float_known",
            "flags", "flag_known", "role_ids", "side_ids")
STATS_PATH = "/tmp/pa4_p3_batch_stats.json"


def row(spec, op, *, w, st, obs, level=0, cands=None):
    td = spec.clone()
    td["op"] = torch.tensor(op)
    td["worker_id"] = torch.tensor(w)
    td["native_slot"] = torch.tensor(st.slot)
    td["native_generation"] = torch.tensor(st.generation)
    td["episode_id"] = torch.tensor(st.episode_id)
    td["seat"] = torch.tensor(0)
    td["request_local_id"] = torch.tensor(w)
    td["policy_version"] = torch.tensor(1)
    td["branch_level"] = torch.tensor(level)
    td["request_key"] = torch.tensor(w)
    for k in OBS_KEYS:
        td[k] = obs[k]
    if cands is not None:
        from agent.types.observation import ObservationLayout
        from pa4.actions.conditional_policy import build_branch_table

        t = build_branch_table([cands], ObservationLayout())
        k = len(cands)
        td["candidate_count"] = torch.tensor(k)
        td["action_ids"][:k] = t.action_ids[0, 0, :k]
        td["legal_mask"][:k] = t.mask[0, 0, :k]
        td["entity_token"][:k] = t.entity_token[0, 0, :k]
        td["move_token"][:k] = t.move_token[0, 0, :k]
    return td


def enabled_service():
    """Factory that records its instrumentation at exit."""
    def build():
        svc = batched_factory.build_service()

        orig = svc.forward

        def wrapped(td):
            out = orig(td)
            try:
                with open(STATS_PATH, "w") as fh:
                    json.dump({
                        "model_forward_count": svc.model_forward_count,
                        "forward_batch_sizes": list(svc.forward_batch_sizes),
                        "step_rows": svc.step_rows,
                        "encode_calls": svc.encode_calls,
                        "cache_high_water": svc.cache_high_water,
                        "model_device": svc.model_device,
                    }, fh)
            except Exception:
                pass
            return out

        svc.forward = wrapped
        return svc

    return build


def main():
    os.environ["PA4_POLICY_DEVICE"] = "cuda"
    os.environ["PA4_P3_STATS_PATH"] = STATS_PATH
    try:
        os.remove(STATS_PATH)
    except OSError:
        pass

    envs = []
    for w in (0, 1):
        e = PA4EnvBase(batch_size=1, seed=1 + w)
        e.reset()
        st = e._slots[0]
        envs.append((e, st, e._observe_one(st, 0)))

    req_sp, res_sp = request_spec(), response_spec()
    srv = InferenceServer(
        model=None, policy_factory=build_instrumented_service, transport="process_slot",
        service_backend="process", request_spec=req_sp, response_spec=res_sp,
        num_clients=2, max_batch_size=8, min_batch_size=2, timeout=0.5,
    )
    srv.start()
    clients = srv.clients(2)

    for w, (e, st, obs) in enumerate(envs):
        clients[w].submit(row(req_sp, 0, w=w, st=st, obs=obs)).result()

    cands, futs = [], []
    for w, (e, st, obs) in enumerate(envs):
        c = e._candidates(st, 0, [])
        cands.append(c)
        futs.append(clients[w].submit(
            row(req_sp, 1, w=w, st=st, obs=obs, level=0, cands=c)))

    outs = [f.result() for f in futs]
    for w, o in enumerate(outs):
        idx = int(o["selected_index"])
        print(f"worker{w}: idx={idx} K={len(cands[w])} err={int(o['error_code'])}")
        assert 0 <= idx < len(cands[w])

    srv.shutdown()

    with open(STATS_PATH) as fh:
        stats = json.load(fh)
    print()
    print("=== SERVICE INSTRUMENTATION (authoritative) ===")
    print(json.dumps(stats, indent=2))

    sizes = stats["forward_batch_sizes"]
    assert any(s >= 2 for s in sizes), f"NO B>=2 FORWARD: {sizes}"
    assert "cuda" in stats["model_device"], stats["model_device"]
    print()
    print("G3_B_CHECKPOINT_PASS: real CUDA forward with batch >= 2 observed")


if __name__ == "__main__":
    main()
