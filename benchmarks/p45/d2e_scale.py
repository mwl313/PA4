"""D2e: where does encode saturate? Batch 1..64, plus repeated-call cost.

If a batch of 1 costs the same as 64, the kernel is latency/overhead-bound and
BATCHING IS THE ENTIRE WIN. If it grows with batch, we are compute-bound.
"""
import json
import sys
import time

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pa4.env.native_env import PA4EnvBase
from pa4.learner.learner_eval import _stack_observations
from agent.types.observation import ObservationBatch


def build_model(seed=20261006):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def make_obs(views):
    from agent.types.ragged import FIELDS, RaggedBatch
    obs_list = []
    for v in views:
        kw = {k: torch.as_tensor(v[k]).to("cuda") for k in
              ("token_mask", "categories", "category_known", "floats",
               "float_known", "flags", "flag_known", "role_ids", "side_ids")}
        row = v.get("_ragged_row") or {}
        ragged = {}
        for fam, fields in FIELDS.items():
            counts = np.asarray(torch.as_tensor(
                row[f"ragged.{fam}_counts"]).numpy(), dtype=np.int64)[None, :]
            vals = torch.as_tensor(row[f"ragged.{fam}_values"])
            packed = [vals[t, :int(counts[0, t])].numpy().astype(np.float32)
                      for t in range(counts.shape[-1]) if int(counts[0, t])]
            flat = (np.concatenate(packed, 0) if packed
                    else np.zeros((0, len(fields)), np.float32))
            ragged[fam] = RaggedBatch.from_rows([flat], counts,
                                                len(fields)).to("cuda")
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
            ragged=ragged,
        ).to("cuda"))
    return obs_list


def main():
    # 64 distinct real observations.
    env = PA4EnvBase(batch_size=8, seed=20261006)
    env.reset()
    views = []
    for _ in range(2):  # 2 seats x 8 slots x several steps
        for i in range(8):
            for side in (0, 1):
                try:
                    views.append(env._observe_one(env._slots[i], side))
                except Exception:
                    pass
        import torch as t
        env.step(env.legal_action_packet(t.Generator().manual_seed(1)))
    views = views[:64]
    obs_list = make_obs(views)
    model = build_model().cuda().eval()

    print(f"{'batch':>6s} {'kernel_ms':>10s} {'per_row_ms':>11s} {'scaling':>9s}")
    rows = []
    base = None
    for n in (1, 2, 4, 8, 16, 32, 64):
        batch = _stack_observations(obs_list[:n], "cuda")
        with torch.no_grad():
            model.encode(batch)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        reps = 20
        start.record()
        for _ in range(reps):
            with torch.no_grad():
                model.encode(batch)
        end.record()
        torch.cuda.synchronize()
        k = start.elapsed_time(end) / reps
        if base is None:
            base = k
        rows.append({"batch": n, "kernel_ms": round(k, 3),
                     "per_row_ms": round(k / n, 3),
                     "vs_batch1": round(k / base, 2)})
        print(f"{n:6d} {k:10.3f} {k/n:11.3f} {k/base:8.2f}x")

    print()
    print(json.dumps({
        "measurements": rows,
        "verdict": ("latency/overhead-bound: cost barely grows with batch"
                    if rows[-1]["kernel_ms"] < 3 * rows[0]["kernel_ms"]
                    else "compute-bound: cost scales with batch"),
    }, indent=1))


if __name__ == "__main__":
    main()
