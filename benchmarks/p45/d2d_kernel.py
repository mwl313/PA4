"""D2d: separate GPU kernel time from transport wait, definitively.

Uses CUDA events around an in-process encode at several batch sizes so we know
the real kernel cost, then compares against the measured transport wait.

If kernel(64 rows) << wait(64 requests), transport dominates by construction.
"""
import json
import sys
import tempfile
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
    env = PA4EnvBase(batch_size=8, seed=20261006)
    env.reset()
    views = []
    for i in range(8):
        st = env._slots[i]
        views.append(env._observe_one(st, 0))

    model = build_model().cuda().eval()
    obs_list = make_obs(views)

    out = {}
    print(f"{'batch':>6s} {'kernel_ms':>10s} {'per_row_ms':>11s} {'wall_ms':>9s}")
    for n in (1, 2, 4, 8):
        batch = _stack_observations(obs_list[:n], "cuda")
        with torch.no_grad():
            model.encode(batch)
        torch.cuda.synchronize()

        # CUDA-event timing (device-side, no launch overhead bias)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        reps = 20
        start.record()
        for _ in range(reps):
            with torch.no_grad():
                model.encode(batch)
        end.record()
        torch.cuda.synchronize()
        kernel_ms = start.elapsed_time(end) / reps

        # Wall timing (includes Python launch overhead)
        t0 = time.perf_counter()
        for _ in range(reps):
            with torch.no_grad():
                model.encode(batch)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) / reps * 1000

        out[n] = {"kernel_ms": round(kernel_ms, 3),
                  "per_row_ms": round(kernel_ms / n, 3),
                  "wall_ms": round(wall_ms, 3),
                  "launch_overhead_ms": round(wall_ms - kernel_ms, 3)}
        print(f"{n:6d} {kernel_ms:10.3f} {kernel_ms/n:11.3f} {wall_ms:9.3f}")

    print()
    print(json.dumps({
        "by_batch_size": out,
        "interpretation": {
            "kernel": "device-side encode duration via CUDA events",
            "per_row": "kernel_ms / batch, shows batching amortisation",
            "launch_overhead": "wall - kernel; Python/dispatch cost per call",
        },
    }, indent=1))


if __name__ == "__main__":
    main()
