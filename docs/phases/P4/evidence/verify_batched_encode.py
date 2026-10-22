"""Does batched encode change any number? It must not."""
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner.ingest import build_cohort
from pa4.learner.learner_eval import evaluate_stored_batch
from pa4.learner.sink import LosslessBuffer


def build_model(seed=20261006):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed)).cuda()


def main():
    cfg = BatchedConfig(num_envs=4, target_matches=2, max_steps=6000, seed=1,
                        device="cuda", run_id="batchcheck", transition_capacity=256)
    sink = LosslessBuffer(capacity=100_000)
    run_batched(cfg, store=sink)
    cohort = build_cohort(sink.records())
    rows = cohort.value_rows()[:8]

    model = build_model()

    # batched (current implementation)
    got = evaluate_stored_batch(model, rows, device="cuda")
    joints = [float(x.detach()) for x in got["joint_logprob"]]
    vals = [float(x.detach()) for x in got["values"]]

    # per-row reference: evaluate each row in its own call
    ref_j, ref_v = [], []
    for r in rows:
        single = evaluate_stored_batch(model, [r], device="cuda")
        ref_j.append(float(single["joint_logprob"][0].detach()))
        ref_v.append(float(single["values"][0].detach()))

    dj = [abs(a - b) for a, b in zip(joints, ref_j)]
    dv = [abs(a - b) for a, b in zip(vals, ref_v)]
    print("max |joint(batched) - joint(per-row)| :", f"{max(dj):.3e}")
    print("max |value(batched) - value(per-row)| :", f"{max(dv):.3e}")
    ok = max(dj) < 1e-5 and max(dv) < 1e-5
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
