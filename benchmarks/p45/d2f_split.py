"""D2f: split evaluate_stored_batch into its parts, as a first-class artifact.

The 73.6% figure was previously used in the docs but existed only as an
unreproducible in-session measurement. This script produces it as evidence:
for one real cohort, time
  A. observation rebuild (Python)
  B. multi-level branch-table construction (Python)
  C. batched encoder (GPU)
  D. the per-row evaluate_encoded loop (the suspect)
so the per-row scorer share is traceable.
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

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner import learner_eval as le
from pa4.learner.ingest import build_cohort
from pa4.learner.parity import build_observation
from pa4.learner.sink import LosslessBuffer
from pa4.learner.snapshot import save_snapshot


def build_model(seed=20261006):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def main():
    d = tempfile.mkdtemp(prefix="p45_split2_")
    base = build_model()
    snap = save_snapshot(base.state_dict(), d, policy_version=1, committed_matches=0)

    cfg = BatchedConfig(num_envs=4, target_matches=8, max_steps=8000, seed=3,
                        device="cuda", run_id="p45-split", policy_version=1,
                        snapshot_path=snap.path, transition_capacity=200_000)
    sink = LosslessBuffer(capacity=200_000)
    run_batched(cfg, store=sink)
    cohort = build_cohort(sink.records())
    rows = cohort.value_rows()
    model = build_model().cuda().train()
    n = len(rows)

    le.evaluate_stored_batch(model, rows[:4], device="cuda")
    torch.cuda.synchronize()

    REP = 3
    t0 = time.perf_counter()
    for _ in range(REP):
        le.evaluate_stored_batch(model, rows, device="cuda")
    torch.cuda.synchronize()
    full = (time.perf_counter() - t0) / REP

    t0 = time.perf_counter()
    for _ in range(REP):
        obs = [build_observation(r, device="cuda") for r in rows]
    torch.cuda.synchronize()
    obs_t = (time.perf_counter() - t0) / REP

    t0 = time.perf_counter()
    for _ in range(REP):
        for i, r in enumerate(rows):
            a = le._table_roster(obs[i].active_to_roster())
            le._multi_level_table_plain(r, a)
    torch.cuda.synchronize()
    tbl_t = (time.perf_counter() - t0) / REP

    batched = le._stack_observations(obs, "cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REP):
        model.encode(batched)
    torch.cuda.synchronize()
    enc_t = (time.perf_counter() - t0) / REP

    encoded_all = model.encode(batched)
    t0 = time.perf_counter()
    for _ in range(REP):
        for i, r in enumerate(rows):
            single = le._slice_encoded(encoded_all, i)
            a = le._table_roster(obs[i].active_to_roster())
            table = le._multi_level_table_plain(r, a).to("cuda")
            sel = torch.tensor([[int(b["selected_index"]) for b in r.branches]],
                               dtype=torch.long, device="cuda")
            model.evaluate_encoded(single, table, selected=sel)
    torch.cuda.synchronize()
    ev_t = (time.perf_counter() - t0) / REP

    out = {
        "rows": n,
        "repeats": REP,
        "total_ms": round(full * 1000, 1),
        "components_ms": {
            "obs_rebuild": round(obs_t * 1000, 1),
            "branch_tables": round(tbl_t * 1000, 1),
            "batched_encoder": round(enc_t * 1000, 1),
            "per_row_evaluate": round(ev_t * 1000, 1),
        },
        "share_pct": {
            "obs_rebuild": round(obs_t / full * 100, 1),
            "branch_tables": round(tbl_t / full * 100, 1),
            "batched_encoder": round(enc_t / full * 100, 1),
            "per_row_evaluate": round(ev_t / full * 100, 1),
        },
        "per_row_evaluate_ms_per_row": round(ev_t / n * 1000, 3),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
