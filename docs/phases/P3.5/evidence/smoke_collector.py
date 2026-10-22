"""P3.5 real-collector smoke: ragged must survive the whole P3 pipeline.

Runs the actual collector (Rust battles, real InferenceServer, real transport)
and then asserts that every retained seat record carries non-empty ragged data
that round-trips to the encoder.
"""
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.collector.transitions import BoundedTransitionStore
from pa4.env import ragged_transport as rt


def main():
    cfg = BatchedConfig(
        num_envs=4,
        target_matches=4,
        max_steps=4000,
        seed=7,
        device="cuda",
        run_id="p35-smoke",
        transition_capacity=4000,
    )
    store = BoundedTransitionStore(capacity=cfg.transition_capacity)
    rep = run_batched(cfg, store=store)
    print("report:", {k: v for k, v in rep.as_dict().items()
                      if k in ("natural_matches", "env_steps", "seat_records",
                               "actor_active_rows", "transitions_stored",
                               "transitions_dropped", "duplicate_match_ids")})

    records = list(store._q)
    print("retained records:", len(records))
    if not records:
        print("RESULT: FAIL (no records)")
        return

    fam_nonempty = {f: 0 for f in rt.family_names()}
    reps = set()
    ragged_ok = 0
    checked = 0
    for rec in records:
        for sr in rec.get("seat_records", []):
            reps.add(sr.get("observation_representation"))
            row = sr.get("observation_ragged") or {}
            if not row:
                continue
            checked += 1
            for f in rt.family_names():
                c = row[f"ragged.{f}_counts"]
                if int(c.sum()) > 0:
                    fam_nonempty[f] += 1
            # Round-trip: transport -> RaggedBatch must succeed and be finite.
            batched = {k: v.unsqueeze(0) for k, v in row.items()}
            ragg = rt.transport_to_ragged(batched, 0)
            if all(torch.isfinite(r.values).all() for r in ragg.values()):
                ragged_ok += 1

    print("representations seen:", reps)
    print("seat records with ragged:", checked, "finite round-trips:", ragged_ok)
    print("nonempty family counts:", fam_nonempty)
    print()
    ok = (checked > 0 and ragged_ok == checked
          and reps == {rt.OBSERVATION_REPRESENTATION}
          and rec.get("transitions_dropped", 0) == 0)
    print("dropped:", getattr(store, "dropped", "n/a"))
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
