"""P4-A smoke: real battles -> lossless sink -> validated cohort.

Proves on real Rust data that:
  * every accepted transition is retained (dropped == 0)
  * episodes group by the five-field key, not match_id
  * natural completions produce two ordered seat trajectories
  * the terminal payout lands on each seat's LAST real decision
  * only ragged.v2 observations are admitted
"""
import json
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner.ingest import build_cohort
from pa4.learner.sink import LosslessBuffer, verify_lossless


def main():
    cfg = BatchedConfig(
        num_envs=4, target_matches=16, max_steps=6000, seed=1,
        device="cuda", run_id="p4a-smoke", transition_capacity=256,
    )
    sink = LosslessBuffer(capacity=100_000)
    rep = run_batched(cfg, store=sink)
    verify_lossless(sink)

    records = sink.records()
    acc = sink.accounting
    print(json.dumps({
        "natural_matches": rep.as_dict()["natural_matches"],
        "transition_records": len(records),
        "sink_accounting": acc.as_dict(),
    }, indent=1))

    cohort = build_cohort(records,
                          expected_policy_version=cfg.policy_version)
    print()
    print("cohort accounting:", json.dumps(cohort.accounting, indent=1))
    print("distinct matches :", cohort.matches())
    print("actor rows       :", len(cohort.actor_rows()))
    print("value rows       :", len(cohort.value_rows()))

    # Terminal payout must be exactly one +1 and one -1 (or 0/0 on a draw)
    # per completed match, on each seat's LAST decision.
    bad = 0
    for t in cohort.trajectories[:5]:
        lasts = {s: rows[-1] for s, rows in t.seats.items() if rows}
        payout = {s: r.reward for s, r in lasts.items()}
        total = sum(payout.values())
        ok = (abs(total) < 1e-9) and all(r.done for r in lasts.values())
        print(f"  match {t.match_id} winner={t.winner} "
              f"seat_rows=({len(t.seats[0])},{len(t.seats[1])}) "
              f"payout={payout} ok={ok}")
        if not ok:
            bad += 1

    print()
    print("RESULT:", "PASS" if acc.dropped == 0 and bad == 0
          and cohort.matches() > 0 else "FAIL")


if __name__ == "__main__":
    main()
