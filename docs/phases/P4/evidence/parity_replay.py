"""P4-B parity probe: replay stored ragged.v2 rows under frozen weights."""
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.actions.conditional_policy import ConditionalPolicy
from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner import parity
from pa4.learner.ingest import build_cohort
from pa4.learner.sink import LosslessBuffer


def main():
    cfg = BatchedConfig(num_envs=4, target_matches=2, max_steps=6000, seed=1,
                        device="cuda", run_id="p4b-tb", transition_capacity=256)
    sink = LosslessBuffer(capacity=100_000)
    run_batched(cfg, store=sink)
    cohort = build_cohort(sink.records())
    rows = cohort.value_rows()[:12]
    print("rows to replay:", len(rows))

    scorer = ConditionalPolicy(seed=20261006, device="cpu", policy_version=1)
    print("scorer fingerprint:", scorer.fingerprint[:16], "...")

    worst_b = 0.0
    worst_j = 0.0
    for r in rows:
        res = parity.replay_seat_row(scorer, r, device="cpu")
        worst_b = max(worst_b, res["max_branch_delta"])
        worst_j = max(worst_j, res["joint_delta"])

    print("max branch delta:", f"{worst_b:.3e}")
    print("max joint  delta:", f"{worst_j:.3e}")
    ok = worst_b <= 1e-5 and worst_j <= 1e-5
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
