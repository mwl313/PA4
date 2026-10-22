"""P4-B — frozen-policy provenance probe.

The continuation prompt warns that "same seed" is not provenance. This probe
answers, on a REAL collected cohort:

  1. What fingerprint do the stored behavior records carry?
  2. Does rebuilding PA3Model(seed=20261006) reproduce that exact fingerprint?
  3. Does the served scorer's own weight_fingerprint agree?
  4. Can the stored ragged.v2 observation reproduce the stored joint logprob?

If (2)/(3)/(4) hold, theta_0 is genuine provenance for P4-B replay.
"""
import json
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.actions.batched_scorer import model_fingerprint
from pa4.actions.conditional_policy import weight_fingerprint
from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner.sink import LosslessBuffer

SEED = 20261006


def main():
    cfg = BatchedConfig(num_envs=4, target_matches=4, max_steps=6000, seed=1,
                        device="cuda", run_id="p4b-probe", transition_capacity=256,
                        policy_seed=SEED)
    sink = LosslessBuffer(capacity=100_000)
    run_batched(cfg, store=sink)
    records = sink.records()

    stored = {r["policy_weight_fingerprint"] for r in records}
    stored_versions = {int(r["policy_version"]) for r in records}
    print("records                :", len(records))
    print("stored fingerprints    :", stored)
    print("stored policy versions :", stored_versions)

    seeded_fp = model_fingerprint(SEED)
    print("model_fingerprint(seed):", seeded_fp)

    # Rebuild the model independently from the seed and fingerprint the weights.
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(SEED)
    model = PA3Model(PA3Config(initialization="random", seed=SEED))
    direct_fp = weight_fingerprint(model)
    print("direct weight fingerprint:", direct_fp)

    print()
    print("stored == seeded helper :", stored == {seeded_fp})
    print("stored == direct weights:", stored == {direct_fp})

    # Observation contract hash and representation
    reps = {sr.get("observation_representation")
            for r in records for sr in r["seat_records"]}
    raggeds = [sr.get("observation_ragged") for r in records
               for sr in r["seat_records"]]
    nonempty = sum(1 for rg in raggeds if rg and
                   sum(int(v.sum()) for k, v in rg.items() if k.endswith("_counts")) > 0)
    print("representations        :", reps)
    print("seat records           :", len(raggeds), "with non-empty ragged:", nonempty)
    print()
    print(json.dumps({
        "records": len(records),
        "stored_fingerprint": sorted(stored),
        "seeded_helper_fingerprint": seeded_fp,
        "direct_weight_fingerprint": direct_fp,
        "provenance_match": stored == {direct_fp},
        "representations": sorted(x for x in reps if x),
        "seat_records_with_nonempty_ragged": nonempty,
    }, indent=1))


if __name__ == "__main__":
    main()
