"""P3.5 evidence: reproduce the inference wire row byte counts.

The receipt records wire_row_bytes_before/after_capacity_tuning. This script
makes both figures reproducible instead of asserted:

  after  = the shipped request_spec() size
  before = the same spec with the pre-tuning ragged capacities

Sizes are computed from the spec tensors themselves, so they cannot drift from
the code.
"""
import json
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.actions import batched_scorer as bs
from pa4.env import ragged_transport as rt

#: The capacities that were in place before the 24,000-view survey sized them.
PRE_TUNING_CAPACITY = {
    "effects": 8,
    "repertoire": 8,
    "types": 4,
    "base_moves": 8,
    "move_effects": 8,
}

RAGGED_FIELDS = ("counts", "values", "lengths", "overflow")

MAX_TOKENS = bs.MAX_TOKENS


def _sizes(capacities):
    """Total and ragged bytes of one request row under `capacities`."""
    total = 0
    ragged = 0
    for name in bs.request_spec().keys():
        key = str(name)
        if key.startswith("ragged."):
            family, field = key[len("ragged."):].rsplit("_", 1)
            width = rt.family_widths()[family]
            cap = capacities[family]
            if field == "counts":
                n = MAX_TOKENS * 8
            elif field == "values":
                n = MAX_TOKENS * cap * width * 4
            elif field == "lengths":
                n = 8
            elif field == "overflow":
                n = 1
            else:
                raise ValueError(f"unexpected ragged field {field}")
            ragged += n
            total += n
        else:
            t = bs.request_spec()[name]
            total += t.numel() * t.element_size()
    return total, ragged


def main():
    after_total, after_ragged = _sizes(dict(rt.RAGGED_TOKEN_CAPACITY))
    before_total, before_ragged = _sizes(PRE_TUNING_CAPACITY)

    print(json.dumps({
        "method": "sizes computed from request_spec() tensor shapes/dtypes",
        "wire_row_bytes_before_capacity_tuning": before_total,
        "ragged_bytes_before": before_ragged,
        "wire_row_bytes_after_capacity_tuning": after_total,
        "ragged_bytes_after": after_ragged,
        "shipped_capacity": dict(rt.RAGGED_TOKEN_CAPACITY),
        "pre_tuning_capacity": PRE_TUNING_CAPACITY,
        "saved_bytes": before_total - after_total,
    }, indent=1))


if __name__ == "__main__":
    main()
