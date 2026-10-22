"""P3.5 capacity survey: what padding capacity do real battles actually need?

Runs real battles and records the maximum per-token row count for each family,
plus the distribution, so the transport capacity is evidence-based rather than
guessed. Capacity is a PADDING bound; overflow still fails closed.
"""
import json
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pa4.env.native_env import PA4EnvBase

FAMS = ("effect_counts", "repertoire_counts", "type_counts",
        "base_move_counts", "move_effect_counts")


def main():
    env = PA4EnvBase(batch_size=8, seed=424242)
    env.reset()

    maxima = {k: 0 for k in FAMS}
    totals = {k: 0 for k in FAMS}
    views = 0
    hist = {k: {} for k in FAMS}

    for step in range(1500):
        for st in list(env._slots):
            for side in (0, 1):
                try:
                    f, r = env._engine.observe_fixed_batch(
                        [(st.slot, st.generation)], [side])
                    p = env._mod.parse_batch(f, r, 1)
                except Exception:
                    continue
                views += 1
                for k in FAMS:
                    c = np.asarray(p[k])
                    m = int(c.max())
                    if m > maxima[k]:
                        maxima[k] = m
                    totals[k] = max(totals[k], int(c.sum()))
                    hist[k][m] = hist[k].get(m, 0) + 1
        env.step(env.legal_action_packet(torch.Generator().manual_seed(step)))

    out = {
        "seat_views": views,
        "steps": 1500,
        "batches": 8,
        "max_rows_per_token": maxima,
        "max_total_rows_per_view": totals,
        "distribution_max_per_token": {
            k: {str(m): n for m, n in sorted(hist[k].items())} for k in FAMS
        },
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
