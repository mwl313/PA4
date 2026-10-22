"""P3.5 golden check: transport must be byte-exact vs PA3's canonical path.

For real battles, for every seat view:
  1. take the native payload (fixed+ragged) exactly as PA4 fetches it
  2. PA3 canonical  -> RaggedBatch via from_native
  3. P3.5 transport -> transport_to_ragged
  4. assert counts/values/host_counts are IDENTICAL (not "close")

The encoder/value comparison lives in verify_encoder_parity.py.
"""
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pa4.env.native_env import PA4EnvBase
from pa4.env import ragged_transport as rt

from agent.types.ragged import FIELDS


def canonical_ragged(payload):
    """PA3's own canonical conversion — the golden reference.

    `from_native` is the exact path PA3 uses to turn a native payload into
    `RaggedBatch` objects for the encoder.
    """
    from agent.types.ragged import from_native

    return from_native(payload, 1)




def main():
    env = PA4EnvBase(batch_size=4, seed=20261006)
    env.reset()

    checked = 0
    mismatch = 0
    fam_seen = {k: 0 for k in FIELDS}
    max_rows = {k: 0 for k in FIELDS}

    for step in range(120):
        for st in list(env._slots):
            for side in (0, 1):
                try:
                    fixed, ragged = env._engine.observe_fixed_batch(
                        [(st.slot, st.generation)], [side])
                    payload = env._mod.parse_batch(fixed, ragged, 1)
                except Exception:
                    continue

                # --- golden: PA3 canonical splitter ---
                gold = canonical_ragged(payload)

                # --- P3.5: transport round-trip ---
                trans = rt.payload_to_transport(payload, 1)
                mine = rt.transport_to_ragged(trans, 0)

                for fam in FIELDS:
                    g, m = gold[fam], mine[fam]
                    gc = np.asarray(g.counts.detach().cpu().numpy())
                    mc = np.asarray(m.counts.detach().cpu().numpy())
                    if not np.array_equal(gc, mc):
                        mismatch += 1
                        print(f"COUNT MISMATCH {fam} step={step} slot={st.slot} side={side}")
                        continue
                    if not np.array_equal(g.host_counts, m.host_counts):
                        mismatch += 1
                        print(f"HOST_COUNT MISMATCH {fam}")
                        continue
                    gv = g.values.detach().cpu().numpy().astype(np.float32)
                    mv = m.values.detach().cpu().numpy().astype(np.float32)
                    if gv.shape != mv.shape or not np.array_equal(gv, mv):
                        mismatch += 1
                        print(f"VALUE MISMATCH {fam}: shapes {gv.shape} vs {mv.shape}")
                        continue
                    if gc.sum():
                        fam_seen[fam] += 1
                    max_rows[fam] = max(max_rows[fam], int(gc.max()) if gc.size else 0)
                checked += 1

        act = env.legal_action_packet(torch.Generator().manual_seed(step))
        env.step(act)

    print()
    print(f"views checked : {checked}")
    print(f"mismatches    : {mismatch}")
    print()
    print("=== nonempty family observations (real battles) ===")
    for fam in FIELDS:
        print(f"  {fam:14s} views_nonempty={fam_seen[fam]:5d}  max_rows_per_token={max_rows[fam]}")
    print()
    print("RESULT:", "BYTE-EXACT PASS" if mismatch == 0 and checked > 0 else "FAIL")


if __name__ == "__main__":
    main()
