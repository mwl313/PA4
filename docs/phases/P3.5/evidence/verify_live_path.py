"""P3.5 end-to-end: repaired LIVE PA4 path vs PA3 canonical encoder."""
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pa4.env.native_env import PA4EnvBase
from pa4.env import ragged_transport as _rt
from pa4.actions.batched_scorer import BatchedConditionalScorer, _ragged_from_td

from agent.model.pa3_model import PA3Model
from agent.types.observation import ObservationBatch
from agent.types.ragged import FIELDS


def live_ragged(players, r, side, dev):
    """Exactly the production route: players[r, side] -> wire row -> RaggedBatch."""
    row = players[r, side]
    flat = {}
    for f in _rt.family_names():
        for field in ("counts", "values", "lengths", "overflow"):
            flat[f"ragged.{f}_{field}"] = row[f"ragged.{f}_{field}"].unsqueeze(0)
    return _ragged_from_td(flat, [0], dev)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    torch.manual_seed(20261006)
    ref = PA3Model().to(dev)
    ref.configure_precision("A")
    ref.eval()

    env = PA4EnvBase(batch_size=4, seed=20261006)
    td = env.reset()

    worst_tok = worst_glob = worst_v = 0.0
    views = 0
    nonempty = {f: 0 for f in FIELDS}

    for step in range(40):
        obs_td = env._observation_td()
        for r, st in enumerate(list(env._slots)):
            for side in (0, 1):
                if not bool(obs_td["players"]["observation_valid"][r, side]):
                    continue  # Wait/Finished: stale leaves are the contract
                try:
                    fixed, ragged = env._engine.observe_fixed_batch(
                        [(st.slot, st.generation)], [side])
                    payload = env._mod.parse_batch(fixed, ragged, 1)
                    gold = ObservationBatch.from_native_payload(payload, dtype=torch.float32)
                except Exception:
                    continue
                for f in FIELDS:
                    if int(gold.ragged[f].counts.sum()) > 0:
                        nonempty[f] += 1

                rv = obs_td["players"][r, side]
                live = ObservationBatch(
                    token_mask=rv["token_mask"].to(dev).unsqueeze(0),
                    categories=rv["categories"].to(dev).unsqueeze(0).long(),
                    category_known=rv["category_known"].to(dev).unsqueeze(0),
                    floats=rv["floats"].to(dev).unsqueeze(0).float(),
                    float_known=rv["float_known"].to(dev).unsqueeze(0),
                    flags=rv["flags"].to(dev).unsqueeze(0),
                    flag_known=rv["flag_known"].to(dev).unsqueeze(0),
                    role_ids=rv["role_ids"].to(dev).unsqueeze(0).long(),
                    side_ids=rv["side_ids"].to(dev).unsqueeze(0).long(),
                    ragged=live_ragged(obs_td["players"], r, side, dev),
                )

                with torch.no_grad():
                    ea = ref.encode(gold.to(dev))
                    eb = ref.encode(live)
                    va = ref.value(ea)
                    vb = ref.value(eb)
                worst_tok = max(worst_tok, (ea.tokens - eb.tokens).abs().max().item())
                worst_glob = max(worst_glob,
                                 (ea.global_repr - eb.global_repr).abs().max().item())
                worst_v = max(worst_v, (va - vb).abs().max().item())
                views += 1

        act = env.legal_action_packet(torch.Generator().manual_seed(step))
        td = env.step(act)

    print()
    print(f"views compared       : {views}")
    print(f"max |d token|        : {worst_tok:.3e}")
    print(f"max |d global_repr|  : {worst_glob:.3e}")
    print(f"max |d V|            : {worst_v:.3e}")
    print()
    print("nonempty family views:", nonempty)
    print()
    ok = views > 0 and worst_tok <= 1e-6 and worst_glob <= 1e-6 and worst_v <= 1e-6
    print("LIVE-PATH PARITY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
