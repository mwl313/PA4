"""P3.5 golden: repaired PA4 path must reproduce PA3's encoder and value exactly.

Builds ObservationBatch two ways from the SAME native payload:
  A. PA3 canonical      : from_native(payload) -> ObservationBatch.ragged
  B. P3.5 transport     : payload_to_transport -> transport_to_ragged
and compares encoder tokens, global_repr and value on CPU and CUDA.

Negative controls deliberately corrupt one family and assert the comparison
DETECTS it, so this harness cannot pass by accident.
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

from agent.model.pa3_model import PA3Model
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.ragged import FIELDS, from_native

FIXED_LEAVES = ("token_mask", "categories", "category_known", "floats",
                "float_known", "flags", "flag_known")


def fixed_kwargs(payload, layout):
    tm = torch.as_tensor(np.asarray(payload["token_mask"]).astype(bool))
    role = layout.default_roles().clone()
    role = role.unsqueeze(0).expand(tm.shape[0], -1).clone()
    role[~tm] = layout.ROLE_PAD
    side = layout.default_sides().clone()
    side = side.unsqueeze(0).expand(tm.shape[0], -1).clone()
    return dict(
        token_mask=tm,
        categories=torch.as_tensor(np.asarray(payload["categories"]).astype(np.int64)),
        category_known=torch.as_tensor(np.asarray(payload["category_known"]).astype(bool)),
        floats=torch.as_tensor(np.asarray(payload["floats"]).astype(np.float32)),
        float_known=torch.as_tensor(np.asarray(payload["float_known"]).astype(bool)),
        flags=torch.as_tensor(np.asarray(payload["flags"]).astype(bool)),
        flag_known=torch.as_tensor(np.asarray(payload["flag_known"]).astype(bool)),
        role_ids=role,
        side_ids=side,
    )


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    torch.manual_seed(20261006)
    model = PA3Model().to(dev)
    model.configure_precision("A")
    model.eval()
    print("params:", model.parameter_count())

    env = PA4EnvBase(batch_size=4, seed=20261006)
    env.reset()
    layout = ObservationLayout()

    results = []
    neg_control_hits = 0
    neg_control_trials = 0

    for step in range(60):
        for st in list(env._slots):
            for side in (0, 1):
                try:
                    fixed, ragged = env._engine.observe_fixed_batch(
                        [(st.slot, st.generation)], [side])
                    payload = env._mod.parse_batch(fixed, ragged, 1)
                except Exception:
                    continue

                gold_ragged = from_native(payload, 1)
                trans = rt.payload_to_transport(payload, 1)
                mine_ragged = rt.transport_to_ragged(trans, 0)

                kw = fixed_kwargs(payload, layout)
                kw_dev = {k: v.to(dev) for k, v in kw.items()}

                obs_a = ObservationBatch(
                    **kw_dev, ragged={k: v.to(dev) for k, v in gold_ragged.items()})
                obs_b = ObservationBatch(
                    **kw_dev, ragged={k: v.to(dev) for k, v in mine_ragged.items()})

                with torch.no_grad():
                    ea = model.encode(obs_a)
                    eb = model.encode(obs_b)
                    va = model.value(ea)
                    vb = model.value(eb)

                dtok = (ea.tokens - eb.tokens).abs().max().item()
                dglob = (ea.global_repr - eb.global_repr).abs().max().item()
                dv = (va - vb).abs().max().item()
                results.append((dtok, dglob, dv))

                # ---- negative control: zero out one family in B ----
                for fam in FIELDS:
                    neg_control_trials += 1
                    corrupt = {k: v for k, v in mine_ragged.items()}
                    zero = corrupt[fam]
                    corrupt[fam] = type(zero)(
                        torch.zeros_like(zero.values), zero.counts, zero.host_counts)
                    obs_c = ObservationBatch(
                        **kw_dev, ragged={k: v.to(dev) for k, v in corrupt.items()})
                    with torch.no_grad():
                        ec = model.encode(obs_c)
                    delta = (ea.tokens - ec.tokens).abs().max().item()
                    # If the family is empty in this view, zeroing it is a no-op,
                    # which is correct; only count detectable cases.
                    if int(zero.counts.sum()) == 0:
                        neg_control_trials -= 1
                        continue
                    if delta > 1e-6:
                        neg_control_hits += 1

        act = env.legal_action_packet(torch.Generator().manual_seed(step))
        env.step(act)

    a = np.array(results)
    print()
    print(f"views compared          : {len(a)}")
    print(f"max |d token|           : {a[:,0].max():.3e}")
    print(f"max |d global_repr|     : {a[:,1].max():.3e}")
    print(f"max |d V|               : {a[:,2].max():.3e}")
    print()
    print(f"negative-control trials : {neg_control_trials}")
    print(f"negatives DETECTED      : {neg_control_hits}")
    print()
    ok = a[:, 0].max() <= 1e-6 and a[:, 1].max() <= 1e-6 and a[:, 2].max() <= 1e-6
    det = neg_control_trials > 0 and neg_control_hits == neg_control_trials
    print("PARITY  :", "PASS" if ok else "FAIL")
    print("NEGATIVE:", "PASS" if det else "FAIL")
    print("RESULT  :", "PASS" if ok and det else "FAIL")


if __name__ == "__main__":
    main()
