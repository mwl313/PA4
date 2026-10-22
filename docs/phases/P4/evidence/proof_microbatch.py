"""M4 proof: microbatch accumulation must equal the full-minibatch objective.

The auditor's finding was that the accumulation helper had no callers, so this
claim was untested. Here we compare, on identical inputs:

  A. one forward over ALL rows, one backward  (the reference objective)
  B. the updater's real loop with microbatch_rows < effective_minibatch_rows

The resulting PARAMETER VECTORS must agree to FP32 tolerance if accumulation is
row-weighted correctly. A naive equal-micro average would diverge.
"""
import sys

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import ppo_loss
from pa4.learner.update import PpoUpdater, UpdateConfig


class Probe(torch.nn.Module):
    """Trainable stand-in whose outputs depend on parameters."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, n, offset=0.0):
        one = torch.ones(n) * self.w
        return {
            "joint_logprob": one + offset,
            "values": one * 0.1,
            "entropy": one * 0.0,
            "uniform_kl": one * 0.0,
        }


def make_batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "rows": list(range(n)),
        "old_lp": torch.zeros(n),
        "adv": torch.randn(n, generator=g),
        "ret": torch.randn(n, generator=g),
        # Strongly unequal actor counts per microbatch: the first micro has
        # almost no actor rows, the last has many. A naive equal average gets
        # this materially wrong; row-weighted accumulation does not.
        "actor_mask": torch.tensor([i >= 4 for i in range(n)]),
        "value_mask": torch.ones(n, dtype=torch.bool),
        "generator": torch.Generator().manual_seed(seed),
    }


def run(probe, batch, micro, mini, epochs=1):
    up = PpoUpdater(probe, UpdateConfig(ppo_epochs=epochs,
                                        effective_minibatch_rows=mini,
                                        microbatch_rows=micro), lr=1e-1)
    res = up.run(batch, lambda sub: probe(len(sub)))
    return probe.w.detach().clone(), res


def main():
    # A: micro == mini  (single forward per minibatch)
    a_probe = Probe()
    a_w, a_res = run(a_probe, make_batch(), micro=32, mini=32)

    # B: micro < mini    (real accumulation)
    b_probe = Probe()
    b_w, b_res = run(b_probe, make_batch(), micro=8, mini=32)

    # C: naive equal-average control, which must NOT match
    c_probe = Probe()
    up = PpoUpdater(c_probe, UpdateConfig(ppo_epochs=1,
                                          effective_minibatch_rows=32,
                                          microbatch_rows=8), lr=1e-1)
    # emulate a naive implementation: each micro scaled 1/num_micros
    batch = make_batch()
    up.optimizer.zero_grad(set_to_none=True)
    n_micros = 4
    for m0 in range(0, 32, 8):
        sub = batch["rows"][m0:m0 + 8]
        got = c_probe(len(sub))
        am = batch["actor_mask"][m0:m0 + 8]
        vm = batch["value_mask"][m0:m0 + 8]
        terms = ppo_loss.ppo_terms(
            got["joint_logprob"], batch["old_lp"][m0:m0 + 8],
            batch["adv"][m0:m0 + 8], batch["ret"][m0:m0 + 8], got["values"],
            actor_mask=am, value_mask=vm, entropy=got["entropy"],
            kl=got["uniform_kl"], actor_weight=1.0 / n_micros,
            value_weight=1.0 / n_micros)
        terms.total.backward()
    torch.nn.utils.clip_grad_norm_(c_probe.parameters(), 0.5)
    up.optimizer.step()
    c_w = c_probe.w.detach().clone()

    d_ab = float((a_w - b_w).abs().max())
    d_ac = float((a_w - c_w).abs().max())
    print(f"A (single forward) w      = {float(a_w):.10f}")
    print(f"B (accumulated micros) w  = {float(b_w):.10f}")
    print(f"C (naive equal average) w = {float(c_w):.10f}")
    print(f"|A - B| = {d_ab:.3e}   <- accumulation must match")
    print(f"|A - C| = {d_ac:.3e}   <- naive average must NOT match")
    print(f"optimizer steps A={a_res.optimizer_steps} B={b_res.optimizer_steps}")
    # Accumulation must match the reference exactly; the naive control must
    # be measurably different (it is wrong, just slightly).
    ok = (d_ab <= 1e-9 and d_ac > 1e-7
          and a_res.optimizer_steps == b_res.optimizer_steps)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
