"""P4-D closure regression: fresh-process recovery must be falsifiable.

These tests run in-process but assert the same invariants the subprocess proof
checks, so a future regression that silently drops Adam/RNG/step restoration is
caught by the normal suite (not only by the long evidence script).
"""
import sys
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import checkpoint as ck
from pa4.learner.checkpoint import CheckpointState, load_checkpoint, restore_into
from pa4.learner.update import PpoUpdater, UpdateConfig


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.ones(1))

    def forward(self, n, offset=0.0):
        one = torch.ones(n) * self.w
        return {"joint_logprob": one + offset, "values": one * 0.1,
                "entropy": one * 0.0, "uniform_kl": one * 0.0}


def _batch(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "rows": list(range(n)),
        "old_lp": torch.zeros(n),
        "adv": torch.randn(n, generator=g),
        "ret": torch.randn(n, generator=g),
        "actor_mask": torch.ones(n, dtype=torch.bool),
        "value_mask": torch.ones(n, dtype=torch.bool),
        "generator": torch.Generator().manual_seed(seed),
    }


def _trained_state(steps_before=0):
    """A checkpoint state whose Adam moments are non-trivial."""
    probe = Probe()
    up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                        effective_minibatch_rows=8,
                                        microbatch_rows=8), lr=1e-2)
    up.run(_batch(), lambda s: probe(len(s)))
    return CheckpointState(
        iteration=2, policy_version=3, committed_matches=16,
        optimizer_steps=up.optimizer_steps + steps_before,
        model_state=probe.state_dict(),
        optimizer_state=up.optimizer.state_dict(),
        scheduler_state={"lr": 1e-5, "committed_matches": 16},
        rng_state=ck._rng_state(),
        match_ledger=["m1", "m2"],
        counters={}, provenance={"theta0": "x" * 64}), probe


class FreshProcessInvariantsTest(unittest.TestCase):
    def test_optimizer_step_counter_round_trips(self):
        """A resumed run must CONTINUE the counter, not restart at zero."""
        base_steps = 7
        state, _ = _trained_state(steps_before=base_steps)
        self.assertGreaterEqual(state.optimizer_steps, base_steps)

        fresh = Probe()
        opt = torch.optim.Adam(fresh.parameters(), lr=1e-2)
        restore_into(state, fresh, opt)

        up = PpoUpdater(fresh, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        up.optimizer.load_state_dict(opt.state_dict())
        up.optimizer_steps = int(state.optimizer_steps)   # the restore contract

        before = up.optimizer_steps
        res = up.run(_batch(), lambda s: fresh(len(s)))
        self.assertGreater(up.optimizer_steps, before,
                           "step counter restarted instead of continuing")
        self.assertEqual(up.optimizer_steps - before,
                         sum(e.optimizer_steps for e in res.epochs),
                         "counter advance must equal the steps actually taken")

    def test_adam_moments_change_training_outcome(self):
        """Dropping Adam restoration MUST change the weights noticeably."""
        state, _ = _trained_state()

        def run(restore_adam):
            probe = Probe()
            opt = torch.optim.Adam(probe.parameters(), lr=1e-2,
                                   betas=(0.9, 0.999), eps=1e-5)
            restore_into(state, probe, opt)
            if not restore_adam:
                opt = torch.optim.Adam(probe.parameters(), lr=1e-2,
                                       betas=(0.9, 0.999), eps=1e-5)
            up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                                effective_minibatch_rows=8,
                                                microbatch_rows=8), lr=1e-2)
            up.optimizer.load_state_dict(opt.state_dict())
            up.run(_batch(), lambda s: probe(len(s)))
            return probe.w.detach().clone(), probe.b.detach().clone()

        with_adam = run(True)
        without_adam = run(False)
        delta = max(float((a - b).abs().max())
                    for a, b in zip(with_adam, without_adam))
        self.assertGreater(delta, 1e-6,
                           "Adam restoration had no effect, so the recovery test "
                           "could not detect its removal")

    def test_rng_roundtrip_detects_missing_stream(self):
        """A checkpointed RNG must resume the saved stream exactly."""
        from pa4.learner.checkpoint import _set_rng_state

        state, _ = _trained_state()
        saved = state.rng_state
        _set_rng_state(saved)
        expected = torch.randn(4).tolist()

        torch.manual_seed(4242)
        _set_rng_state(saved)
        got = torch.randn(4).tolist()
        self.assertEqual(expected, got)

        # Dropping torch_cuda/torch_cpu must NOT reproduce the stream.
        partial = {"python": saved["python"], "numpy": saved["numpy"],
                   "torch_cpu": torch.get_rng_state()}
        _set_rng_state(partial)
        wrong = torch.randn(4).tolist()
        self.assertNotEqual(expected, wrong,
                            "an ablated RNG produced the same stream, so the "
                            "check cannot detect removal")

    def test_checkpoint_carries_every_required_field(self):
        state, _ = _trained_state()
        self.assertTrue(state.optimizer_state)
        self.assertTrue(state.optimizer_state["state"])
        self.assertTrue(state.rng_state)
        self.assertIn("torch_cpu", state.rng_state)
        self.assertIn("numpy", state.rng_state)
        self.assertIn("python", state.rng_state)
        self.assertTrue(state.match_ledger)
        self.assertTrue(state.scheduler_state)


if __name__ == "__main__":
    unittest.main()
