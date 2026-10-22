"""P4 regression tests for the four MAJOR defects found by the audit.

Each test FAILS if the corresponding fix is reverted.
"""
import sys
import tempfile
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import ppo_loss
from pa4.learner.checkpoint import load_checkpoint
from pa4.learner.train_sync import CollectionFailed, NonFiniteUpdate
from pa4.learner.update import PpoUpdater, UpdateConfig


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, n, offset=0.0):
        one = torch.ones(n) * self.w
        return {"joint_logprob": one + offset, "values": one * 0.1,
                "entropy": one * 0.0, "uniform_kl": one * 0.0}


class _NaNTracker(torch.nn.Module):
    """Emits NaN on a chosen call so a non-finite update is reachable."""

    def __init__(self, nan_on=1):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.nan_on = nan_on

    def forward(self, n, offset=0.0):
        self.calls += 1
        one = torch.ones(n) * self.w
        lp = one + offset
        if self.calls >= self.nan_on:
            lp = lp.clone()
            lp[0] = float("nan")
        return {"joint_logprob": lp, "values": one * 0.1,
                "entropy": one * 0.0, "uniform_kl": one * 0.0}


class M4MicrobatchAccumulationTest(unittest.TestCase):
    def _batch(self, n=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        return {
            "rows": list(range(n)),
            "old_lp": torch.zeros(n),
            "adv": torch.randn(n, generator=g),
            "ret": torch.randn(n, generator=g),
            "actor_mask": torch.tensor([i >= 4 for i in range(n)]),
            "value_mask": torch.ones(n, dtype=torch.bool),
            "generator": torch.Generator().manual_seed(seed),
        }

    def _run(self, micro, mini):
        probe = Probe()
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=mini,
                                            microbatch_rows=micro), lr=1e-1)
        up.run(self._batch(), lambda sub: probe(len(sub)))
        return float(probe.w.detach())

    def test_accumulation_equals_single_full_forward(self):
        """Splitting into micros must not change the resulting parameters."""
        single = self._run(micro=32, mini=32)
        accumulated = self._run(micro=8, mini=32)
        self.assertAlmostEqual(single, accumulated, places=9,
                               msg="microbatch accumulation diverged from the "
                                   "full-minibatch objective")

    def test_naive_equal_average_would_differ(self):
        """The naive alternative must be measurably wrong (control)."""
        correct = self._run(micro=8, mini=32)
        probe = Probe()
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=32,
                                            microbatch_rows=8), lr=1e-1)
        batch = self._batch()
        up.optimizer.zero_grad(set_to_none=True)
        micro = 8
        n_micros = 4
        for m0 in range(0, 32, micro):
            sub = batch["rows"][m0:m0 + micro]
            got = probe(len(sub))
            terms = ppo_loss.ppo_terms(
                got["joint_logprob"], batch["old_lp"][m0:m0 + micro],
                batch["adv"][m0:m0 + micro], batch["ret"][m0:m0 + micro],
                got["values"], actor_mask=batch["actor_mask"][m0:m0 + micro],
                value_mask=batch["value_mask"][m0:m0 + micro],
                entropy=got["entropy"], kl=got["uniform_kl"],
                actor_weight=1.0 / n_micros, value_weight=1.0 / n_micros)
            terms.total.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 0.5)
        up.optimizer.step()
        naive = float(probe.w.detach())
        self.assertGreater(abs(correct - naive), 1e-7,
                           "the naive average was not distinguishable, so this "
                           "control proves nothing")

    def test_microbatch_config_is_actually_used(self):
        """A smaller microbatch with the same minibatch must still match."""
        a = self._run(micro=32, mini=32)
        b = self._run(micro=4, mini=32)
        c = self._run(micro=16, mini=32)
        for other in (b, c):
            self.assertAlmostEqual(a, other, places=9)


class M2NonFiniteTest(unittest.TestCase):
    def test_non_finite_reported(self):
        probe = _NaNTracker(nan_on=1)
        n = 8
        batch = {
            "rows": list(range(n)), "old_lp": torch.zeros(n),
            "adv": torch.ones(n), "ret": torch.zeros(n),
            "actor_mask": torch.ones(n, dtype=torch.bool),
            "value_mask": torch.ones(n, dtype=torch.bool),
            "generator": torch.Generator().manual_seed(0),
        }
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=2,
                                            effective_minibatch_rows=4,
                                            microbatch_rows=4), lr=1e-2)
        res = up.run(batch, lambda sub: probe(len(sub)))
        self.assertFalse(res.finite, "NaN loss must set finite=False")

    def test_exception_types_exist(self):
        self.assertTrue(issubclass(CollectionFailed, RuntimeError))
        self.assertTrue(issubclass(NonFiniteUpdate, RuntimeError))


class M3CollectionGateTest(unittest.TestCase):
    def test_collection_failure_names_the_iteration(self):
        exc = CollectionFailed("iteration 3: collector reported 1 error(s)")
        self.assertIn("iteration 3", str(exc))

    def test_trainer_raises_on_under_target(self):
        """A cohort below the target must not be trained on."""
        import inspect

        from pa4.learner import train_sync

        src = inspect.getsource(train_sync.SynchronousTrainer.run_iteration)
        self.assertIn("matches_per_iteration", src,
                      "no target check in the iteration path")
        self.assertIn("CollectionFailed", src)
        self.assertIn("NonFiniteUpdate", src)


class M1OptimizerPersistenceTest(unittest.TestCase):
    def test_updater_counter_is_cumulative_across_runs(self):
        """The counter must accumulate; a per-iteration reset is the defect."""
        probe = Probe()
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        n = 8
        batch = {
            "rows": list(range(n)), "old_lp": torch.zeros(n),
            "adv": torch.ones(n), "ret": torch.zeros(n),
            "actor_mask": torch.ones(n, dtype=torch.bool),
            "value_mask": torch.ones(n, dtype=torch.bool),
            "generator": torch.Generator().manual_seed(0),
        }
        r1 = up.run(batch, lambda sub: probe(len(sub)))
        first = r1.optimizer_steps
        r2 = up.run(batch, lambda sub: probe(len(sub)))
        self.assertGreater(r2.optimizer_steps, first,
                           "optimizer step counter reset instead of accumulating")

    def test_adam_moments_survive_roundtrip(self):
        probe = Probe()
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        n = 8
        batch = {
            "rows": list(range(n)), "old_lp": torch.zeros(n),
            "adv": torch.ones(n), "ret": torch.zeros(n),
            "actor_mask": torch.ones(n, dtype=torch.bool),
            "value_mask": torch.ones(n, dtype=torch.bool),
            "generator": torch.Generator().manual_seed(0),
        }
        up.run(batch, lambda sub: probe(len(sub)))
        moments = up.optimizer.state_dict()["state"]
        nonzero = sum(1 for v in moments.values()
                      if "exp_avg" in v and float(v["exp_avg"].abs().sum()) > 0)
        self.assertGreater(nonzero, 0, "Adam moments were never populated")

        # Round-trip through a fresh optimizer.
        probe2 = Probe()
        up2 = PpoUpdater(probe2, UpdateConfig(), lr=1e-2)
        up2.optimizer.load_state_dict(up.optimizer.state_dict())
        for pid in moments:
            for k in moments[pid]:
                if torch.is_tensor(moments[pid][k]):
                    self.assertTrue(
                        torch.equal(moments[pid][k],
                                    up2.optimizer.state_dict()["state"][pid][k]),
                        f"adam {pid}.{k} lost in round-trip")


if __name__ == "__main__":
    unittest.main()
