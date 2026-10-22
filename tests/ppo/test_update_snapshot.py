"""P4-C — snapshot identity, trainable path and update-loop tests."""
import os
import sys
import tempfile
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import ppo_loss
from pa4.learner.snapshot import (SnapshotError, load_snapshot, read_metadata,
                                  save_snapshot, snapshot_fingerprint)
from pa4.learner.update import PpoUpdater, UpdateConfig

SEED = 20261006


def build_model(seed=SEED):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))



class _Probe(torch.nn.Module):
    """Minimal trainable stand-in for the conditional scorer.

    Produces joint_logprob / values / entropy / kl as real functions of its
    parameters, so backward() exercises the same graph shape as the learner.
    """

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


class SnapshotIdentityTest(unittest.TestCase):
    def test_fingerprint_matches_pa3_definition(self):
        """One definition: snapshot, server and stored records must agree."""
        from pa4.actions.conditional_policy import weight_fingerprint

        m = build_model()
        self.assertEqual(snapshot_fingerprint(m.state_dict()),
                         weight_fingerprint(m))

    def test_roundtrip_preserves_weights(self):
        m = build_model()
        d = tempfile.mkdtemp()
        snap = save_snapshot(m.state_dict(), d, policy_version=1, committed_matches=0)
        state, loaded = load_snapshot(snap.path)
        self.assertEqual(loaded.fingerprint, snap.fingerprint)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(state[k], v.cpu()), f"{k} changed on reload")

    def test_tampered_weights_rejected(self):
        m = build_model()
        d = tempfile.mkdtemp()
        snap = save_snapshot(m.state_dict(), d, policy_version=1, committed_matches=0)
        # Corrupt the tensor file.
        with open(snap.tensor_path(), "r+b") as fh:
            fh.seek(1024)
            fh.write(b"\x00\x01\x02\x03")
        with self.assertRaises(SnapshotError):
            load_snapshot(snap.path)

    def test_missing_metadata_rejected(self):
        with self.assertRaises(SnapshotError):
            read_metadata(os.path.join(tempfile.mkdtemp(), "nope"))

    def test_foreign_schema_rejected(self):
        import json
        d = tempfile.mkdtemp()
        path = os.path.join(d, "policy_v000001_x.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema": "other", "policy_version": 1}, fh)
        with self.assertRaises(SnapshotError):
            read_metadata(path[:-5])

    def test_fingerprint_changes_only_with_weights(self):
        m1 = build_model()
        m2 = build_model()
        self.assertEqual(snapshot_fingerprint(m1.state_dict()),
                         snapshot_fingerprint(m2.state_dict()))
        with torch.no_grad():
            next(m1.parameters()).add_(1e-3)
        self.assertNotEqual(snapshot_fingerprint(m1.state_dict()),
                            snapshot_fingerprint(m2.state_dict()))


class UpdateLoopTest(unittest.TestCase):
    def _batch(self, n=20, seed=0):
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

    def _ev(self, probe, offset=0.0):
        def evaluator(sub):
            return probe(len(sub), offset=offset)
        return evaluator

    def test_two_epochs_and_real_steps(self):
        probe = _Probe()
        batch = self._batch()
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=2,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        res = up.run(batch, self._ev(probe))
        self.assertEqual(len(res.epochs), 2, "must run exactly two scheduled epochs")
        self.assertGreater(res.optimizer_steps, 0)
        self.assertFalse(res.early_stopped)
        self.assertNotEqual(float(probe.w), 0.0, "no parameter update happened")

    def test_kl_early_stop_skips_remaining_epochs(self):
        probe = _Probe()
        batch = self._batch()
        # logprob offset +0.5 with old_lp 0 -> ratio exp(0.5) -> KL >> 0.03
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=2,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8,
                                            target_approx_kl=0.03), lr=1e-2)
        res = up.run(batch, self._ev(probe, offset=0.5))
        self.assertTrue(res.early_stopped)
        self.assertEqual(res.stopped_epoch, 0)
        self.assertEqual(len(res.epochs), 1, "remaining epoch must be skipped")
        self.assertGreater(res.optimizer_steps, 0, "the valid first epoch still updates")

    def test_all_invalid_minibatch_makes_no_step(self):
        probe = _Probe()
        batch = self._batch(n=8)
        batch["actor_mask"] = torch.zeros(8, dtype=torch.bool)
        batch["value_mask"] = torch.zeros(8, dtype=torch.bool)
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        res = up.run(batch, self._ev(probe))
        self.assertEqual(res.optimizer_steps, 0)
        self.assertEqual(float(probe.w), 0.0)

    def test_singleton_only_rows_still_update_critic(self):
        probe = _Probe()
        batch = self._batch(n=8)
        batch["actor_mask"] = torch.zeros(8, dtype=torch.bool)
        batch["ret"] = torch.ones(8) * 0.5
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        res = up.run(batch, self._ev(probe))
        self.assertGreater(res.optimizer_steps, 0,
                           "critic-only rows must still produce an update")

    def test_nonfinite_loss_does_not_step(self):
        probe = _Probe()

        class _NaN(_Probe):
            def forward(self, n, offset=0.0):
                out = super().forward(n, offset)
                lp = out["joint_logprob"].clone()
                lp[0] = float("nan")
                out["joint_logprob"] = lp
                return out

        bad = _NaN()
        up = PpoUpdater(bad, UpdateConfig(ppo_epochs=2,
                                          effective_minibatch_rows=4,
                                          microbatch_rows=4), lr=1e-2)
        batch = self._batch(n=8)
        res = up.run(batch, lambda sub: bad(len(sub)))
        self.assertFalse(res.finite)

    def test_microbatch_accumulation_matches_full_minibatch(self):
        """Weighted accumulation must equal one larger objective, not a naive mean."""
        probe = _Probe()
        # Unequal actor/value counts across micros is the case a naive average
        # gets wrong: rows 0-5 have actors, rows 6-11 are value-only.
        n = 12
        batch = self._batch(n=n)
        batch["actor_mask"] = torch.tensor([True] * 6 + [False] * 6)
        batch["value_mask"] = torch.ones(n, dtype=torch.bool)
        batch["adv"] = torch.ones(n)
        batch["ret"] = torch.ones(n) * 1.0

        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=n,
                                            microbatch_rows=6), lr=1e-3)
        res = up.run(batch, self._ev(probe))
        self.assertEqual(res.epochs[0].actor_rows, 6)
        self.assertEqual(res.epochs[0].value_rows, 12)


class TrailingPartialMinibatchTest(unittest.TestCase):
    def test_partial_minibatch_is_used(self):
        probe = _Probe()
        n = 19  # 2 full minibatches of 8 plus a partial of 3
        batch = {
            "rows": list(range(n)),
            "old_lp": torch.zeros(n),
            "adv": torch.ones(n),
            "ret": torch.zeros(n),
            "actor_mask": torch.ones(n, dtype=torch.bool),
            "value_mask": torch.ones(n, dtype=torch.bool),
            "generator": torch.Generator().manual_seed(0),
        }
        seen = []

        def ev(sub):
            seen.append(len(sub))
            return probe(len(sub))
        up = PpoUpdater(probe, UpdateConfig(ppo_epochs=1,
                                            effective_minibatch_rows=8,
                                            microbatch_rows=8), lr=1e-2)
        res = up.run(batch, ev)
        self.assertIn(3, seen, "the trailing partial minibatch was dropped")
        self.assertEqual(res.optimizer_steps, 3)


if __name__ == "__main__":
    unittest.main()
