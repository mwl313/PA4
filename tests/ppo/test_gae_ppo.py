"""P4-B — GAE and PPO golden parity + falsifiable negative controls.

`pa4.learner.golden` is a separate implementation from `pa4.learner.gae` and
`pa4.learner.ppo_loss`. Comparing them is meaningful only because neither calls
the other.
"""
import math
import sys
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import gae, golden, ppo_loss


class GaeGoldenTest(unittest.TestCase):
    def test_single_row_terminal(self):
        # One decision, terminal reward +1, bootstrap 0.
        adv, ret = gae.seat_gae([0.5], [1.0], [True])
        g_adv, g_ret = golden.gae([0.5], [1.0], [True])
        self.assertAlmostEqual(float(adv[0]), g_adv[0], places=6)
        self.assertAlmostEqual(float(ret[0]), g_ret[0], places=6)
        # delta = 1.0 + 0 - 0.5 = 0.5
        self.assertAlmostEqual(g_adv[0], 0.5, places=6)

    def test_three_row_hand_computed(self):
        values = [0.0, 0.0, 0.0]
        rewards = [0.0, 0.0, 1.0]
        dones = [False, False, True]
        adv, ret = gae.seat_gae(values, rewards, dones)
        g_adv, g_ret = golden.gae(values, rewards, dones)
        for t in range(3):
            self.assertAlmostEqual(float(adv[t]), g_adv[t], places=6)
            self.assertAlmostEqual(float(ret[t]), g_ret[t], places=6)
        # With gamma=1, lambda=.95 and zero values:
        # d2 = 1, d1 = .95*1 = .95, d0 = .95*.95 = .9025
        self.assertAlmostEqual(float(adv[2]), 1.0, places=6)
        self.assertAlmostEqual(float(adv[1]), 0.95, places=6)
        self.assertAlmostEqual(float(adv[0]), 0.95 ** 2, places=6)

    def test_signed_reward_per_side(self):
        for reward in (1.0, -1.0, 0.0):
            adv, ret = gae.seat_gae([0.25], [reward], [True])
            g_adv, _ = golden.gae([0.25], [reward], [True])
            self.assertAlmostEqual(float(adv[0]), g_adv[0], places=6)
            self.assertAlmostEqual(float(adv[0]), reward - 0.25, places=6)

    def test_chain_must_end_done(self):
        with self.assertRaises(ValueError):
            gae.seat_gae([0.0, 0.0], [0.0, 1.0], [False, False])

    def test_terminal_row_ignores_any_later_value(self):
        """A terminal row's delta must not depend on a value beyond it."""
        # Single-row chain and the same row as the last of a longer chain must
        # give that row the same advantage: nothing after a done=True leaks in.
        adv1, _ = gae.seat_gae([0.0], [1.0], [True])
        adv2, _ = gae.seat_gae([0.0, 0.0], [0.0, 1.0], [False, True])
        self.assertAlmostEqual(float(adv2[-1]), float(adv1[0]), places=6)
        # ... and the earlier row is NOT affected by the terminal payout beyond
        # the lambda discount (gamma=1, lambda=.95).
        self.assertAlmostEqual(float(adv2[0]), 0.95 * 1.0, places=6)


class NormalizationTest(unittest.TestCase):
    def test_population_std_matches_golden(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, True, False])
        ours = gae.normalize_advantages(adv, mask)
        theirs = golden.normalize_advantages(adv.tolist(), mask.tolist())
        abs_d, _, ok = golden.compare(theirs, ours)
        self.assertTrue(ok, f"normalization diverged by {abs_d}")

    def test_normalized_over_actor_rows_only(self):
        adv = torch.tensor([10.0, 10.0, 10.0, 1000.0])
        mask = torch.tensor([True, True, True, False])
        out = gae.normalize_advantages(adv, mask)
        # Actor rows were all equal, so they normalize to ~0; the masked-out
        # row must not drag the statistics.
        self.assertAlmostEqual(float(out[:3].abs().max()), 0.0, places=5)

    def test_returns_stay_unnormalized(self):
        _, ret = gae.seat_gae([0.0, 0.0], [0.0, 1.0], [False, True])
        self.assertAlmostEqual(float(ret[-1]), 1.0, places=6)


class PpoGoldenTest(unittest.TestCase):
    def _inputs(self, n=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        old = torch.randn(n, generator=g) - 2.0
        new = old + 0.05 * torch.randn(n, generator=g)
        adv = torch.randn(n, generator=g)
        ret = torch.randn(n, generator=g)
        val = torch.randn(n, generator=g)
        return new, old, adv, ret, val

    def test_policy_loss_matches_golden(self):
        new, old, adv, ret, val = self._inputs()
        mask = torch.ones(6, dtype=torch.bool)
        got = ppo_loss.ppo_terms(new, old, adv, ret, val,
                                 actor_mask=mask, value_mask=mask)
        want = golden.ppo_policy_loss(new.tolist(), old.tolist(), adv.tolist())
        abs_d, rel_d, ok = golden.compare(float(want), float(got.policy))
        self.assertTrue(ok, f"policy loss delta abs={abs_d:.3e} rel={rel_d:.3e}")

    def test_value_loss_matches_golden(self):
        new, old, adv, ret, val = self._inputs()
        mask = torch.ones(6, dtype=torch.bool)
        got = ppo_loss.ppo_terms(new, old, adv, ret, val,
                                 actor_mask=mask, value_mask=mask)
        want = golden.value_loss(val.tolist(), ret.tolist())
        abs_d, rel_d, ok = golden.compare(float(want), float(got.value))
        self.assertTrue(ok, f"value loss delta abs={abs_d:.3e} rel={rel_d:.3e}")

    def test_ratio_is_one_before_update(self):
        """With identical weights the ratio must be 1 and KL exactly 0."""
        n = 8
        lp = torch.linspace(-3.0, -1.0, n)
        adv = torch.ones(n)
        ret = torch.zeros(n)
        val = torch.zeros(n)
        mask = torch.ones(n, dtype=torch.bool)
        got = ppo_loss.ppo_terms(lp, lp, adv, ret, val,
                                 actor_mask=mask, value_mask=mask)
        self.assertAlmostEqual(got.approx_kl, 0.0, places=9)
        self.assertAlmostEqual(got.clip_fraction, 0.0, places=9)
        # L_pi = -mean(A) = -1
        self.assertAlmostEqual(float(got.policy), -1.0, places=6)

    def test_actor_and_value_denominators_are_separate(self):
        """Actor loss must ignore value-only rows and vice versa."""
        lp = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        adv = torch.tensor([1.0, 1.0, 100.0, 100.0])
        ret = torch.zeros(4)
        val = torch.zeros(4)
        actor_mask = torch.tensor([True, True, False, False])
        value_mask = torch.tensor([True, True, True, True])
        got = ppo_loss.ppo_terms(lp, lp, adv, ret, val,
                                 actor_mask=actor_mask, value_mask=value_mask)
        self.assertEqual(got.actor_rows, 2)
        self.assertEqual(got.value_rows, 4)
        # -mean([1,1]) = -1 regardless of the big masked advantages
        self.assertAlmostEqual(float(got.policy), -1.0, places=6)

    def test_singleton_rows_do_not_affect_actor_loss(self):
        lp = torch.tensor([-1.0, -2.0])
        adv = torch.tensor([1.0, 999.0])
        ret = torch.zeros(2)
        val = torch.zeros(2)
        actor_mask = torch.tensor([True, False])
        value_mask = torch.tensor([True, True])
        got = ppo_loss.ppo_terms(lp, lp, adv, ret, val,
                                 actor_mask=actor_mask, value_mask=value_mask)
        self.assertAlmostEqual(float(got.policy), -1.0, places=6)

    def test_no_actor_rows_is_safe(self):
        lp = torch.tensor([-1.0, -2.0])
        got = ppo_loss.ppo_terms(lp, lp, torch.zeros(2), torch.zeros(2),
                                 torch.zeros(2),
                                 actor_mask=torch.zeros(2, dtype=torch.bool),
                                 value_mask=torch.ones(2, dtype=torch.bool))
        self.assertEqual(got.actor_rows, 0)
        self.assertTrue(torch.isfinite(got.total))

    def test_trailing_partial_minibatch_not_dropped(self):
        slices = list(ppo_loss.minibatch_slices(70, 64))
        covered = sorted(int(i) for s in slices for i in s)
        self.assertEqual(covered, list(range(70)))

    def test_microbatch_weights_sum_to_one(self):
        aw, vw = ppo_loss.microbatch_weights([10, 5, 3], [12, 6, 2])
        self.assertAlmostEqual(sum(aw), 1.0, places=9)
        self.assertAlmostEqual(sum(vw), 1.0, places=9)

    def test_lear_count_schedule(self):
        self.assertAlmostEqual(ppo_loss.learning_rate(0), 1e-5, places=12)
        self.assertAlmostEqual(ppo_loss.learning_rate(250_000), 3e-4, places=12)
        self.assertAlmostEqual(ppo_loss.learning_rate(100_000_000), 3e-5, places=12)
        # monotone increasing through warmup
        self.assertLess(ppo_loss.learning_rate(0), ppo_loss.learning_rate(125_000))


class GoldenNegativeControls(unittest.TestCase):
    """The comparison must DETECT wrong values, not always agree."""

    def test_detects_perturbed_policy_loss(self):
        new = torch.tensor([-1.0, -1.5])
        old = torch.tensor([-1.1, -1.6])
        adv = torch.tensor([1.0, -1.0])
        want = golden.ppo_policy_loss(new.tolist(), old.tolist(), adv.tolist())
        for delta in (1e-1, 1e-2, 1e-3):
            abs_d, _, ok = golden.compare(float(want), float(want) + delta)
            self.assertFalse(ok, f"comparison accepted a {delta} perturbation")

    def test_detects_mispaired_advantages(self):
        # Under ratio==1 the objective is -mean(A) in expectation, but the rows
        # are NOT symmetric once clipping is active, so a real mispairing must
        # change the value. Use a ratio well outside the clip band.
        # Row 0: ratio = exp(1.0) ~ 2.72 (clipped).
        # Row 1: ratio = exp(0.0) = 1.0 (NOT clipped).
        new = torch.tensor([-0.5, -1.0])
        old = torch.tensor([-1.5, -1.0])
        adv = torch.tensor([1.0, -1.0])
        a = golden.ppo_policy_loss(new.tolist(), old.tolist(), adv.tolist())
        b = golden.ppo_policy_loss(new.tolist(), old.tolist(), [-1.0, 1.0])
        _, _, ok = golden.compare(float(a), float(b))
        self.assertFalse(ok, f"mispairing not detected: {a} vs {b}")

    def test_symmetric_advantages_are_genuinely_equal(self):
        """Documents WHY the naive swap test cannot discriminate.

        With ratio == 1 and symmetric +/- advantages the clipped surrogate is
        -mean(A) == 0 whichever order they appear in. This is correct
        mathematics; the discriminating test above uses a clipped ratio.
        """
        new = torch.tensor([-1.0, -1.0])
        old = torch.tensor([-1.0, -1.0])
        adv = torch.tensor([1.0, -1.0])
        a = golden.ppo_policy_loss(new.tolist(), old.tolist(), adv.tolist())
        b = golden.ppo_policy_loss(new.tolist(), old.tolist(), [-1.0, 1.0])
        self.assertAlmostEqual(float(a), float(b), places=9)

    def test_detects_wrong_gae_lambda(self):
        g_adv, _ = golden.gae([0.0] * 3, [0.0, 0.0, 1.0], [False, False, True])
        ours, _ = gae.seat_gae([0.0] * 3, [0.0, 0.0, 1.0], [False, False, True],
                               gae_lambda=0.5)
        _, _, ok = golden.compare(g_adv[0], float(ours[0]))
        self.assertFalse(ok, "wrong lambda was not detected")


if __name__ == "__main__":
    unittest.main()
