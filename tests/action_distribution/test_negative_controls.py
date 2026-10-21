"""G1-E and G1-A/B negative controls requested by read-only review.

These are falsification tests: they fail loudly when an invariant breaks.
"max K=18, overflow=False" does not test overflow because 18 is far below 64;
these tests force the boundary.

Uses stdlib unittest (P1 must not change the frozen dependency lock).
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402

from pa4.actions.conditional_policy import (  # noqa: E402
    CANDIDATE_CAPACITY,
    ConditionalPolicy,
    build_branch_table,
)
from pa4.actions.native_reference import NativeReference  # noqa: E402


class OverflowBoundTest(unittest.TestCase):
    """G1-E: capacity is a hard wall, not a silent slice."""

    @classmethod
    def setUpClass(cls):
        from agent.types.observation import ObservationLayout

        cls.layout = ObservationLayout()
        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        cls.slot, cls.gen = h[0]

    def test_capacity_is_64(self):
        self.assertEqual(CANDIDATE_CAPACITY, 64)

    def test_exactly_capacity_is_accepted(self):
        cands = [(0, 0, 255, 0, i % 6, 0) for i in range(CANDIDATE_CAPACITY)]
        table = build_branch_table([cands], self.layout)
        self.assertEqual(int(table.mask.sum()), CANDIDATE_CAPACITY)

    def test_one_over_capacity_raises(self):
        """65 legal candidates MUST raise, never truncate to 64."""
        cands = [(0, 0, 255, 0, i % 6, 0) for i in range(CANDIDATE_CAPACITY + 1)]
        with self.assertRaises(RuntimeError) as ctx:
            build_branch_table([cands], self.layout)
        msg = str(ctx.exception).lower()
        self.assertIn("capacity", msg)
        self.assertIn("truncat", msg)

    def test_overflow_is_not_silent_through_the_policy(self):
        policy = ConditionalPolicy(seed=20261006)
        h = self.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        slot, gen = h[0]
        obs = self.ref.observe_batch([(slot, gen)], [0]).to("cpu")
        enc = policy.encode(obs)
        cands = [(0, 0, 255, 0, i % 6, 0) for i in range(CANDIDATE_CAPACITY + 1)]
        with self.assertRaises(RuntimeError):
            policy.select_branch(enc, cands)

    def test_real_max_k_is_below_capacity(self):
        h = self.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        slot, gen = h[0]
        k = len(self.ref.candidates(slot, gen, 0, []))
        self.assertLess(k, CANDIDATE_CAPACITY)


class IllegalSelectionTest(unittest.TestCase):
    """G1-A negative control.

    Sampling cannot draw an illegal index because the mask is applied before
    softmax, which alone makes "illegal_selected == 0" tautological. Here an
    illegal position is pushed in directly and must sit at the FP32 floor with
    exactly zero probability.
    """

    @classmethod
    def setUpClass(cls):
        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        cls.slot, cls.gen = h[0]
        cls.obs = cls.ref.observe_batch([(cls.slot, cls.gen)], [0]).to("cpu")
        cls.policy = ConditionalPolicy(seed=20261006)

    def test_illegal_slots_have_floor_logprob_and_zero_probability(self):
        from agent.model.scorer import masked_log_softmax

        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        enc = self.policy.encode(self.obs)
        table = build_branch_table([cands], self.policy.layout)
        with torch.no_grad():
            ev = self.policy.model.evaluate_encoded(
                enc, table, selected=torch.tensor([[0]])
            )
        lp = masked_log_softmax(ev.logits[0, 0], table.mask[0, 0])
        k = len(cands)
        floor = torch.finfo(torch.float32).min
        self.assertTrue(bool((lp[k:] <= floor).all()), "illegal slot above FP32 floor")
        self.assertTrue(bool((lp[k:].exp() == 0).all()), "illegal slot has nonzero probability")

    def test_policy_never_selects_beyond_k(self):
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        enc = self.policy.encode(self.obs)
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            idx, lp, _k, _table = self.policy.select_branch(enc, cands, g)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(cands), f"draw {seed} selected an illegal index")
            self.assertTrue(torch.isfinite(torch.tensor(lp)))


class HiddenStateCarryTest(unittest.TestCase):
    """G1-B: level-1 scoring must depend on WHICH action level 0 chose.

    A single-path test cannot detect a silent reset to initial_state().
    This compares two different level-0 choices against the SAME level-1 table.
    """

    @classmethod
    def setUpClass(cls):
        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        cls.slot, cls.gen = h[0]
        cls.obs = cls.ref.observe_batch([(cls.slot, cls.gen)], [0]).to("cpu")
        cls.policy = ConditionalPolicy(seed=20261006)

    def _advance(self, enc, table0, pick):
        fresh = self.policy.model.scorer.initial_state(
            enc.tokens.shape[0], device=enc.tokens.device, dtype=torch.float32
        )
        keys = self.policy.model.scorer.build_keys(
            enc.tokens, table0.action_ids[:, 0], table0.entity_token[:, 0],
            table0.move_token[:, 0],
        )
        sel_index = torch.full((1, 1, keys.shape[-1]), pick, dtype=torch.long)
        selected = torch.gather(keys, 1, sel_index).squeeze(1)
        return self.policy.model.scorer.advance_keys(selected, fresh)

    def test_level1_logprob_depends_on_level0_choice(self):
        enc = self.policy.encode(self.obs)
        c0 = self.ref.candidates(self.slot, self.gen, 0, [])
        self.assertGreaterEqual(len(c0), 2)

        values = []
        for pick in (0, 1):
            table0 = build_branch_table([c0], self.policy.layout)
            hidden = self._advance(enc, table0, pick)
            c1 = self.ref.candidates(self.slot, self.gen, 0, [c0[pick]])
            table1 = build_branch_table([c1], self.policy.layout)
            with torch.no_grad():
                step1 = self.policy.model.sample_level_step(
                    enc, hidden, table1, temperature=1.0,
                    generator=torch.Generator().manual_seed(11),
                )
            values.append(float(step1.logprob[0]))

        self.assertNotAlmostEqual(
            values[0], values[1], places=6,
            msg="level-1 logprob identical for different level-0 picks: hidden may be reset",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
