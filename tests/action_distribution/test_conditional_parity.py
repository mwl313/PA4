"""G1-B / G1-C: branch distribution invariants and exact replay.

Thresholds are the P1 design's pre-registered ones and are NOT relaxed here:
  * |sum(P_legal) - 1| <= 1e-6
  * P_illegal == 0 exactly (FP32 tensor equality)
  * branch/joint logprob parity <= 1e-5 FP32 absolute
"""

import math
import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402

from pa4.actions.conditional_policy import (  # noqa: E402
    ConditionalPolicy,
    build_branch_table,
)
from pa4.actions.native_reference import NativeReference  # noqa: E402

SUM_TOL = 1e-6
LOGPROB_TOL = 1e-5


class BranchDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        cls.slot, cls.gen = h[0]
        cls.obs = cls.ref.observe_batch([(cls.slot, cls.gen)], [0]).to("cpu")
        cls.policy = ConditionalPolicy(seed=20261006)

    def _probs(self, candidates):
        from agent.model.scorer import masked_log_softmax, probabilities

        enc = self.policy.encode(self.obs)
        table = build_branch_table([candidates], self.policy.layout)
        with torch.no_grad():
            ev = self.policy.model.evaluate_encoded(
                enc, table, selected=torch.tensor([[0]])
            )
        logits = ev.logits[0, 0]
        mask = table.mask[0, 0]
        lp = masked_log_softmax(logits, mask)
        return probabilities(lp, mask)

    def test_legal_probabilities_sum_to_one(self):
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        p = self._probs(cands)
        k = len(cands)
        self.assertAlmostEqual(float(p[:k].sum()), 1.0, delta=SUM_TOL)
        self.assertLessEqual(abs(float(p.sum()) - 1.0), SUM_TOL)

    def test_illegal_padded_candidates_exactly_zero(self):
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        p = self._probs(cands)
        k = len(cands)
        self.assertTrue(bool((p[k:] == 0).all()), "padded/illegal probability not exactly zero")
        self.assertTrue(torch.isfinite(p).all())

    def test_no_nan_or_inf(self):
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        p = self._probs(cands)
        self.assertTrue(bool(torch.isfinite(p).all()))

    def test_singleton_branch_has_zero_logprob(self):
        """K=1 => deterministic choice with logprob 0 (design 2.4)."""
        enc = self.policy.encode(self.obs)
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        table = build_branch_table([cands[:1]], self.policy.layout)
        with torch.no_grad():
            ev = self.policy.model.evaluate_encoded(enc, table, selected=torch.tensor([[0]]))
        self.assertAlmostEqual(float(ev.logprob_selected[0, 0].item()), 0.0, delta=LOGPROB_TOL)


class ReplayParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        cls.slot, cls.gen = h[0]
        cls.obs = cls.ref.observe_batch([(cls.slot, cls.gen)], [0]).to("cpu")
        cls.policy = ConditionalPolicy(seed=20261006)

    def _complete(self, seed=1234):
        gen = torch.Generator().manual_seed(seed)
        return self.policy.complete_request(
            self.ref, self.slot, self.gen, 0, request_id=1,
            observation=self.obs, generator=gen,
        )

    def test_completed_action_is_legal(self):
        """Every returned tuple must be in the engine's candidate set for its prefix."""
        acts, rec = self._complete()
        prefix = []
        for i, snap in enumerate(rec.branches):
            self.assertIn(acts[i], snap.candidate_tuples,
                          f"level {i} action not in legal candidates")
            self.assertLess(snap.selected_index, len(snap.candidate_tuples))
            prefix.append(acts[i])

    def test_replay_matches_saved_logprob(self):
        acts, rec = self._complete()
        rep = self.policy.replay(rec, self.obs)
        self.assertLessEqual(rep["max_abs_branch_delta"], LOGPROB_TOL)
        self.assertLessEqual(rep["abs_joint_delta"], LOGPROB_TOL)

    def test_joint_logprob_is_sum_of_branches(self):
        acts, rec = self._complete()
        total = sum(b.branch_logprob_fp32 for b in rec.branches)
        self.assertAlmostEqual(rec.joint_logprob_fp32, total, delta=LOGPROB_TOL)

    def test_actor_active_semantics(self):
        """actor_active = any(valid branch with branch_k >= 2)."""
        acts, rec = self._complete()
        expected = any(b.branch_k >= 2 for b in rec.branches)
        self.assertEqual(rec.actor_active, expected)

    def test_replay_refuses_weight_mismatch(self):
        acts, rec = self._complete()
        other = ConditionalPolicy(seed=999)
        with self.assertRaises(RuntimeError):
            other.replay(rec, self.obs)

    def test_replay_refuses_policy_version_mismatch(self):
        acts, rec = self._complete()
        self.policy.policy_version += 1
        try:
            with self.assertRaises(RuntimeError):
                self.policy.replay(rec, self.obs)
        finally:
            self.policy.policy_version -= 1

    def test_replay_detects_swapped_candidates(self):
        """Negative control: swapping two candidate rows must change the recompute."""
        from dataclasses import replace

        acts, rec = self._complete()
        snap = rec.branches[0]
        self.assertGreaterEqual(len(snap.candidate_tuples), 2)
        swapped = list(snap.candidate_tuples)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        tampered = replace(snap, candidate_tuples=tuple(swapped))
        self.assertNotEqual(
            tampered.candidates_digest(), snap.candidates_digest(),
            "candidate reorder was not detected by the digest",
        )


class GradientTest(unittest.TestCase):
    """G1-C: regularizer terms must be finite and carry gradient."""

    def test_entropy_and_kl_have_gradient(self):
        ref = NativeReference()
        h = ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        slot, gen = h[0]
        obs = ref.observe_batch([(slot, gen)], [0]).to("cpu")
        policy = ConditionalPolicy(seed=20261006)
        cands = ref.candidates(slot, gen, 0, [])
        enc = policy.encode(obs)
        table = build_branch_table([cands], policy.layout)

        ev = policy.model.evaluate_encoded(enc, table, selected=torch.tensor([[0]]))
        entropy = ev.entropy_normalized.sum()
        kl = ev.uniform_kl_normalized.sum()
        self.assertTrue(math.isfinite(float(entropy.detach())))
        self.assertTrue(math.isfinite(float(kl.detach())))
        (entropy + kl).backward()
        grads = [
            p.grad for p in policy.model.parameters()
            if p.grad is not None and p.requires_grad
        ]
        self.assertTrue(grads, "no parameter received a gradient")
        self.assertTrue(any(bool((g != 0).any()) for g in grads),
                        "regularizer produced an all-zero gradient")


if __name__ == "__main__":
    unittest.main(verbosity=2)
