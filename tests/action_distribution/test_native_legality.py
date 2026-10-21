"""G1-A: real Rust legality.

Preview duplicate exclusion, Normal/Mega prefix dependence, signed targets
unclamped, and no silent candidate truncation.

Uses stdlib unittest: the P1 frozen lock has no test-only dependency and P1 must
not change the dependency stack.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

from pa4.actions.native_reference import NativeReference  # noqa: E402


class NativeLegalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = NativeReference()
        cls.ref.verify_contract()

    def _preview(self):
        h = self.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        return h[0]

    def test_observation_contract_is_pinned(self):
        c = self.ref.observation_contract
        self.assertEqual(c["schema_version"], 2)
        self.assertEqual(c["observation_tokens"], 96)
        self.assertEqual(c["fixed_bytes"], 41986)
        self.assertEqual(
            c["contract_hash"],
            "dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35",
        )

    def test_preview_has_four_branches(self):
        slot, gen = self._preview()
        req = self.ref.request(slot, gen, 0)
        self.assertEqual(req.kind_label, "Preview")
        self.assertEqual(req.branch_count, 4)
        self.assertTrue(req.produces_actor_row)

    def test_preview_duplicate_pick_excluded(self):
        """A member picked at level j must disappear from later levels."""
        slot, gen = self._preview()
        prefix = []
        k_prev = None
        for _ in range(4):
            cands = self.ref.candidates(slot, gen, 0, prefix)
            self.assertTrue(cands, "level must have legal candidates")
            if k_prev is not None:
                self.assertEqual(len(cands), k_prev - 1,
                                 f"expected one fewer: {k_prev} -> {len(cands)}")
            for chosen in prefix:
                self.assertNotIn(chosen, cands, "already-picked action reappeared")
            prefix.append(cands[0])
            k_prev = len(cands)

    def test_candidates_are_engine_ordered_and_total(self):
        slot, gen = self._preview()
        cands = self.ref.candidates(slot, gen, 0, [])
        self.assertEqual(len(cands), 6)
        self.assertEqual(sorted(a[4] for a in cands), [0, 1, 2, 3, 4, 5])

    def test_no_overflow_against_padding(self):
        slot, gen = self._preview()
        self.assertLessEqual(len(self.ref.candidates(slot, gen, 0, [])), 64)

    def _to_normal(self):
        slot, gen = self._preview()
        prefix = []
        for _ in range(4):
            prefix.append(self.ref.candidates(slot, gen, 0, prefix)[0])
        self.ref.step([(slot, gen, [(0, prefix), (1, prefix)])])
        return slot, gen

    def test_signed_targets_not_clamped(self):
        slot, gen = self._to_normal()
        b0 = self.ref.candidates(slot, gen, 0, [])
        targets = {a[3] for a in b0}
        self.assertTrue(any(t < 0 for t in targets), f"negative target missing: {targets}")
        self.assertGreaterEqual(min(targets), -2)

    def test_mega_resource_is_prefix_dependent(self):
        slot, gen = self._to_normal()
        b0 = self.ref.candidates(slot, gen, 0, [])
        saw = False
        for cand in b0:
            b1 = self.ref.candidates(slot, gen, 0, [cand])
            if any(a[5] == 1 for a in b1):
                saw = True
                break
        self.assertTrue(saw, "no branch-1 Mega availability found")

    def test_wait_and_finished_produce_no_actor_row(self):
        from pa4.actions.native_reference import POLICY_REQUEST_KINDS, REQUEST_KIND_LABELS

        self.assertNotIn(3, POLICY_REQUEST_KINDS)
        self.assertNotIn(4, POLICY_REQUEST_KINDS)
        self.assertEqual(REQUEST_KIND_LABELS[3], "Wait")
        self.assertEqual(REQUEST_KIND_LABELS[4], "Finished")


if __name__ == "__main__":
    unittest.main(verbosity=2)
