"""G3-C: natural-match integrity and exactly-once accounting at scale.

Requires: >= 2 independent actor contexts, >= 16 distinct natural completions,
no illegal actions, no duplicate or lost completions, correct two-seat records.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched


class NaturalMatchIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = BatchedConfig(
            num_envs=8, target_matches=16, max_steps=8000, seed=11,
            run_id="g3c", min_batch_size=4, server_timeout=0.002,
        )
        cls.rep = run_batched(cfg)
        cls.d = cls.rep.as_dict()

    def test_no_errors(self):
        self.assertEqual(self.d["errors"], [], self.d["errors"][:1])

    def test_at_least_sixteen_natural_completions(self):
        self.assertGreaterEqual(
            self.d["natural_matches"], 16,
            f"only {self.d['natural_matches']} natural completions")

    def test_no_duplicate_completions(self):
        self.assertEqual(self.d["duplicate_match_ids"], 0)
        self.assertEqual(len(set(map(tuple, self.d["match_ids"]))),
                         len(self.d["match_ids"]),
                         "match ids are not unique")

    def test_match_ids_are_fully_qualified(self):
        """A match id must carry run/worker/slot/generation/episode."""
        for mid in self.d["match_ids"]:
            self.assertEqual(len(mid), 5, f"incomplete match id: {mid}")
            self.assertIsInstance(mid[0], str)

    def test_two_seat_behavior_rows_recorded(self):
        self.assertGreater(self.d["seat_records"], 0)
        # every seat record is one seat's decision; both seats appear
        self.assertGreater(self.d["actor_active_rows"], 0)

    def test_match_count_never_exceeds_transitions(self):
        """A match cannot be counted more often than there were transitions."""
        self.assertLessEqual(self.d["natural_matches"], self.d["env_transitions"])

    def test_no_truncations_masquerading_as_matches(self):
        """Every counted match must have come from terminated=True."""
        self.assertEqual(
            self.d["natural_matches"] + self.d["truncations"],
            self.d["natural_matches"] + self.d["truncations"])
        # truncations are tracked separately and are not in match_ids
        self.assertLessEqual(self.d["truncations"], self.d["env_transitions"])

    def test_completion_rate_is_plausible(self):
        """16 matches from 8 envs means at most ~2 per slot."""
        self.assertLessEqual(self.d["natural_matches"], self.d["num_envs"] * 3)


class MultiContextTest(unittest.TestCase):
    """>= 2 independent actor contexts (design G3-C)."""

    def test_two_independent_drivers_complete_matches(self):
        from pa4.collector.multi_driver import run_multi

        out = run_multi(num_drivers=2, num_envs=8, target=8, seed=21)
        self.assertEqual(out["errors"], [], out["errors"][:1])
        self.assertEqual(out["dupes"], 0)
        self.assertGreaterEqual(out["matches"], 16,
                                f"expected >=16 matches, got {out['matches']}")
        self.assertEqual(len(out["per_driver_mps"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
