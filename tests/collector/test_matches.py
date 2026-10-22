"""G3-C: natural-match integrity and exactly-once accounting at scale.

Requires: >= 2 independent actor contexts, >= 16 distinct natural completions,
no illegal actions, no duplicate or lost completions, correct two-seat records.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch
from tensordict import TensorDict

from pa4.collector.batched_core import BatchedConfig, BatchedReport
from pa4.collector.run_batched import account_step, run_batched
from pa4.collector.transitions import BoundedTransitionStore
from pa4.env.native_env import SlotState


class NaturalMatchIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = BatchedConfig(
            num_envs=8, target_matches=16, max_steps=8000, seed=11,
            run_id="g3c", min_batch_size=4, server_timeout=0.002,
            warmup_steps=4, transition_capacity=64,
        )
        cls.store = BoundedTransitionStore(cfg.transition_capacity)
        cls.rep = run_batched(cfg, store=cls.store)
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
        self.assertGreater(self.d["actor_active_rows"], 0)
        # both seats really decided in this run, not just one
        self.assertGreater(self.d["seats_by_seat"][0], 0,
                           "seat 0 produced no decision records")
        self.assertGreater(self.d["seats_by_seat"][1], 0,
                           "seat 1 produced no decision records")

    def test_match_count_never_exceeds_transitions(self):
        """A match cannot be counted more often than there were transitions."""
        self.assertLessEqual(self.d["natural_matches"], self.d["env_transitions"])

    def test_no_truncations_masquerading_as_matches(self):
        """Every counted match must have come from terminated=True.

        The rule itself is proven directly in AccountingUnitTest; here we check
        the live run's ledger is internally consistent and that this run saw no
        truncation that could have been mistaken for a match.
        """
        self.assertEqual(self.d["natural_matches"], len(self.d["match_ids"]))
        self.assertEqual(len(set(map(tuple, self.d["match_ids"]))),
                         len(self.d["match_ids"]))
        self.assertEqual(self.d["errors"], [], self.d["errors"][:1])
        self.assertEqual(self.d["truncations"], 0,
                         "a truncation appeared in this run; the exclusion path "
                         "must be re-verified")

    def test_completion_rate_is_plausible(self):
        """16 matches from 8 envs means at most ~2 per slot."""
        self.assertLessEqual(self.d["natural_matches"], self.d["num_envs"] * 3)


class MultiContextTest(unittest.TestCase):
    """>= 2 independent actor worker processes (design G3-C)."""

    @classmethod
    def setUpClass(cls):
        from pa4.collector.multi_driver import run_multi

        cls.out = run_multi(num_drivers=2, num_envs=8, target=8, seed=21,
                            central=True)

    def test_two_independent_drivers_complete_matches(self):
        out = self.out
        self.assertTrue(out["central"])
        self.assertEqual(out["errors"], [], out["errors"][:1])
        self.assertEqual(out["dupes"], 0)
        self.assertGreaterEqual(out["matches"], 16,
                                f"expected >=16 matches, got {out['matches']}")
        self.assertEqual(len(out["per_driver_mps"]), 2)

    def test_one_forward_mixes_two_worker_processes(self):
        """G3-B: a single forward must contain requests from different workers."""
        out = self.out
        self.assertGreaterEqual(
            out["max_worker_ids_per_forward"], 2,
            f"no forward mixed workers: {out['service_stats'].get('forward_run_ids')}")
        self.assertGreaterEqual(out["cross_worker_forwards"], 1)
        self.assertGreaterEqual(out["transitions_stored"], 1)
        self.assertGreaterEqual(out["steady_matches"], 1)


class TransitionRetentionTest(unittest.TestCase):
    """P3-C: bounded PA4ActorTransitionV1 records with replay necessities."""

    REQUIRED = ("run_id", "worker_id", "native_slot", "native_generation",
                "episode_id", "step_index", "request_ids", "policy_version",
                "policy_weight_fingerprint", "observation_contract_hash",
                "real_p2_action_packet", "reward", "done", "terminated",
                "truncated", "match_completed", "winner", "request_revised",
                "match_id", "seat_records")

    @classmethod
    def setUpClass(cls):
        cfg = BatchedConfig(
            num_envs=2, target_matches=2, max_steps=400, seed=31,
            run_id="g3c-transitions", min_batch_size=2, server_timeout=0.002,
            warmup_steps=1, transition_capacity=8,
        )
        cls.store = BoundedTransitionStore(cfg.transition_capacity)
        cls.rep = run_batched(cfg, store=cls.store)
        cls.d = cls.rep.as_dict()

    def test_records_were_retained(self):
        self.assertEqual(self.d["errors"], [], self.d["errors"][:1])
        self.assertGreater(self.d["transitions_stored"], 0)
        self.assertLessEqual(len(self.store), self.store.capacity)
        self.assertEqual(self.d["transitions_retained"], len(self.store))

    def test_record_has_every_replay_field(self):
        rec = next(iter(self.store))
        for key in self.REQUIRED:
            self.assertIn(key, rec)
        self.assertEqual(len(rec["policy_weight_fingerprint"]), 64,
                         "weight fingerprint is not a sha256")
        self.assertEqual(rec["policy_weight_fingerprint"],
                         self.rep.policy_weight_fingerprint)
        self.assertEqual(rec["observation_contract_hash"],
                         "dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35")

    def test_pre_action_observation_and_candidates_stored(self):
        rec = next(iter(self.store))
        self.assertTrue(rec["seat_records"])
        for seat in rec["seat_records"]:
            self.assertEqual(set(seat["observation"]),
                             {"token_mask", "categories", "category_known", "floats",
                              "float_known", "flags", "flag_known", "role_ids",
                              "side_ids"})
            for leaf in seat["observation"].values():
                self.assertEqual(leaf.device.type, "cpu")
            self.assertTrue(seat["branches"])
            for b in seat["branches"]:
                self.assertIn(b["selected_index"], range(len(b["candidates"])))
                self.assertGreaterEqual(b["branch_k"], 1)

    def test_action_packet_is_the_real_six_key_packet(self):
        rec = next(iter(self.store))
        self.assertEqual(set(rec["real_p2_action_packet"]),
                         {"action", "action_branch_count", "action_present",
                          "action_request_id", "action_episode_id",
                          "action_native_generation"})

    def test_match_records_carry_a_match_id(self):
        recs = list(self.store)
        completed = [r for r in recs if r["match_completed"]]
        for r in completed:
            self.assertIsNotNone(r["match_id"])
            self.assertIn(r["match_id"], self.rep.match_ids)

    def test_stage_timing_and_windows_recorded(self):
        self.assertGreater(self.rep.setup_seconds, 0.0)
        self.assertGreaterEqual(self.rep.steady_seconds, 0.0)
        for key in ("request_info", "views", "encode", "step_levels",
                    "packet", "env_step", "accounting", "reset"):
            self.assertIn(key, self.rep.stage_seconds)
            self.assertGreaterEqual(self.rep.stage_seconds[key], 0.0)

    def test_lost_completion_guard_matches_env(self):
        self.assertEqual(self.rep.env_completed_matches, self.rep.natural_matches)


class BoundedStoreTest(unittest.TestCase):
    def test_drops_oldest_and_counts(self):
        s = BoundedTransitionStore(2)
        s.put({"i": 1})
        s.put({"i": 2})
        s.put({"i": 3})
        self.assertEqual([r["i"] for r in s], [2, 3])
        self.assertEqual((s.stored, s.dropped), (3, 1))
        self.assertEqual(s.summary(), {"capacity": 2, "stored": 3,
                                       "dropped": 1, "retained": 2})

    def test_rejects_zero_capacity(self):
        with self.assertRaises(ValueError):
            BoundedTransitionStore(0)


class AccountingUnitTest(unittest.TestCase):
    """Exactly-once accounting, driven directly with synthetic step outputs.

    A live run with zero truncations cannot prove the exclusion rule, so the rule
    is exercised here on the same `account_step` the collector calls.
    """

    def _out(self, completed=(), truncated=(), revised=()):
        n = 1
        return TensorDict(
            {
                "match_completed": torch.tensor([bool(completed)], dtype=torch.bool),
                "truncated": torch.tensor([bool(truncated)], dtype=torch.bool),
                "request_revised": torch.tensor([bool(revised)], dtype=torch.bool),
            },
            batch_size=[n],
        )

    def _rep(self):
        return BatchedReport(run_id="unit")

    def test_natural_completion_is_counted_once(self):
        rep, seen = self._rep(), set()
        slots = [SlotState(slot=3, generation=7, episode_id=2)]
        account_step(self._out(completed=True), slots, seen, rep, "unit", 5)
        self.assertEqual(rep.natural_matches, 1)
        self.assertEqual(rep.match_ids, [["unit", 5, 3, 7, 2]])
        self.assertEqual(rep.duplicate_match_ids, 0)
        self.assertEqual(rep.truncations, 0)

    def test_repeated_completion_is_a_duplicate_not_a_second_match(self):
        rep, seen = self._rep(), set()
        slots = [SlotState(slot=3, generation=7, episode_id=2)]
        out = self._out(completed=True)
        account_step(out, slots, seen, rep, "unit", 0)
        account_step(out, slots, seen, rep, "unit", 0)
        self.assertEqual(rep.natural_matches, 1, "the same match was counted twice")
        self.assertEqual(rep.duplicate_match_ids, 1)

    def test_truncation_is_never_a_match(self):
        rep, seen = self._rep(), set()
        slots = [SlotState(slot=0, generation=1, episode_id=4)]
        account_step(self._out(truncated=True), slots, seen, rep, "unit", 0)
        self.assertEqual(rep.natural_matches, 0)
        self.assertEqual(rep.match_ids, [])
        self.assertEqual(rep.truncations, 1)

    def test_completed_and_truncated_together_is_an_error(self):
        rep, seen = self._rep(), set()
        slots = [SlotState(slot=0, generation=1, episode_id=4)]
        account_step(self._out(completed=True, truncated=True), slots, seen, rep, "unit", 0)
        self.assertEqual(len(rep.errors), 1)
        self.assertIn("both true", rep.errors[0])

    def test_revision_is_not_a_match(self):
        rep, seen = self._rep(), set()
        slots = [SlotState(slot=0, generation=1, episode_id=4)]
        account_step(self._out(revised=True), slots, seen, rep, "unit", 0)
        self.assertEqual(rep.natural_matches, 0)
        self.assertEqual(rep.revisions, 1)

    def test_revised_rows_are_excluded_from_behaviour_records(self):
        from pa4.collector.transitions import accepted_records

        records = [{"row": 0, "seat": 0}, {"row": 1, "seat": 0}, {"row": 1, "seat": 1}]
        kept = accepted_records(records, {0})
        self.assertEqual([(r["row"], r["seat"]) for r in kept], [(1, 0), (1, 1)])
        self.assertEqual(len(accepted_records(records, set())), 3)

    def test_next_episode_same_slot_is_a_new_match(self):
        rep, seen = self._rep(), set()
        account_step(self._out(completed=True),
                     [SlotState(slot=0, generation=1, episode_id=4)],
                     seen, rep, "unit", 0)
        account_step(self._out(completed=True),
                     [SlotState(slot=0, generation=1, episode_id=5)],
                     seen, rep, "unit", 0)
        self.assertEqual(rep.natural_matches, 2)
        self.assertEqual(rep.duplicate_match_ids, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
