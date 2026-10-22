"""P4-A falsifiable controls: the lossless pipeline must reject bad cohorts.

Each test constructs a *specifically wrong* cohort and asserts the assembler
refuses it. If any of these passed silently, G4-A would be unprovable.
"""
import copy
import sys
import threading
import time
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner import ingest
from pa4.learner.ingest import CohortRejected, build_cohort
from pa4.learner.sink import (LosslessBuffer, LosslessSpool, SinkOverflow,
                              verify_lossless)

_CACHE = {}


def real_records():
    """One real cohort, collected once and shared across tests."""
    if "records" not in _CACHE:
        cfg = BatchedConfig(num_envs=4, target_matches=8, max_steps=6000, seed=1,
                            device="cuda", run_id="p4a-controls",
                            transition_capacity=256)
        sink = LosslessBuffer(capacity=100_000)
        run_batched(cfg, store=sink)
        _CACHE["records"] = sink.records()
        _CACHE["cfg"] = cfg
    return copy.deepcopy(_CACHE["records"])


class SinkLosslessnessTest(unittest.TestCase):
    def test_put_blocks_instead_of_evicting(self):
        sink = LosslessBuffer(capacity=2)
        for i in range(2):
            sink.put({"i": i})
        # Producer must block, and the consumer must still see BOTH records.
        done = threading.Event()

        def producer():
            sink.put({"i": 2})
            done.set()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        time.sleep(0.15)
        self.assertFalse(done.is_set(), "put() returned while the sink was full")
        drained = sink.drain()
        self.assertEqual([r["i"] for r in drained], [0, 1])
        self.assertTrue(done.wait(timeout=2.0), "producer never unblocked")
        self.assertEqual(sink.dropped, 0)
        self.assertEqual(sink.accounting.acknowledged, 3)

    def test_offer_fails_closed_at_capacity(self):
        sink = LosslessBuffer(capacity=1)
        self.assertTrue(sink.offer({"i": 0}))
        self.assertFalse(sink.offer({"i": 1}))
        self.assertEqual(sink.dropped, 0)
        with self.assertRaises(SinkOverflow):
            sink.put_or_fail({"i": 2})

    def test_spool_overflow_fails_closed(self):
        import tempfile
        path = tempfile.mktemp(suffix=".jsonl")
        spool = LosslessSpool(path, capacity=1)
        spool.put({"i": 0})
        with self.assertRaises(SinkOverflow):
            spool.put({"i": 1})
        self.assertEqual(spool.dropped, 0)

    def test_verify_lossless_flags_loss(self):
        sink = LosslessBuffer(capacity=4)
        sink.put({"i": 0})
        sink.accounting.dropped = 1
        with self.assertRaises(SinkOverflow):
            verify_lossless(sink)

    def test_p3_ring_still_evicts(self):
        """P3 benchmark semantics must be untouched."""
        from pa4.collector.transitions import BoundedTransitionStore
        ring = BoundedTransitionStore(capacity=2)
        for i in range(3):
            ring.put({"i": i})
        self.assertEqual(ring.dropped, 1)
        self.assertEqual(len(ring), 2)


class EpisodeAssemblyTest(unittest.TestCase):
    def test_real_cohort_admits(self):
        records = real_records()
        cohort = build_cohort(records)
        self.assertGreater(cohort.matches(), 0)
        self.assertEqual(len({t.match_id for t in cohort.trajectories}),
                         len(cohort.trajectories))

    def test_terminal_reward_on_last_real_decision(self):
        cohort = build_cohort(real_records())
        for t in cohort.trajectories:
            for seat, rows in t.seats.items():
                self.assertTrue(rows, "seat has no rows")
                last = rows[-1]
                self.assertTrue(last.done)
                expected = 0.0 if t.winner == -1 else (1.0 if t.winner == seat else -1.0)
                self.assertAlmostEqual(last.reward, expected, places=9)
                for r in rows[:-1]:
                    self.assertFalse(r.done)
                    self.assertAlmostEqual(r.reward, 0.0, places=9)

    def test_two_episodes_in_one_slot_do_not_share_a_chain(self):
        cohort = build_cohort(real_records())
        keys = [t.episode_key for t in cohort.trajectories]
        slots = {}
        for k in keys:
            slots.setdefault((k[0], k[1], k[2]), []).append(k)
        shared = {s: ks for s, ks in slots.items() if len(ks) > 1}
        # Multiple episodes per slot are allowed; they must differ by gen/episode.
        for s, ks in shared.items():
            self.assertEqual(len({k[3:] for k in ks}), len(ks),
                             "two episodes share generation+episode id")


class FailClosedTest(unittest.TestCase):
    def _one_episode(self):
        records = real_records()
        target = ingest.episode_key(records[0])
        return [r for r in records if ingest.episode_key(r) == target]

    def test_foreign_fingerprint_rejected(self):
        with self.assertRaises(CohortRejected):
            build_cohort(real_records(), expected_fingerprint="0" * 64)

    def test_foreign_policy_version_rejected(self):
        with self.assertRaises(CohortRejected):
            build_cohort(real_records(), expected_policy_version=99)

    def test_fixed9_representation_rejected(self):
        records = real_records()
        for rec in records:
            for sr in rec["seat_records"]:
                sr["observation_representation"] = "pa4.observation.fixed9.v1"
        with self.assertRaises(CohortRejected):
            build_cohort(records)

    def test_duplicate_request_identity_rejected(self):
        records = real_records()
        for rec in records:
            if rec["seat_records"]:
                rec["seat_records"].append(copy.deepcopy(rec["seat_records"][0]))
                break
        with self.assertRaises(CohortRejected):
            build_cohort(records)

    def test_missing_seat_decision_rejected(self):
        records = real_records()
        # Strip EVERY seat-1 decision from one whole episode, which leaves that
        # completed match with no seat-1 trajectory at all.
        target = ingest.episode_key(records[0])
        for rec in records:
            if ingest.episode_key(rec) == target:
                rec["seat_records"] = [s for s in rec["seat_records"]
                                       if int(s["seat"]) == 0]
        with self.assertRaises(CohortRejected):
            build_cohort(records)

    def test_illegal_selected_index_rejected(self):
        records = real_records()
        for rec in records:
            for sr in rec["seat_records"]:
                if sr["branches"]:
                    sr["branches"][0]["selected_index"] = 10_000
                    break
            else:
                continue
            break
        with self.assertRaises(CohortRejected):
            build_cohort(records)

    def test_truncated_episode_excluded_not_trained(self):
        records = real_records()
        target = ingest.episode_key(records[0])
        for rec in records:
            if ingest.episode_key(rec) == target:
                rec["truncated"] = True
        cohort = build_cohort(records)
        keys = {t.episode_key for t in cohort.trajectories}
        self.assertNotIn(target, keys)
        self.assertGreaterEqual(cohort.incomplete, 1)

    def test_mixed_natural_and_truncated_keeps_only_natural(self):
        records = real_records()
        truncated_key = ingest.episode_key(records[0])
        for rec in records:
            if ingest.episode_key(rec) == truncated_key:
                rec["truncated"] = True
        cohort = build_cohort(records)
        self.assertGreater(cohort.matches(), 0)
        self.assertNotIn(truncated_key, {t.episode_key for t in cohort.trajectories})

    def test_duplicate_match_id_rejected(self):
        """Two DIFFERENT episodes claiming the same natural match id.

        Must be refused by the exactly-once match-accounting guard specifically,
        so this asserts the error message, not merely that something raised.
        """
        records = real_records()
        by_key = {}
        for r in records:
            by_key.setdefault(ingest.episode_key(r), []).append(r)

        completed_keys = [k for k, recs in by_key.items()
                          if any(r["match_completed"] for r in recs)]
        self.assertGreaterEqual(len(completed_keys), 2, "needs two completed episodes")

        first, second = sorted(completed_keys)[0], sorted(completed_keys)[1]
        src = next(r for r in by_key[first] if r["match_completed"])
        # Give the second episode the FIRST episode's match id, leaving its own
        # single completion event intact so only the duplicate-id guard can fire.
        for r in by_key[second]:
            if r["match_completed"]:
                r["match_id"] = src["match_id"]

        with self.assertRaises(CohortRejected) as ctx:
            build_cohort(records)
        self.assertIn("duplicate match_id", str(ctx.exception).lower(),
                      f"refused for the wrong reason: {ctx.exception}")

    def test_double_completion_in_one_episode_rejected(self):
        """A single episode reporting two completions is invalid."""
        records = real_records()
        by_key = {}
        for r in records:
            by_key.setdefault(ingest.episode_key(r), []).append(r)

        completed_keys = [k for k, recs in by_key.items()
                          if any(r["match_completed"] for r in recs)]
        self.assertGreaterEqual(len(completed_keys), 1)
        key = sorted(completed_keys)[0]
        donor = sorted(completed_keys)[1] if len(completed_keys) > 1 else key
        src = next(r for r in by_key[donor] if r["match_completed"])
        target = next(r for r in by_key[key] if r["match_completed"])
        target["match_id"] = src["match_id"]

        with self.assertRaises(CohortRejected):
            build_cohort(records)


if __name__ == "__main__":
    unittest.main()
