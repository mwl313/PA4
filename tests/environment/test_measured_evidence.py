"""G2-B/G2-D evidence tests (F5): every metric the receipt claims is asserted here.

The read-only review correctly found that `illegal_submissions: 0`,
`replacement_observed: true` and the observed request-kind pairs were prose, not
measurements. These tests measure them.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from pa4.env.native_env import (  # noqa: E402
    KIND_FINISHED,
    KIND_LABELS,
    MAX_BRANCHES,
    PA4EnvBase,
)


def play_measured(env, rng_base=1701, max_steps=400):
    """Play a full battle, returning measured counters."""
    env.reset()
    kinds_seen = set()
    steps = 0
    illegal = 0
    revisions = 0
    while steps < max_steps:
        st = env._slots[0]
        kinds = env._kinds(st)
        kinds_seen.add((KIND_LABELS[kinds[0]], KIND_LABELS[kinds[1]]))
        if all(k == KIND_FINISHED for k in kinds):
            return {"finished": True, "terminated": False, "steps": steps,
                    "kinds": kinds_seen, "illegal": illegal, "revisions": revisions}
        packet = env.legal_action_packet(torch.Generator().manual_seed(rng_base + env._seed * 100000 + steps))
        # audit every emitted row against real candidates BEFORE submitting
        for side in (0, 1):
            if not bool(packet["action_present"][0, side]):
                continue
            k = int(packet["action_branch_count"][0, side])
            prefix = []
            for i in range(k):
                cands = env._candidates(st, side, prefix)
                row = tuple(int(v) for v in packet["action"][0, side, i])
                if row not in cands:
                    illegal += 1
                prefix.append(row)
        out = env.step(packet)
        steps += 1
        revisions += int(out["next", "request_revised"][0])
        if bool(out["next", "done"][0, 0]):
            return {
                "finished": True,
                "terminated": bool(out["next", "terminated"][0, 0]),
                "truncated": bool(out["next", "truncated"][0, 0]),
                "winner": int(out["next", "winner"][0]),
                "reward": out["next", "reward"][0].flatten().tolist(),
                "completed": bool(out["next", "match_completed"][0]),
                "steps": steps, "kinds": kinds_seen,
                "illegal": illegal, "revisions": revisions,
            }
        td = out["next"]
    return {"finished": False, "steps": steps, "kinds": kinds_seen,
            "illegal": illegal, "revisions": revisions}


# The exact table RECEIPT.json publishes. If the implementation changes, this
# test fails and the receipt must be re-measured rather than silently drifting.
DOCUMENTED_TRACE = (
    {"seed": 1, "steps": 20, "winner": 0, "reward": [1.0, -1.0]},
    {"seed": 2, "steps": 11, "winner": 1, "reward": [-1.0, 1.0]},
    {"seed": 3, "steps": 19, "winner": 0, "reward": [1.0, -1.0]},
    {"seed": 4, "steps": 14, "winner": 1, "reward": [-1.0, 1.0]},
)
DOCUMENTED_KIND_PAIRS = {
    ("Preview", "Preview"),
    ("Normal", "Normal"),
    ("Replacement", "Replacement"),
    ("Replacement", "Wait"),
    ("Wait", "Replacement"),
}


class MeasuredEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = []
        for seed in (1, 2, 3, 4):
            env = PA4EnvBase(batch_size=1, seed=seed)
            r = play_measured(env)
            r["seed"] = seed
            r["env_completed"] = env.completed_matches
            r["truncated_episodes"] = env._slots[0].truncated_episodes
            cls.results.append(r)

    def test_four_natural_completions_measured(self):
        done = [r for r in self.results if r.get("terminated")]
        self.assertEqual(len(done), 4, self.results)
        for r in done:
            self.assertFalse(r["truncated"])
            self.assertTrue(r["completed"])

    def test_illegal_submissions_is_measured_zero(self):
        total = sum(r["illegal"] for r in self.results)
        self.assertEqual(total, 0,
                         f"measured {total} illegal submissions: {self.results}")

    def test_no_duplicate_completions(self):
        for r in self.results:
            self.assertEqual(r["env_completed"], 1, r)
            self.assertEqual(r["truncated_episodes"], 0, r)

    def test_replacement_observed_measured(self):
        seen = set()
        for r in self.results:
            seen |= r["kinds"]
        self.assertTrue(
            any("Replacement" in pair for pair in seen),
            f"replacement never observed; measured pairs: {sorted(seen)}")

    def test_wait_observed_measured(self):
        seen = set()
        for r in self.results:
            seen |= r["kinds"]
        self.assertTrue(any("Wait" in pair for pair in seen),
                        f"wait never observed; measured pairs: {sorted(seen)}")

    def test_zero_operational_errors(self):
        """A native operational error raises, so reaching here means none occurred."""
        for r in self.results:
            self.assertTrue(r["finished"], r)

    def test_documented_trace_matches_receipt(self):
        """steps_per_game / winners / rewards in RECEIPT.json are measured here."""
        got = [(r["seed"], r["steps"], r.get("winner"), r.get("reward"))
               for r in self.results]
        want = [(d["seed"], d["steps"], d["winner"], d["reward"])
                for d in DOCUMENTED_TRACE]
        self.assertEqual(got, want,
                         "measured trace no longer matches the receipt table")

    def test_no_revisions_measured(self):
        """`revisions_observed: 0` is asserted, not prose."""
        for r in self.results:
            self.assertEqual(r["revisions"], 0, r)

    def test_documented_request_kind_pairs_measured(self):
        """`request_kind_pairs_observed` is asserted as an exact measured set."""
        seen = set().union(*(r["kinds"] for r in self.results))
        self.assertEqual(seen, DOCUMENTED_KIND_PAIRS,
                         f"measured pairs differ from the receipt: {sorted(seen)}")


class InvalidSubmissionTest(unittest.TestCase):
    """F4/F5: an invalid packet must raise, never be counted as a revision."""

    def setUp(self):
        self.env = PA4EnvBase(batch_size=1, seed=1)
        self.env.reset()

    def _packet(self):
        return self.env.legal_action_packet(torch.Generator().manual_seed(3))

    def test_illegal_candidate_rejected_explicitly(self):
        p = self._packet()
        # replace a real action with a tuple that cannot be legal
        p["action"][0, 0, 0] = torch.tensor([255, 255, 255, 127, 255, 255])
        with self.assertRaises(RuntimeError) as ctx:
            self.env.step(p)
        self.assertIn("illegal action", str(ctx.exception))

    def test_wrong_branch_count_rejected(self):
        p = self._packet()
        # shrink a real prefix by one -> no longer the native branch count
        k = int(p["action_branch_count"][0, 0])
        if k > 1:
            p["action_branch_count"][0, 0] = k - 1
            with self.assertRaises(RuntimeError) as ctx:
                self.env.step(p)
            self.assertIn("branch count mismatch", str(ctx.exception))
        else:
            # Not a skip: the receipt claims multi-branch requests are exercised,
            # so a single-branch-only reset would mean the evidence is missing.
            self.fail("first seat has a single-branch request; wrong-branch-count "
                      "rejection was not exercised")

    def test_extra_actions_for_wait_seat_rejected(self):
        """Find a Wait state and submit an action for it."""
        for step in range(60):
            st = self.env._slots[0]
            kinds = self.env._kinds(st)
            if any(k == 3 for k in kinds):
                side = kinds.index(3)
                p = self._packet()
                p["action_present"][0, side] = True
                p["action_branch_count"][0, side] = 1
                with self.assertRaises(RuntimeError):
                    self.env.step(p)
                return
            out = self.env.step(self.env.legal_action_packet(
                torch.Generator().manual_seed(step)))
            if bool(out["next", "done"][0, 0]):
                break
        # Not a skip: `wait_observed: true` is a receipt claim, so failing to
        # reach a Wait state must fail rather than silently reduce coverage.
        self.fail("no Wait state reached in this seed; Wait rejection not exercised")

    def test_stale_generation_rejected(self):
        p = self._packet()
        p["action_native_generation"] = p["action_native_generation"] + 99
        with self.assertRaises(RuntimeError) as ctx:
            self.env.step(p)
        self.assertIn("generation", str(ctx.exception))

    def test_stale_episode_id_rejected(self):
        p = self._packet()
        p["action_episode_id"] = p["action_episode_id"] + 99
        with self.assertRaises(RuntimeError) as ctx:
            self.env.step(p)
        self.assertIn("episode", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
