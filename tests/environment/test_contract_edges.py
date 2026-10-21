"""G2-B/G2-D contract tests raised by read-only review.

1. A native truncation must NEVER be counted as a naturally completed match and
   must never pay a terminal winner reward (PA4-P2-DESIGN-v1.0 L180).
2. Cross-episode staleness: a packet from a previous episode must be rejected,
   even though request_id restarts at (1,1) after reset.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from pa4.env.native_env import KIND_FINISHED, PA4EnvBase  # noqa: E402


class TruncationSemanticsTest(unittest.TestCase):
    """Drives the truncation branch directly, since natural play rarely hits it."""

    def _env_with_synthetic_result(self, term, trunc, winner):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        real_step = env._engine.step_batch

        def fake_step(specs):
            return [(True, term, trunc, winner, None, 4, 4)]

        env._engine = _Proxy(env._engine, fake_step)
        packet = env.legal_action_packet()
        return env, env.step(packet)

    def test_truncation_is_not_a_completed_match(self):
        env, out = self._env_with_synthetic_result(term=False, trunc=True, winner=0)
        self.assertTrue(bool(out["next", "truncated"][0, 0]))
        self.assertFalse(bool(out["next", "terminated"][0, 0]))
        self.assertFalse(bool(out["next", "match_completed"][0]),
                         "truncation must not set match_completed")
        self.assertEqual(env.completed_matches, 0,
                         "truncation must not increment completed_matches")
        self.assertEqual(out["next", "reward"][0].flatten().tolist(), [0.0, 0.0],
                         "truncation must not pay a terminal win reward")
        self.assertEqual(env._slots[0].truncated_episodes, 1)

    def test_termination_is_a_completed_match(self):
        env, out = self._env_with_synthetic_result(term=True, trunc=False, winner=0)
        self.assertTrue(bool(out["next", "terminated"][0, 0]))
        self.assertTrue(bool(out["next", "match_completed"][0]))
        self.assertEqual(env.completed_matches, 1)
        self.assertEqual(out["next", "reward"][0].flatten().tolist(), [1.0, -1.0])

    def test_both_flags_recorded_explicitly(self):
        env, out = self._env_with_synthetic_result(term=True, trunc=True, winner=1)
        # both native flags surface truthfully
        self.assertTrue(bool(out["next", "terminated"][0, 0]))
        self.assertTrue(bool(out["next", "truncated"][0, 0]))
        # terminated dominates for counting, but the raw flags are not hidden
        self.assertTrue(bool(out["next", "match_completed"][0]))


class CrossEpisodeStalenessTest(unittest.TestCase):
    def test_previous_episode_packet_rejected_after_reset(self):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        # take one step so request_ids advance beyond (1,1)
        env.step(env.legal_action_packet())
        stale = env.legal_action_packet()
        stale["action_request_id"] = torch.zeros_like(stale["action_request_id"]) + 1

        # full reset restarts request_ids at (1,1); episode and generation change
        env.reset()
        self.assertEqual(env._slots[0].request_ids, (1, 1))

        # a stale packet carrying an old generation must be rejected
        with self.assertRaises(Exception):
            env.step(stale)


class ConcludedRowTest(unittest.TestCase):
    """A step after the episode ended must re-report how it ACTUALLY ended.

    Regression for review finding N1: the earlier F6 fix set terminated=True
    unconditionally, so a truncated episode became a fabricated natural
    completion (and lost the winner) on the following step.
    """

    def _empty_packet(self, env):
        st = env._slots[0]
        return TensorDict(
            {
                "action": torch.zeros((1, 2, 4, 6), dtype=torch.long),
                "action_branch_count": torch.zeros((1, 2), dtype=torch.long),
                "action_present": torch.zeros((1, 2), dtype=torch.bool),
                "action_request_id": torch.tensor([[st.request_ids[0], st.request_ids[1]]]),
                "action_episode_id": torch.full((1, 2), st.episode_id),
                "action_native_generation": torch.full((1, 2), st.generation),
            },
            batch_size=[1],
        )

    def _env_with_synthetic_result(self, term, trunc, winner):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        env._engine = _Proxy(
            env._engine, lambda specs: [(True, term, trunc, winner, None, 4, 4)])
        env.step(env.legal_action_packet())
        return env

    def test_post_truncation_step_does_not_fabricate_termination(self):
        env = self._env_with_synthetic_result(term=False, trunc=True, winner=0)
        out = env.step(self._empty_packet(env))
        self.assertTrue(bool(out["next", "done"][0, 0]))
        self.assertTrue(bool(out["next", "truncated"][0, 0]),
                        "truncation must survive a post-terminal step")
        self.assertFalse(bool(out["next", "terminated"][0, 0]),
                         "a truncated episode must never become a natural termination")
        self.assertFalse(bool(out["next", "match_completed"][0]))
        self.assertEqual(env.completed_matches, 0)
        self.assertEqual(int(out["next", "winner"][0]), 0)

    def test_post_termination_step_preserves_winner(self):
        env = self._env_with_synthetic_result(term=True, trunc=False, winner=1)
        out = env.step(self._empty_packet(env))
        self.assertTrue(bool(out["next", "done"][0, 0]))
        self.assertTrue(bool(out["next", "terminated"][0, 0]))
        self.assertFalse(bool(out["next", "truncated"][0, 0]))
        self.assertEqual(int(out["next", "winner"][0]), 1,
                         "the winner must not be lost on a post-terminal step")
        self.assertEqual(env.completed_matches, 1)

    def test_concluded_row_rejects_a_new_submission(self):
        env = self._env_with_synthetic_result(term=True, trunc=False, winner=0)
        packet = env.legal_action_packet()
        packet["action_present"][0, 0] = True
        with self.assertRaises(RuntimeError) as ctx:
            env.step(packet)
        self.assertIn("already concluded", str(ctx.exception))


class RevisionPathTest(unittest.TestCase):
    """F4/N2: a genuine native accepted=False is a revision, never a completion."""

    def test_accepted_false_is_a_nonterminal_revision(self):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        env._engine = _Proxy(
            env._engine, lambda specs: [(False, False, False, None, None, 1, 2)])
        st = env._slots[0]
        before_req = st.request_ids
        out = env.step(env.legal_action_packet())
        self.assertTrue(bool(out["next", "request_revised"][0]))
        self.assertFalse(bool(out["next", "done"][0, 0]))
        self.assertFalse(bool(out["next", "terminated"][0, 0]))
        self.assertFalse(bool(out["next", "match_completed"][0]))
        self.assertEqual(out["next", "reward"][0].flatten().tolist(), [0.0, 0.0])
        self.assertEqual(env.completed_matches, 0)
        self.assertEqual(st.revisions, 1)
        self.assertEqual(st.step_index, 0, "a revision is not a native advance")
        self.assertNotEqual(st.request_ids, before_req,
                            "a revision must evict the stale request identity")

    def test_illegal_packet_raises_before_submission(self):
        """Pre-validation means an illegal packet never becomes a 'revision'."""
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        env._engine = _Proxy(
            env._engine, lambda specs: [(False, False, False, None, None, 1, 2)])
        p = env.legal_action_packet()
        p["action"][0, 0, 0] = torch.tensor([255, 255, 255, 127, 255, 255])
        with self.assertRaises(RuntimeError) as ctx:
            env.step(p)
        self.assertIn("illegal action", str(ctx.exception))
        self.assertEqual(env._slots[0].revisions, 0,
                         "an invalid submission must not be counted as a revision")


class RequiredIdentityTest(unittest.TestCase):
    """F2/N5: identity fields are required, not opt-out."""

    def test_packet_without_episode_id_rejected(self):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        p = env.legal_action_packet()
        del p["action_episode_id"]
        with self.assertRaises(RuntimeError) as ctx:
            env.step(p)
        self.assertIn("action_episode_id", str(ctx.exception))

    def test_packet_without_generation_rejected(self):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        p = env.legal_action_packet()
        del p["action_native_generation"]
        with self.assertRaises(RuntimeError) as ctx:
            env.step(p)
        self.assertIn("action_native_generation", str(ctx.exception))


class _Proxy:
    """Transparent proxy so a single native call can be substituted."""

    def __init__(self, real, step_fn):
        self._real = real
        self._step_fn = step_fn

    def __getattr__(self, name):
        return getattr(self._real, name)

    def step_batch(self, specs):
        return self._step_fn(specs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
