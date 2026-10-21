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
