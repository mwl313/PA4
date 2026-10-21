"""G2-B and G2-D: real battle stepping, two-seat barrier, privacy and staleness."""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from pa4.env.native_env import (  # noqa: E402
    KIND_FINISHED,
    KIND_LABELS,
    KIND_WAIT,
    MAX_BRANCHES,
    PA4EnvBase,
)


def play_full(env, seed_rng=1701, max_steps=400):
    """Play to natural termination; return a result dict."""
    td = env.reset()
    steps = 0
    kinds_seen = set()
    while steps < max_steps:
        st = env._slots[0]
        kinds = env._kinds(st)
        kinds_seen.add((KIND_LABELS[kinds[0]], KIND_LABELS[kinds[1]]))
        if all(k == KIND_FINISHED for k in kinds):
            return {"finished": True, "steps": steps, "kinds": kinds_seen}
        packet = env.legal_action_packet(torch.Generator().manual_seed(seed_rng + steps))
        td = env.step(packet)
        steps += 1
        if bool(td["next", "done"][0, 0]):
            return {
                "finished": True,
                "steps": steps,
                "terminated": bool(td["next", "terminated"][0, 0]),
                "truncated": bool(td["next", "truncated"][0, 0]),
                "winner": int(td["next", "winner"][0]),
                "reward": td["next", "reward"][0].flatten().tolist(),
                "match_completed": bool(td["next", "match_completed"][0]),
                "kinds": kinds_seen,
            }
        td = td["next"]
    return {"finished": False, "steps": steps, "kinds": kinds_seen}


class BattleStepTest(unittest.TestCase):
    def test_four_distinct_seeded_games_complete_naturally(self):
        results = []
        for seed in (1, 2, 3, 4):
            env = PA4EnvBase(batch_size=1, seed=seed)
            r = play_full(env)
            r["seed"] = seed
            r["env_completed"] = env.completed_matches
            results.append(r)
        completed = [r for r in results if r.get("terminated")]
        self.assertEqual(len(completed), 4, f"expected 4 natural completions: {results}")
        for r in results:
            self.assertFalse(r["truncated"])
            self.assertTrue(r["match_completed"])
            self.assertEqual(r["env_completed"], 1, "episode counted more than once")

    def test_reward_matches_winner(self):
        for seed in (1, 2, 3, 4):
            env = PA4EnvBase(batch_size=1, seed=seed)
            r = play_full(env)
            self.assertIn(r["winner"], (0, 1))
            if r["winner"] == 0:
                self.assertEqual(r["reward"], [1.0, -1.0], f"seed {seed}")
            else:
                self.assertEqual(r["reward"], [-1.0, 1.0], f"seed {seed}")

    def test_wait_seat_submits_nothing(self):
        """A seat in Wait must never be marked present nor given a fake action."""
        seen_wait = False
        for seed in (1, 2, 3):
            env = PA4EnvBase(batch_size=1, seed=seed)
            env.reset()
            for _ in range(200):
                st = env._slots[0]
                kinds = env._kinds(st)
                packet = env.legal_action_packet(torch.Generator().manual_seed(hash((seed, _)) % 10000))
                for side in (0, 1):
                    if kinds[side] == KIND_WAIT:
                        seen_wait = True
                        self.assertFalse(
                            bool(packet["action_present"][0, side]),
                            "Wait seat was marked as submitting an action")
                        self.assertEqual(int(packet["action_branch_count"][0, side]), 0)
                if all(k == KIND_FINISHED for k in kinds):
                    break
                packet = env.legal_action_packet(torch.Generator().manual_seed(seed * 100 + _))
                td = env.step(packet)
                if bool(td["next", "done"][0, 0]):
                    break
        self.assertTrue(seen_wait, "no Wait state observed; cannot claim Wait handling")

    def test_only_one_step_batch_call_per_advance(self):
        env = PA4EnvBase(batch_size=2, seed=9)
        env.reset()
        calls = {"n": 0}
        real = env._engine.step_batch

        class CountingEngine:
            def __getattr__(self, name):
                return getattr(real_engine, name)

            def step_batch(self, specs):
                calls["n"] += 1
                return real(specs)

        real_engine = env._engine
        env._engine = CountingEngine()
        env.step(env.legal_action_packet())
        self.assertEqual(calls["n"], 1, "step() must submit exactly one native batch")

    def test_zero_completions_on_nonterminal_steps(self):
        env = PA4EnvBase(batch_size=1, seed=1)
        env.reset()
        packet = env.legal_action_packet()
        out = env.step(packet)
        if not bool(out["next", "done"][0, 0]):
            self.assertEqual(env.completed_matches, 0)
            self.assertEqual(out["next", "match_completed"].tolist(), [False])


class RejectionTest(unittest.TestCase):
    """G2-D: stale/ill-formed packets must be rejected, never defaulted."""

    def setUp(self):
        self.env = PA4EnvBase(batch_size=1, seed=1)
        self.env.reset()

    def _packet(self, **over):
        p = self.env.legal_action_packet(torch.Generator().manual_seed(1))
        for k, v in over.items():
            p[k] = v
        return p

    def test_stale_request_id_rejected(self):
        p = self._packet()
        p["action_request_id"] = p["action_request_id"] - 1
        with self.assertRaises(RuntimeError):
            self.env.step(p)

    def test_present_mismatch_rejected(self):
        p = self._packet()
        p["action_present"] = ~p["action_present"]
        with self.assertRaises(RuntimeError):
            self.env.step(p)

    def test_branch_count_overflow_rejected(self):
        p = self._packet()
        p["action_branch_count"] = p["action_branch_count"] + MAX_BRANCHES
        with self.assertRaises(RuntimeError):
            self.env.step(p)

    def test_zero_branch_count_for_requesting_seat_rejected(self):
        p = self._packet()
        p["action_branch_count"] = torch.zeros_like(p["action_branch_count"])
        with self.assertRaises(RuntimeError):
            self.env.step(p)


class PrivacyTest(unittest.TestCase):
    """G2-D: the actor view must not leak the opponent or private state."""

    def test_player_view_excludes_other_seat(self):
        env = PA4EnvBase(batch_size=1, seed=2)
        env.reset()
        st = env._slots[0]
        full = env._observe_one(st, 0)
        view = env.player_view(0)
        for k in ("token_mask", "categories", "floats", "flags", "role_ids", "side_ids"):
            self.assertTrue(torch.equal(view[k], full[k]), f"{k} differs from own view")
        self.assertNotIn("players", view.keys())
        self.assertIn("request_kind", view.keys(), "request metadata belongs to this seat only")

    def test_engine_private_keys_absent(self):
        env = PA4EnvBase(batch_size=1, seed=2)
        env.reset()
        view = env.player_view(0)
        forbidden = {"seed", "rng", "native_slot", "native_generation", "opponent",
                     "private", "teams", "winner"}
        self.assertFalse(forbidden & set(view.keys()))

    def test_bad_seat_rejected(self):
        env = PA4EnvBase(batch_size=1, seed=2)
        env.reset()
        for bad in (-1, 2, 99):
            with self.assertRaises(ValueError):
                env.player_view(bad)

    def test_knownness_masks_preserved(self):
        """Unknown opponent data must stay unknown, not be zero-filled as known."""
        env = PA4EnvBase(batch_size=1, seed=2)
        env.reset()
        st = env._slots[0]
        obs = env._observe_one(st, 0)
        # knownness tensors must exist and be boolean, and unknown slots must be False
        for known_key in ("category_known", "float_known", "flag_known"):
            self.assertEqual(obs[known_key].dtype, torch.bool)
        self.assertTrue(bool((~obs["category_known"]).any()),
                        "no unknown categories at all: knownness not exercised")

    def test_view_differs_between_seats(self):
        """The two seats must not receive identical (leaked) observations."""
        env = PA4EnvBase(batch_size=1, seed=2)
        env.reset()
        st = env._slots[0]
        a = env._observe_one(st, 0)
        b = env._observe_one(st, 1)
        differing = [k for k in ("side_ids", "categories", "role_ids")
                     if not torch.equal(a[k], b[k])]
        self.assertTrue(differing,
                        "seat views are byte-identical; side scoping is not demonstrated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
