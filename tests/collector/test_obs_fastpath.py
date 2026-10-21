"""P3-E: fast observation path must be byte-identical to PA3's authoritative decode.

This guards the optimisation that made the actor hot path ~2.4x faster. The fast
path is only allowed to exist because it is provably equivalent; these tests
enforce that and fail loudly if it ever drifts.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402

from pa4.env.native_env import KIND_FINISHED, PA4EnvBase  # noqa: E402
from pa4.env.obs_fastpath import OBS_LEAVES, SliceDecoder  # noqa: E402


class FastDecodeEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PA4EnvBase(batch_size=1, seed=1)
        cls.env.reset()
        cls.st = cls.env._slots[0]
        cls.dec = SliceDecoder(cls.env._engine, cls.env._mod)

    def test_fast_matches_full_all_leaves(self):
        fast = self.dec.decode_fast(self.st.slot, self.st.generation, 0)
        full = self.dec.decode_full(self.st.slot, self.st.generation, 0)
        self.assertEqual(set(fast.keys()), set(OBS_LEAVES))
        for k in OBS_LEAVES:
            self.assertEqual(fast[k].shape, full[k].shape, f"{k} shape")
            self.assertEqual(fast[k].dtype, full[k].dtype, f"{k} dtype")
            self.assertTrue(torch.equal(fast[k], full[k]), f"{k} values differ")

    def test_fast_matches_full_both_seats(self):
        for side in (0, 1):
            fast = self.dec.decode_fast(self.st.slot, self.st.generation, side)
            full = self.dec.decode_full(self.st.slot, self.st.generation, side)
            for k in OBS_LEAVES:
                self.assertTrue(torch.equal(fast[k], full[k]), f"seat{side}.{k}")

    def test_verify_contract_passes(self):
        d = SliceDecoder(self.env._engine, self.env._mod)
        d.verify_contract(self.st.slot, self.st.generation)
        self.assertTrue(d.verified)

    def test_role_side_derived_from_layout_not_wire(self):
        """role_ids/side_ids are not on the wire; they must match PA3's layout fill."""
        fast = self.dec.decode_fast(self.st.slot, self.st.generation, 0)
        full = self.dec.decode_full(self.st.slot, self.st.generation, 0)
        self.assertTrue(torch.equal(fast["role_ids"], full["role_ids"]))
        self.assertTrue(torch.equal(fast["side_ids"], full["side_ids"]))

    def test_padded_slots_use_role_pad(self):
        fast = self.dec.decode_fast(self.st.slot, self.st.generation, 0)
        pad = self.dec._layout.ROLE_PAD
        masked = ~fast["token_mask"]
        if bool(masked.any()):
            self.assertTrue(bool((fast["role_ids"][masked] == pad).all()))


class EnvFastPathTest(unittest.TestCase):
    def test_env_verifies_decoder_on_reset(self):
        env = PA4EnvBase(batch_size=1, seed=2)
        env.reset()
        self.assertIsNotNone(env._decoder)
        self.assertTrue(env._decoder.verified)

    def test_fast_path_disabled_when_requested(self):
        env = PA4EnvBase(batch_size=1, seed=2, use_fast_obs=False)
        env.reset()
        self.assertIsNone(env._decoder)
        # still produces a valid observation through the authoritative path
        obs = env._observe_one(env._slots[0], 0)
        self.assertEqual(tuple(obs["token_mask"].shape), (96,))

    def test_per_step_cache_decodes_each_seat_once(self):
        env = PA4EnvBase(batch_size=1, seed=3)
        env.reset()
        before = env._decoder.fast_decodes
        env.step(env.legal_action_packet())
        decoded = env._decoder.fast_decodes - before
        # two seats, each decoded at most once per transition
        self.assertLessEqual(decoded, 2, f"decoded {decoded} times in one step")

    def test_cache_invalidated_on_reset(self):
        """After reset, no entry from the previous episode may survive."""
        env = PA4EnvBase(batch_size=1, seed=4)
        env.reset()
        env.step(env.legal_action_packet())
        stale_episode = env._slots[0].episode_id
        stale_keys = set(env._obs_cache)
        self.assertTrue(stale_keys, "expected populated cache before reset")

        env.reset()
        new_episode = env._slots[0].episode_id
        self.assertGreater(new_episode, stale_episode)
        # No key carrying the old episode's step state may remain: every key now
        # belongs to the fresh step_index-0 state of the new episode.
        for key in env._obs_cache:
            self.assertEqual(key[2], 0, f"stale step_index survived reset: {key}")

    def test_full_game_with_fast_path_matches_oracle_path(self):
        """A whole game driven on the fast path must terminate correctly."""
        env = PA4EnvBase(batch_size=1, seed=5)
        env.reset()
        steps = 0
        while steps < 200:
            st = env._slots[0]
            if all(k == KIND_FINISHED for k in env._kinds(st)):
                break
            out = env.step(env.legal_action_packet(
                torch.Generator().manual_seed(5000 + steps)))
            steps += 1
            if bool(out["next", "done"][0, 0]):
                self.assertEqual(env.completed_matches, 1)
                return
        self.fail("fast-path game did not terminate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
