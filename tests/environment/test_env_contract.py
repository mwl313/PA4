"""G2-A: real EnvBase contract and TorchRL spec conformance."""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402
from torchrl.envs import EnvBase  # noqa: E402
from torchrl.envs.utils import check_env_specs  # noqa: E402

from pa4.env.native_env import (  # noqa: E402
    KIND_FINISHED,
    KIND_LABELS,
    KIND_WAIT,
    MAX_BRANCHES,
    PA4EnvBase,
)

EXPECTED_STATE_KEYS = {
    "native_slot", "native_generation", "episode_id", "step_index",
    "request_revised", "match_completed", "winner",
}
EXPECTED_PLAYER_KEYS = {
    "token_mask", "categories", "category_known", "floats", "float_known",
    "flags", "flag_known", "role_ids", "side_ids", "observation_valid",
    "request_kind", "decision_required", "request_id", "branch_count",
}


class EnvContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PA4EnvBase(batch_size=1, seed=1, auto_reset=True)
        cls.env.verify_contract()

    def test_is_a_real_torchrl_envbase(self):
        self.assertIsInstance(self.env, EnvBase)
        self.assertEqual(self.env.batch_size, torch.Size([1]))

    def test_observation_contract_pinned(self):
        m = self.env.module
        self.assertEqual(m.OBSERVATION_TOKENS, 96)
        self.assertEqual(m.OBSERVATION_FIXED_BYTES, 41986)
        self.assertEqual(m.CONTRACT_HASH,
                         "dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35")

    def test_reset_returns_fixed_keys(self):
        td = self.env.reset()
        self.assertTrue(EXPECTED_STATE_KEYS.issubset(set(td.keys())))
        self.assertTrue(EXPECTED_PLAYER_KEYS.issubset(set(td["players"].keys())))

    def test_observation_shapes(self):
        td = self.env.reset()
        p = td["players"]
        self.assertEqual(tuple(p["token_mask"].shape), (1, 2, 96))
        self.assertEqual(tuple(p["categories"].shape), (1, 2, 96, 32))
        self.assertEqual(tuple(p["floats"].shape), (1, 2, 96, 50))
        self.assertEqual(tuple(p["flags"].shape), (1, 2, 96, 40))
        self.assertEqual(p["token_mask"].dtype, torch.bool)
        self.assertEqual(p["floats"].dtype, torch.float32)
        self.assertEqual(p["categories"].dtype, torch.long)

    def test_check_env_specs_passes_for_real(self):
        env = PA4EnvBase(batch_size=1, seed=3, auto_reset=True)
        env.verify_contract()
        check_env_specs(env)  # raises on any mismatch

    def test_legal_random_action_comes_from_engine(self):
        env = PA4EnvBase(batch_size=1, seed=5)
        env.reset()
        packet = env.legal_action_packet()
        st = env._slots[0]
        kinds = env._kinds(st)
        for side in (0, 1):
            if kinds[side] in (0, 1, 2):
                self.assertTrue(bool(packet["action_present"][0, side]))
                k = int(packet["action_branch_count"][0, side])
                self.assertGreaterEqual(k, 1)
                self.assertLessEqual(k, MAX_BRANCHES)
                # every emitted row must be a real candidate for its prefix
                prefix = []
                for i in range(k):
                    cands = env._candidates(st, side, prefix)
                    row = tuple(int(v) for v in packet["action"][0, side, i])
                    self.assertIn(row, cands, f"seat{side} level{i} not engine-legal")
                    prefix.append(row)
            else:
                self.assertFalse(bool(packet["action_present"][0, side]))


class KindSemanticsTest(unittest.TestCase):
    """Request-kind vocabulary is inherited from Rust, not invented."""

    def test_kind_labels(self):
        self.assertEqual(KIND_LABELS[0], "Preview")
        self.assertEqual(KIND_LABELS[1], "Normal")
        self.assertEqual(KIND_LABELS[2], "Replacement")
        self.assertEqual(KIND_LABELS[3], "Wait")
        self.assertEqual(KIND_LABELS[4], "Finished")
        self.assertEqual(KIND_WAIT, 3)
        self.assertEqual(KIND_FINISHED, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
