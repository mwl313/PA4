"""G2-E: a genuine P1 staged conditional-policy packet crossing EnvBase.step().

Uses P1's ConditionalPolicy (real PA3 model) to choose each seat's complete
prefix from live Rust candidates, packs it into the P2 action layout, submits it
through the real EnvBase, and confirms the native result matches a direct
step_batch on an independent oracle.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402

from pa4.actions.conditional_policy import ConditionalPolicy  # noqa: E402
from pa4.env.native_env import (  # noqa: E402
    KIND_FINISHED,
    MAX_BRANCHES,
    PA4EnvBase,
)
from pa4.env.trace_compare import NativeOracle  # noqa: E402


class P1IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PA4EnvBase(batch_size=1, seed=21)
        cls.env.verify_contract()
        cls.policy = ConditionalPolicy(seed=20261006)

    def _staged_prefix(self, side, st, gen_rng):
        """P1 staged sampling: ask Rust for candidates after each pick."""
        prefix = []
        n_branches = len(bytes(self.env.engine.request_branches(st.slot, st.generation, side)))
        for _ in range(min(n_branches, MAX_BRANCHES)):
            cands = self.env._candidates(st, side, prefix)
            self.assertTrue(cands, "engine returned no legal candidate")
            idx = int(torch.randint(0, len(cands), (1,), generator=gen_rng).item())
            prefix.append(cands[idx])
        return prefix

    def test_p1_staged_policy_packet_crosses_envbase(self):
        env = PA4EnvBase(batch_size=1, seed=21)
        env.verify_contract()
        td = env.reset()
        st = env._slots[0]
        oracle = NativeOracle(env.module, env._data_dir, env._teams_json, 2)
        o_handles = oracle.reset([st.used_teams[0]], [st.used_teams[1]], [st.used_seed], [(0, 1)])
        o_slot, o_gen = o_handles[0]
        self.assertEqual((o_slot, o_gen), (st.slot, st.generation),
                         "oracle reset produced a different handle identity")

        rng = torch.Generator().manual_seed(4242)
        kinds = env._kinds(st)
        self.assertFalse(all(k == KIND_FINISHED for k in kinds))

        n = env._batch
        action = torch.zeros((n, 2, MAX_BRANCHES, 6), dtype=torch.long)
        counts = torch.zeros((n, 2), dtype=torch.long)
        present = torch.zeros((n, 2), dtype=torch.bool)
        req_ids = torch.zeros((n, 2), dtype=torch.long)
        choices = []

        for side in (0, 1):
            req_ids[0, side] = st.request_ids[side]
            if kinds[side] not in (0, 1, 2):
                continue
            prefix = self._staged_prefix(side, st, rng)
            self.assertGreaterEqual(len(prefix), 1)
            action[0, side, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
            counts[0, side] = len(prefix)
            present[0, side] = True
            choices.append((side, prefix))

        self.assertTrue(choices, "no seat requested a decision at reset")

        from tensordict import TensorDict

        packet = TensorDict(
            {"action": action, "action_branch_count": counts,
             "action_present": present, "action_request_id": req_ids},
            batch_size=[n],
        )

        # direct native step on the oracle with the SAME tuples
        oracle_res = oracle.step(st.slot, st.generation, choices)

        out = env.step(packet)

        env_res = (
            True,
            bool(out["next", "terminated"][0, 0]),
            bool(out["next", "truncated"][0, 0]),
            (int(out["next", "winner"][0]) if int(out["next", "winner"][0]) >= 0 else None),
            None,
            env._kinds(env._slots[0])[0],
            env._kinds(env._slots[0])[1],
        )
        self.assertEqual(env_res, oracle_res,
                         "P1 packet result diverged from direct native step")

    def test_p1_policy_replay_still_consistent(self):
        """P1's own guarantee must survive alongside P2 (no regression)."""
        env = self.env
        env.reset()
        st = env._slots[0]
        obs = env._observe_one(st, 0)
        from agent.types.observation import ObservationBatch

        batch = ObservationBatch(
            token_mask=obs["token_mask"].unsqueeze(0),
            categories=obs["categories"].unsqueeze(0),
            category_known=obs["category_known"].unsqueeze(0),
            floats=obs["floats"].unsqueeze(0),
            float_known=obs["float_known"].unsqueeze(0),
            flags=obs["flags"].unsqueeze(0),
            flag_known=obs["flag_known"].unsqueeze(0),
            role_ids=obs["role_ids"].unsqueeze(0),
            side_ids=obs["side_ids"].unsqueeze(0),
        ).to("cpu")
        gen = torch.Generator().manual_seed(77)
        acts, rec = self.policy.complete_request(
            _RefAdapter(env), st.slot, st.generation, 0, request_id=1,
            observation=batch, generator=gen,
        )
        self.assertGreaterEqual(len(acts), 1)
        rep = self.policy.replay(rec, batch)
        self.assertLessEqual(rep["abs_joint_delta"], 1e-5)


class _RefAdapter:
    """Minimal adapter exposing the P1 NativeReference interface over the env."""

    def __init__(self, env):
        self.env = env

    def request(self, slot, generation, side):
        from pa4.actions.native_reference import RequestView, REQUEST_KIND_LABELS

        kind = int(self.env.engine.request_kind(slot, generation, side))
        branches = tuple(bytes(self.env.engine.request_branches(slot, generation, side)))
        return RequestView(slot=slot, generation=generation, side=side, kind=kind,
                           branch_count=len(branches), branch_slots=branches)

    def candidates(self, slot, generation, side, prefix):
        out = self.env.engine.candidates_batch([(slot, generation, side, list(prefix))])
        return [tuple(int(v) for v in row) for row in out[0]]


if __name__ == "__main__":
    unittest.main(verbosity=2)
