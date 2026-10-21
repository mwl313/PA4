"""G2-E: a genuine P1 staged CONDITIONAL-POLICY packet crossing EnvBase.step().

This is the test the read-only review correctly found was fake: the previous
version picked candidates with torch.randint, so no P1 policy ever reached
EnvBase. Here each requesting seat's complete prefix is produced by
`ConditionalPolicy.complete_request` (the real PA3 model and scorer), then packed
into the P2 action layout and submitted through the real EnvBase.

Gate G2-E (PA4-P2-DESIGN-v1.0 L270): "At least one P1 staged two-seat policy packet
crosses the real EnvBase/native boundary".
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from agent.types.observation import ObservationBatch  # noqa: E402

from pa4.actions.conditional_policy import ConditionalPolicy  # noqa: E402
from pa4.env.native_env import (  # noqa: E402
    KIND_FINISHED,
    MAX_BRANCHES,
    PA4EnvBase,
)
from pa4.env.trace_compare import NativeOracle  # noqa: E402


def obs_to_batch(env, st, side):
    """Wrap one seat's decoded view as a PA3 ObservationBatch (batch of 1)."""
    o = env._observe_one(st, side)
    return ObservationBatch(
        token_mask=o["token_mask"].unsqueeze(0),
        categories=o["categories"].unsqueeze(0),
        category_known=o["category_known"].unsqueeze(0),
        floats=o["floats"].unsqueeze(0),
        float_known=o["float_known"].unsqueeze(0),
        flags=o["flags"].unsqueeze(0),
        flag_known=o["flag_known"].unsqueeze(0),
        role_ids=o["role_ids"].unsqueeze(0),
        side_ids=o["side_ids"].unsqueeze(0),
    ).to("cpu")


class _RefAdapter:
    """Expose P1's NativeReference interface over the env's native engine."""

    def __init__(self, env):
        self.env = env

    def request(self, slot, generation, side):
        from pa4.actions.native_reference import RequestView

        kind = int(self.env.engine.request_kind(slot, generation, side))
        branches = tuple(bytes(self.env.engine.request_branches(slot, generation, side)))
        return RequestView(slot=slot, generation=generation, side=side, kind=kind,
                           branch_count=len(branches), branch_slots=branches)

    def candidates(self, slot, generation, side, prefix):
        out = self.env.engine.candidates_batch([(slot, generation, side, list(prefix))])
        return [tuple(int(v) for v in row) for row in out[0]]


class P1PolicyPacketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = ConditionalPolicy(seed=20261006)

    def test_p1_conditional_policy_packet_crosses_envbase(self):
        env = PA4EnvBase(batch_size=1, seed=21)
        env.verify_contract()
        env.reset()
        st = env._slots[0]
        adapter = _RefAdapter(env)

        oracle = NativeOracle(env.module, env._data_dir, env._teams_json, 2)
        o_slot, o_gen = oracle.reset(
            [st.used_teams[0]], [st.used_teams[1]], [st.used_seed], [(0, 1)])[0]
        self.assertEqual((o_slot, o_gen), (st.slot, st.generation))

        kinds = env._kinds(st)
        self.assertFalse(all(k == KIND_FINISHED for k in kinds))

        n = env._batch
        action = torch.zeros((n, 2, MAX_BRANCHES, 6), dtype=torch.long)
        counts = torch.zeros((n, 2), dtype=torch.long)
        present = torch.zeros((n, 2), dtype=torch.bool)
        req_ids = torch.zeros((n, 2), dtype=torch.long)
        ep_ids = torch.zeros((n, 2), dtype=torch.long)
        gens = torch.zeros((n, 2), dtype=torch.long)
        choices = []
        records = []

        for side in (0, 1):
            req_ids[0, side] = st.request_ids[side]
            ep_ids[0, side] = st.episode_id
            gens[0, side] = st.generation
            if kinds[side] not in (0, 1, 2):
                continue
            obs = obs_to_batch(env, st, side)
            gen = torch.Generator().manual_seed(9000 + side)
            acts, rec = self.policy.complete_request(
                adapter, st.slot, st.generation, side, request_id=side + 1,
                observation=obs, generator=gen,
            )
            # Prove the prefix really came from the P1 policy: replay it.
            rep = self.policy.replay(rec, obs)
            self.assertLessEqual(rep["abs_joint_delta"], 1e-5)
            self.assertEqual(len(acts), len(rec.branches))
            records.append(rec)

            action[0, side, :len(acts)] = torch.tensor(acts, dtype=torch.long)
            counts[0, side] = len(acts)
            present[0, side] = True
            choices.append((side, acts))

        self.assertTrue(choices, "no seat requested a decision at reset")
        self.assertEqual(len(records), len(choices))
        # At least one seat produced a genuine multi-branch actor row.
        self.assertTrue(any(r.actor_active for r in records),
                        "no actor-active row: the policy did not really decide")

        packet = TensorDict(
            {
                "action": action, "action_branch_count": counts,
                "action_present": present, "action_request_id": req_ids,
                "action_episode_id": ep_ids, "action_native_generation": gens,
            },
            batch_size=[n],
        )

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
                         "P1 policy packet diverged from direct native step")
        self.assertFalse(bool(out["next", "request_revised"][0]),
                         "a policy-produced legal packet must not be revised")

    def test_p1_packet_completes_a_full_battle(self):
        """Drive an entire battle with P1 policy decisions on both seats."""
        env = PA4EnvBase(batch_size=1, seed=21)
        env.verify_contract()
        env.reset()
        adapter = _RefAdapter(env)
        steps = 0
        policy_decisions = 0

        while steps < 400:
            st = env._slots[0]
            kinds = env._kinds(st)
            if all(k == KIND_FINISHED for k in kinds):
                break
            n = env._batch
            action = torch.zeros((n, 2, MAX_BRANCHES, 6), dtype=torch.long)
            counts = torch.zeros((n, 2), dtype=torch.long)
            present = torch.zeros((n, 2), dtype=torch.bool)
            req_ids = torch.zeros((n, 2), dtype=torch.long)
            ep_ids = torch.zeros((n, 2), dtype=torch.long)
            gens = torch.zeros((n, 2), dtype=torch.long)
            for side in (0, 1):
                req_ids[0, side] = st.request_ids[side]
                ep_ids[0, side] = st.episode_id
                gens[0, side] = st.generation
                if kinds[side] not in (0, 1, 2):
                    continue
                obs = obs_to_batch(env, st, side)
                acts, _rec = self.policy.complete_request(
                    adapter, st.slot, st.generation, side,
                    request_id=steps * 10 + side,
                    observation=obs,
                    generator=torch.Generator().manual_seed(1000 + env._seed * 100000 + steps * 10 + side),
                )
                policy_decisions += 1
                action[0, side, :len(acts)] = torch.tensor(acts, dtype=torch.long)
                counts[0, side] = len(acts)
                present[0, side] = True

            packet = TensorDict(
                {"action": action, "action_branch_count": counts,
                 "action_present": present, "action_request_id": req_ids,
                 "action_episode_id": ep_ids, "action_native_generation": gens},
                batch_size=[n],
            )
            out = env.step(packet)
            steps += 1
            if bool(out["next", "done"][0, 0]):
                self.assertTrue(bool(out["next", "terminated"][0, 0]))
                self.assertEqual(env.completed_matches, 1)
                self.assertGreater(policy_decisions, 0)
                return
        self.fail("P1-policy battle did not terminate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
