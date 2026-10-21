"""G2-C: deterministic parity between PA4EnvBase and an independent native oracle."""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402

from pa4.env.native_env import (  # noqa: E402
    KIND_FINISHED,
    KIND_LABELS,
    MAX_BRANCHES,
    PA4EnvBase,
)
from pa4.env.trace_compare import (  # noqa: E402
    NativeOracle,
    ParityReport,
    compare_boundary,
    compare_candidates,
    compare_step_result,
)


def make_oracle(env: PA4EnvBase) -> NativeOracle:
    return NativeOracle(env.module, env._data_dir, env._teams_json, workers=env._workers)


def run_lockstep(env: PA4EnvBase, max_steps: int = 400) -> ParityReport:
    """Drive the env and an independent oracle with identical actions."""
    report = ParityReport()
    oracle = make_oracle(env)
    row = 0
    td = env.reset()
    st = env._slots[row]
    slot, gen = st.slot, st.generation

    # reset the oracle with the exact same inputs
    oracle.reset([st.used_teams[0]], [st.used_teams[1]], [st.used_seed], [(0, 1)])

    for step in range(max_steps):
        compare_boundary(step, env, oracle, row, slot, gen, report)
        kinds = env._kinds(st)
        if all(k == KIND_FINISHED for k in kinds):
            break

        choices = []
        for side in (0, 1):
            if kinds[side] not in (0, 1, 2):
                continue
            nb = len(bytes(env.engine.request_branches(slot, gen, side)))
            prefix = []
            for _ in range(min(nb, MAX_BRANCHES)):
                cands = env._candidates(st, side, prefix)
                idx = int(torch.randint(0, len(cands), (1,),
                                        generator=torch.Generator().manual_seed(step * 10 + side)).item())
                prefix.append(cands[idx])
            choices.append((side, prefix))

        if not choices:
            break
        compare_candidates(step, env, oracle, row, slot, gen, choices, report)

        # submit identical packets to both
        n = env._batch
        action = torch.zeros((n, 2, MAX_BRANCHES, 6), dtype=torch.long)
        counts = torch.zeros((n, 2), dtype=torch.long)
        present = torch.zeros((n, 2), dtype=torch.bool)
        req_ids = torch.zeros((n, 2), dtype=torch.long)
        ep_ids = torch.zeros((n, 2), dtype=torch.long)
        gens = torch.zeros((n, 2), dtype=torch.long)
        for side, acts in choices:
            action[row, side, :len(acts)] = torch.tensor(acts, dtype=torch.long)
            counts[row, side] = len(acts)
            present[row, side] = True
        for side in (0, 1):
            req_ids[row, side] = st.request_ids[side]
            ep_ids[row, side] = st.episode_id
            gens[row, side] = st.generation

        from tensordict import TensorDict

        packet = TensorDict({"action": action, "action_branch_count": counts,
                             "action_present": present, "action_request_id": req_ids,
                             "action_episode_id": ep_ids,
                             "action_native_generation": gens},
                            batch_size=[n])
        out = env.step(packet)
        oracle_res = oracle.step(slot, gen, choices)

        env_res = (
            True,
            bool(out["next", "terminated"][row, 0]),
            bool(out["next", "truncated"][row, 0]),
            (int(out["next", "winner"][row]) if int(out["next", "winner"][row]) >= 0 else None),
            None,
            env._kinds(env._slots[row])[0],
            env._kinds(env._slots[row])[1],
        )
        compare_step_result(step, env_res, oracle_res, report)

        report.steps = step + 1
        if env_res[1] or env_res[2]:
            report.terminal_matched = (env_res[1], env_res[2], env_res[3]) == (
                oracle_res[1], oracle_res[2], oracle_res[3])
            break
        td = out["next"]
    return report


class EpisodeParityTest(unittest.TestCase):
    def test_seeded_full_game_parity(self):
        for seed in (1, 2, 3, 4):
            with self.subTest(seed=seed):
                env = PA4EnvBase(batch_size=1, seed=seed)
                env.verify_contract()
                report = run_lockstep(env)
                self.assertGreater(report.steps, 0)
                self.assertTrue(
                    report.ok,
                    f"seed {seed} parity mismatch: {report.first_mismatch}")
                self.assertTrue(report.terminal_matched,
                                f"seed {seed} terminal result disagreed with oracle")

    def test_deterministic_reset_same_seed(self):
        a = PA4EnvBase(batch_size=1, seed=7)
        b = PA4EnvBase(batch_size=1, seed=7)
        ta, tb = a.reset(), b.reset()
        for k in ("request_kind", "branch_count", "request_id"):
            self.assertTrue(torch.equal(ta["players", k], tb["players", k]))
        oa, ob = a._observe_one(a._slots[0], 0), b._observe_one(b._slots[0], 0)
        for k, v in oa.items():
            self.assertTrue(torch.equal(v, ob[k]), f"{k} differs on identical seed")

    def test_partial_reset_preserves_other_slot(self):
        env = PA4EnvBase(batch_size=2, seed=11)
        env.reset()
        before_1 = (env._slots[1].slot, env._slots[1].generation,
                    env._slots[1].episode_id)
        # advance slot 1 a little
        packet = env.legal_action_packet()
        env.step(packet)
        mid_1 = (env._slots[1].slot, env._slots[1].generation,
                 env._slots[1].episode_id, env._slots[1].step_index)
        # partial reset of slot 0 only, via the TorchRL _reset mask
        from tensordict import TensorDict

        mask = TensorDict({"_reset": torch.tensor([True, False])}, batch_size=[2])
        env.reset(mask)
        after_1 = (env._slots[1].slot, env._slots[1].generation,
                   env._slots[1].episode_id, env._slots[1].step_index)
        self.assertEqual(mid_1, after_1, "unrelated live slot changed on partial reset")
        self.assertGreater(env._slots[0].episode_id, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
