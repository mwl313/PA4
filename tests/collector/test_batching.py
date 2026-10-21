"""G3-B: real central CUDA batching with isolated per-row results.

These tests assert the things the design says must NOT be faked:
  * a genuine model forward with B >= 2 from independent requests
  * the model actually on CUDA
  * every row's result corresponds to its own request identity
  * cache isolation between concurrent requests
  * fail-closed behaviour for stale/duplicate/overflowed requests
"""

import os
import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch

from pa4.actions.batched_scorer import (
    BatchedConditionalScorer, ERR_COUNT_MISMATCH, ERR_NO_CACHE, ERR_OVERFLOW,
    ERR_STALE_IDENTITY, request_spec,
)
from pa4.env.native_env import PA4EnvBase

OBS = ("token_mask", "categories", "category_known", "floats", "float_known",
       "flags", "flag_known", "role_ids", "side_ids")


def make_row(env, st, obs, *, w, op=0, level=0, cands=None, table=None,
             gen=None, pv=1):
    td = request_spec().clone()
    td["op"] = torch.tensor(op)
    td["worker_id"] = torch.tensor(w)
    td["native_slot"] = torch.tensor(st.slot)
    td["native_generation"] = torch.tensor(
        st.generation if gen is None else gen)
    td["episode_id"] = torch.tensor(st.episode_id)
    td["seat"] = torch.tensor(0)
    td["request_local_id"] = torch.tensor(w)
    td["policy_version"] = torch.tensor(pv)
    td["request_key"] = torch.tensor(w)
    td["branch_level"] = torch.tensor(level)
    for k in OBS:
        td[k] = obs[k]
    if table is not None:
        k = len(cands)
        td["candidate_count"] = torch.tensor(k)
        td["action_ids"] = table.action_ids[0, 0]
        td["legal_mask"] = table.mask[0, 0]
        td["entity_token"] = table.entity_token[0, 0]
        td["move_token"] = table.move_token[0, 0]
    return td


def branch_table(env, st, obs, seat=0):
    from agent.types.observation import ObservationBatch
    from pa4.actions.conditional_policy import build_branch_table

    ob = ObservationBatch(
        token_mask=obs["token_mask"].unsqueeze(0), categories=obs["categories"].unsqueeze(0),
        category_known=obs["category_known"].unsqueeze(0), floats=obs["floats"].unsqueeze(0),
        float_known=obs["float_known"].unsqueeze(0), flags=obs["flags"].unsqueeze(0),
        flag_known=obs["flag_known"].unsqueeze(0), role_ids=obs["role_ids"].unsqueeze(0),
        side_ids=obs["side_ids"].unsqueeze(0))
    cands = env._candidates(st, seat, [])
    return cands, build_branch_table([cands], ob.layout, ob.active_to_roster())


class BatchedForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = []
        for w in range(4):
            e = PA4EnvBase(batch_size=1, seed=1 + w)
            e.reset()
            st = e._slots[0]
            cls.envs.append((e, st, e._observe_one(st, 0)))

    def test_model_is_on_cuda(self):
        svc = BatchedConditionalScorer(device="cuda")
        svc._ensure_model()
        self.assertIn("cuda", svc.model_device)

    def test_one_forward_covers_multiple_independent_requests(self):
        svc = BatchedConditionalScorer(device="cuda")
        rows = [make_row(e, st, obs, w=w) for w, (e, st, obs) in enumerate(self.envs)]
        svc(torch.stack(rows))  # ENCODE all four

        steps = []
        for w, (e, st, obs) in enumerate(self.envs):
            cands, table = branch_table(e, st, obs)
            steps.append(make_row(e, st, obs, w=w, op=1, level=0,
                                  cands=cands, table=table))
        out = svc(torch.stack(steps))

        self.assertEqual(svc.model_forward_count, 1,
                         "expected ONE forward for the four STEP rows")
        self.assertEqual(svc.forward_batch_sizes, [4])
        self.assertTrue(bool((out["error_code"] == 0).all()))
        # each row got a valid selection for its own candidate set
        for w in range(4):
            idx = int(out["selected_index"][w])
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, int(out["branch_k"][w]))

    def test_rows_are_independent(self):
        """Different requests must get different cached state, not aliases."""
        svc = BatchedConditionalScorer(device="cuda")
        rows = [make_row(e, st, obs, w=w) for w, (e, st, obs) in enumerate(self.envs)]
        svc(torch.stack(rows))
        self.assertEqual(svc.cache_size(), 4, "each request must own a cache entry")

    def test_duplicate_encode_is_rejected(self):
        svc = BatchedConditionalScorer(device="cuda")
        row = make_row(*self.envs[0], w=0)
        svc(torch.stack([row]))
        out = svc(torch.stack([row]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY)

    def test_overflow_rejected(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        svc(torch.stack([make_row(e, st, obs, w=0)]))
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table)
        r["candidate_count"] = torch.tensor(65)
        r["legal_mask"] = torch.ones(64, dtype=torch.bool)
        out = svc(torch.stack([r]))
        self.assertEqual(int(out["error_code"][0]), ERR_OVERFLOW)

    def test_count_mask_mismatch_rejected(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        svc(torch.stack([make_row(e, st, obs, w=0)]))
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table)
        r["candidate_count"] = torch.tensor(3)      # disagrees with mask
        out = svc(torch.stack([r]))
        self.assertEqual(int(out["error_code"][0]), ERR_COUNT_MISMATCH)

    def test_unknown_key_fails_closed(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=99, op=1, level=0, cands=cands, table=table)
        out = svc(torch.stack([r]))
        self.assertEqual(int(out["error_code"][0]), ERR_NO_CACHE)

    def test_wrong_level_fails_closed_and_evicts(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        svc(torch.stack([make_row(e, st, obs, w=0)]))
        self.assertEqual(svc.cache_size(), 1)
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=0, op=1, level=3, cands=cands, table=table)
        out = svc(torch.stack([r]))
        self.assertEqual(int(out["error_code"][0]), ERR_NO_CACHE)
        self.assertEqual(svc.cache_size(), 0, "rejected step must evict the entry")

    def test_stale_generation_is_a_different_key(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        svc(torch.stack([make_row(e, st, obs, w=0)]))
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table,
                     gen=st.generation + 7)
        out = svc(torch.stack([r]))
        self.assertEqual(int(out["error_code"][0]), ERR_NO_CACHE)

    def test_reset_evicts_only_the_addressed_entry(self):
        svc = BatchedConditionalScorer(device="cuda")
        rows = [make_row(e, st, obs, w=w) for w, (e, st, obs) in enumerate(self.envs[:2])]
        svc(torch.stack(rows))
        self.assertEqual(svc.cache_size(), 2)
        e, st, obs = self.envs[0]
        out = svc(torch.stack([make_row(e, st, obs, w=0, op=2)]))
        self.assertEqual(int(out["error_code"][0]), 0)
        self.assertEqual(svc.cache_size(), 1, "only the addressed entry should go")


if __name__ == "__main__":
    unittest.main(verbosity=2)
