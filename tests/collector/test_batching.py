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


_FP_CACHE = {}


def default_fingerprint():
    """The fingerprint of the weights every scorer in these tests serves."""
    if "fp" not in _FP_CACHE:
        from pa4.actions.batched_scorer import fingerprint_id, model_fingerprint

        _FP_CACHE["fp"] = fingerprint_id(model_fingerprint())
    return _FP_CACHE["fp"]


def make_row(env, st, obs, *, w, op=0, level=0, cands=None, table=None,
             gen=None, pv=1, fp=None):
    td = request_spec().clone()
    td["op"] = torch.tensor(op)
    td["policy_fingerprint"] = torch.tensor(
        default_fingerprint() if fp is None else fp)
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
        # each row got a valid selection for ITS OWN candidate set
        for w in range(4):
            cands_w = steps[w]["candidate_count"].item()
            idx = int(out["selected_index"][w])
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, int(out["branch_k"][w]))
            self.assertEqual(int(out["branch_k"][w]), int(cands_w),
                             f"row {w} branch_k does not match its own candidates")
            self.assertEqual(int(out["request_key"][w]), w,
                             f"row {w} response came back under another key")

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

    def test_same_local_id_different_worker_does_not_alias(self):
        """G3-D: the P1 cache-key collision. Same slot/gen/episode/seat/id,
        different worker -> two independent entries, never one aliased row."""
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        a = make_row(e, st, obs, w=0)
        b = make_row(e, st, obs, w=0)
        b["worker_id"] = torch.tensor(1)
        out = svc(torch.stack([a, b]))
        self.assertTrue(bool((out["error_code"] == 0).all()))
        self.assertEqual(svc.cache_size(), 2,
                         "two workers with the same local id must not share a cache entry")

        # and each worker's STEP resolves against its own entry
        cands, table = branch_table(e, st, obs)
        sa = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table)
        sb = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table)
        sb["worker_id"] = torch.tensor(1)
        out = svc(torch.stack([sa, sb]))
        self.assertTrue(bool((out["error_code"] == 0).all()), out["error_code"])
        self.assertEqual(svc.cache_size(), 2)

    def test_mixed_policy_version_rejected_without_corrupting_valid_entry(self):
        """G3-D: a STEP with a different policy version must not be accepted, and
        must not corrupt the legitimate same-identity entry of the live version."""
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        svc(torch.stack([make_row(e, st, obs, w=0, pv=1)]))
        cands, table = branch_table(e, st, obs)

        stale = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table, pv=2)
        out = svc(torch.stack([stale]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY,
                         "a mixed-version request was accepted")

        # the pv=1 entry must still be intact and usable
        live = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table, pv=1)
        out = svc(torch.stack([live]))
        self.assertEqual(int(out["error_code"][0]), 0,
                         "the rejected version corrupted the live cache entry")

    def test_foreign_version_on_encode_rejected(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        out = svc(torch.stack([make_row(e, st, obs, w=0, pv=2)]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY)
        self.assertEqual(svc.cache_size(), 0)

    def test_zero_fingerprint_is_not_a_wildcard(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        out = svc(torch.stack([make_row(e, st, obs, w=0, fp=0)]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY)

    def test_duplicate_step_identity_in_one_batch_rejected(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        svc(torch.stack([make_row(e, st, obs, w=0)]))
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table)
        out = svc(torch.stack([r, r]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY)
        self.assertEqual(int(out["error_code"][1]), ERR_STALE_IDENTITY)
        self.assertEqual(svc.cache_size(), 0, "rejected duplicates must evict the entry")

    def test_same_worker_id_different_run_id_does_not_alias(self):
        """Production identities namespace by driver run_id, not only worker_id."""
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        a = make_row(e, st, obs, w=0)
        b = make_row(e, st, obs, w=0)
        a["run_id"] = torch.tensor(0)
        b["run_id"] = torch.tensor(1)
        out = svc(torch.stack([a, b]))
        self.assertTrue(bool((out["error_code"] == 0).all()))
        self.assertEqual(svc.cache_size(), 2)

    def test_fabricated_identity_fails_closed(self):
        """A STEP whose identity was never encoded must not be served."""
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        cands, table = branch_table(e, st, obs)
        r = make_row(e, st, obs, w=0, op=1, level=0, cands=cands, table=table)
        for field in ("worker_id", "native_slot", "native_generation",
                      "episode_id", "seat", "request_local_id", "request_key"):
            r[field] = torch.tensor(999)
        out = svc(torch.stack([r]))
        self.assertEqual(int(out["error_code"][0]), ERR_NO_CACHE)

    def test_duplicate_identity_in_one_batch_is_rejected(self):
        """Two rows claiming the same identity must not overwrite each other."""
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        row = make_row(e, st, obs, w=0)
        out = svc(torch.stack([row, row]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY)
        self.assertEqual(int(out["error_code"][1]), ERR_STALE_IDENTITY)
        self.assertEqual(svc.cache_size(), 0,
                         "a duplicate batch must not leave a half-written cache entry")

    def test_wrong_weight_fingerprint_rejected(self):
        """A client serving different weights must be refused, not silently used."""
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        wrong = (default_fingerprint() + 12345) & ((1 << 63) - 1)
        out = svc(torch.stack([make_row(e, st, obs, w=0, fp=wrong)]))
        self.assertEqual(int(out["error_code"][0]), ERR_STALE_IDENTITY)
        self.assertEqual(svc.cache_size(), 0)

    def test_response_carries_server_fingerprint(self):
        svc = BatchedConditionalScorer(device="cuda")
        e, st, obs = self.envs[0]
        out = svc(torch.stack([make_row(e, st, obs, w=0)]))
        self.assertEqual(int(out["policy_fingerprint"][0]), svc.fingerprint_id)
        self.assertEqual(svc.fingerprint_id, default_fingerprint())

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
