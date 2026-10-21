"""Server-side guards added after model review (G1-E on the wire path).

The reviewer correctly observed that build_branch_table() raising on overflow
does not protect the transport: a fixed (64,6) buffer would silently slice
candidates past 64, and a rejected STEP left its cache entry behind.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402

from pa4.actions.server_probe import (  # noqa: E402
    CANDIDATE_CAPACITY,
    ERR_COUNT_MISMATCH,
    ERR_NO_CACHE,
    ERR_OVERFLOW,
    ConditionalScorerService,
    request_spec,
)

OBS_KEYS = ("token_mask", "categories", "category_known", "floats", "float_known",
            "flags", "flag_known", "role_ids", "side_ids")


def _row(spec, obs, op, key, level=0, pv=1):
    td = spec.clone()
    td["op"] = torch.tensor(op)
    td["request_key"] = torch.tensor(key)
    td["policy_version"] = torch.tensor(pv)
    td["branch_level"] = torch.tensor(level)
    if obs is not None:
        for a in OBS_KEYS:
            src = getattr(obs, a).detach().cpu()[0]
            if a == "floats":
                src = src.to(torch.float32)
            elif a in ("categories", "role_ids", "side_ids"):
                src = src.to(torch.long)
            else:
                src = src.to(torch.bool)
            td[a] = src
    return torch.stack([td])


class ServerGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pa4.actions.native_reference import NativeReference

        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        slot, gen = h[0]
        cls.obs = cls.ref.observe_batch([(slot, gen)], [0]).to("cpu")
        cls.spec = request_spec()
        cls.svc = ConditionalScorerService(seed=20261006)

    def _encode(self, key=1):
        return self.svc(_row(self.spec, self.obs, op=0, key=key))

    def test_overflow_on_wire_path_is_rejected(self):
        ack = self._encode(40)
        self.assertEqual(int(ack["error_code"]), 0)
        td = self.spec.clone()
        td["op"] = torch.tensor(1)
        td["request_key"] = torch.tensor(40)
        td["policy_version"] = torch.tensor(1)
        td["branch_level"] = torch.tensor(0)
        td["candidate_count"] = torch.tensor(CANDIDATE_CAPACITY + 1)
        td["legal_mask"] = torch.ones(CANDIDATE_CAPACITY, dtype=torch.bool)
        out = self.svc(torch.stack([td]))
        self.assertEqual(int(out["error_code"]), ERR_OVERFLOW)

    def test_candidate_count_mask_mismatch_is_rejected(self):
        self._encode(41)
        td = self.spec.clone()
        td["op"] = torch.tensor(1)
        td["request_key"] = torch.tensor(41)
        td["policy_version"] = torch.tensor(1)
        td["branch_level"] = torch.tensor(0)
        td["candidate_count"] = torch.tensor(5)
        td["legal_mask"] = torch.ones(3, dtype=torch.bool)  # disagrees with count
        out = self.svc(torch.stack([td]))
        self.assertEqual(int(out["error_code"]), ERR_COUNT_MISMATCH)

    def test_rejected_step_evicts_cache(self):
        self._encode(42)
        self.assertGreaterEqual(self.svc.cache_size(), 1)
        td = self.spec.clone()
        td["op"] = torch.tensor(1)
        td["request_key"] = torch.tensor(42)
        td["policy_version"] = torch.tensor(1)
        td["branch_level"] = torch.tensor(3)  # out of order
        td["candidate_count"] = torch.tensor(2)
        td["legal_mask"] = torch.tensor([True, True] + [False] * (CANDIDATE_CAPACITY - 2))
        self.svc(torch.stack([td]))
        # entry must be gone: a follow-up STEP on the same key cannot resume
        td["branch_level"] = torch.tensor(0)
        out = self.svc(torch.stack([td]))
        self.assertEqual(int(out["error_code"]), ERR_NO_CACHE)

    def test_capacity_constant_matches_pa3(self):
        self.assertEqual(CANDIDATE_CAPACITY, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
