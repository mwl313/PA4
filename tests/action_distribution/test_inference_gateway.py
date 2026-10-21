"""G1-D: real TorchRL InferenceServer, two independent clients, staged prefix.

ProcessSlotTransport preallocates a fixed number of slots, so clients are created
ONCE and reused; calling client() more than num_slots times is an error.

Asserts:
  * the server process starts and stays alive;
  * two distinct clients each drive an ENCODE/STEP staged exchange;
  * caches are isolated per request_key;
  * request identity mismatches fail closed instead of reusing stale state;
  * every selected index lies inside the engine's real candidate range.
"""

import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch  # noqa: E402
from torchrl.modules.inference_server import InferenceServer  # noqa: E402

from pa4.actions.conditional_policy import build_branch_table  # noqa: E402
from pa4.actions.native_reference import NativeReference  # noqa: E402
from pa4.actions.server_factory import build_service  # noqa: E402
from pa4.actions.server_probe import (  # noqa: E402
    ERR_NO_CACHE,
    OP_ENCODE,
    OP_RESET,
    OP_STEP,
    request_spec,
    response_spec,
)

OBS_FLOAT = {"floats"}
OBS_LONG = {"categories", "role_ids", "side_ids"}
OBS_KEYS = ("token_mask", "categories", "category_known", "floats", "float_known",
            "flags", "flag_known", "role_ids", "side_ids")
NUM_SLOTS = 2


def _payload(spec, op, key, level=0, pv=1):
    td = spec.clone()
    td["op"] = torch.tensor(op)
    td["request_key"] = torch.tensor(key)
    td["policy_version"] = torch.tensor(pv)
    td["branch_level"] = torch.tensor(level)
    return td


def _encode(spec, key, obs, pv=1):
    td = _payload(spec, OP_ENCODE, key, 0, pv)
    for attr in OBS_KEYS:
        src = getattr(obs, attr).detach().cpu()[0]
        if attr in OBS_FLOAT:
            src = src.to(torch.float32)
        elif attr in OBS_LONG:
            src = src.to(torch.long)
        else:
            src = src.to(torch.bool)
        td[attr] = src
    return td


def _step(spec, key, level, candidates, layout, pv=1):
    td = _payload(spec, OP_STEP, key, level, pv)
    table = build_branch_table([candidates], layout)
    k = len(candidates)
    td["candidate_count"] = torch.tensor(k)
    td["action_ids"][:k] = table.action_ids[0, 0, :k]
    td["legal_mask"][:k] = table.mask[0, 0, :k]
    td["entity_token"][:k] = table.entity_token[0, 0, :k]
    td["move_token"][:k] = table.move_token[0, 0, :k]
    return td


class InferenceGatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.types.observation import ObservationLayout

        cls.layout = ObservationLayout()
        cls.ref = NativeReference()
        h = cls.ref.reset([0], [1], [(7, 8, 9, 10)], [(0, 1)])
        cls.slot, cls.gen = h[0]
        cls.obs = cls.ref.observe_batch([(cls.slot, cls.gen)], [0]).to("cpu")

        cls.req_sp = request_spec()
        cls.res_sp = response_spec()
        cls.server = InferenceServer(
            model=None, policy_factory=build_service, transport="process_slot",
            service_backend="process", request_spec=cls.req_sp,
            response_spec=cls.res_sp, num_clients=NUM_SLOTS, max_batch_size=8,
        )
        cls.server.start()
        # Allocate every slot exactly once and reuse them across tests.
        cls.clients = cls.server.clients(NUM_SLOTS)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.server.shutdown()
        except Exception:
            pass

    def test_server_is_alive(self):
        self.assertTrue(bool(self.server.is_alive))

    def test_two_independent_clients(self):
        self.assertEqual(len(self.clients), NUM_SLOTS)
        self.assertIsNot(self.clients[0], self.clients[1])

    def test_staged_prefix_completes_via_server(self):
        client = self.clients[0]
        ack = client.submit(_encode(self.req_sp, 1, self.obs)).result()
        self.assertEqual(int(ack["error_code"]), 0)
        self.assertGreaterEqual(int(ack["cache_size"]), 1)

        prefix, total = [], 0.0
        for level in range(4):
            cands = self.ref.candidates(self.slot, self.gen, 0, prefix)
            out = client.submit(_step(self.req_sp, 1, level, cands, self.layout)).result()
            self.assertEqual(int(out["error_code"]), 0, f"level {level} failed")
            idx = int(out["selected_index"])
            self.assertTrue(0 <= idx < len(cands), "index outside real candidates")
            prefix.append(cands[idx])
            total += float(out["branch_logprob"])
        self.assertEqual(len(prefix), 4)
        client.submit(_payload(self.req_sp, OP_RESET, 1)).result()

    def test_two_clients_different_requests(self):
        c0, c1 = self.clients
        a0 = c0.submit(_encode(self.req_sp, 10, self.obs)).result()
        a1 = c1.submit(_encode(self.req_sp, 11, self.obs)).result()
        self.assertEqual(int(a0["error_code"]), 0)
        self.assertEqual(int(a1["error_code"]), 0)
        self.assertGreaterEqual(int(a1["cache_size"]), 2, "caches not isolated per request")

        for key, cl in ((10, c0), (11, c1)):
            cands = self.ref.candidates(self.slot, self.gen, 0, [])
            out = cl.submit(_step(self.req_sp, key, 0, cands, self.layout)).result()
            self.assertEqual(int(out["error_code"]), 0)
            self.assertEqual(int(out["request_key"]), key, "response key mismatch")
            cl.submit(_payload(self.req_sp, OP_RESET, key)).result()

    def test_wrong_level_fails_closed(self):
        client = self.clients[0]
        client.submit(_encode(self.req_sp, 20, self.obs)).result()
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        out = client.submit(_step(self.req_sp, 20, 2, cands, self.layout)).result()
        self.assertEqual(int(out["error_code"]), ERR_NO_CACHE)
        client.submit(_payload(self.req_sp, OP_RESET, 20)).result()

    def test_unknown_request_key_fails_closed(self):
        client = self.clients[0]
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        out = client.submit(_step(self.req_sp, 999, 0, cands, self.layout)).result()
        self.assertEqual(int(out["error_code"]), ERR_NO_CACHE)

    def test_wrong_policy_version_fails_closed(self):
        client = self.clients[0]
        client.submit(_encode(self.req_sp, 30, self.obs, pv=1)).result()
        cands = self.ref.candidates(self.slot, self.gen, 0, [])
        out = client.submit(_step(self.req_sp, 30, 0, cands, self.layout, pv=2)).result()
        self.assertEqual(int(out["error_code"]), ERR_NO_CACHE)
        client.submit(_payload(self.req_sp, OP_RESET, 30)).result()


if __name__ == "__main__":
    unittest.main(verbosity=2)
