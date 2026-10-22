"""G3-D: transport and lifetime correctness.

Forces the failure modes the design names explicitly and asserts each fails
closed WITHOUT contaminating other requests.
"""

import gc
import sys
import unittest

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch
from torchrl.modules.inference_server import InferenceServer

from pa4.actions.batched_factory_instrumented import build_instrumented_service
from pa4.actions.batched_scorer import request_spec, response_spec
from pa4.env.native_env import PA4EnvBase

from tests.collector.test_batching import branch_table, make_row


class TransportLifetimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e = PA4EnvBase(batch_size=1, seed=3)
        e.reset()
        st = e._slots[0]
        cls.env, cls.st, cls.obs = e, st, e._observe_one(st, 0)

    def _server(self, n_slots=4):
        srv = InferenceServer(
            model=None, policy_factory=build_instrumented_service,
            transport="process_slot", service_backend="process",
            request_spec=request_spec(), response_spec=response_spec(),
            num_clients=n_slots, max_batch_size=8, min_batch_size=1, timeout=0.0,
        )
        srv.start()
        return srv

    def test_clients_have_distinct_slots(self):
        """The defect that blocked P3: client() caches one slot."""
        srv = self._server()
        try:
            cs = srv.clients(4)
            ids = [c.client_id for c in cs]
            self.assertEqual(len(set(ids)), 4,
                             f"clients share slots: {ids}")
        finally:
            srv.shutdown()

    def test_multi_client_response_permutation_is_safe(self):
        """Four concurrent requests must each receive their own answer."""
        srv = self._server()
        try:
            cs = srv.clients(4)
            for w in range(4):
                cs[w].submit(make_row(self.env, self.st, self.obs, w=w)).result()
            futs = []
            for w in range(4):
                cands, table = branch_table(self.env, self.st, self.obs)
                futs.append((w, cs[w].submit(make_row(
                    self.env, self.st, self.obs, w=w, op=1, level=0,
                    cands=cands, table=table))))
            # Resolve in REVERSE order: a response must follow its own request,
            # not completion order.
            for w, f in reversed(futs):
                out = f.result()
                self.assertEqual(int(out["error_code"]), 0)
                self.assertEqual(int(out["request_key"]), w,
                                 "response key does not match its request")
        finally:
            srv.shutdown()

    def test_clean_shutdown_after_use(self):
        srv = self._server(2)
        cs = srv.clients(2)
        cs[0].submit(make_row(self.env, self.st, self.obs, w=0)).result()
        srv.shutdown()
        self.assertFalse(bool(srv.is_alive))

    def test_shutdown_with_inflight_request_does_not_hang(self):
        srv = self._server(2)
        cs = srv.clients(2)
        cs[0].submit(make_row(self.env, self.st, self.obs, w=0))  # never read
        srv.shutdown()          # must return, not deadlock
        self.assertFalse(bool(srv.is_alive))

    def test_repeated_start_stop_cycles(self):
        for _ in range(2):
            srv = self._server(2)
            cs = srv.clients(2)
            cs[0].submit(make_row(self.env, self.st, self.obs, w=0)).result()
            srv.shutdown()
            del srv
            gc.collect()

    def test_one_inflight_per_client_is_enforced(self):
        srv = self._server(2)
        try:
            cs = srv.clients(2)
            cs[0].submit(make_row(self.env, self.st, self.obs, w=0))
            with self.assertRaises(RuntimeError):
                cs[0].submit(make_row(self.env, self.st, self.obs, w=1))
        finally:
            srv.shutdown()


class FailClosedTest(unittest.TestCase):
    """Invalid input must raise, never silently become a successful match."""

    def test_illegal_action_rejected_by_env(self):
        e = PA4EnvBase(batch_size=1, seed=5)
        e.reset()
        pkt = e.legal_action_packet()
        pkt["action"][0, 0, 0] = torch.tensor([255, 255, 255, 127, 255, 255])
        with self.assertRaises(RuntimeError):
            e.step(pkt)

    def test_stale_request_id_rejected(self):
        e = PA4EnvBase(batch_size=1, seed=6)
        e.reset()
        pkt = e.legal_action_packet()
        pkt["action_request_id"] = pkt["action_request_id"] + 5
        with self.assertRaises(RuntimeError):
            e.step(pkt)

    def test_stale_generation_rejected(self):
        e = PA4EnvBase(batch_size=1, seed=7)
        e.reset()
        pkt = e.legal_action_packet()
        pkt["action_native_generation"] = pkt["action_native_generation"] + 9
        with self.assertRaises(RuntimeError):
            e.step(pkt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
