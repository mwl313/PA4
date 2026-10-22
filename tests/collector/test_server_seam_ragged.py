"""P3.5 F3: non-empty ragged through the REAL inference-server seam.

The unit tests elsewhere exercise the actor side and the encoder helper
separately. This test goes through the same objects as production:

    PA4EnvBase players TD
      -> run_batched._views_from_players
      -> StagedInference.encode_all
      -> InferenceServer (process_slot)
      -> BatchedConditionalScorer._obs_batch
      -> PA3 encoder

The instrumented service writes a digest of the encoder output and of the
ragged row it actually received. The test rebuilds the same observation
canonically with PA3's `from_native` / `ObservationBatch.from_native_payload`
and requires exact equality. Removing the ragged forwarding makes this test
fail; the falsifiable demonstration is recorded in the P3.5 receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from agent.model.config import PA3Config
from agent.model.pa3_model import PA3Model
from agent.types.observation import ObservationBatch
from agent.types.ragged import from_native
from pa4.actions.batched_factory_instrumented import build_instrumented_service
from pa4.actions.batched_scorer import (
    fingerprint_id, model_fingerprint, request_spec, response_spec,
)
from pa4.collector.batched_core import StagedInference
from pa4.collector.run_batched import _views_from_players
from pa4.env import ragged_transport as rt
from pa4.env.native_env import PA4EnvBase

FAMILIES = ("effects", "repertoire", "types", "base_moves", "move_effects")


def _tensor_feed(h, t):
    t = t.detach().to("cpu").contiguous()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())


def _encoded_digest(encoded):
    h = hashlib.sha256()
    for name in ("tokens", "global_repr", "token_mask"):
        h.update(name.encode())
        _tensor_feed(h, getattr(encoded, name))
    return h.hexdigest()


def _ragged_digest_from_batch(ragged: dict, n: int = 1) -> str:
    """Digest a PA3 RaggedBatch dict exactly as the probe service does."""
    h = hashlib.sha256()
    for family in FAMILIES:
        section = ragged[family]
        counts = torch.as_tensor(section.counts, dtype=torch.long)
        values = torch.as_tensor(section.values, dtype=torch.float32)
        if counts.dim() == 1:
            counts = counts.unsqueeze(0)
        if values.dim() == 3 and values.shape[0] == 1:
            values = values[0]
        for key, tensor in (
            (f"ragged.{family}_counts", counts),
            (f"ragged.{family}_values", values),
        ):
            h.update(key.encode())
            _tensor_feed(h, tensor)
    return h.hexdigest()


class ServerSeamRaggedTest(unittest.TestCase):
    def test_real_server_seam_receives_nonempty_ragged(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        env = PA4EnvBase(batch_size=1, seed=20261006)
        td = env.reset()
        st = env._slots[0]
        views = _views_from_players(td["players"], [(0, 0, None, None)])
        view = views[0]
        self.assertEqual(view["_observation_representation"],
                         rt.OBSERVATION_REPRESENTATION)
        nonempty = {f: int(view["_ragged_row"][f"ragged.{f}_counts"].sum())
                    for f in FAMILIES}
        self.assertGreater(sum(nonempty.values()), 0, "test view has no ragged rows")
        # The repair is about real features; base moves and move effects are
        # populated in every surveyed view. Repertoire can legitimately be empty
        # at a fresh preview, so it is not required here.
        for family in ("base_moves", "move_effects"):
            self.assertGreater(nonempty[family], 0, f"{family} unexpectedly empty")

        probe_path = tempfile.mktemp(prefix="pa4-ragged-seam-", suffix=".json")
        old_probe = os.environ.get("PA4_P3_ENCODE_PROBE_PATH")
        old_device = os.environ.get("PA4_POLICY_DEVICE")
        old_seed = os.environ.get("PA4_POLICY_SEED")
        os.environ["PA4_P3_ENCODE_PROBE_PATH"] = probe_path
        os.environ["PA4_POLICY_DEVICE"] = dev
        os.environ["PA4_POLICY_SEED"] = "20261006"

        from torchrl.modules.inference_server import InferenceServer

        srv = InferenceServer(
            model=None, policy_factory=build_instrumented_service,
            transport="process_slot", service_backend="process",
            request_spec=request_spec(), response_spec=response_spec(),
            num_clients=2, max_batch_size=64, min_batch_size=1, timeout=0.0,
        )
        srv.start()
        try:
            clients = srv.clients(2)
            inf = StagedInference(
                clients, request_spec(), 1, run_id_int=0,
                fingerprint_id=fingerprint_id(model_fingerprint()),
                result_timeout=300.0,
            )
            ident = (0, st.slot, st.generation, st.episode_id, 0, 0)
            inf.encode_all([(ident, view)])
            self.assertTrue(os.path.exists(probe_path),
                            "instrumented service wrote no encoder probe")
            with open(probe_path) as fh:
                probe = json.load(fh)
        finally:
            srv.shutdown()
            for key, value in (("PA4_P3_ENCODE_PROBE_PATH", old_probe),
                               ("PA4_POLICY_DEVICE", old_device),
                               ("PA4_POLICY_SEED", old_seed)):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            try:
                os.unlink(probe_path)
            except OSError:
                pass

        # Canonical PA3 path for the same native state.
        fixed, ragged_blob = env._engine.observe_fixed_batch(
            [(st.slot, st.generation)], [0])
        payload = env._mod.parse_batch(fixed, ragged_blob, 1)
        gold_ragged = from_native(payload, 1)
        gold_obs = ObservationBatch.from_native_payload(
            payload, dtype=torch.float32).to(dev)
        for family in ("base_moves", "move_effects"):
            self.assertGreater(int(gold_obs.ragged[family].counts.sum()), 0,
                               f"canonical {family} is empty")

        torch.manual_seed(20261006)
        model = PA3Model(PA3Config(initialization="random", seed=20261006))
        model = model.to(dev).eval()
        with torch.no_grad():
            gold_encoded = model.encode(gold_obs)

        self.assertEqual(probe["ragged"],
                         _ragged_digest_from_batch(gold_ragged, 1),
                         "server did not receive the canonical ragged payload")
        self.assertEqual(probe["encoded"], _encoded_digest(gold_encoded),
                         "server encoder output differs from the canonical path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
