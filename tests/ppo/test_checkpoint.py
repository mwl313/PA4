"""P4-D — checkpoint completeness, atomicity, restore parity and negatives."""
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

from pa4.learner import checkpoint as ck
from pa4.learner.checkpoint import (CheckpointError, CheckpointState,
                                    load_checkpoint, restore_into,
                                    save_checkpoint, verify_restore_parity)

SEED = 20261006


def build_model(seed=SEED):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def make_trained(it=1):
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-5)
    loss = sum((p ** 2).sum() for p in model.parameters())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    return model, opt


def make_state(model, opt, iteration=1, ledger=None):
    return CheckpointState(
        iteration=iteration, policy_version=iteration + 1,
        committed_matches=16 * iteration, optimizer_steps=4 * iteration,
        model_state=model.state_dict(),
        optimizer_state=opt.state_dict(),
        scheduler_state={"lr": 3e-4, "committed_matches": 16 * iteration},
        rng_state=ck._rng_state(),
        match_ledger=ledger if ledger is not None else [f"m{iteration}"],
        counters={"dropped": 0},
        provenance={"pa4_git_sha": "test"},
    )


class CheckpointCompletenessTest(unittest.TestCase):
    def test_not_model_only(self):
        model, opt = make_trained()
        d = tempfile.mkdtemp()
        save_checkpoint(make_state(model, opt), d)
        loaded = load_checkpoint(d)
        self.assertIsNotNone(loaded.optimizer_state)
        self.assertTrue(loaded.optimizer_state["state"], "adam moments missing")
        self.assertIsNotNone(loaded.rng_state)
        self.assertEqual(loaded.match_ledger, ["m1"])
        self.assertIsNotNone(loaded.scheduler_state)

    def test_restore_is_exact(self):
        model, opt = make_trained()
        d = tempfile.mkdtemp()
        save_checkpoint(make_state(model, opt), d)
        loaded = load_checkpoint(d)

        fresh = build_model(seed=1)
        fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-9)
        restore_into(loaded, fresh, fresh_opt)

        for k, v in fresh.state_dict().items():
            self.assertTrue(torch.equal(loaded.model_state[k], v.cpu()), f"{k} differs")

        a = loaded.optimizer_state["state"]
        b = fresh_opt.state_dict()["state"]
        self.assertEqual(set(a), set(b))
        for pid in a:
            for key in a[pid]:
                if torch.is_tensor(a[pid][key]):
                    self.assertTrue(
                        torch.equal(a[pid][key], b[pid][key]),
                        f"adam {pid}.{key} differs")

    def test_restore_parity_report_self(self):
        model, opt = make_trained()
        d = tempfile.mkdtemp()
        save_checkpoint(make_state(model, opt), d)
        a = load_checkpoint(d)
        b = load_checkpoint(d)
        rep = verify_restore_parity(a, b)
        self.assertTrue(rep["parity_pass"], rep)
        self.assertEqual(rep["max_model_delta"], 0.0)

    def test_parity_detects_different_state(self):
        model, opt = make_trained()
        d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        save_checkpoint(make_state(model, opt, iteration=1), d1)
        save_checkpoint(make_state(model, opt, iteration=2), d2)
        rep = verify_restore_parity(load_checkpoint(d1), load_checkpoint(d2))
        self.assertFalse(rep["parity_pass"])
        self.assertFalse(rep["counters_match"])

    def test_previous_checkpoint_retained(self):
        model, opt = make_trained()
        d = tempfile.mkdtemp()
        p1 = save_checkpoint(make_state(model, opt, iteration=1), d)
        p2 = save_checkpoint(make_state(model, opt, iteration=2), d)
        self.assertTrue(os.path.exists(p1), "old checkpoint was deleted")
        self.assertTrue(os.path.exists(p2))
        # pointer now names the newest
        with open(os.path.join(d, "latest.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["iteration"], 2)


class CheckpointNegativeTest(unittest.TestCase):
    def setUp(self):
        self.model, self.opt = make_trained()
        self.dir = tempfile.mkdtemp()
        self.path = save_checkpoint(make_state(self.model, self.opt), self.dir)

    def test_model_only_rejected(self):
        p = os.path.join(self.dir, "model_only.pt")
        torch.save({"schema": ck.CHECKPOINT_SCHEMA,
                    "model_state": self.model.state_dict()}, p)
        with self.assertRaises(CheckpointError):
            load_checkpoint(p)

    def test_tampered_weights_rejected(self):
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        key = sorted(payload["model_state"])[0]
        payload["model_state"][key] = payload["model_state"][key].clone()
        payload["model_state"][key].view(-1)[0] += 1.0
        p = os.path.join(self.dir, "tampered.pt")
        torch.save(payload, p)
        with self.assertRaises(CheckpointError):
            load_checkpoint(p)

    def test_truncated_rejected(self):
        p = os.path.join(self.dir, "trunc.pt")
        with open(self.path, "rb") as fh:
            raw = fh.read()
        with open(p, "wb") as fh:
            fh.write(raw[: len(raw) // 2])
        with self.assertRaises(CheckpointError):
            load_checkpoint(p)

    def test_foreign_schema_rejected(self):
        p = os.path.join(self.dir, "foreign.pt")
        torch.save({"schema": "other.v1"}, p)
        with self.assertRaises(CheckpointError):
            load_checkpoint(p)

    def test_null_optimizer_rejected_on_write(self):
        state = make_state(self.model, self.opt)
        state.optimizer_state = None
        with self.assertRaises(CheckpointError):
            save_checkpoint(state, tempfile.mkdtemp())

    def test_bad_pointer_sha_rejected(self):
        d = tempfile.mkdtemp()
        p = save_checkpoint(make_state(self.model, self.opt), d)
        with open(os.path.join(d, "latest.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["sha256"] = "0" * 64
        with open(os.path.join(d, "latest.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        with self.assertRaises(CheckpointError):
            load_checkpoint(d)

    def test_missing_pointer_rejected(self):
        with self.assertRaises(CheckpointError):
            load_checkpoint(tempfile.mkdtemp())


class RngRoundTripTest(unittest.TestCase):
    def test_rng_state_restored(self):
        """Restoring a checkpoint must reproduce the SAVED RNG stream.

        The contract is round-trip fidelity, not equality with an unrelated
        seed: capture the stream at save time, consume randomness, restore, and
        confirm the stream resumes identically.
        """
        model, opt = make_trained()
        d = tempfile.mkdtemp()

        torch.manual_seed(1234)
        save_checkpoint(make_state(model, opt), d)
        expected = torch.randn(4)          # what the saved stream produces
        torch.randn(8)                     # disturb the live stream

        loaded = load_checkpoint(d)
        restore_into(loaded, build_model(seed=1))
        got = torch.randn(4)
        self.assertTrue(torch.equal(expected, got),
                        "the saved RNG stream was not restored")

    def test_python_and_numpy_rng_restored(self):
        import random as _random

        import numpy as np

        model, opt = make_trained()
        d = tempfile.mkdtemp()
        _random.seed(99)
        np.random.seed(99)
        save_checkpoint(make_state(model, opt), d)
        exp_py = [_random.random() for _ in range(3)]
        exp_np = np.random.rand(3).tolist()

        _random.seed(1)
        np.random.seed(1)
        restore_into(load_checkpoint(d), build_model(seed=1))

        self.assertEqual([_random.random() for _ in range(3)], exp_py)
        self.assertEqual(np.random.rand(3).tolist(), exp_np)


if __name__ == "__main__":
    unittest.main()
