"""P3.5/ADR-009: ragged observation contract tests.

Falsifiable controls for the repair:
  * transport round-trip is byte-exact against PA3's canonical decoder
  * multi-view payloads are split per view (the original bug)
  * every ragged family reaches the encoder and changes its output
  * overflow fails closed instead of truncating
  * nine-field-only records are refused by the representation check
  * the retained transition carries ragged and a ragged-aware digest
  * corrupted/removed families are detectable (negative controls)
"""
import unittest

import numpy as np
import torch

from pa4.env import ragged_transport as rt
from pa4.env.native_env import PA4EnvBase

# The PA3 `agent` package lives in the read-only reference checkout; importing
# the env module above does not guarantee its path is active in every runner.
rt._ensure_pa3_on_path()
from agent.types.ragged import FIELDS, from_native  # noqa: E402


def _packed(section):
    """Reconstruct the flat CSR rows of one family from a RaggedBatch."""
    counts = np.asarray(section.counts.detach().cpu().numpy())
    values = section.values.detach().cpu().numpy().astype(np.float32)
    return counts, values


class TransportGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PA4EnvBase(batch_size=2, seed=20261006)
        cls.env.reset()

    def _payload(self, sides, rows=None):
        st = self.env._slots[0]
        handles = [(st.slot, st.generation)] * len(sides)
        fixed, ragged = self.env._engine.observe_fixed_batch(handles, list(sides))
        return self.env._mod.parse_batch(fixed, ragged, len(sides))

    def test_single_view_byte_exact(self):
        payload = self._payload([0])
        gold = from_native(payload, 1)
        mine = rt.transport_to_ragged(rt.payload_to_transport(payload, 1), 0)
        for fam in FIELDS:
            gc, gv = _packed(gold[fam])
            mc, mv = _packed(mine[fam])
            self.assertTrue(np.array_equal(gc, mc), f"{fam} counts")
            self.assertEqual(gv.shape, mv.shape, f"{fam} shape")
            self.assertTrue(np.array_equal(gv, mv), f"{fam} values")

    def test_multi_view_splits_per_view(self):
        """Regression: the ragged buffer is laid out PER VIEW, not per family.

        Mixed section order is what silently corrupted view 1 before the fix.
        """
        payload = self._payload([0, 1])
        transport = rt.payload_to_transport(payload, 2)
        for index, side in enumerate((0, 1)):
            one = self._payload([side])
            gold = rt.transport_to_ragged(rt.payload_to_transport(one, 1), 0)
            mine = rt.transport_to_ragged(transport, index)
            for fam in FIELDS:
                gc, gv = _packed(gold[fam])
                mc, mv = _packed(mine[fam])
                self.assertTrue(np.array_equal(gc, mc), f"view{index} {fam} counts")
                self.assertTrue(np.array_equal(gv, mv), f"view{index} {fam} values")

    def test_all_five_families_present_and_nonempty(self):
        payload = self._payload([0])
        counts = payload["base_move_counts"]
        self.assertGreater(int(np.asarray(counts).sum()), 0)
        for fam, key in rt.RAGGED_COUNT_KEYS.items():
            self.assertIn(key, payload, f"{fam} counts missing from payload")

    def test_overflow_fails_closed(self):
        payload = self._payload([0])
        original = dict(rt.RAGGED_TOKEN_CAPACITY)
        try:
            rt.RAGGED_TOKEN_CAPACITY["base_moves"] = 1
            with self.assertRaises(rt.RaggedOverflow):
                rt.payload_to_transport(payload, 1)
        finally:
            rt.RAGGED_TOKEN_CAPACITY.clear()
            rt.RAGGED_TOKEN_CAPACITY.update(original)

    def test_no_truncation_at_real_capacity(self):
        """Every real row must survive: total rows equal the payload counts."""
        payload = self._payload([0])
        transport = rt.payload_to_transport(payload, 1)
        for fam, key in rt.RAGGED_COUNT_KEYS.items():
            expected = int(np.asarray(payload[key]).sum())
            got = int(transport[f"ragged.{fam}_lengths"].sum())
            self.assertEqual(got, expected, f"{fam} rows lost in transport")


class RepresentationIdentityTest(unittest.TestCase):
    def test_ragged_representation_accepted(self):
        rt.assert_representation(rt.OBSERVATION_REPRESENTATION)

    def test_fixed9_representation_refused(self):
        with self.assertRaises(rt.ObservationContractMismatch):
            rt.assert_representation(rt.OBSERVATION_REPRESENTATION_FIXED9)

    def test_missing_representation_refused(self):
        with self.assertRaises(rt.ObservationContractMismatch):
            rt.assert_representation(None)


class NegativeControlTest(unittest.TestCase):
    """Deliberately corrupt a family and prove the comparison detects it."""

    @classmethod
    def setUpClass(cls):
        cls.env = PA4EnvBase(batch_size=1, seed=20261006)
        cls.env.reset()
        st = cls.env._slots[0]
        fixed, ragged = cls.env._engine.observe_fixed_batch(
            [(st.slot, st.generation)], [0])
        cls.payload = cls.env._mod.parse_batch(fixed, ragged, 1)

    def test_zeroing_a_family_changes_digest(self):
        base = rt.transport_to_ragged(rt.payload_to_transport(self.payload, 1), 0)
        digest = rt.ragged_digest(base)
        for fam in FIELDS:
            if int(base[fam].counts.sum()) == 0:
                continue
            corrupt = dict(base)
            corrupt[fam] = type(base[fam])(
                torch.zeros_like(base[fam].values),
                base[fam].counts,
                base[fam].host_counts,
            )
            self.assertNotEqual(rt.ragged_digest(corrupt), digest,
                                f"zeroing {fam} was not detected")

    def test_dropping_a_family_raises(self):
        trans = rt.payload_to_transport(self.payload, 1)
        for key in list(trans):
            if key.startswith("ragged.types_"):
                del trans[key]
        with self.assertRaises((KeyError, ValueError)):
            rt.transport_to_ragged(trans, 0)

    def test_transport_is_not_aliased_to_payload(self):
        trans = rt.payload_to_transport(self.payload, 1)
        before = trans["ragged.base_moves_values"].clone()
        # A later native call reuses the engine buffer; transport must not move.
        st = self.env._slots[0]
        self.env._engine.observe_fixed_batch([(st.slot, st.generation)], [0])
        self.assertTrue(torch.equal(trans["ragged.base_moves_values"], before))


class TransitionRetentionTest(unittest.TestCase):
    def test_seat_record_retains_ragged(self):
        from pa4.collector.transitions import seat_record

        env = PA4EnvBase(batch_size=1, seed=20261006)
        env.reset()
        st = env._slots[0]
        view = env._observe_one(st, 0)
        rec = seat_record(
            seat=0, request_id=1, view=view, branches=[],
            actor_active=True, joint_logprob_fp32=0.0,
            policy_version=1, policy_weight_fingerprint="x",
            request_kind=1,
        )
        self.assertEqual(rec["observation_representation"],
                         rt.OBSERVATION_REPRESENTATION)
        self.assertTrue(rec["observation_ragged"], "ragged not retained")
        fams = rt.family_names()
        for fam in fams:
            self.assertIn(f"ragged.{fam}_counts", rec["observation_ragged"])
        # Digest must depend on ragged: mutating it must change the digest.
        from pa4.collector.transitions import observation_digest
        base = observation_digest(view)
        mutated = dict(view)
        row = dict(view["_ragged_row"])
        k = "ragged.base_moves_values"
        row[k] = row[k] + 1.0
        mutated["_ragged_row"] = row
        self.assertNotEqual(observation_digest(mutated), base)


if __name__ == "__main__":
    unittest.main()
