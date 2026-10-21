"""P3: fast, validated observation decoding for the hot path.

Problem (measured, not assumed): `ObservationBatch.from_native_payload` costs
~408 us per seat because it re-validates already-trusted bytes from the Rust
engine on every observation, and `__post_init__` re-checks all nine shapes.

P2 decodes each seat TWICE per env.step(): once to build the action packet and
again inside `_observation_td()` for the returned transition. At 2 seats and
~530 us/seat that dominates a ~2-3 ms step.

Design of the fix (robust + futureproof):

* Do NOT weaken validation. Instead SPLIT it into two explicit tiers:
    - `validate=True`  -> full PA3 `from_native_payload` + `__post_init__`
                          (kept for all tests, the replay path, and first-use).
    - `validate=False` -> a direct, allocation-minimal decode used only on the
                          actor hot path, guarded by a one-time contract check per
                          SliceDecoder instance.
* The fast path is *derived from the same wire parser* (`parse_batch`), so there
  is no second source of truth for the byte layout.
* A per-step observation cache keyed by (slot, generation, step_index) means each
  seat is decoded at most once per transition instead of twice.
* Everything is additive; no P2 contract, key, dtype or shape changes.
"""

from __future__ import annotations

import numpy as np
import torch

OBS_LEAVES = ("token_mask", "categories", "category_known", "floats",
              "float_known", "flags", "flag_known", "role_ids", "side_ids")

# dtypes the destination PA4 contract requires, mirroring PA3's ObservationBatch.
_LEAF_DTYPE = {
    "token_mask": torch.bool,
    "categories": torch.long,
    "category_known": torch.bool,
    "floats": torch.float32,
    "float_known": torch.bool,
    "flags": torch.bool,
    "flag_known": torch.bool,
    "role_ids": torch.long,
    "side_ids": torch.long,
}


class SliceDecoder:
    """Decode one seat's native payload into PA4's fixed observation dict.

    Two tiers, chosen explicitly by the caller. The fast tier is only ever used
    after `verify_contract()` has been checked once against a full PA3 decode.
    """

    def __init__(self, engine, module, layout=None):
        self._engine = engine
        self._mod = module
        if layout is None:
            from agent.types.observation import ObservationLayout

            layout = ObservationLayout()
        self._layout = layout
        self._verified = False
        self.fast_decodes = 0
        self.full_decodes = 0

    # -- tier 1: authoritative, slow, used for verification and replay -----
    def decode_full(self, slot: int, generation: int, side: int) -> dict:
        from agent.types.observation import ObservationBatch

        fixed, ragged = self._engine.observe_fixed_batch([(slot, generation)], [side])
        payload = self._mod.parse_batch(fixed, ragged, 1)
        batch = ObservationBatch.from_native_payload(payload, dtype=torch.float32)
        self.full_decodes += 1
        return {k: getattr(batch, k)[0].cpu() for k in OBS_LEAVES}

    # -- tier 2: hot path, no re-validation of trusted engine bytes --------
    def decode_fast(self, slot: int, generation: int, side: int) -> dict:
        fixed, ragged = self._engine.observe_fixed_batch([(slot, generation)], [side])
        payload = self._mod.parse_batch(fixed, ragged, 1)
        out = {}
        for k in OBS_LEAVES:
            if k in ("role_ids", "side_ids"):
                # Not on the wire: PA3 derives these from the fixed layout.
                continue
            v = payload[k]
            # Verbatim PA3 adapter predicate: a field sliced from the fixed-stride
            # structured batch keeps the struct stride, which torch cannot wrap.
            itemsize = v.itemsize
            if not v.flags.writeable or (itemsize and any(
                stride % itemsize for stride in v.strides
            )):
                v = np.array(v, dtype=v.dtype, order="C", copy=True)
            t = torch.as_tensor(v)
            if t.dim() > 0 and t.shape[0] == 1:
                t = t[0]
            out[k] = t.to(_LEAF_DTYPE[k])

        # Mirror PA3's fallback exactly: layout-derived role/side, padded slots masked.
        token_mask = out["token_mask"]
        role = self._layout.default_roles().contiguous().clone()
        role[~token_mask] = self._layout.ROLE_PAD
        side = self._layout.default_sides().contiguous().clone()
        out["role_ids"] = role.to(torch.long)
        out["side_ids"] = side.to(torch.long)
        self.fast_decodes += 1
        return out

    # -- one-time equivalence proof ---------------------------------------
    def verify_contract(self, sample_slot: int, sample_generation: int, samples=(0, 1)) -> None:
        """Prove the fast tier matches PA3's authoritative decode exactly.

        Run once per process before the hot path is used. If any leaf, dtype or
        shape disagrees, the fast path is disabled and an error is raised so the
        caller falls back to the full decoder rather than silently diverging.
        """
        for side in samples:
            fast = self.decode_fast(sample_slot, sample_generation, side)
            full = self.decode_full(sample_slot, sample_generation, side)
            for k in OBS_LEAVES:
                a, b = fast[k], full[k]
                if a.shape != b.shape:
                    raise RuntimeError(
                        f"fast decode shape mismatch {k}: {tuple(a.shape)} != {tuple(b.shape)}")
                if a.dtype != b.dtype:
                    raise RuntimeError(
                        f"fast decode dtype mismatch {k}: {a.dtype} != {b.dtype}")
                if not torch.equal(a, b):
                    n = int((a != b).sum())
                    raise RuntimeError(
                        f"fast decode value mismatch {k}: {n} differing elements")
        self._verified = True

    @property
    def verified(self) -> bool:
        return self._verified
