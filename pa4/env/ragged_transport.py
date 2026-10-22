"""P3.5 — canonical ragged observation transport for PA4.

ADR-009. PA3's encoder adds all five "ragged" feature families before the
encoder, and the pinned native engine emits them in every observation payload.
Before P3.5, PA4 decoded them with the canonical `parse_batch` and then threw
them away, so `ObservationBatch.__post_init__` silently substituted empty
ragged sections and the policy was blind to movesets, PP, types, status and
secondary move effects.

This module owns the *only* place ragged data is converted between:

    native blob (bytes)  ->  numpy views (PA3 parse_batch)
                         ->  PA4 transport tensors (pad + counts + lengths)
                         ->  agent.types.ragged.RaggedBatch (encoder input)

Design rules (P3.5 requirements 4 and 5):

* The canonical PA3 decoder is reused; nothing here reimplements parsing.
* Counts are the authority. Padding rows are never reachable by the encoder
  because `RaggedBatch` consumes CSR `values[counts]`, and our packed tensors
  are unpacked strictly by `counts`.
* Capacity is a *transport* bound with fail-closed overflow. It is never a
  semantic truncation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

#: PA3 reference checkout (read-only). The `agent` package that owns
#: `agent.types.ragged` and the encoder lives here.
DEFAULT_PA3_REF = "/home/pa4/projects/_pa3_ref"
DEFAULT_PY_PKG = "/home/pa4/.pa3_build/engine/python"


def _ensure_pa3_on_path() -> None:
    """Make the pinned PA3 packages importable.

    Mirrors `pa4.env.native_env._ensure_paths` so this module behaves the same
    whether it is imported first or after the environment module.
    """
    for p in (os.environ.get("PA4_ENGINE_SO_DIR", DEFAULT_PY_PKG),
              os.environ.get("PA4_PA3_REF", DEFAULT_PA3_REF)):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _fields():
    _ensure_pa3_on_path()
    from agent.types.ragged import FIELDS  # noqa: PLC0415

    return FIELDS

# Versioned representation identity. Records that predate the repair carry
# OBSERVATION_REPRESENTATION_FIXED9 and must be refused at P4 ingest.
OBSERVATION_REPRESENTATION = "pa4.observation.ragged.v2"
OBSERVATION_REPRESENTATION_FIXED9 = "pa4.observation.fixed9.v1"

#: The nine fixed leaves. Kept named for backward compatibility of call sites.
OBS_LEAVES = (
    "token_mask", "categories", "category_known", "floats",
    "float_known", "flags", "flag_known", "role_ids", "side_ids",
)

#: Family -> (counts key on the PA3 payload, numpy structured dtype key, width).
#: Widths come from PA3's ragged contract; we read them from FIELDS at import.
RAGGED_COUNT_KEYS = {
    "effects": "effect_counts",
    "repertoire": "repertoire_counts",
    "types": "type_counts",
    "base_moves": "base_move_counts",
    "move_effects": "move_effect_counts",
}

#: Transport padding capacity per token, per family.
#:
#: These are *padding* capacities, not semantic limits. Measured maxima over
#: 24,000 real seat-views (docs/phases/P3.5/evidence/capacity_survey.json):
#:   effects      max 2/token   (0 in 3030 views, 1 in 20960, 2 in 10)
#:   repertoire   max 4/token
#:   types        max 2/token   (structurally 1-2: dual-type)
#:   base_moves   max 4/token   (structurally 4: a Pokemon knows <= 4 moves)
#:   move_effects max 3/token
#: Capacity is set to observed max + a safety margin. Overflow raises
#: (fail closed) rather than truncating; raising a capacity requires re-running
#: the survey. Oversizing these is expensive: the padded tensors are shipped on
#: the inference wire, so every spare slot costs round-trip bytes.
RAGGED_TOKEN_CAPACITY = {
    "effects": 4,
    "repertoire": 6,
    "types": 3,
    "base_moves": 5,
    "move_effects": 5,
}

TOKENS = 96


class RaggedOverflow(RuntimeError):
    """A family exceeded its transport padding capacity.

    Raised instead of truncating. P3.5 requirement 5 forbids silent loss of
    variable-length data; raising capacity (and re-measuring) is the only
    sanctioned response.
    """


class ObservationContractMismatch(RuntimeError):
    """A record's observation representation does not match what this code needs.

    P3.5 requirement 8: nine-field-only records must never enter the repaired
    PPO training pipeline.
    """


def family_widths() -> dict[str, int]:
    """Row width of each family, taken from PA3's own ragged contract."""
    return {name: len(fields) for name, fields in _fields().items()}


def family_names() -> tuple[str, ...]:
    return tuple(_fields())


# ---------------------------------------------------------------------------
# native payload -> transport tensors
# ---------------------------------------------------------------------------
def _structured_view(flat: np.ndarray, dtype, width: int) -> np.ndarray:
    """View a flat ragged byte buffer region as structured rows.

    The buffer arrives as a slice of the whole ragged payload, so it does not
    own its memory. `np.ascontiguousarray` is NOT enough here: for a slice of a
    larger contiguous array it returns a view, and `.view(dtype)` then
    reinterprets from the BASE buffer's origin, silently reading unrelated
    bytes. An explicit owning copy is required for correctness.
    """
    owned = np.array(flat, dtype=np.uint8, order="C", copy=True)
    if owned.size % dtype.itemsize:
        raise ValueError(
            f"ragged region of {owned.size} bytes is not a multiple of "
            f"the {dtype.itemsize}-byte row size")
    return owned.view(dtype)


def payload_to_transport(payload: dict, n: int) -> dict:
    """Convert a PA3 `parse_batch` payload (n views) into transport tensors.

    Returns a dict of plain CPU tensors:

        ragged.{family}_counts  int64   [n, 96]
        ragged.{family}_values  float32 [n, 96, cap, width]
        ragged.{family}_lengths int64   [n]
        ragged.{family}_overflow bool   [n]

    The payload's ragged buffer concatenates every view's five sections in
    request order, exactly as `split_batch_ragged_all` consumes it. Offsets are
    recomputed here from the authoritative count arrays on every call — no
    cached offset is trusted.
    """
    FIELDS = _fields()
    import pa3_engine.observation as obs_mod  # noqa: PLC0415

    dtype_for = {
        "effects": obs_mod.EFFECT,
        "repertoire": obs_mod.REPERTOIRE,
        "types": obs_mod.TYPE,
        "base_moves": obs_mod.BASE_MOVE,
        "move_effects": obs_mod.MOVE_EFFECT,
    }

    out: dict[str, torch.Tensor] = {}
    flat = np.frombuffer(payload["ragged"], dtype=np.uint8)

    FAMILIES = ("effects", "repertoire", "types", "base_moves", "move_effects")
    dtype_for = {
        "effects": obs_mod.EFFECT,
        "repertoire": obs_mod.REPERTOIRE,
        "types": obs_mod.TYPE,
        "base_moves": obs_mod.BASE_MOVE,
        "move_effects": obs_mod.MOVE_EFFECT,
    }

    # The native ragged buffer is laid out PER VIEW: view 0's five sections,
    # then view 1's five sections, ... (see pa3_engine.observation's
    # split_batch_ragged). We must walk view-major, not family-major.
    per_family: dict[str, dict[str, np.ndarray]] = {f: {} for f in FAMILIES}
    offset = 0
    for b in range(n):
        for family in FAMILIES:
            counts = np.asarray(payload[RAGGED_COUNT_KEYS[family]])
            if counts.shape != (n, TOKENS):
                raise ValueError(
                    f"ragged counts for {family} have shape {counts.shape}, "
                    f"expected {(n, TOKENS)}")
            if not np.issubdtype(counts.dtype, np.integer):
                raise ValueError(f"ragged counts for {family} are not integers")
            if np.any(counts < 0):
                raise ValueError(f"negative ragged count for {family}")

            dt = dtype_for[family]
            width = len(FIELDS[family])
            total_rows = int(counts[b].sum())
            nbytes = total_rows * dt.itemsize
            chunk = flat[offset:offset + nbytes]
            if chunk.size != nbytes:
                raise ValueError(
                    f"ragged buffer truncated for {family} view {b}: "
                    f"needed {nbytes} got {chunk.size}")
            offset += nbytes

            cap = RAGGED_TOKEN_CAPACITY[family]
            overflow = counts > cap
            if bool(overflow.any()):
                worst = int(counts.max())
                raise RaggedOverflow(
                    f"ragged family {family!r} has {worst} rows for one token, above "
                    f"the transport capacity {cap}. Refusing to truncate; raise "
                    f"RAGGED_TOKEN_CAPACITY[{family!r}] and re-run the capacity evidence.")

            rows = _structured_view(chunk, dt, width)
            # Flatten structured -> plain float32 using PA3's OWN canonical helper
            # so sub-array fields (move_effects.boosts[7]) expand identically.
            from agent.types.ragged import _matrix  # noqa: PLC0415

            cols = _matrix(rows, family)
            if cols.shape != (total_rows, width):
                raise ValueError(
                    f"family {family!r} flattened to {cols.shape}, "
                    f"expected {(total_rows, width)}")
            per_family[family][b] = cols

    # Assemble padded per-view tensors with a vectorised scatter.
    for family in FAMILIES:
        counts = np.asarray(payload[RAGGED_COUNT_KEYS[family]]).astype(np.int64)
        cap = RAGGED_TOKEN_CAPACITY[family]
        width = len(FIELDS[family])
        values = np.zeros((n, TOKENS, cap, width), dtype=np.float32)

        # Build destination indices without any per-token Python loop:
        # row i belongs to (view, token, offset-within-token).
        total = int(counts.sum())
        if total:
            per_token = counts  # [n, tokens]
            view_of_row = np.repeat(np.arange(n), per_token.sum(axis=1))
            token_of_row = np.repeat(
                np.tile(np.arange(TOKENS), (n, 1)).ravel(), per_token.ravel())
            # within-token offset: 0..k-1 for each (view, token) slot
            k_flat = per_token.ravel()
            offsets = np.arange(total) - np.repeat(
                np.cumsum(k_flat) - k_flat, k_flat)
            flat_cols = (np.concatenate([per_family[family][b] for b in range(n)], axis=0)
                         if total else None)
            if flat_cols.shape[0] != total:
                raise ValueError(f"ragged row accounting mismatch for {family}")
            values[view_of_row, token_of_row, offsets] = flat_cols

        out[f"ragged.{family}_counts"] = torch.from_numpy(counts.copy())
        out[f"ragged.{family}_values"] = torch.from_numpy(values)
        out[f"ragged.{family}_lengths"] = torch.from_numpy(counts.sum(axis=1))
        out[f"ragged.{family}_overflow"] = torch.from_numpy(
            (counts > cap).any(axis=1).copy())

    if offset != flat.size:
        raise ValueError(
            f"ragged buffer has {flat.size - offset} trailing bytes after all views")

    return out


def transport_rows(td: dict, row: int) -> dict:
    """Extract one view's ragged transport tensors (already on CPU).

    The overflow flag is retained deliberately: dropping it would make a
    decoded row unable to prove it was not silently clamped. `payload_to_
    transport` already raises on overflow, so a valid row always carries
    `False`; the flag is still part of the row contract and is validated by
    `validate_transport_row`.
    """
    out = {}
    for key in list(td):
        if key.startswith("ragged."):
            out[key] = td[key][row]
    return out


def validate_transport_row(row, where: str = "view") -> None:
    """Validate one view's ragged transport leaves; fail closed on corruption.

    This is the redundant decode-side gate. The encode path already raises
    `RaggedOverflow`, but a stored, hand-built or future-produced transport must
    not be able to smuggle counts above the padding cap into a slice such as
    `values[t, :k]`, where Python would silently clamp to the available rows.
    """
    FIELDS = _fields()
    for family, fields in FIELDS.items():
        cap = RAGGED_TOKEN_CAPACITY[family]
        width = len(fields)
        keys = {field: f"ragged.{family}_{field}"
                for field in ("counts", "values", "lengths", "overflow")}
        missing = [key for key in keys.values() if key not in row]
        if missing:
            raise ObservationContractMismatch(
                f"{where}: missing ragged leaves {missing}; expected the "
                f"{OBSERVATION_REPRESENTATION!r} contract")
        counts = row[keys["counts"]]
        values = row[keys["values"]]
        lengths = row[keys["lengths"]]
        overflow = row[keys["overflow"]]
        if not torch.is_tensor(counts) or counts.dim() != 1 or counts.numel() != TOKENS:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_counts must be int64[{TOKENS}], "
                f"got {type(counts).__name__} {tuple(counts.shape) if torch.is_tensor(counts) else ''}")
        if counts.dtype != torch.long:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_counts dtype {counts.dtype} != torch.long")
        if bool((counts < 0).any()):
            raise ObservationContractMismatch(
                f"{where}: negative ragged count for {family}")
        worst = int(counts.max().item()) if counts.numel() else 0
        if worst > cap:
            raise RaggedOverflow(
                f"{where}: ragged family {family!r} has {worst} rows for one "
                f"token, above the transport capacity {cap}. Refusing to "
                "truncate; raise RAGGED_TOKEN_CAPACITY and re-measure.")
        if not torch.is_tensor(values) or values.dim() != 3 or tuple(values.shape) != (
                TOKENS, cap, width):
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_values must be float32"
                f"[{TOKENS}, {cap}, {width}], got "
                f"{tuple(values.shape) if torch.is_tensor(values) else type(values).__name__}")
        if values.dtype != torch.float32:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_values dtype {values.dtype} != torch.float32")
        if not torch.is_tensor(lengths) or lengths.dtype != torch.long:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_lengths must be torch.long")
        expected = int(counts.sum().item())
        got = int(lengths.item()) if lengths.numel() == 1 else -1
        if got != expected:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_lengths {got} != sum(counts) {expected}")
        if not torch.is_tensor(overflow) or overflow.numel() != 1:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_overflow must be a scalar bool")
        if overflow.dtype != torch.bool:
            raise ObservationContractMismatch(
                f"{where}: ragged.{family}_overflow dtype {overflow.dtype} != torch.bool")
        if bool(overflow.item()):
            raise RaggedOverflow(
                f"{where}: ragged family {family!r} carries an overflow flag; "
                "the transport must not contain clamped rows")


# ---------------------------------------------------------------------------
# transport tensors -> PA3 RaggedBatch (encoder input)
# ---------------------------------------------------------------------------
def transport_to_ragged(td: dict, index: int, device: str | torch.device = "cpu"):
    """Rebuild PA3 `RaggedBatch` objects for one view from transport tensors.

    `td` is a stacked transport dict ([B, ...] tensors). `index` selects a view.
    Returns {family: RaggedBatch}. Ordering, counts, dtype and values are
    preserved exactly, so the encoder sees byte-identical features to the
    canonical PA3 path.
    """
    FIELDS = _fields()
    from agent.types.ragged import RaggedBatch  # noqa: PLC0415

    first_key = f"ragged.{next(iter(FIELDS))}_counts"
    if td[first_key].dim() == 2:
        row = transport_rows(td, index)
    else:
        row = {k: td[k] for k in td if k.startswith("ragged.")}
    validate_transport_row(row, where=f"transport_to_ragged[{index}]")

    out = {}
    for family, fields in FIELDS.items():
        counts_i = row[f"ragged.{family}_counts"]
        values_i = row[f"ragged.{family}_values"]
        counts_np = np.asarray(counts_i.detach().to("cpu").numpy(), dtype=np.int64)
        if counts_np.ndim == 1:
            counts_np = counts_np[None, :]

        width = len(fields)
        # Pack only the real rows, in token order, exactly as CSR requires.
        packed = []
        for t in range(counts_np.shape[-1]):
            k = int(counts_np[0, t])
            if k:
                packed.append(values_i[t, :k].detach().to("cpu").numpy().astype(np.float32))
        flat = np.concatenate(packed, axis=0) if packed else np.zeros((0, width), np.float32)
        out[family] = RaggedBatch.from_rows([flat], counts_np, width).to(device)
    return out


# ---------------------------------------------------------------------------
# identity checks
# ---------------------------------------------------------------------------
def assert_representation(record_representation: str | None) -> None:
    """Refuse records that are not the repaired representation (requirement 8)."""
    if record_representation == OBSERVATION_REPRESENTATION:
        return
    got = record_representation or OBSERVATION_REPRESENTATION_FIXED9
    raise ObservationContractMismatch(
        f"observation representation {got!r} != required {OBSERVATION_REPRESENTATION!r}; "
        "records collected before the ADR-009 repair cannot enter repaired PPO training")


def ragged_digest(ragged: dict) -> str:
    """Order-sensitive sha256 over all five families' counts and values."""
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    for family in ("effects", "repertoire", "types", "base_moves", "move_effects"):
        section = ragged[family]
        h.update(family.encode())
        h.update(np.ascontiguousarray(section.counts.detach().cpu().numpy()).tobytes())
        h.update(np.ascontiguousarray(
            section.values.detach().cpu().numpy().astype(np.float32)).tobytes())
    return h.hexdigest()
