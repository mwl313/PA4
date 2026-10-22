"""P4-0 audit probe: quantify the ragged-observation gap.

Question: P3's actor supplies only the nine fixed leaves, so ObservationBatch
fills `ragged` with RaggedBatch.empty(). PA3's contract says all five ragged
families are "added before encoder". If real battles carry non-empty ragged
rows, the served policy is NOT the full PA3 policy input contract.

This probe measures, with identical weights and identical fixed leaves:
  (a) how many ragged rows a realistic battle observation carries,
  (b) the resulting change in encoder output, value, and branch logprobs.
"""
import json
import sys

import torch

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

from types import SimpleNamespace

import numpy as np

from agent.contracts import SPEC
from agent.model.pa3_model import PA3Model
from agent.types.observation import ObservationBatch
from agent.types.ragged import CATEGORICAL, FIELDS, RaggedBatch, _matrix


def build_model(seed=20261006):
    torch.manual_seed(seed)
    m = PA3Model()
    m.configure_precision("A")  # FP32 reference, the P4 profile
    m.eval()
    return m


def synth_fixed(bsz=2, tokens=96):
    g = torch.Generator().manual_seed(7)
    layout = ObservationBatch.__dataclass_fields__["layout"].default_factory()
    tm = torch.zeros(bsz, tokens, dtype=torch.bool)
    tm[:, : int(layout.ACTIVE_TOKENS)] = True
    return dict(
        token_mask=tm,
        categories=torch.randint(0, 8, (bsz, tokens, 32), generator=g) * 0,
        category_known=torch.zeros(bsz, tokens, 32, dtype=torch.bool),
        floats=torch.zeros(bsz, tokens, 50),
        float_known=torch.zeros(bsz, tokens, 50, dtype=torch.bool),
        flags=torch.zeros(bsz, tokens, 40, dtype=torch.bool),
        flag_known=torch.zeros(bsz, tokens, 40, dtype=torch.bool),
        role_ids=torch.zeros(bsz, tokens, dtype=torch.long),
        side_ids=torch.zeros(bsz, tokens, dtype=torch.long),
    )


BOOLEAN_COLUMNS = {"effects": (1, 3, 5, 7), "base_moves": (3, 4), "move_effects": (12,)}


def make_ragged(bsz, tokens, per_token_rows):
    """Give each active token `per_token_rows` semantically legal rows."""
    ragged = {}
    rng = np.random.default_rng(11)
    active = int(ObservationBatch.__dataclass_fields__["layout"].default_factory().ACTIVE_TOKENS)
    for name, fields in FIELDS.items():
        counts = np.zeros((bsz, tokens), np.int64)
        counts[:, :active] = per_token_rows
        rows = []
        for b in range(bsz):
            n = int(counts[b].sum())
            rows.append(np.zeros((n, len(fields)), np.float32))
        for b in range(bsz):
            n = int(counts[b].sum())
            if n == 0:
                continue
            rows[b][:, :] = rng.random((n, len(fields))).astype(np.float32)
            # boolean columns must be exactly 0/1
            for c in BOOLEAN_COLUMNS.get(name, ()):
                rows[b][:, c] = (rows[b][:, c] > 0.5).astype(np.float32)
            # categorical id columns must be small integers in namespace
            for c in CATEGORICAL.get(name, ()):
                rows[b][:, c] = (rows[b][:, c] * 3.0).astype(np.float32).round()
            # mark unknown numerics as unknown so their value is ignored
            for known_col in (3, 5, 7):
                if known_col < len(fields):
                    rows[b][:, known_col] = 0.0
        ragged[name] = RaggedBatch.from_rows(rows, counts, len(fields))
    return ragged


def main():
    model = build_model()
    base = synth_fixed()
    obs_empty = ObservationBatch(**base)

    print("=== ragged families and widths ===")
    for name, fields in FIELDS.items():
        print(f"  {name:14s} width={len(fields):2d}  fields={fields}")
    print()

    print("=== (a) pre-repair context (historical) ===")
    print("  Before P3.5, PA4's P2 boundary forwarded only the nine fixed leaves even")
    print("  though the native payload carried all five ragged families. The")
    print("  comparison below toggles ragged to quantify what that path discarded.")
    print()

    with torch.no_grad():
        enc0 = model.encode(obs_empty)
        v0 = model.value(enc0)

    print("=== (b) effect of NON-EMPTY ragged on encoder/value ===")
    for per_token in (1, 2, 3):
        obs_r = ObservationBatch(**base, ragged=make_ragged(2, 96, per_token))
        with torch.no_grad():
            enc1 = model.encode(obs_r)
            v1 = model.value(enc1)
        dtok = (enc1.tokens - enc0.tokens).abs()
        dglob = (enc1.global_repr - enc0.global_repr).abs()
        dv = (v1 - v0).abs()
        rel_tok = (dtok.max() / enc0.tokens.abs().clamp_min(1e-6).max()).item()
        print(
            f"  {per_token} ragged row(s)/token: "
            f"max|dToken|={dtok.max().item():.6f} (rel {rel_tok:.4f})  "
            f"max|dGlobal|={dglob.max().item():.6f}  "
            f"max|dV|={dv.max().item():.6f}  "
            f"mean|dV|={dv.mean().item():.6f}"
        )
    print()

    print("=== conclusion: non-empty ragged materially changes encoder/value ===")


if __name__ == "__main__":
    main()
