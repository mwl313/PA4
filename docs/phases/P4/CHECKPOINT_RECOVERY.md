# P4-D — Checkpoint and Recovery

**Phase:** P4
**Design:** `PA4-P4-DESIGN-v1.0` §6
**Status:** PASS

---

## 1. What a checkpoint contains

`PA4CheckpointV1` (`pa4/learner/checkpoint.py`) stores everything needed to
resume training identically:

| Field | Present |
|---|---|
| model `state_dict` | ✅ |
| **Adam moments** (both exp_avg and exp_avg_sq per parameter) | ✅ 142 entries measured |
| optimizer-step counter | ✅ |
| scheduler / match-clock state | ✅ |
| RNG: python, numpy, torch CPU, torch CUDA | ✅ all four |
| committed unique-match ledger | ✅ |
| iteration / policy version / committed matches | ✅ |
| provenance (PA4 SHA, design id, observation representation, both fingerprints) | ✅ |
| safety counters (dropped, incomplete, generated) | ✅ |

**A model-only save is rejected.** `save_checkpoint` refuses to write when
`optimizer_state`, `scheduler_state`, `rng_state` or `match_ledger` is `None`,
and `load_checkpoint` rejects any payload with those fields null.

## 2. Commit is a transaction

```
collect -> validate -> update in memory -> write checkpoint -> THEN advance
        the committed clock and publish theta_{k+1}
```

1. Write to a temp file **in the same directory**, `fsync`.
2. `os.replace` to the versioned final path.
3. Compute SHA256, write `latest.json` temp, `fsync`, `os.replace`.
4. Only now advance `committed_matches`, `optimizer_steps`, the ledger and the
   policy version.

A crash before step 3 leaves the previous checkpoint authoritative: no clock
advance, no new policy version. The previous good checkpoint is never deleted
(both are retained; `latest.json` names the newest).

## 3. Measured restore fidelity

`evidence/checkpoint_state.txt`:

| Item | Result |
|---|---|
| Adam state entries | 142 |
| RNG streams restored | `numpy`, `python`, `torch_cpu`, `torch_cuda` |
| Max model restore delta | **0.000e+00** |
| Max Adam moment delta | **0.000e+00** |
| Ledger | intact |

## 4. Corruption rejection

`evidence/checkpoint_corruption.txt` — all five modes refused:

| Mode | Result |
|---|---|
| Tampered weights (file structurally intact) | **rejected** (payload checksum) |
| Truncated file | rejected |
| Wrong pointer SHA | rejected |
| Model-only save | rejected |
| Foreign schema | rejected |

A real gap was found here: `load_checkpoint(path)` skipped checksum verification
whenever it was handed a **bare file** rather than a directory, so a corrupted
checkpoint loaded silently. Integrity is now verified regardless of how the
artifact is addressed, with a payload-level checksum over the model tensors for
the no-manifest case.

## 5. Continuous vs interrupted-and-resumed (4 iterations)

A controlled comparison at a policy-version boundary. Both branches consume the
**same on-policy cohorts**, each validated against its own frozen fingerprint.

- **Branch A:** 4 iterations in one process.
- **Branch B:** 2 iterations → save checkpoint → **reload from disk** → 2 more.

### The control that makes this honest

Before comparing branches, the same update was run **twice in one process** with
identical inputs:

```
control (same update twice): max abs 5.960e-08
```

CUDA kernels are not bitwise-reproducible, so **bitwise weight equality is not an
achievable criterion on this GPU**. The design anticipates this and predeclares
`atol=1e-5, rtol=1e-4` for CUDA FP32.

### Result against the predeclared bound

`evidence/resume_parity.txt`:

| Metric | Value |
|---|---|
| Resume max abs weight delta | `1.192e-07` |
| Worst excess over `atol + rtol·\|b\|` | `−9.994e-06` (inside) |
| Tensors passing the criterion | **142 / 142** |
| Max Adam moment delta | `6.29e-09` |
| Counters / committed matches / ledger | identical |

**Two criterion errors were made and corrected:**

1. The first version demanded bitwise equality → failed on `1.19e-07`, which the
   control shows is nondeterminism, not a defect.
2. The second used bare relative error → failed on near-zero weights where
   relative error explodes (`rel = 67`) despite absolute deltas of `1e-7`. The
   predeclared bound is combined: `|a−b| ≤ atol + rtol·|b|`.

Neither relaxation was invented: the tolerance is the one declared in the design
before measurement.

## 6. Reproduce

```bash
cd /home/pa4/projects/PokeAgent4.0
uv run --frozen python docs/phases/P4/evidence/proof_checkpoint_state.py
uv run --frozen python docs/phases/P4/evidence/proof_checkpoint_corruption.py
uv run --frozen python docs/phases/P4/evidence/control_nondeterminism.py
uv run --frozen python docs/phases/P4/evidence/proof_resume.py
uv run --frozen python -m unittest tests.ppo.test_checkpoint -v
```

## 7. Limitations

- The resume comparison uses pre-recorded cohorts, permitted by the design for
  this narrow restart check only. It is **not** a substitute for the separate
  20-iteration live-Rust trial (see `FINAL_REPORT.md`).
- Exact bitwise equality is unavailable on CUDA; all CUDA comparisons use the
  predeclared FP32 bound. CPU-only comparisons in the test suite are exact.
