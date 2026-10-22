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
**same stored cohorts**, each validated against its own frozen fingerprint.

- **Branch A:** 4 iterations in one process.
- **Branch B:** 2 iterations → save checkpoint → **reload from disk** → 2 more.

### The control that makes this honest

Before comparing branches, the same update was run **twice in one process** with
identical inputs:

```
control (same update twice): max abs 1.192e-07
```

CUDA kernels are not bitwise-reproducible, so **bitwise weight equality is not an
achievable criterion on this GPU**. The design anticipates this and predeclares
`atol=1e-5, rtol=1e-4` for CUDA FP32.

### Result against the predeclared bound

`evidence/resume_parity.txt`:

| Metric | Value |
|---|---|
| Resume max abs weight delta | `1.192e-07` |
| Worst excess over `atol + rtol·\|b\|` | `−9.993e-06` (inside) |
| Tensors passing the criterion | **142 / 142** |
| Max Adam moment delta | `4.307e-09` |
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
uv run --frozen python docs/phases/P4/evidence/fresh_process_recovery.py
uv run --frozen python docs/phases/P4/evidence/fresh_process_ablation.py \
    <work_dir> <work_dir>/B_ckpt <work_dir>/cohorts
uv run --frozen python -m unittest tests.ppo.test_checkpoint -v
uv run --frozen python -m unittest tests.ppo.test_fresh_process -v
```

## 7. Limitations

- The resume comparison uses pre-recorded cohorts, permitted by the design for
  this narrow restart check only. It is **not** a substitute for the separate
  20-iteration live-Rust trial (see `FINAL_REPORT.md`).
- Exact bitwise equality is unavailable on CUDA; all CUDA comparisons use the
  predeclared FP32 bound. CPU-only comparisons in the test suite are exact.

---

## 8. Fresh-process recovery (design §6.3) — the acceptance gap

The earlier proof reloaded the checkpoint **in the same process**, which does not
exercise "terminate and relaunch a new process". That gap is now closed by
`evidence/fresh_process_recovery.py`, which:

1. builds theta_0 and collects four cohorts, each validated against the
   snapshot fingerprint that produced it. NOTE: all four are produced by
   theta_0, so they are not on-policy for iterations 2-4. The comparison is
   still valid because BOTH branches consume identical data; this check
   isolates checkpoint fidelity, not on-policy collection;
2. runs iterations 1-4 continuously (branch A);
3. runs iterations 1-2 and commit a full checkpoint (branch B);
4. **launches a genuinely new interpreter and CUDA context** via `subprocess`;
5. the child loads the checkpoint from disk and completes iterations 3-4;
6. the parent compares model, Adam moments, steps, scheduler, ledger and accepted
   cohort identities against the child's final artifacts; the child additionally
   proves a measured RNG round-trip (`check_rng_roundtrip`) from the loaded
   checkpoint.

### Measured result (`evidence/fresh_process_recovery.txt`)

| Field | Value |
|---|---|
| Parent PID → child PID | `19` → `178` (`distinct_processes: true`) |
| Child device | `NVIDIA GeForce RTX 4060 Ti` (fresh CUDA context) |
| Model tensors within bound | **142/142** |
| Model max abs delta | `1.192e-07` |
| Worst excess over the declared bound | `-9.992e-06` (inside) |
| Adam state ids match | **true** |
| Adam params with non-zero `exp_avg` | **139 / 142** |
| Adam max abs delta | `5.331e-09` |
| Optimizer steps | **23 = 23** |
| LR | identical |
| Committed matches | **33 = 33** |
| Match ledger | identical |
| RNG restored | **true** |

### Falsifiability (`evidence/fresh_process_ablation.txt`)

A recovery test that passes regardless of restoration proves nothing. The
ablation child replays the resume four ways against the **same** checkpoint:

| Ablation | Detection | Magnitude |
|---|---|---|
| baseline (full restore) | delta vs itself | `0.000e+00` |
| **Adam moments dropped** | **detected** | max weight delta **`1.059e-04`** |
| **RNG blob dropped** | **detected** | measured stream mismatch (`rng_restored: false`); weights alone move only `7.451e-09` |
| **step counter dropped** | **detected** | counter `9 vs 23` |

Dropping Adam restoration moves the weights by `1.059e-04`, far outside
the predeclared FP32 tolerance, so the recovery gate genuinely fails without it.
RNG loss does not show up in the weights at that magnitude, so it is detected by
observing that the restored stream does not reproduce the checkpointed
continuation.
`tests/ppo/test_fresh_process.py` pins these invariants in the normal suite.

---

## 9. Evidence history

Two runs exist and both are retained:

| Artifact | Run | Status |
|---|---|---|
| `resume_parity.txt` | in-process reload (first attempt) | **superseded** — retained as history; this is the run the audit flagged for not spawning a process |
| `fresh_process_recovery.txt` | subprocess relaunch (§8) | **authoritative** |
| `fresh_process_ablation.txt` | subprocess relaunch, 4 ablations | **authoritative** |
| `checkpoint_state.txt`, `checkpoint_corruption.txt` | field/corruption checks | current |

Every number in this document is read from the artifact named beside it. A first
attempt at this reconciliation cited values from an earlier run than the
regenerated artifacts; the audit caught that, and the tables were rebuilt from
the files rather than adjusted by hand. Superseded runs are named as superseded.
