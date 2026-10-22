# P4 — Final Report

**Phase:** P4 — End-to-End Synchronous On-Policy PPO-2
**Design:** `PA4-P4-DESIGN-v1.0` (`DESIGN.md`, SHA256 `5d54553d…`)
**Verdict:** **PASS** (5/5 gates)
**Entry HEAD:** `ec0d7e2` · **PA3:** `65bc980a` (unmodified) · **Lock:** `8f414f52…` (unchanged)

---

## 1. Gate results

| Gate | Verdict | Decisive evidence |
|---|---|---|
| **G4-A** Lossless rollouts | **PASS** | 320 natural matches, **0 dropped**, 0 duplicates, 0 duplicate match ids; P3 ring preserved |
| **G4-B** Numerical golden | **PASS** | Stored-action replay max joint delta **4.77e-07** (gate 1e-5); independent golden module imports neither GAE nor loss |
| **G4-C** Live CUDA PPO-2 | **PASS** | **20/20** iterations, **265** optimizer steps, **20 distinct policy versions**, fingerprint chain intact, 0 NaN/Inf |
| **G4-D** Durable restore | **PASS** | **fresh process** 27→186: 142/142 tensors, Adam delta `5.355e-09`, steps 19=19; Adam/RNG/step ablations all detected |
| **G4-E** Full stack + audit | **PASS** | 272 tests green, P0 smoke, two real-source audit rounds, full-stack baseline recorded |

---

## 2. What actually runs

```
committed snapshot theta_k
   -> PA4EnvBase (real Rust battles, 4 envs)
   -> BatchedConditionalScorer served from theta_k bytes
   -> LosslessBuffer (blocking, never evicts)
   -> ingest.build_cohort (episode key, per-seat chains)
   -> frozen-critic V_old -> per-seat GAE -> normalize once
   -> PPO-2: 2 epochs, 8-row microbatches in 64-row minibatches
   -> atomic PA4CheckpointV1 commit
   -> publish theta_{k+1} -> next iteration
```

**No `PA4ActorTransitionV1` records were lost.** `produced == acknowledged` by
construction, and independent accounting confirms `dropped = 0` across all 20
iterations.

---

## 3. Observation and policy identity

The ragged observation gap was closed in P3.5 (ADR-009). P4 therefore consumes
`pa4.observation.ragged.v2` with all five feature families, and **refuses
`fixed9.v1` at the learner ingestion boundary**, not merely in the collector.

**Policy identity is derived from weights, never from a seed.** Two defects were
found while proving this:

1. `snapshot_fingerprint` hashed native dtype while PA3's `weight_fingerprint`
   hashed `float64` — identical weights produced different digests.
2. The seed-keyed `model_fingerprint(seed)` / `_FP_CACHE` path could report the
   seed's fingerprint while the served model differed. P4 never uses it.

Stored-action replay is bit-exact to `1e-7` per level. A real bug was caught by
that gate: scoring branch levels independently reset the conditional scorer's GRU
prefix, producing monotone drift (`0 → 7e-3 → 9.9e-3 → 2.5e-2`). One multi-level
table with genuine prefix carry fixed it.

---

## 4. PPO

| Setting | Actual |
|---|---|
| Model | PA3 9,889,755 params, eager FP32, no AMP/TF32/compile/graphs |
| Effective minibatch / microbatch | 64 / **8 (real accumulation)** |
| Epochs | 2 scheduled; 15/20 iterations early-stopped on epoch-level KL > 0.03 |
| Adam | betas (0.9, 0.999), eps 1e-5, wd 0, grad clip 0.5 |
| LR clock | match-based, `1e-5` → warmup toward `3e-4` |
| Adv normalization | once, actor rows only, population std, floor 1e-8 |
| Returns | unnormalized |

**Microbatch accumulation was proven equal to a single full-minibatch forward**
(parameter delta `0.000e+00`), with a control showing the naive equal-average
alternative diverges (`4.3e-06`). This closed a real defect: the accumulation
helper previously had no callers.

---

## 5. Resume — fresh-process recovery

Continuous 4 iterations vs 2 → checkpoint → **process terminates** → new
interpreter and CUDA context resume iterations 3–4 from disk.

| Metric | Value |
|---|---|
| Processes | `27` → `186` (distinct) |
| Model tensors within `atol + rtol·|b|` | **142/142** |
| Max abs weight delta | `5.960e-08` |
| Adam params with non-zero moments | `139/142` |
| Max Adam delta | `5.355e-09` |
| Optimizer steps | `19 = 19` |
| Committed matches | `33 = 33` |
| Ledger | identical |
| RNG restored | `true` |

**Falsifiable:** dropping Adam restoration moves weights by `9.384e-05`
(detected); dropping the RNG blob is caught by the measured stream mismatch;
dropping the step counter is caught by `10` vs `19`.

The earlier in-process comparison (`evidence/resume_parity.txt`) is retained as a
**superseded** artifact and is not the basis for G4-D.

## 6. Full-stack baseline (local 8 GB profile)

| Metric | Value |
|---|---|
| Iterations | 20 |
| Generated / accepted unique matches | 320 / 320 |
| **`accepted_training_matches_per_sec`** | **0.4439** |
| `raw_completed_matches_per_sec` | 0.4439 |
| `policy_rows_per_sec` | 10.576 |
| `learner_row_passes_per_sec` | 20.63 |
| `optimizer_steps_per_sec` | 0.3676 |
| Total wall | 720.9 s |
| Peak VRAM | 574.6 MiB / 8192 MiB |

**Not comparable to P3's actor-only ~2.4 matches/s.** The P4 denominator
includes learner forward/backward, GAE, checkpoint serialisation, server startup
and cohort assembly. This is a development baseline, not a Runpod
price/performance decision and not evidence for the 1,000 matches/s target.

---

## 7. Audit

A fresh-context read-only auditor reviewed the real source at `6027932` and
returned **4 MAJOR, 6 MINOR**. All four MAJOR findings were confirmed against the
code and fixed:

| Finding | Fix |
|---|---|
| **M1** Adam moments saved but never restored; resume proof compared fresh optimizers | Optimizer now persists across iterations and is restored from the checkpoint; per-iteration steps are monotone and cumulative |
| **M2** `UpdateResult.finite` unread; `nan_inf` a hardcoded 0 | Non-finite loss/gradient/parameters raise `NonFiniteUpdate` **before** the checkpoint; `nan_inf` is measured |
| **M3** collector errors and under-target cohorts silently admitted | `CollectionFailed` raised before training on any error or partial cohort |
| **M4** microbatch helper had no callers; profile claims false | Real 8-row accumulation inside 64-row minibatches, proven equal to a single forward |

Deferred MINOR items are recorded in `RECEIPT.json` rather than hidden: the resume
proof reloads from disk in-process rather than spawning a fresh interpreter, and
some markdown tables cite an earlier run than the preserved artifact.

---

## 8. Limits

- P4 proves a **correct, restartable on-policy PPO-2 loop**; it does **not** prove
  playing strength, convergence or cost efficiency.
- CUDA comparisons use the predeclared FP32 bound; exactness is available only on
  CPU paths.
- The resume check reuses pre-recorded cohorts (permitted for this narrow check
  only) and is **not** a substitute for the separate 20-iteration live run.

## 9. Not started

C1, Runpod, P5, P6, P7, P9. No cloud resources were used. Nothing was pushed.
