# P4.5 — Historical Reconciliation

**Design:** `PA4-P45-DIAG-DESIGN-v1.0`
**Baseline:** P4 20-iteration trial at `d0f42f1`
**Evidence:** `evidence/historical_reconstruction.json` (machine-generated)

---

## 1. What was recomputed and from what

Both authoritative P4 artifacts were parsed by code, not read by hand:

| Source | SHA256 |
|---|---|
| `docs/phases/P4/FULL_STACK_BASELINE.csv` | `0553cb8a5985a0ff00b3581fc30ce4e04b3f0e9035c389d55a6859b29b2df85a` |
| `docs/phases/P4/evidence/full_run.json` | `6872eefd3360dcddccb1235ebff04c02388c33bcf31b5e9242c0edc1e8b4ff12` |

Neither file was modified. Their hashes are additionally recorded in
`ENVIRONMENT.json` under `protected_p4_evidence`.

## 2. Stage totals (20 iterations)

| Category | Sum (s) | Share of row sum |
|---|---:|---:|
| `collection_seconds` | **373.347** | **51.81%** |
| `update_seconds` | **332.808** | **46.19%** |
| `checkpoint_seconds` | **11.211** | **1.56%** |
| Residual inside rows | **3.176** | **0.44%** |
| **CSV `wall_seconds` total** | **720.541** | 100% |

These reproduce the design's §1.2 figures exactly.

## 3. The 0.407 s discrepancy — reconciled, not forced equal

- Outer run stopwatch (from `full_run.json` `total_wall_seconds`): **720.948 s**
- Sum of per-iteration `wall_seconds` rows: **720.541 s**
- Difference: **0.407 s**

**Explanation.** The per-iteration timer opens inside
`SynchronousTrainer.run_iteration` (`t_iter = time.perf_counter()`) and closes
after the checkpoint publish. The outer stopwatch opens in `full_run.py` before
constructing `SynchronousTrainer` and closes after `trainer.run()` returns. The
0.407 s is therefore **run-level setup and teardown** outside any single
iteration: object construction, the first import warm-up and the final return.

**This is a boundary difference, not measurement error, and the two numbers are
deliberately not equated.** P4's published `0.4439 matches/s` uses the *outer*
denominator (720.948 s), which is the more conservative of the two:

```
320 accepted / 720.948 = 0.44387  ->  0.4439 m/s
320 accepted / 720.541 = 0.44411  ->  0.4441 m/s
```

## 4. Per-iteration spread (median, not mean)

Read directly from `evidence/historical_reconstruction.json` `per_iteration`:

| Stage | min | median | max | spread |
|---|---:|---:|---:|---:|
| collection | 16.704 s | 18.805 s | 21.777 s | 27.0% |
| update | 10.551 s | 17.303 s | 19.619 s | 52.4% |
| checkpoint | 0.53 s | 0.545 s | 0.715 s | 34.0% |
| wall | 29.503 s | 36.453 s | 40.95 s | 31.4% |

The **update stage varies most (52.4%)** because
the number of accepted row decisions per match differs between iterations, so
per-row learner work is not constant. Collection varies
27.0%.

## 5. What each historical timer actually contains

Established by reading `pa4/learner/train_sync.py`, not inferred:

| Timer | Opens | Closes | Contains |
|---|---|---|---|
| `collection_seconds` | before `run_batched(bcfg, store=sink)` | after it returns | server create/start, snapshot load into the child, initial reset, every ENCODE and prefix-STEP round trip and its wait, native step/reset, transition materialisation, sink put |
| `update_seconds` | after cohort build | after the PPO updater returns | frozen critic construction + state load, **per-row** `build_observation` / `encode` / `value` / `.item()`, GAE, advantage normalisation, per-microbatch row rebuild, ragged stacking, batched trainable encoder, **per-row** `evaluate_encoded`, backward, clip, Adam |
| `checkpoint_seconds` | before `save_checkpoint` | after it and the snapshot publish | `torch.save`, fsync, sha256, atomic rename, pointer publication |
| `wall_seconds` | start of the iteration | after publish | collection + update + checkpoint + in-iteration residual (cohort assembly, publish) |

**None of these are GPU kernel times.** They are parent wall intervals that
include CPU, IPC, queue waits and device work. All fine-grained attribution in
`BOTTLENECK_RANKING.csv` is nested *inside* these intervals and must not be
summed with them.

## 6. Why this matters for the Amdahl bounds

The two halves are close to equal (51.8% / 46.2%), which caps any single-sided
optimisation at about **2.08×** and **1.86×** respectively. This is the arithmetic
reason the diagnosis recommends that both halves be addressed, and the reason a
learner-only patch cannot deliver the throughput target.
