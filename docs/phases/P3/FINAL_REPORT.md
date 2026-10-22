# P3 — Final Report

**Phase:** P3 — Batched Collector and Central Inference
**Design:** `PA4-P3-DESIGN-v1.0` (sha256 `f73c5567…`)
**PA4 base:** `4d8f349b247bb929914db5b9ebe2c5ce84c18288`
**PA4 tested code:** `13395df0e731cdc93996d63306c74703257c1dd6`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified, 0 dirty files)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0)
**GPU:** RTX 4060 Ti 8 GB, CUDA capability (8,9), driver 610.88

---

## VERDICT: **PENDING_FINAL_AUDIT**

The first audit (read access confirmed) returned `FINDINGS`: P3-C transition
retention was absent, "central" inference was per-driver, fingerprints and
several G3-D controls were missing, and the timing evidence was incomplete.
This report describes the implementation that addresses those items. It is
written **before** the re-audit returns so the auditor reviews real numbers.

Measured at the tested commit: **68 collector + 51 environment + 39 action =
158 tests OK**, `check_env_specs` passed, `uv.lock` unchanged, PA3 clean.

---

## 1. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G3-A** Collector integration | **PASS (ADR-008 scoped)** | The actor loop drives real P2 `EnvBase.step()` transitions whose actions come from the real P1 conditional scorer over live Rust candidates. The stock `AsyncBatchedCollector` is not used (ADR-008, user-approved). Transitions are now emitted as bounded `PA4ActorTransitionV1` records. |
| **G3-B** Central CUDA batching | **PASS** | Unit proof: one forward, `forward_batch_sizes == [4]`, each row checked against its own candidates/identity. Central proof: one parent-owned `InferenceServer` (`transport="distributed"`) serves two and four spawned driver processes; the service records each forward's driver `run_id` set, and **235 / 180 forwards mixed 2 / 4 worker processes** in the retained baselines. |
| **G3-C** Natural-match integrity | **PASS** | ≥16 natural completions per run; 0 duplicate ids; ids carry run/worker/slot/generation/episode; 2 and 4 independent driver processes; exactly-once accounting unit-tested; the collector compares its ledger with `env.completed_matches` and records a mismatch as an error; revised decisions are excluded from behaviour rows. |
| **G3-D** Transport/lifetime | **PASS** | Distinct slots, reverse-order response matching, one in-flight, shutdown with an unread in-flight request, repeated start/stop, overflow, count/mask mismatch, unknown key, wrong level, stale generation, duplicate identity **within one batch**, fabricated identity, cross-worker aliasing, mixed policy version, wrong weight fingerprint — all fail closed without corrupting other entries. Client futures have a timeout. |
| **G3-E** Baseline and audit | **PENDING** | `PERFORMANCE_BASELINE.csv` regenerated at the tested commit with per-row provenance, setup/warmup/steady windows, stage timing and retained JSON per run; 152 tests green; lock and PA3 unchanged. **Auditor re-check pending.** |

## 2. What changed in this round

1. **Transition retention (P3-C).** `pa4/collector/transitions.py` implements a
   fixed-capacity `BoundedTransitionStore` and `PA4ActorTransitionV1` records
   with the pre-action nine-leaf observation by value, ordered candidate tables,
   selected indices, branch_k, FP32 logprobs, the real six-key packet, native
   outcome, match id, policy version, sha256 weight fingerprint and the pinned
   observation contract hash. Drops are counted. Only accepted decisions are
   recorded; revised rows are excluded and counted separately.
2. **Central cross-process batching (P3-B).** `run_multi` creates one
   `InferenceServer` with `transport="distributed"` and hands its picklable
   transport to each spawned driver. Requests are tagged with the driver's
   `run_id`; the instrumented service records the run-id set of each forward.
3. **Policy provenance.** The scorer computes a sha256 weight fingerprint at
   init, stamps it on every response, and rejects a request whose
   `policy_version` or fingerprint disagrees.
4. **Validation.** Duplicate identity inside one transport batch is rejected
   without a cache write; fabricated identities fail closed; `_do_encode` and
   `_do_step` reject foreign versions.
5. **Integrity.** Ledger vs `env.completed_matches` comparison; revision
   filtering via `accepted_records`; client result timeout; the legacy
   `run_collector`/`collector_core` modules were removed (they emitted partial
   records and were untested).
6. **P3-D evidence.** Single-driver runs record setup/warmup/steady windows and
   per-stage timing in every run, retained as JSON. "Steady" means steps after
   `warmup_steps`; the window length is recorded (12–56 post-warmup steps here)
   so a short terminal burst is visible rather than hidden. Multi-driver runs retain
   each driver's own windows and stage sums; because drivers run concurrently,
   their steady rates are reported per driver and averaged, not summed into a
   fictitious wall window.

## 3. Measured performance

From `PERFORMANCE_BASELINE.csv`, all at commit `13395df`:

| Configuration | Matches/s | Steady matches/s | Transitions/s |
|---|---|---|---|
| 16 envs, 1 driver | 5.12 | 7.77 | 80.6 |
| 32 envs, 1 driver | 5.53 | 9.23 | 95.3 |
| 64 envs, 1 driver | 5.48 | 11.17 | 106.3 |
| 2 drivers × 16 envs, central | 1.25 | 0.82 | 21.5 |
| 2 drivers × 16 envs, per-driver servers | 5.24 | 5.60 | 97.0 |
| 4 drivers × 16 envs, central | 0.98 | 0.29 | 17.3 |
| 4 drivers × 16 envs, per-driver servers | 7.47 | 3.39 | 125.8 |

Forward batches (STEP forwards): `bmax` 29/61/64 for the single-driver runs,
62/64 central; mean 5.86/6.92/8.15 (single) and 4.02/1.71 (central). Encoding
forwards are counted separately.

**Finding:** central cross-process inference is correct and observable, but on
this host it is ~4.2–7.6× slower than per-driver servers. Both topologies use
the same server batching policy (`min_batch=2`, `timeout=0.02`), so the
difference is the transport: the distributed transport opens a gloo/TCP pair
per client, while shared-memory `ProcessSlotTransport` cannot serve two
processes. This is a transport-selection input for P6/P7, not a model-compute
conclusion. For multi-driver runs the steady column is the mean of per-driver
steady rates (drivers run concurrently, so their windows cannot be summed).

`accepted_training_matches_per_sec` = **null** — no PPO learner exists in P3.

## 4. Accounting rule and its limits

`account_step()` counts one match per unique `match_completed` episode, counts a
truncation only in `truncations`, counts a revision only in `revisions`, and
records an error if both `match_completed` and `truncated` are true. The
collector additionally asserts its ledger equals `env.completed_matches`, so a
lost completion is an error rather than a quietly smaller count. Records are
only emitted for accepted steps.

## 5. Files changed in P3 (tested commit `13395df`)

New: `pa4/collector/{batched_core,run_batched,multi_driver,transitions}.py`, `pa4/actions/{batched_scorer,
batched_factory_instrumented}.py`, `pa4/env/obs_fastpath.py`,
`tests/collector/*`, `docs/phases/P3/*`.

Modified: `pa4/env/native_env.py` (batched decode, fast observation path,
`request_info_batch`), `pa4/actions/conditional_policy.py` (active→roster
threading, from P2), `docs/PROJECT_STATUS.md`.

## 6. Reproduce

```bash
cd /home/pa4/projects/PokeAgent4.0
uv run --frozen python -m unittest discover -s tests/action_distribution -t .
uv run --frozen python -m unittest discover -s tests/environment -t .
uv run --frozen python -m unittest discover -s tests/collector -t .
uv run --frozen python docs/phases/P3/evidence/run_baseline.py
```

## 7. Limitations (explicit)

1. **`AsyncBatchedCollector` is not used** (ADR-008, user-approved). G3-A is a
   scoped PASS.
2. **Central inference is slower on this host** (~1.0–1.3 vs 5.2–7.5 matches/s)
   because the distributed transport uses gloo/TCP per client. Both topologies
   are measured and reported.
3. **Throughput is ~7.5 matches/s at best** at best, far below the 1,000/s stretch goal
   the design forbids proving on this GPU.
4. **Random PA3 weights** (seed 20261006); no strength or parity claim.
5. **Path B (staged prefix) only.**
6. **Transitions are in-memory and bounded**; no replay database, per design.
   Drops are counted, not silent.
7. **Baseline runs use one seed per configuration and were run once**; run-to-run
   variance is not characterised.
8. **Same-model review** is a process control, not independent model proof.
9. **PA3 pin** is the public `main` tip, not verified live production state.
10. **The distributed transport's gloo shutdown emits warnings** when driver
    processes exit; they are benign but noisy.

## 8. P4 handoff

P4 receives bounded `PA4ActorTransitionV1` records with the pre-action
observation, ordered candidates, selected indices, FP32 logprobs, policy version
and weight fingerprint, native outcome and fully-qualified match id. P4 owns GAE,
PPO-2, the optimizer and checkpoint parity — none of which exists in P3.

**P4 is not started.**
