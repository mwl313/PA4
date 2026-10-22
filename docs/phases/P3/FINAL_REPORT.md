# P3 — Final Report

**Phase:** P3 — Batched Collector and Central Inference
**Design:** `PA4-P3-DESIGN-v1.0` (sha256 `f73c5567…`)
**PA4 base:** `4d8f349b247bb929914db5b9ebe2c5ce84c18288`
**PA4 tested code:** `297cf8b7048c6ec1f1841fb00b443a868963520b`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified, 0 dirty files)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0)
**GPU:** RTX 4060 Ti 8 GB, CUDA capability (8,9), driver 610.88

---

## VERDICT: **PARTIAL**

The auditor now reads the actual WSL repository (`ACCESS_OK`, literal file
reads) and returned **`VERDICT: FINDINGS`** at
`297cf8b7048c6ec1f1841fb00b443a868963520b`. The earlier PARTIAL reason —
auditor access — is resolved; the review it was finally able to perform found
**unmet design deliverables**, not just missing evidence:

1. **P3-C transition retention is absent (BLOCKER).** §3.2 of the design and
   `TENSORDICT_TRANSITION_V1.md` describe bounded `PA4ActorTransitionV1` records
   holding the pre-action nine-leaf observation, ordered candidate tables, policy
   version/weight fingerprint and a bounded queue. On the tested path
   `run_batched` builds seat records and discards them after counting; nothing is
   retained.
2. **"Central" inference is per-driver.** Each driver process creates its own
   `InferenceServer`; the four-row B≥2 test is four rows in one process, so no
   forward ever mixes two worker processes as P3-B requires.
3. **No weight fingerprint / observation-contract hash** is stamped on ENCODE,
   STEP or RESET (design §3.4).
4. **Validation gaps:** a duplicate identity inside one transport batch silently
   overwrites (both rows return OK), and a foreign `policy_version` is accepted
   on ENCODE because the service never compares it to its own.
5. **No independent lost-completion guard:** `env.completed_matches` is never
   compared with the collector's ledger.
6. **Revisions count as behaviour rows**; `request_revised` does not filter the
   seat records.
7. **P3-D timing evidence is incomplete:** no cold/steady split, no retained
   stage breakdown, several headline numbers have no artifact, and the
   "~40 ms/forward" transport residual is an inference, not a measurement.

G3-A–G3-D are supported only in the scoped sense below. This is an honest
PARTIAL, not a completed phase.

---

## 1. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G3-A** Collector integration | **PASS (ADR-008 scoped)** | The actor loop drives real P2 `EnvBase.step()` transitions whose actions come from the real P1 conditional scorer over live Rust candidates. The stock `AsyncBatchedCollector` is not used; ADR-008 (user-approved) records why. |
| **G3-B** Central CUDA batching | **PASS per driver — NOT central** | `model_forward_count == 1` with `forward_batch_sizes == [4]` for four independent rows in one driver, device `cuda:0`; per-row `branch_k`, `request_key` and selection checked; cache isolation asserted. Real collection reached `bmax` 63/64. **No forward mixes two worker processes, and each driver owns its own server.** |
| **G3-C** Natural-match integrity | **PASS, with an unverified completeness property** | ≥16 natural completions per run (33–65 across baselines); 0 operational errors; 0 duplicate ids; ids carry run/worker/slot/generation/episode; 2 independent driver processes verified. Exactly-once accounting has direct unit tests. **No independent check that no completion was lost** (`env.completed_matches` is never compared), and revised decisions are counted as behaviour rows. |
| **G3-D** Transport/lifetime | **PARTIAL** | Distinct client slots, response key matching, one in-flight, shutdown with an unread in-flight request, repeated start/stop, overflow, count/mask mismatch, unknown key, wrong level, stale generation, cross-worker aliasing and STEP mixed-version are covered. **Not covered:** duplicate identity within one batch (silently overwrites), foreign-version ENCODE (accepted), missing/fabricated identity, and a bounded collector queue. |
| **G3-E** Baseline and audit | **PARTIAL** | CSV with per-row provenance and 136 tests green, lock unchanged, PA3 clean — but the auditor returned **FINDINGS** (§ above), several headline numbers have no retained artifact, and no cold/steady or stage breakdown exists. |

Test totals (Lead-executed, `uv run --frozen`): **46 collector**, **51 environment**,
**39 action-distribution** = **136 OK**.

## 2. Measured performance

Baselines are from `PERFORMANCE_BASELINE.csv`. The 16/32/64-env rows were measured
at `a8e38b8`; `run_batched.py`, `batched_core.py` and `batched_scorer.py` are
byte-identical between `a8e38b8` and the tested commit (only `multi_driver.py`,
tests and docs were added), and each CSV row records its own commit. Multi-driver
rows were measured at `4079610`, where `multi_driver.py` first exists.

| Configuration | Matches/s | Transitions/s | bmax | bmean |
|---|---|---|---|---|
| 16 envs, 1 driver | 5.12 | 79.4 | 31 | 3.78 |
| 32 envs, 1 driver | **6.04** | **101.1** | 63 | 5.15 |
| 64 envs, 1 driver (min_batch=16) | 5.78 | 104.0 | 64 | 9.16 |
| 2 driver processes × 16 envs | 4.40 | 78.0 | — | — |
| 4 driver processes × 16 envs | 5.53 | 91.8 | — | — |

Separate accumulation sweep (documented in `BATCHING_EVIDENCE.md` §3; raw stats
file not retained):

| 64 envs, min_batch=256 | 4.42 matches/s | forwards 874 → 318 | mean batch 15.5 |

**These are different runs and must not be combined.** An earlier revision of the
receipt paired the sweep's 15.5 mean with the baseline's 5.78 matches/s and
104.0 transitions/s; that was a transcription error and is corrected here.

Caveats on the table:

- `policy_forwards` in the CSV counts **STEP forwards only**; `model.encode` is a
  model forward but is not included, so it is not "all policy forwards".
- `window_type` says `cold+steady`, but no warmup/steady split is implemented:
  the single-driver timer starts after server start and env reset, while the
  multi-driver timer starts before process spawn. The 2×16 / 4×16 comparison to
  one driver is therefore **not controlled**, and "sublinear because the GPU is
  shared" is not demonstrated.
- The 15.5 sweep, 0.09 GiB peak, the GPU encode/step timings, 2291→1410 µs and
  63.7 KiB/row have **no retained artifact**. They are reported, but not
  reproducible from the committed tree.

`accepted_training_matches_per_sec` = **null** — no PPO learner exists in P3.

## 3. The defect that blocked G3-B

`ProcessInferenceServer.client()` **caches a single client** and returns the same
object on every call, so `[srv.client() for _ in range(n)]` produced n references
to **slot 0**. Because `ProcessSlotTransport` allows one in-flight request per
client, only one request was ever outstanding and every forward was B=1.

Measured: `client() -> 0,0` (same object) vs `clients(3) -> 1,2,3` (distinct).

**Fix:** `srv.clients(n)`. That one change turned
`forward_batch_sizes: [1,1,1,…]` into `[4,4,4,…]`.

## 4. Why throughput plateaus — measured, not guessed

Forward count fell 2.7× (874 → 318) when accumulation was tuned, **but throughput
did not move** (4.42 → 4.42 matches/s). Direct GPU timing explains it:

| Op | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| `model.encode` | 4.80 ms | 5.47 ms | 7.09 ms | 11.53 ms |
| `sample_level_step` | 2.60 ms | 2.40 ms | 2.46 ms | — |

GPU work is **nearly flat in batch size** as measured in the separate timing
probe (no retained artifact), so model compute does not explain the plateau.
The residual per-forward cost (~14 ms at 32 envs; ~45 ms in the accumulation
sweep, which includes the wait) is **consistent with** transport round-trip
overhead, but it was **not separately instrumented**, so the causal attribution
is an inference, not a measurement.

**This is a candidate finding for P6/P7, to be instrumented before relying on it.**

## 5. Accounting and negative controls (what the tests actually prove)

- `account_step()` is the single accounting rule used by the collector. Unit
  tests drive it with synthetic step outputs: one completion → one match id;
  the same completion again → a duplicate, not a second match; a truncation →
  `truncations` only, never a match; a revision → not a match; a new episode on
  the same slot → a new match; `match_completed and truncated` together →
  recorded error.
- The live G3-C run asserts its ledger is internally consistent and that it
  observed zero truncations, so the exclusion path is additionally covered
  synthetically rather than assumed.
- **Limits of this rule:** `account_step` keys on P2's `match_completed`, which
  P2 sets only for natural termination; it does not re-check `terminated` itself,
  and a row reporting both `match_completed` and `truncated` is counted as a
  match **and** recorded as an error. The collector also never compares its
  ledger with `env.completed_matches`, so a lost completion would not be caught
  independently.
- G3-D's cross-worker case uses the exact P1 collision shape: two rows with the
  same slot/generation/episode/seat/local id but different `worker_id` must own
  two cache entries. A mixed-version STEP must be rejected while leaving the
  live version's entry usable.

## 6. Optimisations applied (all semantics-preserving)

| Change | Effect |
|---|---|
| Batched decode: one `observe_fixed_batch` for all requesting seats | removed 2·N native calls per step |
| `decode_batch()`, proven byte-identical to per-seat decode | enables the above safely |
| Preallocated observation buffers | 2291 → 1410 µs per `_observation_td` at N=16 |
| Collector reuses the env's already-decoded observations | removed a second decode pass |
| `request_info_batch` for all rows/seats | 16 µs vs 2 native calls per row |
| Removed redundant per-row candidate revalidation from the hot path | Rust enforces legality on submission; strict mode retained |

## 6a. Files changed in P3 (tested commit `297cf8b`)

New: `pa4/collector/{batched_core,run_batched,multi_driver,collector_core,run_collector}.py`,
`pa4/actions/{batched_scorer,batched_factory_instrumented}.py`,
`pa4/env/obs_fastpath.py`, `tests/collector/*`, `docs/phases/P3/*`,
`pa4/collector/__init__.py`.
Modified: `pa4/env/native_env.py` (batched decode, fast observation path,
`request_info_batch` use), `pa4/actions/conditional_policy.py` (active→roster
threading, from P2), `docs/PROJECT_STATUS.md`.

Reproduce:

```bash
cd /home/pa4/projects/PokeAgent4.0
uv run --frozen python -m unittest discover -s tests/action_distribution -t .
uv run --frozen python -m unittest discover -s tests/environment -t .
uv run --frozen python -m unittest discover -s tests/collector -t .
```

Per-run performance rows are regenerated by `docs/phases/P3/evidence/scale_test.py`
and `tune_test.py`; those scripts do **not** reproduce every number in the
tables above (see the caveats in §2).

## 7. Limitations (explicit)

1. **`AsyncBatchedCollector` is not used.** ADR-008 (user-approved) amends the
   master criterion for P3 only. G3-A is a scoped PASS.
1a. **P3-C transition retention is absent**: no bounded `PA4ActorTransitionV1`
   records, so P4 cannot replay behaviour from stored evidence yet.
1b. **Inference is per-driver, not central**: no forward mixes two worker
   processes.
1c. **No weight/observation fingerprint** is stamped on P3 exchanges.
1d. **Validation gaps**: duplicate identity within one batch and foreign-version
   ENCODE are not rejected.
1e. **No lost-completion guard**; revisions count as behaviour rows.
1f. **P3-D evidence gaps**: no cold/steady split, no retained stage breakdown,
   and several headline numbers have no artifact.
2. **Throughput ~6 matches/s**, far below the 1,000/s stretch goal, which the
   design forbids proving on the 4060 Ti (§0.2).
3. **Bottleneck is TorchRL transport overhead**, measured but not resolved;
   fixing it is outside P3 scope.
4. **Baselines were measured at `a8e38b8`**; the collector code is identical at
   the tested commit, but the numbers were not re-measured there.
5. **The accumulation-sweep raw stats file was not retained**; its row is
   documented but not independently re-derivable from an artifact.
6. **Random PA3 weights** (seed 20261006). No strength or weight-parity claim.
7. **Path B (staged prefix) only.** Path A would remove sequential level
   round-trips but is out of scope.
8. **Transitions are in-memory only.** No replay database, per design.
9. **Multi-driver scaling is sublinear** — drivers share one GPU.
10. **Same-model review** is a process control, not independent model proof.

## 8. P4 handoff

P4 receives bounded, versioned behaviour records containing: the actual pre-action
observation, ordered candidate tables per level, selected indices, FP32 branch and
joint logprobs, policy version and weight fingerprint, native reward/termination,
and a fully-qualified match id. P4 owns GAE, PPO-2, the optimizer and checkpoint
parity — none of which exists in P3.

**P4 is not started.**
