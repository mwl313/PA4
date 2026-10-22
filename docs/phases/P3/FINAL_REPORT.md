# P3 — Final Report

**Phase:** P3 — Batched Collector and Central Inference
**Design:** `PA4-P3-DESIGN-v1.0` (sha256 `f73c5567…`)
**PA4 base:** `4d8f349b247bb929914db5b9ebe2c5ce84c18288`
**PA4 tested code:** `297cf8b7048c6ec1f1841fb00b443a868963520b`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified, 0 dirty files)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0)
**GPU:** RTX 4060 Ti 8 GB, CUDA capability (8,9), driver 610.88

---

## VERDICT: **PENDING_FINAL_AUDIT**

G3-A–G3-D pass on executed tests and measured numbers. G3-E's second conjunct is
a read-only `pa4-auditor` that reads the **actual WSL repository files**. A
re-audit of this commit is in progress from a WSL-launched Lead session; access
has already been demonstrated (`ACCESS_OK`, literal file reads). This document is
written before the audit returns so that the auditor reviews the real numbers, and
the verdict is set to PASS only if that review is clean.

The previous PARTIAL was for the auditor-access conjunct alone; it never claimed
the implementation failed.

---

## 1. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G3-A** Collector integration | **PASS (ADR-008 scoped)** | The actor loop drives real P2 `EnvBase.step()` transitions whose actions come from the real P1 conditional scorer over live Rust candidates. The stock `AsyncBatchedCollector` is not used; ADR-008 (user-approved) records why. |
| **G3-B** Central CUDA batching | **PASS** | `model_forward_count == 1` with `forward_batch_sizes == [4]` for four independent requests, device `cuda:0`; each row's `branch_k`, `request_key` and selected index checked against its **own** request; cache isolation asserted. Real collection reached `bmax = 63` (retained stats) and `bmax = 64` (baseline sweep). |
| **G3-C** Natural-match integrity | **PASS** | ≥16 natural completions per run (33–65 across baselines); 0 operational errors; 0 duplicate ids; ids fully qualified `(run, worker, slot, generation, episode)`; 2 independent driver processes verified. Exactly-once accounting now has direct unit tests, not a tautology. |
| **G3-D** Transport/lifetime | **PASS** | 4 concurrent clients each receive their own response; one in-flight per client enforced; clean shutdown after use and with an unread in-flight request; repeated start/stop; overflow 65, count/mask mismatch, unknown key, wrong level, stale generation, duplicate encode, **cross-worker local-id aliasing** and **mixed policy version** all fail closed without corrupting the live entry. |
| **G3-E** Baseline and audit | **PENDING** | `PERFORMANCE_BASELINE.csv` (5 measured configurations, provenance per row); 136 tests green; frozen lock unchanged; PA3 clean. **Auditor conjunct under re-check.** |

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

GPU work is **nearly flat in batch size** (peak allocation 0.09 GiB), so the
residual ~40 ms per forward is **TorchRL `ProcessSlotTransport` round-trip
overhead** — not model compute, not payload (63.7 KiB/row ≈ 40 µs at
shared-memory bandwidth), and not this collector.

**This is an actionable P3 finding for P6/P7.**

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

## 7. Limitations (explicit)

1. **`AsyncBatchedCollector` is not used.** ADR-008 (user-approved) amends the
   master criterion for P3 only. G3-A is a scoped PASS.
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
