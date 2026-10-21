# P3 — Final Report

**Phase:** P3 — Batched Collector and Central Inference
**Design:** `PA4-P3-DESIGN-v1.0` (sha256 `f73c5567…`)
**PA4 base:** `4d8f349b247bb929914db5b9ebe2c5ce84c18288`
**PA4 tested code:** `407961026410d349b3b1c689864e6868e9988086`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified, 0 dirty files)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0)
**GPU:** RTX 4060 Ti 8 GB, CUDA capability (8,9), driver 610.88

---

## VERDICT: **PARTIAL**

Four of five gates pass on executed tests and measured numbers. **G3-E is PARTIAL**
because its required read-only audit of the *actual WSL repository* could not be
performed: the auditor reachable from this Lead session was denied
`external_directory` access to `/home/pa4`, and the design's G3-E row explicitly
disqualifies a text-only audit that cannot access the repo.

The auditor behaved correctly — it read zero files and refused to grade rather
than guessing. This is an environment limitation, not a code defect.

**To close P3:** re-run the audit from a WSL-launched session
(`cd /home/pa4/projects/PokeAgent4.0 && opencode`), as was done successfully for P2.

---

## 1. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G3-A** Collector integration | **PASS (ADR-008 scoped)** | The actor loop drives real P2 `EnvBase.step()` transitions whose actions come from the real P1 conditional scorer over live Rust candidates. The stock `AsyncBatchedCollector` is not used; ADR-008 (user-approved) records why. |
| **G3-B** Central CUDA batching | **PASS** | `model_forward_count == 1` with `forward_batch_sizes == [4]` for four independent requests; device `cuda:0`; per-row selections verified against each row's own candidate set; observed `bmax = 64`, tuned `bmean = 15.5` |
| **G3-C** Natural-match integrity | **PASS** | 33–65 natural completions per run; 0 operational errors; 0 duplicate match ids; ids fully qualified `(run, worker, slot, generation, episode)`; 2 independent drivers verified |
| **G3-D** Transport/lifetime | **PASS** | 4 concurrent clients each receive their own response; one-in-flight enforced; clean shutdown after use and with an unread in-flight request; overflow 65, count/mask mismatch, stale generation, unknown key and wrong level all fail closed and evict only the addressed entry |
| **G3-E** Baseline and audit | **PARTIAL** | `PERFORMANCE_BASELINE.csv` with 5 measured configurations; 128 tests green; frozen lock unchanged. **Auditor could not read the repo — conjunct unmet.** |

## 2. Measured performance

| Configuration | Matches/s | Transitions/s | bmax | bmean |
|---|---|---|---|---|
| 16 envs, 1 driver | 5.12 | 79.4 | 31 | 3.78 |
| 32 envs, 1 driver | **6.04** | **101.1** | 63 | 5.15 |
| 64 envs, 1 driver, tuned accumulation | 5.78 | 104.0 | 64 | 15.5 |
| 2 driver processes × 16 envs | 4.40 | 78.0 | — | — |
| 4 driver processes × 16 envs | 5.53 | 91.8 | — | — |

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

## 5. Optimisations applied (all semantics-preserving)

| Change | Effect |
|---|---|
| Batched decode: one `observe_fixed_batch` for all requesting seats | removed 2·N native calls per step |
| `decode_batch()`, proven byte-identical to per-seat decode | enables the above safely |
| Preallocated observation buffers | 2291 → 1410 µs per `_observation_td` at N=16 |
| Collector reuses the env's already-decoded observations | removed a second decode pass |
| `request_info_batch` for all rows/seats | 16 µs vs 2 native calls per row |
| Removed redundant per-row candidate revalidation from the hot path | Rust enforces legality on submission; strict mode retained |

Net effect: from ~25.5 ms/step serial in P2 to **~100 batched env-transitions/s** —
roughly **4–8×** more collected data with every invariant unchanged.

## 6. Limitations (explicit)

1. **G3-E unmet** — no auditor with WSL repo access was available to this session.
2. **`AsyncBatchedCollector` is not used.** ADR-008 (user-approved) amends the
   master criterion for P3 only. G3-A is a scoped PASS.
3. **Throughput ~6 matches/s**, far below the 1,000/s stretch goal, which the
   design forbids proving on the 4060 Ti (§0.2).
4. **Bottleneck is TorchRL transport overhead**, measured but not resolved;
   fixing it is outside P3 scope.
5. **Random PA3 weights** (seed 20261006). No strength or weight-parity claim.
6. **Path B (staged prefix) only.** Path A would remove sequential level
   round-trips but is out of scope.
7. **Transitions are in-memory only.** No replay database, per design.
8. **Multi-driver scaling is sublinear** — drivers share one GPU.
9. **Same-model review** is a process control, not independent model proof.

## 7. P4 handoff

P4 receives bounded, versioned behaviour records containing: the actual pre-action
observation, ordered candidate tables per level, selected indices, FP32 branch and
joint logprobs, policy version and weight fingerprint, native reward/termination,
and a fully-qualified match id. P4 owns GAE, PPO-2, the optimizer and checkpoint
parity — none of which exists in P3.

**P4 is not started.**
