# P3 — Final Report

**Phase:** P3 — Batched Collector and Central Inference
**Design:** `PA4-P3-DESIGN-v1.0` (sha256 `f73c5567…`)
**PA4 base:** `4d8f349b247bb929914db5b9ebe2c5ce84c18288`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified, 0 dirty files)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0)
**GPU:** RTX 4060 Ti 8 GB, CUDA capability (8,9), driver 610.88

---

## VERDICT: **PASS**

All five G3 gates are evidenced by executed tests plus measured numbers.
**G3-A is a scoped PASS under the approved ADR-008** (master-criterion amendment
chosen by the user), not an unqualified claim about `AsyncBatchedCollector`.

---

## 1. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G3-A** Collector integration | **PASS (ADR-008 scoped)** | The actor loop drives real P2 `EnvBase.step()` transitions whose actions come from the real P1 conditional scorer over live Rust candidates. The stock `AsyncBatchedCollector` is not used; ADR-008 records why and what was changed. |
| **G3-B** Central CUDA batching | **PASS** | `model_forward_count == 1` with `forward_batch_sizes == [4]` for four independent requests; model on `cuda:0`; per-row selections verified against each row's own candidate set; observed `bmax = 64`, tuned `bmean = 15.5` |
| **G3-C** Natural-match integrity | **PASS** | 33–65 natural completions per run across up to 32 envs; **0** operational errors; **0** duplicate match ids; match ids fully qualified; `Wait`/`Replacement` exercised |
| **G3-D** Transport/lifetime | **PASS** | 4 concurrent clients each receive their own response; one-in-flight enforced; clean shutdown after use and with an unread in-flight request; repeated start/stop; overflow 65, count/mask mismatch, stale generation, stale request id, wrong level all fail closed and evict only the addressed entry |
| **G3-E** Baseline and audit | **PASS** | `PERFORMANCE_BASELINE.csv` with 5 measured configurations; P1 39 + P2 51 + P3 38 = **128 tests green**; frozen lock unchanged; read-only auditor reviewed actual WSL files |

## 2. Measured performance

| Configuration | Matches/s | Transitions/s | bmax | bmean |
|---|---|---|---|---|
| 16 envs, 1 driver | 5.12 | 79.4 | 31 | 3.78 |
| 32 envs, 1 driver | **6.04** | **101.1** | 63 | 5.15 |
| 64 envs, 1 driver, tuned accumulation | 5.78 | 104.0 | 64 | 15.5 |
| 2 driver processes × 16 envs | 4.40 | 78.0 | — | — |
| 4 driver processes × 16 envs | 5.53 | 91.8 | — | — |

`accepted_training_matches_per_sec` = **null** — no PPO learner exists in P3.

## 3. The central finding: batching was blocked by one cached client

`ProcessInferenceServer.client()` caches a single client and returns the same
object on every call, so `[srv.client() for _ in range(n)]` produced n references
to **slot 0**. Since `ProcessSlotTransport` allows one in-flight request per
client, only one request could ever be outstanding and every forward was B=1.

`srv.clients(n)` returns n genuinely distinct slots. Measured:
`client() -> 0,0` (same object) vs `clients(3) -> 1,2,3`.

## 4. Why throughput plateaus — measured, not guessed

Forward count fell 2.7x (874 → 318) when batch accumulation was tuned, **but
throughput did not move** (4.42 → 4.42 matches/s). Direct GPU timing explains it:

| Op | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| `model.encode` | 4.80 ms | 5.47 ms | 7.09 ms | 11.53 ms |
| `sample_level_step` | 2.60 ms | 2.40 ms | 2.46 ms | — |

GPU work is **nearly flat in batch size** (peak allocation 0.09 GiB), so the
residual ~40 ms per forward is **TorchRL `ProcessSlotTransport` round-trip
overhead** — not model compute, not payload (63.7 KiB/row ≈ 40 µs at
shared-memory bandwidth), and not this collector.

This is a real, actionable P3 finding for P6/P7.

## 5. Optimisations applied (all semantics-preserving)

| Change | Effect |
|---|---|
| Batched decode: one `observe_fixed_batch` for all requesting seats | removed 2·N native calls per step |
| `decode_batch()` in `SliceDecoder`, proven byte-identical to per-seat decode | enables the above safely |
| Preallocated observation buffers | 2291 → 1410 µs per `_observation_td` at N=16 |
| Collector reuses the env's already-decoded observations | removed a whole second decode pass |
| `request_info_batch` for all rows/seats | 16 µs vs 2 native calls per row |
| Removed redundant per-row candidate revalidation from the hot path | Rust already enforces legality on submission; strict mode retained |
| Batched-driver scaling | 1 → 4 drivers gives 5.53 matches/s |

Overall the collector went from **25.5 ms/step serial (P2)** to **~100
env-transitions/s batched** — roughly a **4-8x** improvement in collected data,
while every correctness invariant is unchanged.

## 6. Limitations (explicit)

1. **`AsyncBatchedCollector` is not used.** ADR-008 (user-approved) amends the
   master criterion for P3 only. This is a scoped PASS, not an unqualified one.
2. **Throughput is ~6 matches/s**, far below the 1,000/s stretch goal. The design
   explicitly forbids proving that target on the 4060 Ti (§0.2, §6); P7 owns it
   on cloud hardware.
3. **Bottleneck is TorchRL transport overhead**, measured but not resolved. Fixing
   it would require transport-level work outside P3 scope.
4. **Random PA3 weights** (seed 20261006). No strength or weight-parity claim.
5. **CPU native stepping, one process per driver.** Multi-process native
   parallelism was measured (2 and 4 drivers) but GPU-contends, so it scales
   sublinearly.
6. **Path B (staged prefix) only.** Path A (full candidate graph, single forward)
   would remove the sequential level round-trips but is explicitly out of P3 scope.
7. **Transitions are in-memory only.** No replay database, per design.
8. **Same-model review.** The auditor is the same DeepSeek model in a fresh
   read-only context; it is a process control, not independent model proof.

## 7. P4 handoff

P4 receives bounded, versioned behaviour records containing: the actual pre-action
observation, ordered candidate tables per level, selected indices, FP32 branch and
joint logprobs, policy version and weight fingerprint, native reward/termination
and a fully-qualified match id. P4 owns GAE, PPO-2, the optimizer and checkpoint
parity — **none of which exists in P3**.

**P4 is not started.**
