# P4.5 — Architecture Decision

**Design:** `PA4-P45-DIAG-DESIGN-v1.0` §4.7
**Constraint:** this document recommends; it **implements nothing**.

---

## 1. The measurement that governs every decision

| Batch size | Encode kernel | Per row |
|---|---|---|
| 1 | **5.58 ms** | **5.58 ms** |
| 8 | 6.06 ms | 0.76 ms |
| 16 | 7.11 ms | 0.44 ms |
| 32 | 10.27 ms | 0.32 ms |
| **64** | **10.85 ms** | **0.17 ms** |

A **64× larger batch costs 1.94× more.** Per-row kernel cost falls **32.8×**.

This is a **latency-bound** kernel: roughly 5 ms of fixed cost (weight streaming
from VRAM + launch + device-side dispatch) regardless of how much work is in
the batch. Every per-row or per-small-batch call throws away ~5 ms.

**Consequence:** the dominant lever in this codebase is *batch width*, not
FLOPs, not IPC speed, and not the GPU model.

## 2. Transport measurement

Per inference request (`evidence/d2c_split.json`):

| Component | ms |
|---|---|
| payload prep (Python) | 0.179 |
| submit (IPC send) | 0.343 |
| **wait** (queue + decode + GPU + return) | **11.792** |

Wait is **65.99× prep**. The measured in-process batched encode is 47.1 ms for 183 rows — i.e. batching is far cheaper than this transport path. So a large share of the 11.792 ms is **queue and round-trip
dwell**, not computation.

## 3. Amdahl bounds (measured, sequential topology)

| Candidate | fraction of wall | **upper bound** |
|---|---:|---:|
| Eliminate collection entirely | 0.518 | **2.08×** |
| Eliminate update entirely | 0.462 | **1.86×** |
| Batch both learner paths | 0.413 | **1.70×** |
| Eliminate `step_levels` | 0.322 | 1.47× |
| Eliminate `ppo_loop` per-row | 0.306 | 1.44× |
| Eliminate `encode` | 0.167 | 1.20× |
| Eliminate per-row V_old | 0.107 | 1.12× |

**These are upper limits with everything else unchanged. They are not
predictions and they cannot be added** — collection and update improvements
multiply, they do not sum.

**No single fix exceeds 2.08×, and doing all the learner batching perfectly
gives at most 1.70×.** With the collection transport also batched, the
compounded ceiling is **1.70 × 1.47 ≈ 2.50×** if the STEP transport barrier were
removed entirely (which is itself a bound, not a forecast).

Under the current topology, **~1.1 matches/s is the realistic ceiling with
everything batched**. The design's D6 requirement is satisfied: the
architectural question is not "which micro-optimisation" but "does the
topology itself have to change".

## 4. Candidate architectures

### Option A — In-process batched inference (leading candidate)

Bring the policy model into the actor process and score all pending decisions in
one forward per phase.

| Dimension | Assessment |
|---|---|
| Evidence for | Batch-64 per-row cost is **32.8× lower** than batch-1; `step_levels` + `encode` are 94% of accounted collection and both are per-request round trips; wait is 67× prep |
| Correctness risk | **Medium.** Must preserve the exact `ObservationBatch` including all five ragged families, ordered candidate tables, saved indices and GRU prefix carry. PA3's `evaluate_encoded` already handles multi-level prefix carry, and P4-B proved stored-action replay to `1e-7` through that same call |
| Implementation complexity | Medium: replaces `StagedInference` with a local scorer; the learner already loads snapshots and the collection path already builds `ObservationBatch` objects |
| TorchRL compatibility | **Reduces** TorchRL surface. ADR-008 already permits this: stock `AsyncBatchedCollector` cannot query staged Rust candidates |
| GPU residency | Actor and learner cannot both hold the model; P4 already mandates sequential residency, so this is consistent |
| Expected full-stack effect | Removes the transport wait *and* widens batches. Compounded Amdahl ceiling ≈ **2.5×**, not 10–30× |

**What would falsify it:** an A/B where in-process inference produces a
different `joint_logprob` beyond `1e-5` from the current served path on the same
observations, or where actor+learner residency exceeds 8 GB.

### Option B — Optimised PA3-style actor adapter

Adapt PA3's production actor loop, which was measured at 43.5 natural
matches/s during PA3 training on dual V100s.

| Dimension | Assessment |
|---|---|
| Evidence for | PA3 achieved materially higher actor throughput; its loop is designed for batched tokenised actor traffic |
| Evidence against | **The comparison is not like-for-like**: dual V100 vs single 4060 Ti, different observation transport, different policy versioning, and PA3's number is a full training rate, not actor-only |
| Correctness risk | **High.** PA3's actor has its own trajectory and accounting conventions that must be re-proven against PA4's verified contracts (Ragged v2, snapshot identity, per-seat GAE, lossless ingestion) |
| Implementation complexity | High: effectively a second actor implementation to keep in parity |
| Expected effect | Unknown from current evidence. **Not measured on this hardware** |

**Judgment:** this is a promising *source of design ideas* (especially its batch
assembly) but it is **not currently evidence-backed as a wholesale
replacement**. Adopting it would put the P4 correctness gains at risk for an
unmeasured gain.

### Option C — Current TorchRL InferenceServer, tuned and persistent

Keep the architecture; make the server persistent across iterations and batch
more aggressively.

| Dimension | Assessment |
|---|---|
| Evidence for | Cheapest change; no correctness risk to the served path |
| Evidence against | The wait is 67× prep, so the round-trip structure itself is the cost; tuning `min_batch_size`/`timeout` was already explored in P3 and larger values were *slower*. Persistence saves ~0.2 s/iteration of startup against a 36 s iteration |
| Expected effect | Bounded by `eliminate_encode + eliminate_step_levels` ≈ **1.5×**, and likely much less |

**Judgment:** worth doing *as part of* A, not instead of it.

### Option D — Batch the learner paths only (no actor change)

| Dimension | Assessment |
|---|---|
| Evidence for | `vold_encode` + `ppo_loop` are 89.4% of learner time; batching the encoder already proved numerically identical (`4.8e-07`) |
| Correctness risk | **Low–medium:** must pad branch tables to a common level count without disturbing prefix carry. P4-B's prefix-carry bug is the exact failure mode to guard against, and the existing parity tests can gate it |
| Expected effect | **≤1.70×**, realistically 1.4–1.6× |

**Judgment: this is the right *first* implementation step** regardless of what
happens to the actor, because it is bounded, testable and independent.

## 5. Recommendation

1. **Implement Option D first.** It is the lowest-risk, highest-confidence step,
   gated by the existing replay-parity tests. Expected ≤1.70×, realistically
   1.4–1.6×.
2. **Then evaluate Option A** with a bounded A/B: same observations, same
   snapshots, compare served logprobs and wall time. This is the only path that
   attacks both the transport wait and the batch width together.
3. **Do not adopt Option B** without a direct measurement on this hardware.
   Its evidence is confounded by different GPUs and a different metric.
4. **Treat Option C as a satellite of A**, not an alternative.

## 6. Honest ceiling statement

Even with **D + A** fully implemented, the compounded Amdahl ceiling on this
topology is **~2.5×**, taking 0.44 → roughly **1.1 matches/s**. (1.70 for the
learner times 1.47 for the STEP barrier; removing *all* collection would be
2.08×, which is not a claim made here.)

**That is still far below a serious training rate.** Reaching hundreds to
thousands of matches/s requires changing what is computed per decision, not how
fast it is computed — for example, running many battles in one batched
environment step, or removing the per-decision Python materialisation entirely.
Those are P6/P7 designs, not P4.5 decisions.

## 7. C1 and P5 readiness (different question from throughput)

- **C1 (Runpod compatibility smoke):** P4.5 found **no correctness regression**.
  The runtime is slow but correct. C1 is a *compatibility* check, not a training
  campaign, and the current architecture is safe to validate there. Recommend
  proceed **when authorized**.
- **P5 (evaluation league):** P5 needs correct matches and policy identity, both
  of which P4 verified. Slow collection makes P5 experiments slower but not
  wrong. Recommend P5 may proceed in parallel with optimisation work, provided
  the league size is sized to the measured rate.

Neither recommendation is an approval for a costly production training run.

## 8. Falsifiable follow-up A/B experiments

| Experiment | Tests | Falsified if |
|---|---|---|
| E1: batched learner scoring vs per-row | D's correctness | max joint-logprob delta > 1e-5 on stored rows |
| E2: in-process inference vs InferenceServer on identical observations | A's correctness | served logprob delta > 1e-5, or VRAM > 8 GB |
| E3: batch-width sweep on the actor path | the latency hypothesis | throughput does not improve with wider batches |
| E4: PA3 actor adapter on identical cohorts | B's applicability | it fails any PA4 contract gate |
