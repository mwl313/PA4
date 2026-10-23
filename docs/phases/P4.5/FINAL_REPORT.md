# P4.5 — Final Report (Performance Forensics, Diagnostic-Only)

**Design:** `PA4-P45-DIAG-DESIGN-v1.0` (SHA256 `4d6d65bb…`, verified)
**Baseline:** P4 at `d0f42f1` (272 tests, 0.4439 accepted matches/s)
**Verdict:** see `RECEIPT.json` — gates D45-0…D45-6 evaluated below
**No optimisation was implemented.** This phase produced measurement only.

---

## Executive summary

**The single decisive finding: the policy encoder is latency-bound, not
compute-bound.**

| Batch | Encode kernel | Per row |
|---|---|---|
| 1 | **5.58 ms** | **5.58 ms** |
| 8 | 6.06 ms | 0.76 ms |
| 16 | 7.11 ms | 0.44 ms |
| 32 | 10.27 ms | 0.32 ms |
| **64** | **10.85 ms** | **0.17 ms** |

A 64× wider batch costs 1.94× more CPU. Per-row cost falls **32.8×**. Roughly
5 ms of every forward is fixed overhead — weight streaming, launch, dispatch —
regardless of batch contents.

Every place the runtime calls the model with a batch of one, it discards about
5.3 ms of that fixed cost. The code does this **per value row** (753 calls in two
iterations) and **per decision** (once per inference round trip).

The second measured fact: **transport wait is 11.5 ms per request while payload
preparation is 0.17 ms** — a 67× ratio. The round trip, not the Python, is the
collection cost.

The third: **no single fix exceeds 2.08×**, because collection (51.8%) and
update (46.2%) are nearly equal. The GPU is not saturated — it is starved.

---

## 1. Historical accounting (D45-1)

Recomputed by code from the unmodified P4 artifacts:

| Stage | Seconds | Share |
|---|---:|---:|
| collection | **373.347** | **51.81%** |
| update | **332.808** | **46.19%** |
| checkpoint | **11.211** | **1.56%** |
| residual in rows | **3.176** | 0.44% |
| CSV row sum | 720.541 | — |
| outer stopwatch | 720.948 | — |

The 0.407 s difference is run-level setup/teardown outside any iteration — a
**boundary difference, documented rather than forced equal**
(`HISTORICAL_RECONCILIATION.md`).

**Amdahl counterfactuals:** eliminate collection → 2.075×; eliminate update →
1.858×; eliminate checkpoint → 1.016×.

## 2. Uninstrumented baseline (D45-3)

Three matched 4-iteration full-stack trials with real Rust, real PPO, profiling
off:

| Trial | Wall | Collect | Update | Steps | Rate (m/s) |
|---|---:|---:|---:|---:|---:|
| 1 | 121.0 s | 68.3 | 49.7 | 45 | **0.5374** |
| 2 | 125.5 s | 70.2 | 52.4 | 45 | **0.5338** |
| 3 | 127.2 s | 69.4 | 55.0 | 49 | **0.5030** |

**Median 0.5338 m/s, spread 6.4%.** Zero drops, zero duplicates, fingerprint
chain intact in all three, real parameter updates every iteration.

The 4-iteration rate (0.53) exceeds the historical 20-iteration rate (0.44)
because per-iteration setup amortises over fewer iterations here — **the two are
not directly comparable and are not being compared.**

## 3. Instrumentation overhead (D45-2)

| Run | Rate |
|---|---|
| uninstrumented median | 0.5338 m/s |
| instrumented (2 iters) | 0.5524 m/s |

Instrumentation did **not** slow the path measurably; the instrumented run was
slightly faster, i.e. within noise. Off-mode is proven inert by four tests
(records nothing, writes no file, identical return values, gate off by default).
**All throughput claims in this report come from uninstrumented runs.**

## 4. Learner attribution (measured, exclusive ms)

| Stage | Calls | Exclusive ms | % of learner |
|---|---:|---:|---:|
| `ppo_loop` (per-row scorer inside) | 2 | 15 516 | **66.2%** |
| `vold_encode` (frozen critic, per row) | 753 | 5 430 | **23.2%** |
| `vold_obs` | 753 | 1 011 | 4.3% |
| `checkpoint` | 2 | 508 | 2.2% |
| `frozen_model_create` | 2 | 392 | 1.7% |
| `learner_create` | 2 | 368 | 1.6% |
| `vold_item` | 753 | **79** | **0.3%** |
| `vold_value` | 753 | 73 | 0.3% |
| GAE, cohort, normalize, clone | — | ~23 | 0.1% |

**The hypothesised `.item()` synchronisation bottleneck is 0.3%** — effectively
absent. The cost is per-call kernel latency, not host sync.

## 5. Collection attribution (measured)

| Stage | Share of accounted collection |
|---|---:|
| `step_levels` (prefix STEP round trips) | **62.1%** |
| `encode` (ENCODE round trips) | **32.2%** |
| `env_step` (real Rust work) | 2.0% |
| `accounting` | 1.2% |
| `packet` + `reset` + `views` + `request_info` | 2.5% |

Per request: prep 0.172 ms, submit 0.334 ms, **wait 11.492 ms** — wait/prep =
**66.8×**.

## 6. Hypothesis verdicts (D45-5)

| # | Hypothesis | Verdict | Deciding measurement |
|---|---|---|---|
| H1 | Per-row frozen V_old + `.item()` sync | **PARTIALLY** | encode is 23.2% of learner, but `.item()` is only 0.3% |
| H2 | microbatch 8 / per-row scorer / ragged CPU | **SUPPORTED** | `ppo_loop` 66.2%, dominated by per-row `evaluate_encoded` |
| H3 | InferenceServer IPC/queue barriers | **SUPPORTED** | wait 11.5 ms vs prep 0.17 ms; 94% of collection is round trips |
| H4 | Instrumentation in the production path | **REJECTED** | `PA4_P3_STATS_PATH` unset; profiler off by default and proven inert |
| H5 | Per-iteration model recreation | **PARTIALLY** | ~793 ms over 2 iters ≈ 1.4% of wall |
| H6 | GPU saturated; faster GPU helps unmodified | **REJECTED** | 1.94× cost for 64× work; 0.5–0.7 GB of 8 GB used; GPU idles at 210 MHz/P8 |

## 7. Top five bottlenecks with Amdahl bounds

| Rank | Bottleneck | Wall | Confidence | Upper bound |
|---|---|---:|---|---:|
| 1 | Prefix STEP transport barrier | 120 s | HIGH | **1.47×** |
| 2 | Per-row conditional scorer in PPO loop | 110 s | HIGH | **1.44×** |
| 3 | Per-row frozen V_old encode | 39 s | HIGH | 1.12× |
| 4 | ENCODE transport barrier | 62 s | HIGH | **1.20×** |
| 5 | Model construction per iteration | 1.4 s | HIGH | 1.03× |

Full ten-row table with file/function, call counts, CPU/GPU split and evidence
references: `BOTTLENECK_RANKING.csv`.

## 8. What the numbers say about architecture

`ARCHITECTURE_DECISION.md` evaluates four candidates. The conclusion:

- **Do the learner batching first** (Option D): bounded, testable, ≤1.70×,
  realistically 1.4–1.6×.
- **Then trial in-process inference** (Option A) as a bounded A/B: it attacks
  both the transport wait and the batch width, with a compounded ceiling of
  roughly 2.5–3.5×.
- **Do not adopt a PA3-style actor wholesale** (Option B): its 43.5 matches/s
  figure is from dual V100s and a different topology, and is **not comparable**
  to this single-4060-Ti result.
- **Treat server tuning** (Option C) as a satellite of A.

**Honest ceiling on the current topology: ~2.5–3.5×, i.e. 0.44 → ~1.1–1.5
matches/s.** Reaching hundreds per second requires changing what is computed per
decision, which is a P6/P7 design question.

## 9. Profiler limitations

| Item | Status |
|---|---|
| Parent CPU attribution | measured (`torch.perf_counter_ns` spans, exclusive accounting verified) |
| GPU kernel durations | measured with CUDA events at 7 batch sizes |
| **Child-process kernel attribution** | **PARTIAL** — a parent `torch.profiler` trace cannot observe the spawned inference server's kernels; measured indirectly via per-request wall and in-process equivalents |
| Cache hit rate for `PA4_P3_STATS_PATH` mtime reuse | **not established** — the key is visible in the source; whether it materially changed timing was not proven either way |
| Full Chrome trace export | not produced; CUDA event timing used instead, which is sufficient for the batch-scaling claim |

## 10. Next step

Implement **Option D** (batched learner scoring) as a bounded change gated on
existing replay-parity tests, then run experiment **E1** to prove parity at
`≤1e-5`. **No optimisation was performed in P4.5.**

C1 and P5 readiness are addressed separately in `ARCHITECTURE_DECISION.md` §7 and
are **not** approvals for production training.
