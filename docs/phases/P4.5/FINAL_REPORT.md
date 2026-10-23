# P4.5 — Final Report (Performance Forensics, Diagnostic-Only)

**Design:** `PA4-P45-DIAG-DESIGN-v1.0` (SHA256 `4d6d65bb…`, verified)
**Baseline:** P4 at `d0f42f1` (272 tests, 0.4439 accepted matches/s). `d0f42f1` is the P4 *closure* commit; the P4 receipt records the *tested code* SHA as `7689ed1`. The two differ because P4's own receipt and docs landed after its tested-code commit, so `d0f42f1` contains no training-code change beyond `7689ed1`. Both are recorded in `ENVIRONMENT.json`.
**Verdict:** see `RECEIPT.json` — gates D45-0…D45-6 evaluated below
**No optimisation was implemented.** This phase produced measurement only.

---

## Executive summary

**The single decisive finding: the policy encoder is latency-bound, not
compute-bound.**

| Batch | Encode kernel | Per row |
|---|---|---|
| 1 | 5.583 ms | 5.583 ms |
| 8 | 5.634 ms | 0.704 ms |
| 16 | 7.168 ms | 0.448 ms |
| 32 | 10.286 ms | 0.321 ms |
| 64 | 10.85 ms | 0.17 ms |

A 64× wider batch costs 1.94× more CPU. Per-row cost falls **32.8×**. Roughly
5 ms of every forward is fixed overhead — weight streaming, launch, dispatch —
regardless of batch contents.

Every place the runtime calls the model with a batch of one, it discards about
5.3 ms of that fixed cost. The code does this **per value row** (753 calls in two
iterations) and **per decision** (once per inference round trip).

The second measured fact: **transport wait is 11.792 ms per request while payload
preparation is 0.179 ms** — a 66× ratio. The round trip, not the Python, is the
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
because fixed per-run setup is spread over 4 iterations here versus 20 in the historical trial (fewer iterations amortise setup LESS, so the 4-iteration rate is flattered by other factors and the two are not comparable) — **the two are
not directly comparable and are not being compared.**

## 3. Instrumentation overhead (D45-2)

| Run | Rate |
|---|---|
| uninstrumented median | 0.5338 m/s |
| instrumented (2 iters) | 0.5524 m/s |

**This is not a matched control:** the instrumented run used 2 iterations
while the uninstrumented median used 4, so per-iteration setup amortises
differently and the two rates are not directly comparable. The reliable
evidence for zero overhead is structural, not statistical: off-mode is
asserted inert (no counters, no file, gate defaults off, identical return
value) and the only production read of `prof.enabled` under the P4 loop is the guarded per-iteration dump at the end of `run_iteration`, which records nothing when disabled.
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

Within one learner evaluation the four components are measured **disjointly** by hoisting branch-table construction out of the evaluate loop (`evidence/d2f_learner_split.json`, 167 rows, 1094.7 ms): `evaluate_encoded` 52.2%, branch-table construction 25.6%, observation rebuild 17.6%, batched encoder 4.7% — which sum to 100.0% (the printed one-decimal values differ by 0.1 from rounding). The encoder is the one part already batched, and it is the cheapest.

**The hypothesised `.item()` synchronisation bottleneck is 0.3%** — effectively
absent. The cost is per-call kernel latency, not host sync.

## 5. Collection attribution (measured)

| Stage | Share of accounted collection |
|---|---:|
| `step_levels` (prefix STEP round trips) | **62.1%** |
| `encode` (ENCODE round trips) | **32.3%** |
| `env_step` (real Rust work) | 1.9% |
| `accounting` | 1.2% |
| `packet` + `reset` + `views` + `request_info` | 2.5% |

Per request: prep 0.179 ms, submit 0.343 ms, **wait 11.792 ms** — wait/prep = **65.99×**.

The existing collection stage timers cover 15807.7 ms of the 18760.0 ms measured collection wall. The remaining **2952.2 ms (15.7% of collection wall)** is not attributed by those timers; it includes service startup, snapshot load and shutdown (`d2b_collect.json`).

## 6. Hypothesis verdicts (D45-5)

| # | Hypothesis | Verdict | Deciding measurement |
|---|---|---|---|
| H1 | Per-row frozen V_old + `.item()` sync | **PARTIALLY** | encode is 23.2% of learner, but `.item()` is only 0.3% |
| H2 | microbatch 8 / per-row scorer / ragged CPU | **SUPPORTED** | `ppo_loop` is 66.2% of learner time; within one evaluation `evaluate_encoded` alone is 52.2% and branch-table construction 25.6%, both per row |
| H3 | InferenceServer IPC/queue barriers | **SUPPORTED** | wait 11.792 ms vs prep 0.179 ms (66×); step_levels + encode are 94.4% of accounted collection |
| H4 | Instrumentation in the production path | **REJECTED** | `PA4_P3_STATS_PATH` unset; profiler off by default and structurally inert. The instrumented run measured 0.5524 m/s vs 0.5338 uninstrumented, i.e. slightly *faster*, consistent with noise and not with overhead |
| H5 | Per-iteration model recreation | **PARTIALLY** | frozen critic rebuilt + state reloaded each iteration (the learner itself is built once and restored); measured 11.2 s attributed to construction over the 20-iteration baseline, 1.6% of wall |
| H6 | GPU saturated; faster GPU helps unmodified | **REJECTED** | 1.94× cost for 64× work — the batch-scaling measurement alone decides this; peak VRAM and a single idle-state clock sample are NOT used as utilization evidence |

## 7. Top five bottlenecks with Amdahl bounds

Every value below is copied from `BOTTLENECK_RANKING.csv`, which computes
each bottleneck's attributed wall time on ONE basis: the measured stage
share applied to the 20-iteration totals (collection 373.347 s,
update 332.808 s). Amdahl is 1/(1-f) with f = attributed/wall.

| Rank | Bottleneck | Attributed wall | % of 720.5 s | Confidence | Amdahl bound |
|---|---|---:|---:|---|---:|
| 1 | prefix STEP transport barrier | 231.85 s | 32.18% | HIGH | **1.474×** |
| 2 | ENCODE transport barrier | 120.59 s | 16.74% | HIGH | **1.201×** |
| 3 | per-row conditional scorer evaluation | 115.01 s | 15.96% | HIGH | **1.19×** |
| 4 | per-row frozen V_old encode | 77.12 s | 10.7% | HIGH | **1.12×** |
| 5 | branch-table construction | 56.4 s | 7.83% | HIGH | **1.085×** |

The ten ranked bottlenecks together attribute 649.4 s of 720.5 s (90.1%), on a single non-overlapping basis.
Full ten-row table with file/function, call counts (in section 4/5), CPU/GPU split and evidence
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
