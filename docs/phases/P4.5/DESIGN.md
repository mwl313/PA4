# PokeAgent 4.0 — P4.5 Performance Forensics, Diagnostic-Only Design

**Document ID:** `PA4-P45-DIAG-DESIGN-v1.0`  
**Date:** 2026-10-23 (Asia/Seoul)  
**Status:** Execution-ready design; **not** an optimization approval or a new production profile  
**Baseline repository:** `mwl313/PA4`, reviewed remote `main` at `d0f42f1e127f166864aa94bff7e6b2b7f864016a`  
**Immutable PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`  
**Execution location:** `/home/pa4/projects/PokeAgent4.0`, WSL2 Ubuntu 24.04, RTX 4060 Ti 8GB  
**Operator:** `pa4-lead` / DeepSeek; fresh-context, read-only `pa4-auditor`  
**Dependency reference:** PyTorch 2.9.1+cu128, TorchRL 0.14.1, TensorDict 0.14.3, frozen `uv.lock` SHA256 `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e`.

## 0. Executive decision

P0, P1, P2, P3, P3.5 and P4 have passed their scoped gates. P4 proves an end-to-end, live, synchronous on-policy PPO-2 system. It **does not** prove high throughput or competitive policy strength. The next diagnostic task is to identify, quantify and classify the bottlenecks **without changing learning behavior**. This is intentionally a narrow, local, read-only-with-respect-to-the-baseline work package that precedes the separately authorized C1 cloud compatibility smoke and P5 self-play/evaluation work.

**Required output:** An evidence-backed attribution of the approximately 721 seconds consumed by the P4 20-iteration trial, identification of the top causes of CPU stalls, CUDA kernel work, transport waits, repeated encoding, model reloads and data marshaling, and a ranked, falsifiable future optimization roadmap. **No speed optimization is to be implemented or promoted during P4.5.**

### 0.1 Hard boundaries

- Freeze the accepted P4 commit as an immutable reference. Make instrumentation-only changes on a **new local branch** named `diagnostics/p45-performance-forensics` or an appropriately collision-safe equivalent; do not rewrite P4 history.
- Do not alter Rust battle semantics, PA3 pinned source, observations, five ragged families, legal candidates, selected actions, action distributions, reward accounting, PPO loss, optimizer, early-stop, LR schedule, checkpoint schema, RNG semantics or benchmark acceptance conditions.
- No change to `uv.lock`, package versions, compilation mode, mixed precision, TF32/BF16, batch size, actor topology, async policy logic, policy model architecture or TorchRL integration. No C1, P5 or production training; no RunPod expenditure.
- Preserve real game completion, full two-seat accounting, **lossless** cohorts, behavior-policy fingerprint continuity, full Ragged v2 and checkpoint integrity.
- Code added to instrument/profile the system must be **default-off**. Existing uninstrumented behavior must remain unchanged, preferably with tests proving this.
- Audit and locally commit diagnostic artifacts. **Do not push** or merge to the protected/reference baseline without user approval.

## 1. Established facts and what is not yet established

### 1.1 Authoritative repository evidence

1. `docs/PROJECT_STATUS.md` at the reviewed `main` reports P4 PASS, 272 tests, 20 completed real PPO iterations, 320 generated/accepted natural matches, zero drops/duplicates, 265 optimizer steps, 20 policy versions, `0.4439` accepted matches/s, peak recorded torch VRAM allocation `574.6 MiB`.
2. `docs/phases/P4/FINAL_REPORT.md`, `RECEIPT.json`, `FULL_STACK_BASELINE.csv` and `evidence/full_run.json` are the current ground truth for the 20-iteration trial. The fresh-process checkpoint recovery is separately evidenced and PASS, not an open blocker.
3. `pa4/learner/train_sync.py::SynchronousTrainer.run_iteration` performs a strictly sequential collect -> cohort -> frozen critic V_old -> GAE -> PPO -> atomic checkpoint cycle. The 8GB development profile uses 4 environments, 16 accepted matches per iteration, effective minibatch 64, microbatch 8, PPO-2, eager FP32.
4. `pa4/collector/run_batched.py` drives the staged `ENCODE` and prefix-by-prefix `STEP` requests; `pa4/collector/batched_core.py::StagedInference` waits for `future.result()` per phase. `pa4/actions/batched_scorer.py` executes the actual frozen PA3 model on the GPU.
5. `pa4/actions/batched_factory_instrumented.py::_Instrumented.forward` writes a JSON statistics file whenever the `PA4_P3_STATS_PATH` path is enabled. **Whether this per-forward file-write path materially contributes in the final P4 configuration is a hypothesis to measure, not a proven bottleneck.**
6. `train_sync.py` computes frozen `V_old` with an **individual forward per value row** and immediately converts GPU values using `.item()` inside a Python loop. This creates a plausible launch and synchronization bottleneck; measure it.
7. `pa4/learner/learner_eval.py::evaluate_stored_batch` batches `model.encode` across each microbatch but constructs observations per row, assembles ragged data through CPU/NumPy and then evaluates conditional branches row-by-row. Separate the costs instead of labeling all of it “GPU training.”
8. `docs/phases/P3/INTEGRATION_DECISION.md` documents the approved ADR-008 custom staged Collector fallback: stock `AsyncBatchedCollector` could not legally query the Rust-native prefix candidate API. This is a *compatibility fact*, not proof that TorchRL itself is the bottleneck.
9. Historical P3 and P3.5 actor-only numbers are under different configurations; they are **not** comparable to the P4 full-stack benchmark. Historical P3.5 `stage_perf.txt` reports substantial `ENCODE` and `STEP` wall time, which motivates but does not prove the current P4 diagnosis.

### 1.2 Recomputed P4 accounting from `FULL_STACK_BASELINE.csv`

The 20 CSV rows sum approximately to:

| Category | Sum | Share of per-row wall sum |
|---|---:|---:|
| `collection_seconds` | `373.347 s` | `51.81%` |
| `update_seconds` | `332.808 s` | `46.19%` |
| `checkpoint_seconds` | `11.211 s` | `1.56%` |
| Residual / orchestration | `3.176 s` | `0.44%` |
| CSV `wall_seconds` total | `720.541 s` | `100%` |

`FINAL_REPORT.md` reports `720.9 s` and `RECEIPT.json` reports `720.948 s`. **Do not silently equate these totals.** The P4.5 report must document the accounting boundary (per-iteration sums versus outer-run stopwatch, startup/export/teardown) and reconcile the discrepancy numerically. P4 `0.4439/s` uses the full-run outer denominator.

These categories are **parent wall intervals**, not GPU kernel time. `update_seconds` contains frozen critic work, GAE, CPU data preparation and PPO compute; `collection_seconds` contains server setup/teardown, environment work, inference, transfer and waits. They are not fine-grained attributions.

### 1.3 Useful speedup ceilings, not forecasts

For the measured sequential topology, assuming all other stages unchanged, completely eliminating the Collector wall interval yields an ideal upper bound of approximately `720.541/(720.541-373.347) = 2.08x`; eliminating the update interval instead yields `~1.86x`. Checkpoint elimination alone is `~1.016x`. These are **Amdahl counterfactuals**, not achievable speedups. They demonstrate why both halves must ultimately be improved. When stages overlap in future P6, do not reuse sequential Amdahl arithmetic unchanged.

## 2. Questions this diagnosis MUST answer

### Collector / actor

- What proportion of the `~373 s` comes from (a) service startup and shutdown, (b) snapshot load, (c) native game advance, (d) native observation decode and ragged transport, (e) ENCODE submission and results, (f) each prefix-level candidate request, tensor preparation, transfer, service queue, GPU inference and `result()` wait, (g) reset and transition assembly, (h) metrics/filesystem writes?
- How many `ENCODE` calls and prefix `STEP` calls happen per seat-decision, per match and per policy iteration? What is the forward-batch-size distribution, not just the maximum?
- Are there serial barriers when some seats have different numbers of prefix branches? What fraction of slots and rows are idle at each level?
- What are payload bytes **actually** transferred per request, which fixed/ragged leaves cause padding, and how many copies are taken per request? Distinguish nominal allocated tensor bytes from actual transport traffic.
- Does `PA4_P3_STATS_PATH` cause a file write per server forward in the P4 run, how many such writes occur, and what is their measured time? Do not disable the metric path as an “optimization” during this diagnosis.
- How much of `future.result` wait is real GPU kernel time versus queue dwell, Python prep, IPC serialization, scheduling or synchronization?

### Learner

- Of `~333 s`, what fraction belongs to repeated frozen critic model construction, state loading, per-row V_old `encode` and `.item()` synchronization, GAE/cohort formation, observation `build_observation`, Ragged `_stack_observations` NumPy conversion, trainable encoder, per-row `evaluate_encoded`, autograd/backward, grad clip, Adam and optimizer-state reload?
- How many model `encode` calls, conditional GRU scorer forwards, CUDA launches, `.item`/`.cpu` syncs, host->device and device->host transfers happen per accepted match and per optimizer step?
- Is the GPU mostly waiting on Python, actually busy with kernels, bandwidth-limited, or memory constrained? **GPU peak allocation of 574.6 MiB is not a GPU utilization metric.**
- Is `microbatch_rows=8` causing small CUDA kernels and host dispatch overhead? Record effective actual batch sizes and occupancy indicators. **Do not change microbatch size in P4.5.**

### Full stack / architecture

- Is the validated P4 throughput dominated by CPU work, transport latency, GPU kernel compute, snapshot/process lifecycle, or some interaction? Rank by absolute recoverable wall time with uncertainty.
- Would a much faster GPU materially help **without** modifying the present code? Give conditional bounds based on measured GPU-critical-path share, not generic FLOPS estimates.
- Which of three later architectures has the strongest evidence: (A) in-process policy inference preserving current TorchRL-compatible learner, (B) optimized PA3-style actor adapter preserving PA4 trajectories, or (C) current TorchRL InferenceServer tuned and kept persistent? Specify what follow-up A/B would falsify each claim.
- Can the next C1 compatibility smoke proceed on the untouched P4 architecture? A diagnosis should not itself block compatibility testing unless it uncovers correctness regressions.

## 3. Measurement vocabulary, accounting and invariant definitions

### 3.1 Top-level mutually exclusive wall-clock stages

Measure **one run** with `time.perf_counter_ns()` or `time.monotonic_ns()` in the driver; mark explicit boundaries. The reported total must be decomposed into **non-overlapping** parent-level intervals:

`run_initialization`, `iteration_setup`, `collection`, `post_collection_validation_and_assembly` (specify if folded into `update` historical timing), `update`, `checkpoint_and_publish`, `report_and_teardown`, `unattributed`.

The legacy 4-stage P4 CSV (`collection`, `update`, `checkpoint`, `wall`) must be reproduced unmodified. Fine-grained instrumentation is **nested** within those stages. Provide both inclusive and exclusive timing, with an explicit parent-child tree. Do **not** sum nested child timers with their parents or sum overlapping process work as if it were sequential.

For asynchronous child processes, use tagged per-process monotonic times plus request/response correlation IDs and, where available, server enqueue/dequeue timestamps. Parent wall wait includes subprocess work; service CPU/GPU time is a **causal sub-attribution**, not additional wall time.

For reference, metric formulas:

- `accepted_training_matches_per_sec = accepted unique natural matches / outer end-to-end wall seconds`.
- `actor_only_natural_matches_per_sec = completed non-truncated matches / actor-only wall seconds`.
- `policy_rows_per_sec = accepted actor-eligible rows / same wall denominator`.
- `learner_row_passes_per_sec = rows actually evaluated in optimization across epochs / wall seconds`.
- `effective_cuda_batch_size` must be measured from real forwards; request batch size is not equivalent to GPU batch size when grouping or serial fallbacks occur.
- `launch_count`, `cuda_kernel_sum_ms`, `cuda_critical_path_ms`, `cpu_self_time_ms`, `cpu_wait_ms`, `H2D_bytes`, `D2H_bytes`, `synchronization_count` must distinguish direct measurement, aggregation and estimation.

### 3.2 Synchronization discipline

PyTorch CUDA launches are asynchronous. Un-synchronized Python `perf_counter` intervals around CUDA launches typically measure enqueue time rather than kernel completion. Use `torch.profiler` **CPU + CUDA** activity in a bounded window, or CUDA events on the same stream with sync once outside the profiled span; avoid `torch.cuda.synchronize()` at every row/request. If explicit synchronization is necessary to isolate a GPU phase, do it **only in an isolated diagnostic run** and label its impact. Do not turn on `CUDA_LAUNCH_BLOCKING=1` for throughput measurements.

Correctly distinguish `CPU self time`, CUDA kernel duration sums, overlapping kernel work and wall critical path. CUDA kernel time sums can overlap and **must not** be treated as GPU wall busy time. If CUPTI is unsupported or the trace lacks GPU data, mark GPU claims `UNVERIFIED` rather than inferring from CPU wrappers.

### 3.3 Instrumentation overhead and negative controls

Instrumented runs are used for attribution, **never** to claim production throughput. Evaluate instrumented versus matched uninstrumented runs. Report profiler slowdown ratio and counter overhead independently. A trace is acceptable even if overhead exceeds 10% provided the slowdown is disclosed and all throughput conclusions are based on uninstrumented evidence; if instrumentation perturbs scheduling enough to invalidate a causal attribution, use lower-impact probes or mark it unverified.

Required falsifiability tests:

1. Turning profiling **off** produces exactly the original code path and no event files or tracer operations; checkpoint/policy records and 272 inherited regressions remain valid.
2. Deliberately inserting a controlled CPU sleep into **an isolated diagnostic fixture only**, not real production code, makes the matching labeled interval increase and its parent wall reconcile. Remove fixture after test; never report fixture speed as production.
3. A deliberately nested timing fixture proves that inclusive/exclusive accounting does not double count.
4. Request correlation tests show per-stage counters never silently lose/duplicate a request even when a policy request errors.
5. Deliberately invoking a missing CUDA activity capability causes the GPU-proof portion to return `UNVERIFIED`, not `PASS` with zeroes.

## 4. Profiling execution plan

### 4.1 D0 — Local source and environment freeze

- Capture `git status --porcelain`, HEAD, branch, `git diff`, PA3 pinned SHA/dirtiness, `uv lock --check` and lock SHA256.
- Check `nvidia-smi`, `torch.cuda.get_device_properties`, torch/TorchRL/TensorDict versions, CPU model/logical cores, system memory, `torch.get_num_threads`, thread env vars, WSL kernel and CUDA driver.
- Record GPU clocks/power/pstate/temperature/SM utilization samples if available at low frequency, clearly noting monitor granularity; do not reconfigure GPU power limits, BIOS, drivers or system packages.
- Read and stat the P4 historical baseline, CSV schema and SHA256 of original artifacts. **Never edit them.**
- Create a diagnostic branch and a new `docs/phases/P4.5/` tree and `benchmarks/p45/` scripts. Do not use an existing training checkpoint directory for any new run.

### 4.2 D1 — Reconcile the historical baseline

Programmatically parse original `FULL_STACK_BASELINE.csv` and `evidence/full_run.json`. Recompute row counts, 320/320 accepted, step counts, per-stage sums, medians, spread and percent shares; show exact outer versus inner stopwatch difference. Verify attribution code definitions by source reading, including the precise `update_seconds` boundary in `train_sync.py`.

Generate `historical_reconstruction.json` and `historical_reconstruction.md`, each with source file hash, parser SHA and machine-checked arithmetic. If the source artifacts disagree by more than rounding, report discrepancy and its likely boundary; **do not change original evidence or fabricate reconciliation**.

### 4.3 D2 — Low-overhead wall stage instrumentation

Add a **default-off** `PA4_P45_PROFILE=1` or explicit diagnostic-wrapper mechanism. Reuse existing timers wherever they are sufficient; avoid duplicate clocks. Emit buffered JSONL/CSV **once per parent iteration or on shutdown**, not once per request, unless measuring per-request logging overhead in a separate controlled trace.

Record at minimum:

**Collector stages:** server creation and start; server policy snapshot load; initial native reset; `request_info`; `_views_from_players` and ragged validation; encoder request prep; `ENCODE` `submit`, ready/wait, response; each prefix level's candidate construction, request prep, `submit`, wait, response and cache operations; full packet construction; `env.step`; transition clone/digest/write to lossless sink; `account_step`; resets; server shutdown; legacy per-forward stats write. Ensure one named catch-all `collector_other`.

**Learner stages:** checkpoint/snapshot read and model init/load; cohort validation and row building; frozen policy creation/load; **frozen V_old observation build, encoder, value and `.item()`**; GAE and normalization; microbatch row selection; stored row build; Ragged stacking and NumPy path; batched trainable encoder; conditional scorer per row; PPO loss terms; `.backward`; clip grad; optimizer step; gradient diagnostics; checkpoint serialization, SHA, `fsync`, atomic pointer publication; other.

Collect operation **counts**, row counts, shape histograms, bytes, identity and policy-version metadata separate from durations. No secret keys, private file paths beyond local internal report, or full Pokémon battle replay payloads in public diagnostic artifacts.

### 4.4 D3 — Uninstrumented repeated end-to-end baseline

The **fixed** small-profile configuration must be:

```yaml
num_envs: 4
matches_per_iteration: 16
iterations: 4
microbatch_rows: 8
effective_minibatch_rows: 64
ppo_epochs: 2
precision: eager_fp32
seed: 20261006
policy_snapshot: actual versioned weights, updated after each iteration
```

Use a new run directory per trial. Prefer three matched uninstrumented trials (at least two if runtime or instability prevents three); capture per-iteration and aggregate metrics. Make an explicit cold start/warm split. Preserve all real PPO updates and checkpoints. Do not disable required checkpoint/replay validation in benchmark mode. Historical 20-iteration rate is an anchor, not assumed identical to 4-iteration runs.

Report median and range (and distribution when feasible) for accepted matches/s, collection/update/checkpoint intervals, accepted rows, epoch early stops, actual optimizer step counts, GPU memory and model-forward counts. If matching seeds still yield differing trajectory composition, report match length, seat-decisions, branch counts and policy divergence rather than pretending identical CPU work.

### 4.5 D4 — Short bounded deep CPU + CUDA traces

Run a **separate, instrumented** short window after warmup, ideally one live actor iteration and one real learner iteration. Profile code inside spawned inference-server process separately from driver/learner: a parent-only `torch.profiler` trace does **not** observe child CUDA kernels or CPU operations.

- Use `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], ...)` with bounded activation and export of raw Chrome/Perfetto trace; check installed PyTorch 2.9.1 API and CUDA profiler capability before use.
- Annotate with `torch.profiler.record_function("pa4.p45.<stage>")`. Include distinct parent and child PID/worker names, relevant request identity and policy version in metadata (not every record if this explodes traces).
- First gather low-impact traces (`record_shapes=False`, `with_stack=False`, `profile_memory=False`). For a **smaller bounded span only**, enable stacks/shapes/memory where needed; do not trace all 20 iterations with stacks on.
- Export an operator table sorted by both CPU self time and CUDA self time. Capture kernel launches,Memcpy activity, CPU and CUDA idle gaps if trace supports it, per-stage number of model forwards and measured batch sizes.
- Identify CUDA synchronization sites: `.item()`, `.cpu().numpy()`, synchronous transfers, `torch.equal()` on CUDA, scalar host conversion, custom error checks. Count only calls that **actually synchronize** by evidence, not all calls with `.item` substring.
- Use CUDA events in an isolated corroboration fixture where cross-process profiler instrumentation is unavailable; do not insert synchronizations into production baseline.
- Trace exports must be size bounded and separated from source control if large (`.gitignore` or excluded artifact pointer and local hashes). Fail with a clear `GPU_TRACE_UNAVAILABLE` limitation if GPU events cannot be collected.

### 4.6 D5 — Honest isolation experiments (no optimizations)

Three diagnostic-only, *faithful* paths:

1. **Actor-only, same config**: Run real Rust battles with same 4 environments, actual Ragged v2 observations, frozen actual policy snapshot and identical seat policy, terminating after comparable 16/64 natural matches. Use the existing producer without disabling legitimate observations or needed inference; no surrogate random actor. Label the difference from full-stack actor startup and counters.
2. **Learner-only, frozen accepted cohorts**: Export or record **real** accepted P4 cohorts from an isolated test run; replay valid stored actor+value rows against an identical policy snapshot and full PPO-2 gradients in a throwaway workspace. Retain the same microbatch, early-stop and Adam policy for the measured window. This run is **not a new on-policy training campaign** and must not be merged into any production ledger.
3. **Per-stage isolated compute/transfer:** Time an identical fixed set of real observations through native parsing, ragged stacking, V_old and trainable scorer using separate diagnostic runner functions. Label it microbenchmark, not full-stack matches/s. Do not use synthetic zero-ragged samples to represent production.

Prefer observational instrumentation over intervention. If you must make a switchable diagnostic intervention (e.g., stats-write disabled) to establish cause, run it only on throwaway, frozen actor-only artifacts, label it **counterfactual ablation rather than production optimization**, compare matched runs and restore default. No changed path becomes the accepted P4 default.

### 4.7 D6 — Hypothesis ranking, architectural decision and audit

Produce a **top-ten** bottleneck table (top five deeply analyzed), with for each:

- Exact file/function + source SHA, call/launch counts and input shapes.
- Absolute attributable baseline wall milliseconds or seconds; percentages of the correct, non-overlapping parent denominator.
- Separately measured CPU self time, CUDA kernel time, queue/wait time, transfer/copy time and any overlap (or mark `not measured`).
- Confidence (`HIGH`, `MEDIUM`, `LOW`) with direct artifacts, variance and disconfirming data.
- Physical/architectural cause (e.g., repeated per-row GPU sync, IPC, tiny CUDA kernel batches, checkpoint lifecycle), and a future **not implemented** fix candidate.
- Correct sequential Amdahl bound and separate cost/risk/behavior-contract impact. Never sum the maximum possible speedups of dependent fixes.

Decision among post-diagnostic options: retain TorchRL with persistent InferenceServer, replace inference transport with in-process calls, adapt PA3-style optimized actor, reorganize per-row Learner evaluation, or a bounded Rust-owned actor protocol. State which option(s) need a small A/B experiment and which lack evidence. Recommend proceed/not proceed for C1 and P5 **based on correctness and integration readiness**, not production throughput alone.

## 5. Deliverable specification

Create under `docs/phases/P4.5/`:

| Deliverable | Contents |
|---|---|
| `DESIGN.md` | Verbatim copy of this authored design, ID/hash recorded |
| `FINAL_REPORT.md` | One-page executive findings followed by full substantiation, uncertainties and decisions |
| `RECEIPT.json` | Test code SHA, canonical baseline, GPU/system versions, hashes, gates, counts, verdict, limitations |
| `HISTORICAL_RECONCILIATION.md` | 20-iteration stage arithmetic and evidence boundary explanation |
| `BOTTLENECK_RANKING.csv` | Function/stage, inclusive/exclusive time, call counts, kernel/CPU/wait, confidence, evidence |
| `PERF_RUNS.csv` | Trial configuration, git SHA, wall, accepted matches, stages, validation, duration |
| `ENVIRONMENT.json` | OS/driver/GPU/CPU/threads, package versions, lock/PA3 hashes |
| `TIMING_DICTIONARY.md` | Definitions and parent-child timing map, overhead assumptions |
| `ARCHITECTURE_DECISION.md` | Falsifiable A/B proposals and recommendations for C1/P5/P6/P7 |
| `evidence/` | Machine-readable measurements, unit negative controls, profiler tables and trace hashes; large trace files need only manifest/path, not Git payload |

Create `benchmarks/p45/` with runnable scripts and add strictly isolated profiling tests under `tests/diagnostics/` or another logical test location. Do not invent command flags; inspect the actual test runner, prefer `uv run --frozen` and `python -m unittest` conventions already used in PA4.

### Receipt skeleton (illustrative keys, not invented measured values)

```json
{
  "phase": "P4.5",
  "design_id": "PA4-P45-DIAG-DESIGN-v1.0",
  "mode": "diagnostic_only",
  "baseline_git_sha": null,
  "diagnostic_git_sha": null,
  "pa3_reference_sha": "65bc980abd8e0bbec14158b9ddc2884bb1726a4e",
  "lock_sha256": "8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e",
  "gates": {},
  "number_of_uninstrumented_trials": 0,
  "baseline_accepted_matches_per_sec": null,
  "timing_coverage_fraction": null,
  "unattributed_wall_fraction": null,
  "instrumentation_overhead_ratio": null,
  "cuda_trace_available": null,
  "policy_parity_passed": null,
  "optimization_implemented": false,
  "verdict": "NOT_RUN"
}
```

## 6. Explicit P4.5 diagnostic acceptance gates

- **D45-0 Source freeze: PASS** only if P4 evidence is preserved, HEAD and clean/dirty state recorded, PA3 immutable, lock unchanged, correct environment, instrumenter isolated.
- **D45-1 Accounting: PASS** only if 20-iteration historical stage totals are recomputed by code and historical stopwatch discrepancy is explicitly reconciled; no silent rewrites.
- **D45-2 Instrumentation: PASS** only if all major categories have explicit parent/exclusive timers, child-process inference is instrumented or its absence is documented, nested double-count prevention is tested, off-mode is identical, no count loss.
- **D45-3 Baseline: PASS** only if at least two uninstrumented matched full-stack runs use **real** Rust + PPO, with 0 dropped/duplicates, real changing policy, 0 operational errors and numeric correctness gates unchanged, per-stage logs and variation documented.
- **D45-4 CPU/GPU evidence: PASS** only if both parent and inference child CPU and GPU activity are observed or a precise `PARTIAL` status is stated with no unsupported utilization/compute claims. Reproducible operator and critical-path evidence must support at least the top three claims.
- **D45-5 Root-cause ranking: PASS** only if >=90% of parent wall time is attributed to mutually exclusive top-level stages, >=80% of each major parent interval is classified into exclusive subcategories or residual explicitly named, top five bottlenecks ranked with direct measurable evidence and credible uncertainty. If measurements cannot meet coverage, verdict is `PARTIAL`, not a manufactured PASS.
- **D45-6 Invariance, audit and next-step decision: PASS** only if 272 inherited tests (or a higher naturally increased count) and P0 smoke pass, P4 behavior contracts remain intact, a fresh-context read-only auditor actually accesses the checked-out diagnostic source, findings are addressed or clearly marked blocking, and resulting docs contain actionable and falsifiable next A/B experiments. Do not require another 20-iteration full run solely to audit diagnostic instrumentation.

Overall `P4.5 PASS` only when all seven gates truly PASS; otherwise `PARTIAL` or `BLOCKED` with evidence. Passing P4.5 certifies **diagnostic quality**, never faster training.

## 7. Safety, bounded runtime and stop conditions

Recommended experimental budget: three uninstrumented 4-iteration trials (may be two with explicit explanation), one short 4-iteration light-instrumented trial, one bounded parent+child deep trace, one actor-only real-policy collection, and one real-cohort learner-only rehearsal. Use the minimum trace to identify a dominant root cause; avoid unbounded recording, massive reruns or a full 20-iteration replay when the original P4 run has already established correctness.

Stop immediately, without changing correctness criteria, if:

- Any policy/observation contract, fingerprint, logprob, loss, optimizer behavior, real completion or lossless accounting changes due to instrumentation.
- Native PA3 reference or `uv.lock` changes.
- Evidence of dropped/duplicate matches, inconsistent policy version, non-finite gradients or invalid checkpoints.
- Trace collection threatens out-of-memory, system stability, disk capacity or correctness; disable only the **diagnostic profiler**, keep P4 code unmodified, and mark limitation.
- The work would provision a cloud instance, require credentials, change global OpenCode configuration, rewrite PPO/Actor architecture, or push/merge without authorization.

Ordinary engineering blockers should be diagnosed and corrected within the instrumentation scope without waiting for routine approvals. A failed probe is **not** a reason to make training behavior faster/different; report it honestly.

## 8. Source references

PA4 source-of-truth at reviewed GitHub `main`:

- https://github.com/mwl313/PA4/blob/main/docs/PROJECT_STATUS.md
- https://github.com/mwl313/PA4/blob/main/docs/master/PA4_MASTER_ARCHITECTURE.md
- https://github.com/mwl313/PA4/blob/main/docs/phases/P4/FINAL_REPORT.md
- https://github.com/mwl313/PA4/blob/main/docs/phases/P4/RECEIPT.json
- https://github.com/mwl313/PA4/blob/main/docs/phases/P4/FULL_STACK_BASELINE.csv
- https://github.com/mwl313/PA4/blob/main/pa4/collector/run_batched.py
- https://github.com/mwl313/PA4/blob/main/pa4/actions/batched_factory_instrumented.py
- https://github.com/mwl313/PA4/blob/main/pa4/learner/train_sync.py
- https://github.com/mwl313/PA4/blob/main/pa4/learner/learner_eval.py

Profiler behavior reference:

- PyTorch `torch.profiler`: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch CUDA asynchrony and event timing: https://docs.pytorch.org/docs/stable/notes/cuda.html

*End of design.*
