# P4.5 — Timing Dictionary

**Design:** `PA4-P45-DIAG-DESIGN-v1.0` §3

Definitions for every measurement term used in this phase, plus the
parent-child map. Where a term is estimated rather than measured, it says so.

---

## 1. Top-level, mutually exclusive parent stages

These are the only intervals that may be summed. They do not overlap.

| Stage | Boundary in code | What it contains |
|---|---|---|
| `run_initialization` | `full_run.py` entry → before `SynchronousTrainer` construction | process start, imports, snapshot seed model |
| `iteration_setup` | start of `run_iteration` → before `run_batched` | theta_k load, snapshot metadata read, record construction |
| `collection` | `run_batched` call → return | see §2 |
| `post_collection_validation_and_assembly` | folded into `update` (P4 behaviour) | cohort build, provenance validation |
| `update` | after cohort build → after PPO updater returns | see §3 |
| `checkpoint_and_publish` | before `save_checkpoint` → after snapshot publish | see §4 |
| `report_and_teardown` | loop end → process exit | summary JSON, CSV write |
| `unattributed` | anything not covered above | measured as wall minus the rest |

**Historical four-stage P4 CSV is reproduced unmodified**: `collection`,
`update`, `checkpoint`, `wall`. P4.5 adds finer stages *inside* those, never
instead of them.

## 2. Collection child stages

`run_batched` already records these (`STAGES` in `pa4/collector/run_batched.py`):

| Stage | Definition |
|---|---|
| `request_info` | querying the native engine for request kinds and branch counts |
| `views` | extracting the nine fixed leaves + ragged transport rows per requesting seat |
| `encode` | ENCODE round trip for every requesting seat: payload assembly, `submit`, `result()` wait, layout/roster extraction |
| `step_levels` | all prefix STEP levels: candidate construction, `build_branch_table`, `submit`, `result()` wait, cache update |
| `packet` | assembling the six-key native action packet |
| `env_step` | the Rust `env.step` call itself |
| `accounting` | transition materialisation, digests, lossless sink put |
| `reset` | native resets for finished rows |

**Not covered by these timers** (documented as a limitation, not estimated):
inference-service startup, snapshot load inside the child, per-request
submit-vs-queue-vs-wait split, server shutdown, per-forward stats write.
D2c and D2d isolate several of these separately.

## 3. Learner child stages (added by P4.5)

| Stage | Definition |
|---|---|
| `update.cohort_build` | `build_cohort`: grouping, validation, seat trajectory assembly |
| `update.frozen_model_create` | constructing the frozen critic module |
| `update.frozen_state_load` | `load_state_dict` + `eval()` |
| `update.vold_obs` | `build_observation` per value row |
| `update.vold_encode` | `frozen.encode` per value row |
| `update.vold_value` | `frozen.value` per value row |
| `update.vold_item` | `.item()` host synchronisation per value row |
| `update.gae` | per-seat GAE over all completed chains |
| `update.normalize` | advantage normalisation |
| `update.learner_create` | trainable model construction + committed state restore |
| `update.learner_clone_before` | parameter snapshot taken for the change check |
| `update.ppo_loop` | the full `PpoUpdater.run`: per-microbatch row rebuild, ragged stacking, batched encoder, per-row `evaluate_encoded`, backward, clip, Adam, for all epochs |
| `update.checkpoint` | `save_checkpoint` |
| `collect.run_batched` | the whole collection call (parent of §2) |

## 4. Accounting rules

1. **Exclusive vs inclusive.** `inclusive` is the span's own elapsed time.
   `exclusive` subtracts time spent in *direct* children. Computed from a
   thread-local stack, so `parent_exclusive + Σ children_exclusive ==
   parent_inclusive` holds by construction. Verified by
   `tests/diagnostics/test_profiler_controls.py`.
2. **Never sum across processes.** Parent wall time already *contains* child
   process work. Child CPU/GPU times are causal sub-attribution, not addable
   wall time.
3. **Never sum nested stages with their parents.** `BOTTLENECK_RANKING.csv`
   uses exclusive time only.
4. **CUDA asynchrony.** A `perf_counter` interval around a CUDA launch measures
   enqueue time, not kernel completion. Device work is measured with CUDA events
   (`d2d`, `d2e`) or reported as unavailable, never inferred from CPU wrappers.
5. **Kernel time is not GPU busy time.** Summed kernel durations may overlap;
   they are not wall occupancy.

## 5. Metric definitions

| Metric | Formula | Source |
|---|---|---|
| `accepted_training_matches_per_sec` | accepted unique natural matches / outer wall seconds | `d3_baseline.json` |
| `actor_only_natural_matches_per_sec` | completed non-truncated matches / actor-only wall | `d2b_collect.json` |
| `policy_rows_per_sec` | accepted actor-eligible rows / wall | P4 CSV |
| `learner_row_passes_per_sec` | rows evaluated across epochs / wall | P4 CSV |
| `effective_cuda_batch_size` | measured from real forwards | `d2e_scale.json` |
| `instrumentation_overhead_ratio` | instrumented rate ÷ uninstrumented rate | `d2_run` vs `d3_baseline` |

## 6. Instrumentation overhead

The profiler is **off unless `PA4_P45_PROFILE=1`**. Off-mode is proven inert by
four tests (records nothing, writes no file, returns identical values, gate
defaults to disabled). Instrumented throughput is reported only as a ratio
against the uninstrumented median, never as a production rate.

## 7. Confidence vocabulary

| Level | Meaning |
|---|---|
| `HIGH` | direct measurement, reproduced across runs, no alternative explanation |
| `MEDIUM` | direct measurement, single run or with unmeasured co-factor |
| `LOW` | inferred from correlated stages; mechanism plausible but not isolated |
| `not measured` | explicitly not attributed; never replaced with an estimate |
