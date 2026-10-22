# P3 — PA4ActorTransitionV1 Specification

Document ID: `PA4-P3-TRANSITION-V1`
Design: `PA4-P3-DESIGN-v1.0` §3.2

P3 stores bounded, in-memory transition records so P4 can replay behaviour
exactly. No database, no long-lived archive, no PPO advantage computation.

Implementation: `pa4/collector/transitions.py` (`BoundedTransitionStore`,
`seat_record`, `transition`, `accepted_records`), emitted by
`pa4/collector/run_batched.py`. The store has a fixed capacity
(`BatchedConfig.transition_capacity`, default 256); when full it drops the
oldest record and increments `transitions_dropped`, which the report exposes
alongside `transitions_stored` and `transitions_retained`. Records are emitted
only for decisions used in an accepted native step: rows whose step was revised
are excluded and counted separately. Ownership is copy-on-store — every tensor
is detached and cloned to CPU, so a later env/memory reuse cannot mutate a
retained record.

---

## 1. What is recorded per transition

| Field | Meaning |
|---|---|
| `run_id` | immutable run identity |
| `worker_id` / `driver` | which actor context produced it |
| `native_slot`, `native_generation` | native handle lifetime (never crosses a process) |
| `episode_id`, `step_index` | strict per-slot accounting |
| `request_ids[2]` | per-seat request identity at decision time |
| `policy_version` | frozen policy version (never changes during P3) |
| `policy_weight_fingerprint` | sha256 over the exact weights served |
| `observation_contract_hash` | pinned PA3 observation contract |
| `real_p2_action_packet` | the exact six-key packet submitted to `env.step` |
| `reward[2]`, `done`, `terminated`, `truncated` | native outcome, unchanged |
| `match_completed`, `winner`, `request_revised` | exactly-once accounting |
| `seat_records[]` | one per seat that actually decided |

## 2. Per-seat behaviour record

Only seats whose request kind is Preview/Normal/Replacement produce a record.
`Wait`/`Finished` produce **nothing** — no fabricated pass, no empty row.

| Field | Meaning |
|---|---|
| `seat` | 0 or 1 |
| `request_id` | the seat's own request identity |
| `observation` | **pre-action** nine-leaf view (by value, not a digest) |
| `actor_active` | true iff any branch had `branch_k >= 2` |
| `joint_logprob_fp32` | sum of per-branch FP32 logprobs |
| `branches[]` | ordered per-level evidence |

Each `branches[j]`:

| Field | Meaning |
|---|---|
| `level` | 0..3 |
| `candidates` | **ordered** canonical six-field tuples, exactly as Rust returned them |
| `selected_index` | index into that ordered list |
| `branch_k` | real legal candidate count |
| `branch_logprob_fp32` | FP32 logprob of the chosen index |

All of the first table's fields are present in every stored record; the
`tests/collector/test_matches.py::TransitionRetentionTest` asserts the field
set, the nine observation leaves (CPU tensors), candidate membership of each
selected index, the six-key packet, the sha256 weight fingerprint, the pinned
observation contract hash, and that a stored `match_completed` record carries a
match id present in the ledger.

## 3. Replay requirements satisfied

P4 can recompute a stored behaviour logprob because the record retains:

1. the **actual pre-action observation** (not only a digest),
2. the **ordered candidate table** per level,
3. the **selected index** (never resampled),
4. the **branch_k** used for masking,
5. the **policy version and weight fingerprint**.

The P1 `replay()` path already proves this works: it refuses on weight, version or
observation mismatch and reproduces branch and joint logprobs with delta `0.0` on
the tested fixtures (P1 receipt).

**Not claimed:** independent PA3 golden numerical parity. Replay is
self-consistency against stored evidence, and the P2 oracle shares the observation
decoder, so decoder bugs would be invisible to both.

## 4. Match identity (exactly-once)

```
match_id = (run_id, worker_id, native_slot, native_generation, episode_id)
```

A match is committed **only** when Rust reports `terminated=True` AND
`match_completed=True` for that episode, and only once per identity.

| Event | Counted as a match? |
|---|---|
| natural `terminated=True` | **yes, once** |
| native truncation | no (tracked in `truncations`) |
| `accepted=False` revision | no |
| operational error | no — raises |
| dropped/abandoned episode | no |
| the second seat of the same battle | no — same `match_id` |

Verified by `test_matches.py::test_no_duplicate_completions` and
`test_match_ids_are_fully_qualified`.

## 5. Storage bounds

Transitions are emitted through a bounded queue; the collector keeps only
per-request inference cache state (evicted on RESET and on every rejection path).
No unbounded retention, no replay database, no serialized corpus.
