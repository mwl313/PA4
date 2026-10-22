# P4 — Trajectory Contract v1

**Phase:** P4
**Design:** `PA4-P4-DESIGN-v1.0` §3
**Status:** implemented in P4-A

This document defines the producer → consumer boundary between P3's actor and
P4's learner. It exists because P3's `BoundedTransitionStore` **evicts** on
overflow and must never silently feed PPO training.

---

## 1. Two stores, deliberately different

| | `BoundedTransitionStore` (P3) | `LosslessTrainingSink` (P4) |
|---|---|---|
| Purpose | P3 benchmark throughput | PPO training input |
| Overflow | evicts oldest, `dropped += 1` | **blocks** (backpressure) or fails the iteration |
| Default | yes (P3 path) | opt-in only |
| `dropped == 0` guaranteed | **no** | **yes, by construction** |

P3's ring keeps its exact benchmark semantics so P3's receipts stay
reproducible. P4 training mode never uses it. `run_batched` accepts an injected
store; P4 injects the lossless sink.

---

## 2. Accounting counters (per iteration)

Every iteration records these separately. A successful iteration requires
`dropped == 0` and `produced == acknowledged`.

| Counter | Meaning |
|---|---|
| `produced` | accepted transitions emitted by the actor |
| `delivered` | handed to the sink |
| `acknowledged` | durably retained by the sink |
| `dropped` | evicted or lost (**must be 0**) |
| `duplicates` | same identity offered twice |
| `revisions` | rows superseded by a revised native step |
| `incomplete` | episodes not naturally terminated |
| `accepted_for_training` | distinct natural matches admitted |

---

## 3. Episode key — not `match_id`

Early transitions have no `match_id`. Grouping by it (or by `None`) would merge
unrelated episodes.

**Canonical episode key:**

```
(run_id, worker_id, native_slot, native_generation, episode_id)
```

The terminal `match_id` is attached to the whole accumulated episode by matching
that five-field key. Two successive episodes in the same native slot differ by
`generation` and/or `episode_id` and therefore **never share a GAE chain**.

---

## 4. Seat trajectories

For each episode, exactly two ordered sequences:

```
(episode_key, seat=0)   and   (episode_key, seat=1)
```

Rules:

1. Only accepted `Preview` / `Normal` / `Replacement` requests create rows.
   `Wait` / `Finished` produce **no row at all** — no fabricated timestep.
2. Ordering is by `step_index`, then `request_id`; identity must be strictly
   monotone with no duplicates.
3. A seat's next learning row is its **next real policy decision**, which may be
   several native steps later.
4. `actor_active = True` iff at least one branch has `K >= 2`. Deterministic
   `K == 1` rows still contribute to the critic.
5. The terminal result (`+1/-1/0`) is assigned to the **last accepted decision of
   each seat**, even if that seat did not act in the final engine step
   (the `(Replacement, Wait)` case).
6. `done = True` is set on that final decision row for that seat.
7. GAE never crosses seats.

---

## 5. Rejected cohorts (fail closed)

The whole cohort is rejected — never partially trained on — if any row shows:

- missing first decision, or a terminal with no prior seat rows
- swapped seat, duplicate request identity
- cross-generation or cross-episode contamination
- `observation_representation != "pa4.observation.ragged.v2"`
- policy version / weight fingerprint mismatch
- a revised (non-accepted) action
- malformed mask, or a selected index outside the stored candidate table
- an incomplete / truncated / errored episode counted as a natural completion

---

## 6. Observation fidelity

Each seat row retains, per `PA4ActorTransitionV1` plus the P3.5 additions:

- the nine fixed pre-action leaves (by value, CPU)
- **all five ragged families** (`observation_ragged`)
- `observation_representation` (`pa4.observation.ragged.v2`)
- ordered tuple6 candidates per branch, legal mask, token refs, `branch_k`,
  `selected_index`
- FP32 branch and joint logprobs
- `request_kind`, `request_id`, `seat`, `step_index`
- policy version, weight fingerprint, precision profile

**Legacy `pa4.observation.fixed9.v1` records are refused at learner ingestion**
via `ragged_transport.assert_representation`, not merely in the collector.

---

## 7. Incomplete episodes

P4's initial mode is completed-only. At a policy rollover, unfinished episodes
are handled **explicitly**:

- drain them under the same frozen policy, or
- discard them and count them in `incomplete`

They are never relabelled onto a new policy version, never marked terminal, and
never carried as on-policy rows for `theta_{k+1}`.
