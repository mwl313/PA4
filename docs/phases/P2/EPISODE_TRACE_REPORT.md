# P2 — Episode Trace Report

Document ID: `PA4-P2-EPISODE-TRACE`
Design: `PA4-P2-DESIGN-v1.0`
Command: `uv run --frozen python -m unittest discover -s tests/environment -t .`

All numbers below are measured on the pinned stack. Nothing is projected.

---

## 1. Natural game completion (G2-B)

| Seed | Steps | Terminated | Truncated | Winner | Reward (seat0, seat1) | Match completions |
|---|---|---|---|---|---|---|
| 1 | 20 | yes | no | 0 | `+1.0, -1.0` | 1 |
| 2 | 11 | yes | no | 1 | `-1.0, +1.0` | 1 |
| 3 | 19 | yes | no | 0 | `+1.0, -1.0` | 1 |
| 4 | 14 | yes | no | 1 | `-1.0, +1.0` | 1 |

This exact table is asserted by
`tests/environment/test_measured_evidence.py::test_documented_trace_matches_receipt`,
so it cannot drift away from the receipt unnoticed. (An earlier revision of this
document published `18/22/17/17` with winners `0,0,1,1`; that was measured before
the harness RNG defect in §4 item 4 was fixed, and is superseded.)

- **Naturally completed: 4 / 4.** All reached Rust `terminated=True`, none via
  truncation or harness timeout.
- **Reward↔winner mapping verified in both directions.** Winner 0 pays
  `(+1,-1)`; winner 1 pays `(-1,+1)`.
- **`duplicate_completion_ids = 0`.** Each episode commits exactly one match;
  `match_completed` and `env.completed_matches` were cross-checked.
- **`operational_errors = 0`.** A native `operational_error` raises instead of
  being recorded.

## 2. Two-seat protocol states actually observed (G2-B)

Request-kind pairs seen in the seeded games:

| Pair | Meaning |
|---|---|
| `(Preview, Preview)` | both seats decide — 4 picks each |
| `(Normal, Normal)` | both seats decide — up to 2 branches each |
| `(Replacement, Replacement)` | both seats decide — forced replacement on both sides |
| `(Replacement, Wait)` | **asymmetric**: one seat decides, peer waits |
| `(Wait, Replacement)` | **asymmetric**, mirrored |

The exact measured set above is asserted by
`test_documented_request_kind_pairs_measured`; the earlier four-pair list omitted
`(Replacement, Replacement)`, which seed 2 actually produces.

- `wait_observed = True` — a `Wait` seat was never marked present and never given
  a fabricated action (asserted in `test_wait_seat_submits_nothing`).
- `replacement_observed = True` — real forced replacement appeared naturally; no
  fixture was forged.
- **`illegal_submissions = 0`.** Every emitted action row was asserted to be a
  member of the engine's candidate list for its prefix.

## 3. Independent-oracle parity (G2-C)

`pa4/env/trace_compare.py` constructs a **second** `NativeEngine` at the same
pinned SHA and replays the exact action tuples. Compared at every boundary:

- both seats' `request_kind` and `branch_count`
- ordered legal candidate lists for each selected prefix
- all nine decoded observation leaves for both seats (including knownness masks)
- native step result `(accepted, terminated, truncated, winner, error, kind_a, kind_b)`
- final winner and termination

Result: **4 seeds × every boundary, zero mismatches**; terminal results agree.

Also verified:

- **Deterministic reset**: identical seed + teams + roles reproduce identical
  requests and identical player views across two independently constructed envs.
- **Partial reset**: resetting slot 0 via the `_reset` mask leaves slot 1's
  `slot`, `generation`, `episode_id` and `step_index` untouched.

## 4. Bugs found by this testing (real, not cosmetic)

1. **Oracle used a re-derived seed** computed *after* reset incremented
   `episode_id`, so it replayed the wrong episode — surfaced as
   `invalid action prefix`. Fixed by recording the exact `used_seed` /
   `used_teams` submitted at reset.
2. **Partial reset used `reset_batch`**, which replaces every native slot and
   invalidates the untouched row's handle (`environment handle`). Fixed by using
   `reset_slots_batch` for only the masked slots.
3. **Spec dtype mismatch**: torchrl `Binary` *defaults* to `int8` while the env
   emits `bool`. `check_env_specs` failed (`fake=torch.int8 and real=torch.bool`)
   until boolean leaves used `Unbounded(dtype=torch.bool)`. The spec was
   corrected to match truthful data, not the data cast to hide the mismatch.
4. **Harness RNG defect (Lead's own measurement code).** The action RNG was
   seeded from the step index only, so all four `distinct_seeds` played the same
   action sequence; the published games were identical and all won by seat 1.
   Seeding now includes `env._seed`, and the four games are genuinely distinct.
5. **A false G2-E claim.** At commit `67f1d59` the "P1 staged policy packet"
   test selected candidates with `torch.randint`; no `ConditionalPolicy` call
   reached `EnvBase.step()`, although the docstring and receipt said it did. A
   read-only review caught it. The test now calls
   `ConditionalPolicy.complete_request` per requesting seat, verifies the record
   by replay, asserts an actor-active row, and a second test drives a full battle
   with P1 policy decisions on both seats.
6. **Post-terminal truncation fabrication.** After a truncated episode, the next
   step re-reported `terminated=True, truncated=False` and lost the winner. A
   concluded row now re-reports the raw flags and winner recorded when the episode
   ended (`test_contract_edges.py::ConcludedRowTest`).

## 5. What this does not show

- No performance claim. Elapsed times are not benchmarked or optimised in P2.
- No GPU simulation claim; native stepping is CPU.
- Observed max legal candidates remains **18** (inherited fixture bound, not a
  global maximum).
- Coverage is 4 seeded games, not 1,220 teams.
