# P2 — Episode Trace Report

Document ID: `PA4-P2-EPISODE-TRACE`
Design: `PA4-P2-DESIGN-v1.0`
Command: `uv run --frozen python -m unittest discover -s tests/environment -t .`

All numbers below are measured on the pinned stack. Nothing is projected.

---

## 1. Natural game completion (G2-B)

| Seed | Steps | Terminated | Truncated | Winner | Reward (seat0, seat1) | Match completions |
|---|---|---|---|---|---|---|
| 1 | 18 | yes | no | 0 | `+1.0, -1.0` | 1 |
| 2 | 22 | yes | no | 0 | `+1.0, -1.0` | 1 |
| 3 | 17 | yes | no | 1 | `-1.0, +1.0` | 1 |
| 4 | 17 | yes | no | 1 | `-1.0, +1.0` | 1 |

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
| `(Replacement, Wait)` | **asymmetric**: one seat decides, peer waits |
| `(Wait, Replacement)` | **asymmetric**, mirrored |

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
3. **Spec dtype mismatch**: torchrl `Binary` is `int8`; the env emits `bool`.
   `check_env_specs` failed until boolean leaves used
   `Unbounded(dtype=torch.bool)`. The spec was corrected to match truthful data,
   not the data cast to hide the mismatch.

## 5. What this does not show

- No performance claim. Elapsed times are not benchmarked or optimised in P2.
- No GPU simulation claim; native stepping is CPU.
- Observed max legal candidates remains **18** (inherited fixture bound, not a
  global maximum).
- Coverage is 4 seeded games, not 1,220 teams.
