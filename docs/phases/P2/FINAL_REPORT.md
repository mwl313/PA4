# P2 — Final Report

**Phase:** P2 — Rust ↔ TorchRL Environment Adapter
**Design:** `PA4-P2-DESIGN-v1.0`
**PA4 base (P1 HEAD):** `892f136f3704b773257a377c9a41c4fa50c74770`
**PA4 tested code:** `136d333fa089edd12c36af53ae2c5ddc85f664a8`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0/P1)

---

## VERDICT: **PASS** (5/5 gates)

G2-E's two conjuncts are both met. The P1 `ConditionalPolicy` produces the packet
that crosses the real `EnvBase`/native boundary, the P1 regression is green, the
lock and PA3 reference are unchanged, and a read-only `pa4-auditor` ran **from
WSL with actual repository read access** and returned `VERDICT: CLEAN` on the
tested commit. The previous PARTIAL existed only because no auditor session could
read the WSL tree; that access problem is resolved and was demonstrated
(`ACCESS_OK`, literal file read) before the review began.

---

## 1. What was built

| File | Role |
|---|---|
| `pa4/env/native_env.py` | `PA4EnvBase(EnvBase)` — native-backed environment, identity/terminal accounting |
| `pa4/env/trace_compare.py` | Independent `NativeOracle` + lockstep comparator |
| `tests/environment/test_env_contract.py` | G2-A (7 tests) |
| `tests/environment/test_battle_step.py` | G2-B / G2-D (14 tests) |
| `tests/environment/test_episode_parity.py` | G2-C (3 tests) |
| `tests/environment/test_p1_integration.py` | G2-E P1 policy crossing (2 tests) |
| `tests/environment/test_contract_edges.py` | truncation, staleness, revision, concluded-row contracts (11 tests) |
| `tests/environment/test_measured_evidence.py` | measured metrics + rejection tests (14 tests) |

Total: **51 environment tests**. No Collector, no PPO, no benchmarking, no cloud work.
One focused P1 change was required (see §3, F1); P1 regression remains 39/39.

## 2. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G2-A** Real EnvBase spec | **PASS** | Genuine `EnvBase` subclass; real `check_env_specs succeeded!`; fixed keys/shapes/dtypes in `TENSORDICT_SPEC.md`; `rand_action` is engine-sourced from `candidates_batch` |
| **G2-B** Battle & barrier | **PASS** | 4/4 distinct seeded games terminated naturally (20/11/19/14 steps); rewards match the native winner in both directions; `Wait` seats never present; exactly one `step_batch` per advance; 0 operational errors, 0 illegal submissions, 0 duplicate completions, 0 revisions, 0 truncations |
| **G2-C** Source parity | **PASS** | Second independent `NativeEngine`; lockstep over 4 seeds comparing request kinds, branch counts, ordered candidates per prefix, all 9 observation leaves per seat, full native step tuple and outcome — zero mismatches; deterministic reset; partial reset preserves the untouched slot |
| **G2-D** Privacy & lifetime | **PASS** | `player_view(seat)` returns only that seat's view; no engine-private keys; bad seats raise; knownness masks preserved; stale `request_id`, native `generation`, `episode_id`, present mismatches, wrong branch counts and extra actions on `Wait`/`Finished` all raise **before** submission; no failure path becomes a completed match |
| **G2-E** P1 integration & audit | **PASS** | `ConditionalPolicy.complete_request` produces each requesting seat's prefix, is verified by replay (`abs_joint_delta <= 1e-5`), asserts an actor-active row, and the packet crosses `EnvBase.step()` matching a direct oracle `step_batch`; a second test plays a **full battle** with P1 decisions on both seats. P1 regression **39/39 green**; lock/PA3 unchanged; auditor read the WSL repo and returned CLEAN |

Test totals (Lead-executed, `uv run --frozen`): **51 environment tests OK**, **39 P1 regression tests OK**, `check_env_specs succeeded!`.

## 3. Defects found and fixed during P2

Each was found by running real engine code or by a read-only review that read the
actual files — not by inspection prose.

1. **Spec dtype mismatch.** `Binary` *defaults* to `int8` while the env emits
   `bool`; `check_env_specs` failed (`fake=torch.int8 and real=torch.bool`).
   Boolean leaves use `Unbounded(dtype=torch.bool)`. The spec was corrected to
   match truthful data rather than casting data to satisfy a wrong spec.
2. **Nested Composite shape.** `players` needed the full `[N,2]` shape, not `[2]`.
3. **Oracle seed off-by-one.** The oracle re-derived its seed after reset
   incremented `episode_id`, replaying the wrong episode. Fixed by recording the
   exact `used_seed`/`used_teams` submitted.
4. **Partial reset destroyed siblings.** `reset_batch` replaces every slot,
   invalidating untouched handles. Fixed with `reset_slots_batch` on masked slots.
5. **Truncation counted as a completed match.** The guard was `(term or trunc)`.
   Completion now requires `term`; truncation is tracked in `truncated_episodes`
   with no match and no payout.
6. **F1 (BLOCKER, review-found): a false G2-E claim.** At `67f1d59` the
   "P1 staged policy packet" test selected candidates with `torch.randint`;
   `ConditionalPolicy` appeared only in a separate replay test that never called
   `env.step()`, although the report and receipt said the packet crossed. Fixed:
   the policy now produces the submitted prefixes. PA3 refuses to resolve token
   refs for non-PICK actions without the player-known active→roster map, so
   `build_branch_table`/`ConditionalPolicy` thread `observation.active_to_roster()`
   (additive, default-compatible; P1 had only exercised PICK actions).
7. **Harness RNG defect.** The measurement harness seeded actions from the step
   index only, so all four "distinct seeds" played an identical sequence. Seeding
   now includes `env._seed`; the four games are genuinely distinct.
8. **N1 (review-found): post-terminal flag fabrication.** After a truncated
   episode, the next step re-reported `terminated=True, truncated=False` and lost
   the winner. Concluded rows now record the raw native flags/winner at commit and
   re-report exactly those.
9. **Identity/validation gaps (review-found F2–F6).** Slot/generation/episode
   identity is now mandatorily carried and checked; supplied prefix length must
   equal the native branch count; illegal rows raise before `step_batch`;
   `accepted=False` is a counted revision, never a completion; prohibited empty
   `step_batch` is no longer issued.

## 4. Review history (read-only auditor, fresh contexts)

| Round | Access | Commit | Result |
|---|---|---|---|
| 1 | Windows `opencode.exe` | `67f1d59` | `INSUFFICIENT_EVIDENCE`; denied `permission.rejected: external_directory` on every WSL path |
| 2 | WSL, repo read confirmed | `67f1d59` | `FINDINGS`: **F1 BLOCKER** (false P1-packet claim) + F2–F6 |
| 3 | WSL | `e953495` | F1/F3/F4 resolved; F5/F6 partial; **N1** post-terminal fabrication confirmed |
| 4 | WSL | `e310cce` | N1/N2/N3/N5 resolved; no code blocker; receipt values found stale |
| 5 | WSL | `136d333` | **`VERDICT: CLEAN`** |

Round 2's F1 finding was correct and is the reason this phase was not passed
earlier. The auditor is the **same model** as the Lead: this is a process control
and a second reading of the files, **not** independent-model verification.
Correctness rests on the executed tests and the PA3 reference contracts.

## 5. Limitations (explicit)

1. **4 seeded games only** — exactly the G2-B minimum, no margin.
2. **Oracle shares the PA4 decoder path.** It is a second native engine, so it
   catches engine/wrapper divergence, but a decoder bug would be invisible to it.
3. **No GPU simulation.** Native stepping is CPU; no throughput claim.
4. **Truncation is proven at the envelope level, not by a real truncating
   episode.** A synthetic native result drives `_Proxy`; natural play produced no
   truncation. The accounting contract is tested, the native occurrence is not
   observed.
5. **Max legal candidates 18** is an inherited fixture bound, not a global max.
6. **Random-weight policy** (seed 20261006); no strength claim.
7. **Minor uncovered test edges** (auditor M1–M4, M7): the concluded-row
   `present=False,count>0` branch has no direct test; `truncated_episodes`
   persistence across reset is implemented but unasserted; the parity test
   hard-codes env `accepted=True`; the P1 two-seat test asserts an actor-active
   row rather than both seats; there is no `len(results)==len(specs)` guard.
   None affects a gate result.
8. Elapsed time is not benchmarked or optimised.
9. PA3 pin is the public `main` tip, not verified live production state.

## 6. P3 handoff

- EnvBase shapes/semantics: `TENSORDICT_SPEC.md` (fixed keys, dtypes, privacy
  boundary, post-terminal contract).
- Action packet schema: all six keys required; `request_ids` increment per advance
  and per revision, restart at `(1,1)` on reset; identity mismatch raises.
- P1 `ENCODE/STEP/RESET` cache semantics and per-request policy version.
- Real `check_env_specs` receipt and small natural-game traces.
- **Carry forward:** CPU-only policy server, no proven GPU batching, PA3 pinned to
  a public SHA only, P1 replay needs the original observation payload, truncation
  never observed natively, and the auditor shares the Lead's model.

**P3 is not started.**

---

## 7. Correction record (kept deliberately)

The first P2 submission (`67f1d59`) was **wrong** and was reported as PARTIAL for
the wrong reason. A reviewer with repository access found that the test named
"P1 staged policy packet" used `torch.randint`; no P1 policy ever reached
`EnvBase.step()`, while the receipt claimed `p1_action_packet_integration_pass:
true`. The fix required threading PA3's player-known active→roster map, and the
test now proves policy provenance by replay.

A second error surfaced while fixing that: the measurement harness produced four
identical games, so the "4 distinct seeded games" claim was weaker than stated.

The PARTIAL verdict that followed was deliberately **not** upgraded on the
strength of the code alone: the design requires an auditor that reads the actual
repository, and for several sessions no such auditor existed. Once the Lead ran
from WSL, access was demonstrated with a literal file read, the review found a
real blocker, and the gate was only passed after the blocker and its residuals
were fixed and re-reviewed.
