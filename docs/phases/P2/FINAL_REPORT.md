# P2 — Final Report

**Phase:** P2 — Rust ↔ TorchRL Environment Adapter
**Design:** `PA4-P2-DESIGN-v1.0`
**PA4 base (P1 HEAD):** `892f136f3704b773257a377c9a41c4fa50c74770`
**PA4 tested code:** `67f1d59dccf51d1be7f39d7452c86a0dcf30033a`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (unmodified)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0/P1)

---

## VERDICT: **PARTIAL**

**Four gates pass. G2-E is PARTIAL** because its auditor leg — an independent
read-only review of the *actual WSL repository files* — could not be performed:
this Lead session runs as the Windows `opencode.exe` rooted at
`C:\Users\임민우\Desktop\PA4`, and the auditor was denied
`permission.rejected: external_directory` on every WSL path. The design states
explicitly that inline-only review is not an acceptable P2 independent file
review (§P2-D, §5).

Nothing about the implementation is being marked failed. The gate simply has an
unmet conjunct, and I will not report PASS for something that did not happen.

---

## 1. What was built

| File | Role |
|---|---|
| `pa4/env/native_env.py` | `PA4EnvBase(EnvBase)` — real native-backed environment |
| `pa4/env/trace_compare.py` | Independent `NativeOracle` + lockstep comparator |
| `tests/environment/test_env_contract.py` | G2-A (7 tests) |
| `tests/environment/test_battle_step.py` | G2-B / G2-D (14 tests) |
| `tests/environment/test_episode_parity.py` | G2-C (3 tests) |
| `tests/environment/test_p1_integration.py` | G2-E (2 tests) |
| `tests/environment/test_contract_edges.py` | review-driven edge contracts (4 tests) |

No Collector, no PPO, no benchmarking, no cloud work.

## 2. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **G2-A** Real EnvBase spec | **PASS** | Genuine `EnvBase` subclass; real **`check_env_specs succeeded!`**; fixed keys/shapes/dtypes documented in `TENSORDICT_SPEC.md`; legal random action asserted row-by-row against engine candidates |
| **G2-B** Battle & barrier | **PASS** | 4/4 seeded games terminated naturally (18/22/17/17 steps); rewards match winner in **both** directions; `Wait` seats never marked present; exactly **one** `step_batch` per advance; 0 operational errors, 0 illegal submissions, 0 duplicate completions |
| **G2-C** Source parity | **PASS** | Independent second `NativeEngine`; lockstep over 4 seeds comparing request kinds, branch counts, ordered candidates per prefix, all 9 observation leaves per seat, full native step tuple and outcome — **zero mismatches**; deterministic reset; partial reset preserves the untouched slot |
| **G2-D** Privacy & lifetime | **PASS** | `player_view(seat)` returns only that seat's view; no engine-private keys; bad seats raise; knownness masks preserved; stale request ids, present mismatches, branch overflow and zero counts all rejected; cross-episode stale packet rejected |
| **G2-E** P1 integration & audit | **PARTIAL** | P1 staged packet **did** cross `EnvBase.step()` and match a direct oracle `step_batch`; P1 regression **39/39 green**; lock/PA3 unchanged. **Auditor could not read WSL files — conjunct unmet.** |

Test totals: **30 environment tests OK**, **39 P1 regression tests OK**.

## 3. Defects found and fixed during P2

Each was found by running real engine code, not by inspection alone.

1. **Spec dtype mismatch.** torchrl `Binary` is `int8`; the env emits `bool`.
   `check_env_specs` failed until boolean leaves used
   `Unbounded(dtype=torch.bool)`. The spec was corrected to match truthful data
   rather than casting data to satisfy a wrong spec.
2. **Nested Composite shape.** `players` needed the full `[N,2]` shape, not `[2]`.
3. **Oracle seed off-by-one.** The oracle re-derived its seed *after* reset
   incremented `episode_id`, replaying the wrong episode (`invalid action
   prefix`). Fixed by recording the exact `used_seed`/`used_teams` submitted.
4. **Partial reset destroyed siblings.** `reset_batch` replaces every slot,
   invalidating untouched handles (`environment handle`). Fixed with
   `reset_slots_batch` on the masked slots only.
5. **Truncation counted as a completed match** — *found by the read-only review*.
   The guard was `(term or trunc)`, which set `match_completed` and incremented
   `completed_matches` for a native truncation. Now completion requires `term`;
   truncation is tracked in `truncated_episodes` with no match and no payout.

## 4. Review history

- Auditor **round 1**: `INSUFFICIENT_EVIDENCE`. It refused to grade a summary,
  and could not open any WSL file (`permission.rejected: external_directory`).
  It raised 8 points; **2 were substantive and correct** (truncation-as-
  completion; my prose test-count miscount of 9+15 vs the real 7+14=26), 1 was
  correct-but-not-a-bug (partial-reset arity — verified correct), and the rest
  were already handled or unverifiable without file access.
- I fixed the truncation defect and added 4 edge tests rather than arguing.

**Auditor independence remains a structural limitation:** the same-model
auditor runs in the Windows domain and cannot read the WSL repo. Fixing this
requires relaunching the Lead from WSL (`cd ~/projects/PokeAgent4.0 && opencode`).

## 5. Limitations (explicit)

1. **Auditor did not read the WSL repo.** G2-E is partial for this reason alone.
2. **4 seeded games only** — exactly the G2-B minimum, no margin.
3. **Oracle shares the PA4 decoder path.** It is a second native engine, so it
   catches engine/wrapper divergence, but a decoder bug would be invisible to it.
4. **No GPU simulation.** Native stepping is CPU; no throughput claim.
5. **Max legal candidates 18** is an inherited fixture bound, not a global max.
6. **Random-weight policy** from P1 is unchanged; no strength claim.
7. Elapsed time is not benchmarked or optimised.

## 6. P3 handoff

- EnvBase shapes/semantics: `TENSORDICT_SPEC.md` (fixed keys, dtypes, privacy boundary).
- Action packet schema and acceptance/revision behaviour; `request_ids` increment per step and per revision, reset to `(1,1)` at episode start.
- P1 `ENCODE/STEP/RESET` cache semantics and per-request policy version.
- Real `check_env_specs` receipt and small natural-game traces.
- **Carry forward:** CPU-only policy server, no proven GPU batching, PA3 pinned to public SHA only, P1 replay needs the original observation payload, and P2's auditor leg is unmet.

**P3 is not started.**
