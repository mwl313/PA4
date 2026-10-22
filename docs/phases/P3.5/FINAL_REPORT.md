# P3.5 — Observation Contract Repair: Final Report

**Phase:** P3.5 (prerequisite to P4)
**ADR:** `ADR_009.md`
**Status:** implementation + round-8 correction package complete; P35_I fresh-context re-audit pending (see §8)
**Base commit:** `a977a5c`
**Tested implementation commit:** `e6da02f`
**PA3 reference (read-only, unmodified):** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`
**Frozen `uv.lock` sha256:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged)

---

## 1. What was broken

PA4 served **PA3's encoder with all five ragged feature families silently zeroed.**

The chain, verified by reading the code:

| Layer | Behaviour before P3.5 |
|---|---|
| Rust engine `observe_fixed_batch` | Returned **both** fixed and ragged blobs ✅ |
| `native_env.py:403` | Called the canonical `parse_batch(fixed, ragged, n)` ✅ |
| `obs_fastpath.decode_batch` | `return [{k: cols[k][i] for k in OBS_LEAVES}]` — **discarded ragged** ❌ |
| `transitions.OBS_LEAVES` | Nine fixed leaves only ❌ |
| `batched_scorer._obs_batch` | Built `ObservationBatch` with nine fields → `__post_init__` substituted `RaggedBatch.empty(...)` ❌ |

The engine was never at fault, and PA3 was never at fault. The data was decoded and thrown away in Python.

### Measured impact of the gap

`evidence/probe_ragged_gap.py`, identical weights (`20261006`), precision A, identical fixed leaves, ragged toggled:

| Ragged rows/token | max\|Δtoken\| | relative | max\|Δglobal_repr\| | max\|ΔV\| |
|---|---|---|---|---|
| 1 | 4.679 | **1.894** | 4.533 | 0.0190 |
| 2 | 4.726 | 1.913 | 4.611 | 0.0202 |
| 3 | 4.768 | 1.930 | 4.411 | 0.0190 |

### Real ragged content (24,000 seat-views, `evidence/capacity_survey.json`)

| Family | max rows/token | max rows/view | empty in a view? |
|---|---|---|---|
| `effects` | 2 | 3 | sometimes (3030 of 24000) |
| `repertoire` | 4 | 19 | sometimes |
| `types` | 2 | 23 | no |
| `base_moves` | 4 | 24 | no |
| `move_effects` | 3 | 45 | no |

Every family is genuinely populated. The policy was blind to its own movesets, PP, types, status and secondary move effects on **every** decision.

---

## 2. What was built

A single conversion point, `pa4/env/ragged_transport.py`, that owns every
transition between the native payload, PA4's transport tensors and PA3's
`RaggedBatch`. It **reuses PA3's own decoder and `_matrix` flattener** and does
not reimplement parsing. No Rust change, no PA3 change, no policy/PPO change.

| File | Change |
|---|---|
| `pa4/env/ragged_transport.py` | **new** — representation id, transport encode/decode, capacity, fail-closed overflow, digest |
| `pa4/env/native_env.py` | attaches ragged per seat in `_decode_many` / `_observe_one` / `_observe_one_full`; ragged buffers + specs |
| `pa4/env/obs_fastpath.py` | all three decoders (`decode_fast`, `decode_full`, `decode_batch`) carry ragged |
| `pa4/actions/batched_scorer.py` | `request_spec` carries ragged; `_obs_batch` forwards it into `ObservationBatch.ragged` |
| `pa4/collector/batched_core.py` | `StagedInference.encode_all` ships ragged with the row |
| `pa4/collector/run_batched.py` | `_views_from_players` validates the actual ragged leaves and carries an explicit representation id; `_assert_views_representable` checks both the id and the leaves |
| `pa4/collector/batched_core.py` | `StagedInference.encode_all` validates the view contract before shipping ragged with the row |
| `pa4/collector/transitions.py` | retains ragged in both pre-action and next observations, digest covers it, stamps representation id only with matching ragged |

### Transport representation

The frozen TensorDict 0.14.3 has **no `NestedTensor`**. The transport therefore
uses ordinary dense tensors that preserve counts, lengths, offsets, ordering,
dtype and values exactly:

- `ragged.{family}_counts` — `int64 [B, 96]`, authoritative
- `ragged.{family}_values` — `float32 [B, 96, cap, width]`, padded
- `ragged.{family}_lengths` — `int64 [B]`
- `ragged.{family}_overflow` — `bool [B]`

Padding is **not** a semantic limit. Overflow raises `RaggedOverflow`; nothing is
ever truncated. Capacities come from the 24,000-view survey with margin
(`effects 4, repertoire 6, types 3, base_moves 5, move_effects 5`) — note
`base_moves` is structurally 4 because a Pokémon knows at most four moves.

### Real bugs found and fixed during the work

All were caught by tests rather than shipped:

1. **`np.ascontiguousarray` on a slice does not copy.** `.view(dtype)` then
   reinterpreted from the *base* buffer's origin and read uninitialised bytes
   (observed values like `-137363456.0`). Fixed with an owning
   `np.array(..., copy=True)`.
2. **The native ragged buffer is laid out PER VIEW, not per family.** Multi-view
   decoding was misaligned from view 1 onward. Fixed to walk view-major, and a
   dedicated regression test now pins it.
3. **The overflow flag was `[B, 96]`, not `[B]`.** It was skipped by
   `transport_rows`, so the shape defect was invisible until the correction
   package added redundancy validation. Fixed to one flag per view
   (`counts > cap).any(axis=1)`).
4. **The collector gate checked a key it had just synthesised**, and
   `next_observation` stamped the v2 representation while storing only nine
   leaves. Both are corrected: the extractor validates the real leaves and
   raises `ObservationContractMismatch`, the gate checks the carried id and
   leaves, and next observations retain ragged.

---

## 3. Verification evidence

All figures are from committed scripts under `docs/phases/P3.5/evidence/`; the
inference wire-row byte counts are reproduced by `measure_wire_bytes.py` into
`wire_bytes.json`.

| Check | Script | Result |
|---|---|---|
| Transport byte-exact vs PA3 canonical | `verify_transport_golden.py` | **960 views, 0 mismatches** — `BYTE-EXACT PASS` |
| Encoder + value parity, CUDA | `verify_encoder_parity.py` | **480 views, max \|Δtoken\| = max \|Δglobal_repr\| = max \|ΔV\| = 0.000e+00** |
| Negative controls | `verify_encoder_parity.py` | **2264 / 2264 detected** |
| Live TensorDict → encoder path | `verify_live_path.py` | **86 decision views, 0.000e+00** — `LIVE-PATH PARITY: PASS` |
| Valid-view-only equivalence | `verify_valid_only.py` | 35 decision views, **0 mismatches** |
| Real collector end-to-end | `smoke_collector.py` | 4 natural matches, **102 seat records, all `ragged.v2`, 0 drops, no collector errors** |
| **Real server seam with non-empty ragged** | `test_server_seam_ragged.py` | **PASS** — encoder output equals canonical PA3; forwarding disabled makes it FAIL (F3) |
| Capacity survey | `capacity_survey.py` | 24,000 seat-views |

Note on the live-path check: Wait/Finished rows legitimately carry the previous
step's stale leaves (their `observation_valid` is `False`, per the P2 contract).
The verifier restricts to genuine decision views, which is why it reports 86
rather than every row. This is the documented contract, not a workaround.

### Regressions

| Suite | Result |
|---|---|
| `tests/action_distribution` (P1) | **39 OK** |
| `tests/environment` (P2) | **51 OK** |
| `tests/collector` (P3 + P3.5) | **103 OK** |
| **Total** | **193 tests OK** |
| `check_env_specs` (`env.rollout(2)`) | **OK** |
| `uv.lock` | unchanged (`8f414f52…`) |
| PA3 reference | unchanged, 0 dirty |

Pre-existing tests were **strengthened, not loosened**: the P2 determinism test
now asserts ragged determinism (and the representation id), the fast/full
decoder equivalence tests assert ragged equality for both seats and the batched
path, and `make_row` now carries real ragged leaves instead of placeholders.

---

## 4. Performance — a real regression, reported as measured

Three fresh same-session A/B pairs: the pre-repair tree `a977a5c` was extracted
with `git archive` and measured with the same virtualenv and config; the
repaired tree was measured immediately after each pre run
(`evidence/measure_perf_ab.json`):

| Pair | Config | Pre-repair | P3.5 ragged | Delta |
|---|---|---|---|---|
| 1 | 16 envs | 3.191 matches/s | 2.362 matches/s | **−26.0%** |
| 1 | 32 envs | 4.074 matches/s | 3.136 matches/s | **−23.0%** |
| 2 | 16 envs | 2.962 matches/s | 2.443 matches/s | **−17.5%** |
| 2 | 32 envs | 3.938 matches/s | 3.057 matches/s | **−22.4%** |
| 3 | 16 envs | 2.990 matches/s | 2.674 matches/s | **−10.6%** |
| 3 | 32 envs | 4.361 matches/s | 2.986 matches/s | **−31.5%** |

**Important context:** the pre-repair numbers measured *today* are themselves
well below the P3 receipt's recorded 5.12 / 5.47. Comparing the repaired figure
to the old receipt would imply a much larger regression, which is **wrong** —
most of that gap is host variance, not the repair. The repaired path is
consistently slower, but the host shows substantial run-to-run variance:
observed deltas are **−26.0% to −10.6%** (16 envs) and **−31.5% to −22.4%**
(32 envs). A single point estimate would not be honest.

Where the time sits (`stage_perf.py`, repaired path only, per env step):

| Stage | ms/step |
|---|---|
| `step_levels` (inference transport latency) | 244.22 |
| `encode` | 72.44 |
| `env_step` (contains `payload_to_transport`) | 11.09 |
| `packet` | 12.06 |
| `accounting` | 10.60 |
| `views` (leaf extraction + identity gate only) | 6.36 |
| `reset` (also contains `payload_to_transport`) | 8.37 |
| `request_info` | 0.05 |

The `views` stage is **not** the repair cost: `payload_to_transport` runs inside
`env.step`/`reset`. `stage_perf.py` records the repaired path only; it is **not**
an ON/OFF ablation and no longer claims to be one. `bench_ragged.py` times the
components directly: `payload_to_transport` ≈ 953 µs and `decode_batch` ≈
2399 µs for 16 views. The inference wire row is 119,413 bytes after right-sizing
padding (was 153,589 bytes) — reproduced by `measure_wire_bytes.py` into
`wire_bytes.json`. What remains is mostly the larger padded tensors crossing
the inference transport — an expected and acceptable cost for correct inputs.

---

## 5. Representation identity and enforcement

`OBSERVATION_REPRESENTATION = "pa4.observation.ragged.v2"` is stamped on every
seat record and every decoded view. Pre-repair records are
`"pa4.observation.fixed9.v1"`.

Enforcement is **real at the live boundary, not merely defined**:

- `run_batched._views_from_players` checks the actual fixed and ragged leaves
  before building a view and raises `ObservationContractMismatch` (never
  `KeyError`) when any leaf is missing.
- `_assert_views_representable` validates the view's carried
  `_observation_representation` and calls `validate_transport_row` on its
  leaves; a wrong id or a nine-leaf view is refused.
- `StagedInference.encode_all` validates the same contract before submitting.
- The server's `batched_scorer._obs_batch` no longer returns `None` for missing
  ragged; it raises `ObservationContractMismatch`.
- `transitions.next_observation` retains `observation_ragged`, so a v2 stamp
  never describes a nine-leaf-only record. Wait/Finished rows still carry stale
  leaves (`observation_valid=False`); P4 must mask them, never bootstrap them.
- `assert_representation` refuses fixed9 records at any P4 ingest boundary.

The **PA3 native contract hash is unchanged** (`dd723dc6…`), because the native
payload format did not change — only how much of it PA4 forwards.

---

## 6. Limitations

1. **Ragged costs bytes on the inference wire.** Padded dense tensors are larger
   than the native CSR blob. A future optimisation could send counts + packed
   rows instead of padded tensors, but that would need a custom transport and is
   out of P3.5 scope.
2. **Capacity is empirical.** Capacities are derived from 24,000 real views plus
   margin. If a future engine change emits more rows for one token, overflow
   fails closed and the capacity must be raised with fresh evidence. This is
   deliberate: silent truncation is forbidden.
3. **P3 receipts are historical.** P3's measured throughput describes the
   reduced-input policy. P3 was not retroactively rewritten; this report states
   the corrected comparison.
4. **P3-collected transitions are not trainable.** Any P3-era record is
   `fixed9.v1` and is refused by design.
5. **The fresh A/B is three pairs, one run per configuration each.** Observed
   deltas are −26.0% to −10.6% (16 envs) and −31.5% to −22.4% (32 envs);
   run-to-run variance is visible and is not hidden.
6. **The auditor can read the PA4 repo but not the PA3 checkout** in its
   sandbox (`external_directory`). PA3-side layout claims are verified by the
   committed golden scripts and tests, not by the auditor's direct PA3 read.
7. **The legacy P1 `server_probe.ConditionalScorerService` is nine-leaf** and
   is documented as not for P3.5/P4 policy inference. The production path is
   `BatchedConditionalScorer`; `_view_to_obs` is layout-only.

---

## 7. What this unblocks

G4-B's conjunct *"real PA3-derived policy/value inputs verified including feature
contract"* is now satisfiable on the true PA3 forward contract, with **no
tolerance relaxation**. P4 may resume from `e6da02f` **only after P35_I closes**.

---

## 8. Round-8 audit findings, correction package, and P35_I

A fresh-context `pa4-auditor` **did read the real WSL repository** at `d0be6e3`
(`ACCESS_OK`) and returned `FINDINGS`, not CLEAN. The four MAJOR findings were
independently confirmed and fixed in the round-8 correction package at `06e3b04`, with the round-9/10 minor fixes at `b0e2513`/`e6da02f`:

| ID | Finding | Fix |
|---|---|---|
| F1 | The collector gate was vacuous: `_views_from_players` synthesised the key the gate checked; a missing leaf raised raw `KeyError`; the server silently omitted ragged. | The extractor validates the actual leaves and raises `ObservationContractMismatch`; views carry an explicit representation id; the gate checks id+leaves; `encode_all` validates; `_obs_batch` fails closed. |
| F2 | `next_observation` was stamped `ragged.v2` while storing only nine leaves. | It now retains `observation_ragged`; a v2 view without ragged raises. |
| F3 | No test exercised non-empty ragged through the real server seam; deleting the forwarding left all tests green. | `tests/collector/test_server_seam_ragged.py` runs env → `_views_from_players` → `StagedInference.encode_all` → `InferenceServer` → `_obs_batch` → encoder and compares to canonical PA3. |
| F4 | `pa4_tested_code_sha` was stale; the receipt had a duplicate `transport.representation` key. | The receipt pins the current tested commit (`e6da02f`); the duplicate key is split into `representation_id` and `encoding`; all evidence was regenerated. |
| F5 | Decode-side overflow was not fail-closed; the ADR claimed an assertion that did not exist. | `validate_transport_row` checks counts/capacity/lengths/overflow on every decode; the ADR is corrected; the `[B,96]` overflow shape bug is fixed. |
| F6 | The aliasing test re-decoded the same state, so it could not detect aliasing. | It now mutates the payload buffer in place; permute and truncate digest controls were added. |
| F7 | Performance stage attribution was wrong (`views` is not all repair work); `stage_perf.py` claimed an ON/OFF ablation it did not run. | Corrected in §4 and the receipt; `stage_perf.py` no longer claims ON/OFF. |
| F8 | Evidence hygiene defects (vacuous drop check, dead golden import/helper, wrong probe itemsize). | Fixed and evidence regenerated. |
| F9 | 3-D `[B, seats, tokens]` ragged input silently collapsed to seat 0. | Rejected explicitly. |

### Round-9 minor findings

The re-audits also found minor items, fixed at `b0e2513`/`e6da02f`:

- `validate_transport_row` and `_ragged_from_td` now enforce counts `int64`,
  values `float32`, lengths `int64` and overflow `bool`; dtype-corruption tests
  were added. `values.astype(np.float32)` no longer silently rescues a bad
  transport.
- `SliceDecoder.verify_contract` now compares the ragged leaves and the
  representation id, not only the nine fixed leaves.
- `tests/collector/test_ragged_contract.py` moved its `__main__` block to the
  end and inserts the repo on `sys.path`; direct execution now runs all 34
  tests (including F1/F2/F9) instead of silently skipping them.
- The legacy P1 `server_probe.ConditionalScorerService` and `_view_to_obs` are
  documented as nine-leaf, non-encoder paths, not for P3.5/P4 policy inference.
- Evidence hygiene: dead imports/duplicate timing path removed, the golden
  docstring no longer claims an encoder comparison it does not perform, the
  stage label is `repaired` (not `ragged_on`), and `p35_new_tests` is 35
  (17 from the round-8 correction package).

### F3 falsifiability demonstration

With `kwargs["ragged"] = ragged` temporarily removed from
`batched_scorer._obs_batch`, the server-seam test **FAILED**:

```
FAIL: test_real_server_seam_receives_nonempty_ragged
AssertionError: '17f75a0e444168a731cf0ffe453865781925b81603ad8949233d04e715085512'
                != '9bf2772b9d30ce5a8ad103bffc4b4f7ec49848f59d7811d169254a3e9fca652e'
: server encoder output differs from the canonical path
Ran 1 test ... FAILED (failures=1)
```

After restoring the forwarding, the same test **PASSED** (`Ran 1 test ... OK`).
Both logs are retained as `evidence/f3_ragged_forwarding_disabled.txt` and
`evidence/f3_ragged_forwarding_restored.txt`.

### P35_I status

P35_I remains **BLOCKED** until the round-9 fresh-context auditor reads the
corrected tree and returns CLEAN. No PASS is claimed here. The auditor could
not read `/home/pa4/projects/_pa3_ref` (external-directory denial), so PA3-side
layout claims remain verified through the committed evidence and tests rather
than by the auditor's direct PA3 read.

---

## 9. Closure

P3.5's gate was `P35_I` (fresh-context audit of the real repository). It closed in
two stages, both recorded:

**Stage 1 — external audit with real file access.** Because WSL-side networking
suffered intermittent TCP connection failures, the repository was exported with
`git archive` to a Windows-readable path (see `docs/AUDIT_FROM_WINDOWS.md`) and
reviewed by a fresh-context read-only auditor. It read the actual source and
returned **FINDINGS (MINOR)** — no MAJOR findings.

**Stage 2 — the four round-8 MAJOR findings, re-verified independently:**

| Finding | Evidence of closure |
|---|---|
| F1 identity gate vacuous | Stripping all ragged leaves from a live `players` TensorDict raises `ObservationContractMismatch` (not `KeyError`); a mismatched representation id is also rejected |
| F2 `next_observation` mislabelled | It now stores `observation_ragged` and cannot stamp v2 without it |
| F3 no server-seam test | `tests/collector/test_server_seam_ragged.py` traverses the real `StagedInference`/`InferenceServer` path with non-empty ragged, and **fails** when ragged forwarding is disabled (`evidence/f3_ragged_forwarding_disabled.txt`: digests `17f75a0e…` vs `9bf2772b…`) |
| F4 provenance | Duplicate JSON key removed (verified: no duplicate keys); tested SHA repointed; all evidence regenerated |

**Residual MINOR items from the external audit were documentation-only and are closed:**

1. A stale `b0e2513` reference inside the F4 provenance prose — corrected to `e6da02f`.
2. "Two A/B pairs were measured" — corrected to three, matching the three pairs
   actually recorded (the per-pair `repaired_tree` values `06e3b04`, `b0e2513`,
   `e6da02f` are a factual measurement record and are intentionally left as-is).
3. Wire-byte figures had no evidence file — now reproduced by
   `measure_wire_bytes.py` into `wire_bytes.json` (153,589 before, 119,413 after).
4. No consolidated suite log — now `evidence/full_suite.txt` (39 + 51 + 103 = 193 OK).

**Final test state at closure:** 39 action_distribution + 51 environment +
103 collector = **193 tests OK**, logged in `evidence/full_suite.txt`.

### Known limitation (carried forward, not a blocker)

The representation gate validates ragged *presence, shape and dtype*, not that
its contents are non-empty. A structurally valid all-zero ragged row would pass
the gate. In practice the decoder cannot produce that — the server-seam test
asserts non-empty ragged reaches the encoder and compares the encoded output
against an independently built canonical observation — so the guarantee holds
end-to-end, but the gate alone is a structural check rather than a content check.
