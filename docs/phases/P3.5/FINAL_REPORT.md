# P3.5 — Observation Contract Repair: Final Report

**Phase:** P3.5 (prerequisite to P4)
**ADR:** `ADR_009.md`
**Status:** implementation + round-8 correction package complete; P35_I fresh-context re-audit pending (see §8)
**Base commit:** `a977a5c`
**Tested implementation commit:** `06e3b04`
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

All figures are from committed scripts under `docs/phases/P3.5/evidence/`.

| Check | Script | Result |
|---|---|---|
| Transport byte-exact vs PA3 canonical | `verify_transport_golden.py` | **960 views, 0 mismatches** — `BYTE-EXACT PASS` |
| Encoder + value parity, CUDA | `verify_encoder_parity.py` | **480 views, max \|Δtoken\| = max \|Δglobal_repr\| = max \|ΔV\| = 0.000e+00** |
| Negative controls | `verify_encoder_parity.py` | **2264 / 2264 detected** |
| Live TensorDict → encoder path | `verify_live_path.py` | **86 decision views, 0.000e+00** — `LIVE-PATH PARITY: PASS` |
| Valid-view-only equivalence | `verify_valid_only.py` | 35 decision views, **0 mismatches** |
| Real collector end-to-end | `smoke_collector.py` | 4 natural matches, **91 seat records, all `ragged.v2`, 0 drops** |
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
| `tests/collector` (P3 + P3.5) | **95 OK** |
| **Total** | **185 tests OK** |
| `check_env_specs` (`env.rollout(2)`) | **OK** |
| `uv.lock` | unchanged (`8f414f52…`) |
| PA3 reference | unchanged, 0 dirty |

Pre-existing tests were **strengthened, not loosened**: the P2 determinism test
now asserts ragged determinism (and the representation id), the fast/full
decoder equivalence tests assert ragged equality for both seats and the batched
path, and `make_row` now carries real ragged leaves instead of placeholders.

---

## 4. Performance — a real regression, reported as measured

Fresh same-session A/B: the pre-repair tree `a977a5c` was extracted with
`git archive` and measured with the same virtualenv and config; the repaired
tree `06e3b04` was measured minutes later (`evidence/measure_perf_ab.json`):

| Config | Pre-repair (this host, now) | P3.5 ragged | Delta |
|---|---|---|---|
| 16 envs | 3.191 matches/s | 2.362 matches/s | **−26%** |
| 32 envs | 4.074 matches/s | 3.136 matches/s | **−23%** |

**Important context:** the pre-repair numbers measured *today* are themselves
well below the P3 receipt's recorded 5.12 / 5.47. Comparing the repaired figure
to the old receipt would imply a much larger regression, which is **wrong** —
most of that gap is host variance, not the repair. The attributable cost in this
A/B is **23–26%**, one run per configuration.

Where the time sits (`stage_perf.py`, repaired path only, per env step):

| Stage | ms/step |
|---|---|
| `step_levels` (inference transport latency) | 236.18 |
| `encode` | 71.63 |
| `env_step` (contains `payload_to_transport`) | 13.03 |
| `packet` | 11.75 |
| `accounting` | 10.11 |
| `views` (leaf extraction + identity gate only) | 8.26 |
| `reset` (also contains `payload_to_transport`) | 7.26 |
| `request_info` | 0.04 |

The `views` stage rose from 2.91 to 8.26 ms/step because the gate now validates
every ragged leaf rather than checking a key. The `views` stage is **not** the
repair cost: `payload_to_transport` runs inside `env.step`/`reset`.
`stage_perf.py` records the repaired path only; it is **not** an ON/OFF ablation
and no longer claims to be one. `bench_ragged.py` times the
components directly: `payload_to_transport` ≈ 948 µs and `decode_batch` ≈ 1741 µs
for 16 views. The inference wire row is 119 KB after right-sizing padding
(was 154 KB). What remains is mostly the larger padded tensors crossing the
inference transport — an expected and acceptable cost for correct inputs.

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
5. **The fresh A/B is one run per configuration.** Run-to-run variance is not
   characterised; the measured deltas are −26%/−23% in this A/B, not a
   universal constant.
6. **The auditor can read the PA4 repo but not the PA3 checkout** in its
   sandbox (`external_directory`). PA3-side layout claims are verified by the
   committed golden scripts and tests, not by the auditor's direct PA3 read.

---

## 7. What this unblocks

G4-B's conjunct *"real PA3-derived policy/value inputs verified including feature
contract"* is now satisfiable on the true PA3 forward contract, with **no
tolerance relaxation**. P4 may resume from `06e3b04` **only after P35_I closes**.

---

## 8. Round-8 audit findings, correction package, and P35_I

A fresh-context `pa4-auditor` **did read the real WSL repository** at `d0be6e3`
(`ACCESS_OK`) and returned `FINDINGS`, not CLEAN. The four MAJOR findings were
independently confirmed and fixed in the correction package at `06e3b04`:

| ID | Finding | Fix |
|---|---|---|
| F1 | The collector gate was vacuous: `_views_from_players` synthesised the key the gate checked; a missing leaf raised raw `KeyError`; the server silently omitted ragged. | The extractor validates the actual leaves and raises `ObservationContractMismatch`; views carry an explicit representation id; the gate checks id+leaves; `encode_all` validates; `_obs_batch` fails closed. |
| F2 | `next_observation` was stamped `ragged.v2` while storing only nine leaves. | It now retains `observation_ragged`; a v2 view without ragged raises. |
| F3 | No test exercised non-empty ragged through the real server seam; deleting the forwarding left all tests green. | `tests/collector/test_server_seam_ragged.py` runs env → `_views_from_players` → `StagedInference.encode_all` → `InferenceServer` → `_obs_batch` → encoder and compares to canonical PA3. |
| F4 | `pa4_tested_code_sha` was stale; the receipt had a duplicate `transport.representation` key. | The receipt pins `06e3b04`; the duplicate key is split into `representation_id` and `encoding`; all evidence was regenerated. |
| F5 | Decode-side overflow was not fail-closed; the ADR claimed an assertion that did not exist. | `validate_transport_row` checks counts/capacity/lengths/overflow on every decode; the ADR is corrected; the `[B,96]` overflow shape bug is fixed. |
| F6 | The aliasing test re-decoded the same state, so it could not detect aliasing. | It now mutates the payload buffer in place; permute and truncate digest controls were added. |
| F7 | Performance stage attribution was wrong (`views` is not all repair work); `stage_perf.py` claimed an ON/OFF ablation it did not run. | Corrected in §4 and the receipt; `stage_perf.py` no longer claims ON/OFF. |
| F8 | Evidence hygiene defects (vacuous drop check, dead golden import/helper, wrong probe itemsize). | Fixed and evidence regenerated. |
| F9 | 3-D `[B, seats, tokens]` ragged input silently collapsed to seat 0. | Rejected explicitly. |

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
