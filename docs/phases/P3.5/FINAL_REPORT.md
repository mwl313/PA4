# P3.5 — Observation Contract Repair: Final Report

**Phase:** P3.5 (prerequisite to P4)
**ADR:** `ADR_009.md`
**Status:** implementation complete with **one unresolved environmental gate** (fresh-context auditor access; see §8)
**Base commit:** `a977a5c`
**Tested implementation commit:** `765fb31`
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
| `pa4/collector/run_batched.py` | `_views_from_players` includes ragged; **representation gate enforced** |
| `pa4/collector/transitions.py` | retains ragged, digest covers it, stamps representation id |

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

### Two real bugs found and fixed during the work

Both were caught by tests rather than shipped:

1. **`np.ascontiguousarray` on a slice does not copy.** `.view(dtype)` then
   reinterpreted from the *base* buffer's origin and read uninitialised bytes
   (observed values like `-137363456.0`). Fixed with an owning
   `np.array(..., copy=True)`.
2. **The native ragged buffer is laid out PER VIEW, not per family.** Multi-view
   decoding was misaligned from view 1 onward. Fixed to walk view-major, and a
   dedicated regression test now pins it.

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
| Real collector end-to-end | `smoke_collector.py` | 4 natural matches, 108 seat records, all `ragged.v2`, 0 drops |
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
| `tests/collector` (P3 + P3.5) | **82 OK** |
| **Total** | **172 tests OK** |
| `check_env_specs` (`env.rollout(2)`) | **OK** |
| `uv.lock` | unchanged (`8f414f52…`) |
| PA3 reference | unchanged, 0 dirty |

Two pre-existing tests were **strengthened, not loosened**: the P2 determinism
test now also asserts ragged determinism, and the fast/full decoder equivalence
test now asserts ragged equality instead of only the nine fixed leaves.

---

## 4. Performance — a real regression, reported as measured

Measured on the same host, **minutes apart**, same config, with the repair
stashed and restored (`measure_perf.py`):

| Config | Pre-repair (this host, now) | P3.5 ragged | Delta |
|---|---|---|---|
| 16 envs | 3.129 matches/s | 2.42 matches/s | **−23%** |
| 32 envs | 4.309 matches/s | 2.87 matches/s | **−33%** |

**Important context:** the pre-repair numbers measured *today* (3.13 / 4.31) are
themselves well below the P3 receipt's recorded 5.12 / 5.47. Comparing the
repaired figure to the old receipt would imply a 2.1× regression, which is
**wrong** — most of that gap is host variance, not the repair. The honest
attributable cost is **23–33%**.

Where the cost sits (`stage_perf.py`, per env step):

| Stage | ms/step |
|---|---|
| `step_levels` (transport latency) | 225.45 |
| `encode` | 64.81 |
| `env_step` | 12.13 |
| `packet` | 12.13 |
| `reset` | 6.65 |
| `accounting` | 6.61 |
| **`views` (all P3.5 work)** | **2.91** |

The repair itself is ~2.9 ms/step. Two optimisations were applied during the
work: `payload_to_transport` went 1773 µs → 418 µs (4.2×) by removing per-token
Python loops and per-view `_matrix` calls, and the inference wire row went
154 KB → 119 KB by right-sizing padding from evidence. What remains is mostly
the larger padded tensors crossing the inference transport — an expected and
acceptable cost for correct inputs.

---

## 5. Representation identity

`OBSERVATION_REPRESENTATION = "pa4.observation.ragged.v2"` is stamped on every
seat record. Pre-repair records are `"pa4.observation.fixed9.v1"`.

Enforcement is **real, not merely defined**: `run_batched._assert_views_representable`
runs in the collector loop and raises `ObservationContractMismatch` for any view
without ragged. Two tests prove a nine-leaf-only view is refused.

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

---

## 7. What this unblocks

G4-B's conjunct *"real PA3-derived policy/value inputs verified including feature
contract"* is now satisfiable on the true PA3 forward contract, with **no
tolerance relaxation**. P4 may resume from `765fb31` with the corrected
observation contract.

---

## 8. Unresolved gate — auditor access (BLOCKED, environmental)

A fresh-context `pa4-auditor` was spawned and returned **`INSUFFICIENT_EVIDENCE`**:

```
{"error":{"type":"permission.rejected","message":"Permission denied: external_directory"}}
```

It read **zero files**. It correctly refused to approve on the strength of a
summary, and explicitly did not fabricate findings. Its sandbox is rooted at
`C:\Users\임민우\Desktop\PA4`, which contains only design documents.

This is the **same environmental blocker recorded for G3-E**: only an OpenCode
session launched *from WSL inside the repository* gives an auditor read access.
The auditors available to a Windows-rooted session inherit that root.

I am **not** claiming an auditor PASS. This phase is therefore reported as:

> **P3.5 implementation: PASS on all executable gates. Auditor review: BLOCKED.**

The three substantive concerns the auditor raised before being blocked were
independently adjudicated by executable tests and recorded: byte layout
(`test_multi_view_splits_per_view`), aliasing
(`test_transport_is_not_aliased_to_payload`), and fail-closed overflow
(`test_overflow_fails_closed`, `test_no_truncation_at_real_capacity`). Its
concern #5 — that the identity gate was only defined, never enforced — was
**valid and was acted on**, which is why `_assert_views_representable` now runs
in the collector loop.

To close this gate, relaunch OpenCode from WSL:
`wsl` → `cd /home/pa4/projects/PokeAgent4.0` → `opencode`, then have
`pa4-auditor` review commit `765fb31` read-only.
