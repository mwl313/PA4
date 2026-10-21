# P1 — Final Report

**Phase:** P1 — Conditional Action Feasibility Gate
**Design:** `PA4-P1-DESIGN-v1.0`
**PA4 base:** `a900f4793afb4417c1f3226a5d423aca9c038e43`
**PA4 tested:** `aa1a624776f5f646efb231e7c4fce3c8950be3d0`
**PA3 reference:** `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (read-only, unmodified)
**Dependency lock:** `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged from P0)

**Verdict: PASS** — with the explicit limitations in §6.

---

## 1. What was actually built

| File | Role |
|---|---|
| `pa4/actions/native_reference.py` | Sole legality gateway to the real Rust engine |
| `pa4/actions/conditional_policy.py` | Staged prefix selection, replay record, digests |
| `pa4/actions/server_probe.py` | Real TorchRL server: ENCODE/STEP/RESET + cache |
| `pa4/actions/server_factory.py` | Module-level picklable `policy_factory` |
| `tests/action_distribution/*.py` | 39 gate tests (stdlib `unittest`) |

No `EnvBase`, collector, PPO loss, optimizer or checkpoint I/O — all out of P1 scope.

## 2. Real components, not mocks

- **Rust**: `pa3-engine` built from the pinned SHA with Rust **1.90.0** (PA3's own
  `rust-toolchain.toml`) and the existing `python` feature (PyO3 0.26), in a scratch
  copy at `/home/pa4/.pa3_build`. PA3 itself stayed clean (0 modified files).
- **Policy**: the real PA3 `PA3Model` — **9,889,755 parameters**, matching PA3's
  source-of-truth figure. Random initialisation with fixed seed `20261006`.
- **Server**: real `torchrl.modules.inference_server.ProcessInferenceServer` over
  `process_slot` transport, started, driven and shut down.

## 3. The three questions, answered

**Q1 — Complete legal conditional player action without changing Rust rules?**
Yes. Every candidate table comes from `candidates_batch(handle, side, prefix)`.
The policy only scores/samples rows the engine produced; `illegal_actions = 0`
across all completed requests.

**Q2 — Single batch or staged prefix?**
**Staged prefix (path B).** Measured: Preview K = 6 → 5 → 4 → 3 as picks are made.
The engine computes branch *j*'s set after branches `< j`, so tables cannot be
pre-built in one pass. It still runs through **one** server with a fixed layout.

**Q3 — Minimum metadata for exact recomputation?**
Ordered per-branch candidate tuples, legal mask, entity/move token refs,
selected index, `branch_k`, per-branch FP32 logprob, plus policy weight
fingerprint, policy version, observation content digest and contract identity.

## 4. Gate results

| Gate | Result | Decisive evidence |
|---|---|---|
| **G1-A** Rust legality | **PASS** | 8 tests: Preview 4 branches, duplicate exclusion with exact K decrement, 6 distinct level-0 members, signed targets −2..2 unclamped, Mega prefix dependence, Wait/Finished excluded |
| **G1-B** Branch distribution | **PASS** | `sum(P_legal)=1` within 1e-6; padded/illegal probability **exactly 0**; no NaN/Inf; K=1 ⇒ logprob 0; hidden-carry test compares two level-0 picks |
| **G1-C** Replay / gradient | **PASS** | Replay delta **0.0** (gate ≤1e-5), no RNG on replay path, refuses weight/version/observation mismatch, validates mask and index, entropy+KL finite with nonzero gradient |
| **G1-D** Central inference | **PASS (CPU)** | Real `ProcessInferenceServer` alive; 2 slots; staged 4-level completion through the transport; per-request cache isolation; fail-closed on wrong level/key/version |
| **G1-E** Bounded representation | **PASS** | 64 accepted, **65 raises** on both the builder and the wire path; `candidate_count` must equal mask sum; max observed K=18, depth 4 |

Test command:
```
uv run --frozen python -m unittest discover -s tests/action_distribution -t .
Ran 39 tests ... OK
```

## 5. Bugs found and fixed during P1 (each with evidence)

1. **`sample_levels()` resets the GRU every call.** It restarts from
   `Scorer.initial_state()`, so a per-level server call would silently discard the
   prefix state. Fixed by calling `sample_level_step` and carrying `hidden` in the
   cache. This is exactly the "stale hidden" failure G1-B guards.
2. **`ProcessSlotTransport` slot specs must be unbatched.** Batched specs produced
   `invalid observation shape`; the server stacks across slots.
3. **Extra rank on `legal_mask`** produced `(1,1,1,64)` and
   `torch.multinomial: prob_dist must be 1 or 2 dim`.
4. **Wire path had no overflow check** — a fixed `(64,6)` buffer would slice
   candidates past 64 silently. Now `ERR_OVERFLOW`.
5. **`candidate_count` vs `mask.sum()`** mismatch was accepted. Now
   `ERR_COUNT_MISMATCH`.
6. **Rejected STEP left its cache entry behind**, allowing stale reuse. Now evicted.
7. **Replay trusted a construction-time fingerprint**; now recomputed live.
8. **Digests were blind to dtype/shape**; now included.

### One reviewer finding that was a false positive

The read-only review argued replay scores each branch with a fresh hidden, making
a 0.0 delta self-comparison. Checked against the PA3 source: `evaluate_encoded`
→ `_run_branches` replays the branch loop from `initial_state` and advances hidden
through the stored prefix. Verified empirically — scoring level 1 with the correct
level-0 prefix gives `-1.608508587`, with a wrong prefix `-1.611402273`
(delta `2.89e-03`). Replay is genuinely hidden-aware. Recorded here rather than
"fixed", because nothing was broken.

## 6. Limitations (explicit)

1. **CPU only.** No GPU central-inference claim; `G1-D` passes as a CPU transport
   proof. The design says a CPU fallback cannot be reported as a GPU PASS.
2. **Random weights**, seed `20261006` — not a production checkpoint. No claim
   about trained strength or production weight parity.
3. **Small fixture set.** Max K=18 is an observed fixture maximum, not a proven
   global bound.
4. **No numerical comparison against a PA3 golden reference** beyond replay
   self-consistency; no PPO loss/optimizer parity (that is P4).
5. **`policy_cache_peak_bytes` not measured**; cache is small CPU `EncodedState`.
6. **Observation is not persisted** in the record — replay needs the original
   observation supplied by the caller (digest verified).
7. **Same-model review** is a process control, not independent verification.
8. **Auditor could not read the repo** — it reviewed inline source only, and did
   not independently reproduce the test run or commit hashes.

## 7. P2/P3 handoff

- **P2**: submit one completed player `Request` per seat through
  `step_batch`; handle `Wait`/`Finished`; two-seat simultaneous barrier.
- **P3**: adopt the request/response spec and cache key/lifecycle; measure real
  8 GB limits; decide whether `ProcessSlotTransport` carries collector volume.
- **P4**: replay record plus stored observation is the behavior-logprob source.

**P2 is not started.**
