# P1 — Action Distribution Design

Document ID: `PA4-P1-ACTION-DISTRIBUTION`
Parent: `PA4-P1-DESIGN-v1.0`
PA3 reference: `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (read-only)

---

## 1. The three P1 questions, answered

### Q1 — Can the TorchRL policy server sample a complete legal conditional player action without changing Rust legality?

**Yes.** Demonstrated live. Every candidate table comes from
`NativeEngine.candidates_batch(handle, side, prefix)`. The policy only scores and
samples rows the engine supplied; it never generates, filters or reorders them.
Masking happens inside PA3's `masked_log_softmax` before any sampling math, so
illegal candidates carry probability exactly 0.

### Q2 — Single batched request, or staged prefix protocol?

**Staged prefix (path B) is required.** Measured, not assumed:

| Level | Legal K | Why it changes |
|---|---|---|
| Preview 0 | 6 | all party members |
| Preview 1 | 5 | member picked at level 0 removed |
| Preview 2 | 4 | two members removed |
| Preview 3 | 3 | three members removed |

The Rust engine computes branch *j*'s candidate set **after** the branches chosen
before it, so the tables cannot be pre-built in one pass. A single request would
have to carry the whole legal prefix tree, which PA3 does not expose.

The staged protocol still runs through **one** `InferenceServer` with a fixed
TensorDict layout, using an `op` code (ENCODE / STEP / RESET). The encoder runs
once per request at ENCODE; every STEP reuses the cached `EncodedState`.

### Q3 — Minimum persisted metadata for exact logprob recomputation

`ActionReplayRecordV1` stores, per request:

- `request_id`, `policy_version`, **policy weight fingerprint** (sha256 over all
  `state_dict` tensors), `precision_profile`
- `observation_digest` (content digest over the decoded token tensors, not just
  a pointer), `schema_version`, `contract_hash`
- per branch, **in order**: `branch_slot`, ordered `candidate_tuples`
  (canonical 6-field rows exactly as the engine returned them), `legal_mask`,
  `entity_token_refs`, `move_token_refs`, `selected_index`, `branch_k`,
  `branch_logprob_fp32`
- `joint_logprob_fp32`, `actor_active`, `request_kind`

Replay **never resamples**: it re-evaluates the stored index against the stored
table and recomputes the sum. It refuses if weights, policy version or
observation content differ from the record.

---

## 2. Semantics implemented

```
log P(a_0:L-1 | s) = sum over valid j of log P(a_j | s, a_<j, C_j(a_<j))
joint_logprob_fp32 = sum(branch_logprob_fp32)
actor_active       = any(valid branch with branch_k >= 2)
```

- Probabilities and log-probabilities are computed in FP32 (`logits.float()`
  then masked log-softmax). PA3's floor is `torch.finfo(torch.float32).min`.
- `K = 1` ⇒ deterministic pick, branch logprob 0.
- `Wait` / `Finished` produce no actor row (`POLICY_REQUEST_KINDS = {Preview,
  Normal, Replacement}`).
- The GRU prefix state advances across levels **within one request**; it is not
  match-level recurrent memory.

## 3. Capacity and overflow policy

- Candidate capacity **64**, max branches **4** — asserted against real data.
- Measured maximum legal candidates: **18** (Normal, branch level 1).
- `build_branch_table` raises on overflow. **No silent truncation**, ever.

## 4. Server protocol (fixed layout)

| op | meaning |
|---|---|
| 0 `ENCODE` | observation → cached `EncodedState`; encoder runs once |
| 1 `STEP` | score cached state against one level's candidates, sample, advance GRU |
| 2 `RESET` | drop the request's cache entry |

Fail-closed conditions (all tested): unknown request key, wrong `branch_level`
out of order, wrong `policy_version`. In each case the server returns
`ERR_NO_CACHE` and never reuses stale state.

## 5. Gradient scope

Entropy and `KL(U||π)` are computed with `log K` normalization and remain
attached to the graph; gradients are finite and nonzero. **No PPO loss and no
optimizer are implemented in P1** — that is P4's contract.

## 6. Known limitations

1. The model is **randomly initialized** with a fixed seed (`20261006`), not a
   production checkpoint. Comparisons are candidate-vs-reference on identical
   weights. This is **not** a claim about trained strength or weight parity.
2. Reference parity is verified by **replay of stored choices**, not by
   byte-identical sampling trajectories (batch reorder changes `multinomial`
   consumption order, which the design explicitly permits).
3. The server runs on **CPU** in P1. GPU execution is not claimed.
4. Only a small set of real fixtures was exercised; this is not full game
   coverage (that is P2).
