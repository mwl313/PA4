# P0-D — PokeAgent 3.0 Reference Freeze & P1 Handoff

Document ID: `PA4-P0-SOURCE-FREEZE-v1.0`
Scope: P0-D. Read-only source metadata and contract facts required by P1.
No PA3 code is migrated, no checkpoints downloaded, no fixtures generated.

---

## 1. Pinned reference

| Item | Value |
|---|---|
| Source URI | `https://github.com/mwl313/PokeAgent3.0` |
| Default branch | `main` |
| **Pinned commit SHA** | **`65bc980abd8e0bbec14158b9ddc2884bb1726a4e`** |
| Commit date | 2026-10-17 19:20:41 +0900 |
| Commit subject | `docs(ops): record the checkpoint team-stats path dependency that breaks resume` |
| Repository pushed_at | 2026-10-17T10:29:51Z |
| Acquired as | `git clone --depth 1 --branch main --single-branch`, then verified `git rev-parse HEAD` |
| Local location | `/root/projects/_pa3_ref/pa3` (separate directory, read-only intent) |

### Limitation — stated explicitly

- This SHA is the **current `main` tip** of the public repository at freeze time. It is
  **not** claimed to be the live production training state. PA3's own
  `docs/TRAINING_SOURCE_OF_TRUTH.md` records that production was **stopped** on
  2026-10-16 pending a precision decision, and points at a delivery branch
  (`codex/v2-secondary-precision-stability`) and a tested executable
  `9d3d6051d6d17cc517259a3e6807bb067c717a9e`. That branch and executable were **not**
  independently verified by PA4 and are **not** substituted for the pinned SHA.
- No active PA3 production machine was accessed. The user's live training sources and
  checkpoints were not inspected in P0.
- **PA3 was not modified.** The clone was not edited; `git status --porcelain` in the
  reference clone was empty (clean) after inspection. PA4 wrote no artifacts into PA3.

---

## 2. Contract inventory (read-only facts)

Files inspected are limited to what P1 needs. Hashes are `sha256` of the file at the
pinned SHA.

| Contract | Source path | sha256 |
|---|---|---|
| Engine entry / module map | `engine/src/lib.rs` | `890de6351ab82397e6c7a842a98c091a050faad2af9f4fc137ed39a6eb31e38d` |
| Action schema | `engine/src/actions.rs` | `fe396c19946b11872fe57a8e437905e50cd977a3dcc5e5215cbd3cda0f02b241` |
| Observation encoding | `engine/src/observation.rs` | `edd53f78a8f3fe518407878b4a8935eaa54bd2110db10e945c5ccd67cce6d000` |
| Token count constant | `engine/src/knowledge.rs` | `ba00304cfde701bf414215457350deaba7640500dbef2f6dc73836c9b7f589e2` |
| Batch transition API | `engine/src/batch.rs` | `e33c72b70bde6fee4e45a941e90a44e49f5a39d1a224e95272ab93ef93231d03` |
| PyO3 binding | `engine/src/python.rs` | `56b5bdd48dc9121991fbf07a18cac417c3bca3e3efc5701f84797335579853e3` |
| Conditional sampler / joint logprob | `agent/model/pa3_model.py` | `9f314425f0679fe031b566f356663c5c6fdaf191c08d4c21dd5ed41dab357dd3` |
| Masked log-softmax scoring | `agent/model/scorer.py` | `2c3e3792f3c0fd88195b252c16ec9e248bc97a49c121acfcc36164b2ee505cab` |
| Precision contract | `agent/model/precision.py` | `fb1eaacb1ee3830e31904ed715479772f5f1ebc42a1c321c260bc6a75bceeb51` |
| PPO loss | `agent/ppo/losses.py` | `8d10e5f5eef5bcc2f6f052f9dfc26b5957d12a5cc73b741dbec003b47e4b40be` |
| GAE | `agent/ppo/gae.py` | `5d97b90904e2d0eb94a936dd93a72fbaebb9b93ff982fe405dff1aef2047910b` |
| Learner loop | `agent/ppo/learner.py` | `134a62caf890e044c0788eaedc0c5fa3b421e9fe8d55eaf2432041aae716223c` |
| Training source of truth | `docs/TRAINING_SOURCE_OF_TRUTH.md` | `36189ca03365e9b68d448d8b9d5c5a291c602b9b1e9cd061e2787ef3fe4eabc3` |

Crate facts: workspace members `["engine"]`; package `pa3-engine` v0.1.0, `edition = 2024`,
`rust-version = "1.90"`, lib `pa3_engine` (`rlib` + `cdylib`), feature `python` enables
`pyo3 0.26` (extension-module; optional). Pinned Rust toolchain: `1.90.0`
(`rust-toolchain.toml`, components rustfmt + clippy).
Engine format constant: `FORMAT = "gen9championsvgc2026regmc"`.

---

## 3. Contract facts for P1

### 3.1 Engine

- `engine/src/lib.rs` declares modules: `actions, assets, batch, battle, damage, effects,
  knowledge, legality, observation, python (feature-gated), queue, rng, state, stats`.
- The native batch API lives in `engine/src/batch.rs` (`BattleBatch`): `reset_batch`,
  `reset_slots`, `request(handle, side)`, `observe_batch`, `observe_encoded_batch_into`,
  `step_batch`, `step_batch_into`, `snapshot`, `restore`, `enable_trace`, `export_trace`.
  Handles are opaque `(slot, generation)` pairs from `reset_batch`.
- `legality.rs` is **team legality** (`validate_team`, `Verdict`, `LegalityReport`), not
  per-action legality. Per-action legality is expressed by the `Request` candidate
  tables (see 3.3) and enforced at execution.

### 3.2 Observation — 96-token contract

- `knowledge.rs`: `OBSERVATION_TOKENS = 96`.
- `observation.rs`: `SCHEMA_VERSION = 2`, `CATEGORY_COUNT = 32`, `FLOAT_COUNT = 50`,
  `FLAG_COUNT = 40`.
- `python.rs` defines `OBSERVATION_FIXED_BYTES` from those constants and packs
  little-endian (`put_u16`/`put_f32`), one packed blob per **view** (per side), with
  explicit `category_known` / `float_known` / `flag_known` masks. Hidden opponent
  information is therefore represented as *unknown* rather than fabricated.
- Layout is mirrored by `engine/python/pa3_observation.py`.

### 3.3 Action — prefix-dependent candidates

- `actions.rs`: `CANDIDATE_PADDING = 64`, `NO_SLOT = 0xFF`.
  - `RequestKind = {Preview, Normal, Replacement, Wait, Finished}`
  - `ActionKind = {Pick, Move, Switch, Pass}`
  - `Resource = {None, Mega}`
  - `AtomicAction { kind, own_slot, move_slot, target_location, switch_destination, resource }`
  - `Request { kind, slots: [SlotRequest; 2], bench, revive_targets, preview_roster }`
- A complete **player** request is a `Request` covering both of that player's active
  slots; `JoinAction` in PA4 terms = one complete player request.
- `SlotRequest` carries per-slot legality inputs (`moves`, `trapped`, `can_mega`,
  `locked_move`, `locked_recharge`, `last_active`, `reviving`, ...), which is why later
  candidate tables depend on earlier picks.

### 3.4 Probability — conditional joint logprob in FP32

- `python.rs`: `Branch = Vec<ActionTuple>` (an **action prefix**);
  `EnvSpec = (u32, u32, Vec<(u8, Branch)>)` submits per-side branches.
- `scorer.py`: `LOGPROB_FLOOR = torch.finfo(torch.float32).min`;
  `masked_log_softmax` applies the legal mask **before** any log-prob math, so illegal
  candidates get `LOGPROB_FLOOR` (exactly zero probability after masking).
- `pa3_model.py` `SamplingResult`: `selected [B,K]`, `logprob_selected [B,K]`,
  **`request_logprob [B]` computed as `logprob_selected.sum(dim=-1)`**, `branch_k [B,K]`,
  `actor_active [B] bool`, `probabilities [B,K,P]`.
- `actor_active = (branch_k >= 2).any(dim=-1)`: a request only produces an actor row when
  at least one branch actually had ≥2 real candidates.
- `LevelSamplingResult` documents that the engine's mask for branch *j* is computed
  **after** the branches selected before it ("preview picks cannot repeat a member, a
  used Mega flag disappears from the second slot"), so a faithful collector **cannot**
  pre-build all branch tables in one pass.

### 3.5 Learning — PPO-2 and masks

- `losses.py`: `policy_loss = -mean(min(ratio*A, clamp(ratio, 1±eps)*A))` over actor rows;
  `value_loss = 0.5 * MSE(value, return)` with **no value clipping**;
  `approx_kl = ((ratio - 1) - log(ratio.clamp_min(1e-12))).mean()`.
- `gae.py`: `compute_gae(gamma=1.0, gae_lambda=0.95)`, with
  `delta = r + gamma*next_V - V`, `next_adv = delta + gamma*lambda*(0 if terminal else 1)*next_adv`,
  plus `compute_gae_for_rows` which groups per side and orders by `request_index`.
- `learner.py`: `for epoch in range(self.config.ppo_epochs)` — PPO-2 means two epochs;
  row-masked means (`_row_masked_mean`) separate actor rows from value rows.
- Two-seat accounting is handled outside the loss (per-side sequences), consistent with
  "both seats learn; one natural battle counted once".

### 3.6 Precision reference

- `precision.py` defines versioned `PrecisionContract`s (modes A…E) with
  `identity = f"pa3-precision-{mode}-v{1 or 2}"`, controlling observation quantization,
  encoder/scorer/GRU AMP and `ragged_effects_fp32`.
- Baseline for PA4 P1/P4 comparison is the **FP32 reference**: masks and log-probs are
  computed and reduced in FP32, `LOGPROB_FLOOR` is an FP32 value.
- PA3's production candidate profile at 2026-10-16 recorded `precision.mode = "E"` with
  `fp16-reduction-off`. PA4 must **not** adopt that as the reference; P0 fixes FP32 and
  no tolerance may be relaxed to make a parity test pass.

---

## 4. P1 Handoff — exactly three questions

1. **Can the TorchRL policy server sample a complete legal conditional player action
   without changing the Rust legality rules?**
   (i.e. can `InferenceServer` + a custom distribution over the full conditional request
   produce a complete `Request` whose every branch is legal, while the Rust engine
   remains the sole authority on legality?)

2. **Is a single batched request feasible, or is a staged prefix protocol required?**
   (Given `Branch = Vec<ActionTuple>` and the documented prefix-dependence of branch
   masks, does one request carry all branch candidate tables, or must sampling proceed
   prefix-by-prefix with batched round trips?)

3. **What minimum branch metadata is necessary for exact behavior-logprob
   recomputation?**
   (Given `joint_logprob = sum(branch logprob)` over valid branches and
   `actor_active = (branch_k >= 2).any()`, what exactly must be persisted — candidate
   ids/masks per branch, `branch_k`, padding capacity, policy version, observation
   contract hash — so the stored logprob can be exactly recomputed?)

**No solution to these questions is implemented in P0.**
