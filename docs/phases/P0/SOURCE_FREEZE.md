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
| Local location | `/home/pa4/projects/_pa3_ref` (separate directory, read-only intent) |

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
- **PA3 was not modified** — measured with an explicit `-C` so the command cannot silently
  run against PA4:

  ```
  $ git -C /home/pa4/projects/_pa3_ref rev-parse HEAD
  65bc980abd8e0bbec14158b9ddc2884bb1726a4e
  $ git -C /home/pa4/projects/_pa3_ref rev-parse origin/main
  65bc980abd8e0bbec14158b9ddc2884bb1726a4e     # HEAD == remote pin
  $ git -C /home/pa4/projects/_pa3_ref status --porcelain | wc -l
  0                                             # working tree clean
  $ git -C /home/pa4/projects/_pa3_ref reflog | wc -l
  1                                             # clone entry only, no local commits
  $ git -C /home/pa4/projects/_pa3_ref diff --stat origin/main HEAD | wc -l
  0                                             # no divergence from the pinned ref
  ```

  **Clone provenance note:** the reference is a **single-branch shallow clone**
  (`--depth 1 --single-branch`). `origin/main` is therefore a *clone-time* ref, not
  a live upstream check; it is not evidence that `main` has not advanced since.
  To confirm today, run `git ls-remote <uri> refs/heads/main`.

  Performed at environment-migration time (2026-10-21):

  ```
  $ git ls-remote https://github.com/mwl313/PokeAgent3.0 refs/heads/main
  65bc980abd8e0bbec14158b9ddc2884bb1726a4e	refs/heads/main
  ```

  Upstream `main` still equals the pinned SHA, so the pin is current as of that check.

  Baseline for "unmodified" is the freshly cloned `origin/main` at
  `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`; every check above compares against it.

- **PA4 isolation** — measured:
  - PA4 root `/home/pa4/projects/PokeAgent4.0` and PA3 reference
    `/home/pa4/projects/_pa3_ref` are distinct real paths.
  - `sys.path` inside the PA4 venv contains **no** PA3 entry (`NO_PA3_ON_SYS_PATH`).
  - The real PA3 package names are not importable from the PA4 venv:
    `pa3_engine` → `ModuleNotFoundError: No module named 'pa3_engine'`;
    `pa3_observation` → same; `agent` → same.
  - No PA4 source/config file references a PA3 local path; only documentation mentions the
    upstream URL.

  **Isolation caveat:** an `ImportError`/`ModuleNotFoundError` proves the module is not
  resolvable in this venv; it does not prove the absence of a vendored copy under a
  different name, nor does it inspect symlinks. No such copy exists in the committed tree
  (18 tracked files at the amendment commit, listed in `git ls-files`), but this check is name-based, not a
  sandbox guarantee.

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

Constants, with exact file:line:

| Constant | Value | Location |
|---|---|---|
| `OBSERVATION_TOKENS` | `96` | `engine/src/knowledge.rs:7` |
| `SCHEMA_VERSION` | `2` | `engine/src/observation.rs:34` |
| `CONTRACT_HASH` | `dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35` | `engine/src/observation.rs:35` |
| `CATEGORY_COUNT` | `32` | `engine/src/observation.rs:37` |
| `FLOAT_COUNT` | `50` | `engine/src/observation.rs:38` |
| `FLAG_COUNT` | `40` | `engine/src/observation.rs:39` |

**Critical reconciliation — the reviewer's Finding 8 resolved.** `CATEGORY_COUNT`,
`FLOAT_COUNT` and `FLAG_COUNT` are **per-token** field counts, *not* components summed to
reach 96. `TokenFeatures` (`engine/src/observation.rs:152-159`) is:

```rust
pub struct TokenFeatures {
    pub categories:     [u16; CATEGORY_COUNT],   // 32
    pub category_known: [bool; CATEGORY_COUNT],  // 32
    pub floats:         [f32; FLOAT_COUNT],      // 50
    pub float_known:    [bool; FLOAT_COUNT],     // 50
    pub flags:          [bool; FLAG_COUNT],      // 40
    pub flag_known:     [bool; FLAG_COUNT],      // 40
}
```

So there are **96 tokens**, each carrying 32+50+40 fields (times two, counting the `*_known`
mask columns). Per-token token-payload bytes
(`32*2 + 32 + 50*4 + 50 + 40 + 40 = 426`) plus the `5*2 = 10` byte role/side block gives
**436 bytes/token**. The full fixed blob, following `engine/src/python.rs:27-36` verbatim:

```
OBSERVATION_FIXED_BYTES = 2              // schema_version (u16)            -> put_u16
                        + 32             // CONTRACT_BYTES hash             -> extend_from_slice
                        + 96 * 1         // token_mask: one presence byte per token
                        + 96 * 32 * 2    // categories (u16 x 32)           -> pack_tokens
                        + 96 * 32        // category_known (u8 x 32)
                        + 96 * 50 * 4    // floats (f32 x 50)
                        + 96 * 50        // float_known (u8 x 50)
                        + 96 * 40        // flags (u8 x 40)
                        + 96 * 40        // flag_known (u8 x 40)
                        + 96 * 5 * 2     // 5 ragged length rows (u16 each):
                                         //   effect_repertoire/type/base_move/move_effect
                        = 41,986 bytes
```

Provenance of the two terms absent from `TokenFeatures`: `pack_fixed_into`
(all Rust citations in this document are at the pinned SHA
`65bc980abd8e0bbec14158b9ddc2884bb1726a4e`, path `engine/src/...`)
(`engine/src/python.rs:137-160`) writes `schema_version`, then `CONTRACT_BYTES`, then
`for present in view.token_mask { out.push(u8::from(present)) }` (**that is the `96*1`**),
then `pack_tokens(...)`, then five loops over `effect_ranges`, `repertoire_ranges`,
`type_ranges`, `base_move_ranges`, `move_effect_ranges`, each `put_u16(len)` (**that is the
`96*5*2`**). `TokenFeatures` at `observation.rs:152-159` covers only the middle block.

(Recomputed independently: token payload `96 * 426 = 40,896`; plus `2 + 32 + 96 + 960 =
1,090` → `41,986`. This matches the constant's literal expression.)

**A "token" counts 96 entity slots; category/float/flag counts are per slot.** P1 must not
treat 96 as `32 + 50 + 40 + …`.

Packing (from `engine/src/python.rs`):

- Little-endian (`put_u16`, `put_f32`); one packed blob **per view** (per side).
- Explicit `category_known` / `float_known` / `flag_known` masks, so hidden opponent
  information is represented as *unknown* rather than fabricated or zero-filled silently.
- Layout mirrored by `engine/python/pa3_observation.py`.

Action-side constant: `CANDIDATE_PADDING = 64` (`engine/src/actions.rs:10`).

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
