# P4-0 — Preflight, implementation audit and input fidelity

**Phase:** P4
**Design:** `PA4-P4-DESIGN-v1.0` (see `docs/phases/P4/DESIGN.md`, SHA256 `5d54553d…`)
**Recorded:** 2026-10-22
**Entry HEAD:** `ec0d7e2` (P3.5 PASS, pushed)

---

## 1. Repository and environment state

| Item | Recorded value |
|---|---|
| Workspace | `/home/pa4/projects/PokeAgent4.0` |
| Branch | `main` |
| HEAD at P4-0 | `ec0d7e2` |
| Worktree | clean (only `docs/phases/P4/` added by this preflight) |
| `uv.lock` SHA256 | `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` |
| `uv lock --check` | passes ("Resolved 37 packages") |
| PA3 reference SHA | `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` |
| PA3 dirty files | 0 |
| Python / torch / TorchRL / TensorDict | 3.12.3 / 2.9.1+cu128 / 0.14.1 / 0.14.3 |
| GPU | RTX 4060 Ti 8 GB |
| `scripts/p0_smoke.py` | `P0_TENSORDICT_2X4_PASS`, `P0_CUDA_8X8_PASS`, `P0_TORCHRL_API_IMPORT_PASS` |

## 2. Inherited test baseline — actual counts

The prompt states the inherited baseline is **158** (39 P1 + 51 P2 + 68 P3). The
actual inherited count is **193**, because P3.5 added 35 tests when repairing the
observation contract:

| Suite | Tests | Result |
|---|---|---|
| `tests/action_distribution` (P1) | 39 | OK |
| `tests/environment` (P2) | 51 | OK |
| `tests/collector` (P3 + P3.5) | **103** | OK |
| **Total** | **193** | **OK** |

P4 gates are evaluated against **193**, not 158. The design's G4-E wording
("158 inherited tests still green, or explain changed actual count") is satisfied
by this explanation: P3.5 is an approved phase that added tests, and no inherited
test was removed or weakened.

## 3. P3 PASS interpretation

P3 is recorded PASS with G3-A scoped under ADR-008. Per design §0.1 the user-approved
ADR-008 remains binding: the stock `AsyncBatchedCollector` is **not** required as
the outer loop, and must not be reintroduced merely to satisfy a generic master
description.

**P3 PASS is not any PPO training proof.** P3 recorded
`accepted_training_matches_per_sec = null`; no value head, GAE, PPO loss,
optimizer or checkpoint exists in the inherited tree.

## 4. Audit A — how does the actor obtain its weights?

**Finding: the actor is hardcoded to seed-initialized weights.**

- `pa4/actions/batched_scorer.py:316-330` `_ensure_model()` constructs
  `PA3Model(PA3Config(initialization="random", seed=self._seed))` and there is
  **no `load_state_dict` anywhere in the class or the factory path**.
- `pa4/actions/batched_factory.py:10-14` passes only `PA4_POLICY_SEED`,
  `PA4_POLICY_VERSION` and the device.

**Consequence if left unfixed:** all 20 P4 iterations would collect with the same
seed-1 weights, the fingerprint would never change, and the "on-policy" loop
would silently be a single frozen random policy with a learner attached. This is
precisely the failure the prompt warns about (P4 prompt §1 bullet 4).

**P4 requirement:** the frozen inference service must be able to load the
**current committed model snapshot** every iteration, and P4 must prove a real
fingerprint/version change with new weights actually served. This is addressed in
P4-C (service recreation at each committed version).

## 5. Audit B — observation and ragged feature fidelity (G4-B blocker check)

**Finding: the ragged gap is already closed by P3.5, and the actor/learner input
contract is now reconstructible.**

P3.5 (ADR-009) repaired the observation contract:

- `pa4/actions/batched_scorer.py` carries ragged in `request_spec()`
  (`_ragged_spec_rows`, line 78+) and forwards it through `_ragged_from_td`
  into `ObservationBatch.ragged`, failing closed on any missing leaf.
- `pa4/collector/transitions.py` retains `observation_ragged` and
  `observation_representation` per seat record (lines 132-133, 171-172).
- All five PA3 families (`effects`, `repertoire`, `types`, `base_moves`,
  `move_effects`) are measured non-empty in real battles (P3.5
  `capacity_survey.json`, 24,000 seat-views).

**Therefore the exact observation the real P3/P4 actor consumes can be
reconstructed by the learner**, including ragged. There is no missing required
policy-input field, so **G4-B is not blocked by an input-contract gap**.

Two residual obligations for P4 are recorded here so they are not lost:

1. **Only `pa4.observation.ragged.v2` may be trained on.** Legacy `fixed9.v1`
   records must be refused at the *learner ingestion boundary* (not merely in the
   collector). `ragged_transport.assert_representation` exists and must be called
   by P4 ingest.
2. **Replay fidelity must be re-proven end-to-end at the learner**, not inferred
   from P3.5's collector-side proof: the P4-B parity harness must rebuild
   `ObservationBatch` from a *stored transition record* (including ragged) and
   reproduce the stored FP32 joint logprob within `<=1e-5` under frozen weights.

## 6. Audit C — the lossy transition store

**Finding: confirmed lossy, and confirmed to be the default.**

- `pa4/collector/transitions.py:30-47` `BoundedTransitionStore.put()` does
  `self._q.popleft()` when full and increments `dropped`. It is a ring.
- `pa4/collector/run_batched.py:82` silently defaults to
  `BoundedTransitionStore(cfg.transition_capacity)` when no store is passed.
- P3's own baseline recorded large drop counts (e.g. 752 dropped of 1,008 stored
  in the 16-env run).

**P4 requirement (P4-A):** a separate **training-only** lossless sink with
blocking backpressure or a bounded spool that fails the iteration rather than
evicting. P3's benchmark ring semantics must remain unchanged for P3
reproducibility.

## 7. Files inspected for the action/record path

| File | Role |
|---|---|
| `pa4/collector/run_batched.py` | P3 actor loop, staged inference, match accounting |
| `pa4/collector/transitions.py` | `PA4ActorTransitionV1` records and the bounded ring |
| `pa4/collector/batched_core.py` | `StagedInference`, `BatchedConfig`, report |
| `pa4/actions/batched_scorer.py` | server model, request/response spec, ragged forward |
| `pa4/actions/conditional_policy.py` | conditional scorer, `weight_fingerprint` |
| `pa4/env/native_env.py` | EnvBase, two-seat barrier, ragged observation assembly |
| `pa4/env/ragged_transport.py` | P3.5 ragged transport + representation identity |

## 8. Conclusions carried into P4-A..E

1. Entry HEAD `ec0d7e2`; 193 inherited tests green; lock and PA3 frozen.
2. **Observation contract is complete** (ragged v2). G4-B is not input-blocked,
   but must be re-proven at the learner on stored records.
3. **Actor weight loading must be added** — the current path cannot serve an
   updated snapshot.
4. **Lossless training ingestion must be added** — the existing store evicts.
5. Legacy `fixed9.v1` records must be rejected at learner ingestion.
