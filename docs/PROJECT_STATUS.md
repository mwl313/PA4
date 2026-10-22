# PA4 Project Status

**Last updated:** 2026-10-22
**Current phase:** P4 — Synchronous On-Policy PPO-2 (**closed, fresh-process recovery verified**)
**Current verdict:** **P4 PASS** (G4-A..G4-E; two audit rounds closed; G4-D proven across a process restart)
**Last closed phase:** P4 **PASS** (prior receipts preserved unchanged)
**Next phase:** C1 — not started (requires separate authorization; nothing provisioned)

| Phase | Verdict | Evidence |
|---|---|---|
| **P0** Discovery / environment freeze | **PASS** (4/4) | `docs/phases/P0/{ENVIRONMENT.md,COMPATIBILITY.md,SOURCE_FREEZE.md}` |
| **P1** Conditional action feasibility | **PASS** (G1-D CPU-scoped) | `docs/phases/P1/RECEIPT.json` |
| **P2** Rust ↔ TorchRL environment adapter | **PASS** (5/5) | `docs/phases/P2/{FINAL_REPORT.md,RECEIPT.json}` |
| **P3** Batched collector / central inference | **PASS** (5/5) | `docs/phases/P3/{FINAL_REPORT.md,RECEIPT.json,PERFORMANCE_BASELINE.csv}` |
| **P3.5** Observation contract repair | **PASS** (9/9) | `docs/phases/P3.5/{FINAL_REPORT.md,RECEIPT.json,ADR_009.md}` |
| **P4** Synchronous on-policy PPO-2 | **PASS** (G4-A..E) | `docs/phases/P4/{FINAL_REPORT.md,RECEIPT.json,FULL_STACK_BASELINE.csv}` |

PASS establishes the phase contracts only. It does not prove Collector
scalability, PPO stability, agent strength or throughput.

## Environment (WSL2)

- **Distro:** `Ubuntu-24.04`, user `pa4`, repo `/home/pa4/projects/PokeAgent4.0` (ext4)
- **Remote:** `https://github.com/mwl313/PA4.git` (origin)
- **Stack:** Python 3.12 / torch `2.9.1+cu128` / torchrl `0.14.1` / tensordict `0.14.3` / Rust `1.90.0`
- **Lock:** `uv.lock` sha256 `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` (unchanged since P0)
- **PA3 reference:** `https://github.com/mwl313/PokeAgent3.0` @ `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`, worktree clean, read-only
- **Agents:** `pa4-lead` (primary, writable) + `pa4-auditor` (subagent, read-only)

## P2 outcome

- `PA4EnvBase(EnvBase)` over one native `NativeEngine` with per-row battle slots.
- Real `check_env_specs succeeded!`; fixed TensorDict contract in `TENSORDICT_SPEC.md`.
- 4 distinct seeded natural games (20/11/19/14 steps, winners `0/1/0/1`), rewards
  match the native winner in both directions; 0 operational errors, 0 illegal
  submissions, 0 revisions, 0 duplicate completions.
- Independent second-engine oracle: zero parity mismatches over 4 seeds.
- P1 `ConditionalPolicy` packet crosses `EnvBase.step()` and matches a direct
  native `step_batch`; a full battle is driven by P1 policy decisions.
- **51 environment tests OK**, **39 P1 regression tests OK**.

## Review status

`pa4-auditor` now runs from WSL and can read the actual repository. Access was
proven with a literal file read (`ACCESS_OK`) before the review. Five review
rounds are recorded in `docs/phases/P2/RECEIPT.json`; the final round at
`136d333` returned `VERDICT: CLEAN`. Seven P3 rounds are recorded in
`docs/phases/P3/RECEIPT.json`; round 7 returned `CLEAN` after the stale-value
fixes.

The auditor is the **same model** as the Lead. This is a process control and a
second reading of real files, **not** independent-model verification.


## P3 — Batched Collector and Central Inference (closed)

**Status: PASS (5/5).** G3-A is a scoped PASS under the user-approved ADR-008
(the stock `AsyncBatchedCollector` is not used). G3-E closed on the round-7
read-only audit, which returned `CLEAN` after the stale-wording fixes below.
The evidence is in this repository; the auditor is the same DeepSeek model in a
fresh context, so this is a process control, not independent-model proof.

- Tested code: `13395df0e731cdc93996d63306c74703257c1dd6`; baseline evidence
  regenerated at `5c9096a` by the committed generator.
- **158 tests green:** P1 39 + P2 51 + P3 68 (re-run on the final tree).
- Real CUDA batching, both topologies:
  - per driver: one forward covering 4 independent rows, `cuda:0`, bmax 64;
  - **central**: one parent-owned `InferenceServer` (`transport="distributed"`)
    serves 2 and 4 spawned drivers; 233/180 forwards mixed 2/4 worker
    processes.
- P3-C **bounded `PA4ActorTransitionV1` retention** implemented: pre-action
  observations, ordered candidates, selected indices, FP32 logprobs, the real
  packet, native outcome, match id, policy version and sha256 weight
  fingerprint; drops are counted.
- Measured throughput (steady / wall): single driver 7.9–9.7 / 5.1–5.5
  matches/s; 2 drivers central 0.81 / 1.24, per-driver 6.19 / 5.70; 4 drivers
  central 0.32 / 1.05, per-driver 3.27 / 7.19. For multi-driver runs the steady
  figure is the mean of per-driver steady rates (drivers run concurrently).
- **Finding:** central cross-process inference is correct but 4.60–6.85× slower
  on the wall measure (and 7.64–10.22× on the per-driver steady mean) than
  per-driver servers on this host, with both topologies using the same server
  batching policy. Per-match encode and staged forward time both rise; the
  split between transport and service/CPU contention is not isolated. A
  transport-selection input for P6/P7.
- Key earlier finding retained: `ProcessInferenceServer.client()` caches one
  client; `srv.clients(n)` returns distinct slots.
- Addressed from the audits: transition retention, central cross-worker batching,
  weight/observation fingerprints, duplicate-in-batch and foreign-version
  rejection, lost-completion guard, revision filtering, cold/steady + stage
  timing, retained artifacts, a generator that reproduces every retained
  baseline artifact, removal of the untested legacy collector, stale docs.
- The P3-C record now matches the design schema: `schema_version`,
  `next_observation`, per-level legal mask/token refs, observation digest and
  per-seat policy provenance.

## P3.5 — Observation Contract Repair (correction package applied; re-audit pending)

**Status: PASS (9/9). Closed; superseded by P4.** Historical note — the
round-8 fresh-context auditor read the real repo and returned `FINDINGS`; the
four MAJOR findings (vacuous identity gate, mislabelled `next_observation`, no
server-seam ragged test, stale provenance/duplicate key) and F5–F9 were fixed
at `e6da02f` with tests that fail if the fix is reverted.

- **193 tests green:** P1 39 + P2 51 + collector 103; `check_env_specs` OK.
- Transport byte-exact on **960** real views, encoder/value parity on CUDA
  **0.000e+00** over **480** views, **2264/2264** negative controls detected.
- Real server seam with **non-empty ragged** passes; disabling the forwarding
  makes it fail (both runs retained in `evidence/f3_ragged_forwarding_*.txt`).
- Collector smoke: 4 natural matches, **102 seat records, all `ragged.v2`, 0 drops, no collector errors**.
- Fresh same-session A/B, three pairs: observed deltas **−26.0% to −10.6%**
  (16 envs) and **−31.5% to −22.4%** (32 envs); `views` (6.36 ms/step) is only
  leaf extraction and the gate, not the repair cost (`payload_to_transport`
  runs inside `env_step`/`reset`).
- PA3 untouched (0 dirty), `uv.lock` unchanged.
- **P35_I closed PASS** after the Windows-export audit (see ADR-009).

## Open limitations (carried forward, not blocking)

1. Auditor independence is same-model only.
2. PA3 pin is the public `main` tip, not verified live production state.
3. P1's original inference server is CPU-only; P3 proves GPU batching separately (per-driver and cross-process central).
4. P1 replay needs the original observation payload.
5. P2 covers 4 seeded games; native truncation was never observed (accounting is
   proven with a synthetic native result).
6. Oracle shares the PA4 decoder path, so decoder bugs would be invisible to it.
7. Minor P2 test-coverage edges are listed in `docs/phases/P2/RECEIPT.json`
   (`limitations`); none affects a gate.

## Immediate next task

**P4 is closed.** P3.5 closed PASS and P4 closed PASS with G4-D proven across a
process restart. The audit history for both phases is in
`docs/phases/P3.5/RECEIPT.json` and `docs/phases/P4/RECEIPT.json`.

**C1 (Runpod compatibility) is the next phase but is NOT started.** It requires
separate authorization and spends real money; nothing has been provisioned.


## Artifacts

- `docs/master/PA4_MASTER_ARCHITECTURE.md` (authoritative master, v1.2)
- `docs/phases/P0/` — environment, compatibility, source freeze, receipt
- `docs/phases/P1/` — action distribution design, receipt, final report
- `docs/phases/P2/` — design, TensorDict spec, episode trace report, final report, receipt
- `docs/phases/P3/` — design, ADR-008, integration decision, batching evidence, transition spec, baseline CSV + stage timing, final report, receipt

## P4 — Synchronous on-policy PPO-2 (closed)

**Status: PASS (G4-A..G4-E).** A strictly synchronous on-policy PPO-2 loop over
real Rust battles, with lossless ingestion, per-seat GAE, snapshot-based policy
identity, real Adam updates and atomic checkpoints.

- **272 tests green:** P1 39 + P2 51 + P3 103 + P4 79. The inherited baseline is
  193 (the P4 prompt's 158 predates the approved P3.5 phase, which added 35).
- **20-iteration trial** on the RTX 4060 Ti: 320 generated / 320 accepted natural
  matches, 0 dropped, 0 duplicates, 265 optimizer steps, 20 distinct policy
  versions, fingerprint chain intact, 0 NaN/Inf, peak VRAM 574.6 MiB,
  0.4439 accepted training matches/s.
- **Fresh-process recovery:** parent `27` → child `186`, distinct processes and a
  new CUDA context; 142/142 tensors within the predeclared FP32 bound, Adam
  delta `5.355e-09`, steps `19 = 19`, matches `33 = 33`, ledger identical.
- **Falsifiable:** dropping Adam restoration moves weights by `9.384e-05`;
  dropping the RNG blob is caught by a measured stream mismatch; dropping the
  step counter is caught by `10` vs `19`.
- **Four audit rounds** are recorded in `docs/phases/P4/RECEIPT.json`. Round 1
  found four MAJOR training-loop defects; round 2 found that the closure
  documentation cited values absent from the artifacts. Both were fixed rather
  than argued.

### Known limitations

- P4 proves a correct, restartable loop. It does **not** prove playing strength,
  convergence or cost efficiency.
- Training is **CPU-bound in Python** (~3.5% of learner time is GPU encode), so a
  faster GPU would not currently help; optimisation is deferred.
- CUDA kernels are not bitwise-reproducible; CUDA comparisons use the
  predeclared FP32 bound.
- The recovery comparison reuses pre-recorded theta_0 cohorts; both branches
  consume identical data, which is what isolates checkpoint fidelity.

**C1 (Runpod compatibility) is not started and requires separate authorization.**
No cloud resources have been used.
