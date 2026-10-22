# PA4 Project Status

**Last updated:** 2026-10-22
**Current phase:** P3.5 — Observation Contract Repair (**correction package complete; re-audit pending**)
**Current verdict:** **P3.5 implementation PASS**, fresh-context audit **BLOCKED** (round-8 findings fixed; round-9 re-audit pending)
**Last closed phase:** P3 **PASS** (receipt preserved unchanged)
**Next phase:** P4 — not started; blocked on the P3.5 auditor gate

| Phase | Verdict | Evidence |
|---|---|---|
| **P0** Discovery / environment freeze | **PASS** (4/4) | `docs/phases/P0/{ENVIRONMENT.md,COMPATIBILITY.md,SOURCE_FREEZE.md}` |
| **P1** Conditional action feasibility | **PASS** (G1-D CPU-scoped) | `docs/phases/P1/RECEIPT.json` |
| **P2** Rust ↔ TorchRL environment adapter | **PASS** (5/5) | `docs/phases/P2/{FINAL_REPORT.md,RECEIPT.json}` |
| **P3** Batched collector / central inference | **PASS** (5/5) | `docs/phases/P3/{FINAL_REPORT.md,RECEIPT.json,PERFORMANCE_BASELINE.csv}` |
| **P3.5** Observation contract repair | **Implementation PASS; re-audit pending** | `docs/phases/P3.5/{FINAL_REPORT.md,RECEIPT.json,ADR_009.md}` |

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

**Status: implementation PASS; P35_I BLOCKED pending round-9 audit.** The
round-8 fresh-context auditor read the real repo and returned `FINDINGS`; the
four MAJOR findings (vacuous identity gate, mislabelled `next_observation`, no
server-seam ragged test, stale provenance/duplicate key) and F5–F9 were fixed
at `b0e2513` with tests that fail if the fix is reverted.

- **189 tests green:** P1 39 + P2 51 + collector 99; `check_env_specs` OK.
- Transport byte-exact on **960** real views, encoder/value parity on CUDA
  **0.000e+00** over **480** views, **2264/2264** negative controls detected.
- Real server seam with **non-empty ragged** passes; disabling the forwarding
  makes it fail (both runs retained in `evidence/f3_ragged_forwarding_*.txt`).
- Collector smoke: 4 natural matches, **111 seat records, all `ragged.v2`, 0 drops**.
- Fresh same-session A/B, two pairs: **−26.0%/−23.0%** and **−17.5%/−22.4%**
  (16/32 envs); `views` (8.56 ms/step) is only leaf extraction and the gate,
  not the repair cost (`payload_to_transport` runs inside `env_step`/`reset`).
- PA3 untouched (0 dirty), `uv.lock` unchanged.
- **P35_I remains BLOCKED** until the round-9 fresh-context audit returns CLEAN.

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

**P3.5 re-audit.** Do not start P4. The round-9 fresh-context `pa4-auditor`
must read the corrected tree (`06e3b04` + this metadata commit) and return
CLEAN before P35_I can be set PASS. P4 (GAE, PPO-2, optimizer, checkpoint
parity) is not started. Audit history is in `docs/phases/P3.5/RECEIPT.json`
and `docs/phases/P3/RECEIPT.json`.

## Artifacts

- `docs/master/PA4_MASTER_ARCHITECTURE.md` (authoritative master, v1.2)
- `docs/phases/P0/` — environment, compatibility, source freeze, receipt
- `docs/phases/P1/` — action distribution design, receipt, final report
- `docs/phases/P2/` — design, TensorDict spec, episode trace report, final report, receipt
- `docs/phases/P3/` — design, ADR-008, integration decision, batching evidence, transition spec, baseline CSV + stage timing, final report, receipt
