# PA4 Project Status

**Last updated:** 2026-10-22
**Current phase:** P3 — Batched Collector and Central Inference (**open**)
**Current verdict:** **P3 PARTIAL** (G3-A–G3-D scoped PASS, G3-E not passed; audit findings open)
**Last closed phase:** P2 **PASS** (5/5)
**Next phase:** P4 — not started

| Phase | Verdict | Evidence |
|---|---|---|
| **P0** Discovery / environment freeze | **PASS** (4/4) | `docs/phases/P0/{ENVIRONMENT.md,COMPATIBILITY.md,SOURCE_FREEZE.md}` |
| **P1** Conditional action feasibility | **PASS** (G1-D CPU-scoped) | `docs/phases/P1/RECEIPT.json` |
| **P2** Rust ↔ TorchRL environment adapter | **PASS** (5/5) | `docs/phases/P2/{FINAL_REPORT.md,RECEIPT.json}` |
| **P3** Batched collector / central inference | **PARTIAL** (open) | `docs/phases/P3/{FINAL_REPORT.md,RECEIPT.json}` |

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
`136d333` returned `VERDICT: CLEAN`.

The auditor is the **same model** as the Lead. This is a process control and a
second reading of real files, **not** independent-model verification.


## P3 — Batched Collector and Central Inference (open)

**Status: PARTIAL.** G3-A–G3-D are supported in the scoped sense; G3-E did **not**
pass. The earlier PARTIAL reason (auditor could not read WSL) is resolved — the
auditor now reads the repo — and the review it produced found **unmet design
deliverables**, not just missing access.

- Tested code: `297cf8b7048c6ec1f1841fb00b443a868963520b`
- Real CUDA batching **per driver** is proven: one model forward covering 4
  independent requests, `cuda:0`; real runs reached `bmax` 63/64. The receipt
  reports per-run means 3.78/5.15/9.16.
- Measured: 5.1–6.0 natural matches/s single driver; 2/4 drivers 4.4/5.5
  matches/s. The multi-driver comparison is not controlled (startup is inside
  the multi-driver wall clock, outside the single-driver one).
- 136 tests green: P1 39 + P2 51 + P3 46.
- Key finding: `ProcessInferenceServer.client()` caches one client, so
  `[srv.client() …]` aliased slot 0 and every forward was B=1; `srv.clients(n)`
  returns distinct slots.

**Unmet or unproven (auditor FINDINGS, `docs/phases/P3/RECEIPT.json`):**
1. **P3-C transition retention is absent.** `PA4ActorTransitionV1` pre-action
   observations, ordered candidate tables, weight fingerprints and a bounded
   queue are documented but never retained on the tested path; seat records are
   built and discarded after counting.
2. **"Central" inference is per-driver.** Each driver owns its own
   `InferenceServer`; no forward ever carries requests from two worker
   processes, contrary to P3-B's "different workers" wording.
3. **No weight fingerprint / observation-contract hash** on P3 exchanges.
4. **Validation gaps:** duplicate identity within one transport batch overwrites
   silently; a foreign `policy_version` is accepted on ENCODE.
5. **No independent lost-completion guard** (`env.completed_matches` is never
   compared with the ledger).
6. **Revisions are counted as behaviour rows.**
7. **P3-D timing gaps:** no cold/steady split, no retained stage breakdown,
   several headline numbers have no artifact, and the "~40 ms" residual is an
   unmeasured inference.
8. `run_collector.py` still uses the aliasing `srv.client()` pattern and is
   untested.
9. `INTEGRATION_DECISION.md` still records ADR-008 as proposed/blocked.

## Open limitations (carried forward, not blocking)

1. Auditor independence is same-model only.
2. PA3 pin is the public `main` tip, not verified live production state.
3. P1 inference server is CPU-only; GPU batching is unproven.
4. P1 replay needs the original observation payload.
5. P2 covers 4 seeded games; native truncation was never observed (accounting is
   proven with a synthetic native result).
6. Oracle shares the PA4 decoder path, so decoder bugs would be invisible to it.
7. Minor P2 test-coverage edges are listed in `docs/phases/P2/RECEIPT.json`
   (`limitations`); none affects a gate.

## Immediate next task

**Do not start P4.** P3 remains open: either implement the unmet P3-C/P3-D
deliverables listed above, or record an explicit scope amendment. Both require
user authorisation. The P3 audit findings, in severity order, are in
`docs/phases/P3/RECEIPT.json` under `audit_findings`.

## Artifacts

- `docs/master/PA4_MASTER_ARCHITECTURE.md` (authoritative master, v1.2)
- `docs/phases/P0/` — environment, compatibility, source freeze, receipt
- `docs/phases/P1/` — action distribution design, receipt, final report
- `docs/phases/P2/` — design, TensorDict spec, episode trace report, final report, receipt
