# PA4 Project Status

**Last updated:** 2026-10-22
**Current phase:** P2 — Rust ↔ TorchRL Environment Adapter (**complete**)
**Current verdict:** **P2 PASS** (5/5 gates)
**Next phase:** P3 — not started, awaiting explicit authorisation

| Phase | Verdict | Evidence |
|---|---|---|
| **P0** Discovery / environment freeze | **PASS** (4/4) | `docs/phases/P0/{ENVIRONMENT.md,COMPATIBILITY.md,SOURCE_FREEZE.md}` |
| **P1** Conditional action feasibility | **PASS** (G1-D CPU-scoped) | `docs/phases/P1/RECEIPT.json` |
| **P2** Rust ↔ TorchRL environment adapter | **PASS** (5/5) | `docs/phases/P2/{FINAL_REPORT.md,RECEIPT.json}` |

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

**Do not start P3 automatically.** P3 (batched Collector + InferenceServer) and
P4 (PPO) require explicit user authorisation. Read the P2 handoff in
`docs/phases/P2/FINAL_REPORT.md` §6 first.

## Artifacts

- `docs/master/PA4_MASTER_ARCHITECTURE.md` (authoritative master, v1.2)
- `docs/phases/P0/` — environment, compatibility, source freeze, receipt
- `docs/phases/P1/` — action distribution design, receipt, final report
- `docs/phases/P2/` — design, TensorDict spec, episode trace report, final report, receipt
