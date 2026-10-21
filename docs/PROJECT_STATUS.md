# PA4 Project Status

**Last updated:** 2026-10-21
**Current phase:** P0 — Discovery, Environment Freeze, Minimal Compatibility Proof
**Overall P0 verdict:** **PASS** (4/4 gates)

| Gate | Result | Evidence |
|---|---|---|
| G0-A Dev host (WSL2 + CUDA + 8×8 tensor) | **PASS** | `docs/phases/P0/ENVIRONMENT.md` §5 |
| G0-B Locked dependencies + TorchRL APIs + TensorDict 2×4 | **PASS** | `docs/phases/P0/COMPATIBILITY.md` §1–2 |
| G0-C DeepSeek-only routing + read-only auditor | **PASS** | `docs/phases/P0/COMPATIBILITY.md` §3 |
| G0-D PA3 reference freeze + P1 handoff | **PASS** | `docs/phases/P0/SOURCE_FREEZE.md` |

PASS establishes **development readiness only**. It does not prove that the battle engine,
the conditional action sampler or the PPO learner works.

## Pinned facts

- **PA4 repo:** `/root/projects/PokeAgent4.0` (WSL2 Ubuntu 24.04, ext4)
- **PA3 reference:** `https://github.com/mwl313/PokeAgent3.0` @ `65bc980abd8e0bbec14158b9ddc2884bb1726a4e` (read-only)
- **Stack:** Python 3.12 / torch `2.9.1+cu128` / torchrl `0.14.1` / tensordict `0.14.3`
- **Lock:** `uv.lock` sha256 `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e`
- **Agents:** `pa4-lead` (primary, writable) + `pa4-auditor` (subagent, read-only)

## Current blocker

None blocking P1.

## Open limitations (carried forward, not blocking)

1. Rust `1.90.0` toolchain is **not installed** in WSL yet; P1/P2 must install the pinned
   toolchain before building `pa3-engine`.
2. The pinned PA3 SHA is the **public `main` tip**, not verified live production state.
   PA3 records production as stopped since 2026-10-16.
3. `pa4-auditor` runs in the Windows domain and **cannot read the WSL ext4 repo**; its
   positive read control was demonstrated on a workspace-internal Windows path, so it did
   **not** audit the actual PA4 deliverables. Moving the auditor workspace into WSL is
   required before it can perform real diff review in P1.
4. Auditor read-only status is **capability absence** (denied tools are removed from the
   registry), not present-and-refused (see `docs/DECISIONS.md` D-004).
5. Lead and auditor are the **same model**; review is a process control, not independent
   verification.

## Review status

A read-only `pa4-auditor` review of the P0 artifacts was performed (fresh context) and
returned `INSUFFICIENT_EVIDENCE` on the first evidence set. Its findings were acted on:

| Finding | Action |
|---|---|
| Smoke script used optimizable `assert`; no version assertions | Fixed + negative control added (D-005) |
| WSL2 host identity asserted, not evidenced | `uname`/`/proc/version`/`os-release` added (ENVIRONMENT.md §1) |
| Lock hash unsupported by a lock artifact | `uv lock --check`, `uv tree`, lock graph added |
| PA3-unmodified unevidenced | `git status`/`reflog` output added (SOURCE_FREEZE.md §1) |
| PA4 isolation unevidenced | `sys.path` and `import pa3_engine` checks added |
| Observation width unreconciled | Resolved: 41,986 bytes, per-token counts (D-006) |
| Auditor could not audit deliverables | Recorded as an explicit limitation, not claimed as covered |

Remaining gaps are recorded rather than closed by assertion.

## Immediate next task

Begin **P1 — Conditional Action Feasibility Gate**: answer the three handoff questions in
`docs/phases/P0/SOURCE_FREEZE.md` §4 with the smallest working proof. Do not build the
full `EnvBase`, collector or PPO before G1.

## Artifacts

- `docs/phases/P0/ENVIRONMENT.md`, `COMPATIBILITY.md`, `SOURCE_FREEZE.md`, `RECEIPT.json`
- `docs/phases/P0/DESIGN.md` (authoritative P0 design, verbatim)
- `docs/master/PA4_MASTER_ARCHITECTURE.md` (authoritative master, verbatim)
