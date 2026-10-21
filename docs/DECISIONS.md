# Decisions

Append-only. One entry per decision that affects PA4 scope, contracts or tooling.

---

## D-001 — P0 scope deferral relative to Master v1.2

**Date:** 2026-10-21
**Status:** accepted
**Authority:** `PA4-P0-DESIGN-v1.1` §0.3 ("Fast-Track exclusions take precedence")

Master `PA4-MASTER-ARCH-v1.2` lists broader P0-style deliverables (golden fixtures, CI/CD,
Docker, extensive receipts, full PA3 test suites). The P0 fast-track design explicitly
defers their **implementation**, not their eventual verification:

| Deferred item | Now owned by |
|---|---|
| Conditional action correctness / joint logprob parity | P1 |
| `EnvBase` adapter, game-rule correctness, observation parity | P2 |
| Collector + `InferenceServer` + `ProcessSlotTransport` integration | P3 |
| PPO / GAE / gradient parity, checkpoint restore | P4 |
| Docker, Cloud compatibility smoke | C1 |
| CI/CD, benchmarks, dashboards, deployment | P7–P9 / as specified |

**Not changed:** PA4's final success standards, the four G0 gates, the requirement that
objective tests (not reviews) decide gate outcomes.
**Rationale:** P0 must be a foundation gate, not a training sprint; P1 needs an isolated,
reproducible, safe starting point.

---

## D-002 — Frozen dependency tuple

**Date:** 2026-10-21
**Status:** accepted (evidence in `docs/phases/P0/COMPATIBILITY.md`)

One stable tuple was resolved and locked; no version sweep was performed:

| Component | Version | Why |
|---|---|---|
| Python | 3.12 | system interpreter on Ubuntu 24.04; pinned via `.python-version` |
| torch | `2.9.1+cu128` | stable CUDA 12.8 wheel from the official PyTorch index; compatible with host driver 610.88 |
| torchrl | `0.14.1` | newest stable TorchRL exposing all five required endpoints |
| tensordict | `0.14.3` | satisfies torchrl's `>=0.14.3,<0.15.0`; avoids the `0.15.x` line |

Package manager: `uv` (single lockfile `uv.lock`).
**Rejected:** torchrl `0.15.x`/tensordict `0.15.x` (outside TorchRL 0.14.1's constraint
without a source patch), nightly builds, extra RL/visualization extras.
**Rationale:** TorchRL 0.14 docs name the required APIs; the installed release was
verified empirically rather than assumed from documentation.

---

## D-003 — Agent roles are DeepSeek-only (no Qwen / MiniDC)

**Date:** 2026-10-21
**Status:** accepted (fixed in Master v1.2; recorded here for operational clarity)

`pa4-lead` is the sole writable primary agent; `pa4-auditor` is a fresh-context read-only
subagent on the same model. MiniDC/Qwen are **not** part of the PA4 development path and
no MiniDC configuration was added or validated.

**Known limitation:** same-model review is a process control only. It is **not**
independent model verification. PA3 reference contracts and automated tests remain the
authority.

---

## D-004 — G0-C auditor boundary: capability absence, not prompt compliance

**Date:** 2026-10-21
**Status:** accepted, with a recorded evidence limitation

`pa4-auditor` was hardened to an explicit default-deny profile
(`"*": "deny"` plus `edit`, `shell`, `task`, `webfetch`, `websearch`, `external_directory`
all `deny`; only `read`/`glob`/`grep`/`list` allowed).

Measured outcome: in a live fresh-context invocation the auditor's **tool registry
contained only `read`, `glob`, `grep`** — no edit, shell or task tool existed to invoke,
and an attempted proof-file read/write left no file behind.

**Limitation recorded, not hidden:** this is *capability absence*, which is a strictly
stronger guarantee than "the tool existed but refused". It is however **not** the
"present-and-refused" evidence the design's §4 wording imagines, because OpenCode v2
removes denied tools from the registry entirely. Additionally, the auditor runs in the
**Windows** domain while PA4 sources live on the **WSL ext4** filesystem; therefore the
auditor's positive read control was demonstrated on a workspace-internal file at
`Desktop\PA4\tmp\`, not on the WSL repo itself. Both facts are reported as-is.

**Rationale:** a read-only reviewer with no write/exec capability satisfies the security
boundary; the gate is not inflated to claim more than was observed.
