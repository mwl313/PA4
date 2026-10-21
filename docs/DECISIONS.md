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

**Post-review correction (v2):** the first receipt draft said only "capability absence" in a
footnote while the gate row read a flat `PASS`. After read-only review flagged that the
auditor never touched the actual PA4 deliverables, the receipt was changed to record
`auditor_could_reach_pa4_wsl_repo: false` and a per-conjunct status block, so the gap is
visible in the machine-readable record rather than buried in prose.

---

## D-005 — Smoke script hardened after read-only review

**Date:** 2026-10-21
**Status:** accepted

Read-only review (Finding 10) correctly identified that `scripts/p0_smoke.py` used bare
`assert`, which `python -O` strips, and that it printed versions without asserting them
equal to the pins. Either defect would let a drifted environment still print every `_PASS`
line and exit 0.

Changes:

1. Replaced bare `assert` with an explicit `check()` that raises `SystemExit`.
2. Assert installed `torch`/`torchrl`/`tensordict` equal the pinned versions.
3. Assert each required API is a real class with a non-empty `__init__` signature and
   record its provenance module.

Verified with a **negative control**: a copy with `torchrl` expected pinned to `9.9.9`
exits `1` with `P0_SMOKE_FAIL: torchrl installed='0.14.1' expected='9.9.9'` even under
`python -O`. A gate script that cannot fail is not evidence.

---

## D-006 — Observation width interpretation fixed

**Date:** 2026-10-21
**Status:** accepted

Review (Finding 8) asked how `OBSERVATION_TOKENS = 96` reconciles with
`CATEGORY_COUNT = 32`, `FLOAT_COUNT = 50`, `FLAG_COUNT = 40`. Answer, verified against
`TokenFeatures` at `engine/src/observation.rs:152-159`: the three counts are **per-token**
field counts, not components of 96. The full packed blob is
`OBSERVATION_FIXED_BYTES = 41,986`, which equals the literal expression in
`engine/src/python.rs:27-36`.

An earlier draft of `SOURCE_FREEZE.md` stated 45,548 bytes from faulty mental arithmetic;
that figure was recomputed in the environment and replaced with 41,986 before commit.
P1 must not treat 96 as `32 + 50 + 40 + …`.
