# PokeAgent 4.0 — P0 Detailed Design: Fast-Track Foundation

**Document ID:** `PA4-P0-DESIGN-v1.1`  
**Date:** 2026-10-21 (KST)  
**Status:** DESIGN READY, NOT IMPLEMENTED  
**Parent:** `PA4-MASTER-ARCH-v1.2` (2026-10-21)  
**Implementation phase:** P0 — Discovery, Environment Freeze, Minimal Compatibility Proof  
**Next phase:** P1 — Conditional Action Compatibility Gate

> **One-line objective:** Windows/WSL2 개발 머신에서 PA4의 의존성, 코딩 에이전트, PA3 참조 계약을 최소한의 증거로 고정하고, P1을 즉시 시작할 수 있는 독립 작업공간을 만든다. **P0에서는 배틀 환경도 PPO도 구현하지 않는다.**

---

## 0. 결정과 범위 제한

### 0.1 고정 결정

| 항목 | P0 계약 |
|---|---|
| Developer workstation | Windows 11, WSL2 Ubuntu 24.04, RTX 4060 Ti 8GB |
| MiniDC | **Not used by PA4 development**; existing service/global config left unchanged |
| Coding harness | OpenCode inside WSL2 |
| Lead & Implementer | DeepSeek V4.1-Flash-1017 API, OpenCode `deepseek/deepseek-flash`; verify actual route at execution |
| Implementation concurrency | **One writable DeepSeek session at a time**, no Qwen delegation or Swarm |
| Auditor | Fresh DeepSeek context, read-only permissions; **same-model review is not independent model verification** |
| RL foundation | PyTorch, TorchRL, TensorDict, exact compatible released versions pinned in lockfile |
| PA3 source | **Read-only, SHA-pinned reference**, no modification to active PA3 project |
| Cloud | No Runpod rental in P0; choice of production GPU stays open |
| Test principle | Four small smoke gates; no production tests or performance tuning |

### 0.2 Strict in scope

1. WSL2 Linux development workspace availability and CUDA device detection (no driver rewrites unless proven broken).
2. Separate PA4 Git repository/worktree initialization, safety guardrails and a minimal deterministic dependency lock.
3. TorchRL API availability check **in the actual installed environment** (not a full Collector integration).
4. OpenCode DeepSeek-only route, one direct edit/tool-call smoke, one separate DeepSeek read-only reviewer invocation and permission check.
5. PA3 source SHA, reference engine/policy/collector/PPO module paths and contract inventory; a narrow P1 handoff.
6. Minimal P0 receipt, status and explicit PASS / BLOCKED decision.

### 0.3 Explicitly out of scope — do not start

- TorchRL `EnvBase` subclass implementation, Rust-Python adapters, complete battles, action distribution, Joint Action enumeration.
- PPO rollout collection, GAE parity, gradient parity, training loops, self-play, evaluation league or checkpoint migration.
- Whole PA3 test suite, all fixture replay, long-running tests, benchmarks, load tests, simulated network failures or fault injection.
- CUDA Graphs, `torch.compile` performance, FP16/BF16 exploration, model compression, microbatch autotuning or >40/1000 matches/s performance claims.
- Multi-GPU/DDP, Runpod rental, persistent cloud volumes, dashboard/UI, CI/CD pipelines, database, monitoring service, Kubernetes or automated deployment.
- Full PA3 code vendoring, large checkpoint downloads, bulk docs migration, new research branch or framework alternatives unless a hard blocker is evidenced.

**P0 stop rule:** If a change is not required to pass G0 and unblock P1, it is deferred. No speculative refactor. No extended dependency compatibility matrix. No more than one stable pinned stack.

**Relationship to Master v1.2:** The master describes wider P0 deliverables (golden tests, CI, Docker, extensive receipts). This P0 fast-track design explicitly defers their **implementation**, not their eventual verification. P1 owns action correctness, P2 environment/game correctness, P4 PPO correctness, C1 cloud/container validation. Record this single scoped deferral in `docs/DECISIONS.md`; do not change PA4's final success standards.

---

## 1. Minimal target architecture

```text
Windows 11 / NVIDIA Driver
           |
  WSL2 Ubuntu 24.04 (Linux filesystem)
           |
  ~/projects/PokeAgent4.0   [NEW writable Git repo]
       |               |
  OpenCode              .venv (PyTorch, TorchRL, TensorDict)
       |                            |
       |                       RTX 4060 Ti 8GB
       |
       +-- DeepSeek API: Lead + Implementer (writable session)
       +-- DeepSeek API: Auditor (fresh read-only context)

  ~/projects/PokeAgent3.0-reference [READ-ONLY SNAPSHOT]
       ^ only used to capture file/SHA/contract references

  Runpod: NOT USED IN P0
```

**Hard boundary:** OpenCode's write workspace is PA4. PA3 reference must not be modified by tasks, and production PA3 checkpoints, runs and branches are never touched. Repo naming is illustrative; reuse a separately isolated existing PA4 workspace if present instead of force-reinitializing it.

---

## 2. Work packages, responsibilities, exact scope

### P0-A — Host / WSL2 preflight

**Owner:** DeepSeek Lead/Implementer only.  
**Allowed output:** `docs/phases/P0/ENVIRONMENT.md`, no software source changes.

Required observations:

- Linux is WSL2 Ubuntu 24.04, Python and Rust toolchains available, disk/workspace writable, git available.
- `nvidia-smi` sees RTX 4060 Ti and PyTorch reports CUDA available; make **one small** CUDA tensor operation.
- Source directory under WSL Linux home, not `/mnt/c`.
- WSL/Windows NVIDIA driver is used. **Never install a separate Linux NVIDIA kernel driver inside WSL.**

Suggested commands (adapt Python/Rust installation only if missing):

```bash
uname -a
cat /etc/os-release | grep -E 'NAME=|VERSION_ID='
nvidia-smi
python3 --version
cargo --version
git --version
```

After PyTorch is installed:

```bash
python - <<'PY'
import torch
print('torch=', torch.__version__, 'cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA unavailable on dev machine'
x = torch.ones((8, 8), device='cuda')
assert x.sum().item() == 64
print('gpu_name=', torch.cuda.get_device_name(0))
PY
```

**Exit condition:** CUDA smoke passes; no GPU memory-intensive test. If no CUDA, mark BLOCKED with exact error and stop. Do not mask it by silently switching all later test expectations to CPU.

### P0-B — Minimal PA4 repo and reproducible dependency stack

**Owner:** DeepSeek Lead/Implementer (one bounded work unit) followed by fresh read-only DeepSeek review.  
**Allowed changes:** PA4 repo scaffold, package manifest/lock, tiny smoke script; no Rust/PPO/game implementation.

Required files:

```text
PokeAgent4.0/
├── AGENTS.md
├── README.md
├── opencode.json                 # no secrets or personal IPs committed
├── pyproject.toml
├── uv.lock                       # or a single equivalent fully pinned lock method
├── .gitignore
├── configs/
│   └── dev_8gb.yaml             # declarative device profile only
├── docs/
│   ├── master/PA4_MASTER_ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   ├── DECISIONS.md
│   └── phases/P0/
│       ├── DESIGN.md
│       ├── ENVIRONMENT.md
│       ├── SOURCE_FREEZE.md
│       ├── COMPATIBILITY.md
│       └── RECEIPT.json
└── scripts/
    └── p0_smoke.py              # one minimal import/API/GPU check
```

Use one Python environment manager: **`uv` preferred**, pinned stable Python and versioned package set. PyTorch should be installed from the official CUDA-compatible build selector, then TorchRL/TensorDict compatible stable releases should be locked. Avoid nightly packages, extra RL integrations, lint bundles and visualization dependencies.

**Framework version policy:** As of drafting, published TorchRL 0.14 line documents `AsyncBatchedCollector`, `InferenceServer` and `ProcessSlotTransport`; TorchRL 0.14 requires TensorDict >=0.14.2. Do **not** assume a specific PyTorch version from this design. The implementer must resolve and freeze one installable compatible tuple (`python`, `torch`, `torchrl`, `tensordict`), report versions and lock hash, then stop exploring versions if the API smoke passes.

**Single TorchRL API smoke** (`scripts/p0_smoke.py`; no environment workers):

```python
import torch
import torchrl
import tensordict
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.collectors import AsyncBatchedCollector
from torchrl.modules.inference_server import InferenceServer, ProcessSlotTransport

print('torch', torch.__version__)
print('torchrl', getattr(torchrl, '__version__', 'unknown'))
print('tensordict', getattr(tensordict, '__version__', 'unknown'))
assert torch.cuda.is_available()
td = TensorDict({'observation': torch.zeros(2, 4)}, batch_size=[2])
assert td['observation'].shape == (2, 4)
print('P0_TORCHRL_API_IMPORT_PASS')
```

This proves only API availability and a tiny TensorDict operation; **it does not prove our action compatibility, custom EnvBase contract, Collector throughput or process transport reliability**. Those belong to P1-P3.

**No needless repetition:** Run installation/import smoke once after environment resolution and once from a clean `uv sync --frozen`-equivalent reproduction, not repeated version sweeps.

### P0-C — OpenCode DeepSeek-only routing and read-only audit boundary

**Owner:** DeepSeek Lead/Implementer with `pa4-auditor` in a separate fresh context.  
**Allowed changes:** PA4 project-local `opencode.json`, `AGENTS.md`, ignored `tmp/p0_agent_smoke.txt`, `docs/phases/P0/COMPATIBILITY.md`. Do not modify the user's global OpenCode configuration.

Steps:

1. Record actual `opencode --version` inside WSL2. Use the installed version's recognized config schema.
2. Reuse the user's working DeepSeek OpenCode provider credentials; project-local config selects `deepseek/deepseek-flash`. Confirm model selection and **a real tool call**. Do not duplicate API credentials in JSON, source, logs or receipts.
3. Configure `pa4-lead` as **primary, write-capable** and `pa4-auditor` as a **subagent with fresh context and read-only tools**. Disable shell/edit/task and write-capable MCP tools for the auditor, and verify actual permissions.
4. Perform one tiny handoff: Lead writes a fixed marker to ignored PA4 scratch file; auditor reads that marker in a new context and reports `PASS_REVIEW` (or a precise issue) without changing files. Record actual tool results in `COMPATIBILITY.md`. Remove generated scratch afterward.
5. Avoid extra agent roles, Qwen provider validation, custom queues, workflows, new MCP servers, or background swarm orchestration.

**Operational detail:** The Auditor uses the same DeepSeek model but a fresh, restricted context. It is an additional review, **not** an independent model oracle. Objective correctness comes from Gate tests and PA3 reference contracts.

**Failure behavior:** If DeepSeek cannot invoke file tools, mark `BLOCKED` with a sanitized error after at most one targeted fix. If the reviewer is able to write or use a shell, mark `BLOCKED`; a natural-language instruction to be read-only is insufficient.

### P0-D — Reference-source freeze and P1 handoff contracts

**Owner:** DeepSeek Lead/Implementer (direct source metadata collection).  
**Allowed output:** two short reference documents; no copied checkpoints, no generated fixtures.

Source resolution procedure:

1. Inspect the user's designated PA3 source repository and identify **one immutable source SHA** for reference. Do not mistake an older project's `PROJECT_STATUS.md` production progress for the actual latest training state.
2. Record source URI, branch and commit SHA; if the user's active local training sources are not available from the Windows machine, record that limitation explicitly rather than inventing confirmation.
3. Identify only the source files required by P1: Rust engine entry point and native candidate API, observation schema, request/branch semantics, conditional sampler/logprob, PPO loss contract, and the one canonical precision/source document.
4. Capture minimal contract facts without implementing them. No full legacy code audit or fixture extraction here.

**Required output:** `docs/phases/P0/SOURCE_FREEZE.md` with:

| Contract | Evidence to record | P1 usage |
|---|---|---|
| Engine | Rust crate path, entry APIs, source SHA | native action legality reference |
| Observation | 96-token contract, masks, hidden info | request payload shapes |
| Action | Prefix-dependent candidate availability, legal masks, complete player request | P1 custom distribution feasibility |
| Probability | `joint_logprob = sum(valid branch logprob)`; FP32 logprob | action parity oracle |
| Learning | PPO-2 target, actor/value mask, two-seat match accounting | later P4 contract reference |
| Precision | Pa3 FP32 reference mode, no permissive tolerance changes | later P1/P4 numerical baseline |

Also include `P1_HANDOFF.md` (could be a section in `SOURCE_FREEZE.md`) with **exactly three questions**:

- Can the TorchRL policy server sample a complete legal conditional player action without changing the Rust legality rules?
- Is a single batched request feasible, or is a staged prefix protocol required?
- What minimal saved branch metadata is necessary for exact behavior-logprob recomputation?

No benchmark numbers or claims about current production status without independent live evidence.

---

## 3. Execution order and handoffs

| Sequence | Work package | Assigned execution mode | Blocking dependency |
|---|---|---|---|
| 1 | P0-A host/WSL/GPU preflight | DeepSeek Lead | None |
| 2 | P0-B repo scaffold + TorchRL lock | DeepSeek Implementer | A |
| 3 | P0-C OpenCode DeepSeek-only and reviewer smoke | DeepSeek Lead + fresh read-only Auditor | A + installed OpenCode |
| 4 | P0-D source freeze + P1 handoff | DeepSeek Lead/Implementer | Accessible PA3 source |
| 5 | Four G0 checks and commit | DeepSeek Lead, automated tests, read-only Auditor | A-D |

One writable agent context at a time, auditor read-only. No custom subagent swarm. A model's narrative report is not a test receipt.

### Essential limits (scope creep prevention)

- One PA4 P0 branch and one G0 commit series, no speculative TorchRL forks.
- Exactly four P0 Gate checks; no extra test suites or performance experiments.
- No alternative framework investigations unless the pinned TorchRL API proves unavailable.
- No source code beyond minimal scaffold/config and tiny smoke script.
- No PA3 module migration until P1/P2 and no mutation of PA3's original worktree.

---

## 4. Minimal Verification Gate G0

| Gate | Exactly what to check | Pass evidence | Explicit exclusions |
|---|---|---|---|
| **G0-A Dev host** | WSL2, CUDA detected, one 8×8 torch CUDA tensor | `ENVIRONMENT.md` command outputs | VRAM stress / CPU perf |
| **G0-B Locked dependencies** | `uv sync --frozen`-equivalent, TorchRL/TensorDict API import + TensorDict 2×4 operation | `p0_smoke.py` output, package versions + lock hash | Collector environment start, GPU training |
| **G0-C DeepSeek-only routing** | DeepSeek Lead tool write to ignored scratch, fresh `pa4-auditor` reads marker with edit/shell denied | sanitized `COMPATIBILITY.md` with real tool results and permission checks | Qwen, provider swarms, stress tests |
| **G0-D Source freeze** | Read-only PA3 SHA, P1 paths and invariants, PA4 isolated writable repo | `SOURCE_FREEZE.md`, `git status` on PA3 and PA4 | full PA3 tests, checkpoints or fixture replay |

**G0 PASS only if all four pass.** Keep failures precise. One diagnostic repair cycle per blocked gate is permitted before escalating the blocker to the user; do not perform hours of speculative rewrites. PASS does not mean Rust gameplay or PPO train is integrated.

**Protected PA3 handling:** Prefer an independent read-only archive/clone and verify the source worktree is unchanged. A clone's read-only intent is not a filesystem guarantee. The Lead uses `external_directory=ask` for strictly authorized **read-only** PA3 reference inspection, never edits PA3; the Auditor has external access denied and no shell. Do not store PA4-generated artifacts in PA3.

**Minimal deterministic receipt:**

```json
{
  "phase": "P0",
  "verdict": "PASS_OR_BLOCKED",
  "pa4_commit": "ACTUAL_SHA",
  "pa3_reference_commit": "ACTUAL_SHA",
  "host": "wsl2_ubuntu_24_04_win11",
  "gpu": "rtx_4060_ti_8gb",
  "torch": "ACTUAL_VERSION",
  "torchrl": "ACTUAL_VERSION",
  "tensordict": "ACTUAL_VERSION",
  "dependency_lock_sha256": "ACTUAL_SHA256",
  "opencode_version": "ACTUAL_VERSION",
  "deepseek_model_route": "ACTUAL_ROUTE",
  "auditor_model_route": "ACTUAL_ROUTE",
  "auditor_read_only_verified": "ACTUAL_TRUE_OR_FALSE",
  "gates": {
    "G0_A": "PASS_OR_BLOCKED",
    "G0_B": "PASS_OR_BLOCKED",
    "G0_C": "PASS_OR_BLOCKED",
    "G0_D": "PASS_OR_BLOCKED"
  },
  "notes": "NO_SECRETS_NO_GUESSING"
}
```

Never fill placeholders with assumed values. Commit only a final, actually measured receipt. For startup iteration, an empty template file is acceptable but must not claim PASS.

---

## 5. Tiny files and responsibilities

**`AGENTS.md` (short):** one writable DeepSeek session at a time, P0 boundaries, PA3 non-mutation, required commands, failure escalation, no API keys in Git, separate DeepSeek auditor read-only. Do not create an agent orchestration framework.

**`docs/PROJECT_STATUS.md` (short):** current phase, last green gate, source SHA, current blocker, one immediate next task, relevant artifacts. Update once at final G0, not on every tool call.

**`docs/DECISIONS.md`:** one entry for P0 scope deferral from master v1.2 and one chosen dependency/version tuple. DeepSeek-only agent-role change is fixed in the master; do not repeat the historical Qwen decision as an active configuration.

**`configs/dev_8gb.yaml`:** only device / precision / debug batch defaults; a placeholder schema, not an operational PPO profile. Do not invent final training hyperparameters or GPU speed projections.

**`README.md`:** minimal setup and three commands to validate environment; no installation encyclopedia.

**`docs/phases/P0/COMPATIBILITY.md`:** installed API proof, OpenCode tool-call results and minimal known limitations. Keep outputs concise and sanitized.

**`SOURCE_FREEZE.md`:** metadata and P1 reference paths, no massive copied PA3 files.

---

## 6. Contingencies and rollback

| Problem | Fast-track handling | Stop point |
|---|---|---|
| WSL CUDA unavailable | Verify Windows NVIDIA driver / WSL update, rerun one tiny tensor test | BLOCKED if still unavailable |
| PyTorch TorchRL dependency conflict | Choose one compatible **stable release tuple**, resolve once and pin | BLOCKED if no supported combination without source modifications |
| TorchRL import exists but process transport missing | Confirm installed package/API vs stable docs, try one matching released version | BLOCKED, no Collector rewrite in P0 |
| DeepSeek provider works but file tools fail | Verify WSL2 provider authentication, OpenCode permissions and actual model route; one targeted correction | BLOCKED with sanitized tool error |
| Auditor can modify worktree | Restrict edit + shell and test on scratch only | BLOCKED; security boundary required |
| Latest PA3 production commit unknown | Pin accessible source SHA and record that actual production may differ | PASS with limitation only if P1's reference SHA is unambiguous; never claim live production sync |
| 8GB OOM during 8×8 smoke | Fix basic device availability; do not shrink below trivial smoke as fake success | BLOCKED |

**Rollback:** Keep P0 on independent PA4 branch. If installation or config fails, restore pinned manifest and delete only PA4-generated scratch files. Never run destructive cleanup against PA3, Windows-wide installs or MiniDC inference server without explicit user authorization.

---

## 7. Definition of Done and P1 start contract

P0 is **PASS** when all of the following hold:

- WSL2 + 4060 Ti runs the trivial torch CUDA smoke.
- Frozen PyTorch + TorchRL + TensorDict stack can be installed reproducibly and exposes named API imports.
- OpenCode DeepSeek Lead writes a single controlled scratch marker with a real tool call. A fresh read-only DeepSeek auditor reads and reviews it; edit and shell operations remain unavailable.
- An immutable PA3 reference SHA and minimal actionable P1 contracts exist. PA3 source is unchanged.
- `RECEIPT.json`, `COMPATIBILITY.md`, `SOURCE_FREEZE.md` and `PROJECT_STATUS.md` contain verifiable facts; no values fabricated or tests silently skipped.
- The PA4 branch is committed with no credentials; P1 can start without GPU rental or further infrastructure expansion.

**P1 is authorized to:** build the **smallest working proof** of PA3 legal conditional action sampling and exact joint logprob with the pinned TorchRL stack. P1 must not treat P0 imports as proof of compatibility.

**P1 is not authorized to:** implement full Rust EnvBase, collector, PPO or Runpod infrastructure before its action feasibility gate. Those belong to P2-P4/C1.

---

## 8. Authoritative implementation references

- Master architecture: `PA4_MASTER_ARCHITECTURE_v1.2_DEEPSEEK_ONLY_2026-10-21.md` (include unchanged copy under `docs/master/` at initialization).
- TorchRL 0.14 `AsyncBatchedCollector`: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.collectors.AsyncBatchedCollector.html
- TorchRL `ProcessSlotTransport`: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.inference_server.ProcessSlotTransport.html
- TorchRL release notes: https://github.com/pytorch/rl/releases
- OpenCode agents/permissions: https://opencode.ai/docs/agents/
- OpenCode custom provider: https://opencode.ai/docs/providers/
- DeepSeek model alias: https://api-docs.deepseek.com/updates/
- PA3 owner repository: https://github.com/mwl313/PokeAgent3.0 (P0 must pin actual commit SHA)

**Final design rule:** P0 is a **foundation gate, not a training sprint**. Do only enough to make P1 independent, reproducible and safe. Everything else is deferred by default.
