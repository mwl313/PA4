# P0-B / P0-C — Compatibility & Agent Routing Evidence

Document ID: `PA4-P0-COMPAT-v1.1`
Scope: P0-B (locked dependency stack + TorchRL API) and P0-C (DeepSeek-only routing, auditor boundary).
All values below are measured on the actual host. No assumed values.

> **Revision note (v1.1):** hardened `scripts/p0_smoke.py` to (a) fail under `-O` by using an
> explicit `check()` instead of bare `assert`, (b) assert installed versions equal the pins,
> (c) verify each API is a real class with an `__init__` signature. Added a negative-control
> run proving the script fails on version drift. Added lock-graph and host-identity evidence.
>
> **Revision note (v1.2):** added `scripts/p0_negative_control.py`, which mutates the
> **committed** script four ways (torchrl pin, tensordict pin, TensorDict shape, CUDA sum) and
> requires each variant to exit non-zero under `-O`. Result: `NEGATIVE_CONTROL_PASS variants=4`.
> G0-C is now recorded as a **scoped** pass, not a blanket one.

---

## 1. Locked dependency stack (P0-B, G0-B)

| Component | Pinned version | Source |
|---|---|---|
| Python | 3.12.3 (`/usr/bin/python3.12`, `.python-version` = 3.12) | Ubuntu 24.04 |
| torch | `2.9.1+cu128` | `download.pytorch.org/whl/cu128` (explicit index) |
| torchrl | `0.14.1` | PyPI (published 2026-10-10) |
| tensordict | `0.14.3` | PyPI |
| numpy | 2.5.4 (transitive) | PyPI |
| triton | 3.5.1 (transitive) | PyPI |

Resolution rationale:

- TorchRL `0.14.1` requires `tensordict>=0.14.3,<0.15.0`; `0.14.3` is the newest allowed
  `0.14.x` release, chosen over the newer `0.15.x` line to satisfy that constraint without
  a source patch.
- TorchRL `0.14.1` requires `torch>=2.1.0`. `torch 2.9.1+cu128` is a stable CUDA 12.8 build
  supported by the host driver (610.88).
- No nightly builds, no extra RL integrations, no visualization extras.

### Lock evidence (measured in-session)

```
$ uv lock --check
Resolved 37 packages in 0.74ms                     # exit 0

$ uv tree --frozen --depth 1
pa4 v0.0.1
├── tensordict v0.14.3
├── torch v2.9.1+cu128
└── torchrl v0.14.1

$ python3 inspect_lock.py
torch -> 2.9.1+cu128
torchrl -> 0.14.1
tensordict -> 0.14.3
numpy -> 2.5.4
cu128 index referenced in lock: True
packages in lock: 37
pypi.org/simple referenced: True

$ sha256sum uv.lock
8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e  uv.lock
```

`pyproject.toml` intentionally carries **no** hash pins; `uv.lock` is the single source of
truth for the full transitive graph (37 packages) and is committed.

### Reproducibility check (frozen)

| Step | Result |
|---|---|
| `rm -rf .venv && uv sync --frozen` | exit 0 |
| `uv run --frozen python scripts/p0_smoke.py` | exit 0, all PASS markers reproduced |

---

## 2. TorchRL API availability (measured on installed `torchrl 0.14.1`)

### Positive run — `uv run --frozen python scripts/p0_smoke.py`, exit 0

```
python 3.12.3
torch 2.9.1+cu128
torch.cuda 12.8
torchrl 0.14.1
tensordict 0.14.3
version_pins_ok {'torch': '2.9.1+cu128', 'torchrl': '0.14.1', 'tensordict': '0.14.3'}
api_import_ok TensorDict | module: tensordict._td
api_import_ok EnvBase | module: torchrl.envs.common
api_import_ok AsyncBatchedCollector | module: torchrl.collectors._async_batched
api_import_ok InferenceServer | module: torchrl.modules.inference_server._server
api_import_ok ProcessSlotTransport | module: torchrl.modules.inference_server._process_slot
tensordict_2x4_ok (2,) (2, 4)
P0_TENSORDICT_2X4_PASS
gpu_name NVIDIA GeForce RTX 4060 Ti
cuda_8x8_sum 64.0
P0_CUDA_8X8_PASS
P0_TORCHRL_API_IMPORT_PASS
```

Checks performed: `inspect.isclass`, non-empty `__init__` signature, module provenance.
This distinguishes real classes from `None` aliases or stubs.

### Negative control — proves the script actually fails

`scripts/p0_negative_control.py` is **committed to the repo** (tracked, not ignored) and grows
the tree from 17 to 18 files; the earlier "16 tracked files" figure predated it. It rewrites
the **committed** script with four independent
mutations and requires each variant to exit non-zero under `python -O` (where bare
`assert`s would be stripped):

```
PASS wrong_torchrl_pin:   exit=1 matched='P0_SMOKE_FAIL: torchrl'
PASS wrong_tensordict_pin: exit=1 matched='P0_SMOKE_FAIL: tensordict'
PASS wrong_tensordict_shape: exit=1 matched='P0_SMOKE_FAIL'
PASS wrong_cuda_sum:      exit=1 matched='P0_SMOKE_FAIL: 8x8 cuda sum'
NEGATIVE_CONTROL_PASS variants=4
```

The control operates on **temporary copies**, never the working tree, so it cannot mutate the
committed script. `wrong_tensordict_shape` asserts only the generic `P0_SMOKE_FAIL` prefix
(the broadest of the four); the other three match specific messages.

So the version-pin, TensorDict-shape and CUDA branches are each demonstrated to be fail-able,
not just optimizable. The script is not a rubber stamp.

### `__init__` signatures (recorded for P1 planning only)

| Symbol | `__init__` parameters (abridged) |
|---|---|
| `AsyncBatchedCollector` | `create_env_fn, policy, policy_factory, frames_per_batch, total_frames, max_batch_size, min_batch_size, server_timeout, transport, device, backend, env_backend, env_exchange, envs_per_worker, transition_chunk_size, policy_backend, reset_at_each_iter, postproc, exploration_type, replay_buffer, post_collect_hook, yield_completed_trajectories, weight_sync, weight_sync_model_id, verbose, create_env_kwargs, worker_affinity, driver_affinity, server_config, device_config, policy_version, policy_version_key` |
| `InferenceServer` | `model, transport, policy_factory, service_backend, service_backend_options, transport_options, request_spec, response_spec, num_clients, max_batch_size, static_batch_size, min_batch_size, timeout, collate_fn, device, policy_device, output_device, collect_stats, stats_window_size, weight_sync, weight_sync_model_id, server_config, device_config, shutdown_event, policy_version, policy_version_key` |
| `ProcessSlotTransport` | `request_spec, response_spec, num_slots, ctx, copy_result` |
| `EnvBase` | `device, batch_size, run_type_checks, allow_done_after_reset` |

`AsyncBatchedCollector` exposes `policy_version` / `policy_version_key` — recorded as an
observation relevant to the P6 staleness contract; not exercised in P0.

### Known limitations (explicit)

- Proves **import/class availability and version equality** only. It does **not** prove
  action-contract compatibility, a custom `EnvBase`, collector throughput, or process
  transport reliability under load. Those are P1–P3 obligations.
- TorchRL was not re-exported, patched or forked. One version resolution only; no sweep.

---

## 3. OpenCode DeepSeek-only routing (P0-C)

| Item | Measured value |
|---|---|
| OpenCode CLI | `opencode v2.1.2` (Windows-native install) |
| `pa4-lead` | `primary`, `deepseek/deepseek-flash`, write-capable |
| `pa4-auditor` | `subagent`, `deepseek/deepseek-flash`, `"*": "deny"` + explicit `edit`/`shell`/`task`/`webfetch`/`websearch`/`external_directory` `deny`; only `read`/`glob`/`grep`/`list` allowed |

Auditor config also carries **both** a wildcard deny and explicit per-tool denies, so the
boundary does not depend on implicit wildcard inheritance alone.

### G0-C live smoke (measured — three attempts, reported truthfully)

| Attempt | Setup | Result |
|---|---|---|
| 1 | Auditor asked to read a WSL path | `permission.rejected: external_directory`. Auditor correctly refused to fabricate a result and returned `INSUFFICIENT_EVIDENCE`. |
| 2 | Auditor asked to read a temp-dir path | Same `external_directory` rejection. |
| 3 | Target placed **inside the workspace root** | **Read succeeded.** |

Attempt 3 observations (fresh-context `pa4-auditor`):

```
TOOLS_IN_REGISTRY:      read, glob, grep
MARKER_TEXT:            PA4_P0_DEEPSEEK_TOOL_SMOKE_OK     (verbatim read)
WRITE attempt:          no write-capable tool in registry
SHELL attempt:          no shell tool in registry
TASK attempt:           no task tool in registry
PROOF_FILE_CREATED:     FALSE
```

Lead-side independent verification: `auditor_write_proof.txt` **absent** from the directory
listing; only `p0_agent_smoke.txt` present (29 bytes).

Lead-side tool write: the marker file was created and read back through real tool calls, and
`git check-ignore -v` confirmed it matched `.gitignore:10:tmp/`.

### G0-C conjunct-by-conjunct status

| Conjunct (per `PA4-P0-DESIGN-v1.1` §4 / §7) | Status | Evidence |
|---|---|---|
| Lead performs a real **write** tool call to ignored scratch | **met** | marker file created, read back, `git check-ignore` matched `.gitignore:10:tmp/` |
| Fresh-context auditor **reads the marker** | **met** | attempt 3: auditor returned `PA4_P0_DEEPSEEK_TOOL_SMOKE_OK` verbatim (read of the actual file, not inference) |
| Auditor **edit denied** | **met (by absence)** | registry = `read, glob, grep` only; no edit/write/patch existed |
| Auditor **shell denied** | **met (by absence)** | no shell/bash/execute in registry |
| Auditor **cannot nest** agents | **met (by absence)** | no task tool in registry |
| Auditor reviews the PA4 deliverables | **not met** | Windows domain vs WSL ext4; see below |

### What G0-C does and does not establish

- **Established:** `pa4-lead` performs real write tool calls; a fresh-context `pa4-auditor`
  can read and report exact file content; the auditor had **no** edit, shell or task
  capability available to it at all.
- **Scope of the PASS:** the design's fail condition is *"if the auditor can modify files or
  execute shell commands, report G0-C as BLOCKED"*. That condition is **not** triggered, so
  G0-C passes **for the read-only boundary and the lead write only**.
- **Not covered — auditor deliverable review:** `pa4-auditor` runs in the **Windows** domain
  (`C:\Users\...\Desktop\PA4`) while PA4 sources live on the WSL ext4 filesystem. The
  auditor's positive read control used a workspace-internal **Windows** file. It therefore did
  **not** review the PA4 deliverables under `/home/pa4/projects/...`. Recorded as
  `auditor_covered_pa4_deliverables: false`. To obtain real diff review in P1, the auditor
  workspace must move into WSL.
- **Limitation — capability absence, not present-and-refused:** OpenCode v2 removes denied
  tools from an agent's registry entirely, so there is no "tool existed and refused" error to
  observe. Absence is a stronger in-session guarantee, but a different mechanism than the
  design's wording suggests. Recorded as `D-004`.
- **Limitation — same model:** lead and auditor are both `deepseek/deepseek-flash`. Context
  separation is a process control, not independent model verification. Objective correctness
  comes only from automated tests and PA3 reference contracts.

---

## 4. Reproduction commands

```bash
cd ~/projects/PokeAgent4.0
uv lock --check
rm -rf .venv && uv sync --frozen
uv run --frozen python scripts/p0_smoke.py
sha256sum uv.lock
```
