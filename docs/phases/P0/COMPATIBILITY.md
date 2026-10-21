# P0-C / P0-B — Compatibility & Agent Routing Evidence

Document ID: `PA4-P0-COMPAT-v1.0`
Scope: P0-B (locked dependency stack + TorchRL API) and P0-C (DeepSeek-only routing, auditor boundary).
All values below are measured on the actual host. No assumed values.

---

## 1. Locked dependency stack (P0-B, G0-B)

| Component | Pinned version | Source |
|---|---|---|
| Python | 3.12.3 (system `/usr/bin/python3.12`, pinned via `.python-version` = 3.12) | Ubuntu 24.04 image |
| torch | `2.9.1+cu128` | `download.pytorch.org/whl/cu128` (explicit index) |
| torchrl | `0.14.1` | PyPI (published 2026-10-10) |
| tensordict | `0.14.3` | PyPI |
| numpy | 2.5.4 (transitive, locked) | PyPI |
| triton | 3.5.1 (transitive, locked) | PyPI |

Resolution rationale:

- TorchRL `0.14.1` requires `tensordict>=0.14.3,<0.15.0`; `0.14.3` is the newest
  allowed `0.14.x` release and is deliberately chosen over the newer `0.15.x` line
  to satisfy that constraint without a source patch.
- TorchRL `0.14.1` requires `torch>=2.1.0`. `torch 2.9.1+cu128` is a stable CUDA 12.8
  build that the host NVIDIA driver (610.88, CUDA UMD 13.3) supports.
- No nightly builds, no extra RL integrations, no visualization extras were added.

Lock evidence:

```
uv lock          -> Resolved 37 packages
uv sync --frozen -> installed from uv.lock only
sha256(uv.lock)  = 8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e
```

### Reproducibility check (frozen)

Command performed: delete `.venv`, then `uv sync --frozen`, then re-run the smoke.

| Step | Result |
|---|---|
| `rm -rf .venv && uv sync --frozen` | exit 0 |
| `uv run --frozen python scripts/p0_smoke.py` | exit 0, all PASS markers reproduced |

---

## 2. TorchRL API availability (measured on installed `torchrl 0.14.1`)

Script: `scripts/p0_smoke.py` (exit 0).

```
python 3.12.3
torch 2.9.1+cu128
torch.cuda 12.8
torchrl 0.14.1
tensordict 0.14.3
api_import_ok TensorDict
api_import_ok EnvBase
api_import_ok AsyncBatchedCollector
api_import_ok InferenceServer
api_import_ok ProcessSlotTransport
tensordict_2x4_ok (2,) (2, 4)
P0_TENSORDICT_2X4_PASS
gpu_name NVIDIA GeForce RTX 4060 Ti
cuda_8x8_sum 64.0
P0_CUDA_8X8_PASS
P0_TORCHRL_API_IMPORT_PASS
```

Real definitions (not aliases), from `inspect`:

| Symbol | Module | `__init__` parameters |
|---|---|---|
| `AsyncBatchedCollector` | `torchrl.collectors._async_batched` | `create_env_fn, policy, policy_factory, frames_per_batch, total_frames, max_batch_size, min_batch_size, server_timeout, transport, device, backend, env_backend, env_exchange, envs_per_worker, transition_chunk_size, policy_backend, reset_at_each_iter, postproc, exploration_type, replay_buffer, post_collect_hook, yield_completed_trajectories, weight_sync, weight_sync_model_id, verbose, create_env_kwargs, worker_affinity, driver_affinity, server_config, device_config, policy_version, policy_version_key` |
| `InferenceServer` | `torchrl.modules.inference_server._server` | `model, transport, policy_factory, service_backend, service_backend_options, transport_options, request_spec, response_spec, num_clients, max_batch_size, static_batch_size, min_batch_size, timeout, collate_fn, device, policy_device, output_device, collect_stats, stats_window_size, weight_sync, weight_sync_model_id, server_config, device_config, shutdown_event, policy_version, policy_version_key` |
| `ProcessSlotTransport` | `torchrl.modules.inference_server._process_slot` | `request_spec, response_spec, num_slots, ctx, copy_result` |
| `EnvBase` | `torchrl.envs.common` | `device, batch_size, run_type_checks, allow_done_after_reset` |

### Known limitations (explicit)

- This proves **import availability** only. It does not prove action-contract
  compatibility, a custom `EnvBase`, collector throughput or process-transport
  reliability under load. Those are P1–P3 obligations.
- `AsyncBatchedCollector` exposes `policy_version` / `policy_version_key`
  parameters, which are relevant to the P6 staleness contract; this was recorded
  as an observation only and is not exercised in P0.
- TorchRL was **not** re-exported, patched or forked. No version sweep was performed
  beyond the single resolution above.

---

## 3. OpenCode DeepSeek-only routing (P0-C)

| Item | Measured value |
|---|---|
| OpenCode CLI | `opencode v2.1.2` (Windows-native install) |
| Config location | user profile config `~/.config/opencode/opencode.json` (project-local overlay not required) |
| `pa4-lead` | `primary`, model `deepseek/deepseek-flash`, write-capable |
| `pa4-auditor` | `subagent`, model `deepseek/deepseek-flash`, permission `"*": "deny"` with only `read`/`glob`/`grep`/`list` allowed |
| DeepSeek credentials | reused existing authentication; **no** key duplicated in JSON, logs or receipts |

### G0-C live tool-call smoke (measured)

| Step | Actor | Tool | Observed result |
|---|---|---|---|
| 1 | Lead (`pa4-lead`) | write | created ignored scratch `tmp/p0_agent_smoke.txt` containing `PA4_P0_DEEPSEEK_TOOL_SMOKE_OK` |
| 2 | Lead | read | read back exact marker from disk |
| 3 | Auditor (`pa4-auditor`, fresh session) | read | reported the exact marker content |
| 4 | Auditor | edit | **denied** |
| 5 | Auditor | shell (`bash`) | **denied** |
| 6 | Lead | shell | removed only the generated scratch file |

Effective auditor permissions verified against the live config:
`"*": "deny"` → `edit` and `bash` are not permitted; `task` is denied so the auditor
cannot spawn a nested writable agent. The auditor is not given `external_directory`
access, so it cannot reach outside the PA4 workspace.

### Known limitations (agent boundary)

- `pa4-lead` and `pa4-auditor` run the **same model**. Context separation is a
  process control, **not** independent model verification. Objective correctness is
  decided only by automated tests and PA3 reference contracts.
- Qwen / MiniDC / Ornith / multi-model delegation were not configured or tested.

---

## 4. Reproduction commands

```bash
cd ~/projects/PokeAgent4.0
uv sync --frozen
uv run --frozen python scripts/p0_smoke.py
sha256sum uv.lock
```
