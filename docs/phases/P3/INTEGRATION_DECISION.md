# P3 — Integration Decision

Document ID: `PA4-P3-INTEGRATION-DECISION`
Design: `PA4-P3-DESIGN-v1.0` (sha256 `f73c556794bb406c1bd322a093fe7c05f0f330b32b3763d69c492266e313563e`)
Installed stack: torchrl `0.14.1`, tensordict `0.14.3`, torch `2.9.1+cu128`

---

## 1. Preflight (measured)

| Item | Value |
|---|---|
| Repo | `/home/pa4/projects/PokeAgent4.0`, branch `main` |
| HEAD at start | `4d8f349b247bb929914db5b9ebe2c5ce84c18288` |
| Worktree | clean |
| Python | 3.12.3 |
| Rust | 1.90.0 |
| torch | 2.9.1+cu128 |
| torchrl | 0.14.1 |
| tensordict | 0.14.3 |
| `uv.lock` sha256 | `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e` |
| `uv lock --check` | pass ("Resolved 37 packages") |
| PA3 reference | `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`, 0 dirty files |
| GPU in WSL | `NVIDIA GeForce RTX 4060 Ti`, 8188 MiB, driver 610.88 |
| torch CUDA | available; capability (8,9); 6.90 GiB free of 8.00 GiB |
| P1 regression | **39 tests OK** |
| P2 regression | **51 tests OK** |
| P0 smoke | exit 0 (`P0_CUDA_8X8_PASS`) |

Design copied verbatim to `docs/phases/P3/DESIGN.md`; source and destination
sha256 match exactly.

---

## 2. P3-A result: the installed collector cannot drive a staged conditional policy

### 2.1 What the installed API actually does

Inspected source:
`torchrl/collectors/_async_batched.py` (2099 lines, see evidence: sha256 bf5ba4b31e4be0c71adc569bd3fa3db5ce6704ccdd81edfe1ef478f35be47c96).

The acting protocol, from `_process_env_loop` (line ~190) and `_env_loop` (line ~135):

```python
env = env_factory(**create_env_kwargs)
observation = env.reset()
while not shutdown_event.is_set():
    policy_output = client(observation)              # ONE call, obs -> action
    action_td = observation.update(policy_output)
    transition, observation = env.step_and_maybe_reset(action_td)
```

and for the batched coordinator, `_env_batch_loop` (line ~298):

```python
pending_actions[env_id] = clients[env_id].submit(observation)
```

**The contract is strictly `observation -> action`, one exchange per env step.**

`client` is created by `transport.client()` (line 156) or is the user-supplied
`policy` callable. It receives **only a TensorDict**. It holds no reference to
the env, and `PolicyClientModule` is explicitly documented as a *remote*
transport client that "carries no lifecycle rights".

### 2.2 Why that is incompatible with P2/P1

- One P2 `EnvBase.step()` requires a **complete six-key action packet for both
  seats**, built from several staged `ENCODE`/`STEP`/`RESET` exchanges.
- Each staged `STEP` needs the Rust engine's legal candidate list **for the
  current prefix**, which changes after every pick
  (`candidates_batch(handle, side, prefix)`).
- The collector never calls `env.rand_action`, never consults an env-local
  policy, and offers no callback between "observation available" and
  "action required".

So a client in the collector's position cannot know the legal candidate list for
level 0, let alone level 3, without access to the live `NativeEngine` — which the
collector does not provide.

### 2.3 Reproduced experimentally (not merely reasoned)

Three probes were run inside the repo with the real `PA4EnvBase`:

| Probe | What it did | Result |
|---|---|---|
| 1 | `AsyncBatchedCollector(create_env_fn=[PA4EnvBase], policy=trivial_client)` | Collector **constructs**; `next(iter(col))` raises |
| 2 | Same, with a process-local registry so the client could look up the worker's env | Registry is populated in the factory, but the **client is never invoked** — no `[CLIENT INVOKED]` log line |
| 3 | Direct contract check without the collector | P2 obs batch `[1]`, packet batch `[1]`, `action (1,2,4,6)` — well-formed |

The immediate exception, reproduced identically in probes 1 and 2:

```
RuntimeError: A collector worker thread raised an exception.
  cause: Received a new batch size torch.Size([1, 1]) with an existing
  batch_size torch.Size([1]) in a lazy TD. Modifying the batch size of a lazy
  representation of a tensordict is not permitted.
```

This is a real shape incompatibility between P2's env output and the
collector's internal `step_and_maybe_reset` lazy-stack handling, and it fires
**before any policy is consulted**. Fixing it would be possible in principle,
but it would not change the finding in §2.1–2.2: even with shapes corrected, the
client would still have no access to the native candidate API, so
`G3-A`'s requirement — "actions ... chosen by a P1 remote conditional scorer" —
would remain unsatisfiable through this path.

### 2.4 What was deliberately not done

- Did not fork or patch TorchRL.
- Did not write a generic Collector replacement.
- Did not feed the collector a dummy/random policy and call it a pass.
- Did not weaken G3-A.

Per the design's stop-early rule (§1.3, P3-A), one focused compatibility
correction was attempted (the registry route in probe 2). It failed for a
structural reason, so the fallback ADR is raised instead of continuing.

---

## 3. ADR: P3-F conditional fallback (requires user approval)

**Status: APPROVED by the user (option B, 2026-10-22) and implemented in
`pa4/collector/`, with the gaps recorded in `RECEIPT.json` (`audit_findings`).**
The text below is the original proposal, kept verbatim for history.

**Decision requested.** Approve a narrow, TorchRL-compatible actor loop that
owns the staged acting protocol, uses the real `InferenceServer` +
`ProcessSlotTransport` for central CUDA policy evaluation, emits ordinary
TensorDict transitions, and preserves every P2/P1 contract. It does not fork or
rewrite `AsyncBatchedCollector`.

**Consequence for the gate, stated plainly.** A fallback-only result is
**PARTIAL against the preferred integration requirement**, exactly as
§1.3 and §6 require. It cannot be reported as G3-A PASS unless the user approves
this ADR and records the adapted master criterion.

**Why this is the minimal option.** The alternative — making the collector
aware of a live Rust engine mid-step — requires either a TorchRL fork (forbidden
by §1.3 and §6.2.2) or moving legality into Python (forbidden by invariant 1).

---

## 4. Where P3 stands

- **P3-0 preflight:** complete, all values above measured.
- **P3-A:** ADR-008 was **approved** (option B) and the actor-aware adapter is
  implemented; G3-A is a scoped PASS.
- **P3-B / P3-C / P3-D:** implemented in part. Batching is proven per driver;
  cross-worker central batching, bounded transition retention, weight
  fingerprints and the cold/steady breakdown are **not** implemented. See
  `RECEIPT.json` under `audit_findings`.

**P3 verdict: PARTIAL** (audited). The earlier "BLOCKED pending the §3 decision"
was superseded when the user approved ADR-008.
