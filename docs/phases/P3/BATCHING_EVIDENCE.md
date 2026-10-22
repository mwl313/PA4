# P3 — Batching Evidence

Document ID: `PA4-P3-BATCHING-EVIDENCE`
Design: `PA4-P3-DESIGN-v1.0` §4.1, G3-B
GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 610.88, capability (8,9)
Stack: torch 2.9.1+cu128, torchrl 0.14.1, tensordict 0.14.3

Every number below is measured and has a retained artifact in
`docs/phases/P3/evidence/`. An earlier revision of this document quoted
measurements whose raw files were not retained (a 15.5 mean-batch sweep, a
0.09 GiB peak, GPU op timings and a "~40 ms/forward" residual). Those claims
have been **removed**; they are replaced by the retained baseline below.

---

## 1. The defect that blocked P3-B

`ProcessInferenceServer.client()` **caches a single client**:

```python
def client(self) -> Any:
    if self._service_client is None:
        self._service_client = self.transport.client()
    return self._service_client
```

So `[srv.client() for _ in range(n)]` yields **n references to slot 0**.
`ProcessSlotTransport` permits one in-flight request per client, so only one
request could ever be outstanding and **every forward was B=1**.

Measured: `client() -> 0,0` (same object); `clients(3) -> 1,2,3` (distinct).

**Fix:** `srv.clients(n)`. `tests/collector/test_transport.py::test_clients_have_distinct_slots`
guards it, and `run_collector.py` no longer uses the aliasing pattern.

---

## 2. Genuine B ≥ 2, per driver and across worker processes

### 2.1 One forward, four independent requests (unit proof)

`tests/collector/test_batching.py::test_one_forward_covers_multiple_independent_requests`:
four independent envs, four ENCODE rows, then four STEP rows submitted together.

```
model_forward_count == 1
forward_batch_sizes == [4]
every row: branch_k == its own candidate_count
every row: request_key echo == its own worker index
selected_index lies inside its own candidate set
```

### 2.2 One forward mixing **two and four worker processes** (central server)

`tests/collector/test_matches.py::MultiContextTest` runs two spawned driver
processes against **one** parent-owned `InferenceServer` using
`transport="distributed"`. Each driver creates its own actor-side clients from
the pickled transport. Requests carry the driver's `run_id`, and the
instrumented service records the run-id set of every forward.

Measured (seed 21, 2 drivers × 8 envs):

```
max_worker_ids_per_forward >= 2
cross_worker_forwards >= 1
```

The retained 4-driver baseline reaches `max_worker_ids_per_forward = 4` and 190
mixed forwards (see `evidence/baseline_multi4x16_central.json`).

### 2.3 Forward-batch histograms

From `PERFORMANCE_BASELINE.csv` and the retained per-run JSON:

| config | forwards | bmax | mean batch |
|---|---|---|---|
| 16 envs, 1 driver | 358 | 29 | 5.85 |
| 32 envs, 1 driver | 355 | 61 | 6.97 |
| 64 envs, 1 driver | 395 | 64 | 8.45 |
| 2 drivers, central | 896 | 62 | 4.54 |
| 4 drivers, central | 4944 | 63 | 1.71 |

`model_device` is `cuda:0` in every run; the service stamps a sha256 weight
fingerprint and the client rejects a mismatch.

---

## 3. Throughput and its transport cost (measured, both topologies)

`PERFORMANCE_BASELINE.csv` (commit `62c2b2f`):

| config | matches/s (wall) | steady matches/s | transitions/s |
|---|---|---|---|
| 16 envs, 1 driver | 4.02 | 8.46 | 68.1 |
| 32 envs, 1 driver | 3.79 | 9.17 | 79.7 |
| 64 envs, 1 driver | 3.78 | 12.13 | 94.1 |
| 2 drivers × 16 envs, **central** | 1.21 | 1.53 | 21.1 |
| 2 drivers × 16 envs, **per-driver servers** | 5.40 | 11.16 | 93.1 |
| 4 drivers × 16 envs, **central** | 0.97 | 1.12 | 17.3 |
| 4 drivers × 16 envs, **per-driver servers** | 5.48 | 8.11 | 98.4 |

**Finding.** Cross-process central batching is correct and observable, but on
this host it is ~4.5–5.6× slower than one server per driver (5.40/1.21 and
5.48/0.97). The distributed transport opens a gloo/TCP pair per client, and
that cost dominates at 32–64 slots. `ProcessSlotTransport` (shared memory) is
faster but cannot serve two processes. This is a transport-selection input for
P6/P7, not a model-compute limit.

The multi-driver wall clock includes process startup; the steady columns use
each driver's own post-warmup window, and are the fair comparison.

### 3.1 Stage timing (retained, first-hand)

`run_batched` records wall seconds per stage in every baseline JSON. Example,
64 envs / 1 driver (`evidence/baseline_bl_64_16.json`):

| stage | seconds |
|---|---|
| request_info (native kinds+branches, one call) | 0.002 |
| views (reuse env-decoded observations) | 0.071 |
| encode (server round-trip + CUDA encode) | 1.515 |
| step_levels (all staged STEP phases) | 5.049 |
| packet assembly | 0.103 |
| `env.step` (native advance) | 0.244 |
| accounting + transition emission | 0.044 |
| reset | 0.123 |

The server round-trip dominates; native stepping is a small fraction.

### 3.2 Why this is not a 1,000/s claim

The design forbids proving the 1,000/s stretch goal on this GPU. P3 reports
bounded actor-only measurements only, and `accepted_training_matches_per_sec`
is `null` because no learner exists.

---

## 4. Process topology (central mode)

```
parent process
  InferenceServer(transport="distributed", service_backend="process")
      └─ service process: PA3Model on cuda:0, FP32, frozen weights
  driver 0 (spawn)                      driver 1 (spawn)
    PA4EnvBase(N) + NativeEngine          PA4EnvBase(N) + NativeEngine
    clients = transport.client() × 2N     clients = transport.client() × 2N
    env.step() is the only transition     env.step() is the only transition
```

Native handles never leave their driver. The policy receives one seat's view per
request, never the opponent's. Legality comes only from Rust.

---

## 5. Negative controls (all fail closed)

| Case | Result |
|---|---|
| duplicate ENCODE for the same identity (later call) | `ERR_STALE_IDENTITY` |
| **two rows with the same identity in one batch** | both `ERR_STALE_IDENTITY`, no cache write |
| **fabricated identity never encoded** | `ERR_NO_CACHE` |
| `candidate_count = 65` (overflow) | `ERR_OVERFLOW` |
| `candidate_count` != mask sum | `ERR_COUNT_MISMATCH` |
| unknown request key | `ERR_NO_CACHE` |
| STEP at the wrong level | `ERR_NO_CACHE` and the entry is evicted |
| stale `native_generation` | `ERR_NO_CACHE` |
| **same local id under two workers** | two independent entries, never one alias |
| **mixed policy version** | `ERR_STALE_IDENTITY`, live entry intact |
| **wrong weight fingerprint** | `ERR_STALE_IDENTITY` |
| RESET | evicts only the addressed entry |
| illegal action submitted to the env | raises |
| stale `action_request_id` / `native_generation` | raises |
| two in-flight requests on one client | raises (transport-enforced) |

---

## 6. How to reproduce

```bash
cd /home/pa4/projects/PokeAgent4.0
uv run --frozen python docs/phases/P3/evidence/run_baseline.py
uv run --frozen python -m unittest discover -s tests/collector -t .
```
