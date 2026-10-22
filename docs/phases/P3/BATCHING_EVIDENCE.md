# P3 — Batching Evidence

Document ID: `PA4-P3-BATCHING-EVIDENCE`
Design: `PA4-P3-DESIGN-v1.0` §4.1, G3-B
GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 610.88, capability (8,9)
Stack: torch 2.9.1+cu128, torchrl 0.14.1, tensordict 0.14.3

Every number below is measured and is retained either in
`docs/phases/P3/evidence/` or in the named unit test. An earlier revision of this document quoted
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
guards it. The legacy `run_collector.py` (which used the aliasing pattern) was
removed.

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

The retained 4-driver baseline reaches `max_worker_ids_per_forward = 4` and 180
mixed forwards; the 2-driver baseline reaches `max_worker_ids_per_forward = 2`
and 233 mixed forwards (see `evidence/baseline_multi{2,4}x16_central.json`).

### 2.3 Forward-batch histograms

From `PERFORMANCE_BASELINE.csv` and the retained per-run JSON. `forwards` counts
**STEP** model forwards only; encoding forwards are a separate column in the
JSON (`encode_forward_count`).

| config | forwards | bmax | mean batch |
|---|---|---|---|
| 16 envs, 1 driver | 624 | 29 | 5.88 |
| 32 envs, 1 driver | 596 | 61 | 7.17 |
| 64 envs, 1 driver | 588 | 64 | 8.40 |
| 2 drivers, central | 1019 | 62 | 4.02 |
| 4 drivers, central | 4332 | 64 | 1.82 |

`model_device` is `cuda:0` in every run; the service stamps a sha256 weight
fingerprint and the client rejects a mismatch.

---

## 3. Throughput and its transport cost (measured, both topologies)

`PERFORMANCE_BASELINE.csv` (measured at commit `5c9096a`):

| config | matches/s (wall) | steady matches/s | transitions/s |
|---|---|---|---|
| 16 envs, 1 driver | 5.12 | 7.86 | 79.3 |
| 32 envs, 1 driver | 5.47 | 8.88 | 96.6 |
| 64 envs, 1 driver | 5.21 | 9.71 | 104.6 |
| 2 drivers × 16 envs, **central** | 1.24 | 0.81 | 21.3 |
| 2 drivers × 16 envs, **per-driver servers** | 5.70 | 6.19 | 92.5 |
| 4 drivers × 16 envs, **central** | 1.05 | 0.32 | 17.6 |
| 4 drivers × 16 envs, **per-driver servers** | 7.19 | 3.27 | 122.1 |

**Finding.** Cross-process central batching is correct and observable, but on
this host it is 4.60–6.85× slower on the wall measure (5.70/1.24 and 7.19/1.05)
and 7.64–10.22× slower on the per-driver steady mean (6.19/0.81 and 3.27/0.32).
Both topologies use the same server batching policy (`min_batch=2`,
`timeout=0.02`) and the same model. Per-match stage time rises in **both** the
encode round-trip and the staged forward phases (section 3.1), so the split
between the distributed transport (a gloo/TCP pair per client; shared-memory
`ProcessSlotTransport` cannot serve two processes) and service/CPU contention
was **not isolated**. This is a transport-selection input for P6/P7, not a
model-compute conclusion.

The multi-driver wall clock includes process startup. Drivers run concurrently,
so their steady windows cannot be added into one wall window: the multi steady
column is the **mean of the per-driver steady rates**, and the full per-driver
windows are retained in each run's JSON. The wall rate is the aggregate figure.

### 3.1 Stage timing (retained, first-hand)

`run_batched` records wall seconds per stage in every baseline JSON, and
`BASELINE_STAGE_TIMING.csv` collects the same numbers (aggregate and per-driver
for multi-driver runs). Example, 64 envs / 1 driver
(`evidence/baseline_bl_64_16.json`):

| stage | seconds |
|---|---|
| request_info (native kinds+branches, one call) | 0.003 |
| views (reuse env-decoded observations) | 0.107 |
| encode (server round-trip + CUDA encode) | 2.241 |
| step_levels (all staged STEP phases) | 7.152 |
| packet assembly | 0.328 |
| `env.step` (native advance) | 0.320 |
| accounting + transition emission | 0.259 |
| reset | 0.195 |

The server round-trip dominates; native stepping is a small fraction. With four
central drivers the same stages grow per match (encode 0.80 s/match, STEP phases
2.71 s/match) versus the four per-driver servers (0.10 s/match and 0.23 s/match),
which is why the transport-vs-service split is not attributed to one cause.

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
