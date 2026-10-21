# P3 — Batching Evidence

Document ID: `PA4-P3-BATCHING-EVIDENCE`
Design: `PA4-P3-DESIGN-v1.0` §4.1, G3-B
GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 610.88, capability (8,9)
Stack: torch 2.9.1+cu128, torchrl 0.14.1, tensordict 0.14.3

All numbers below are measured. Nothing is projected.

---

## 1. The defect that blocked P3-B for three attempts

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

Measured proof:

```
srv.client()  -> ids 0, 0   (same object)
srv.clients(3)-> ids 1, 2, 3 (distinct)
```

**Fix:** use `srv.clients(n)`. This single line was the difference between
`forward_batch_sizes: [1,1,1,...]` and `[4,4,4,...]`.

---

## 2. Genuine B >= 2 forward, with per-row correctness

Test: `tests/collector/test_batching.py::test_one_forward_covers_multiple_independent_requests`

Four independent `PA4EnvBase` instances, each ENCODEs its own observation, then
all four STEP rows are submitted together:

```
assert svc.model_forward_count == 1
assert svc.forward_batch_sizes == [4]
assert (out["error_code"] == 0).all()
# each row's selected index lies inside ITS OWN candidate set
```

Result: **PASS**. One model forward, four independent requests, four correct
per-row selections.

Additional measured evidence from the instrumented service during real
collection (`/tmp/pa4_p3_svc_*.json`):

| Metric | Observed |
|---|---|
| `model_device` | `cuda:0` |
| `forward_batch_sizes` | max **64** |
| mean batch (n=64, tuned) | **15.5** |
| `cache_high_water` | scales with live requests, not matches |

---

## 3. Why bigger batches did not make it faster (measured, then explained)

Batch accumulation was tuned directly:

| envs | min_batch | timeout | forwards | mean batch | matches/s |
|---|---|---|---|---|---|
| 64 | 1 | 0.0 | 874 | 5.4 | 4.42 |
| 64 | 16 | 0.002 | 504 | 9.3 | 4.23 |
| 64 | 64 | 0.003 | 478 | 10.3 | 4.46 |
| 64 | 128 | 0.005 | 397 | 12.5 | 4.45 |
| 64 | 256 | 0.008 | 318 | 15.5 | 4.42 |

Forward count fell **2.7x** while throughput stayed flat. **The GPU is not the
bottleneck.** Direct measurement confirms it:

| Operation | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| `model.encode` | 4.80 ms | 5.47 ms | 7.09 ms | 11.53 ms |
| `sample_level_step` | 2.60 ms | 2.40 ms | 2.46 ms | — |

Both are nearly **flat in batch size** — batching is essentially free on this GPU
at these widths. Peak GPU allocation: **0.09 GiB**.

**Conclusion: the remaining cost is TorchRL `ProcessSlotTransport` round-trip
overhead (~40 ms per forward beyond GPU work), not model compute, not payload
size (63.7 KiB/row ≈ 40 µs at shared-memory bandwidth), and not this code.**
This is a genuine P3 finding and a direct input to P6/P7.

---

## 4. Process topology

```
┌──────────────────────────────────────────────────────────────┐
│ driver process (one per collector)                           │
│                                                              │
│  PA4EnvBase(batch_size=N)   owns NativeEngine + N Rust slots │
│         │                                                    │
│  StagedInference            owns N distinct client slots     │
│    phase 1: ENCODE every requesting seat (one batched pass)  │
│    phase 2..L: STEP every pending request at level j         │
│    phase L+1: RESET every finished request                   │
│         │                                                    │
│  env.step(packet)  ← the ONLY state transition               │
└─────────────────────┬────────────────────────────────────────┘
                      │ ProcessSlotTransport (CPU slots)
                      ▼
        InferenceServer (process backend)
        PA3Model on cuda:0, eager FP32, frozen weights
```

Ownership matches the design §2.2: native handles never leave the driver; the
server never sees the opponent's view; legality comes only from Rust.

`test_transport.py::test_clients_have_distinct_slots` guards the slot defect from
recurring.

---

## 5. Negative controls (all fail closed)

| Case | Result |
|---|---|
| duplicate ENCODE for the same identity | `ERR_STALE_IDENTITY` |
| `candidate_count = 65` (overflow) | `ERR_OVERFLOW` |
| `candidate_count` != mask sum | `ERR_COUNT_MISMATCH` |
| unknown request key | `ERR_NO_CACHE` |
| STEP at the wrong level | `ERR_NO_CACHE` **and the cache entry is evicted** |
| stale `native_generation` | `ERR_NO_CACHE` |
| RESET | evicts only the addressed entry |
| illegal action submitted to the env | raises |
| stale `action_request_id` / `native_generation` | raises |
| two in-flight requests on one client | raises (transport-enforced) |
