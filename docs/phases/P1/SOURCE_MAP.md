# P1 — Source Map

Document ID: `PA4-P1-SOURCE-MAP`
PA3 pin: `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`
PA4 base commit: `a900f4793afb4417c1f3226a5d423aca9c038e43`

Records exactly how PA3 source is used, and what was rebuilt rather than reused.
No PA3 file was modified.

---

## 1. PA3 usage

| Role | PA3 path | How PA4 uses it |
|---|---|---|
| Native engine (PyO3) | `engine/src/*` | **Built unmodified** at the pinned SHA into `pa3_engine.so`; imported as a package |
| Wire decoders | `engine/python/pa3_engine/observation.py` | Imported unmodified (`parse_batch`) |
| Observation contract | `agent/types/observation.py` | Imported unmodified (`ObservationBatch`, `ObservationLayout`) |
| Action types | `agent/types/actions.py` | Imported unmodified (`AtomicAction`) |
| Candidate batch | `agent/types/requests.py` | Imported unmodified (`ActionRef`, `BranchCandidatesBatch`) |
| Policy model | `agent/model/pa3_model.py` | Imported unmodified (`PA3Model`, `EncodedState`) |
| Scorer / masking | `agent/model/scorer.py` | Imported unmodified (`masked_log_softmax`, `probabilities`) |
| Model config | `agent/model/config.py` | Imported unmodified (`PA3Config`) |

PA3 is added to `sys.path` read-only. PA4 does **not** vendor, copy or patch PA3.

## 2. Build location (PA3 untouched)

PA3 at `/home/pa4/projects/_pa3_ref` stays a clean checkout
(`git status --porcelain` = 0 changed files, HEAD = pinned SHA).

To avoid writing build artifacts into it, a scratch copy is used:

```
/home/pa4/.pa3_build            # build copy, not tracked, not in PA3
/home/pa4/.pa3_build/target/release/libpa3_engine.so
/home/pa4/.pa3_build/engine/python/pa3_engine/pa3_engine.so
```

Built with the toolchain PA3 pins (`rust-toolchain.toml` → **1.90.0**) and the
existing Cargo `python` feature (PyO3 0.26). No Cargo feature was added.

Import path used by PA4:

```python
sys.path.insert(0, "/home/pa4/.pa3_build/engine/python")  # package dir
import pa3_engine            # native ext + parse_batch/parse_candidates
```

**Correction recorded:** an earlier attempt imported a bare `pa3_engine.abi3.so`
file. That module lacks the wire decoders, and `import pa3_engine.observation`
failed with "not a package". The package-directory layout above is required and
is what PA3's own `scripts/build_python.sh` produces.

## 3. New PA4 code (this phase)

| File | Purpose |
|---|---|
| `pa4/actions/native_reference.py` | Sole legality gateway to the Rust engine; contract assertion |
| `pa4/actions/conditional_policy.py` | Staged prefix sampling, replay record, weight/observation digests |
| `pa4/actions/server_probe.py` | Real `InferenceServer` service: ENCODE/STEP/RESET + cache |
| `pa4/actions/server_factory.py` | Module-level picklable `policy_factory` (closures cannot cross the spawn boundary) |
| `tests/action_distribution/test_native_legality.py` | G1-A |
| `tests/action_distribution/test_conditional_parity.py` | G1-B / G1-C |
| `tests/action_distribution/test_inference_gateway.py` | G1-D |

## 4. Deliberate non-reuse / non-implementation

- `build_branch_table` is a **faithful re-expression** of PA3's
  `NativeCollector._level_table`, because the collector itself (and its config,
  device handling and telemetry) is P3 scope. The mask/capacity/fail-loud
  semantics are copied exactly; the surrounding collector is not.
- `sample_levels()` was **not** used inside the server for multi-level stepping:
  it restarts the GRU from `initial_state()` on every call, so a per-level server
  call would silently discard the prefix state. The server calls
  `sample_level_step` and carries `hidden` explicitly. The single-call local path
  still uses `sample_levels` and agrees with the server (see parity tests).
- No PPO, no `EnvBase`, no collector, no optimizer, no checkpoint I/O.

## 5. Dependency stack (unchanged)

`uv.lock` sha256 `8f414f526aad676f6466099da9bf96240f739abae9b9376a5c32fe845340d96e`
— identical to P0. Tests use stdlib `unittest` because `pytest` is not in the
frozen lock and P1 must not change dependencies.
