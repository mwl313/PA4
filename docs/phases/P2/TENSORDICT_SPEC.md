# P2 — TensorDict Specification

Document ID: `PA4-P2-TENSORDICT-SPEC-v1.0`
Design: `PA4-P2-DESIGN-v1.0`
Env: `pa4/env/native_env.py` → `PA4EnvBase(EnvBase)`
Installed stack: torchrl `0.14.1`, tensordict `0.14.3`, torch `2.9.1+cu128`

All shapes include the leading EnvBase batch dimension `N` (default `N=1` in
tests, `N=2` supported and exercised for partial reset).

---

## 1. Observation (env output)

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `players.token_mask` | `[N,2,96]` | `bool` | Per-seat visible tokens |
| `players.categories` | `[N,2,96,32]` | `int64` | Category ids |
| `players.category_known` | `[N,2,96,32]` | `bool` | Knownness mask |
| `players.floats` | `[N,2,96,50]` | `float32` | Continuous features |
| `players.float_known` | `[N,2,96,50]` | `bool` | Knownness mask |
| `players.flags` | `[N,2,96,40]` | `bool` | Binary state |
| `players.flag_known` | `[N,2,96,40]` | `bool` | Knownness mask |
| `players.role_ids` | `[N,2,96]` | `int64` | Layout metadata (PA3 decoder) |
| `players.side_ids` | `[N,2,96]` | `int64` | Layout metadata (PA3 decoder) |
| `players.observation_valid` | `[N,2]` | `bool` | Whether this seat has a player-facing view |
| `players.request_kind` | `[N,2]` | `int64` | 0=Preview,1=Normal,2=Replacement,3=Wait,4=Finished |
| `players.decision_required` | `[N,2]` | `bool` | True only for kinds 0/1/2 |
| `players.request_id` | `[N,2]` | `int64` | Increments on every native advance and every revision; restarts at `(1,1)` when a slot is reset |
| `players.branch_count` | `[N,2]` | `int64` | Real native branch count; 0 for Wait/Finished |
| `native_slot` | `[N]` | `int64` | Handle traceability (not a policy feature) |
| `native_generation` | `[N]` | `int64` | Handle lifetime |
| `episode_id` | `[N]` | `int64` | Strict per-slot accounting |
| `step_index` | `[N]` | `int64` | Strict per-slot accounting |
| `request_revised` | `[N]` | `bool` | Native accepted=False with no error |
| `match_completed` | `[N]` | `bool` | At most once per natural episode |
| `winner` | `[N]` | `int64` | 0, 1, or -1 (draw/unknown) |

## 2. Action (env input)

| Key | Shape | dtype |
|---|---|---|
| `action` | `[N,2,4,6]` | `int64` |
| `action_branch_count` | `[N,2]` | `int64` |
| `action_present` | `[N,2]` | `bool` |
| `action_request_id` | `[N,2]` | `int64` |
| `action_episode_id` | `[N,2]` | `int64` |
| `action_native_generation` | `[N,2]` | `int64` |

`action` rows are canonical six-field native tuples
`(kind, own_slot, move_slot, target_location, switch_destination, resource)`.
`target_location` is **signed** and never clamped. Rows beyond
`action_branch_count` are never submitted.

**All six keys are required.** A packet missing `action_episode_id` or
`action_native_generation` is rejected rather than accepted with stale-packet
detection silently disabled. For each requesting seat the env checks
`request_id`, `episode_id` and native `generation` against the current handle,
requires the supplied prefix length to equal the native branch count, and checks
every row against the engine's ordered candidate list for that prefix. `Wait` /
`Finished` seats must supply `present=False` and `branch_count=0`. Any violation
raises `RuntimeError` before `step_batch` is called, so an invalid packet can
never be absorbed as a native "revision".

### 2.1 Post-terminal contract

After a row reports `done`, that row has no native transition left. A further
`step` is legal only with `action_present=False` and `action_branch_count=0` for
that row; the env re-reports `done=True` together with the terminal
`terminated`/`truncated` flags and `winner` recorded when the episode actually
ended. It never synthesizes a different termination (in particular, a truncated
episode never becomes `terminated=True` afterwards). Any submission for a
concluded row raises.

## 3. Reward / done

| Key | Shape | dtype |
|---|---|---|
| `reward` | `[N,2,1]` | `float32` |
| `done` | `[N,1]` | `bool` |
| `terminated` | `[N,1]` | `bool` |
| `truncated` | `[N,1]` | `bool` |

Terminal reward: winner `0` → `(+1,-1)`; winner `1` → `(-1,+1)`; draw/unknown →
`(0,0)`; every intermediate step `(0,0)`.

## 4. Spec construction notes (measured, not assumed)

- **Boolean leaves are `bool`, and the spec says `bool`.** torchrl 0.14.1
  `Binary` *defaults* to `int8` (it does accept `dtype=`), so the earlier
  `Binary(...)` spec produced `fake=torch.int8 and real=torch.bool` in
  `check_env_specs`. Boolean leaves now use
  `Unbounded(shape=..., dtype=torch.bool)`, which matches the truthful emitted
  dtype. The fix aligns the spec to the data rather than casting the data to
  hide the mismatch.
- The nested `players` `Composite` must carry the **full** shape including the
  outer batch dim (`[N,2]`), not just `[2]`, or TorchRL raises a shape mismatch
  against the parent spec.
- `rand_action()` returns the legal engine-sourced packet **merged into the
  passed tensordict**, because `check_env_specs` calls it with the live
  tensordict.

## 5. Verified

```
uv run --frozen python -c "from torchrl.envs.utils import check_env_specs; ..."
[torchrl][INFO] check_env_specs succeeded!
```

## 6. Privacy boundary

`players` is an **internal two-seat** structure for the action driver.
`player_view(seat)` returns only that seat's own decoded observation plus its own
request metadata. It never returns the other seat's view, engine-private state,
RNG state, teams or the other seat's action history. Unknown opponent fields stay
masked via `*_known`, they are not zero-filled as known.
