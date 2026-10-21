# P1 Handoff

Source: `docs/phases/P0/SOURCE_FREEZE.md` §4 (PA3 @ `65bc980abd8e0bbec14158b9ddc2884bb1726a4e`).
**P1 is authorized** to build the smallest working proof of PA3 legal conditional action
sampling and exact joint logprob on the pinned stack. P1 is **not** authorized to build the
full Rust `EnvBase`, the collector, PPO, or Runpod infrastructure before G1.

Exactly three questions must be answered in P1:

1. **Can the TorchRL policy server sample a complete legal conditional player action
   without changing the Rust legality rules?**
   The Rust engine must remain the sole authority on legality; the policy may only assign
   probability over the candidate tables the engine supplies.

2. **Is a single batched request feasible, or is a staged prefix protocol required?**
   PA3 documents that branch *j*'s mask is computed *after* the branches selected before it
   (`LevelSamplingResult`, `agent/model/pa3_model.py`), so determine whether one request can
   carry all branch candidate tables or whether batched prefix-by-prefix round trips are
   required.

3. **What minimum branch metadata is necessary for exact behavior-logprob recomputation?**
   Given `joint_logprob = sum(branch logprob)` over valid branches and
   `actor_active = (branch_k >= 2).any()`, determine the minimal persisted set (per-branch
   candidate ids/masks, `branch_k`, capacity/padding, policy version, observation contract
   hash) that allows the stored FP32 logprob to be recomputed exactly.

## P1 entry checklist

- [ ] Install Rust `1.90.0` (from PA3 `rust-toolchain.toml`) in WSL2.
- [ ] Build `pa3-engine` with the `python` feature (PyO3 0.26).
- [ ] Reuse the pinned stack: torch `2.9.1+cu128`, torchrl `0.14.1`, tensordict `0.14.3`.
- [ ] Keep PA3 read-only at the pinned SHA; do not modify PA3.
- [ ] Decide A-path (full legal candidate graph, one GPU call) vs B-path (staged prefix),
      or record `BLOCKED` and propose an ADR if `InferenceServer` cannot carry it.

## What P0 did not establish

P0 proves import/API availability and device access only. It does **not** prove action
compatibility, custom `EnvBase` contract correctness, collector throughput or
process-transport reliability.
