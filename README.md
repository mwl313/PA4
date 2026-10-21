# PokeAgent 4.0

PokeAgent 4.0 (PA4) is a TorchRL/TensorDict-based rework of the PokeAgent 3.0
Rust battle engine pipeline. PokeAgent 3.0 remains the read-only reference for
game rules, observations, actions and PPO contracts.

**Current phase: P0 (Fast-Track Foundation).** P0 pins the development host,
dependency stack, DeepSeek-only agent routing and the PA3 reference SHA. P0 does
**not** implement a battle environment, action sampler or PPO learner.

## Status

See `docs/PROJECT_STATUS.md` for the live gate status and
`docs/phases/P0/RECEIPT.json` for measured evidence.

## Setup (WSL2 Ubuntu 24.04)

```bash
# 1. Install the frozen dependency stack into a local .venv
cd ~/projects/PokeAgent4.0
uv sync --frozen

# 2. Run the P0 smoke: pinned API imports + TensorDict 2x4 + 8x8 CUDA tensor
uv run python scripts/p0_smoke.py

# 3. Confirm the exact locked versions
uv run python -c "import torch,torchrl,tensordict;print(torch.__version__,torchrl.__version__,tensordict.__version__)"
```

## Boundaries

- PokeAgent 3.0 is **read-only** and pinned by commit SHA (see `docs/phases/P0/SOURCE_FREEZE.md`).
- No cloud GPU resources are used in P0.
- No API keys or local credentials are committed; see `.gitignore`.
