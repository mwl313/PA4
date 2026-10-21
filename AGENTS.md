# AGENTS.md - PA4 working agreement

## Roles

- **`pa4-lead`** - primary agent, DeepSeek (`deepseek/deepseek-flash`). **Sole writable
  implementer.** Runs project tools and tests. Exactly one writable session at any time.
- **`pa4-auditor`** - subagent, same DeepSeek model in a **fresh context**. Read-only:
  no `edit`, no shell, no nested `task`, no write-capable MCP. Configured as default-deny
  with only `read`/`glob`/`grep`/`list` allowed.
- **Objective verifier** - the test processes themselves (`uv sync --frozen`, `p0_smoke.py`,
  `git`/`sha256sum`). A natural-language review never substitutes for a test result.

Same-model review is **not** independent model verification. Correctness is decided by
automated tests and PA3 reference contracts.

## Hard rules

1. Work inside the PA4 repository only. **Never modify PokeAgent3.0** sources,
   production checkpoints, branches or training processes.
2. Never commit API keys, tokens, `auth.json` or local infrastructure credentials.
3. Do not modify the user's global OpenCode configuration or make network/cloud calls that
   cost money without explicit user approval.
4. Work only within the current phase's detailed design. Do not start future phases.
5. No more than one targeted diagnostic correction per blocked gate before reporting BLOCKED.
6. Never relax an acceptance criterion or skip a gate test to obtain PASS.
7. Record only measured facts in receipts. No fabricated versions, hashes or results.

## Required commands

```bash
cd ~/projects/PokeAgent4.0
uv sync --frozen                                   # reproducible environment
uv run python scripts/p0_smoke.py                  # P0 API + TensorDict + CUDA smoke
git diff --check                                   # whitespace check
sha256sum uv.lock                                  # lock evidence
```

## Failure escalation

If a gate cannot pass after one targeted correction, record the exact sanitized error in
`docs/phases/P0/COMPATIBILITY.md` / `RECEIPT.json`, mark the gate `BLOCKED`, stop, and
report to the user. Do not silently substitute CPU results for CUDA/Linux evidence.
