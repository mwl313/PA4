"""P2-C: independent native oracle and lockstep trace comparison.

Creates a SECOND NativeEngine at the same pinned SHA with identical teams/seeds/
roles and replays the exact action tuples the env submitted. Compares at every
boundary:

  * both seats' request kind and branch count
  * ordered legal candidate lists for the selected prefixes
  * both seats' decoded public observations (all 9 leaves, incl. knownness masks)
  * native step result (accepted/terminated/truncated/winner/error)
  * final winner and termination

Never compares PA4 output to itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

OBS_LEAVES = ("token_mask", "categories", "category_known", "floats", "float_known",
              "flags", "flag_known", "role_ids", "side_ids")


@dataclass
class Mismatch:
    field: str
    step: int
    detail: str


@dataclass
class ParityReport:
    steps: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    terminal_matched: bool = False

    @property
    def ok(self) -> bool:
        return not self.mismatches

    @property
    def first_mismatch(self) -> str | None:
        return None if self.ok else self.mismatches[0].detail


class NativeOracle:
    """A completely separate NativeEngine used only for comparison."""

    def __init__(self, module, data_dir: str, teams_json: str, workers: int = 2):
        self.mod = module
        self.engine = module.NativeEngine(data_dir, teams_json, workers)

    def reset(self, team_a, team_b, seeds, role_map):
        return self.engine.reset_batch(team_a, team_b, seeds, role_map)

    def kinds(self, slot, gen):
        return (int(self.engine.request_kind(slot, gen, 0)),
                int(self.engine.request_kind(slot, gen, 1)))

    def branch_counts(self, slot, gen):
        return tuple(len(bytes(self.engine.request_branches(slot, gen, s))) for s in (0, 1))

    def candidates(self, slot, gen, side, prefix):
        out = self.engine.candidates_batch([(slot, gen, side, list(prefix))])
        return [tuple(int(v) for v in row) for row in out[0]]

    def observe(self, slot, gen, side):
        from agent.types.observation import ObservationBatch

        fixed, ragged = self.engine.observe_fixed_batch([(slot, gen)], [side])
        payload = self.mod.parse_batch(fixed, ragged, 1)
        b = ObservationBatch.from_native_payload(payload, dtype=torch.float32)
        return {k: getattr(b, k)[0].cpu() for k in OBS_LEAVES}

    def step(self, slot, gen, choices):
        return self.engine.step_batch([(slot, gen, choices)])[0]


def _cmp_tensor(a: torch.Tensor, b: torch.Tensor) -> str | None:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"dtype {a.dtype} != {b.dtype}"
    if not torch.equal(a.cpu(), b.cpu()):
        diff = (a.cpu() != b.cpu())
        return f"{int(diff.sum())} differing elements"
    return None


def compare_obs(step: int, env_obs: dict, oracle_obs: dict, label: str,
                report: ParityReport) -> None:
    for k in OBS_LEAVES:
        if k not in env_obs:
            continue
        msg = _cmp_tensor(env_obs[k], oracle_obs[k])
        if msg:
            report.mismatches.append(Mismatch(f"{label}.{k}", step, f"step {step} {label}.{k}: {msg}"))


def compare_boundary(step: int, env, oracle: NativeOracle, row: int, slot: int, gen: int,
                     report: ParityReport) -> None:
    """Compare request kinds, branch counts, candidates and both seat views."""
    st = env._slots[row]
    ek = env._kinds(st)
    ok_ = oracle.kinds(slot, gen)
    if ek != ok_:
        report.mismatches.append(
            Mismatch("request_kind", step, f"step {step} request_kind env={ek} oracle={ok_}"))

    ec = env._branch_counts(st)
    oc = oracle.branch_counts(slot, gen)
    if ec != oc:
        report.mismatches.append(
            Mismatch("branch_count", step, f"step {step} branch_count env={ec} oracle={oc}"))

    for side in (0, 1):
        if ek[side] not in (0, 1, 2):
            continue
        env_obs = env._observe_one(st, side)
        ora_obs = oracle.observe(slot, gen, side)
        compare_obs(step, env_obs, ora_obs, f"seat{side}", report)


def compare_candidates(step: int, env, oracle: NativeOracle, row: int, slot: int, gen: int,
                       choices, report: ParityReport) -> None:
    """Ordered candidate lists must match for each selected prefix."""
    for side, actions in choices:
        prefix = []
        for action in actions:
            ec = env._candidates(env._slots[row], side, prefix)
            oc = oracle.candidates(slot, gen, side, prefix)
            if ec != oc:
                report.mismatches.append(Mismatch(
                    f"candidates.side{side}.lvl{len(prefix)}", step,
                    f"step {step} candidates differ seat{side} level {len(prefix)}: "
                    f"env={len(ec)} oracle={len(oc)}"))
            prefix.append(tuple(action))


def compare_step_result(step: int, env_res, oracle_res, report: ParityReport) -> None:
    if env_res != oracle_res:
        report.mismatches.append(Mismatch(
            "step_result", step, f"step {step} step_result env={env_res} oracle={oracle_res}"))
