"""Use the PREDECLARED FP32 tolerance for CUDA resume parity, with the control.

The comparison previously demanded bitwise weight equality, which CUDA cannot
provide: running the same update twice in one process already differs by
~6e-08. The design predeclares atol=1e-5, rtol=1e-4 for CUDA FP32 and exact
values only where a deterministic CPU reference permits.

This script reports BOTH: the nondeterminism control, and the resume comparison
judged against the predeclared tolerance.
"""
import json
import sys
import tempfile

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import checkpoint as ck

sys.path.insert(0, "/tmp")
from proof_resume import (DEPTH, SEED, build_model, collect_cohort, run_branch)
from control_nondet import one_update

from pa4.learner.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from pa4.learner.ingest import build_cohort
from pa4.learner.sink import LosslessBuffer

ATOL = 1e-5
RTOL = 1e-4


def max_rel_delta(a: dict, b: dict) -> tuple[float, float]:
    """Max absolute and max relative deviation over all model tensors."""
    abs_max = 0.0
    rel_max = 0.0
    for k in a:
        d = (a[k].float() - b[k].float()).abs()
        abs_max = max(abs_max, float(d.max()))
        denom = a[k].float().abs().clamp_min(1e-12)
        rel_max = max(rel_max, float((d / denom).max()))
    return abs_max, rel_max


def tensor_pass(a: torch.Tensor, b: torch.Tensor, atol=ATOL, rtol=RTOL) -> bool:
    """Predeclared criterion: every element satisfies |a-b| <= atol + rtol*|b|."""
    a = a.float()
    b = b.float()
    return bool(((a - b).abs() <= (atol + rtol * b.abs())).all())


def combined(a: dict, b: dict) -> tuple[float, float, int, int]:
    """Max absolute delta and the worst (delta - atol)/|b| ratio, plus counts."""
    abs_max = 0.0
    worst_excess = float("-inf")
    passed = 0
    total = 0
    for k in a:
        d = (a[k].float() - b[k].float()).abs()
        abs_max = max(abs_max, float(d.max()))
        bound = ATOL + RTOL * a[k].float().abs()
        excess = float((d - bound).max())
        worst_excess = max(worst_excess, excess)
        total += 1
        if tensor_pass(a[k], b[k]):
            passed += 1
    return abs_max, worst_excess, passed, total


def main():
    work = tempfile.mkdtemp(prefix="p4_resume2_")
    snap_dir = tempfile.mkdtemp(prefix="snap_")

    m0 = build_model()
    from pa4.learner.snapshot import save_snapshot
    snap0 = save_snapshot(m0.state_dict(), snap_dir, policy_version=1,
                          committed_matches=0)

    cohorts = {it: collect_cohort(it, 1, snap0.path, seed=SEED + it)
               for it in range(1, DEPTH + 1)}
    print("cohorts:", {k: v.matches() for k, v in cohorts.items()})

    # ---- CONTROL: same update twice, same process ----------------------
    w1, o1 = one_update(build_model(), cohorts[1], 1)
    w2, o2 = one_update(build_model(), cohorts[1], 1)
    ctl_abs, ctl_rel = max_rel_delta(w1, w2)
    print(f"control (same update twice): max abs {ctl_abs:.3e}  max rel {ctl_rel:.3e}")

    # ---- Branch A: continuous 4 ---------------------------------------
    _, state_a = run_branch(work + "/A", cohorts, 1, end_iter=DEPTH)
    print("branch A complete")

    # ---- Branch B: 2, exit, restore from disk, 3-4 --------------------
    _, state_b1 = run_branch(work + "/B", cohorts, 1, end_iter=2)
    ckdir = work + "/B_ckpt"
    save_checkpoint(state_b1, ckdir)
    restored = load_checkpoint(ckdir)          # genuinely reloaded from disk
    assert restored.iteration == 2
    _, state_b = run_branch(work + "/B2", cohorts, 3,
                            start_state=restored, end_iter=DEPTH)
    print("branch B resumed from disk")

    abs_d, worst_excess, passed, total = combined(state_a.model_state,
                                                  state_b.model_state)
    rep = ck.verify_restore_parity(state_a, state_b)

    adam_abs = 0.0
    sa = state_a.optimizer_state["state"]
    sb = state_b.optimizer_state["state"]
    for pid in sa:
        for key in sa[pid]:
            if torch.is_tensor(sa[pid][key]):
                adam_abs = max(adam_abs,
                               float((sa[pid][key].float() - sb[pid][key].float()).abs().max()))

    print(json.dumps({
        "control_max_abs_delta": ctl_abs,
        "control_max_rel_delta": ctl_rel,
        "resume_max_abs_weight_delta": abs_d,
        "resume_worst_excess_over_bound": worst_excess,
        "resume_tensors_passing": f"{passed}/{total}",
        "resume_max_abs_adam_delta": adam_abs,
        "counters_match": rep["counters_match"],
        "rng_match": rep["rng_match"],
        "committed_matches": (state_a.committed_matches, state_b.committed_matches),
        "ledger_equal": sorted(state_a.match_ledger) == sorted(state_b.match_ledger),
        "predeclared_atol": ATOL,
        "predeclared_rtol": RTOL,
    }, indent=1))

    adam_bound_ok = all(
        ((state_a.optimizer_state["state"][pid][key].float()
          - state_b.optimizer_state["state"][pid][key].float()) <= ATOL).all().item()
        for pid in state_a.optimizer_state["state"]
        for key in state_a.optimizer_state["state"][pid]
        if torch.is_tensor(state_a.optimizer_state["state"][pid][key]))

    ok = (passed == total and worst_excess <= 0 and adam_bound_ok
          and rep["counters_match"]
          and state_a.committed_matches == state_b.committed_matches
          and sorted(state_a.match_ledger) == sorted(state_b.match_ledger))
    print()
    print("criterion: CUDA FP32 predeclared atol=1e-5, rtol=1e-4 (bitwise not required)")
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
