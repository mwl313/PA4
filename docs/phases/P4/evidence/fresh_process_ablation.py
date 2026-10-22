"""P4-D: prove the fresh-process recovery test is FALSIFIABLE.

Requirement: the recovery test must fail if Adam or RNG restoration is removed.
A test that passes regardless proves nothing.

Runs the ablation child four ways against the SAME committed checkpoint and
reports whether each deliberate gap is DETECTED.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch


def main():
    work, ckpt_dir, cohort_dir = sys.argv[1:4]
    child = os.path.join(HERE, "ablation_child.py")

    results = {}
    for mode in ("full", "no_adam", "no_rng", "no_steps"):
        out = os.path.join(work, f"abl_{mode}.json")
        proc = subprocess.run(
            [sys.executable, child, work, ckpt_dir, cohort_dir, out, mode],
            capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            results[mode] = {"error": proc.stderr[-500:]}
            print(f"[{mode}] FAILED: {proc.stderr[-300:]}")
            continue
        with open(out, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        res["_model"] = torch.load(out + ".model.pt", map_location="cpu",
                                   weights_only=True)
        results[mode] = res

    if "full" not in results or "_model" not in results["full"]:
        print("RESULT: FAIL (baseline ablation run did not complete)")
        return 1

    ref = results["full"]["_model"]
    # The "full" row is the reference. Its delta vs itself is trivially 0 and is
    # NOT a determinism control; the control is the separate nondeterminism
    # script (control_nondeterminism.py). Label it accordingly in the output.

    def delta_vs_baseline(res):
        if "_model" not in res:
            return None
        d = 0.0
        for k in ref:
            d = max(d, float((ref[k].float() - res["_model"][k].float()).abs().max()))
        return d

    summary = {}
    for mode, res in results.items():
        if "_model" not in res:
            summary[mode] = {"error": res.get("error", "unknown")[:200]}
            continue
        summary[mode] = {
            "model_max_abs_delta_vs_full": delta_vs_baseline(res),
            "steps": res["steps"],
            "rng_restored": res["rng_restored"],
            "committed": res["committed"],
            "ledger_len": len(res["ledger"]),
        }

    print(json.dumps(summary, indent=1))

    base = summary["full"]
    d_adam = summary["no_adam"]["model_max_abs_delta_vs_full"]
    d_rng = summary["no_rng"]["model_max_abs_delta_vs_full"]
    steps_gap = summary["no_steps"]["steps"] != base["steps"]
    rng_gap = summary["no_rng"]["rng_restored"] is False

    print()
    print(f"baseline vs itself      : {base['model_max_abs_delta_vs_full']:.3e}")
    print(f"no_adam  model delta    : {d_adam:.3e}   <- must exceed tolerance")
    print(f"no_rng   model delta    : {d_rng:.3e}   (weights alone cannot see RNG)")
    print(f"no_rng   flag detected  : {rng_gap}")
    print(f"no_steps counter gap    : {steps_gap} "
          f"({summary['no_steps']['steps']} vs {base['steps']})")

    # RNG loss is detected by the MEASURED rng_restored flag, not by weights
    # (the weight delta of an RNG ablation is within tolerance).
    ok = (base["model_max_abs_delta_vs_full"] < 1e-9
          and d_adam > 1e-5
          and rng_gap
          and steps_gap)
    print()
    print("RESULT:", "PASS (test is falsifiable)" if ok
          else "FAIL (test cannot detect an ablation)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
