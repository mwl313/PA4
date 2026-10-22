"""Cross-check every P4 doc number against the preserved artifacts.

Asserts artifact-derived values appear in BOTH narrative documents, that the
receipt's counts agree everywhere, that PROJECT_STATUS carries no stale
guidance, and that the superseded in-process artifacts are retained as history.

Written to fail loudly on the regressions that earlier audit rounds found:
  - docs quoting values present in no artifact
  - receipt counts disagreeing with the actual suite
  - "P4 is not started" surviving after closure
"""
import json
import pathlib

ROOT = pathlib.Path("/home/pa4/projects/PokeAgent4.0")
E = ROOT / "docs/phases/P4/evidence"
D = ROOT / "docs/phases/P4"


def block_with(path, required):
    """Find the JSON object in `path` containing every key in `required`."""
    text = path.read_text(encoding="utf-8")
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except Exception:
                        break
                    if isinstance(obj, dict) and all(k in obj for k in required):
                        return obj
                    break
    raise ValueError(f"{path}: no JSON object containing {required}")


def main():
    fr = block_with(E / "fresh_process_recovery.txt",
                    ("model_max_abs_delta", "adam_max_abs_delta", "steps"))
    ab = block_with(E / "fresh_process_ablation.txt",
                    ("no_adam", "no_rng", "no_steps", "full"))
    rp = block_with(E / "resume_parity.txt",
                    ("resume_worst_excess_over_bound",))

    doc = (D / "CHECKPOINT_RECOVERY.md").read_text(encoding="utf-8")
    fin = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")
    receipt = json.loads((D / "RECEIPT.json").read_text(encoding="utf-8"))
    status = (ROOT / "docs/PROJECT_STATUS.md").read_text(encoding="utf-8")

    failures = []

    documented = {
        "model_max_abs_delta": f"{fr['model_max_abs_delta']:.3e}",
        "adam_max_abs_delta": f"{fr['adam_max_abs_delta']:.3e}",
        "steps": f"{fr['steps'][0]} = {fr['steps'][1]}",
        "committed_matches": f"{fr['committed_matches'][0]} = {fr['committed_matches'][1]}",
        "adam_ablation": f"{ab['no_adam']['model_max_abs_delta_vs_full']:.3e}",
        "step_ablation": str(ab["no_steps"]["steps"]),
        "pids": f"{fr['parent_pid']}` → `{fr['child_pid']}",
        "bound_excess": f"{abs(rp['resume_worst_excess_over_bound']):.3e}",
    }

    print("value                  in_recovery  in_report")
    for name, val in documented.items():
        a = val in doc
        b = val in fin
        print(f"  {name:20s} {str(a):11s} {str(b)}   ({val})")
        if not a:
            failures.append(f"{name} ({val}) missing from CHECKPOINT_RECOVERY.md")
        if not b:
            failures.append(f"{name} ({val}) missing from FINAL_REPORT.md")

    if receipt["tests"]["total"] != 272:
        failures.append(f"receipt tests.total={receipt['tests']['total']} != 272")
    if receipt["gate_evidence"]["G4_E"]["total_tests"] != 272:
        failures.append("receipt G4_E.total_tests != 272")
    if receipt["gate_evidence"]["G4_E"]["p4_tests"] != 79:
        failures.append("receipt G4_E.p4_tests != 79")

    if "before P35_I can be set PASS" in status:
        failures.append("PROJECT_STATUS still implies P35_I is unset")
    if "Do not start P4" in status:
        failures.append("PROJECT_STATUS still says 'Do not start P4'")
    if "parity) is not started" in status:
        failures.append("PROJECT_STATUS still says P4 is not started")
    if "closed, fresh-process recovery verified" not in status:
        failures.append("PROJECT_STATUS does not mark P4 closed")

    for name in ("resume_parity.txt", "_resume_helpers.py", "proof_resume_core.py"):
        if not (E / name).exists():
            failures.append(f"superseded artifact deleted: {name}")

    print()
    print(f"receipt tests.total : {receipt['tests']['total']}")
    print(f"receipt G4_E        : {receipt['gate_evidence']['G4_E']['total_tests']} "
          f"/ p4 {receipt['gate_evidence']['G4_E']['p4_tests']}")
    print("status marked closed:", "closed, fresh-process recovery verified" in status)
    print("superseded retained :",
          all((E / n).exists() for n in
              ("resume_parity.txt", "_resume_helpers.py", "proof_resume_core.py")))
    print()
    for f in failures:
        print("  FAIL:", f)
    print("RESULT:", "PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
