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
    # NOTE: the bound excess must come from the FRESH-PROCESS artifact. Taking
    # it from resume_parity.txt (the superseded in-process run) is exactly the
    # misattribution audit round 5 caught.

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
        "step_ablation": f"{ab['no_steps']['steps']} vs {ab['full']['steps']}",
        "pids": f"{fr['parent_pid']}` → `{fr['child_pid']}",
        "bound_excess": f"{fr['worst_excess_over_bound']:.3e}",
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

    stale_markers = [
        "before P35_I can be set PASS",
        "Do not start P4",
        "parity) is not started",
        "re-audit pending",
        "P35_I remains BLOCKED",
        "P35_I BLOCKED",
    ]
    for marker in stale_markers:
        if marker in status:
            failures.append(f"PROJECT_STATUS still contains stale marker: {marker!r}")
    if "closed, fresh-process recovery verified" not in status:
        failures.append("PROJECT_STATUS does not mark P4 closed")

    # Absence rule: numbers belonging to a SUPERSEDED run must not appear
    # in current documents. Presence-only checks let round 6 certify PASS
    # over two tables that still cited the previous run.
    superseded_tokens = {
        '27` → `186': 'old pid pair',
        '5.355e-09': 'old adam delta',
        '9.384e-05': 'old adam ablation',
        '10 vs 19': 'old step ablation pair',
        '19 = 19': 'old step count',
        # Distinctive values of the SUPERSEDED in-process run (resume_parity.txt).
        # Section 5 of CHECKPOINT_RECOVERY is carved out to document that run, so
        # these must not appear anywhere else.
        '4.307e-09': 'in-process adam delta',
        '-9.993e-06': 'in-process bound excess',
    }
    # Strip the deliberately-attributed superseded section before scanning.
    # NOTE: the pattern needs ONE backslash before each dot. An over-escaped
    # raw string (two backslashes) demands a literal backslash, matches
    # nothing, and silently leaves the carve-out dead.
    import re as _re

    _s5 = _re.search(r"## 5\..*?(?=## 6\.)", doc, _re.S)
    if _s5 is None:
        failures.append("crosscheck: could not locate section 5 to exclude")
    else:
        section5 = _s5.group(0)
        doc = doc.replace(section5, "")
        # Self-test: section 5 holds the superseded run's numbers, so the
        # stripped text must have contained at least one of its markers.
        if "4.307e-09" not in section5 and "1.192e-07" not in section5:
            failures.append(
                "crosscheck self-test: section 5 does not look like the "
                "superseded run, so the exclusion may be mis-scoped")

    def _norm(text):
        """Collapse whitespace/backticks so 27 -> 186 and 27->186 both match."""
        text = text.replace("\u2192", "->").replace("\u2014", "-")
        return _re.sub(r"[\s`]+", "", text)

    receipt_text = (D / "RECEIPT.json").read_text(encoding="utf-8")
    for doc_name, doc_text in (
            ("CHECKPOINT_RECOVERY.md", doc),
            ("FINAL_REPORT.md", fin),
            ("PROJECT_STATUS.md", status),
            ("RECEIPT.json", receipt_text)):
        norm = _norm(doc_text)
        for token, label in superseded_tokens.items():
            if _norm(token) in norm:
                failures.append(
                    f"{doc_name} still cites {label}: {token!r}")
    # --- structural RECEIPT check -------------------------------------
    # String scanning cannot see "parent_pid": 27 vs `27` -> `186`, nor a
    # full-precision float vs its 3-digit spelling, so a revert of G4_D would
    # slip through. Compare the parsed fields against the live artifact instead.
    g4d = receipt["gate_evidence"]["G4_D"]
    expect = {
        "parent_pid": fr["parent_pid"],
        "child_pid": fr["child_pid"],
        "model_tensors_passing": fr["model_tensors_passing"],
        "model_max_abs_delta": fr["model_max_abs_delta"],
        "adam_max_abs_delta": fr["adam_max_abs_delta"],
        "optimizer_steps": fr["steps"],
        "committed_matches": fr["committed_matches"],
        "worst_excess_over_bound": fr["worst_excess_over_bound"],
        "adam_params_nonzero": fr["child_adam_nonzero"],
        "ledger_match": fr["ledger_match"],
        "rng_restored": fr["child_rng_restored"],
    }
    for key, want in expect.items():
        got = g4d.get(key)
        same = (got == want) or (
            isinstance(got, float) and isinstance(want, float)
            and abs(got - want) <= 1e-15)
        if not same:
            failures.append(
                f"RECEIPT.json gate_evidence.G4_D.{key} = {got!r} but the "
                f"artifact says {want!r} (stale receipt)")

    # Remaining scalar fields the round-8 enumeration skipped.
    for key, want in {
        "distinct_processes": fr["distinct_processes"],
        "adam_state_ids_match": fr["adam_state_ids_match"],
        "child_device": fr["child_device"],
    }.items():
        got = g4d.get(key)
        if got != want:
            failures.append(
                f"RECEIPT.json gate_evidence.G4_D.{key} = {got!r} but the "
                f"artifact says {want!r} (stale receipt)")

    fals = g4d.get("falsifiability") or {}
    fals_expect = {
        "adam_ablation_model_delta": ab["no_adam"]["model_max_abs_delta_vs_full"],
        "rng_ablation_model_delta": ab["no_rng"]["model_max_abs_delta_vs_full"],
        "rng_ablation_detected_by_stream": ab["no_rng"]["rng_restored"] is False,
        "step_ablation_counter": [ab["no_steps"]["steps"], ab["full"]["steps"]],
        "all_detected": True,
    }
    for key, want in fals_expect.items():
        got = fals.get(key)
        same = (got == want) or (
            isinstance(got, float) and isinstance(want, float)
            and abs(got - want) <= 1e-15)
        if not same:
            failures.append(
                f"RECEIPT.json falsifiability.{key} = {got!r} but the artifact "
                f"says {want!r} (stale receipt)")

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
