"""Cross-check every P4 doc number against the preserved artifacts."""
import json
import pathlib

E = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4/evidence")
D = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4")


def block(p):
    t = p.read_text(encoding="utf-8")
    s = t.index("{")
    d = 0
    for i in range(s, len(t)):
        if t[i] == "{":
            d += 1
        elif t[i] == "}":
            d -= 1
            if d == 0:
                return json.loads(t[s:i + 1])
    raise ValueError(p)


def main():
    fr = block(E / "fresh_process_recovery.txt")
    ab = block(E / "fresh_process_ablation.txt")

    doc = (D / "CHECKPOINT_RECOVERY.md").read_text(encoding="utf-8")
    fin = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")
    receipt = json.loads((D / "RECEIPT.json").read_text(encoding="utf-8"))
    status = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/PROJECT_STATUS.md").read_text(
        encoding="utf-8")

    checks = {
        "model_delta": f"{fr['model_max_abs_delta']:.3e}",
        "adam_delta": f"{fr['adam_max_abs_delta']:.3e}",
        "steps": f"{fr['steps'][0]} = {fr['steps'][1]}",
        "matches": f"{fr['committed_matches'][0]} = {fr['committed_matches'][1]}",
        "no_adam": f"{ab['no_adam']['model_max_abs_delta_vs_full']:.3e}",
        "no_steps": str(ab["no_steps"]["steps"]),
        "pids": f"{fr['parent_pid']}` → `{fr['child_pid']}",
    }

    print("value           in_doc  in_report")
    ok = True
    for name, val in checks.items():
        a = val in doc
        b = val in fin
        print(f"  {name:12s} {str(a):7s} {str(b)}   ({val})")
        if name not in ("pids",):
            ok = ok and a

    print()
    print("tests block total :", receipt["tests"]["total"])
    print("G4_E total_tests  :", receipt["gate_evidence"]["G4_E"]["total_tests"])
    print("G4_E p4_tests     :", receipt["gate_evidence"]["G4_E"]["p4_tests"])
    print("status says closed:", "closed, fresh-process recovery verified" in status)
    print("status says do-not-start-P4:", "Do not start P4" in status)
    print()
    print("RESULT:", "PASS" if ok and receipt["gate_evidence"]["G4_E"]["total_tests"] == 272
          and "Do not start P4" not in status else "FAIL")


if __name__ == "__main__":
    main()
