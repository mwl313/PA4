"""Restore the learner-split evidence line into FINAL_REPORT section 4."""
import json
import pathlib

D = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4.5")
E = D / "evidence"


def load(p, req):
    text = p.read_text(encoding="utf-8")
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
                    if isinstance(obj, dict) and all(k in obj for k in req):
                        return obj
                    break
    raise ValueError(p)


def main():
    ls = load(E / "d2f_learner_split.json", ("share_pct",))
    sp = ls["share_pct"]

    f = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")
    anchor = "**The hypothesised `.item()` synchronisation bottleneck is 0.3%**"
    assert anchor in f, "anchor missing"

    insert = (
        "Within a single learner evaluation the components split as "
        f"**per-row `evaluate_encoded` {sp['per_row_evaluate']}%**, "
        f"branch-table construction {sp['branch_tables']}%, "
        f"observation rebuild {sp['obs_rebuild']}%, and the batched encoder "
        f"only {sp['batched_encoder']}% "
        f"(`evidence/d2f_learner_split.json`, {ls['rows']} rows, "
        f"{ls['total_ms']} ms total). The encoder is the one part already "
        "batched, and it is the cheapest.\n\n")
    f = f.replace(anchor, insert + anchor, 1)
    (D / "FINAL_REPORT.md").write_text(f, encoding="utf-8")
    print(f"restored learner split ({sp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
