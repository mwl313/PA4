"""P4.5 self-consistency check: every reported number must trace to an artifact.

Mirrors the P4 lesson: if a document quotes a measurement, a script should be
able to prove it came from the evidence files.
"""
import csv
import json
import pathlib
import re

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
    fails = []

    hist = load(E / "historical_reconstruction.json", ("stage_sums_s",))
    split = load(E / "d2c_split.json", ("per_request_ms",))
    coll = load(E / "d2b_collect.json", ("stage_share_pct",))
    scale = load(E / "d2e_scale.json", ("measurements",))
    base = load(E / "d3_baseline.json", ("median_matches_per_sec",))
    lsplit = load(E / "d2f_learner_split.json", ("share_pct",))
    receipt = json.loads((D / "RECEIPT.json").read_text(encoding="utf-8"))
    report = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")
    hist_md = (D / "HISTORICAL_RECONCILIATION.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader(open(D / "BOTTLENECK_RANKING.csv", encoding="utf-8")))
    by_n = {m["batch"]: m for m in scale["measurements"]}

    checks = {
        "batch1_kernel": (f"{by_n[1]['kernel_ms']}", report),
        "batch64_kernel": (f"{by_n[64]['kernel_ms']}", report),
        "batch8_per_row": (f"{by_n[8]['per_row_ms']}", report),
        "transport_prep": (f"{split['per_request_ms']['prep']}", report),
        "transport_wait": (f"{split['per_request_ms']['wait']}", report),
        "step_levels_share": (f"{coll['stage_share_pct']['step_levels']}", report),
        "encode_share": (f"{coll['stage_share_pct']['encode']}", report),
        "baseline_median": (f"{base['median_matches_per_sec']}", report),
        "learner_split_row": (f"{lsplit['share_pct']['evaluate_encoded_only']}", report),
        "hist_collection": (f"{hist['stage_sums_s']['collection_seconds']}", hist_md),
        "hist_update": (f"{hist['stage_sums_s']['update_seconds']}", hist_md),
        "hist_outer": (f"{hist['stage_sums_s']['wall_seconds']}", hist_md),
    }
    for name, (val, doc) in checks.items():
        if val not in doc:
            fails.append(f"{name}: {val!r} not found in its document")

    # Per-iteration medians in the markdown must equal the JSON.
    for key in ("collection_seconds", "update_seconds", "checkpoint_seconds",
                "wall_seconds"):
        med = hist["per_iteration"][key]["median"]
        if f"{med}" not in hist_md:
            fails.append(f"per-iteration median {key}={med} missing from historical md")

    # CSV ranks must be sorted by attributed wall seconds, descending.
    walls = [float(r["attributed_wall_s"]) for r in rows]
    if walls != sorted(walls, reverse=True):
        fails.append("BOTTLENECK_RANKING.csv is not sorted by attributed wall")

    # CSV amdahl must equal 1/(1-f) using its own columns.
    wall_total = hist["stage_sums_s"]["wall_seconds"]
    for r in rows:
        f = float(r["attributed_wall_s"]) / wall_total
        want = round(1 / (1 - f), 3)
        got = float(r["amdahl_upper_bound_x"])
        if abs(want - got) > 0.002:
            fails.append(f"amdahl mismatch rank {r['rank']}: {got} vs {want}")

    # Verdict must follow the gate rule.
    part = [k for k, v in receipt["gates"].items() if v != "PASS"]
    if part and receipt["verdict"] == "PASS":
        fails.append(f"verdict PASS while gates not PASS: {part}")
    if part and receipt["verdict"] != "PARTIAL":
        fails.append(f"gates {part} not PASS but verdict is {receipt['verdict']}")

    # Test counts
    if receipt["tests"]["grand_total"] != 286:
        fails.append(f"grand_total {receipt['tests']['grand_total']} != 286")

    # Attribution coverage >= 90% (D45-5)
    cov = sum(walls) / wall_total
    if cov < 0.90:
        fails.append(f"attribution coverage {cov:.3f} < 0.90")

    # ---- round-2 invariants -------------------------------------------
    lsplit = load(E / "d2f_learner_split.json", ("share_pct",))
    total_share = sum(lsplit["share_pct"].values())
    if abs(total_share - 100.0) > 0.6:
        fails.append(f"learner split sums to {total_share:.1f}%, not 100% "
                     "(components overlap)")

    csv_text = (D / "BOTTLENECK_RANKING.csv").read_text(encoding="utf-8")
    stale = {
        "73.6%": "old learner split",
        "94.6%": "inflated coverage",
        "11.5 ms": "old transport wait",
        "66.8×": "old transport ratio",
    }
    docs = {
        "FINAL_REPORT.md": report,
        "ARCHITECTURE_DECISION.md": (D / "ARCHITECTURE_DECISION.md").read_text(encoding="utf-8"),
        "HISTORICAL_RECONCILIATION.md": hist_md,
        "BOTTLENECK_RANKING.csv": csv_text,
        "RECEIPT.json": json.dumps(receipt),
    }
    for name, text in docs.items():
        for token, label in stale.items():
            if token in text:
                fails.append(f"{name} still contains {label}: {token!r}")

    print(f"attribution coverage : {cov*100:.1f}%")
    print(f"learner split sum    : {total_share:.1f}%")
    print(f"verdict              : {receipt['verdict']} (non-PASS gates: {part})")
    print(f"tests                : {receipt['tests']['grand_total']}")
    print(f"csv rows sorted      : {walls == sorted(walls, reverse=True)}")
    print()
    for f in fails:
        print("  FAIL:", f)
    print("RESULT:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
