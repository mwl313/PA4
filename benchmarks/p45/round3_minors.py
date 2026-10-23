"""P4.5 round-3 cleanups: stale echoes in the evidence and decision docs.

MINOR 1 d5_hypotheses.json said "10 negative-control tests" (actual 14)
MINOR 2 d5_hypotheses.json said step_levels+encode = 94.3% (actual 94.4)
MINOR 3 ARCHITECTURE_DECISION still said "67x prep" in two places (actual 65.99)
MINOR 4 RECEIPT encode 32.2 / env_step 2.0 vs artifact 32.3 / 1.9
NIT      d6 batch_per_row_vold used a rounded learner share; FINAL_REPORT says
         the displayed split sums to 100.0% when the rounded values sum to 100.1
"""
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
    coll = load(E / "d2b_collect.json", ("stage_share_pct",))
    split = load(E / "d2c_split.json", ("per_request_ms",))
    lsplit = load(E / "d2f_learner_split.json", ("share_pct",))
    sp = coll["stage_share_pct"]
    round_trip = round(sp["step_levels"] + sp["encode"], 1)

    # ---- 1 + 2: d5_hypotheses stale echoes ----------------------------
    hp = E / "d5_hypotheses.json"
    txt = hp.read_text(encoding="utf-8")
    txt = txt.replace("10 negative-control tests", "14 negative-control tests")
    txt = txt.replace("94.3%", f"{round_trip}%")
    txt = txt.replace("step_levels + encode are 94.3", f"step_levels + encode are {round_trip}")
    hp.write_text(txt, encoding="utf-8")
    print(f"MINOR 1/2: d5 -> 14 tests, {round_trip}% round trips")

    # ---- 3: ARCHITECTURE_DECISION 67x -> 66x -------------------------
    a = D / "ARCHITECTURE_DECISION.md"
    s = a.read_text(encoding="utf-8")
    s = s.replace("**67× prep**", "**65.99× prep**")
    s = s.replace("67× prep", "65.99× prep")
    s = s.replace("Wait is **65.99× prep**.", "Wait is **65.99× prep**.")
    a.write_text(s, encoding="utf-8")
    print("MINOR 3: ARCHITECTURE_DECISION 67x -> 65.99x")

    # ---- 4: RECEIPT shares from the artifact -------------------------
    rp = D / "RECEIPT.json"
    d = json.loads(rp.read_text(encoding="utf-8"))
    d["key_measurements"]["collection_stage_share_pct"] = {
        "step_levels": sp["step_levels"],
        "encode": sp["encode"],
        "env_step": sp["env_step"],
        "accounting": sp["accounting"],
    }
    rp.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"MINOR 4: RECEIPT shares -> encode {sp['encode']}, env_step {sp['env_step']}")

    # ---- NIT: d6 uses the exact learner share, not 0.232 -------------
    attrib = {}
    for line in (E / "d2_learner_attribution.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5 and p[0].startswith("update."):
            try:
                attrib[p[0]] = float(p[2])
            except ValueError:
                pass
    upd_total = sum(attrib.values())
    vold_share = attrib.get("update.vold_encode", 0.0) / upd_total if upd_total else 0

    dd = load(E / "d6_amdahl.json", ("candidates",))
    hist = load(E / "historical_reconstruction.json", ("stage_sums_s",))
    wall = hist["stage_sums_s"]["wall_seconds"]
    ufrac = hist["stage_sums_s"]["update_seconds"] / wall
    for c in dd["candidates"]:
        if c["candidate"] == "batch_per_row_vold":
            f = ufrac * vold_share
            c["fraction_of_wall_removed"] = round(f, 4)
            c["amdahl_upper_bound_x"] = round(1 / (1 - f), 3)
    (E / "d6_amdahl.json").write_text(json.dumps(dd, indent=1), encoding="utf-8")
    print(f"NIT: d6 vold fraction uses exact share {vold_share:.4f}")

    # ---- NIT: FINAL_REPORT rounding sentence -------------------------
    f = D / "FINAL_REPORT.md"
    t = f.read_text(encoding="utf-8")
    t = t.replace(
        f"— summing to {lsplit['share_sum_pct']}%.",
        f"— which sum to {lsplit['share_sum_pct']}% "
        f"(the printed one-decimal values differ by 0.1 from rounding).")
    f.write_text(t, encoding="utf-8")
    print("NIT: rounding sentence clarified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
