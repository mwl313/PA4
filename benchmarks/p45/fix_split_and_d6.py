"""Restore the disjoint split line and regenerate d6_amdahl from the new CSV."""
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
    hist = load(E / "historical_reconstruction.json", ("stage_sums_s",))
    s = hist["stage_sums_s"]

    # ---- 1. FINAL_REPORT: insert the disjoint split paragraph once -----
    f = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")
    marker = "Within a single learner evaluation the components split as"
    para = (
        "Within one learner evaluation the four components are measured "
        f"**disjointly** by hoisting branch-table construction out of the "
        f"evaluate loop (`evidence/d2f_learner_split.json`, {ls['rows']} rows, "
        f"{ls['total_ms']} ms): `evaluate_encoded` {sp['evaluate_encoded_only']}%, "
        f"branch-table construction {sp['branch_tables']}%, observation rebuild "
        f"{sp['obs_rebuild']}%, batched encoder {sp['batched_encoder']}% — "
        f"summing to {ls['share_sum_pct']}%. The encoder is the one part already "
        "batched, and it is the cheapest.")

    # Drop any old version of the paragraph, then insert the corrected one.
    import re
    f = re.sub(r"Within (?:a single|one) learner evaluation the components? .*?"
               r"cheapest\.\n\n", "", f, flags=re.S)
    anchor = "**The hypothesised `.item()` synchronisation bottleneck is 0.3%**"
    assert anchor in f, "anchor missing"
    f = f.replace(anchor, para + "\n\n" + anchor, 1)

    # H2 verdict evidence should cite the disjoint number
    f = re.sub(r"\| H2 \| microbatch 8 / per-row scorer / ragged CPU \| \*\*SUPPORTED\*\* \|.*\n",
               "| H2 | microbatch 8 / per-row scorer / ragged CPU | **SUPPORTED** | "
               f"`ppo_loop` is 66.2% of learner time; within one evaluation "
               f"`evaluate_encoded` alone is {sp['evaluate_encoded_only']}% and "
               f"branch-table construction {sp['branch_tables']}%, both per row |\n",
               f)
    (D / "FINAL_REPORT.md").write_text(f, encoding="utf-8")
    print("FINAL_REPORT: disjoint split restored, H2 cites it")

    # ---- 2. d6_amdahl.json: regenerate from measured shares -----------
    sp_coll = load(E / "d2b_collect.json", ("stage_share_pct",))["stage_share_pct"]
    wall = s["wall_seconds"]
    cfrac = s["collection_seconds"] / wall
    ufrac = s["update_seconds"] / wall

    def bound(f):
        return round(1 / (1 - f), 3)

    out = {
        "denominator": "P4 20-iteration row-sum wall (720.541 s)",
        "measured_shares": {
            "collection": round(cfrac, 4),
            "update": round(ufrac, 4),
            "checkpoint": round(s["checkpoint_seconds"] / wall, 4),
        },
        "learner_split_disjoint_pct": sp,
        "candidates": [
            {"candidate": "eliminate_collection_entirely",
             "fraction_of_wall_removed": round(cfrac, 4),
             "amdahl_upper_bound_x": bound(cfrac),
             "notes": "physically impossible; bounds the actor side"},
            {"candidate": "eliminate_update_entirely",
             "fraction_of_wall_removed": round(ufrac, 4),
             "amdahl_upper_bound_x": bound(ufrac),
             "notes": "physically impossible; bounds the learner side"},
            {"candidate": "eliminate_step_levels_only",
             "fraction_of_wall_removed": round(cfrac * sp_coll["step_levels"] / 100, 4),
             "amdahl_upper_bound_x": bound(cfrac * sp_coll["step_levels"] / 100),
             "notes": "prefix-STEP transport barrier"},
            {"candidate": "eliminate_encode_only",
             "fraction_of_wall_removed": round(cfrac * sp_coll["encode"] / 100, 4),
             "amdahl_upper_bound_x": bound(cfrac * sp_coll["encode"] / 100),
             "notes": "ENCODE transport barrier"},
            {"candidate": "batch_per_row_evaluate",
             "fraction_of_wall_removed": round(
                 ufrac * 0.662 * sp["evaluate_encoded_only"] / 100, 4),
             "amdahl_upper_bound_x": bound(
                 ufrac * 0.662 * sp["evaluate_encoded_only"] / 100),
             "notes": "batch the conditional scorer across rows"},
            {"candidate": "batch_per_row_vold",
             "fraction_of_wall_removed": round(ufrac * 0.232, 4),
             "amdahl_upper_bound_x": bound(ufrac * 0.232),
             "notes": "batch the frozen critic across rows"},
        ],
        "warning": ("Sequential Amdahl bounds are UPPER LIMITS with every other "
                    "stage unchanged. They are not forecasts and cannot be added."),
    }
    (E / "d6_amdahl.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("d6_amdahl.json regenerated from measured shares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
