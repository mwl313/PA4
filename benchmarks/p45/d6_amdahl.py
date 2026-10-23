"""D6: Amdahl upper bounds for each candidate, from measured shares.

These are UPPER BOUNDS for a strictly sequential topology. They are not
predictions and must never be presented as achievable speedups.
"""
import json
import pathlib

E = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4.5/evidence")
D = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4.5")


def load(name, required):
    text = (E / name).read_text(encoding="utf-8")
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
    raise ValueError(name)


def main():
    hist = load("historical_reconstruction.json", ("stage_sums_s",))
    coll = load("d2b_collect.json", ("stage_share_pct",))
    scale = load("d2e_scale.json", ("measurements",))
    split = load("d2c_split.json", ("per_request_ms",))

    s = hist["stage_sums_s"]
    w = s["wall_seconds"]
    collect_frac = s["collection_seconds"] / w
    update_frac = s["update_seconds"] / w

    by_n = {m["batch"]: m for m in scale["measurements"]}
    per_row_1 = by_n[1]["per_row_ms"]
    per_row_64 = by_n[64]["per_row_ms"]

    # Measured sub-shares inside collection.
    sp = coll["stage_share_pct"]
    step_levels_share = sp["step_levels"] / 100.0
    encode_share = sp["encode"] / 100.0

    rows = []

    def add(name, removes_frac, basis, notes):
        """Amdahl: S = 1 / (1 - f) for a sequential topology."""
        rows.append({
            "candidate": name,
            "fraction_of_wall_removed": round(removes_frac, 4),
            "amdahl_upper_bound_x": round(1.0 / (1.0 - removes_frac), 3) if removes_frac < 1 else None,
            "basis": basis,
            "notes": notes,
        })

    add("eliminate_collection_entirely", collect_frac,
        f"collection is {collect_frac*100:.1f}% of wall",
        "physically impossible; bounds the whole actor side")
    add("eliminate_update_entirely", update_frac,
        f"update is {update_frac*100:.1f}% of wall",
        "physically impossible; bounds the whole learner side")
    add("eliminate_step_levels_only", collect_frac * step_levels_share,
        f"step_levels is {sp['step_levels']}% of accounted collection",
        "targets the prefix-STEP transport barrier")
    add("eliminate_encode_only", collect_frac * encode_share,
        f"encode is {sp['encode']}% of accounted collection",
        "targets the ENCODE barrier")
    add("eliminate_vold_per_row", update_frac * 0.232,
        "vold_encode measured at 23.2% of learner (profile dumps)",
        "batch the frozen critic instead of one row per forward")
    add("eliminate_ppo_loop_per_row", update_frac * 0.662,
        "ppo_loop measured at 66.2% of learner, dominated by per-row "
        "evaluate_encoded (isolated split: 73.6% of one learner evaluation)",
        "batch conditional scorer evaluation")
    add("batch_both_learner_paths", update_frac * (0.232 + 0.662),
        "vold_encode + ppo_loop measured at 89.4% of learner",
        "the learner-side batching target")

    out = {
        "denominator": "P4 20-iteration row-sum wall (720.541 s)",
        "measured_shares": {
            "collection": round(collect_frac, 4),
            "update": round(update_frac, 4),
            "checkpoint": round(s["checkpoint_seconds"] / w, 4),
        },
        "kernel_scaling": {
            "batch_1_per_row_ms": per_row_1,
            "batch_64_per_row_ms": per_row_64,
            "per_row_reduction_x": round(per_row_1 / per_row_64, 1),
            "note": "the batching headroom that makes the learner candidates worthwhile",
        },
        "transport": split["per_request_ms"],
        "candidates": rows,
        "warning": (
            "Sequential Amdahl bounds are UPPER LIMITS with every other stage "
            "unchanged. They are not forecasts, and they cannot be added: "
            "collection and update improvements multiply, they do not sum."),
    }
    (E / "d6_amdahl.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'candidate':34s} {'f':>8s} {'upper_x':>9s}")
    for r in rows:
        print(f"{r['candidate']:34s} {r['fraction_of_wall_removed']:8.4f} "
              f"{r['amdahl_upper_bound_x']:9.3f}")
    print()
    print("kernel per-row reduction 1 -> 64:", out["kernel_scaling"]["per_row_reduction_x"], "x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
