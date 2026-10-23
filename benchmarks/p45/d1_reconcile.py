"""D1: recompute the P4 20-iteration ledger from the authoritative artifacts.

Reads FULL_STACK_BASELINE.csv and evidence/full_run.json, recomputes all sums,
and reconciles the outer stopwatch (720.948 s) against the CSV row sum
(720.541 s). Does NOT modify any P4 evidence.
"""
import csv
import hashlib
import json
import os

REPO = "/home/pa4/projects/PokeAgent4.0"
P4 = os.path.join(REPO, "docs/phases/P4")
OUT = os.path.join(REPO, "docs/phases/P4.5")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    csv_path = os.path.join(P4, "FULL_STACK_BASELINE.csv")
    json_path = os.path.join(P4, "evidence/full_run.json")

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(json_path, "r", encoding="utf-8") as fh:
        full = json.load(fh)

    def f(r, k):
        return float(r[k])

    n = len(rows)
    sums = {
        "collection_seconds": sum(f(r, "collection_seconds") for r in rows),
        "update_seconds": sum(f(r, "update_seconds") for r in rows),
        "checkpoint_seconds": sum(f(r, "checkpoint_seconds") for r in rows),
        "wall_seconds": sum(f(r, "wall_seconds") for r in rows),
    }
    residual = sums["wall_seconds"] - (sums["collection_seconds"]
                                        + sums["update_seconds"]
                                        + sums["checkpoint_seconds"])

    outer = float(full["summary"]["total_wall_seconds"])
    rec = {
        "source": {
            "csv": {"file": "docs/phases/P4/FULL_STACK_BASELINE.csv",
                    "sha256": sha256(csv_path), "rows": n},
            "json": {"file": "docs/phases/P4/evidence/full_run.json",
                     "sha256": sha256(json_path)},
        },
        "iterations": n,
        "stage_sums_s": {k: round(v, 3) for k, v in sums.items()},
        "residual_inside_rows_s": round(residual, 3),
        "shares_of_row_sum_pct": {
            "collection": round(sums["collection_seconds"] / sums["wall_seconds"] * 100, 2),
            "update": round(sums["update_seconds"] / sums["wall_seconds"] * 100, 2),
            "checkpoint": round(sums["checkpoint_seconds"] / sums["wall_seconds"] * 100, 2),
            "residual": round(residual / sums["wall_seconds"] * 100, 2),
        },
        "outer_stopwatch_s": outer,
        "outer_minus_row_sum_s": round(outer - sums["wall_seconds"], 3),
    }

    # Cross-check against the JSON summary so the two artifacts agree.
    s = full["summary"]
    rec["json_summary_crosscheck"] = {
        "iterations": s["iterations"],
        "generated_natural_matches": s["generated_natural_matches"],
        "accepted_unique_matches": s["accepted_unique_matches"],
        "optimizer_steps_cumulative_final": s["optimizer_steps_cumulative_final"],
        "dropped_total": s["dropped_total"],
        "duplicates_total": s["duplicates_total"],
        "nan_inf_total": s["nan_inf_total"],
        "accepted_training_matches_per_sec": s["accepted_training_matches_per_sec"],
        "peak_vram_mib": s["peak_vram_mib"],
        "fingerprint_chain_ok": s["fingerprint_chain_ok"],
        "distinct_policy_versions": s["distinct_policy_versions"],
        "early_stopped_iterations": s["early_stopped_iterations"],
    }

    # Per-iteration medians/spread (never just the mean).
    def stats(key):
        vals = sorted(f(r, key) for r in rows)
        mid = len(vals) // 2
        med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        return {"min": round(vals[0], 3), "median": round(med, 3),
                "max": round(vals[-1], 3),
                "spread_pct": round((vals[-1] - vals[0]) / med * 100, 1) if med else None}

    rec["per_iteration"] = {k: stats(k) for k in
                            ("collection_seconds", "update_seconds",
                             "checkpoint_seconds", "wall_seconds")}

    # Amdahl counterfactuals (sequential topology, all else unchanged).
    w = sums["wall_seconds"]
    rec["amdahl_counterfactuals"] = {
        "eliminate_collector_only": round(w / (w - sums["collection_seconds"]), 3),
        "eliminate_update_only": round(w / (w - sums["update_seconds"]), 3),
        "eliminate_checkpoint_only": round(w / (w - sums["checkpoint_seconds"]), 3),
        "eliminate_collector_and_update": round(
            w / (w - sums["collection_seconds"] - sums["update_seconds"]), 3),
        "note": "upper bounds for a strictly sequential topology; NOT forecasts",
    }

    # What the two named timers actually contain (source-reading result).
    rec["timer_semantics"] = {
        "collection_seconds": (
            "train_sync.run_iteration: from the call into run_batched() until it "
            "returns. Includes server creation/start, snapshot load into the "
            "inference child, initial reset, every ENCODE/STEP round trip and its "
            "wait, native step/reset, transition materialisation and sink put."),
        "update_seconds": (
            "train_sync.run_iteration: from after cohort build until the PPO "
            "updater returns. Includes frozen critic construction + state load, "
            "per-row V_old build_observation/encode/value/.item, GAE, advantage "
            "normalisation, per-microbatch row rebuild, ragged stacking, batched "
            "trainable encoder, per-row evaluate_encoded, backward, clip, Adam."),
        "checkpoint_seconds": (
            "torch.save of the full state, fsync, sha256, atomic rename, pointer "
            "publication and the policy snapshot write."),
        "wall_seconds": (
            "per-iteration perf_counter span covering collection + update + "
            "checkpoint + in-iteration residual (cohort assembly, publish)."),
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "evidence", "historical_reconstruction.json"),
              "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
