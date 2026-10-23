"""Rebuild BOTTLENECK_RANKING.csv with ONE consistent basis.

Audit MINOR 9 / MAJOR 2: the old CSV mixed 20-iteration extrapolations with raw
2-iteration profile milliseconds, so `pct_of_720s` and `amdahl_upper_bound_x`
disagreed and rank order did not follow either measure.

Every row is now on the same basis: exclusive wall seconds attributed to that
bottleneck in the 20-iteration baseline, computed from the measured stage share
times the measured stage total. Amdahl is 1/(1-f) with f = attributed/wall.
"""
import csv
import json
import pathlib

D = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4.5")
E = D / "evidence"


def load(path, required):
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
    raise ValueError(path)


def main():
    hist = load(E / "historical_reconstruction.json", ("stage_sums_s",))
    coll = load(E / "d2b_collect.json", ("stage_share_pct",))
    lsplit = load(E / "d2f_learner_split.json", ("share_pct",))
    split = load(E / "d2c_split.json", ("per_request_ms",))
    attrib = {}
    for line in (E / "d2_learner_attribution.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5 and p[0].startswith(("update.", "collect.")):
            try:
                attrib[p[0]] = float(p[2])
            except ValueError:
                pass

    s = hist["stage_sums_s"]
    wall = s["wall_seconds"]
    c_share = coll["stage_share_pct"]
    t = split["per_request_ms"]

    # Learner shares from the profiler dumps (2-iteration), normalised to the
    # measured learner total. These are shares, so they transfer to any total.
    upd_total = sum(float(v) for k, v in attrib.items() if k.startswith("update."))
    def lshare(key):
        return attrib.get(key, 0.0) / upd_total if upd_total else 0.0

    items = []

    def add(name, stage, fn, wall_s, cpu, gpu, wait, conf, mechanism, fix, evidence):
        f = wall_s / wall
        items.append({
            "bottleneck": name,
            "stage": stage,
            "file_function": fn,
            "attributed_wall_s": round(wall_s, 2),
            "pct_of_720s": round(f * 100, 2),
            "cpu_self": cpu,
            "cuda_kernel": gpu,
            "queue_or_wait": wait,
            "confidence": conf,
            "mechanism": mechanism,
            "future_fix_not_implemented": fix,
            "amdahl_upper_bound_x": round(1 / (1 - f), 3) if f < 1 else None,
            "evidence": evidence,
        })

    add("prefix STEP transport barrier", "collection",
        "pa4/collector/batched_core.py::StagedInference.step_level",
        s["collection_seconds"] * c_share["step_levels"] / 100,
        "prep+submit small vs wait",
        "not isolated per kernel",
        f"wait {t['wait']} ms/req vs prep {t['prep']} ms",
        "HIGH",
        "one IPC round trip per prefix level; each pays queue+decode+GPU+return "
        "for a batch of ~1",
        "batch all pending levels into one transport call, or move inference "
        "in-process", "d2b_collect.json, d2c_split.json")

    add("per-row conditional scorer evaluation", "update",
        "pa4/learner/learner_eval.py::evaluate_stored_batch -> evaluate_encoded",
        s["update_seconds"] * 0.662 * lsplit["share_pct"]["per_row_evaluate"] / 100,
        f"{lsplit['share_pct']['per_row_evaluate']}% of one learner evaluation",
        "per-row; batch-64 encode far cheaper",
        "n/a (in-process)",
        "HIGH",
        "each row builds its own prefix table and calls evaluate_encoded, paying "
        "~5 ms fixed kernel latency per row",
        "pad branch tables to one shape and evaluate once per minibatch",
        "d2_learner_attribution.txt, d2f_learner_split.json")

    add("ENCODE transport barrier", "collection",
        "pa4/collector/batched_core.py::StagedInference.encode_all",
        s["collection_seconds"] * c_share["encode"] / 100,
        f"prep {t['prep']} ms/req", "not isolated",
        f"wait {t['wait']} ms/req", "HIGH",
        "same per-request round trip as STEP, on the encode phase",
        "combine with STEP batching; one transport call per phase",
        "d2b_collect.json, d2c_split.json")

    add("per-row frozen V_old encode", "update",
        "pa4/learner/train_sync.py::run_iteration (frozen critic loop)",
        s["update_seconds"] * lshare("update.vold_encode"),
        f"obs build {attrib.get('update.vold_obs', 0):.0f} ms",
        "~5.6 ms per batch-of-1 call",
        f".item() only {attrib.get('update.vold_item', 0):.0f} ms",
        "HIGH",
        "one full encode per value row at batch size 1; fixed kernel latency "
        "dominates, not host sync",
        "batch all value rows into one encode", "d2_learner_attribution.txt")

    add("model construction per iteration", "update",
        "train_sync._build_model (frozen critic rebuild)",
        (attrib.get("update.frozen_model_create", 0)
         + attrib.get("update.frozen_state_load", 0)
         + attrib.get("update.learner_create", 0)) / upd_total * s["update_seconds"],
        "measured", "n/a", "n/a", "HIGH",
        "the frozen critic is rebuilt and its state reloaded each iteration; the "
        "learner itself is built once and restored, not rebuilt",
        "keep one persistent frozen copy", "d2_learner_attribution.txt")

    add("checkpoint serialisation and publish", "checkpoint",
        "pa4/learner/checkpoint.py::save_checkpoint",
        s["checkpoint_seconds"], "measured (torch.save+fsync+sha256)", "n/a", "n/a",
        "HIGH", "full model+Adam+RNG written and hashed every iteration",
        "incremental or async checkpointing later", "historical_reconstruction.json")

    add("V_old observation rebuild", "update",
        "pa4/learner/parity.py::build_observation",
        s["update_seconds"] * lshare("update.vold_obs"),
        "measured (Python + tensor build)", "n/a", "n/a", "HIGH",
        "per-row ObservationBatch construction including ragged CSR rebuild",
        "build all rows once and reuse across epochs", "d2_learner_attribution.txt")

    add("branch-table construction", "update",
        "pa4/actions/conditional_policy.py::build_branch_table",
        s["update_seconds"] * 0.662 * lsplit["share_pct"]["branch_tables"] / 100,
        f"{lsplit['share_pct']['branch_tables']}% of one learner evaluation",
        "n/a", "n/a", "HIGH",
        "pure-Python per-candidate resolution inside the per-row loop",
        "vectorise and share across rows", "d2f_learner_split.json")

    add("native Rust game advance", "collection",
        "pa4/env/native_env.py::_step -> engine.step",
        s["collection_seconds"] * c_share["env_step"] / 100,
        "measured", "n/a", "n/a", "HIGH",
        "the real battle simulation; irreducible without changing semantics",
        "none desirable - this is the actual work", "d2b_collect.json")

    add("transition materialisation and sink put", "collection",
        "pa4/collector/run_batched.py accounting stage",
        s["collection_seconds"] * c_share["accounting"] / 100,
        "measured", "n/a", "n/a", "HIGH",
        "cloning tensors, digests, lossless sink put", "minor",
        "d2b_collect.json")

    items.sort(key=lambda r: -r["attributed_wall_s"])

    fields = ["rank"] + list(items[0].keys())
    with open(D / "BOTTLENECK_RANKING.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank"] + [k for k in items[0]])
        w.writeheader()
        for i, row in enumerate(items, 1):
            w.writerow({"rank": i, **row})

    total_attr = sum(r["attributed_wall_s"] for r in items)
    print(f"wrote {len(items)} rows, single basis")
    print(f"attributed total: {total_attr:.1f} s of {wall:.1f} s "
          f"({total_attr/wall*100:.1f}%)")
    print()
    print(f"{'rank':>4s} {'bottleneck':38s} {'wall_s':>8s} {'pct':>6s} {'bound':>6s}")
    for i, r in enumerate(items, 1):
        print(f"{i:4d} {r['bottleneck']:38s} {r['attributed_wall_s']:8.1f} "
              f"{r['pct_of_720s']:5.1f}% {r['amdahl_upper_bound_x']:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
