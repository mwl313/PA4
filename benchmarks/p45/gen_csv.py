"""Generate BOTTLENECK_RANKING.csv and PERF_RUNS.csv from measured evidence."""
import csv
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
    split = load("d2c_split.json", ("per_request_ms",))
    scale = load("d2e_scale.json", ("measurements",))
    base = load("d3_baseline.json", ("median_matches_per_sec",))
    attrib = {}
    for line in (E / "d2_learner_attribution.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith(("update.", "collect.")):
            try:
                attrib[parts[0]] = {"calls": int(parts[1]),
                                    "excl_ms": float(parts[2]),
                                    "pct": float(parts[3].rstrip("%"))}
            except ValueError:
                pass

    s = hist["stage_sums_s"]
    wall = s["wall_seconds"]
    collect_share = s["collection_seconds"] / wall
    update_share = s["update_seconds"] / wall
    by_n = {m["batch"]: m for m in scale["measurements"]}

    def pct_of_wall(fraction_of_stage, stage_share):
        return round(fraction_of_stage * stage_share * 100, 2)

    rows = []

    def add(rank, name, stage, fn, calls, wall_ms, cpu, gpu, wait, conf,
            mechanism, fix, amdahl):
        rows.append({
            "rank": rank,
            "bottleneck": name,
            "stage": stage,
            "file_function": fn,
            "calls_per_run": calls,
            "attributable_wall_ms": wall_ms,
            "pct_of_720s": pct_of_wall(wall_ms / (wall * 1000), 1.0),
            "cpu_self": cpu,
            "cuda_kernel": gpu,
            "queue_or_wait": wait,
            "confidence": conf,
            "mechanism": mechanism,
            "future_fix_not_implemented": fix,
            "amdahl_upper_bound_x": amdahl,
            "evidence": "evidence/d2*.json, d2_learner_attribution.txt",
        })

    # 1. Prefix STEP transport barrier
    sl = coll["stages_ms"]["step_levels"]
    add(1, "prefix STEP transport barrier", "collection",
        "pa4/collector/batched_core.py::StagedInference.step_level",
        "per request per level",
        round(s["collection_seconds"] * coll["stage_share_pct"]["step_levels"] / 100 * 1000, 0),
        "measured (prep+submit small)", "not isolated per call",
        f"measured: wait {split['per_request_ms']['wait']} ms vs prep "
        f"{split['per_request_ms']['prep']} ms",
        "HIGH",
        "one IPC round trip per prefix level; each pays queue + decode + "
        "GPU + return for a batch of ~1",
        "batch all pending levels into one transport call, or move inference "
        "in-process", 1.474)

    # 2. Per-row conditional scorer in the PPO loop
    add(2, "per-row conditional scorer evaluation", "update",
        "pa4/learner/learner_eval.py::evaluate_stored_batch -> "
        "model.evaluate_encoded (per row)",
        "1 per value row per epoch",
        round(s["update_seconds"] * 0.662 * 1000, 0),
        "measured: 73.6% of one learner evaluation is this loop",
        f"kernel: per-row; batch-64 encode is {by_n[64]['kernel_ms']} ms "
        f"vs {by_n[1]['kernel_ms']} ms at batch 1",
        "n/a (in-process)",
        "HIGH",
        "each row builds its own prefix table and calls evaluate_encoded, "
        "paying ~5 ms fixed kernel latency per row",
        "pad all rows' branch tables to one shape and evaluate once per "
        "minibatch", 1.440)

    # 3. Per-row frozen V_old
    add(3, "per-row frozen V_old encode", "update",
        "pa4/learner/train_sync.py::run_iteration (frozen critic loop)",
        attrib.get("update.vold_encode", {}).get("calls", 0),
        round(s["update_seconds"] * 0.232 * 1000, 0),
        f"obs build {attrib.get('update.vold_obs', {}).get('excl_ms', 0)} ms",
        f"~{by_n[1]['kernel_ms']} ms per call (batch of 1)",
        f".item() only {attrib.get('update.vold_item', {}).get('excl_ms', 0)} ms",
        "HIGH",
        "one full encode per value row at batch size 1; fixed kernel latency "
        "dominates, not host synchronisation",
        "batch all value rows into one encode (already proven equivalent for "
        "the trainable encoder)", 1.120)

    # 4. ENCODE transport barrier
    en = coll["stages_ms"]["encode"]
    add(4, "ENCODE transport barrier", "collection",
        "pa4/collector/batched_core.py::StagedInference.encode_all",
        "1 per requesting seat per step",
        round(s["collection_seconds"] * coll["stage_share_pct"]["encode"] / 100 * 1000, 0),
        f"prep {split['per_request_ms']['prep']} ms/req",
        "not isolated",
        f"wait {split['per_request_ms']['wait']} ms/req",
        "HIGH",
        "same per-request round trip as STEP, on the encode phase",
        "combine with the STEP batching work; one transport call per phase", 1.201)

    # 5. Frozen critic / learner model construction per iteration
    fc = attrib.get("update.frozen_model_create", {}).get("excl_ms", 0)
    lc = attrib.get("update.learner_create", {}).get("excl_ms", 0)
    add(5, "model construction per iteration", "update",
        "train_sync._build_model (frozen critic + learner)",
        2,
        round(fc + lc, 0),
        "measured", "n/a", "n/a", "HIGH",
        "two 9.9M-parameter models built from scratch each iteration",
        "keep one persistent frozen copy; reuse the resident learner", 1.030)

    # 6. Checkpoint serialisation
    add(6, "checkpoint serialisation and publish", "checkpoint",
        "pa4/learner/checkpoint.py::save_checkpoint",
        1,
        round(s["checkpoint_seconds"] * 1000, 0),
        "measured (torch.save + fsync + sha256)", "n/a", "n/a", "HIGH",
        "full model + Adam + RNG state written and hashed every iteration",
        "incremental or async checkpointing in a later phase", 1.016)

    # 7. GAE / normalisation
    add(7, "GAE and advantage normalisation", "update",
        "pa4/learner/gae.py::gae_for_cohort / normalize_advantages",
        1,
        round(attrib.get("update.gae", {}).get("excl_ms", 0)
              + attrib.get("update.normalize", {}).get("excl_ms", 0), 0),
        "measured (CPU)", "n/a", "n/a", "HIGH",
        "pure CPU arithmetic over ~380 rows", "already negligible", 1.000)

    # 8. Observation rebuild for V_old
    add(8, "V_old observation rebuild", "update",
        "pa4/learner/parity.py::build_observation",
        attrib.get("update.vold_obs", {}).get("calls", 0),
        round(attrib.get("update.vold_obs", {}).get("excl_ms", 0), 0),
        "measured (Python + tensor build)", "n/a", "n/a", "HIGH",
        "per-row ObservationBatch construction including ragged CSR rebuild",
        "build all rows once and reuse across epochs", 1.014)

    # 9. Native game step
    add(9, "native Rust game advance", "collection",
        "pa4/env/native_env.py::_step -> engine.step",
        "per env step",
        round(coll["stages_ms"]["env_step"], 0),
        "measured", "n/a", "n/a", "HIGH",
        "the actual battle simulation; irreducible without changing semantics",
        "none desirable - this is the real work", 1.007)

    # 10. Transition accounting
    add(10, "transition materialisation and sink put", "collection",
        "pa4/collector/run_batched.py accounting stage",
        "per accepted transition",
        round(coll["stages_ms"]["accounting"], 0),
        "measured", "n/a", "n/a", "HIGH",
        "cloning tensors, digests, lossless sink put", "minor", 1.004)

    with open(D / "BOTTLENECK_RANKING.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote BOTTLENECK_RANKING.csv ({len(rows)} rows)")

    # ---- PERF_RUNS.csv ------------------------------------------------
    runs = []
    runs.append({
        "run_id": "p4_historical_20iter", "kind": "historical",
        "git_sha": "d0f42f1", "instrumented": False, "iterations": 20,
        "wall_s": round(wall, 3),
        "accepted_matches": 320,
        "accepted_matches_per_sec": 0.4439,
        "collection_s": round(s["collection_seconds"], 3),
        "update_s": round(s["update_seconds"], 3),
        "checkpoint_s": round(s["checkpoint_seconds"], 3),
        "dropped": 0, "valid": True,
        "note": "authoritative P4 baseline; outer denominator",
    })
    for t in base["trials"]:
        runs.append({
            "run_id": f"p45_base_trial{t['trial']}", "kind": "uninstrumented",
            "git_sha": "d0f42f1+diag", "instrumented": False,
            "iterations": t["iterations"], "wall_s": t["wall_s"],
            "accepted_matches": t["accepted_matches"],
            "accepted_matches_per_sec": t["accepted_matches_per_sec"],
            "collection_s": t["collection_s"], "update_s": t["update_s"],
            "checkpoint_s": t["checkpoint_s"], "dropped": t["dropped"],
            "valid": t["dropped"] == 0 and t["fingerprints_chain_ok"],
            "note": f"optimizer_steps={t['optimizer_steps']}",
        })
    runs.append({
        "run_id": "p45_instrumented_2iter", "kind": "instrumented",
        "git_sha": "d0f42f1+diag", "instrumented": True, "iterations": 2,
        "wall_s": 57.933, "accepted_matches": 32,
        "accepted_matches_per_sec": 0.5524,
        "collection_s": 33.303, "update_s": 22.946, "checkpoint_s": 1.124,
        "dropped": 0, "valid": True,
        "note": "attribution only; NOT a throughput claim",
    })
    with open(D / "PERF_RUNS.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)
    print(f"wrote PERF_RUNS.csv ({len(runs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
