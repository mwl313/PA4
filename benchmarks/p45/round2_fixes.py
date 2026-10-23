"""P4.5 round-2 corrections.

MAJOR-1 the learner split double-counted: `per_row_evaluate` measured a loop that
  itself calls `_multi_level_table_plain`, so branch-table construction is a
  SUBSET of it. The shares summed to 111.7%. The ranking then listed both as
  separate rows, and the "94.6% non-overlapping" claim was false; the distinct
  total is 88.7% and the exclusive per-row scorer is 118.75 s, not 161.27 s.

  Fix: re-measure the split with branch-table construction HOISTED out of the
  per-row loop, so the four components are genuinely disjoint. Then rebuild the
  CSV, the report table and the receipt from that artifact.

MINORs: stale test counts, 73.6 vs 73.2, 11.5 vs 11.792, "not faster" wording,
  ARCHITECTURE_DECISION batch table, ceiling stated three ways, collection
  residual denominator, missing call-count column, false "no production code
  reads prof.enabled", and the backwards amortisation sentence.
"""
import json
import pathlib
import time

ROOT = pathlib.Path("/home/pa4/projects/PokeAgent4.0")
D = ROOT / "docs/phases/P4.5"
E = D / "evidence"


# ---------------------------------------------------------------------------
# 1. Re-measure the learner split with truly disjoint components.
# ---------------------------------------------------------------------------
def remeasure_split():
    import sys
    import tempfile

    for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
              str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    from pa4.collector.batched_core import BatchedConfig
    from pa4.collector.run_batched import run_batched
    from pa4.learner import learner_eval as le
    from pa4.learner.ingest import build_cohort
    from pa4.learner.parity import build_observation
    from pa4.learner.sink import LosslessBuffer
    from pa4.learner.snapshot import save_snapshot
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model

    def build_model(seed=20261006):
        torch.manual_seed(seed)
        return PA3Model(PA3Config(initialization="random", seed=seed))

    d = tempfile.mkdtemp(prefix="p45_split3_")
    base = build_model()
    snap = save_snapshot(base.state_dict(), d, policy_version=1, committed_matches=0)
    cfg = BatchedConfig(num_envs=4, target_matches=8, max_steps=8000, seed=3,
                        device="cuda", run_id="p45-split3", policy_version=1,
                        snapshot_path=snap.path, transition_capacity=200_000)
    sink = LosslessBuffer(capacity=200_000)
    run_batched(cfg, store=sink)
    cohort = build_cohort(sink.records())
    rows = cohort.value_rows()
    model = build_model().cuda().train()
    n = len(rows)
    REP = 3

    le.evaluate_stored_batch(model, rows[:4], device="cuda")
    torch.cuda.synchronize()

    # (a) observation rebuild
    t0 = time.perf_counter()
    for _ in range(REP):
        obs = [build_observation(r, device="cuda") for r in rows]
    torch.cuda.synchronize()
    obs_t = (time.perf_counter() - t0) / REP

    # (b) branch tables, hoisted OUT of the evaluate loop
    t0 = time.perf_counter()
    for _ in range(REP):
        tables = []
        for i, r in enumerate(rows):
            a = le._table_roster(obs[i].active_to_roster())
            tables.append(le._multi_level_table_plain(r, a).to("cuda"))
    torch.cuda.synchronize()
    tbl_t = (time.perf_counter() - t0) / REP

    # (c) batched encoder
    batched = le._stack_observations(obs, "cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REP):
        model.encode(batched)
    torch.cuda.synchronize()
    enc_t = (time.perf_counter() - t0) / REP

    # (d) evaluate_encoded ONLY (tables already built)
    encoded_all = model.encode(batched)
    sels = [torch.tensor([[int(b["selected_index"]) for b in r.branches]],
                         dtype=torch.long, device="cuda") for r in rows]
    t0 = time.perf_counter()
    for _ in range(REP):
        for i in range(n):
            single = le._slice_encoded(encoded_all, i)
            model.evaluate_encoded(single, tables[i], selected=sels[i])
    torch.cuda.synchronize()
    ev_t = (time.perf_counter() - t0) / REP

    total = obs_t + tbl_t + enc_t + ev_t
    out = {
        "rows": n,
        "repeats": REP,
        "method": "branch tables hoisted OUT of the evaluate loop so the four "
                  "components are disjoint and sum to the total",
        "total_ms": round(total * 1000, 1),
        "components_ms": {
            "obs_rebuild": round(obs_t * 1000, 1),
            "branch_tables": round(tbl_t * 1000, 1),
            "batched_encoder": round(enc_t * 1000, 1),
            "evaluate_encoded_only": round(ev_t * 1000, 1),
        },
        "share_pct": {
            "obs_rebuild": round(obs_t / total * 100, 1),
            "branch_tables": round(tbl_t / total * 100, 1),
            "batched_encoder": round(enc_t / total * 100, 1),
            "evaluate_encoded_only": round(ev_t / total * 100, 1),
        },
        "share_sum_pct": round(sum([obs_t, tbl_t, enc_t, ev_t]) / total * 100, 1),
        "per_row_evaluate_ms": round(ev_t / n * 1000, 3),
    }
    (E / "d2f_learner_split.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print("disjoint split measured:", out["share_pct"],
          "sum", out["share_sum_pct"], "%")
    return out


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
    split3 = remeasure_split()

    # ---- rebuild the ranking CSV on the corrected, disjoint basis ------
    import csv
    hist = load(E / "historical_reconstruction.json", ("stage_sums_s",))
    coll = load(E / "d2b_collect.json", ("stage_share_pct",))
    tsplit = load(E / "d2c_split.json", ("per_request_ms",))
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
    upd_total = sum(float(v) for k, v in attrib.items() if k.startswith("update."))
    sh = split3["share_pct"]

    def lshare(key):
        return attrib.get(key, 0.0) / upd_total if upd_total else 0.0

    items = []

    def add(name, stage, fn, wall_s, cpu, gpu, wait, conf, mechanism, fix,
            evidence, shape):
        f = wall_s / wall
        items.append({
            "bottleneck": name, "stage": stage, "file_function": fn,
            "input_shape": shape,
            "attributed_wall_s": round(wall_s, 2),
            "pct_of_720s": round(f * 100, 2),
            "cpu_self": cpu, "cuda_kernel": gpu, "queue_or_wait": wait,
            "confidence": conf, "mechanism": mechanism,
            "future_fix_not_implemented": fix,
            "amdahl_upper_bound_x": round(1 / (1 - f), 3) if f < 1 else None,
            "evidence": evidence, "source_git_sha": "d0f42f1",
        })

    # Per-row scorer = ppo_loop share x evaluate_encoded_only share (disjoint now)
    add("per-row conditional scorer evaluation", "update",
        "pa4/learner/learner_eval.py::evaluate_stored_batch (evaluate_encoded loop)",
        s["update_seconds"] * 0.662 * sh["evaluate_encoded_only"] / 100,
        f"{sh['evaluate_encoded_only']}% of learner evaluation",
        "per-row; batch-64 encode far cheaper", "n/a (in-process)", "HIGH",
        "one evaluate_encoded call per row; fixed kernel latency per call",
        "pad branch tables to one shape and evaluate once per minibatch",
        "d2_learner_attribution.txt, d2f_learner_split.json",
        "[1, <=4 levels, 64, 6] per row")

    add("prefix STEP transport barrier", "collection",
        "pa4/collector/batched_core.py::StagedInference.step_level",
        s["collection_seconds"] * coll["stage_share_pct"]["step_levels"] / 100,
        "prep+submit small vs wait", "not isolated per kernel",
        f"wait {tsplit['per_request_ms']['wait']} ms/req vs prep "
        f"{tsplit['per_request_ms']['prep']} ms", "HIGH",
        "one IPC round trip per prefix level for a batch of ~1",
        "batch all pending levels into one transport call, or move inference "
        "in-process", "d2b_collect.json, d2c_split.json",
        "[1, 64] candidates per level")

    add("ENCODE transport barrier", "collection",
        "pa4/collector/batched_core.py::StagedInference.encode_all",
        s["collection_seconds"] * coll["stage_share_pct"]["encode"] / 100,
        f"prep {tsplit['per_request_ms']['prep']} ms/req", "not isolated",
        f"wait {tsplit['per_request_ms']['wait']} ms/req", "HIGH",
        "same per-request round trip as STEP, on the encode phase",
        "combine with STEP batching; one transport call per phase",
        "d2b_collect.json, d2c_split.json", "[1, 96, ...] observation per request")

    add("per-row frozen V_old encode", "update",
        "pa4/learner/train_sync.py::run_iteration (frozen critic loop)",
        s["update_seconds"] * lshare("update.vold_encode"),
        f"obs build {attrib.get('update.vold_obs', 0):.0f} ms",
        "~5.6 ms per batch-of-1 call",
        f".item() only {attrib.get('update.vold_item', 0):.0f} ms", "HIGH",
        "one full encode per value row at batch size 1",
        "batch all value rows into one encode",
        "d2_learner_attribution.txt", "[1, 96, ...] observation per row")

    add("branch-table construction", "update",
        "pa4/actions/conditional_policy.py::build_branch_table",
        s["update_seconds"] * 0.662 * sh["branch_tables"] / 100,
        f"{sh['branch_tables']}% of learner evaluation", "n/a", "n/a", "HIGH",
        "pure-Python per-candidate resolution; counted separately from the "
        "evaluate loop it feeds",
        "vectorise and share across rows", "d2f_learner_split.json",
        "[1, 1, 64, 6] per row")

    add("V_old observation rebuild", "update",
        "pa4/learner/parity.py::build_observation",
        s["update_seconds"] * lshare("update.vold_obs"),
        "measured (Python + tensor build)", "n/a", "n/a", "HIGH",
        "per-row ObservationBatch construction including ragged CSR rebuild",
        "build all rows once and reuse across epochs", "d2_learner_attribution.txt",
        "[1, 96, 32/50/40] leaves + ragged CSR")

    add("checkpoint serialisation and publish", "checkpoint",
        "pa4/learner/checkpoint.py::save_checkpoint",
        s["checkpoint_seconds"], "measured (torch.save+fsync+sha256)", "n/a", "n/a",
        "HIGH", "full model+Adam+RNG written and hashed every iteration",
        "incremental or async checkpointing later",
        "historical_reconstruction.json", "142 tensors + Adam + RNG")

    add("model construction per iteration", "update",
        "train_sync._build_model (frozen critic rebuild)",
        (attrib.get("update.frozen_model_create", 0)
         + attrib.get("update.frozen_state_load", 0)
         + attrib.get("update.learner_create", 0)) / upd_total * s["update_seconds"],
        "measured", "n/a", "n/a", "HIGH",
        "frozen critic rebuilt and reloaded each iteration; the learner is "
        "built once and restored, not rebuilt",
        "keep one persistent frozen copy", "d2_learner_attribution.txt",
        "9,889,755 params")

    add("native Rust game advance", "collection",
        "pa4/env/native_env.py::_step -> engine.step",
        s["collection_seconds"] * coll["stage_share_pct"]["env_step"] / 100,
        "measured", "n/a", "n/a", "HIGH",
        "the real battle simulation; irreducible without changing semantics",
        "none desirable - this is the actual work", "d2b_collect.json",
        "[4] env rows")

    add("transition materialisation and sink put", "collection",
        "pa4/collector/run_batched.py accounting stage",
        s["collection_seconds"] * coll["stage_share_pct"]["accounting"] / 100,
        "measured", "n/a", "n/a", "HIGH",
        "cloning tensors, digests, lossless sink put", "minor",
        "d2b_collect.json", "per accepted transition")

    items.sort(key=lambda r: -r["attributed_wall_s"])
    with open(D / "BOTTLENECK_RANKING.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank"] + list(items[0].keys()))
        w.writeheader()
        for i, row in enumerate(items, 1):
            w.writerow({"rank": i, **row})

    total_attr = sum(r["attributed_wall_s"] for r in items)
    print(f"CSV rebuilt: total attributed {total_attr:.1f} s "
          f"({total_attr/wall*100:.1f}% of {wall:.1f})")
    print(f"{'rank':>4s} {'bottleneck':40s} {'wall_s':>8s} {'pct':>6s} {'bound':>6s}")
    for i, r in enumerate(items, 1):
        print(f"{i:4d} {r['bottleneck']:40s} {r['attributed_wall_s']:8.1f} "
              f"{r['pct_of_720s']:5.1f}% {r['amdahl_upper_bound_x']:6.3f}")

    # ---- patch the remaining MINORs ----------------------------------
    patch_minors(sh, tsplit)
    return 0


def patch_minors(sh, tsplit):
    import re

    # FINAL_REPORT
    f = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")
    f = f.replace("no production code path reads `prof.enabled`",
                  "the only production read of `prof.enabled` under the P4 "
                  "loop is the guarded per-iteration dump at the end of "
                  "`run_iteration`, which records nothing when disabled")
    f = f.replace(
        "because per-iteration setup amortises over fewer iterations here",
        "because fixed per-run setup is spread over 4 iterations here versus 20 "
        "in the historical trial (fewer iterations amortise setup LESS, so the "
        "4-iteration rate is flattered by other factors and the two are not "
        "comparable)")
    f = f.replace(
        "2952.2 ms unaccounted of 15807.7 ms",
        f"{2952.2} ms unaccounted of {18760} ms measured collection wall "
        f"({round(2952.2/18760*100,1)}%)")
    f = f.replace("call counts", "call counts (in section 4/5)")
    (D / "FINAL_REPORT.md").write_text(f, encoding="utf-8")

    # ARCHITECTURE_DECISION
    a = (D / "ARCHITECTURE_DECISION.md").read_text(encoding="utf-8")
    a = a.replace("Wait is **67× prep**.", "Wait is **65.99× prep**.")
    a = a.replace("11.5 ms per request", "11.792 ms per request")
    a = a.replace("**11.492**", "**11.792**")
    a = a.replace(
        "The measured in-process encode for a 4-row batch is 47.1 ms for 183 rows",
        "The measured in-process batched encode is 47.1 ms for 183 rows")
    a = a.replace("~1.3 matches/s", "~1.1 matches/s")
    a = a.replace("2.5–3.5×", "~2.5×")
    (D / "ARCHITECTURE_DECISION.md").write_text(a, encoding="utf-8")

    # RECEIPT stale counts
    rp = D / "RECEIPT.json"
    d = json.loads(rp.read_text(encoding="utf-8"))
    d["gate_notes"]["D45_6"] = (
        "286 tests green (272 inherited + 14 diagnostics), P0 smoke PASS, P4 "
        "behaviour contracts unchanged, fresh-context auditor verified real file "
        "access, findings addressed.")
    d["key_measurements"]["learner_evaluation_split_pct"] = sh
    d["key_measurements"]["collection_residual_pct"] = round(2952.2 / 18760 * 100, 1)
    rp.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

    # d5_hypotheses stale values
    hp = E / "d5_hypotheses.json"
    h = json.loads(hp.read_text(encoding="utf-8"))
    txt = json.dumps(h)
    txt = txt.replace("73.6%", f"{sh['evaluate_encoded_only']}%")
    txt = txt.replace("~11.5 ms", "11.792 ms")
    txt = txt.replace("is not faster than", "is slightly faster than")
    h = json.loads(txt)
    h["hypotheses"]["H3_ipc_queue_barrier"]["detail"] = (
        "The transport path pays 11.792 ms per request while payload prep is "
        "0.179 ms. step_levels + encode are 94.3% of accounted collection, and "
        "both are dominated by submit/queue/wait.")
    hp.write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")

    print("MINORs patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
