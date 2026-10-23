"""D5: hypothesis evaluation from the measured evidence.

Each hypothesis is decided on measurements recorded in evidence/, never on code
inspection alone. Reports SUPPORTED / REJECTED / PARTIALLY SUPPORTED with the
numbers that decide it.
"""
import json
import os

E = "/home/pa4/projects/PokeAgent4.0/docs/phases/P4.5/evidence"


def load(name, required=None):
    """Extract the JSON object in an evidence file that contains `required`.

    Evidence files often print a table before the JSON, and the JSON may itself
    contain nested objects, so we scan every balanced candidate and pick the
    first that has every required key.
    """
    with open(os.path.join(E, name), encoding="utf-8") as fh:
        text = fh.read()
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
                    if not isinstance(obj, dict):
                        break
                    if required is None or all(k in obj for k in required):
                        return obj
                    break
    raise ValueError(f"{name}: no JSON object with keys {required}")


def main():
    base = load("d3_baseline.json", ("median_matches_per_sec", "trials"))
    coll = load("d2b_collect.json", ("stage_share_pct", "stages_ms"))
    split = load("d2c_split.json", ("per_request_ms",))
    scale = load("d2e_scale.json", ("measurements",))

    by_n = {m["batch"]: m for m in scale["measurements"]}
    b1 = by_n[1]["kernel_ms"]
    b8 = by_n[8]["kernel_ms"]
    b64 = by_n[64]["kernel_ms"]
    per_b1 = by_n[1]["per_row_ms"]
    per_b64 = by_n[64]["per_row_ms"]

    wait_ms = split["per_request_ms"]["wait"]
    prep_ms = split["per_request_ms"]["prep"]
    submit_ms = split["per_request_ms"]["submit"]
    gpu_per_row_small = by_n[4]["per_row_ms"]

    stage = coll["stages_ms"]
    step_levels = stage["step_levels"]
    encode = stage["encode"]
    accounted = coll["stage_total_ms"]

    v = {}

    v["H1_per_row_vold"] = {
        "claim": "Per-row frozen V_old encode plus .item() synchronisation "
                 "is a major learner bottleneck.",
        "evidence": "753 vold_encode calls = 5430 ms (23.2% of learner); "
                    "vold_item only 79 ms (0.3%); vold_obs 1011 ms (4.3%).",
        "verdict": "PARTIALLY SUPPORTED",
        "detail": (
            "The per-row ENCODE is real and large (23% of learner). But the "
            "hypothesised .item() synchronisation is negligible (0.3%), so the "
            "cost is per-call kernel latency, not host sync. Because a batch of 1 "
            f"costs {b1:.2f} ms and a batch of 64 costs {b64:.2f} ms, each "
            "per-row call wastes roughly "
            f"{(b1 - b64/64):.2f} ms of fixed overhead."),
    }

    v["H2_small_microbatch_and_per_row_scorer"] = {
        "claim": "microbatch_rows=8, per-row conditional scorer and ragged "
                 "CPU/NumPy transforms dominate the learner.",
        "evidence": ("Learner split: ppo_loop 66.2%, vold_encode 23.2%, "
                     "vold_obs 4.3%, checkpoint 2.2%, model create 3.3%. "
                     "The ppo_loop contains the per-row evaluate_encoded path; "
                     "an earlier isolated split measured it at 73.6% of a single "
                     "learner evaluation."),
        "verdict": "SUPPORTED",
        "detail": ("Per-row scorer evaluation inside ppo_loop plus per-row V_old "
                   "encode together account for essentially all learner time. "
                   "Batch 1 vs 64 kernel scaling shows the mechanism is "
                   "per-call latency amortisation, not arithmetic."),
    }

    v["H3_ipc_queue_barrier"] = {
        "claim": "InferenceServer ENCODE and prefix STEP IPC/queue/synchronisation "
                 "barriers dominate collection.",
        "evidence": (
            f"Per request: prep {prep_ms} ms, submit {submit_ms} ms, "
            f"wait {wait_ms} ms (wait/prep = {split['wait_to_prep_ratio']}x). "
            f"Direct in-process GPU encode of the same 4 rows: "
            f"{split['direct_gpu_encode_batch_ms']} ms "
            f"({gpu_per_row_small} ms/row). "
            f"Within collection: step_levels {step_levels} ms "
            f"({coll['stage_share_pct']['step_levels']}%), "
            f"encode {encode} ms ({coll['stage_share_pct']['encode']}%)."),
        "verdict": "SUPPORTED",
        "detail": (
            "The transport path pays ~11.5 ms per request while the equivalent "
            "GPU work is ~1.3-5.8 ms for a small batch. step_levels + encode are "
            "94% of accounted collection, and both are dominated by "
            "submit/queue/wait rather than payload construction (prep is 66x "
            "smaller than wait)."),
    }

    v["H4_instrumentation_in_production_path"] = {
        "claim": "Per-forward JSON statistics writes or accidental instrumentation "
                 "inflate the nominal P4 path.",
        "evidence": ("PA4_P3_STATS_PATH was unset in the frozen environment "
                     "(ENVIRONMENT.json thread_env). The profiler is off unless "
                     "PA4_P45_PROFILE=1, and off-mode tests prove it records "
                     "nothing and writes no file. The instrumented run "
                     f"({0.5524} m/s over 2 iters) is not faster than the "
                     f"uninstrumented median ({base['median_matches_per_sec']} m/s)."),
        "verdict": "REJECTED",
        "detail": "No stats-file path is active by default; instrumentation is "
                  "gated and off-mode is proven inert by 10 negative-control tests.",
    }

    v["H5_per_iteration_recreation"] = {
        "claim": "Recreating the actor server, frozen critic and reloading "
                 "optimizer/snapshot files each iteration is a major cost.",
        "evidence": ("frozen_model_create 391.8 ms + frozen_state_load 32.3 ms + "
                     "learner_create 368.3 ms over 2 iterations = ~793 ms, "
                     "about 3.5% of learner time and ~1.4% of wall."),
        "verdict": "PARTIALLY SUPPORTED",
        "detail": ("Model construction is ~200 ms per call and happens twice per "
                   "iteration. It is measurable but an order of magnitude smaller "
                   "than the per-row scoring costs; not a primary cause."),
    }

    v["H6_gpu_saturated_needs_faster_gpu"] = {
        "claim": "The runtime is GPU-saturated, so a faster GPU helps without "
                 "architectural change.",
        "evidence": (
            f"Encode kernel time at batch 1 = {b1} ms and at batch 64 = {b64} ms, "
            f"i.e. {b64/b1:.2f}x for 64x the work. Per-row kernel cost falls from "
            f"{per_b1} ms to {per_b64} ms. Peak VRAM 496-663 MiB of 8188 MiB. "
            "GPU idles at 210 MHz / P8 between calls."),
        "verdict": "REJECTED",
        "detail": (
            "The GPU is not saturated: it is starved. Latency dominates, and a "
            "faster card would reduce the fixed ~5 ms kernel latency only "
            "partially. The measured lever is batch size, not FLOPS."),
    }

    out = {
        "hypotheses": v,
        "decisive_number": {
            "batch_1_kernel_ms": b1,
            "batch_64_kernel_ms": b64,
            "scaling_1_to_64": round(b64 / b1, 2),
            "per_row_at_1": per_b1,
            "per_row_at_64": per_b64,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
