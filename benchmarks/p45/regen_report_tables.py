"""Regenerate the FINAL_REPORT tables directly from BOTTLENECK_RANKING.csv.

Hand-written tables have drifted from the CSV three times. This replaces the
top-five table, the transport line and the residual line with values parsed
from the CSV and the evidence files, so the report cannot disagree with its own
artifacts.
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
    rows = list(csv.DictReader(open(D / "BOTTLENECK_RANKING.csv",
                                    encoding="utf-8")))
    by_rank = {int(r["rank"]): r for r in rows}
    split = load(E / "d2c_split.json", ("per_request_ms",))
    coll = load(E / "d2b_collect.json", ("stage_share_pct", "unaccounted_ms",
                                         "stage_total_ms"))
    hist = load(E / "historical_reconstruction.json", ("stage_sums_s",))

    t = split["per_request_ms"]
    ratio = split["wait_to_prep_ratio"]
    residual_pct = round(coll["unaccounted_ms"] / coll["stage_total_ms"] * 100, 1)

    f = (D / "FINAL_REPORT.md").read_text(encoding="utf-8")

    # ---- transport + residual line (replace whatever is there) ----------
    start = f.index("Per request: prep ")
    end = f.index("## 6. Hypothesis verdicts")
    new_line = (
        f"Per request: prep {t['prep']} ms, submit {t['submit']} ms, "
        f"**wait {t['wait']} ms** — wait/prep = **{ratio}×**.\n\n"
        f"The existing collection stage timers do not cover "
        f"**{residual_pct}%** of measured collection wall "
        f"(`d2b_collect.json`: {coll['unaccounted_ms']} ms unaccounted of "
        f"{coll['stage_total_ms']} ms); that residual includes service startup, "
        f"snapshot load and shutdown.\n\n")
    f = f[:start] + new_line + f[end:]

    # ---- top-five table, straight from the CSV -------------------------
    ts = f.index("## 7. Top five bottlenecks with Amdahl bounds")
    te = f.index("Full ten-row table with file/function")
    s = hist["stage_sums_s"]
    lines = [
        "## 7. Top five bottlenecks with Amdahl bounds",
        "",
        "Every value below is copied from `BOTTLENECK_RANKING.csv`, which computes",
        "each bottleneck's attributed wall time on ONE basis: the measured stage",
        f"share applied to the 20-iteration totals (collection {s['collection_seconds']} s,",
        f"update {s['update_seconds']} s). Amdahl is 1/(1-f) with f = attributed/wall.",
        "",
        "| Rank | Bottleneck | Attributed wall | % of 720.5 s | Confidence | Amdahl bound |",
        "|---|---|---:|---:|---|---:|",
    ]
    for i in range(1, 6):
        r = by_rank[i]
        lines.append(
            f"| {i} | {r['bottleneck']} | {r['attributed_wall_s']} s | "
            f"{r['pct_of_720s']}% | {r['confidence']} | "
            f"**{r['amdahl_upper_bound_x']}×** |")
    lines += [
        "",
        f"The ten ranked bottlenecks together attribute "
        f"{sum(float(r['attributed_wall_s']) for r in rows):.1f} s of "
        f"{s['wall_seconds']:.1f} s "
        f"({sum(float(r['attributed_wall_s']) for r in rows)/s['wall_seconds']*100:.1f}%), "
        "on a single non-overlapping basis.",
        "",
    ]
    f = f[:ts] + "\n".join(lines) + f[te:]

    # ---- H3/H6 evidence wording ----------------------------------------
    f = f.replace(
        "| H3 | InferenceServer IPC/queue barriers | **SUPPORTED** | wait 11.5 ms vs prep 0.17 ms; 94% of collection is round trips |",
        f"| H3 | InferenceServer IPC/queue barriers | **SUPPORTED** | wait {t['wait']} ms vs "
        f"prep {t['prep']} ms ({ratio:.0f}×); step_levels + encode are "
        f"{coll['stage_share_pct']['step_levels'] + coll['stage_share_pct']['encode']:.1f}% "
        "of accounted collection |")
    f = f.replace(
        "| H6 | GPU saturated; faster GPU helps unmodified | **REJECTED** | 1.94× cost for 64× work; 0.5–0.7 GB of 8 GB used; GPU idles at 210 MHz/P8 |",
        "| H6 | GPU saturated; faster GPU helps unmodified | **REJECTED** | "
        "1.94× cost for 64× work — the batch-scaling measurement alone decides "
        "this; peak VRAM and a single idle-state clock sample are NOT used as "
        "utilization evidence |")

    (D / "FINAL_REPORT.md").write_text(f, encoding="utf-8")
    print("FINAL_REPORT tables regenerated from BOTTLENECK_RANKING.csv")
    for i in range(1, 6):
        r = by_rank[i]
        print(f"  rank {i}: {r['bottleneck'][:40]:42s} {r['attributed_wall_s']:>7s}s "
              f"{r['amdahl_upper_bound_x']:>6s}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
