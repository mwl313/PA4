"""Summarise the instrumented profile dumps into a ranked attribution."""
import glob
import json
import os
import sys


def main():
    prof_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(prof_dir, "profile_iter*.json")))
    if not files:
        raise SystemExit(f"no profiles in {prof_dir}")

    agg = {}
    calls = {}
    for path in files:
        d = json.load(open(path, encoding="utf-8"))
        for name, c in d["counters"].items():
            agg[name] = agg.get(name, 0.0) + c["exclusive_ms"]
            calls[name] = calls.get(name, 0) + c["calls"]

    total = sum(agg.values())
    print(f"profiles: {len(files)}   total exclusive across all spans: {total:.1f} ms")
    print()
    print(f"{'stage':34s} {'calls':>7s} {'excl_ms':>10s} {'%':>6s} {'ms/call':>9s}")
    for name, ms in sorted(agg.items(), key=lambda kv: -kv[1]):
        c = calls[name]
        per = ms / c if c else 0
        print(f"{name:34s} {c:7d} {ms:10.1f} {ms/total*100:5.1f}% {per:9.3f}")

    print()
    print("=== learner-only breakdown (update.*) ===")
    upd = {k: v for k, v in agg.items() if k.startswith("update.")}
    ut = sum(upd.values())
    for name, ms in sorted(upd.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {ms:9.1f} ms  {ms/ut*100:5.1f}% of update  "
              f"({calls[name]} calls)")


if __name__ == "__main__":
    main()
