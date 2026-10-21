"""Negative controls for the COMMITTED scripts/p0_smoke.py.

Each variant mutates one expectation and must exit non-zero. Passing this proves
every check branch is live under -O, not just the version pin.
"""

import pathlib
import re
import subprocess
import sys
import tempfile

SRC = pathlib.Path("scripts/p0_smoke.py")
original = SRC.read_text(encoding="utf-8")

# (label, mutated source, expected error fragment)
VARIANTS = [
    (
        "wrong_torchrl_pin",
        original.replace('"torchrl": "0.14.1"', '"torchrl": "9.9.9"'),
        "P0_SMOKE_FAIL: torchrl",
    ),
    (
        "wrong_tensordict_pin",
        original.replace('"tensordict": "0.14.3"', '"tensordict": "0.0.1"'),
        "P0_SMOKE_FAIL: tensordict",
    ),
    (
        "wrong_tensordict_shape",
        original.replace('torch.zeros(2, 4)', 'torch.zeros(2, 5)'),
        "P0_SMOKE_FAIL",
    ),
    (
        "wrong_cuda_sum",
        original.replace("check(total == 64.0", "check(total == 65.0"),
        "P0_SMOKE_FAIL: 8x8 cuda sum",
    ),
]

failures = 0
for label, mutated, expected in VARIANTS:
    if mutated == original:
        print(f"SKIP {label}: mutation did not apply (pattern drifted)")
        failures += 1
        continue
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(mutated)
        path = fh.name
    proc = subprocess.run(
        [sys.executable, "-O", path], capture_output=True, text=True
    )
    combined = proc.stdout + proc.stderr
    ok = proc.returncode != 0 and expected in combined
    print(f"{'PASS' if ok else 'FAIL'} {label}: exit={proc.returncode} "
          f"matched={expected!r}")
    if not ok:
        failures += 1
        print(combined[-300:])
    pathlib.Path(path).unlink()

print(f"NEGATIVE_CONTROL_{'PASS' if failures == 0 else 'FAIL'} variants={len(VARIANTS)}")
sys.exit(0 if failures == 0 else 1)
