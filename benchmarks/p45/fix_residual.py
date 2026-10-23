"""Fix the collection residual sentence to use the correct denominator."""
import pathlib

p = pathlib.Path("/home/pa4/projects/PokeAgent4.0/docs/phases/P4.5/FINAL_REPORT.md")
s = p.read_text(encoding="utf-8")

s = s.replace(
    "The existing collection stage timers do not cover **18.7%** of measured collection wall "
    "(`d2b_collect.json`: 2952.2 ms unaccounted of 15807.7 ms); that residual includes service "
    "startup, snapshot load and shutdown.",
    "The existing collection stage timers cover 15807.7 ms of the 18760.0 ms measured "
    "collection wall. The remaining **2952.2 ms (15.7% of collection wall)** is not "
    "attributed by those timers; it includes service startup, snapshot load and "
    "shutdown (`d2b_collect.json`).")

p.write_text(s, encoding="utf-8")
print("residual denominator corrected")
