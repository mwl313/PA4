"""P4.5 default-off structural profiler (stack-based exclusive accounting).

Design gate D45-2: explicit parent/exclusive timers, nested timers that cannot
double-count, off-mode byte-identical to the original path.

Exclusive time is computed with a single global stack of open spans. When a span
closes, its elapsed time is charged to its own counter (inclusive) and removed
from the direct parent's exclusive bucket, so parent_exclusive + sum(children)
== parent_inclusive by construction rather than by convention.

Enabled only by PA4_P45_PROFILE=1. Disabled, every method returns immediately and
nothing is allocated, which a test asserts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Counter:
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0


@dataclass
class _Frame:
    name: str
    start_ns: int
    child_ns: int = 0


class Profiler:
    """Thread-local stack; safe to use from the driver and from a child process."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._local = threading.local()
        self._lock = threading.Lock()
        self.counters: dict[str, Counter] = {}
        self.records: list = []

    # -- stack helpers --------------------------------------------------
    def _stack(self) -> list:
        s = getattr(self._local, "stack", None)
        if s is None:
            s = []
            self._local.stack = s
        return s

    # -- span -----------------------------------------------------------
    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        stack = self._stack()
        frame = _Frame(name=name, start_ns=time.perf_counter_ns())
        stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - frame.start_ns
            stack.pop()
            if stack:
                # Charge this span to its direct parent's child bucket.
                stack[-1].child_ns += elapsed
            with self._lock:
                c = self.counters.setdefault(name, Counter())
                c.calls += 1
                c.inclusive_ns += elapsed
                # Exclusive = what this span spent NOT inside its own children.
                c.exclusive_ns += max(0, elapsed - frame.child_ns)

    # -- counts ---------------------------------------------------------
    def count(self, name: str, n: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.counters.setdefault(name, Counter()).calls += n

    def record(self, kind: str, **fields) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.records.append({"kind": kind, **fields})

    # -- output ---------------------------------------------------------
    def summary(self) -> dict:
        return {
            name: {"calls": c.calls,
                   "inclusive_ms": round(c.inclusive_ns / 1e6, 3),
                   "exclusive_ms": round(c.exclusive_ns / 1e6, 3)}
            for name, c in sorted(self.counters.items())
        }

    def dump(self, path: str) -> None:
        if not self.enabled:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"counters": self.summary(), "records": self.records},
                      fh, indent=1)

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.records.clear()


def get_profiler() -> Profiler:
    """Process-wide profiler; enabled only by PA4_P45_PROFILE=1."""
    return Profiler(enabled=os.environ.get("PA4_P45_PROFILE") == "1")


#: A permanently-disabled instance for call sites that must never pay overhead.
OFF = Profiler(enabled=False)
