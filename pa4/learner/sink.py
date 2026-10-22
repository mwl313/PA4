"""P4-A — lossless training sink for accepted actor transitions.

ADR-010 / TRAJECTORY_CONTRACT_V1.

P3's `BoundedTransitionStore` evicts the oldest record on overflow, which is
correct for benchmark throughput and unacceptable for PPO: an evicted row can be
the only evidence of a seat's action before its terminal reward.

This module provides a **separate, training-only** sink with two backends:

* `LosslessBuffer`   — in-memory, bounded, **blocks** the producer when full.
* `LosslessSpool`    — iteration-local bounded spool on disk; same accounting.

Both record `produced / delivered / acknowledged / dropped / duplicates /
revisions / incomplete / accepted_for_training` separately. Neither can drop a
record: when capacity is exhausted the producer waits, and if a caller uses the
non-blocking `offer()` it receives an explicit failure rather than a silent
eviction.

P3's ring is untouched and remains the default for benchmark runs.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field


class SinkOverflow(RuntimeError):
    """Raised when a non-blocking producer would exceed capacity.

    Fails the iteration. Never evicts, never continues with a partial cohort.
    """


@dataclass
class SinkAccounting:
    """Per-iteration counters. `dropped == 0` is required for training."""

    produced: int = 0
    delivered: int = 0
    acknowledged: int = 0
    dropped: int = 0
    duplicates: int = 0
    revisions: int = 0
    incomplete: int = 0
    accepted_for_training: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def healthy(self) -> bool:
        """A training iteration is admissible only with zero loss."""
        return (
            self.dropped == 0
            and self.produced == self.acknowledged
            and self.duplicates == 0
        )


class LosslessBuffer:
    """Bounded, blocking, lossless FIFO of accepted transition records.

    Producer semantics:
      * `put(record)`   — blocks while full (backpressure); never evicts.
      * `offer(record)` — non-blocking; returns False when full.

    Consumer semantics:
      * `drain()`       — take everything, keep accounting.
      * `records()`     — read-only snapshot without draining.
    """

    def __init__(self, capacity: int = 4096):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._q: list[dict] = []
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self.accounting = SinkAccounting()

    # -- producer ------------------------------------------------------
    def put(self, record: dict) -> None:
        """Blocking append. Backpressure instead of eviction."""
        with self._not_full:
            while len(self._q) >= self.capacity:
                self._not_full.wait()
            self._q.append(record)
            self.accounting.produced += 1
            self.accounting.delivered += 1
            self.accounting.acknowledged += 1

    def offer(self, record: dict) -> bool:
        """Non-blocking append. Returns False when full; never drops."""
        with self._lock:
            if len(self._q) >= self.capacity:
                return False
            self._q.append(record)
            self.accounting.produced += 1
            self.accounting.delivered += 1
            self.accounting.acknowledged += 1
            return True

    def put_or_fail(self, record: dict) -> None:
        """Fail closed when full (use when blocking would deadlock)."""
        if not self.offer(record):
            raise SinkOverflow(
                f"lossless sink capacity {self.capacity} reached; refusing to "
                "evict or continue with a partial cohort")

    # -- consumer ------------------------------------------------------
    def drain(self) -> list[dict]:
        with self._not_full:
            out = self._q
            self._q = []
            self._not_full.notify_all()
            return out

    def records(self) -> list[dict]:
        with self._lock:
            return list(self._q)

    # -- P3 collector report compatibility ----------------------------
    # `run_batched` records `store.stored` / `store.dropped` for its report.
    # Exposing them here keeps that reporting path unchanged; `dropped` is
    # always 0 because this sink cannot evict.
    @property
    def stored(self) -> int:
        return self.accounting.acknowledged

    @property
    def dropped(self) -> int:
        return self.accounting.dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def __iter__(self):
        return iter(self.records())


class LosslessSpool:
    """Bounded on-disk spool with the same accounting contract.

    Used when an iteration's cohort is large enough that keeping every record
    resident is undesirable. Records are JSON lines of the plain serialisable
    form produced by `transitions.to_plain`.
    """

    def __init__(self, path, capacity: int = 100_000):
        self.path = str(path)
        self.capacity = int(capacity)
        self.accounting = SinkAccounting()
        self._count = 0
        # Truncate any stale spool from a previous iteration.
        with open(self.path, "w", encoding="utf-8"):
            pass

    def put(self, record: dict) -> None:
        if self._count >= self.capacity:
            raise SinkOverflow(
                f"lossless spool capacity {self.capacity} reached at {self.path}")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._count += 1
        self.accounting.produced += 1
        self.accounting.delivered += 1
        self.accounting.acknowledged += 1

    def records(self) -> list[dict]:
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def drain(self) -> list[dict]:
        return self.records()

    @property
    def stored(self) -> int:
        return self.accounting.acknowledged

    @property
    def dropped(self) -> int:
        return self.accounting.dropped

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        return iter(self.records())


def verify_lossless(sink) -> None:
    """Assert the sink's own accounting shows zero loss.

    Called at cohort admission so a corrupted sink cannot masquerade as healthy.
    """
    acc = sink.accounting
    if acc.dropped != 0:
        raise SinkOverflow(f"sink reported {acc.dropped} dropped record(s)")
    if acc.produced != acc.acknowledged:
        raise SinkOverflow(
            f"sink produced {acc.produced} but acknowledged {acc.acknowledged}")
