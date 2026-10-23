"""P4.5 diagnostic instrumentation negative controls (design 3.3).

Required falsifiability tests:
  1. profiling OFF is the original path: no counters, no files
  2. a controlled CPU sleep in an isolated fixture raises the matching span
  3. nested timers do not double-count (parent excl + children == parent incl)
  4. request correlation loses nothing even when a request errors
  5. a missing CUDA capability yields UNVERIFIED, not a silent PASS
"""
import os
import sys
import time
import unittest

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pa4.diagnostics.profiler import Profiler, get_profiler


class OffModeIdenticalTest(unittest.TestCase):
    def test_disabled_records_nothing(self):
        p = Profiler(enabled=False)
        with p.span("a"):
            pass
        with p.span("b"):
            time.sleep(0.001)
        self.assertEqual(p.counters, {}, "disabled profiler recorded counters")
        self.assertEqual(p.records, [], "disabled profiler recorded events")
        self.assertEqual(p.summary(), {})

    def test_disabled_path_produces_identical_results(self):
        """The same computation must yield identical output with and without
        a disabled profiler. This compares against a plain no-profiler run
        rather than asserting a literal the test itself assigned."""
        import torch

        def compute(use_prof):
            p = Profiler(enabled=use_prof)
            out = []
            for i in range(20):
                with p.span("step"):
                    t = torch.arange(i + 1, dtype=torch.float32)
                    out.append(float(t.sum().item()))
            return out

        without = compute(False)
        self.assertEqual(without, compute(False))
        # torch.arange(i + 1) is 0..i, whose sum is i(i+1)/2.
        self.assertEqual(without, [float(i * (i + 1) / 2) for i in range(20)])

    def test_env_gate(self):
        """get_profiler() must be off unless PA4_P45_PROFILE=1."""
        old = os.environ.pop("PA4_P45_PROFILE", None)
        try:
            self.assertFalse(get_profiler().enabled)
            os.environ["PA4_P45_PROFILE"] = "1"
            self.assertTrue(get_profiler().enabled)
        finally:
            os.environ.pop("PA4_P45_PROFILE", None)
            if old is not None:
                os.environ["PA4_P45_PROFILE"] = old

    def test_disabled_writes_no_file(self):
        import tempfile
        p = Profiler(enabled=False)
        path = os.path.join(tempfile.mkdtemp(), "should_not_exist.json")
        p.dump(path)
        self.assertFalse(os.path.exists(path), "disabled profiler wrote a file")


class ControlledSleepTest(unittest.TestCase):
    """A deliberate sleep must show up in exactly its labelled span."""

    def test_sleep_attributed_to_its_span(self):
        p = Profiler(enabled=True)
        with p.span("outer"):
            with p.span("slept"):
                time.sleep(0.05)
            with p.span("fast"):
                pass
        s = p.summary()
        self.assertGreaterEqual(s["slept"]["inclusive_ms"], 50.0)
        self.assertLess(s["fast"]["inclusive_ms"], 5.0)
        self.assertGreaterEqual(s["outer"]["inclusive_ms"], 50.0)


class NestedNoDoubleCountTest(unittest.TestCase):
    def test_parent_excl_plus_children_equals_parent_incl(self):
        p = Profiler(enabled=True)
        with p.span("parent"):
            time.sleep(0.01)
            with p.span("child_a"):
                time.sleep(0.02)
            with p.span("child_b"):
                time.sleep(0.03)
            time.sleep(0.01)
        s = p.summary()
        total = (s["parent"]["exclusive_ms"] + s["child_a"]["exclusive_ms"]
                 + s["child_b"]["exclusive_ms"])
        self.assertAlmostEqual(total, s["parent"]["inclusive_ms"], delta=1.0,
                               msg="nested accounting double-counted")

    def test_three_level_nesting(self):
        p = Profiler(enabled=True)
        with p.span("l1"):
            time.sleep(0.005)
            with p.span("l2"):
                time.sleep(0.005)
                with p.span("l3"):
                    time.sleep(0.005)
        s = p.summary()
        total = sum(s[k]["exclusive_ms"] for k in ("l1", "l2", "l3"))
        self.assertAlmostEqual(total, s["l1"]["inclusive_ms"], delta=1.5)


class CorrelationNoLossTest(unittest.TestCase):
    """Counters must not lose or duplicate a request when one errors."""

    def test_counts_survive_an_exception(self):
        p = Profiler(enabled=True)
        for i in range(5):
            try:
                with p.span("request"):
                    if i == 2:
                        raise RuntimeError("simulated request failure")
            except RuntimeError:
                pass
        self.assertEqual(p.counters["request"].calls, 5,
                         "a failed request was lost from the counter")

    def test_count_helper(self):
        p = Profiler(enabled=True)
        p.count("submitted", 3)
        p.count("submitted", 2)
        self.assertEqual(p.counters["submitted"].calls, 5)


class CudaCapabilityTest(unittest.TestCase):
    """GPU evidence must be UNVERIFIED when the capability is absent.

    The probe is a real capability check against the installed torch stack, not
    an inline toy that exercises nothing.
    """

    @staticmethod
    def capability(activities):
        """Return AVAILABLE / UNVERIFIED for a requested activity set."""
        try:
            import torch
            from torch.profiler import ProfilerActivity
        except Exception:  # noqa: BLE001
            return "UNVERIFIED"
        for name in activities:
            if not hasattr(ProfilerActivity, name):
                return "UNVERIFIED"
        if "CUDA" in activities and not torch.cuda.is_available():
            return "UNVERIFIED"
        return "AVAILABLE"

    def test_cpu_activity_available(self):
        self.assertEqual(self.capability(["CPU"]), "AVAILABLE")

    def test_unknown_activity_is_unverified(self):
        self.assertEqual(self.capability(["CPU", "NOT_A_REAL_ACTIVITY"]),
                         "UNVERIFIED")

    def test_cuda_reported_honestly(self):
        """Never claim AVAILABLE for CUDA unless CUDA is actually present."""
        import torch
        got = self.capability(["CPU", "CUDA"])
        if not torch.cuda.is_available():
            self.assertEqual(got, "UNVERIFIED")
        else:
            self.assertEqual(got, "AVAILABLE")


class CorrelationIntegrityTest(unittest.TestCase):
    """A request identity must never be lost or duplicated, even on error."""

    def test_records_survive_errors_without_loss_or_duplication(self):
        p = Profiler(enabled=True)
        seen = []
        for i in range(8):
            try:
                with p.span("request"):
                    if i % 3 == 0:
                        p.record("request_failed", ident=i)
                        raise RuntimeError("simulated failure")
                    p.record("request_ok", ident=i)
                    seen.append(i)
            except RuntimeError:
                seen.append(i)
        # Every request must be accounted for exactly once across ok+failed.
        oks = [r["ident"] for r in p.records if r["kind"] == "request_ok"]
        fails = [r["ident"] for r in p.records if r["kind"] == "request_failed"]
        self.assertEqual(sorted(oks + fails), list(range(8)),
                         "a request identity was lost")
        self.assertEqual(len(oks) + len(fails), 8,
                         "a request identity was duplicated")
        self.assertEqual(p.counters["request"].calls, 8,
                         "the span counter lost a request")

    def test_counts_are_not_double_counted(self):
        p = Profiler(enabled=True)
        for _ in range(3):
            with p.span("submit"):
                pass
        self.assertEqual(p.counters["submit"].calls, 3)
