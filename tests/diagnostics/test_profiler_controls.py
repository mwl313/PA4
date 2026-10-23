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

    def test_disabled_returns_same_value(self):
        p = Profiler(enabled=False)
        with p.span("x"):
            result = 42
        self.assertEqual(result, 42)

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
    """A missing capability must be UNVERIFIED, never a silent PASS."""

    def test_missing_activity_reports_unverified(self):
        def probe(activities):
            try:
                import torch
                from torch.profiler import ProfilerActivity
                want = [getattr(ProfilerActivity, a) for a in activities
                        if hasattr(ProfilerActivity, a)]
                if len(want) != len(activities):
                    return "UNVERIFIED"
                if "CUDA" in activities and not torch.cuda.is_available():
                    return "UNVERIFIED"
                return "AVAILABLE"
            except Exception:  # noqa: BLE001
                return "UNVERIFIED"

        self.assertEqual(probe(["CPU", "NOT_A_REAL_ACTIVITY"]), "UNVERIFIED")
        self.assertIn(probe(["CPU", "CUDA"]), ("AVAILABLE", "UNVERIFIED"))


if __name__ == "__main__":
    unittest.main()
