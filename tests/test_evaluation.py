import unittest

from pydantic import ValidationError

from tools._evaluation import Tolerance, WorkloadResult, aggregate, normalize_outcome


class EvaluationContractTests(unittest.TestCase):
    def test_latency_only_pass_is_scorable(self):
        evaluation = aggregate(
            [
                WorkloadResult(axes={"n": 1}, outcome="PASSED", latency_ms=1.0),
                WorkloadResult(axes={"n": 2}, outcome="PASSED", latency_ms=4.0),
            ]
        )
        self.assertEqual(evaluation.status, "ALL_PASSED")
        self.assertEqual(evaluation.geomean_latency_ms, 2.0)
        self.assertIsNone(evaluation.geomean_speedup_factor)

    def test_same_run_speedup_is_optional_view(self):
        evaluation = aggregate(
            [
                WorkloadResult(
                    outcome="PASSED",
                    latency_ms=2.0,
                    reference_latency_ms=8.0,
                    speedup_factor=4.0,
                ),
                WorkloadResult(
                    outcome="PASSED",
                    latency_ms=1.0,
                    reference_latency_ms=9.0,
                    speedup_factor=9.0,
                ),
            ]
        )
        self.assertEqual(evaluation.geomean_speedup_factor, 6.0)

    def test_failure_blocks_geomean_and_is_histogrammed(self):
        evaluation = aggregate(
            [
                WorkloadResult(outcome="PASSED", latency_ms=1.0),
                WorkloadResult(outcome="TIMEOUT", diagnostic="timed out"),
            ]
        )
        self.assertEqual(evaluation.status, "FAILED")
        self.assertIsNone(evaluation.geomean_latency_ms)
        self.assertEqual(evaluation.failure_statuses, {"TIMEOUT": 1})

    def test_pass_requires_positive_finite_latency(self):
        for value in (None, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                WorkloadResult(outcome="PASSED", latency_ms=value)

    def test_normalizer_fields_are_atomic(self):
        with self.assertRaises(ValidationError):
            WorkloadResult(
                outcome="PASSED",
                latency_ms=1.0,
                reference_latency_ms=2.0,
            )

    def test_unknown_outcome_is_rejected_unless_normalized(self):
        with self.assertRaises(ValidationError):
            WorkloadResult(outcome="FRAMEWORK_SPECIAL", diagnostic="native")
        self.assertEqual(normalize_outcome("FRAMEWORK_SPECIAL"), "OTHER")

    def test_per_workload_tolerance_survives_aggregation(self):
        tolerance = Tolerance(max_atol=0.01, max_rtol=0.02, required_matched_ratio=0.99)
        evaluation = aggregate(
            [WorkloadResult(outcome="PASSED", latency_ms=1.0, tolerance=tolerance)]
        )
        self.assertEqual(
            evaluation.representative_workloads["small"].tolerance,
            tolerance,
        )

    def test_empty_result_is_failed_and_unscored(self):
        evaluation = aggregate([])
        self.assertEqual(evaluation.status, "FAILED")
        self.assertEqual(evaluation.workload_count, 0)
        self.assertIsNone(evaluation.geomean_latency_ms)


if __name__ == "__main__":
    unittest.main()

