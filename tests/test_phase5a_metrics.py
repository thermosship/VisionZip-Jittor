import json
from pathlib import Path
import unittest

import numpy as np

from visionzip_jittor.phase5a_metrics import (
    compare_generation_traces,
    evaluate_trace_acceptance,
    stable_softmax,
)


class Phase5AMetricTests(unittest.TestCase):
    def compare(
        self,
        uncached,
        cached,
        uncached_ids=(3,),
        cached_ids=(3,),
        max_total_variation_distance=5e-5,
    ):
        return compare_generation_traces(
            uncached_ids,
            [np.asarray(uncached, dtype=np.float32)],
            cached_ids,
            [np.asarray(cached, dtype=np.float32)],
            raw_atol=1e-5,
            raw_rtol=1e-5,
            centered_atol=1e-5,
            centered_rtol=1e-5,
            probability_atol=1e-5,
            probability_rtol=1e-5,
            max_total_variation_distance=max_total_variation_distance,
        )

    def acceptance(self, comparison):
        return evaluate_trace_acceptance(
            comparison,
            require_exact_token_ids=True,
            require_total_variation_bound=True,
        )

    def test_stable_softmax_is_shift_invariant(self):
        logits = np.asarray([[1000.0, 1001.0, 999.0]], dtype=np.float32)
        shifted = logits + np.float32(128.0)
        np.testing.assert_allclose(
            stable_softmax(logits),
            stable_softmax(shifted),
            atol=0.0,
            rtol=0.0,
        )

    def test_exact_trace_passes_every_report(self):
        comparison = self.compare([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]])
        self.assertTrue(comparison["token_ids_exact"])
        self.assertTrue(comparison["raw_logits"]["allclose"])
        self.assertTrue(comparison["centered_logits"]["allclose"])
        self.assertTrue(comparison["softmax_probabilities"]["allclose"])
        self.assertTrue(comparison["probability_distribution"]["within_bound"])
        self.assertTrue(self.acceptance(comparison)["passed"])

    def test_raw_failure_remains_visible_but_does_not_control_acceptance(self):
        uncached = np.asarray([[10.0, 9.0, 8.0]], dtype=np.float32)
        cached = uncached + np.float32(5e-4)
        comparison = self.compare(uncached, cached)
        self.assertFalse(comparison["raw_logits"]["allclose"])
        self.assertEqual(comparison["raw_logits"]["failed_steps"], [0])
        self.assertEqual(comparison["raw_logits"]["role"], "diagnostic_only")
        self.assertTrue(comparison["centered_logits"]["allclose"])
        self.assertTrue(comparison["softmax_probabilities"]["allclose"])
        self.assertTrue(comparison["probability_distribution"]["within_bound"])
        acceptance = self.acceptance(comparison)
        self.assertTrue(acceptance["passed"])
        self.assertTrue(acceptance["raw_logits_are_diagnostic_only"])

    def test_coordinatewise_probability_failure_can_pass_tv_gate(self):
        comparison = self.compare(
            [[0.0, 0.0]],
            [[8e-5, 0.0]],
        )
        self.assertFalse(comparison["softmax_probabilities"]["allclose"])
        self.assertEqual(
            comparison["softmax_probabilities"]["role"],
            "diagnostic_only",
        )
        distribution = comparison["probability_distribution"]
        self.assertLess(distribution["max_total_variation_distance"], 5e-5)
        self.assertTrue(distribution["within_bound"])
        acceptance = self.acceptance(comparison)
        self.assertTrue(acceptance["passed"])
        self.assertTrue(
            acceptance["coordinatewise_probabilities_are_diagnostic_only"]
        )

    def test_centered_diagnostic_can_fail_while_distribution_gate_passes(self):
        comparison = self.compare(
            [[20.0, 0.0, -1.0]],
            [[20.0, 0.001, -1.0]],
        )
        self.assertFalse(comparison["raw_logits"]["allclose"])
        self.assertFalse(comparison["centered_logits"]["allclose"])
        self.assertTrue(comparison["probability_distribution"]["within_bound"])
        self.assertTrue(self.acceptance(comparison)["passed"])

    def test_total_variation_report_matches_definition(self):
        comparison = self.compare(
            [[0.0, 0.0]],
            [[8e-5, 0.0]],
        )
        left = stable_softmax(np.asarray([[0.0, 0.0]], dtype=np.float32))
        right = stable_softmax(
            np.asarray([[8e-5, 0.0]], dtype=np.float32)
        )
        expected = 0.5 * float(np.sum(np.abs(left - right)))
        self.assertAlmostEqual(
            comparison["probability_distribution"][
                "max_total_variation_distance"
            ],
            expected,
            places=15,
        )

    def test_distribution_bound_failure_rejects_trace(self):
        comparison = self.compare([[10.0, 0.0]], [[0.0, 10.0]])
        self.assertFalse(comparison["probability_distribution"]["within_bound"])
        acceptance = self.acceptance(comparison)
        self.assertFalse(acceptance["passed"])
        self.assertFalse(
            acceptance["probability_distribution_requirement_satisfied"]
        )

    def test_token_mismatch_rejects_trace_even_when_logits_match(self):
        comparison = self.compare(
            [[1.0, 2.0]],
            [[1.0, 2.0]],
            uncached_ids=(1,),
            cached_ids=(0,),
        )
        acceptance = self.acceptance(comparison)
        self.assertFalse(acceptance["passed"])
        self.assertFalse(acceptance["token_id_requirement_satisfied"])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            self.compare([[1.0, 2.0]], [[1.0, 2.0, 3.0]])

    def test_invalid_total_variation_bound_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be finite and in"):
            self.compare(
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                max_total_variation_distance=1.1,
            )

    def test_checked_in_config_declares_acceptance_v3(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "configs/phase5a_kv_cache.json").read_text(
                encoding="utf-8-sig"
            )
        )
        self.assertEqual(
            payload["artifact_type"],
            "phase5a_kv_cache_benchmark_config_v3",
        )
        self.assertTrue(payload["require_exact_token_ids"])
        self.assertTrue(payload["require_total_variation_bound"])
        self.assertEqual(payload["raw_logit_diagnostic_atol"], 1e-5)
        self.assertEqual(payload["probability_diagnostic_atol"], 1e-5)
        self.assertEqual(payload["max_total_variation_distance"], 5e-5)


if __name__ == "__main__":
    unittest.main()
