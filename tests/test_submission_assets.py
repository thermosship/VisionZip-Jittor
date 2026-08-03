import unittest

from scripts.build_submission_assets import rolling_mean


class SubmissionAssetTests(unittest.TestCase):
    def test_rolling_mean_uses_prefix_window(self):
        self.assertEqual(rolling_mean([1.0, 2.0, 3.0, 4.0], 2), [1.0, 1.5, 2.5, 3.5])

    def test_rolling_mean_window_larger_than_trace(self):
        actual = rolling_mean([2.0, 4.0, 8.0], 32)
        self.assertEqual(actual, [2.0, 3.0, 14.0 / 3.0])

    def test_rolling_mean_rejects_non_positive_window(self):
        with self.assertRaisesRegex(ValueError, "window must be positive"):
            rolling_mean([1.0], 0)


if __name__ == "__main__":
    unittest.main()
