from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from reference.clip_features import (
        contextual_target_original_indices,
        infer_patch_grid,
        key_projection_to_metric,
        merge_original_indices,
        real_reference_filename,
        resolve_layer_index,
    )
else:
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this environment")
class ClipFeatureHelperTests(unittest.TestCase):
    def test_negative_layer_index(self):
        self.assertEqual(resolve_layer_index(24, -2), 22)
        self.assertEqual(resolve_layer_index(24, 0), 0)
        with self.assertRaises(IndexError):
            resolve_layer_index(24, -25)

    def test_key_projection_matches_official_head_mean(self):
        projection = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12)
        actual = key_projection_to_metric(projection, num_heads=3)
        expected = (
            projection.reshape(2, 5, 3, 4)
            .permute(0, 2, 1, 3)
            .mean(dim=1)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (2, 5, 4))

    def test_original_index_mapping(self):
        remaining = torch.tensor([[2, 4, 5, 8, 9]])
        targets = torch.tensor([[0, 3]])
        merges = torch.tensor([[1, 2, 4]])
        self.assertTrue(
            torch.equal(
                contextual_target_original_indices(remaining, targets),
                torch.tensor([[2, 8]]),
            )
        )
        self.assertTrue(
            torch.equal(
                merge_original_indices(remaining, merges),
                torch.tensor([[4, 5, 9]]),
            )
        )

    def test_patch_grid_and_filename(self):
        self.assertEqual(infer_patch_grid(577), (24, 24))
        self.assertEqual(
            real_reference_filename("clip 64/code exact", "float32"),
            "reference_clip_64_code_exact_float32_real_clip.npz",
        )
        with self.assertRaises(ValueError):
            infer_patch_grid(578)


if __name__ == "__main__":
    unittest.main()
