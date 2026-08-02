from __future__ import annotations

import importlib.util
import unittest


VISUAL_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "PIL")
)
if VISUAL_DEPS_AVAILABLE:
    import numpy as np

    from scripts.visualize_real_clip_tokens import (
        denormalize_pixel_values,
        patch_box,
    )
else:
    np = None


@unittest.skipUnless(
    VISUAL_DEPS_AVAILABLE,
    "NumPy and Pillow are required for visualization tests",
)
class RealClipVisualizationTests(unittest.TestCase):
    def test_denormalize_identity(self):
        values = np.zeros((3, 4, 4), dtype=np.float32)
        image = denormalize_pixel_values(
            values,
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        )
        self.assertEqual(image.size, (4, 4))
        self.assertEqual(image.getpixel((0, 0)), (128, 128, 128))

    def test_patch_box_for_24_by_24_grid(self):
        self.assertEqual(patch_box(1, (24, 24), 336), (0, 0, 14, 14))
        self.assertEqual(
            patch_box(576, (24, 24), 336),
            (322, 322, 336, 336),
        )


if __name__ == "__main__":
    unittest.main()
