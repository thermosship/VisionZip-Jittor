import importlib.util
import unittest

import numpy as np


@unittest.skipIf(
    importlib.util.find_spec("jittor") is None,
    "Jittor is only available in the AutoDL environment",
)
class JittorImportTests(unittest.TestCase):
    def test_core_import(self):
        import jittor as jt

        from visionzip_jittor.core import VisionZip
        from visionzip_jittor.config import VisionZipConfig

        jt.flags.use_cuda = 0
        module = VisionZip(
            VisionZipConfig(dominant_tokens=2, contextual_tokens=1)
        )
        self.assertEqual(module.config.dominant_tokens, 2)

    def test_native_core_executes_on_small_tensor(self):
        import jittor as jt

        from visionzip_jittor.config import VisionZipConfig
        from visionzip_jittor.core import visionzip_compress

        jt.flags.use_cuda = 0
        rng = np.random.default_rng(2026)
        hidden = jt.array(rng.standard_normal((2, 17, 8)).astype("float32"))
        logits = rng.standard_normal((2, 3, 17, 17)).astype("float32")
        logits -= logits.max(axis=-1, keepdims=True)
        attention = np.exp(logits)
        attention /= attention.sum(axis=-1, keepdims=True)
        attentions = jt.array(attention)
        metric = jt.array(rng.standard_normal((2, 17, 4)).astype("float32"))
        config = VisionZipConfig(dominant_tokens=4, contextual_tokens=3)

        output = visionzip_compress(hidden, attentions, metric, config)
        output["compressed_tokens"].sync()

        self.assertEqual(list(output["compressed_tokens"].shape), [2, 8, 8])
        self.assertEqual(list(output["selected_indices"].shape), [2, 5])
        self.assertEqual(list(output["contextual_tokens"].shape), [2, 3, 8])
        ordered = output["dominant_ordered_indices"].numpy()
        self.assertTrue(np.all(ordered[:, 1:] >= ordered[:, :-1]))
        self.assertTrue(np.isfinite(output["compressed_tokens"].numpy()).all())


if __name__ == "__main__":
    unittest.main()
