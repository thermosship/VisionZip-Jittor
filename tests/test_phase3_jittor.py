import importlib.util
import unittest

import numpy as np


@unittest.skipIf(
    importlib.util.find_spec("jittor") is None,
    "Jittor is only available in the AutoDL environment",
)
class Phase3JittorTests(unittest.TestCase):
    def setUp(self):
        import jittor as jt

        jt.flags.use_cuda = 0
        jt.set_global_seed(2026)

    def test_projector_shapes_for_both_variants(self):
        import jittor as jt

        from visionzip_jittor.projector import MultimodalProjector
        from visionzip_jittor.projector_config import ProjectorConfig

        visual = jt.array(
            np.random.default_rng(2026)
            .standard_normal((2, 5, 8))
            .astype("float32")
        )
        for projector_type in ("linear", "mlp2x_gelu"):
            projector = MultimodalProjector(
                ProjectorConfig(
                    projector_type=projector_type,
                    vision_hidden_size=8,
                    language_hidden_size=12,
                    vocab_size=16,
                )
            )
            output = projector(visual)
            output.sync()
            self.assertEqual(list(output.shape), [2, 5, 12])

    def test_frozen_stub_backpropagates_only_into_projector(self):
        import jittor as jt

        from visionzip_jittor.multimodal import (
            FrozenLanguageStub,
            ProjectorFrozenLanguageBridge,
        )
        from visionzip_jittor.projector import MultimodalProjector
        from visionzip_jittor.projector_config import ProjectorConfig

        config = ProjectorConfig(
            projector_type="mlp2x_gelu",
            vision_hidden_size=8,
            language_hidden_size=12,
            vocab_size=16,
            prefix_tokens=2,
            suffix_tokens=3,
            learning_rate=1e-3,
        )
        projector = MultimodalProjector(config)
        language = FrozenLanguageStub(config)
        bridge = ProjectorFrozenLanguageBridge(projector, language)
        optimizer = jt.optim.Adam(
            list(projector.parameters()),
            lr=config.learning_rate,
        )

        rng = np.random.default_rng(2026)
        visual = jt.array(rng.standard_normal((2, 5, 8)).astype("float32"))
        prefix = jt.array(rng.integers(0, 16, (2, 2), dtype=np.int32))
        suffix = jt.array(rng.integers(0, 16, (2, 3), dtype=np.int32))
        projector_before = [
            parameter.numpy().copy() for parameter in projector.parameters()
        ]
        frozen_before = [
            parameter.numpy().copy() for parameter in language.parameters()
        ]

        outputs = bridge(visual, prefix, suffix)
        loss = (outputs["logits"] * outputs["logits"]).mean()
        optimizer.zero_grad()
        optimizer.backward(loss)
        gradients = [
            parameter.opt_grad(optimizer).numpy()
            for parameter in projector.parameters()
        ]
        optimizer.step()
        jt.sync_all()

        self.assertTrue(language.all_parameters_stop_grad())
        self.assertEqual(list(outputs["packed_embeddings"].shape), [2, 10, 12])
        self.assertTrue(all(np.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(
            sum(float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients),
            0.0,
        )
        self.assertTrue(
            any(
                not np.array_equal(before, after.numpy())
                for before, after in zip(projector_before, projector.parameters())
            )
        )
        self.assertTrue(
            all(
                np.array_equal(before, after.numpy())
                for before, after in zip(frozen_before, language.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
