import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipIf(
    importlib.util.find_spec("jittor") is None,
    "Jittor is only available in the AutoDL environment",
)
class Phase4JittorTests(unittest.TestCase):
    def setUp(self):
        import jittor as jt

        jt.flags.use_cuda = 0
        jt.set_global_seed(2026)

    def test_checkpoint_restores_projector_and_adam_exactly(self):
        import jittor as jt

        from visionzip_jittor.phase4_training import (
            load_phase4_checkpoint,
            parameter_sha256,
            save_phase4_checkpoint,
        )
        from visionzip_jittor.projector import MultimodalProjector
        from visionzip_jittor.projector_config import ProjectorConfig

        projector = MultimodalProjector(
            ProjectorConfig(
                projector_type="mlp2x_gelu",
                vision_hidden_size=4,
                language_hidden_size=6,
                vocab_size=16,
            )
        )
        optimizer = jt.optim.Adam(projector.parameters(), lr=1e-3)
        visual = jt.array(
            np.random.default_rng(2026).standard_normal((2, 3, 4)).astype("float32")
        )
        loss = (projector(visual) ** 2).mean()
        optimizer.step(loss)
        jt.sync_all()
        expected_hash = parameter_sha256(projector.parameters())
        expected_m = [value.numpy().copy() for value in optimizer.param_groups[0]["m"]]
        expected_v = [value.numpy().copy() for value in optimizer.param_groups[0]["values"]]
        expected_step = optimizer.n_step

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_phase4_checkpoint(path, projector, optimizer, {"global_step": 1})
            optimizer.step((projector(visual) ** 2).mean())
            metadata = load_phase4_checkpoint(path, projector, optimizer)

        self.assertEqual(metadata["global_step"], 1)
        self.assertEqual(parameter_sha256(projector.parameters()), expected_hash)
        self.assertEqual(optimizer.n_step, expected_step)
        for actual, expected in zip(optimizer.param_groups[0]["m"], expected_m):
            np.testing.assert_array_equal(actual.numpy(), expected)
        for actual, expected in zip(optimizer.param_groups[0]["values"], expected_v):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_tiny_teacher_forced_step_updates_only_projector(self):
        import jittor as jt

        from visionzip_jittor.gpt2 import masked_causal_language_loss
        from visionzip_jittor.multimodal import FrozenLanguageStub
        from visionzip_jittor.phase4_training import build_jittor_training_batch
        from visionzip_jittor.projector import MultimodalProjector
        from visionzip_jittor.projector_config import ProjectorConfig

        class Tokenizer:
            eos_token_id = 7

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [1, 2] if text == "prompt" else [3, 4]

        config = ProjectorConfig(
            projector_type="linear",
            vision_hidden_size=4,
            language_hidden_size=6,
            vocab_size=8,
        )
        projector = MultimodalProjector(config)
        language = FrozenLanguageStub(config)
        optimizer = jt.optim.Adam(projector.parameters(), lr=1e-3)
        visual = jt.array(
            np.random.default_rng(7).standard_normal((1, 3, 4)).astype("float32")
        )
        projector_before = [item.numpy().copy() for item in projector.parameters()]
        language_before = [item.numpy().copy() for item in language.parameters()]
        projected = projector(visual)
        batch = build_jittor_training_batch(
            Tokenizer(), language, projected, ["caption"], "prompt", 8
        )
        logits = language(batch["packed_embeddings"])
        loss = masked_causal_language_loss(logits, batch["labels"], batch["label_mask"])
        optimizer.step(loss)
        jt.sync_all()

        self.assertTrue(np.isfinite(loss.numpy()).all())
        self.assertTrue(
            any(
                not np.array_equal(before, after.numpy())
                for before, after in zip(projector_before, projector.parameters())
            )
        )
        self.assertTrue(
            all(
                np.array_equal(before, after.numpy())
                for before, after in zip(language_before, language.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
