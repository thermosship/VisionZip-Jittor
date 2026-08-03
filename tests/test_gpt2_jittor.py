import unittest

import numpy as np

try:
    import jittor as jt
except (ImportError, OSError):
    jt = None


@unittest.skipIf(jt is None, "Jittor is not installed in this environment")
class NativeGPT2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jt.flags.use_cuda = 0
        from visionzip_jittor.gpt2 import NativeGPT2LMHeadModel
        from visionzip_jittor.gpt2_config import GPT2Config

        cls.config = GPT2Config(
            vocab_size=32,
            n_positions=64,
            n_embd=16,
            n_layer=2,
            n_head=4,
            bos_token_id=31,
            eos_token_id=31,
        )
        cls.model_class = NativeGPT2LMHeadModel

    def test_forward_shape_and_tied_head(self):
        model = self.model_class(self.config)
        ids = jt.array([[1, 2, 3], [4, 5, 6]])
        logits = model(input_ids=ids)
        self.assertEqual(list(logits.shape), [2, 3, 32])
        self.assertNotIn("lm_head.weight", model.weight_targets())

    def test_transformer_blocks_are_registered_as_parameters(self):
        model = self.model_class(self.config)
        self.assertEqual(len(model.blocks()), self.config.n_layer)
        self.assertEqual(len(model.weight_targets()), 4 + 12 * self.config.n_layer)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            8128,
        )

    def test_freeze_marks_every_language_parameter(self):
        model = self.model_class(self.config)
        model.freeze_parameters()
        self.assertTrue(model.all_parameters_stop_grad())

    def test_inputs_embeds_path(self):
        model = self.model_class(self.config)
        ids = jt.array([[1, 2, 3]])
        embeddings = model.embed_tokens(ids)
        logits = model(inputs_embeds=embeddings)
        self.assertEqual(list(logits.shape), [1, 3, 32])

    def test_masked_language_loss_is_finite(self):
        from visionzip_jittor.gpt2 import masked_causal_language_loss

        model = self.model_class(self.config)
        ids = jt.array([[1, 2, 3, 4]])
        logits = model(input_ids=ids)
        mask = jt.array([[0.0, 1.0, 1.0, 1.0]])
        loss = masked_causal_language_loss(logits, ids, mask)
        self.assertTrue(float(loss.numpy().item()) > 0.0)

    def test_cached_decode_matches_full_recompute(self):
        model = self.model_class(self.config)
        prefix = jt.array([[1, 2, 3]])
        cached = model.forward_with_cache(input_ids=prefix)
        self.assertEqual(len(cached["past_key_values"]), self.config.n_layer)
        for key, value in cached["past_key_values"]:
            self.assertEqual(list(key.shape), [1, 4, 3, 4])
            self.assertEqual(list(value.shape), [1, 4, 3, 4])

        next_id = jt.array([[4]])
        decoded = model.forward_with_cache(
            input_ids=next_id,
            past_key_values=cached["past_key_values"],
        )
        full = model(input_ids=jt.array([[1, 2, 3, 4]]))
        np.testing.assert_allclose(
            decoded["logits"].numpy()[:, -1, :],
            full.numpy()[:, -1, :],
            atol=1e-5,
            rtol=1e-5,
        )
        for key, value in decoded["past_key_values"]:
            self.assertEqual(list(key.shape), [1, 4, 4, 4])
            self.assertEqual(list(value.shape), [1, 4, 4, 4])

    def test_cached_greedy_matches_uncached_greedy(self):
        from visionzip_jittor.gpt2 import (
            greedy_generate_from_embeddings,
            greedy_generate_from_embeddings_cached,
        )

        model = self.model_class(self.config)
        initial = model.embed_tokens(jt.array([[1, 2, 3]]))
        expected = greedy_generate_from_embeddings(
            model,
            initial,
            max_new_tokens=5,
        )
        actual = greedy_generate_from_embeddings_cached(
            model,
            initial,
            max_new_tokens=5,
        )
        self.assertEqual(actual, expected)

    def test_cache_layer_count_is_validated(self):
        model = self.model_class(self.config)
        with self.assertRaisesRegex(ValueError, "layer count"):
            model.forward_with_cache(
                input_ids=jt.array([[1]]),
                past_key_values=[],
            )


if __name__ == "__main__":
    unittest.main()
