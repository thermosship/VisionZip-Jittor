import json
import tempfile
import unittest
from pathlib import Path

from visionzip_jittor.gpt2_config import (
    GPT2Config,
    Phase3BConfig,
    load_gpt2_config,
    load_phase3b_config,
)


class GPT2ConfigTests(unittest.TestCase):
    def test_hugging_face_aliases(self):
        config = GPT2Config.from_dict(
            {
                "model_type": "gpt2",
                "vocab_size": 100,
                "max_position_embeddings": 64,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "bos_token_id": 99,
                "eos_token_id": 99,
                "unrelated_hf_key": True,
            }
        )
        self.assertEqual(config.n_positions, 64)
        self.assertEqual(config.n_embd, 32)
        self.assertEqual(config.n_layer, 2)
        self.assertEqual(config.n_head, 4)

    def test_hidden_size_must_divide_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            GPT2Config(n_embd=30, n_head=8).validate()

    def test_phase3b_target_count_is_runtime_checked(self):
        config = Phase3BConfig.from_dict(
            {
                "targets": [" one", " two", " three"],
                "iterations": 1,
            }
        )
        self.assertEqual(len(config.targets), 3)

    def test_unknown_phase3b_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            Phase3BConfig.from_dict({"unknown": 1})

    def test_json_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpt2_path = root / "gpt2.json"
            gpt2_path.write_text(
                json.dumps(GPT2Config().to_dict()),
                encoding="utf-8",
            )
            phase_path = root / "phase.json"
            phase_path.write_text(
                json.dumps(Phase3BConfig().to_dict()),
                encoding="utf-8",
            )
            self.assertEqual(load_gpt2_config(gpt2_path), GPT2Config())
            self.assertEqual(load_phase3b_config(phase_path), Phase3BConfig())


if __name__ == "__main__":
    unittest.main()
