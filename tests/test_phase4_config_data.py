import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visionzip_jittor.phase4_config import Phase4AConfig, load_phase4a_config
from visionzip_jittor.phase4_data import (
    PairedManifest,
    batch_indices_for_step,
    build_label_arrays,
    deterministic_split,
    load_paired_manifest,
    load_precomputed_visual_tokens,
    prepare_text_supervision,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "prompt":
            return [10, 11]
        return [20 + index for index, _ in enumerate(text.split())]


class Phase4ConfigDataTests(unittest.TestCase):
    def setUp(self):
        self.manifest_path = (
            Path(__file__).resolve().parents[1]
            / "manifests/phase4a_tiny_pairs.json"
        )

    def test_versioned_config_loads(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/phase4a_tiny_overfit.json"
        )
        config = load_phase4a_config(path)
        self.assertEqual(config.budget, 64)
        self.assertEqual(config.max_steps, 30)
        self.assertTrue(config.verify_resume)
        self.assertEqual(Phase4AConfig.from_dict(config.to_dict()), config)

    def test_manifest_and_split_are_deterministic(self):
        manifest = load_paired_manifest(self.manifest_path)
        train_a, validation_a = deterministic_split(
            manifest.samples, 1.0 / 3.0, 2026
        )
        train_b, validation_b = deterministic_split(
            tuple(reversed(manifest.samples)), 1.0 / 3.0, 2026
        )
        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(len(train_a), 2)
        self.assertEqual(len(validation_a), 1)
        self.assertEqual(
            {sample.sample_id for sample in train_a + validation_a},
            {"dense", "scene", "text"},
        )

    def test_manifest_rejects_duplicate_ids(self):
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        payload["samples"][1]["id"] = payload["samples"][0]["id"]
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            PairedManifest.from_dict(payload)

    def test_batch_schedule_is_repeatable_and_epoch_shuffled(self):
        first = [batch_indices_for_step(5, 2, 7, step) for step in range(3)]
        repeated = [batch_indices_for_step(5, 2, 7, step) for step in range(3)]
        second_epoch = [
            batch_indices_for_step(5, 2, 7, step) for step in range(3, 6)
        ]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second_epoch)
        self.assertEqual(sorted(item for batch in first for item in batch), list(range(5)))

    def test_target_only_label_masks(self):
        supervision = prepare_text_supervision(
            FakeTokenizer(),
            ["two words", "one"],
            "prompt",
            max_caption_tokens=4,
        )
        arrays = build_label_arrays(
            prompt_tokens=2,
            visual_tokens=3,
            target_ids=supervision["target_ids"],
            target_mask=supervision["target_mask"],
        )
        self.assertEqual(supervision["target_token_counts"], [3, 2])
        self.assertEqual(list(arrays["labels"].shape), [2, 8])
        np.testing.assert_array_equal(arrays["label_mask"][:, :5], 0.0)
        np.testing.assert_array_equal(
            arrays["label_mask"][:, 5:],
            supervision["target_mask"],
        )
        np.testing.assert_array_equal(
            arrays["attention_mask"][:, 5:],
            supervision["target_mask"],
        )

    def test_precomputed_tokens_follow_manifest_order(self):
        manifest = load_paired_manifest(self.manifest_path)
        metadata = {
            "artifact_type": "real_clip_reference_v1",
            "images": [
                {"name": "text.png"},
                {"name": "dense.png"},
                {"name": "scene.png"},
            ],
        }
        compressed = np.stack(
            [
                np.full((2, 4), 10, dtype=np.float32),
                np.full((2, 4), 20, dtype=np.float32),
                np.full((2, 4), 30, dtype=np.float32),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            np.savez_compressed(
                path,
                compressed_tokens=compressed,
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            selected, report = load_precomputed_visual_tokens(
                path,
                manifest.samples,
            )
        np.testing.assert_array_equal(selected[:, 0, 0], [20, 30, 10])
        self.assertEqual(report["source_rows"], [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
