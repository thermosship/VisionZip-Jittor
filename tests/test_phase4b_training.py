import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visionzip_jittor.phase4b_config import load_phase4b_config
from visionzip_jittor.phase4b_data import (
    Phase4BPreparedSample,
    write_prepared_dataset_manifest,
)
from visionzip_jittor.phase4b_features import (
    write_feature_manifest,
    write_feature_shard,
)
from visionzip_jittor.phase4b_training import (
    batch_indices_for_optimizer_step,
    caption_metrics,
    checkpoints_to_remove,
    corpus_bleu,
    deterministic_subset_indices,
    learning_rate_for_optimizer_step,
    load_phase4b_training_features,
    rouge_l_f1,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase4b_commoncatalog_cc_by_8k.json"


class Phase4BTrainingUtilityTests(unittest.TestCase):
    def test_optimizer_schedule_is_repeatable_and_crosses_epoch_exactly(self):
        first = batch_indices_for_optimizer_step(10, 2, 3, 1, 2026)
        repeated = batch_indices_for_optimizer_step(10, 2, 3, 1, 2026)
        next_step = batch_indices_for_optimizer_step(10, 2, 3, 2, 2026)
        self.assertEqual(first, repeated)
        self.assertEqual(tuple(len(item) for item in first), (2, 2, 2))
        self.assertEqual(len({item for batch in first for item in batch}), 6)
        self.assertNotEqual(first, next_step)
        self.assertTrue(all(0 <= item < 10 for batch in next_step for item in batch))

    def test_learning_rate_uses_linear_warmup_and_cosine_decay(self):
        self.assertAlmostEqual(
            learning_rate_for_optimizer_step(1, 1e-4, 2, 6),
            5e-5,
        )
        self.assertAlmostEqual(
            learning_rate_for_optimizer_step(2, 1e-4, 2, 6),
            1e-4,
        )
        self.assertEqual(
            learning_rate_for_optimizer_step(3, 1e-4, 2, 6),
            1e-4,
        )
        self.assertGreater(
            learning_rate_for_optimizer_step(6, 1e-4, 2, 6),
            0.0,
        )
        self.assertLess(
            learning_rate_for_optimizer_step(6, 1e-4, 2, 6),
            learning_rate_for_optimizer_step(5, 1e-4, 2, 6),
        )

    def test_subset_and_checkpoint_retention_are_deterministic(self):
        self.assertEqual(
            deterministic_subset_indices(20, 5, 2026),
            deterministic_subset_indices(20, 5, 2026),
        )
        paths = [Path(f"projector_step_{step:06d}.npz") for step in (1, 2, 3, 4)]
        self.assertEqual(
            checkpoints_to_remove(paths, keep_last=2),
            (paths[0], paths[1]),
        )
        self.assertEqual(
            checkpoints_to_remove(paths, keep_last=2, protected=(paths[0],)),
            (paths[1],),
        )

    def test_caption_metrics_have_fixed_single_reference_policy(self):
        references = ["A red car on the road", "Two cats sit together"]
        perfect = caption_metrics(references, references)
        self.assertAlmostEqual(perfect["bleu_1_single_synthetic_reference"], 1.0)
        self.assertAlmostEqual(perfect["bleu_4_single_synthetic_reference"], 1.0)
        self.assertAlmostEqual(perfect["rouge_l_single_synthetic_reference"], 1.0)
        self.assertEqual(
            perfect["reference_type"],
            "one BLIP-2 synthetic caption per held-out image",
        )
        self.assertEqual(rouge_l_f1("a b c", "a c"), 0.8)
        self.assertEqual(
            corpus_bleu(["a b c"], ["x y"], 1, add_one_smoothing=False),
            0.0,
        )
        self.assertGreater(
            corpus_bleu(["a b c"], ["x y"], 4, add_one_smoothing=True),
            0.0,
        )

    @staticmethod
    def sample(index, split):
        image_bytes = f"image-{index}".encode("utf-8")
        digest = hashlib.sha256(image_bytes).hexdigest()
        return Phase4BPreparedSample(
            sample_id=f"sample-{index}",
            split=split,
            image_path=f"images/sample-{index}.jpg",
            image_sha256=digest,
            caption=f"caption number {index}",
            source_shard="source.parquet",
            source_row=index,
            source_photo_id=str(index),
            source_uid=f"uid-{index}",
            source_image_sha256="a" * 64,
            creator_name="creator",
            title="title",
            source_page_url="https://example.com/page",
            license_name="CC BY 2.0",
            license_url="https://creativecommons.org/licenses/by/2.0/",
            width=512,
            height=512,
        )

    def test_feature_loader_preallocates_and_preserves_exact_sample_order(self):
        config = load_phase4b_config(CONFIG_PATH)
        samples = tuple(
            self.sample(index, "train" if index < 3 else "validation")
            for index in range(4)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_dir = root / "prepared"
            features_dir = root / "features"
            prepared_dir.mkdir()
            features_dir.mkdir()
            prepared = write_prepared_dataset_manifest(
                prepared_dir,
                config,
                samples,
                source_files=[
                    {"path": "source.parquet", "size_bytes": 1, "sha256": "b" * 64}
                ],
                rejection_counts={},
            )
            shards = []
            for shard_index, start in enumerate((0, 2)):
                ids = [sample.sample_id for sample in samples[start : start + 2]]
                arrays = {
                    "sample_ids": np.asarray(ids),
                    "compressed_tokens": np.full(
                        (2, 3, 4), start + 1, dtype=np.float32
                    ),
                    "selected_indices": np.tile(
                        np.asarray([[0, 2]], dtype=np.int16), (2, 1)
                    ),
                    "assignments": np.tile(
                        np.asarray([[0, 1, 0]], dtype=np.int8), (2, 1)
                    ),
                }
                shards.append(write_feature_shard(features_dir, shard_index, arrays))
            write_feature_manifest(
                output_dir=features_dir,
                config_payload=config.to_dict(),
                dataset_samples_sha256=prepared.samples_sha256,
                model_name_or_path=config.features.model_name_or_path,
                model_revision=config.features.model_revision,
                visionzip_config={"name": "fixture"},
                requested_layer_index=-2,
                resolved_layer_index=22,
                storage_dtype="float32",
                token_shape=(3, 4),
                shards=shards,
            )
            loaded = load_phase4b_training_features(
                prepared_dir / "manifest.json",
                features_dir / "manifest.json",
            )
            self.assertEqual(loaded.sample_ids, tuple(item.sample_id for item in samples))
            np.testing.assert_array_equal(loaded.train_indices, [0, 1, 2])
            np.testing.assert_array_equal(loaded.validation_indices, [3])
            np.testing.assert_array_equal(loaded.compressed_tokens[:2], 1.0)
            np.testing.assert_array_equal(loaded.compressed_tokens[2:], 3.0)

            with np.load(features_dir / "features-00001.npz", allow_pickle=False) as data:
                bad_arrays = {key: data[key] for key in data.files}
            bad_arrays["sample_ids"] = bad_arrays["sample_ids"][::-1]
            write_feature_shard(features_dir, 1, bad_arrays)
            # Rebuild the manifest so hashes are valid; the loader must still reject row order.
            bad_shards = [shards[0], write_feature_shard(features_dir, 1, bad_arrays)]
            write_feature_manifest(
                output_dir=features_dir,
                config_payload=config.to_dict(),
                dataset_samples_sha256=prepared.samples_sha256,
                model_name_or_path=config.features.model_name_or_path,
                model_revision=config.features.model_revision,
                visionzip_config={"name": "fixture"},
                requested_layer_index=-2,
                resolved_layer_index=22,
                storage_dtype="float32",
                token_shape=(3, 4),
                shards=bad_shards,
            )
            with self.assertRaisesRegex(ValueError, "sample order mismatch"):
                load_phase4b_training_features(
                    prepared_dir / "manifest.json",
                    features_dir / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
