import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visionzip_jittor.phase4b_config import Phase4BConfig, load_phase4b_config
from visionzip_jittor.phase4b_data import (
    Phase4BPreparedSample,
    assign_exact_splits,
    file_sha256,
    load_prepared_dataset,
    normalize_caption,
    preflight_report,
    prepared_sample_from_row,
    source_image_bytes,
    source_row_rejection,
    source_sample_id,
    write_prepared_dataset_manifest,
)
from visionzip_jittor.phase4b_features import (
    build_feature_index,
    iter_feature_shards,
    load_feature_manifest,
    write_feature_manifest,
    write_feature_shard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase4b_commoncatalog_cc_by_8k.json"


class Phase4BConfigTests(unittest.TestCase):
    def test_pinned_commoncatalog_plan_is_self_consistent(self):
        config = load_phase4b_config(CONFIG_PATH)
        self.assertEqual(config.artifact_type, "phase4b_config_v1")
        self.assertEqual(
            config.dataset.revision,
            "80f50fe4a1ca937f37a11be3f8eee5199d776ff3",
        )
        self.assertEqual(config.dataset.source_row_count, 9621)
        self.assertEqual(config.dataset.source_size_bytes, 1263965106)
        self.assertEqual(config.dataset.target_sample_count, 8192)
        self.assertEqual(config.dataset.train_sample_count, 7168)
        self.assertEqual(config.dataset.validation_sample_count, 1024)
        self.assertEqual(config.training.effective_batch_size, 16)
        self.assertEqual(config.training.max_optimizer_steps, 1344)
        self.assertEqual(config.features.storage_dtype, "float32")

    def test_unknown_config_key_is_rejected(self):
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        payload["surprise"] = True
        with self.assertRaisesRegex(ValueError, "Unknown Phase 4B config keys"):
            Phase4BConfig.from_dict(payload)

    def test_preflight_has_explicit_headroom(self):
        config = load_phase4b_config(CONFIG_PATH)
        report = preflight_report(config, free_bytes=20 * 1024 ** 3)
        self.assertEqual(report["estimated_feature_bytes"], 2181038080)
        self.assertTrue(report["disk_preflight_passed"])
        self.assertGreater(report["estimated_required_bytes"], 8 * 1024 ** 3)

    def test_source_object_ids_must_be_lowercase_hex(self):
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        payload["dataset"]["source_shards"][0]["repository_oid"] = "Z" * 40
        with self.assertRaisesRegex(ValueError, "lowercase 40-character Git oid"):
            Phase4BConfig.from_dict(payload)


class Phase4BPreparationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_phase4b_config(CONFIG_PATH)
        self.good_row = {
            "jpg": {"bytes": b"jpeg bytes", "path": None},
            "blip2_caption": "  a   small bird on a branch  ",
            "status": "success",
            "licensename": "Attribution License",
            "licenseurl": "http://creativecommons.org/licenses/by/2.0/",
            "width": 512,
            "height": 512,
            "photoid": 123,
            "uid": "user@N00",
            "unickname": "Example Creator",
            "title": "Bird",
            "pageurl": "https://www.flickr.com/photos/example/123/",
            "sha256": hashlib.sha256(b"upstream source bytes").hexdigest(),
        }

    def sample(self, index):
        digest = hashlib.sha256(f"image-{index}".encode("utf-8")).hexdigest()
        source_digest = hashlib.sha256(f"source-{index}".encode("utf-8")).hexdigest()
        return Phase4BPreparedSample(
            sample_id=f"sample-{index}",
            split="train",
            image_path=f"images/{index}.jpg",
            image_sha256=digest,
            caption=f"caption number {index}",
            source_shard="source.parquet",
            source_row=index,
            source_photo_id=str(index),
            source_uid=f"user-{index}",
            source_image_sha256=source_digest,
            creator_name=f"creator {index}",
            title=f"title {index}",
            source_page_url=f"https://example.com/image/{index}",
            license_name="Attribution License",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            width=512,
            height=512,
        )

    def test_caption_filter_and_image_extraction(self):
        self.assertEqual(
            normalize_caption(self.good_row["blip2_caption"]),
            "a small bird on a branch",
        )
        self.assertIsNone(source_row_rejection(self.good_row, self.config))
        self.assertEqual(source_image_bytes(self.good_row["jpg"]), b"jpeg bytes")
        self.assertEqual(
            source_sample_id(self.good_row),
            "ccby-" + self.good_row["sha256"][:24],
        )
        embedded_hash = hashlib.sha256(b"jpeg bytes").hexdigest()
        prepared = prepared_sample_from_row(
            row=self.good_row,
            config=self.config,
            source_shard="source.parquet",
            source_row=0,
            image_path="images/example.jpg",
            image_sha256=embedded_hash,
        )
        prepared.validate()
        self.assertEqual(prepared.image_sha256, embedded_hash)
        self.assertEqual(prepared.source_image_sha256, self.good_row["sha256"])
        self.assertNotEqual(prepared.image_sha256, prepared.source_image_sha256)
        bad = dict(self.good_row, licenseurl="https://example.com/license")
        self.assertEqual(source_row_rejection(bad, self.config), "license")

        self.assertEqual(
            source_row_rejection(dict(self.good_row, unickname=""), self.config),
            "creator",
        )
        self.assertEqual(
            source_row_rejection(dict(self.good_row, pageurl=""), self.config),
            "source_page",
        )
        self.assertEqual(
            source_row_rejection(dict(self.good_row, licensename=""), self.config),
            "license_name",
        )
        self.assertEqual(
            source_row_rejection(dict(self.good_row, sha256="not-a-digest"), self.config),
            "source_sha256_format",
        )

    def test_exact_split_is_deterministic_and_preserves_order(self):
        samples = tuple(self.sample(index) for index in range(10))
        first = assign_exact_splits(samples, validation_sample_count=3, seed=2026)
        repeated = assign_exact_splits(samples, validation_sample_count=3, seed=2026)
        self.assertEqual(first, repeated)
        self.assertEqual([item.sample_id for item in first], [item.sample_id for item in samples])
        self.assertEqual(sum(item.split == "validation" for item in first), 3)
        self.assertEqual(sum(item.split == "train" for item in first), 7)

    def test_prepared_manifest_round_trip_and_hash_verification(self):
        samples = assign_exact_splits(
            tuple(self.sample(index) for index in range(4)),
            validation_sample_count=1,
            seed=2026,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample in samples:
                image_path = root / sample.image_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                content = f"image-{sample.source_row}".encode("utf-8")
                image_path.write_bytes(content)
                self.assertEqual(file_sha256(image_path), sample.image_sha256)
            written = write_prepared_dataset_manifest(
                root,
                self.config,
                samples,
                source_files=[{"path": "source.parquet", "size_bytes": 100, "sha256": "0" * 64}],
                rejection_counts={"duplicate": 2},
            )
            loaded_manifest, loaded_samples = load_prepared_dataset(
                root / "manifest.json",
                verify_images=True,
            )
            self.assertEqual(loaded_manifest, written)
            self.assertEqual(loaded_samples, samples)
            (root / "samples.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_prepared_dataset(root / "manifest.json")


class Phase4BFeatureStoreTests(unittest.TestCase):
    def arrays(self, start, count):
        return {
            "sample_ids": np.asarray([f"sample-{index}" for index in range(start, start + count)]),
            "compressed_tokens": np.arange(count * 3 * 4, dtype=np.float32).reshape(count, 3, 4),
            "selected_indices": np.tile(np.asarray([[0, 2]], dtype=np.int16), (count, 1)),
            "assignments": np.tile(np.asarray([[0, 1, 0]], dtype=np.int8), (count, 1)),
        }

    def test_feature_indices_require_integer_dtypes(self):
        arrays = self.arrays(0, 1)
        arrays["assignments"] = arrays["assignments"].astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "integer dtype"):
                write_feature_shard(Path(directory), 0, arrays)

    def test_shards_manifest_iteration_and_index(self):
        config = load_phase4b_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = [
                write_feature_shard(root, 0, self.arrays(0, 2)),
                write_feature_shard(root, 1, self.arrays(2, 1)),
            ]
            manifest = write_feature_manifest(
                output_dir=root,
                config_payload=config.to_dict(),
                dataset_samples_sha256="1" * 64,
                model_name_or_path=config.features.model_name_or_path,
                model_revision=config.features.model_revision,
                visionzip_config={"name": "fixture"},
                requested_layer_index=-2,
                resolved_layer_index=22,
                storage_dtype="float32",
                token_shape=(3, 4),
                shards=shards,
            )
            loaded = load_feature_manifest(root / "manifest.json", verify_shards=True)
            self.assertEqual(loaded, manifest)
            rows = list(iter_feature_shards(root / "manifest.json"))
            self.assertEqual([item[0].sample_count for item in rows], [2, 1])
            self.assertEqual(
                build_feature_index(root / "manifest.json"),
                {
                    "sample-0": ("features-00000.npz", 0),
                    "sample-1": ("features-00000.npz", 1),
                    "sample-2": ("features-00001.npz", 0),
                },
            )


if __name__ == "__main__":
    unittest.main()
