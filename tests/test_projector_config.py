import json
import tempfile
import unittest
from pathlib import Path

from visionzip_jittor.projector_config import (
    ProjectorConfig,
    load_projector_config,
)


class ProjectorConfigTests(unittest.TestCase):
    def test_defaults_match_phase3_smoke_dimensions(self):
        config = ProjectorConfig()
        config.validate()
        self.assertEqual(config.vision_hidden_size, 1024)
        self.assertEqual(config.language_hidden_size, 4096)
        self.assertEqual(config.projector_type, "mlp2x_gelu")

    def test_linear_projector_is_supported(self):
        config = ProjectorConfig(projector_type="linear")
        config.validate()
        self.assertEqual(config.projector_type, "linear")

    def test_invalid_projector_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "projector_type"):
            ProjectorConfig(projector_type="transformer").validate()

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown projector"):
            ProjectorConfig.from_dict({"unexpected": True})

    def test_json_round_trip(self):
        expected = ProjectorConfig(language_hidden_size=256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projector.json"
            path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")
            actual = load_projector_config(path)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
