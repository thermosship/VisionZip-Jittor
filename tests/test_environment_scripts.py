from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ACTIVATE_SCRIPT = ROOT / "environment" / "activate_jittor.sh"


class EnvironmentScriptTests(unittest.TestCase):
    def test_activation_script_is_checkout_relative(self):
        text = ACTIVATE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("BASH_SOURCE[0]", text)
        self.assertIn('cd "${PROJECT_ROOT}"', text)
        self.assertNotIn("cd /root/autodl-tmp/VisionZip-Jittor", text)

    def test_activation_script_supports_clean_environment_overrides(self):
        text = ACTIVATE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("VISIONZIP_JITTOR_ENV", text)
        self.assertIn("VISIONZIP_CACHE_ROOT", text)
        self.assertIn("/root/autodl-tmp/envs/visionzip-jittor", text)
        self.assertIn("/root/autodl-tmp/cache", text)


if __name__ == "__main__":
    unittest.main()
