import tempfile
import unittest
from pathlib import Path

from astrai.utils import checkpoints
from astrai.utils.configuration import load_config


class ConfigurationLoadingTests(unittest.TestCase):
    def test_loads_yaml_from_a_path_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "training:\n  epochs: 3\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config, {"training": {"epochs": 3}})

    def test_checkpoint_module_preserves_load_config_import(self):
        self.assertIs(checkpoints.load_config, load_config)


if __name__ == "__main__":
    unittest.main()
