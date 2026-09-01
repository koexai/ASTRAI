import tempfile
import unittest
import zipfile
from pathlib import Path

from utils.log_experiments import save_code


class SaveCodeTests(unittest.TestCase):
    def test_archives_python_sources_recursively_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            included = {
                "root_script.py",
                "models/model.py",
                "scripts/train.py",
                "tests/test_example.py",
                "utils/helper.py",
            }
            excluded = {
                ".hidden/ignored.py",
                "data/ignored.py",
                "experiments/previous/ignored.py",
                "preprocessed/ignored.py",
                "utils/__pycache__/ignored.py",
                "venv/ignored.py",
            }

            for relative_path in included | excluded:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("pass\n")

            exp_dir = root / "experiments" / "current"
            exp_dir.mkdir(parents=True)

            save_code(exp_dir, folder=root)

            with zipfile.ZipFile(exp_dir / "code.zip") as archive:
                self.assertEqual(set(archive.namelist()), included)

    def test_rejects_empty_source_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            ignored_file = root / "data" / "ignored.py"
            ignored_file.parent.mkdir(parents=True)
            ignored_file.write_text("pass\n")

            exp_dir = root / "experiments" / "current"
            exp_dir.mkdir(parents=True)

            with self.assertRaisesRegex(
                RuntimeError, "No Python source files found"
            ):
                save_code(exp_dir, folder=root)

            self.assertFalse((exp_dir / "code.zip").exists())


if __name__ == "__main__":
    unittest.main()
