import contextlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from astrai.cli import dispatcher
from astrai import paths
from astrai.paths import default_config_path, source_checkout_root


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_runtime_requirements_match_project_dependencies(self):
        project = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        requirements = {
            line.strip()
            for line in (REPOSITORY_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(requirements, set(project["project"]["dependencies"]))

    def test_default_config_is_an_installed_package_resource(self):
        path = default_config_path("default_split.yaml")
        self.assertTrue(path.is_file())
        self.assertIn("data:", path.read_text(encoding="utf-8"))

    def test_packaged_configs_match_source_checkout_configs(self):
        for filename in ("4par.yaml", "default.yaml", "default_split.yaml"):
            self.assertEqual(
                default_config_path(filename).read_bytes(),
                (REPOSITORY_ROOT / "configs" / filename).read_bytes(),
            )

    def test_source_checkout_is_derived_from_package_location(self):
        self.assertEqual(source_checkout_root(), REPOSITORY_ROOT)

    def test_installed_package_does_not_claim_the_current_git_checkout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            installed_package = Path(temp_dir) / "site-packages" / "astrai"
            installed_package.mkdir(parents=True)
            with patch.object(paths, "PACKAGE_ROOT", installed_package):
                self.assertIsNone(paths.source_checkout_root())
                self.assertEqual(paths.source_snapshot_root(), installed_package)

    def test_top_level_help_does_not_import_command_dependencies(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = dispatcher.main(["--help"])
        self.assertEqual(result, 0)
        self.assertIn("infer-real", output.getvalue())

    def test_unknown_command_has_usage_exit_code(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            result = dispatcher.main(["does-not-exist"])
        self.assertEqual(result, 2)
        self.assertIn("unknown command", output.getvalue())


if __name__ == "__main__":
    unittest.main()
