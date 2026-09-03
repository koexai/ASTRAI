import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import preprocess


class PreprocessingRunDirectoryTests(unittest.TestCase):
    def test_creates_timestamped_directory_from_config_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "preprocessed"
            started_at = datetime(2026, 9, 3, 18, 5, 7, tzinfo=timezone.utc)

            with patch.object(preprocess, "_DEFAULT_RUNS_DIR", base_dir):
                run_dir = preprocess._create_run_directory(
                    "configs/research config.yaml",
                    now=started_at,
                )

            self.assertEqual(
                run_dir,
                base_dir / "20260903_180507_research-config",
            )
            self.assertTrue(run_dir.is_dir())

    def test_rejects_an_existing_explicit_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "existing"
            destination.mkdir()

            with self.assertRaisesRegex(
                FileExistsError,
                "Preprocessing output directory already exists",
            ):
                preprocess._create_run_directory(
                    "configs/default_split.yaml",
                    out_dir=destination,
                )


class PreprocessingRunMetadataTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "preprocessing": {
                "pca_components": 3,
                "n_splits": 2,
                "random_seed": 42,
            }
        }

    @staticmethod
    def _write_code_archive(run_dir, folder):
        del folder
        (Path(run_dir) / "code.zip").write_bytes(b"code")

    def test_completed_run_preserves_config_code_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "source.yaml"
            config_contents = "# preserved comment\npreprocessing:\n  n_splits: 2\n"
            config_path.write_text(config_contents, encoding="utf-8")
            run_dir = root / "runs" / "explicit-run"

            output = StringIO()
            with (
                patch.object(preprocess, "_REPOSITORY_ROOT", root),
                patch.object(
                    preprocess,
                    "save_code",
                    side_effect=self._write_code_archive,
                ),
                patch.object(
                    preprocess,
                    "_generate_preprocessing_artefacts",
                ) as generate,
                redirect_stdout(output),
            ):
                result = preprocess.run_preprocessing(
                    self.cfg,
                    out_dir=run_dir,
                    config_path=config_path,
                )

            self.assertEqual(result, str(run_dir))
            self.assertIn(f"--prep {run_dir}", output.getvalue())
            self.assertEqual(
                (run_dir / "config.yaml").read_text(encoding="utf-8"),
                config_contents,
            )
            self.assertTrue((run_dir / "code.zip").is_file())
            generate.assert_called_once_with(self.cfg, run_dir)

            metadata = yaml.safe_load(
                (run_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["preprocessing_artefact_schema_version"],
                1,
            )
            self.assertEqual(metadata["run"]["status"], "completed")
            self.assertIsNotNone(metadata["run"]["completed_at_utc"])
            self.assertEqual(metadata["config"]["snapshot"], "config.yaml")
            self.assertEqual(
                metadata["preprocessing"],
                {
                    "random_seed": 42,
                    "n_splits": 2,
                    "folds": [1, 2],
                },
            )
            self.assertEqual(
                metadata["git"],
                {
                    "commit": None,
                    "branch": None,
                    "working_tree_dirty": None,
                },
            )

    def test_failed_run_records_the_error_without_reusing_the_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "failed-run"

            with (
                patch.object(preprocess, "_REPOSITORY_ROOT", root),
                patch.object(
                    preprocess,
                    "save_code",
                    side_effect=self._write_code_archive,
                ),
                patch.object(
                    preprocess,
                    "_generate_preprocessing_artefacts",
                    side_effect=RuntimeError("synthetic failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    preprocess.run_preprocessing(
                        self.cfg,
                        out_dir=run_dir,
                    )

            metadata = yaml.safe_load(
                (run_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["run"]["status"], "failed")
            self.assertEqual(metadata["run"]["error_type"], "RuntimeError")
            self.assertEqual(
                metadata["run"]["error_message"],
                "synthetic failure",
            )

            with self.assertRaises(FileExistsError):
                preprocess.run_preprocessing(
                    self.cfg,
                    out_dir=run_dir,
                )

    def test_missing_config_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "unused-run"

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Configuration file not found",
            ):
                preprocess.run_preprocessing(
                    self.cfg,
                    out_dir=run_dir,
                    config_path=root / "missing.yaml",
                )

            self.assertFalse(run_dir.exists())

    def test_records_git_revision_branch_and_dirty_state(self):
        with patch.object(
            preprocess,
            "_run_git_command",
            side_effect=["abc123", "feature/run-metadata", " M file.py"],
        ):
            metadata = preprocess._git_metadata(Path("/repository"))

        self.assertEqual(
            metadata,
            {
                "commit": "abc123",
                "branch": "feature/run-metadata",
                "working_tree_dirty": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
