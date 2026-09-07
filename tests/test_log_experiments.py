import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from astrai.utils.log_experiments import (
    ExperimentRun,
    create_experiment_dir,
    create_pipeline_run_id,
    save_code,
)


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


class ExperimentDirectoryTests(unittest.TestCase):
    def test_repeated_timestamp_creates_distinct_directories(self):
        instant = datetime(2026, 9, 6, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            first = create_experiment_dir(temp_dir, now=instant)
            second = create_experiment_dir(temp_dir, now=instant)

        self.assertNotEqual(first, second)
        self.assertTrue(second.endswith("_01"))

    def test_pipeline_identifiers_remain_unique_at_the_same_instant(self):
        instant = datetime(2026, 9, 6, 10, 30, tzinfo=timezone.utc)

        first = create_pipeline_run_id(now=instant)
        second = create_pipeline_run_id(now=instant)

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("pipeline_20260906_103000_000000_"))


class ExperimentRunTests(unittest.TestCase):
    @staticmethod
    def _source_root(root):
        source_root = root / "repository"
        source_root.mkdir()
        (source_root / "train.py").write_text("pass\n", encoding="utf-8")
        return source_root

    @staticmethod
    def _metadata(run_dir):
        return yaml.safe_load(
            (run_dir / "metadata.yaml").read_text(encoding="utf-8")
        )

    def test_completed_run_records_inputs_results_and_artefacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = self._source_root(root)
            prep_dir = root / "preprocessing"
            prep_dir.mkdir()
            (prep_dir / "metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "preprocessing_artefact_schema_version": 3,
                        "run": {"status": "completed"},
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "experiment"

            run = ExperimentRun.start(
                stage="characterizer",
                config={
                    "data": {
                        "n_params": 2,
                        "param_names": ["Mass", "Energy"],
                    }
                },
                exp_dir=run_dir,
                preprocessing_dir=prep_dir,
                folds=[1],
                base_seed=42,
                device="cpu",
                pipeline_run_id="pipeline-123",
                repository_root=source_root,
            )
            checkpoint = run_dir / "model.pth"
            checkpoint.write_bytes(b"weights")
            run.record_fold(
                1,
                {"aggregate": {"R2": np.float64(0.75)}},
                {"model": np.int64(123), "data_loader": 456},
                1.25,
            )
            run.record_checkpoint(1, np.float64(0.75), {"model": checkpoint})
            run.complete({"aggregate": {"R2": {"mean": 0.75}}})

            metadata = self._metadata(run_dir)
            config_snapshot = yaml.safe_load(
                (run_dir / "config.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(metadata["experiment_metadata_version"], 2)
        self.assertEqual(metadata["run"]["status"], "completed")
        self.assertEqual(metadata["run"]["stage"], "characterizer")
        self.assertEqual(metadata["run"]["pipeline_run_id"], "pipeline-123")
        self.assertEqual(metadata["config"]["snapshot"], "config.yaml")
        self.assertEqual(metadata["source"]["snapshot"], "code.zip")
        self.assertEqual(metadata["environment"]["schema_version"], 1)
        self.assertEqual(
            metadata["execution"]["deterministic_algorithms"],
            metadata["environment"]["pytorch"][
                "deterministic_algorithms"
            ]["enabled"],
        )
        self.assertEqual(
            config_snapshot,
            {
                "data": {
                    "n_params": 2,
                    "param_names": ["Mass", "Energy"],
                }
            },
        )
        self.assertEqual(
            metadata["preprocessing"]["artefact_schema_version"], 3
        )
        self.assertEqual(metadata["checkpoint"]["best_fold"], 1)
        self.assertEqual(metadata["checkpoint"]["best_score"], 0.75)
        self.assertEqual(
            set(metadata["artefacts"]),
            {
                "code.zip",
                "config.yaml",
                "model.pth",
                "preprocessing_metadata.yaml",
            },
        )

    def test_checkpoint_accepts_symlinked_experiment_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            physical_root = root / "physical"
            physical_root.mkdir()
            alias_root = root / "alias"

            try:
                alias_root.symlink_to(
                    physical_root,
                    target_is_directory=True,
                )
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Symbolic links are unavailable: {exc}")

            run_dir = alias_root / "experiment"
            run = ExperimentRun.start(
                stage="characterizer",
                config={},
                exp_dir=run_dir,
                repository_root=self._source_root(root),
            )

            checkpoint = run_dir / "model.pth"
            checkpoint.write_bytes(b"weights")
            run.record_checkpoint(1, 0.75, {"model": checkpoint})

            metadata = self._metadata(run.directory)

        self.assertEqual(
            metadata["checkpoint"]["files"]["model"]["path"],
            "model.pth",
        )

    def test_failed_run_preserves_failure_and_partial_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "experiment"
            run = ExperimentRun.start(
                stage="generator",
                config={"data": {"n_params": 1}},
                exp_dir=run_dir,
                repository_root=self._source_root(root),
            )
            run.record_fold(1, {"R2": 0.1}, {"model": 1}, 0.5)
            run.fail(RuntimeError("training stopped"))
            metadata = self._metadata(run_dir)

        self.assertEqual(metadata["run"]["status"], "failed")
        self.assertEqual(metadata["run"]["error_type"], "RuntimeError")
        self.assertEqual(metadata["run"]["error_message"], "training stopped")
        self.assertEqual(len(metadata["results"]["folds"]), 1)

    def test_rejects_non_empty_explicit_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "experiment"
            run_dir.mkdir()
            (run_dir / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "is not empty"):
                ExperimentRun.start(
                    stage="generator",
                    config={},
                    exp_dir=run_dir,
                    repository_root=self._source_root(root),
                )

    def test_invalid_preprocessing_run_leaves_failed_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prep_dir = root / "preprocessing"
            prep_dir.mkdir()
            (prep_dir / "metadata.yaml").write_text(
                yaml.safe_dump({"run": {"status": "running"}}),
                encoding="utf-8",
            )
            run_dir = root / "experiment"

            with self.assertRaisesRegex(
                ValueError,
                "must describe a completed run",
            ):
                ExperimentRun.start(
                    stage="characterizer",
                    config={},
                    exp_dir=run_dir,
                    preprocessing_dir=prep_dir,
                    repository_root=self._source_root(root),
                )

            metadata = self._metadata(run_dir)

        self.assertEqual(metadata["run"]["status"], "failed")
        self.assertEqual(metadata["run"]["error_type"], "ValueError")

    def test_refreshes_effective_execution_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "experiment"
            run = ExperimentRun.start(
                stage="unified",
                config={},
                exp_dir=run_dir,
                repository_root=self._source_root(root),
            )
            execution_environment = {
                "pytorch": {
                    "deterministic_algorithms": {"enabled": True},
                },
                "environment_variables": {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            }

            with patch(
                "astrai.utils.log_experiments.capture_execution_environment",
                return_value=execution_environment,
            ):
                run.record_execution_environment(device="cpu")

            metadata = self._metadata(run_dir)

        self.assertTrue(metadata["execution"]["deterministic_algorithms"])
        self.assertEqual(
            metadata["environment"]["pytorch"],
            execution_environment["pytorch"],
        )
        self.assertEqual(
            metadata["environment"]["environment_variables"],
            execution_environment["environment_variables"],
        )


if __name__ == "__main__":
    unittest.main()
