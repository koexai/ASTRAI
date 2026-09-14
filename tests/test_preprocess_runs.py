import hashlib
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from astrai.cli import preprocess


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
                base_dir.resolve() / "20260903_180507_research-config",
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

            resolved_run_dir = run_dir.resolve()
            self.assertEqual(result, str(resolved_run_dir))
            self.assertIn(f"--prep {resolved_run_dir}", output.getvalue())
            self.assertEqual(
                (run_dir / "config.yaml").read_text(encoding="utf-8"),
                config_contents,
            )
            self.assertTrue((run_dir / "code.zip").is_file())
            generate.assert_called_once_with(self.cfg, resolved_run_dir)

            metadata = yaml.safe_load(
                (run_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["preprocessing_artefact_schema_version"],
                6,
            )
            self.assertEqual(metadata["target_transform"]["name"], "log1p")
            self.assertEqual(metadata["run"]["status"], "completed")
            self.assertIsNotNone(metadata["run"]["completed_at_utc"])
            self.assertEqual(metadata["config"]["snapshot"], "config.yaml")
            self.assertEqual(
                metadata["preprocessing"],
                {
                    "random_seed": 42,
                    "n_splits": 2,
                    "folds": [1, 2],
                    "seed_plan": preprocess.build_pool_preprocessing_seed_plan(42, 2),
                },
            )
            self.assertEqual(
                metadata["array_dtypes"],
                {
                    "model": "float32",
                    "indices": "int64",
                },
            )
            self.assertEqual(metadata["array_artefacts"], {})
            self.assertEqual(metadata["environment"]["schema_version"], 1)
            self.assertIn("python", metadata["environment"])
            self.assertIn("installed_distributions", metadata["environment"])
            self.assertIn("pytorch", metadata["environment"])
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

    def test_interrupted_run_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "interrupted-run"

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
                    side_effect=KeyboardInterrupt("preprocessing interrupted"),
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    preprocess.run_preprocessing(
                        self.cfg,
                        out_dir=run_dir,
                    )

            metadata = yaml.safe_load(
                (run_dir / "metadata.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(metadata["run"]["status"], "failed")
        self.assertEqual(metadata["run"]["error_type"], "KeyboardInterrupt")
        self.assertEqual(
            metadata["run"]["error_message"],
            "preprocessing interrupted",
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

    def test_completed_run_records_array_dtypes_and_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"

            def generate_artefacts(_cfg, destination):
                fold_dir = Path(destination) / "fold_1"
                fold_dir.mkdir()
                np.save(
                    Path(destination) / "x_raw.npy",
                    np.ones((3, 4), dtype=np.float32),
                )
                np.save(
                    fold_dir / "train_idx.npy",
                    np.array([0, 2], dtype=np.int64),
                )

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
                    side_effect=generate_artefacts,
                ),
                redirect_stdout(StringIO()),
            ):
                preprocess.run_preprocessing(self.cfg, out_dir=run_dir)

            metadata = yaml.safe_load(
                (run_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["array_artefacts"],
                {
                    "fold_1/train_idx.npy": {
                        "sha256": hashlib.sha256((run_dir / "fold_1/train_idx.npy").read_bytes()).hexdigest(),
                        "dtype": "int64",
                        "shape": [2],
                    },
                    "x_raw.npy": {
                        "sha256": hashlib.sha256((run_dir / "x_raw.npy").read_bytes()).hexdigest(),
                        "dtype": "float32",
                        "shape": [3, 4],
                    },
                },
            )


class PreprocessingArrayDtypeTests(unittest.TestCase):
    def test_process_fold_persists_model_and_index_contracts(self):
        from astrai.utils.partitions import Partition
        from astrai.utils.preprocessing import fit_preprocessing
        x_raw = np.arange(28, dtype=np.float32).reshape(7, 4)
        y_physical = np.arange(14, dtype=np.float32).reshape(7, 2)
        y_transformed = np.log1p(y_physical)
        partition = Partition(7, (0, 1, 2, 3, 4), (0, 1, 2), (3, 4), (5, 6), outer_fold=1)
        bundle = fit_preprocessing(x_raw, y_transformed, partition, 2, 123)
        augmented = x_raw[:3].astype(np.float64) + 0.125
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(preprocess, "apply_lsst_pipeline", return_value=(augmented, np.ones_like(augmented, dtype=bool))):
                preprocess._process_fold(1, x_raw, y_physical, y_transformed,
                                         partition, bundle, 4, 0.05, 1, temp_dir, 123)
            fold_dir = Path(temp_dir) / "fold_1"
            for path in fold_dir.glob("*.npy"):
                expected = np.int64 if path.stem.endswith("idx") else np.float32
                self.assertEqual(np.load(path).dtype, np.dtype(expected))
            np.testing.assert_array_equal(np.load(fold_dir / "x_train_aug_pca.npy"),
                                          bundle.transform_curves(augmented))
            self.assertFalse((fold_dir / "y_test.npy").exists())

    def test_generation_applies_the_target_transform_once(self):
        cfg = {"data": {"target_transform": "log1p", "n_days": 2, "samples_per_day": 1},
               "augmentation": {"noise_std": 0.05},
               "preprocessing": {"pca_components": 1, "n_splits": 2, "random_seed": 42}}
        x_raw = np.arange(16, dtype=np.float32).reshape(8, 2)
        y_physical = np.arange(16, dtype=np.float32).reshape(8, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            with (patch.object(preprocess, "load_raw_data", return_value=(x_raw, y_physical)),
                  patch.object(preprocess, "fit_preprocessing", wraps=preprocess.fit_preprocessing) as fit):
                preprocess._generate_preprocessing_artefacts(cfg, temp_dir)
            for call in fit.call_args_list:
                np.testing.assert_allclose(call.args[1], np.log1p(y_physical))
            np.testing.assert_array_equal(np.load(Path(temp_dir) / "y_physical.npy"), y_physical)
            np.testing.assert_allclose(np.load(Path(temp_dir) / "y_transformed.npy"), np.log1p(y_physical))
            self.assertFalse((Path(temp_dir) / "y_raw.npy").exists())


if __name__ == "__main__":
    unittest.main()
