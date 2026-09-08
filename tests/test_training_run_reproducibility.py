import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from astrai.cli.train_characterizer import run_characterizer_training
from astrai.cli.train_generator import run_generator_training


class SplitTrainingRunReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "data": {
                "n_params": 2,
                "param_names": ["Mass", "Energy"],
            },
            "preprocessing": {
                "pca_components": 2,
                "n_splits": 2,
                "random_seed": 42,
            },
            "characterizer": {
                "model": {"width": 4, "depth": 1, "dropout": 0.2},
                "training": {
                    "test_fold": 1,
                    "batch_size": 2,
                    "epochs": 2,
                    "learning_rate": 0.01,
                },
                "checkpoint": {
                    "model": "characterizer.pth",
                    "x_scaler": "char_x_scaler.pkl",
                    "y_scaler": "char_y_scaler.pkl",
                    "pca": "char_pca.pkl",
                },
            },
            "generator": {
                "model": {"width": 4, "depth": 1, "dropout": 0.2},
                "training": {
                    "test_fold": 1,
                    "batch_size": 2,
                    "epochs": 2,
                    "learning_rate": 0.01,
                },
                "checkpoint": {
                    "model": "generator.pth",
                    "x_scaler": "gen_x_scaler.pkl",
                    "y_scaler": "gen_y_scaler.pkl",
                    "pca": "gen_pca.pkl",
                },
            },
        }

    @staticmethod
    def _write_preprocessing_artefacts(prep_dir):
        prep_dir = Path(prep_dir)
        fold_dir = prep_dir / "fold_1"
        fold_dir.mkdir(parents=True)

        raw_curves = np.arange(32, dtype=np.float64).reshape(8, 4) / 10
        raw_parameters = np.arange(16, dtype=np.float64).reshape(8, 2) / 10
        x_scaler = StandardScaler().fit(raw_curves)
        y_scaler = StandardScaler().fit(raw_parameters)
        pca = PCA(n_components=2).fit(x_scaler.transform(raw_curves))

        clean_pca = pca.transform(x_scaler.transform(raw_curves[:4]))
        augmented_pca = clean_pca + 0.05
        y_train_scaled = y_scaler.transform(raw_parameters[:4])
        y_test_scaled = y_scaler.transform(raw_parameters[4:6])

        arrays = {
            "train_idx.npy": np.arange(4, dtype=np.int64),
            "test_idx.npy": np.arange(4, 6, dtype=np.int64),
            "x_train_clean_pca.npy": clean_pca,
            "x_train_aug_pca.npy": augmented_pca,
            "x_test_pca.npy": pca.transform(
                x_scaler.transform(raw_curves[4:6])
            ),
            "y_train_scaled.npy": y_train_scaled,
            "y_test_scaled.npy": y_test_scaled,
            "y_test_transformed.npy": raw_parameters[4:6],
            "y_test_physical.npy": np.expm1(raw_parameters[4:6]),
            "x_test_clean.npy": raw_curves[4:6],
        }
        for filename, values in arrays.items():
            np.save(fold_dir / filename, values)

        np.save(prep_dir / "x_raw.npy", raw_curves.astype(np.float32))
        np.save(
            prep_dir / "y_transformed.npy",
            raw_parameters.astype(np.float32),
        )

        joblib.dump(x_scaler, prep_dir / "x_scaler.pkl")
        joblib.dump(y_scaler, prep_dir / "y_scaler.pkl")
        joblib.dump(pca, prep_dir / "pca.pkl")
        (prep_dir / "metadata.yaml").write_text(
            yaml.safe_dump(
                {
                    "preprocessing_artefact_schema_version": 5,
                    "run": {"status": "completed"},
                    "target_transform": {
                        "name": "log1p",
                        "version": 1,
                        "input_space": "physical",
                        "model_space": "transformed",
                        "physical_domain": "non_negative",
                    },
                }
            ),
            encoding="utf-8",
        )

    def _assert_run_metadata(self, run_dir, stage, model_name):
        metadata = yaml.safe_load(
            (run_dir / "metadata.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["run"]["status"], "completed")
        self.assertEqual(metadata["run"]["stage"], stage)
        self.assertEqual(metadata["reproducibility"]["base_seed"], 42)
        self.assertIn(
            "fold_1",
            metadata["reproducibility"]["fold_seed_plans"],
        )
        self.assertEqual(metadata["preprocessing"]["source_run_status"], "completed")
        self.assertEqual(metadata["preprocessing"]["artefact_schema_version"], 5)
        self.assertEqual(
            metadata["checkpoint"]["selected_checkpoint"]["outer_fold"],
            1,
        )
        self.assertIsNotNone(
            metadata["checkpoint"]["selected_checkpoint"]["epoch"]
        )
        self.assertEqual(
            metadata["checkpoint"]["selection_policy"]["dataset"],
            "validation",
        )
        self.assertIn(model_name, metadata["artefacts"])
        self.assertIn("preprocessing_metadata.yaml", metadata["artefacts"])

    @staticmethod
    def _assert_checkpoints_equal(first_path, second_path):
        first = torch.load(first_path, map_location="cpu", weights_only=True)
        second = torch.load(
            second_path,
            map_location="cpu",
            weights_only=True,
        )
        assert first.keys() == second.keys()
        for name in first:
            torch.testing.assert_close(
                first[name],
                second[name],
                rtol=0,
                atol=0,
            )

    def test_characterizer_checkpoints_repeat_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prep_dir = root / "preprocessing"
            first_dir = root / "characterizer-first"
            second_dir = root / "characterizer-second"
            first_dir.mkdir()
            second_dir.mkdir()
            self._write_preprocessing_artefacts(prep_dir)

            with redirect_stdout(StringIO()):
                run_characterizer_training(
                    self.cfg,
                    prep_dir=str(prep_dir),
                    exp_dir=str(first_dir),
                )
                torch.rand(100)
                run_characterizer_training(
                    self.cfg,
                    prep_dir=str(prep_dir),
                    exp_dir=str(second_dir),
                )

            self._assert_checkpoints_equal(
                first_dir / "characterizer.pth",
                second_dir / "characterizer.pth",
            )
            self._assert_run_metadata(
                first_dir,
                "characterizer",
                "characterizer.pth",
            )

    def test_generator_checkpoints_repeat_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prep_dir = root / "preprocessing"
            first_dir = root / "generator-first"
            second_dir = root / "generator-second"
            first_dir.mkdir()
            second_dir.mkdir()
            self._write_preprocessing_artefacts(prep_dir)

            with redirect_stdout(StringIO()):
                run_generator_training(
                    self.cfg,
                    prep_dir=str(prep_dir),
                    exp_dir=str(first_dir),
                )
                torch.rand(100)
                run_generator_training(
                    self.cfg,
                    prep_dir=str(prep_dir),
                    exp_dir=str(second_dir),
                )

            self._assert_checkpoints_equal(
                first_dir / "generator.pth",
                second_dir / "generator.pth",
            )
            self._assert_run_metadata(
                first_dir,
                "generator",
                "generator.pth",
            )


if __name__ == "__main__":
    unittest.main()
