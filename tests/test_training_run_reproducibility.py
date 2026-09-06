import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.train_characterizer import run_characterizer_training
from scripts.train_generator import run_generator_training


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
                    "held_out_fold": 1,
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
                    "held_out_fold": 1,
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
            "x_train_clean_pca.npy": clean_pca,
            "x_train_aug_pca.npy": augmented_pca,
            "x_test_pca.npy": pca.transform(
                x_scaler.transform(raw_curves[4:6])
            ),
            "y_train_scaled.npy": y_train_scaled,
            "y_test_scaled.npy": y_test_scaled,
            "y_test.npy": raw_parameters[4:6],
            "x_test_clean.npy": raw_curves[4:6],
        }
        for filename, values in arrays.items():
            np.save(fold_dir / filename, values.astype(np.float32))

        joblib.dump(x_scaler, prep_dir / "x_scaler.pkl")
        joblib.dump(y_scaler, prep_dir / "y_scaler.pkl")
        joblib.dump(pca, prep_dir / "pca.pkl")

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


if __name__ == "__main__":
    unittest.main()
