import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.plot_results import (
    resolve_diagnostic_fold,
    save_reconstruction_error_csv,
    save_sample_diagnostic_csv,
    select_diagnostic_samples,
    validate_experiment_configs,
    validate_scaled_parameter_artifact,
    validate_shared_parameter_scaling,
)


def make_config():
    return {
        "data": {
            "format": "npy_csv",
            "curves_path": "curves.npy",
            "params_path": "params.csv",
            "params_csv_sep": ";",
            "n_days": 421,
            "n_params": 4,
            "param_names": ["Radius", "Mass", "Energy", "Nichel"],
            "samples_per_day": 1,
        },
        "preprocessing": {
            "pca_components": 32,
            "n_splits": 10,
            "random_seed": 42,
        },
        "augmentation": {
            "noise_std": 0.05,
        },
        "characterizer": {
            "training": {},
        },
        "generator": {
            "training": {
                "held_out_fold": 6,
            },
        },
    }


class DummyParameterScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def transform(self, values):
        return (values - self.mean_) / self.scale_


class DiagnosticConfigurationTests(unittest.TestCase):
    def test_accepts_legacy_characterizer_fold_override(self):
        char_cfg = make_config()
        gen_cfg = make_config()

        fold = resolve_diagnostic_fold(
            char_cfg,
            gen_cfg,
            requested_fold=6,
            characterizer_fold=6,
        )

        self.assertEqual(fold, 6)

    def test_accepts_recorded_characterizer_fold(self):
        char_cfg = make_config()
        gen_cfg = make_config()
        char_cfg["characterizer"]["training"]["held_out_fold"] = 6

        fold = resolve_diagnostic_fold(
            char_cfg,
            gen_cfg,
            requested_fold=6,
        )

        self.assertEqual(fold, 6)

    def test_requires_override_for_legacy_characterizer(self):
        char_cfg = make_config()
        gen_cfg = make_config()

        with self.assertRaisesRegex(
            ValueError,
            "pass --characterizer-fold explicitly",
        ):
            resolve_diagnostic_fold(
                char_cfg,
                gen_cfg,
                requested_fold=6,
            )

    def test_rejects_fold_mismatch(self):
        char_cfg = make_config()
        gen_cfg = make_config()

        with self.assertRaisesRegex(ValueError, "Fold mismatch"):
            resolve_diagnostic_fold(
                char_cfg,
                gen_cfg,
                requested_fold=5,
                characterizer_fold=6,
            )

    def test_rejects_out_of_range_fold(self):
        char_cfg = make_config()
        gen_cfg = make_config()

        with self.assertRaisesRegex(ValueError, "valid range 1-10"):
            resolve_diagnostic_fold(
                char_cfg,
                gen_cfg,
                requested_fold=11,
                characterizer_fold=6,
            )

    def test_rejects_incompatible_experiment_configs(self):
        char_cfg = make_config()
        gen_cfg = make_config()
        gen_cfg["preprocessing"]["pca_components"] = 16

        with self.assertRaisesRegex(
            ValueError,
            "preprocessing.pca_components",
        ):
            validate_experiment_configs(char_cfg, gen_cfg)

    def test_selects_samples_by_augmentation_effect(self):
        x_clean = np.zeros((4, 2))
        x_aug = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [4.0, 4.0],
            ]
        )

        selected = select_diagnostic_samples(
            x_clean,
            x_aug,
            [
                "representative",
                "least-affected",
                "most-affected",
                "2",
            ],
        )

        self.assertEqual(
            [item["index"] for item in selected],
            [1, 0, 3, 2],
        )
        self.assertEqual(
            [item["name"] for item in selected],
            [
                "representative",
                "least_affected",
                "most_affected",
                "sample_2",
            ],
        )

    def test_defaults_to_representative_sample(self):
        x_clean = np.zeros((3, 2))
        x_aug = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [3.0, 3.0],
            ]
        )

        selected = select_diagnostic_samples(x_clean, x_aug)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["name"], "representative")
        self.assertEqual(selected[0]["index"], 1)

    def test_rejects_invalid_sample_selection(self):
        x_clean = np.zeros((2, 2))
        x_aug = np.ones((2, 2))

        with self.assertRaisesRegex(
            ValueError,
            "Unknown sample selection",
        ):
            select_diagnostic_samples(
                x_clean,
                x_aug,
                ["best-model"],
            )

        with self.assertRaisesRegex(ValueError, "out of range"):
            select_diagnostic_samples(
                x_clean,
                x_aug,
                ["10"],
            )

    def test_accepts_identical_parameter_scalers(self):
        char_scaler = DummyParameterScaler(
            mean=[1.0, 2.0],
            scale=[3.0, 4.0],
        )
        gen_scaler = DummyParameterScaler(
            mean=[1.0, 2.0],
            scale=[3.0, 4.0],
        )

        validate_shared_parameter_scaling(
            char_scaler,
            gen_scaler,
        )

    def test_rejects_different_parameter_scalers(self):
        char_scaler = DummyParameterScaler(
            mean=[1.0, 2.0],
            scale=[3.0, 4.0],
        )
        gen_scaler = DummyParameterScaler(
            mean=[1.0, 2.5],
            scale=[3.0, 4.0],
        )

        with self.assertRaisesRegex(
            ValueError,
            "characterizer output and generator input",
        ):
            validate_shared_parameter_scaling(
                char_scaler,
                gen_scaler,
            )

    def test_validates_scaled_parameter_artifact(self):
        scaler = DummyParameterScaler(
            mean=[1.0, 2.0],
            scale=[2.0, 4.0],
        )
        y_test = np.array(
            [
                [1.0, 2.0],
                [3.0, 6.0],
            ]
        )
        y_test_scaled = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )

        validate_scaled_parameter_artifact(
            y_test,
            y_test_scaled,
            scaler,
        )

    def test_rejects_incompatible_scaled_parameter_artifact(self):
        scaler = DummyParameterScaler(
            mean=[1.0, 2.0],
            scale=[2.0, 4.0],
        )
        y_test = np.array([[1.0, 2.0]])
        y_test_scaled = np.array([[1.0, 0.0]])

        with self.assertRaisesRegex(
            ValueError,
            "stored y_test_scaled artefact",
        ):
            validate_scaled_parameter_artifact(
                y_test,
                y_test_scaled,
                scaler,
            )

    def test_saves_sample_diagnostic_with_integer_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample.csv"

            save_sample_diagnostic_csv(
                output_path,
                time_axis=np.array([0.0, 1.0]),
                clean_curve=np.array([10.0, 11.0]),
                augmented_curve=np.array([9.5, 11.5]),
                retained_mask=np.array([False, True]),
                reconstructed_curve=np.array([10.1, 10.9]),
            )

            lines = output_path.read_text().splitlines()
            self.assertEqual(
                lines[0],
                "time_days,clean_target,augmented_model_input,"
                "retained_mask,model_reconstruction",
            )
            self.assertEqual(
                [line.split(",")[3] for line in lines[1:]],
                ["0", "1"],
            )

    def test_saves_reconstruction_error_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "reconstruction_error.csv"
            error_curves = {
                "generator_true_parameters_rmse": np.array([0.1, 0.2]),
                "full_pipeline_clean_input_rmse": np.array([0.3, 0.4]),
                "full_pipeline_augmented_input_rmse": np.array([0.5, 0.6]),
            }

            save_reconstruction_error_csv(
                output_path,
                time_axis=np.array([0.0, 1.0]),
                error_curves=error_curves,
            )

            data = np.genfromtxt(
                output_path,
                delimiter=",",
                names=True,
            )
            self.assertEqual(
                data.dtype.names,
                (
                    "time_days",
                    "generator_true_parameters_rmse",
                    "full_pipeline_clean_input_rmse",
                    "full_pipeline_augmented_input_rmse",
                ),
            )
            np.testing.assert_allclose(
                data["generator_true_parameters_rmse"],
                [0.1, 0.2],
            )
            np.testing.assert_allclose(
                data["full_pipeline_clean_input_rmse"],
                [0.3, 0.4],
            )
            np.testing.assert_allclose(
                data["full_pipeline_augmented_input_rmse"],
                [0.5, 0.6],
            )


if __name__ == "__main__":
    unittest.main()
