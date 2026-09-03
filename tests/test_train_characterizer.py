import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_characterizer


class IdentityScaler:
    def inverse_transform(self, values):
        return values


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.evaluation_mode = False

    def eval(self):
        self.evaluation_mode = True

    def __call__(self, _):
        return torch.tensor(self.predictions, dtype=torch.float32)


class TrainingModel:
    def to(self, _device):
        return self

    def parameters(self):
        return ()

    def state_dict(self):
        return {"weight": 1.0}


def make_parameter_history():
    return {
        "Mass": {
            "RMSE": [1.0, 3.0],
            "RRMSE": [0.1, 0.3],
            "MAE": [0.8, 1.2],
            "R2": [0.7, 0.9],
        },
        "Energy": {
            "RMSE": [2.0, 4.0],
            "RRMSE": [0.2, 0.4],
            "MAE": [1.5, 2.5],
            "R2": [0.6, 0.8],
        },
    }


class CharacterizerMetricIntegrationTests(unittest.TestCase):
    def test_evaluation_returns_named_and_aggregate_metrics(self):
        true = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        predictions = np.array(
            [
                [1.0, 12.0],
                [3.0, 18.0],
                [2.0, 33.0],
            ]
        )
        model = FixedPredictionModel(predictions)

        with mock.patch.object(
            train_characterizer.joblib,
            "load",
            return_value=IdentityScaler(),
        ):
            result = train_characterizer._evaluate_characterizer(
                model,
                np.ones((3, 2)),
                true,
                ("Mass", "Energy"),
                "/tmp/preprocessed",
                torch.device("cpu"),
            )

        self.assertTrue(model.evaluation_mode)
        self.assertEqual(
            tuple(result["per_parameter"]),
            ("Mass", "Energy"),
        )
        self.assertAlmostEqual(
            result["aggregate"]["R2"],
            (0.0 + 0.915) / 2.0,
            places=6,
        )

    def test_invalid_parameter_configuration_precedes_artifact_creation(self):
        cfg = {
            "data": {
                "n_params": 2,
                "param_names": ["Mass"],
            }
        }

        with mock.patch.object(
            train_characterizer,
            "create_experiment_dir",
        ) as create_experiment_dir:
            with self.assertRaisesRegex(
                ValueError,
                "data.n_params does not match data.param_names",
            ):
                train_characterizer.run_characterizer_training(cfg)

        create_experiment_dir.assert_not_called()

    def test_records_each_fold_in_parameter_history(self):
        history = train_characterizer._initialise_parameter_history(
            ("Mass", "Energy")
        )
        first_fold = {
            "Mass": {
                "RMSE": 1.0,
                "RRMSE": 0.1,
                "MAE": 0.8,
                "R2": 0.7,
            },
            "Energy": {
                "RMSE": 2.0,
                "RRMSE": 0.2,
                "MAE": 1.5,
                "R2": 0.6,
            },
        }

        train_characterizer._record_parameter_metrics(history, first_fold)

        self.assertEqual(history["Mass"]["RMSE"], [1.0])
        self.assertEqual(history["Energy"]["R2"], [0.6])

    def test_checkpoint_selection_still_uses_aggregate_r2(self):
        cfg = {
            "data": {
                "n_params": 2,
                "param_names": ["Mass", "Energy"],
            },
            "preprocessing": {
                "pca_components": 1,
                "n_splits": 2,
            },
            "characterizer": {
                "model": {
                    "width": 4,
                    "depth": 1,
                    "dropout": 0.0,
                },
                "training": {
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "held_out_fold": None,
                },
                "checkpoint": {
                    "model": "best_characterizer.pth",
                },
            },
        }
        fold_data = (
            np.ones((2, 1)),
            np.ones((2, 1)),
            np.ones((2, 1)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )
        first_evaluation = {
            "aggregate": {
                "RMSE": 1.0,
                "RRMSE": 0.1,
                "MAE": 0.8,
                "R2": 0.8,
            },
            "per_parameter": {
                "Mass": {
                    "RMSE": 1.0,
                    "RRMSE": 0.1,
                    "MAE": 0.8,
                    "R2": 0.1,
                },
                "Energy": {
                    "RMSE": 2.0,
                    "RRMSE": 0.2,
                    "MAE": 1.5,
                    "R2": 0.2,
                },
            },
        }
        second_evaluation = {
            "aggregate": {
                "RMSE": 0.9,
                "RRMSE": 0.09,
                "MAE": 0.7,
                "R2": 0.7,
            },
            "per_parameter": {
                "Mass": {
                    "RMSE": 0.5,
                    "RRMSE": 0.05,
                    "MAE": 0.4,
                    "R2": 0.99,
                },
                "Energy": {
                    "RMSE": 0.6,
                    "RRMSE": 0.06,
                    "MAE": 0.5,
                    "R2": 0.99,
                },
            },
        }

        with (
            mock.patch.object(
                train_characterizer,
                "_load_fold_data",
                return_value=fold_data,
            ),
            mock.patch.object(train_characterizer, "_train_model"),
            mock.patch.object(
                train_characterizer,
                "_evaluate_characterizer",
                side_effect=(first_evaluation, second_evaluation),
            ),
            mock.patch.object(
                train_characterizer,
                "SplitMLPRegressor",
                return_value=TrainingModel(),
            ),
            mock.patch.object(train_characterizer, "TensorDataset"),
            mock.patch.object(train_characterizer, "DataLoader"),
            mock.patch.object(train_characterizer.torch.optim, "Adam"),
            mock.patch.object(train_characterizer.torch.nn, "MSELoss"),
            mock.patch.object(train_characterizer, "CosineAnnealingLR"),
            mock.patch.object(train_characterizer.torch, "save") as save,
            mock.patch.object(
                train_characterizer,
                "copy_preprocessing_artifacts",
            ),
            mock.patch.object(train_characterizer, "print_final_stats"),
            redirect_stdout(io.StringIO()),
        ):
            train_characterizer.run_characterizer_training(
                cfg,
                prep_dir="/tmp/preprocessed",
                exp_dir="/tmp/experiment",
            )

        save.assert_called_once()

    def test_prints_multi_fold_parameter_statistics(self):
        output = io.StringIO()

        with redirect_stdout(output):
            train_characterizer._print_parameter_final_stats(
                make_parameter_history(),
                held_out_fold=None,
            )

        report = output.getvalue()
        self.assertIn("PER-PARAMETER CHARACTERIZATION (2-Fold Mean)", report)
        self.assertIn("Mass:", report)
        self.assertIn("RMSE: 2.0000  (+/- 1.0000)", report)
        self.assertIn("R2: 0.8000  (+/- 0.1000)", report)

    def test_prints_single_held_out_fold_parameter_values(self):
        history = make_parameter_history()
        for metric_history in history.values():
            for metric_name in metric_history:
                metric_history[metric_name] = metric_history[metric_name][:1]
        output = io.StringIO()

        with redirect_stdout(output):
            train_characterizer._print_parameter_final_stats(
                history,
                held_out_fold=6,
            )

        report = output.getvalue()
        self.assertIn(
            "PER-PARAMETER CHARACTERIZATION (Held-out fold 6)",
            report,
        )
        self.assertIn("Mass: RMSE=1.000000", report)
        self.assertIn("Energy: RMSE=2.000000", report)

    def test_prints_fold_metrics_in_configured_order(self):
        per_parameter = {
            "Mass": {
                "RMSE": 1.0,
                "RRMSE": 0.1,
                "MAE": 0.8,
                "R2": 0.7,
            },
            "Energy": {
                "RMSE": 2.0,
                "RRMSE": 0.2,
                "MAE": 1.5,
                "R2": 0.6,
            },
        }
        output = io.StringIO()

        with redirect_stdout(output):
            train_characterizer._print_parameter_metrics(per_parameter)

        report = output.getvalue()
        self.assertLess(report.index("Mass:"), report.index("Energy:"))
        self.assertIn(
            "RMSE=1.000000 | RRMSE=0.100000 | "
            "MAE=0.800000 | R2=0.700000",
            report,
        )


if __name__ == "__main__":
    unittest.main()
