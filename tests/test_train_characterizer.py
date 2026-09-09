import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch


from astrai.cli import train_characterizer


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
                {"data": {"target_transform": "log1p"}},
            )

        self.assertTrue(model.evaluation_mode)
        self.assertEqual(
            tuple(result["transformed"]["per_parameter"]),
            ("Mass", "Energy"),
        )
        self.assertAlmostEqual(
            result["transformed"]["aggregate"]["R2"],
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
            train_characterizer.ExperimentRun,
            "start",
        ) as start_experiment:
            with self.assertRaisesRegex(
                ValueError,
                "data.n_params does not match data.param_names",
            ):
                train_characterizer.run_characterizer_training(cfg)

        start_experiment.assert_not_called()

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

    def test_complete_evaluation_keeps_checkpoint_metric_in_transformed_space(self):
        true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        predictions = np.array([[1.0, 12.0], [3.0, 18.0], [2.0, 33.0]])
        model = FixedPredictionModel(predictions)

        with mock.patch.object(
            train_characterizer.joblib,
            "load",
            return_value=IdentityScaler(),
        ):
            evaluation = train_characterizer._evaluate_characterizer(
                model,
                np.ones((3, 2)),
                true,
                ("Mass", "Energy"),
                "/tmp/preprocessed",
                torch.device("cpu"),
                {"data": {"target_transform": "log1p"}},
            )

        self.assertAlmostEqual(
            evaluation["transformed"]["aggregate"]["R2"],
            (0.0 + 0.915) / 2.0,
        )
        self.assertNotIn("aggregate", evaluation["physical"])

    def test_prints_multi_fold_parameter_statistics(self):
        output = io.StringIO()

        with redirect_stdout(output):
            train_characterizer._print_parameter_final_stats(
                make_parameter_history(),
                test_fold=None,
            )

        report = output.getvalue()
        self.assertIn("PER-PARAMETER CHARACTERIZATION (2-Fold Mean)", report)
        self.assertIn("Mass:", report)
        self.assertIn("RMSE: 2.0000  (+/- 1.0000)", report)
        self.assertIn("R2: 0.8000  (+/- 0.1000)", report)

    def test_prints_single_test_fold_parameter_values(self):
        history = make_parameter_history()
        for metric_history in history.values():
            for metric_name in metric_history:
                metric_history[metric_name] = metric_history[metric_name][:1]
        output = io.StringIO()

        with redirect_stdout(output):
            train_characterizer._print_parameter_final_stats(
                history,
                test_fold=6,
            )

        report = output.getvalue()
        self.assertIn(
            "PER-PARAMETER CHARACTERIZATION (Test fold 6)",
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
            train_characterizer._print_parameter_metrics(
                per_parameter,
                "transformed",
            )

        report = output.getvalue()
        self.assertLess(report.index("Mass:"), report.index("Energy:"))
        self.assertIn(
            "RMSE=1.000000 | RRMSE=0.100000 | "
            "MAE=0.800000 | R2=0.700000",
            report,
        )


if __name__ == "__main__":
    unittest.main()
