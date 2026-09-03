import unittest

import numpy as np

from utils.metrics import (
    METRIC_NAMES,
    compute_parameter_metrics,
)
from utils.parameter_validation import validate_parameter_names


class ParameterNameValidationTests(unittest.TestCase):
    def test_accepts_ordered_unique_parameter_names(self):
        names = validate_parameter_names(2, ["Mass", "Energy"])

        self.assertEqual(names, ("Mass", "Energy"))

    def test_rejects_invalid_parameter_count(self):
        for n_params in (True, 0, -1, 2.0, "2"):
            with self.subTest(n_params=n_params):
                with self.assertRaisesRegex(
                    ValueError,
                    "data.n_params must be a positive integer",
                ):
                    validate_parameter_names(n_params, ["Mass", "Energy"])

    def test_rejects_invalid_parameter_name_sequences(self):
        invalid_names = (
            None,
            "Mass",
            [],
            ["Mass", ""],
            ["Mass", 2],
        )

        for names in invalid_names:
            with self.subTest(names=names):
                with self.assertRaisesRegex(
                    ValueError,
                    "data.param_names must be a non-empty sequence of names",
                ):
                    validate_parameter_names(2, names)

    def test_rejects_duplicate_or_mismatched_parameter_names(self):
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            validate_parameter_names(2, ["Mass", "Mass"])

        with self.assertRaisesRegex(
            ValueError,
            "data.n_params does not match data.param_names: 2 != 1",
        ):
            validate_parameter_names(2, ["Mass"])


class ParameterMetricTests(unittest.TestCase):
    def setUp(self):
        self.true = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        self.pred = np.array(
            [
                [1.0, 12.0],
                [3.0, 18.0],
                [2.0, 33.0],
            ]
        )

    def test_computes_named_metrics_in_configured_order(self):
        result = compute_parameter_metrics(
            self.true,
            self.pred,
            ["Mass", "Energy"],
        )

        self.assertEqual(
            tuple(result["per_parameter"]),
            ("Mass", "Energy"),
        )
        self.assertEqual(
            tuple(result["per_parameter"]["Mass"]),
            METRIC_NAMES,
        )
        self.assertAlmostEqual(
            result["per_parameter"]["Mass"]["RMSE"],
            np.sqrt(2.0 / 3.0),
        )
        self.assertAlmostEqual(
            result["per_parameter"]["Mass"]["RRMSE"],
            np.sqrt(2.0 / 3.0) / 2.0,
        )
        self.assertAlmostEqual(
            result["per_parameter"]["Mass"]["MAE"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(result["per_parameter"]["Mass"]["R2"], 0.0)
        self.assertAlmostEqual(
            result["per_parameter"]["Energy"]["RMSE"],
            np.sqrt(17.0 / 3.0),
        )
        self.assertAlmostEqual(
            result["per_parameter"]["Energy"]["RRMSE"],
            np.sqrt(17.0 / 3.0) / 20.0,
        )
        self.assertAlmostEqual(
            result["per_parameter"]["Energy"]["MAE"],
            7.0 / 3.0,
        )
        self.assertAlmostEqual(
            result["per_parameter"]["Energy"]["R2"],
            0.915,
        )

    def test_aggregate_is_the_unweighted_mean_across_parameters(self):
        result = compute_parameter_metrics(
            self.true,
            self.pred,
            ["Mass", "Energy"],
        )

        for metric_name in METRIC_NAMES:
            expected = np.mean(
                [
                    result["per_parameter"][name][metric_name]
                    for name in ("Mass", "Energy")
                ]
            )
            self.assertAlmostEqual(
                result["aggregate"][metric_name],
                expected,
            )

    def test_rejects_non_matrix_or_empty_inputs(self):
        invalid_pairs = (
            (np.ones(3), np.ones(3), "two-dimensional"),
            (np.ones((0, 2)), np.ones((0, 2)), "at least one sample"),
        )

        for true, pred, message in invalid_pairs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    compute_parameter_metrics(
                        true,
                        pred,
                        ["Mass", "Energy"],
                    )

    def test_rejects_different_target_and_prediction_shapes(self):
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            compute_parameter_metrics(
                np.ones((3, 2)),
                np.ones((2, 2)),
                ["Mass", "Energy"],
            )

    def test_rejects_parameter_count_different_from_array_width(self):
        with self.assertRaisesRegex(
            ValueError,
            "data.n_params does not match data.param_names: 2 != 1",
        ):
            compute_parameter_metrics(
                self.true,
                self.pred,
                ["Mass"],
            )


if __name__ == "__main__":
    unittest.main()
