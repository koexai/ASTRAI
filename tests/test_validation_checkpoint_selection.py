import unittest

import numpy as np
import torch

from astrai.cli import train, train_characterizer, train_generator
from astrai.utils.metrics import compute_selection_metric
from astrai.utils.training import (
    ValidationSelectionTracker,
    assert_selected_validation_score,
    partition_precomputed_development_data,
    resolve_test_fold,
    resolve_training_control,
    split_development_indices,
    train_supervised_model,
)


class _StateModel:
    def __init__(self):
        self.value = 0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class _IdentityTransformer:
    def inverse_transform(self, values):
        return values


class _FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def eval(self):
        return self

    def __call__(self, _inputs):
        return torch.tensor(self.predictions, dtype=torch.float32)


class TrainingControlTests(unittest.TestCase):
    def test_defaults_select_r2_and_leave_early_stopping_disabled(self):
        control = resolve_training_control({}, "transformed.aggregate.R2")

        self.assertEqual(control["validation_fraction"], 0.1)
        self.assertEqual(
            control["selection_policy"],
            {
                "dataset": "validation",
                "metric": "R2",
                "metric_path": "transformed.aggregate.R2",
                "mode": "max",
            },
        )
        self.assertEqual(
            control["early_stopping"],
            {"enabled": False, "patience": 100, "min_delta": 0.0},
        )

    def test_error_metric_changes_direction_without_changing_the_loop(self):
        control = resolve_training_control(
            {"checkpoint_selection": {"metric": "MAE"}},
            "transformed.aggregate.MAE",
        )

        self.assertEqual(control["selection_policy"]["metric"], "MAE")
        self.assertEqual(control["selection_policy"]["mode"], "min")

    def test_rejects_unknown_selection_metric_and_invalid_stopping_settings(self):
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint-selection"):
            resolve_training_control(
                {"checkpoint_selection": {"metric": "loss"}},
                "loss",
            )
        with self.assertRaisesRegex(ValueError, "patience must be at least 1"):
            resolve_training_control(
                {"early_stopping": {"patience": 0}},
                "R2",
            )

    def test_test_fold_prefers_canonical_name_and_checks_legacy_alias(self):
        self.assertEqual(resolve_test_fold({"test_fold": 3}), 3)
        self.assertEqual(resolve_test_fold({"held_out_fold": 4}), 4)
        with self.assertRaisesRegex(ValueError, "must match"):
            resolve_test_fold({"test_fold": 3, "held_out_fold": 4})
        with self.assertRaisesRegex(ValueError, "must match"):
            resolve_test_fold({"test_fold": None, "held_out_fold": 4})


class ValidationSplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_covers_development_pool(self):
        first = split_development_indices(10, 0.2, 123)
        repeated = split_development_indices(10, 0.2, 123)

        np.testing.assert_array_equal(first[0], repeated[0])
        np.testing.assert_array_equal(first[1], repeated[1])
        self.assertEqual(len(first[1]), 2)
        self.assertEqual(set(first[0]) & set(first[1]), set())
        self.assertEqual(set(first[0]) | set(first[1]), set(range(10)))

    def test_r2_minimum_can_reserve_two_validation_samples(self):
        training, validation = split_development_indices(
            4,
            0.1,
            123,
            minimum_validation_samples=2,
        )

        self.assertEqual(len(training), 2)
        self.assertEqual(len(validation), 2)

    def test_partition_excludes_clean_and_augmented_validation_counterparts(self):
        clean = np.arange(6).reshape(6, 1)
        augmented = clean + 100
        targets = clean + 200
        partition = partition_precomputed_development_data(
            clean,
            augmented,
            targets,
            1 / 3,
            42,
        )

        validation = set(partition["validation_local_indices"])
        training_values = set(partition["training_inputs"][:, 0])
        for index in validation:
            self.assertNotIn(index, training_values)
            self.assertNotIn(index + 100, training_values)


class SelectionTrackerTests(unittest.TestCase):
    def test_min_delta_affects_patience_but_not_strict_checkpoint_selection(self):
        model = _StateModel()
        tracker = ValidationSelectionTracker(
            metric="R2",
            mode="max",
            early_stopping_enabled=True,
            patience=2,
            min_delta=0.2,
        )

        model.value = 1
        self.assertFalse(tracker.observe(1, 1.0, model, 1.0, 0.1))
        model.value = 2
        self.assertFalse(tracker.observe(2, 1.1, model, 0.9, 0.1))
        model.value = 3
        self.assertTrue(tracker.observe(3, 1.05, model, 0.8, 0.1))
        tracker.restore_selected_checkpoint(model)

        self.assertEqual(tracker.selected_epoch, 2)
        self.assertEqual(model.value, 2)
        self.assertEqual(tracker.early_stopping_reference_score, 1.0)

    def test_ties_keep_the_first_best_epoch(self):
        model = _StateModel()
        tracker = ValidationSelectionTracker(metric="R2", mode="max")
        model.value = 1
        tracker.observe(1, 0.5, model, 1.0, 0.1)
        model.value = 2
        tracker.observe(2, 0.5, model, 0.9, 0.1)

        tracker.restore_selected_checkpoint(model)
        self.assertEqual(tracker.selected_epoch, 1)
        self.assertEqual(model.value, 1)

    def test_minimised_metric_selects_every_strict_decrease(self):
        model = _StateModel()
        tracker = ValidationSelectionTracker(metric="RMSE", mode="min")
        for epoch, score in enumerate((2.0, 1.9, 1.9, 1.5), 1):
            model.value = epoch
            tracker.observe(epoch, score, model, 1.0, 0.1)

        self.assertEqual(tracker.selected_epoch, 4)
        self.assertEqual(tracker.selected_score, 1.5)

    def test_disabled_early_stopping_runs_all_epochs_and_restores_best(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=3,
        )
        scores = iter((0.1, 0.3, 0.2))
        tracker = ValidationSelectionTracker(
            metric="R2",
            mode="max",
            early_stopping_enabled=False,
            patience=1,
        )

        result = train_supervised_model(
            model,
            [(torch.ones((2, 1)), torch.ones((2, 1)))],
            optimiser,
            criterion,
            scheduler,
            3,
            torch.device("cpu"),
            validation_score_fn=lambda _model: next(scores),
            selection_tracker=tracker,
        )

        self.assertEqual(result["epochs_completed"], 3)
        self.assertFalse(result["stopped_early"])
        self.assertEqual(result["selected_epoch"], 2)


class SelectionScoreConsistencyTests(unittest.TestCase):
    def test_characterizer_selection_matches_complete_transformed_aggregate(self):
        complete = {
            "transformed": {"aggregate": {"R2": 0.75}},
            "physical": {"per_parameter": {}},
        }

        score = assert_selected_validation_score(
            0.75,
            complete,
            "transformed.aggregate.R2",
        )

        self.assertEqual(score, 0.75)

    def test_unified_selection_matches_characterisation_validation_metric(self):
        complete = {
            "characterization": {
                "transformed": {"aggregate": {"MAE": 0.125}},
                "physical": {"per_parameter": {}},
            },
            "generation": {"MAE": 2.0},
        }

        score = assert_selected_validation_score(
            0.125,
            complete,
            "characterization.transformed.aggregate.MAE",
        )

        self.assertEqual(score, 0.125)

    def test_semantic_divergence_between_selection_paths_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            assert_selected_validation_score(
                0.75,
                {"transformed": {"aggregate": {"R2": 0.7}}},
                "transformed.aggregate.R2",
            )

    def test_lightweight_characterisation_metric_matches_complete_convention(self):
        true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        pred = np.array([[1.0, 12.0], [3.0, 18.0], [2.0, 33.0]])

        score = compute_selection_metric(true, pred, "R2", n_columns=2)

        self.assertAlmostEqual(score, (0.0 + 0.915) / 2.0)

    def test_characterizer_lightweight_and_complete_paths_agree(self):
        true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        pred = np.array([[1.0, 12.0], [3.0, 18.0], [2.0, 33.0]])
        model = _FixedPredictionModel(pred)
        scaler = _IdentityTransformer()
        inputs = np.ones((3, 2))

        predicted = train_characterizer._predict_characterizer_transformed(
            model,
            inputs,
            scaler,
            torch.device("cpu"),
        )
        selected_score = compute_selection_metric(
            true,
            predicted,
            "R2",
            n_columns=2,
        )
        complete = train_characterizer._evaluate_characterizer(
            model,
            inputs,
            true,
            ("Mass", "Energy"),
            "/unused",
            torch.device("cpu"),
            {"data": {"target_transform": "log1p"}},
            y_scaler=scaler,
        )

        assert_selected_validation_score(
            selected_score,
            complete,
            "transformed.aggregate.R2",
        )

    def test_unified_lightweight_and_complete_paths_agree(self):
        true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        pred = np.array([[1.0, 12.0], [3.0, 18.0], [2.0, 33.0]])
        inputs = np.ones((3, 2))
        scaler = _IdentityTransformer()
        model = type(
            "FixedUnifiedModel",
            (),
            {
                "regressor": _FixedPredictionModel(pred),
                "generator": _FixedPredictionModel(inputs),
                "eval": lambda self: self,
            },
        )()

        predicted = train._predict_unified_characterisation_transformed(
            model,
            inputs,
            scaler,
            torch.device("cpu"),
        )
        selected_score = compute_selection_metric(
            true,
            predicted,
            "R2",
            n_columns=2,
        )
        characterisation, generation = train._evaluate_fold(
            model,
            inputs,
            true,
            true,
            inputs,
            scaler,
            scaler,
            scaler,
            2,
            torch.device("cpu"),
            {
                "data": {
                    "n_params": 2,
                    "param_names": ["Mass", "Energy"],
                    "target_transform": "log1p",
                }
            },
        )

        assert_selected_validation_score(
            selected_score,
            {
                "characterization": characterisation,
                "generation": generation,
            },
            "characterization.transformed.aggregate.R2",
        )

    def test_generator_lightweight_and_complete_paths_agree(self):
        target_curves = np.array([[1.0, 2.0], [3.0, 5.0]])
        model = _FixedPredictionModel(target_curves + 0.5)
        transformer = _IdentityTransformer()
        parameters = np.ones((2, 2))

        predictions = train_generator._predict_generator_curves(
            model,
            parameters,
            transformer,
            transformer,
            torch.device("cpu"),
        )
        selected_score = compute_selection_metric(
            target_curves,
            predictions,
            "MAE",
        )
        complete = train_generator._evaluate_generator(
            model,
            parameters,
            target_curves,
            "/unused",
            torch.device("cpu"),
            x_scaler=transformer,
            pca=transformer,
        )

        assert_selected_validation_score(selected_score, complete, "MAE")


if __name__ == "__main__":
    unittest.main()
