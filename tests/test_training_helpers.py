import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from astrai.utils.reproducibility import (
    make_torch_generator,
    seed_data_loader_worker,
)
from astrai.utils.training import (
    build_training_components,
    build_training_loader,
    combine_clean_and_augmented,
    create_metric_history,
    load_fold_arrays,
    print_metric_history,
    record_metric_values,
    select_training_device,
    train_supervised_model,
)


class TrainingHelperTests(unittest.TestCase):
    def test_selects_the_existing_cuda_or_cpu_policy(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(select_training_device(), torch.device("cpu"))
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(select_training_device(), torch.device("cuda"))

    def test_loads_fold_arrays_in_requested_order_and_model_dtype(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "first.npy", np.full((2, 2), 1.0))
            np.save(root / "second.npy", np.full((2, 2), 2.0))

            second, first = load_fold_arrays(
                root,
                ("second.npy", "first.npy"),
            )

        self.assertEqual(second.dtype, np.dtype(np.float32))
        self.assertEqual(first.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(second, np.full((2, 2), 2.0))
        np.testing.assert_array_equal(first, np.full((2, 2), 1.0))

    def test_combines_inputs_and_duplicates_targets_without_reordering(self):
        clean = np.array([[1.0], [2.0]])
        augmented = np.array([[3.0], [4.0]])
        targets = np.array([[10.0], [20.0]])

        inputs, combined_targets = combine_clean_and_augmented(
            clean,
            augmented,
            targets,
        )

        np.testing.assert_array_equal(inputs[:, 0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            combined_targets[:, 0],
            [10.0, 20.0, 10.0, 20.0],
        )

    def test_training_loader_matches_the_previous_seeded_contract(self):
        inputs = np.arange(12, dtype=np.float64).reshape(6, 2)
        targets = np.arange(6, dtype=np.float64).reshape(6, 1)
        actual = build_training_loader(inputs, targets, batch_size=2, seed=42)
        reference = DataLoader(
            TensorDataset(
                torch.FloatTensor(inputs),
                torch.FloatTensor(targets),
            ),
            batch_size=2,
            shuffle=True,
            generator=make_torch_generator(42),
            worker_init_fn=seed_data_loader_worker,
        )

        for actual_batch, reference_batch in zip(actual, reference):
            torch.testing.assert_close(actual_batch[0], reference_batch[0])
            torch.testing.assert_close(actual_batch[1], reference_batch[1])
            self.assertEqual(actual_batch[0].dtype, torch.float32)
            self.assertEqual(actual_batch[1].dtype, torch.float32)

    def test_builds_the_existing_optimisation_components(self):
        model = torch.nn.Linear(2, 1)
        optimizer, criterion, scheduler = build_training_components(
            model,
            {"learning_rate": 0.01, "epochs": 3},
        )

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(criterion, torch.nn.MSELoss)
        self.assertIsInstance(scheduler, CosineAnnealingLR)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(scheduler.T_max, 3)

    def test_training_loop_matches_the_previous_parameter_updates(self):
        torch.manual_seed(42)
        actual_model = torch.nn.Linear(2, 1)
        reference_model = copy.deepcopy(actual_model)
        batches = [
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0], [2.0]]),
            ),
            (
                torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                torch.tensor([[3.0], [4.0]]),
            ),
        ]
        training_cfg = {"learning_rate": 0.01, "epochs": 2}
        actual_components = build_training_components(
            actual_model,
            training_cfg,
        )
        reference_components = build_training_components(
            reference_model,
            training_cfg,
        )

        train_supervised_model(
            actual_model,
            batches,
            *actual_components,
            training_cfg["epochs"],
            torch.device("cpu"),
        )

        optimizer, criterion, scheduler = reference_components
        reference_model.train()
        for _ in range(training_cfg["epochs"]):
            for batch_inputs, batch_targets in batches:
                batch_inputs = batch_inputs.to(torch.device("cpu"))
                batch_targets = batch_targets.to(torch.device("cpu"))
                optimizer.zero_grad()
                predictions = reference_model(batch_inputs)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        for name, actual_value in actual_model.state_dict().items():
            torch.testing.assert_close(
                actual_value,
                reference_model.state_dict()[name],
                rtol=0,
                atol=0,
            )
        self.assertEqual(
            actual_components[0].param_groups[0]["lr"],
            reference_components[0].param_groups[0]["lr"],
        )

    def test_metric_history_preserves_order_and_records_values(self):
        history = create_metric_history()
        metrics = {"RMSE": 1.0, "RRMSE": 0.1, "MAE": 0.8, "R2": 0.9}

        record_metric_values(history, metrics)

        self.assertEqual(tuple(history), ("RMSE", "RRMSE", "MAE", "R2"))
        self.assertEqual(history["R2"], [0.9])

    def test_reports_the_actual_fold_count_and_single_fold(self):
        history = {
            "RMSE": [1.0, 3.0],
            "RRMSE": [0.1, 0.3],
            "MAE": [0.8, 1.2],
            "R2": [0.7, 0.9],
        }
        output = io.StringIO()
        with redirect_stdout(output):
            print_metric_history("GENERATION", history)
        self.assertIn("GENERATION (2-Fold Mean)", output.getvalue())
        self.assertIn("RMSE: 2.0000  (+/- 1.0000)", output.getvalue())

        output = io.StringIO()
        with redirect_stdout(output):
            print_metric_history(
                "GENERATION",
                {name: values[:1] for name, values in history.items()},
                held_out_fold=6,
            )
        self.assertIn("GENERATION (Held-out fold 6)", output.getvalue())
        self.assertIn("R2: 0.7000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
