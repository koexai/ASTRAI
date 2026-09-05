import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.reproducibility import (
    build_preprocessing_seed_plan,
    build_training_seed_plan,
    configure_torch_determinism,
    derive_diagnostic_seed,
    derive_seed,
    make_torch_generator,
    seed_data_loader_worker,
    validate_seed,
)


class SeedDerivationTests(unittest.TestCase):
    def test_rejects_invalid_seed_values(self):
        for value in (True, 1.5, "42"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    validate_seed(value)

        for value in (-1, 2**32):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_seed(value)

    def test_derivation_is_stable_and_namespaced(self):
        self.assertEqual(derive_seed(42, 1), 3329053876)
        self.assertEqual(derive_seed(42, 2, 1), 1385871029)
        self.assertNotEqual(derive_seed(42, 2, 1), derive_seed(42, 2, 2))

    def test_preprocessing_plan_preserves_the_kfold_seed(self):
        plan = build_preprocessing_seed_plan(42, 2)

        self.assertEqual(plan["k_fold"], 42)
        self.assertEqual(plan["pca"], 3329053876)
        self.assertEqual(
            plan["augmentation"],
            {
                "fold_1": 1385871029,
                "fold_2": 1939820029,
            },
        )

    def test_training_stages_and_folds_have_independent_seeds(self):
        characterizer = build_training_seed_plan(42, "characterizer", 1)
        repeated = build_training_seed_plan(42, "characterizer", 1)
        next_fold = build_training_seed_plan(42, "characterizer", 2)
        generator = build_training_seed_plan(42, "generator", 1)

        self.assertEqual(characterizer, repeated)
        self.assertNotEqual(characterizer["model"], next_fold["model"])
        self.assertNotEqual(characterizer["model"], generator["model"])
        self.assertNotEqual(
            characterizer["model"],
            characterizer["data_loader"],
        )

    def test_diagnostic_seed_depends_on_sample_not_execution_order(self):
        first = derive_diagnostic_seed(42, 7)

        derive_diagnostic_seed(42, 2)
        repeated = derive_diagnostic_seed(42, 7)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, derive_diagnostic_seed(42, 8))


class TorchDeterminismTests(unittest.TestCase):
    @staticmethod
    def _train_small_model(base_seed, fold_idx):
        plan = build_training_seed_plan(
            base_seed,
            "characterizer",
            fold_idx,
        )
        configure_torch_determinism(plan["model"])

        inputs = torch.arange(32, dtype=torch.float32).reshape(16, 2) / 10
        targets = inputs.sum(dim=1, keepdim=True)
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=4,
            shuffle=True,
            generator=make_torch_generator(plan["data_loader"]),
            worker_init_fn=seed_data_loader_worker,
        )
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(8, 1),
        )
        optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        model.train()
        for _ in range(3):
            for batch_inputs, batch_targets in loader:
                optimiser.zero_grad()
                loss = criterion(model(batch_inputs), batch_targets)
                loss.backward()
                optimiser.step()

        return {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def test_global_sequences_repeat_after_configuration(self):
        configure_torch_determinism(123)
        first = (
            random.random(),
            np.random.random(),
            torch.rand(3),
        )

        configure_torch_determinism(123)
        second = (
            random.random(),
            np.random.random(),
            torch.rand(3),
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_model_training_repeats_exactly_for_the_same_fold(self):
        first = self._train_small_model(42, 3)

        # Exercise another fold between the two equivalent runs. The result
        # for fold 3 must not depend on this intervening random consumption.
        self._train_small_model(42, 1)
        repeated = self._train_small_model(42, 3)

        self.assertEqual(first.keys(), repeated.keys())
        for name in first:
            with self.subTest(name=name):
                torch.testing.assert_close(
                    first[name],
                    repeated[name],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
