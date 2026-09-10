import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from astrai.cli import train, train_generator
from astrai.models.unified_model import UnifiedModel


class _RecordingModule(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.inputs = []

    def forward(self, values):
        self.inputs.append(values.detach().clone())
        return self.linear(values)


class _RecordingLoss:
    def __init__(self):
        self.targets = []

    def __call__(self, predictions, targets):
        self.targets.append(targets.detach().clone())
        return torch.mean((predictions - targets) ** 2)


class LCGenCleanTargetTests(unittest.TestCase):
    def test_split_generator_pairs_each_parameter_with_one_clean_curve(self):
        clean_curves = np.arange(18).reshape(6, 3)
        augmented_curves = clean_curves + 1000
        parameters = np.arange(12).reshape(6, 2) + 100

        partition = train_generator._partition_generator_development_data(
            clean_curves,
            parameters,
            validation_fraction=1 / 3,
            validation_seed=42,
        )

        training_indices = partition["training_local_indices"]
        validation_indices = partition["validation_local_indices"]
        np.testing.assert_array_equal(
            partition["training_inputs"],
            parameters[training_indices],
        )
        np.testing.assert_array_equal(
            partition["training_targets"],
            clean_curves[training_indices],
        )
        np.testing.assert_array_equal(
            partition["validation_inputs"],
            parameters[validation_indices],
        )
        self.assertEqual(
            len(partition["training_targets"]),
            len(training_indices),
        )
        self.assertFalse(
            np.isin(partition["training_targets"], augmented_curves).any()
        )

    def test_unified_preprocessing_repeats_clean_generator_targets(self):
        clean_curves = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 3.0, 2.0, 5.0],
                [2.0, 0.0, 4.0, 1.0],
                [3.0, 2.0, 1.0, 4.0],
            ]
        )
        augmented_curves = clean_curves + 100.0
        validation_curves = clean_curves[:2] + 0.5
        test_curves = clean_curves[2:] + 0.5
        parameters = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]]
        )

        with mock.patch.object(
            train,
            "apply_lsst_pipeline",
            return_value=(
                augmented_curves,
                np.ones_like(augmented_curves, dtype=bool),
            ),
        ):
            (
                characterizer_inputs,
                parameter_targets,
                generator_targets,
                *_rest,
            ) = train._preprocess_fold(
                clean_curves,
                validation_curves,
                test_curves,
                parameters,
                parameters[:2],
                parameters[2:],
                n_pca=2,
                noise_std=0.05,
                n_days=4,
                samples_per_day=1,
                fold_idx=1,
                augmentation_seed=11,
                pca_seed=12,
            )

        sample_count = len(clean_curves)
        np.testing.assert_allclose(
            characterizer_inputs[:sample_count],
            generator_targets[:sample_count],
        )
        np.testing.assert_allclose(
            generator_targets[sample_count:],
            generator_targets[:sample_count],
        )
        np.testing.assert_allclose(
            parameter_targets[sample_count:],
            parameter_targets[:sample_count],
        )
        self.assertFalse(
            np.allclose(
                characterizer_inputs[sample_count:],
                generator_targets[sample_count:],
            )
        )

    def test_unified_losses_use_augmented_inputs_and_clean_generator_targets(self):
        regressor = _RecordingModule(2, 1)
        generator = _RecordingModule(1, 2)
        model = UnifiedModel(regressor, generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        characterizer_loss = _RecordingLoss()
        generator_loss = _RecordingLoss()
        curve_inputs = torch.tensor([[1.0, 2.0], [101.0, 102.0]])
        parameter_targets = torch.tensor([[3.0], [3.0]])
        clean_targets = torch.tensor([[1.0, 2.0], [1.0, 2.0]])

        model.fit(
            [(curve_inputs, parameter_targets, clean_targets)],
            optimizer,
            characterizer_loss,
            generator_loss,
            torch.device("cpu"),
            epochs=1,
        )

        torch.testing.assert_close(regressor.inputs[0], curve_inputs)
        torch.testing.assert_close(generator.inputs[0], parameter_targets)
        torch.testing.assert_close(
            characterizer_loss.targets[0],
            parameter_targets,
        )
        torch.testing.assert_close(generator_loss.targets[0], clean_targets)
        self.assertFalse(torch.equal(generator_loss.targets[0], curve_inputs))


if __name__ == "__main__":
    unittest.main()
