import unittest

import numpy as np

from astrai.cli import train_generator


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


if __name__ == "__main__":
    unittest.main()
