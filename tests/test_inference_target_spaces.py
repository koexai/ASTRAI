import unittest

import numpy as np
import torch

from astrai.cli import inference, inference_split


class IdentityTransformer:
    def transform(self, values):
        return np.asarray(values)

    def inverse_transform(self, values):
        return np.asarray(values)


class UnifiedModel:
    def __init__(self, scaled_predictions):
        self.scaled_predictions = scaled_predictions

    def regressor(self, _values):
        return torch.tensor(self.scaled_predictions, dtype=torch.float32)

    def generator(self, values):
        return values


class SplitModel:
    def __init__(self, scaled_predictions=None):
        self.scaled_predictions = scaled_predictions

    def __call__(self, values):
        if self.scaled_predictions is None:
            return values
        return torch.tensor(self.scaled_predictions, dtype=torch.float32)


class InferenceTargetSpaceTests(unittest.TestCase):
    def setUp(self):
        self.identity = IdentityTransformer()
        self.device = torch.device("cpu")
        self.cfg = {"data": {"target_transform": "log1p"}}

    def test_unified_characterization_returns_physical_parameters(self):
        model = UnifiedModel([[0.0, 1.0]])
        predicted = inference.characterize(
            model,
            np.ones((1, 2)),
            self.identity,
            self.identity,
            self.identity,
            self.device,
            self.cfg,
        )
        np.testing.assert_allclose(predicted, [[0.0, np.e - 1.0]])

    def test_unified_generation_transforms_physical_parameters_once(self):
        model = UnifiedModel([[0.0, 0.0]])
        generated = inference.generate(
            model,
            np.array([[0.0, np.e - 1.0]]),
            self.identity,
            self.identity,
            self.identity,
            self.device,
            self.cfg,
        )
        np.testing.assert_allclose(generated, [[0.0, 1.0]], rtol=1e-6)

    def test_split_characterization_returns_physical_parameters(self):
        model = SplitModel([[0.0, 1.0]])
        predicted = inference_split.characterize(
            model,
            np.ones((1, 2)),
            self.identity,
            self.identity,
            self.identity,
            self.device,
            self.cfg,
        )
        np.testing.assert_allclose(predicted, [[0.0, np.e - 1.0]])

    def test_split_scaled_handoff_preserves_model_representation(self):
        model = SplitModel()
        scaled = np.array([[-0.5, 1.5]], dtype=np.float32)
        generated = inference_split.generate_from_scaled(
            model,
            scaled,
            self.identity,
            self.identity,
            self.device,
        )
        np.testing.assert_array_equal(generated, scaled)


if __name__ == "__main__":
    unittest.main()
