import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np
import torch

from scripts import train


class UnifiedTrainingReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "data": {
                "n_days": 6,
                "n_params": 2,
                "samples_per_day": 1,
            },
            "model": {
                "pca_components": 2,
                "width": 4,
                "depth": 1,
                "dropout": 0.2,
            },
            "training": {
                "batch_size": 4,
                "epochs": 2,
                "learning_rate": 0.01,
                "n_splits": 2,
                "random_seed": 42,
            },
            "augmentation": {"noise_std": 0.05},
            "loss": {"alpha_char": 1.0, "alpha_gen": 1.0},
            "checkpoint": {},
        }
        self.x_raw = np.linspace(0.1, 4.8, 48).reshape(8, 6)
        self.y_raw = np.linspace(0.1, 1.6, 16).reshape(8, 2)

    def _run_and_capture_checkpoints(self):
        checkpoints = []

        def capture_checkpoint(
            _exp_dir,
            _cfg_checkpoint,
            model,
            _x_scaler,
            _y_scaler,
            _pca,
        ):
            checkpoints.append(
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )

        with (
            patch.object(sys, "argv", ["train.py", "--config", "config.yaml"]),
            patch.object(train, "load_config", return_value=self.cfg),
            patch.object(
                train,
                "load_data",
                return_value=(self.x_raw, self.y_raw),
            ),
            patch.object(train, "create_experiment_dir", return_value="run"),
            patch.object(train, "save_code"),
            patch.object(train, "save_config"),
            patch.object(
                train,
                "save_model_checkpoint",
                side_effect=capture_checkpoint,
            ),
            redirect_stdout(StringIO()),
        ):
            train.main()

        self.assertTrue(checkpoints)
        return checkpoints

    def test_full_unified_training_repeats_exactly(self):
        first = self._run_and_capture_checkpoints()
        torch.rand(100)
        repeated = self._run_and_capture_checkpoints()

        self.assertEqual(len(first), len(repeated))
        for checkpoint_idx, (left, right) in enumerate(zip(first, repeated)):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                with self.subTest(checkpoint=checkpoint_idx, name=name):
                    torch.testing.assert_close(
                        left[name],
                        right[name],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
