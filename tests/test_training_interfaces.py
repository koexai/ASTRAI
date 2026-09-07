import io
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from astrai.cli import train_characterizer, train_generator


@contextmanager
def change_working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class TrainingInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "data": {
                "n_params": 2,
                "param_names": ["Mass", "Energy"],
            },
            "preprocessing": {
                "pca_components": 2,
                "n_splits": 2,
                "random_seed": 42,
            },
            "characterizer": {
                "model": {"width": 4, "depth": 1, "dropout": 0.0},
                "training": {
                    "held_out_fold": 1,
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 0.01,
                },
                "checkpoint": {"model": "characterizer.pth"},
            },
            "generator": {
                "model": {"width": 4, "depth": 1, "dropout": 0.0},
                "training": {
                    "held_out_fold": 1,
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 0.01,
                },
                "checkpoint": {"model": "generator.pth"},
            },
        }

    def _assert_path_and_failure_contract(self, module, run_training):
        experiment = mock.Mock()
        experiment.directory = Path("/tmp/experiment")
        failure = KeyboardInterrupt("training interrupted")

        with tempfile.TemporaryDirectory() as temp_dir:
            working_directory = Path(temp_dir)
            expected_prep_dir = (
                working_directory / "relative-preprocessing"
            ).resolve()
            with (
                change_working_directory(working_directory),
                mock.patch.object(
                    module.ExperimentRun,
                    "start",
                    return_value=experiment,
                ) as start_experiment,
                mock.patch.object(
                    module,
                    "select_training_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.object(
                    module,
                    "_load_fold_data",
                    side_effect=failure,
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_training(
                        self.cfg,
                        prep_dir="relative-preprocessing",
                        exp_dir="experiment",
                    )

        self.assertEqual(
            start_experiment.call_args.kwargs["preprocessing_dir"],
            expected_prep_dir,
        )
        experiment.fail.assert_called_once_with(failure)

    def test_characterizer_resolves_preprocessing_and_records_interruptions(self):
        self._assert_path_and_failure_contract(
            train_characterizer,
            train_characterizer.run_characterizer_training,
        )

    def test_generator_resolves_preprocessing_and_records_interruptions(self):
        self._assert_path_and_failure_contract(
            train_generator,
            train_generator.run_generator_training,
        )


if __name__ == "__main__":
    unittest.main()
