import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils import runtime_environment


class _Distribution:
    def __init__(self, name, version):
        self.metadata = {"Name": name}
        self.version = version


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_distribution_versions_are_canonical_and_sorted(self):
        distributions = [
            _Distribution("PyYAML", "6.0.2"),
            _Distribution("scikit_learn", "1.7.1"),
            _Distribution("NumPy", "2.3.2"),
        ]

        with patch.object(
            runtime_environment.importlib_metadata,
            "distributions",
            return_value=distributions,
        ):
            versions = runtime_environment.installed_distribution_versions()

        self.assertEqual(
            versions,
            {
                "numpy": "2.3.2",
                "pyyaml": "6.0.2",
                "scikit-learn": "1.7.1",
            },
        )

    def test_runtime_snapshot_combines_static_and_execution_details(self):
        execution = {
            "pytorch": {"version": "2.7.0"},
            "environment_variables": {"OMP_NUM_THREADS": "4"},
        }
        with (
            patch.object(
                runtime_environment,
                "installed_distribution_versions",
                return_value={"numpy": "2.3.2"},
            ),
            patch.object(
                runtime_environment,
                "capture_execution_environment",
                return_value=execution,
            ),
            patch.object(
                runtime_environment.platform,
                "python_implementation",
                return_value="CPython",
            ),
            patch.object(
                runtime_environment.platform,
                "python_version",
                return_value="3.12.2",
            ),
            patch.object(
                runtime_environment.platform,
                "system",
                return_value="Darwin",
            ),
            patch.object(
                runtime_environment.platform,
                "release",
                return_value="25.0.0",
            ),
            patch.object(
                runtime_environment.platform,
                "machine",
                return_value="arm64",
            ),
        ):
            snapshot = runtime_environment.capture_runtime_environment("cpu")

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(
            snapshot["python"],
            {"implementation": "CPython", "version": "3.12.2"},
        )
        self.assertEqual(
            snapshot["platform"],
            {
                "system": "Darwin",
                "release": "25.0.0",
                "machine": "arm64",
            },
        )
        self.assertEqual(
            snapshot["installed_distributions"],
            {"numpy": "2.3.2"},
        )
        self.assertEqual(snapshot["pytorch"], execution["pytorch"])

    def test_cpu_device_does_not_query_cuda_device_details(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch.cuda,
                "get_device_name",
                side_effect=AssertionError("CUDA device must not be queried"),
            ),
        ):
            pytorch = runtime_environment.capture_torch_environment("cpu")

        self.assertEqual(
            pytorch["selected_device"],
            {"value": "cpu", "type": "cpu", "index": None},
        )
        self.assertFalse(pytorch["cuda"]["available"])
        self.assertIsInstance(pytorch["threads"]["intra_op"], int)
        self.assertIsInstance(pytorch["threads"]["inter_op"], int)

    def test_selected_cuda_device_records_name_and_capability(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=2),
            patch.object(
                torch.cuda,
                "get_device_name",
                return_value="Test GPU",
            ),
            patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=(9, 0),
            ),
        ):
            pytorch = runtime_environment.capture_torch_environment("cuda")

        self.assertEqual(
            pytorch["selected_device"],
            {
                "value": "cuda",
                "type": "cuda",
                "index": 2,
                "name": "Test GPU",
                "capability": [9, 0],
            },
        )

    def test_only_relevant_environment_variables_are_recorded(self):
        with (
            patch.dict(
                os.environ,
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "OMP_NUM_THREADS": "8",
                    "UNRELATED_SECRET": "do-not-record",
                },
                clear=True,
            ),
            patch.object(
                runtime_environment,
                "capture_torch_environment",
                return_value={},
            ),
        ):
            execution = runtime_environment.capture_execution_environment()

        self.assertEqual(
            execution["environment_variables"],
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "8",
            },
        )

    def test_collection_does_not_consume_random_numbers(self):
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        deterministic_state = torch.are_deterministic_algorithms_enabled()

        with patch.object(
            runtime_environment,
            "installed_distribution_versions",
            return_value={},
        ):
            runtime_environment.capture_runtime_environment("cpu")

        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            torch_state,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            torch.are_deterministic_algorithms_enabled(),
            deterministic_state,
        )


if __name__ == "__main__":
    unittest.main()
