import tempfile
import unittest
from pathlib import Path

from utils.checkpoints import (
    checkpoint_artifact_paths,
    experiment_artifact_path,
)


class ExperimentArtifactPathTests(unittest.TestCase):
    def test_resolves_legacy_checkpoint_path_inside_current_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "experiment"

            result = experiment_artifact_path(
                run_dir,
                "experiments/old-run/model.pth",
            )

        self.assertEqual(result, run_dir / "model.pth")

    def test_checkpoint_manifest_paths_preserve_roles(self):
        paths = checkpoint_artifact_paths(
            "/tmp/experiment",
            {
                "model": "model.pth",
                "x_scaler": "x.pkl",
                "y_scaler": "y.pkl",
                "pca": "pca.pkl",
                "ignored": "notes.txt",
            },
        )

        self.assertEqual(
            paths,
            {
                "model": Path("/tmp/experiment/model.pth"),
                "x_scaler": Path("/tmp/experiment/x.pkl"),
                "y_scaler": Path("/tmp/experiment/y.pkl"),
                "pca": Path("/tmp/experiment/pca.pkl"),
            },
        )


if __name__ == "__main__":
    unittest.main()
