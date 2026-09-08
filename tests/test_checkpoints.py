import tempfile
import unittest
from pathlib import Path

import joblib
import torch

from astrai.utils.checkpoints import (
    checkpoint_artefact_paths,
    experiment_artefact_path,
    save_split_checkpoint,
)


class ExperimentArtefactPathTests(unittest.TestCase):
    def test_resolves_legacy_checkpoint_path_inside_current_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "experiment"

            result = experiment_artefact_path(
                run_dir,
                "experiments/old-run/model.pth",
            )

        self.assertEqual(result, run_dir / "model.pth")

    def test_checkpoint_manifest_paths_preserve_roles(self):
        paths = checkpoint_artefact_paths(
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

    def test_split_checkpoint_saves_model_and_preprocessing_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prep_dir = root / "preprocessing"
            exp_dir = root / "experiment"
            prep_dir.mkdir()
            exp_dir.mkdir()
            preprocessing_files = {
                "x_scaler.pkl": "x.pkl",
                "y_scaler.pkl": "y.pkl",
                "pca.pkl": "pca.pkl",
            }
            for source_name, checkpoint_name in preprocessing_files.items():
                joblib.dump(
                    {"source": checkpoint_name},
                    prep_dir / source_name,
                )
            checkpoint_cfg = {
                "model": "model.pth",
                "x_scaler": "x.pkl",
                "y_scaler": "y.pkl",
                "pca": "pca.pkl",
            }
            model = torch.nn.Linear(2, 1)

            paths = save_split_checkpoint(
                exp_dir,
                checkpoint_cfg,
                model,
                prep_dir,
            )

            self.assertEqual(paths, checkpoint_artefact_paths(exp_dir, checkpoint_cfg))
            self.assertEqual(set(paths), {"model", "x_scaler", "y_scaler", "pca"})
            loaded = torch.load(paths["model"], weights_only=True)
            self.assertEqual(loaded.keys(), model.state_dict().keys())
            for role, filename in (
                ("x_scaler", "x.pkl"),
                ("y_scaler", "y.pkl"),
                ("pca", "pca.pkl"),
            ):
                self.assertEqual(joblib.load(paths[role]), {"source": filename})


if __name__ == "__main__":
    unittest.main()
