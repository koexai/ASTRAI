import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import main as pipeline_main


class SplitPipelineTests(unittest.TestCase):
    def test_passes_the_new_preprocessing_run_to_both_trainers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            prep_out = root / "requested-preprocessing-run"
            completed_prep = str(root / "completed-preprocessing-run")

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "main.py",
                        "--config",
                        str(config_path),
                        "--prep-out",
                        str(prep_out),
                    ],
                ),
                patch.object(
                    pipeline_main,
                    "run_preprocessing",
                    return_value=completed_prep,
                ) as run_preprocessing,
                patch.object(
                    pipeline_main,
                    "create_experiment_dir",
                    side_effect=["characterizer-run", "generator-run"],
                ),
                patch.object(pipeline_main, "save_code"),
                patch.object(pipeline_main, "save_config"),
                patch.object(
                    pipeline_main,
                    "run_characterizer_training",
                ) as run_characterizer,
                patch.object(
                    pipeline_main,
                    "run_generator_training",
                ) as run_generator,
            ):
                pipeline_main.main()

            run_preprocessing.assert_called_once_with(
                {},
                out_dir=str(prep_out),
                config_path=str(config_path),
            )
            run_characterizer.assert_called_once_with(
                {},
                prep_dir=completed_prep,
                exp_dir="characterizer-run",
                config_path=str(config_path),
            )
            run_generator.assert_called_once_with(
                {},
                prep_dir=completed_prep,
                exp_dir="generator-run",
                config_path=str(config_path),
            )

    def test_omitted_prep_out_requests_an_automatic_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("{}\n", encoding="utf-8")

            with (
                patch.object(
                    sys,
                    "argv",
                    ["main.py", "--config", str(config_path)],
                ),
                patch.object(
                    pipeline_main,
                    "run_preprocessing",
                    return_value="preprocessing-run",
                ) as run_preprocessing,
                patch.object(
                    pipeline_main,
                    "create_experiment_dir",
                    side_effect=["characterizer-run", "generator-run"],
                ),
                patch.object(pipeline_main, "save_code"),
                patch.object(pipeline_main, "save_config"),
                patch.object(pipeline_main, "run_characterizer_training"),
                patch.object(pipeline_main, "run_generator_training"),
            ):
                pipeline_main.main()

            run_preprocessing.assert_called_once_with(
                {},
                out_dir=None,
                config_path=str(config_path),
            )


if __name__ == "__main__":
    unittest.main()
