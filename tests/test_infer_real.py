import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from astrai.cli import infer_real


class RealSupernovaInferenceTests(unittest.TestCase):
    def test_characterization_decodes_zero_in_physical_space(self):
        import torch

        class IdentityTransformer:
            @staticmethod
            def transform(values):
                return np.asarray(values)

            @staticmethod
            def inverse_transform(values):
                return np.asarray(values)

        class FixedModel:
            @staticmethod
            def __call__(_values):
                return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        predicted, scaled = infer_real.run_characterization(
            np.array([40.0, 41.0]),
            FixedModel(),
            IdentityTransformer(),
            IdentityTransformer(),
            IdentityTransformer(),
            torch.device("cpu"),
            {"data": {"target_transform": "log1p"}},
        )

        np.testing.assert_array_equal(scaled, [[0.0, 1.0]])
        np.testing.assert_allclose(predicted, [0.0, np.e - 1.0])

    def test_bol_preprocessing_preserves_the_batch_numerical_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bol_TEST_UBVRI.txt"
            path.write_text(
                "# ph Lobs err L+BB err\n"
                "58411.3 1e40 1e38 1e40 1e38\n"
                "58412.3 1e41 1e38 1e41 1e38\n"
                "58413.3 1e42 1e38 1e42 1e38\n",
                encoding="utf-8",
            )

            curve, days, luminosity, errors = infer_real.load_bol_txt(
                path, 58411.3, 3
            )

        np.testing.assert_allclose(days, [0.0, 1.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(luminosity, [40.0, 41.0, 42.0])
        np.testing.assert_allclose(curve, [40.0, 41.0, 42.0], atol=1e-10)
        np.testing.assert_allclose(
            errors,
            np.array([1e-2, 1e-3, 1e-4]) / np.log(10),
        )

    def test_single_mode_uses_catalogue_explosion_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            info = Path(temp_dir) / "info.txt"
            info.write_text(
                "SN_n t_explosion\nSN2018HNA 58411.3\n",
                encoding="utf-8",
            )
            args = Namespace(
                name="SN2018HNA",
                explosion_epoch=None,
                info=str(info),
                output=str(Path(temp_dir) / "out.pdf"),
                bol_file="observations.txt",
            )

            with patch.object(infer_real, "infer_supernova") as run:
                infer_real._single(args, {}, "cpu", "char", "gen")

        self.assertEqual(run.call_args.args[1], 58411.3)

    def test_single_and_batch_expose_the_same_model_arguments(self):
        parser = infer_real.build_parser()
        shared = [
            "--exp-char",
            "char",
            "--exp-gen",
            "gen",
        ]
        single = parser.parse_args(
            ["single", *shared, "--bol-file", "sn.txt", "--name", "SN"]
        )
        batch = parser.parse_args(["batch", *shared])
        self.assertEqual((single.exp_char, single.exp_gen), ("char", "gen"))
        self.assertEqual((batch.exp_char, batch.exp_gen), ("char", "gen"))

    def test_plot_restores_name_and_neutral_explosion_epoch_label(self):
        figure = infer_real.make_plot(
            "SN2018HNA",
            58411.3,
            np.arange(3),
            np.array([40.0, 41.0, 42.0]),
            np.array([40.0, 41.0, 42.0]),
            np.arange(3),
            np.array([40.0, 41.0, 42.0]),
            np.ones(3) * 0.1,
            ["Mass"],
            np.array([10.0]),
        )
        try:
            self.assertIn("SN2018HNA", figure.axes[0].get_title())
            self.assertIn("58411.3", figure.axes[0].get_title())
            self.assertEqual(
                figure.axes[1].get_xlabel(), "Days from explosion"
            )
        finally:
            infer_real.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
