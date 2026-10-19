"""The historical training entry point must fail before touching artefacts."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from astrai.cli import dispatcher, train


class RetiredUnifiedTrainingTests(unittest.TestCase):
    def test_cli_help_remains_available(self):
        with redirect_stdout(StringIO()) as output, self.assertRaises(SystemExit) as raised:
            dispatcher.main(["train", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("retired", output.getvalue())
        self.assertIn("pipeline", output.getvalue())

    def test_cli_rejects_training_without_reading_or_converting_config(self):
        with redirect_stderr(StringIO()) as output, self.assertRaises(SystemExit) as raised:
            dispatcher.main(["train", "--config", "/missing/historical.yaml"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Configurations are not converted automatically", output.getvalue())
        self.assertIn("Historical Unified checkpoints remain supported", output.getvalue())

    def test_programmatic_training_rejects_before_creating_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "experiment"
            with self.assertRaisesRegex(RuntimeError, "retired"):
                train.run_unified_training({}, exp_dir=destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
