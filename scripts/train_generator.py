"""Compatibility wrapper; prefer ``astrai train-generator``."""

from astrai.cli.train_generator import main


if __name__ == "__main__":
    raise SystemExit(main())
