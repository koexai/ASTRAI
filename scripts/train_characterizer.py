"""Compatibility wrapper; prefer ``astrai train-characterizer``."""

from astrai.cli.train_characterizer import main


if __name__ == "__main__":
    raise SystemExit(main())
