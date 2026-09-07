"""Compatibility wrapper; prefer ``astrai preprocess``."""

from astrai.cli.preprocess import main


if __name__ == "__main__":
    raise SystemExit(main())
