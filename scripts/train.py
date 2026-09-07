"""Compatibility wrapper; prefer ``astrai train``."""

from astrai.cli.train import main


if __name__ == "__main__":
    raise SystemExit(main())
