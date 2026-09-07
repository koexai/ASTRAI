"""Compatibility wrapper; prefer ``astrai pipeline``."""

from astrai.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
