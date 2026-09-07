"""Compatibility wrapper; prefer ``astrai infer``."""

from astrai.cli.inference import main


if __name__ == "__main__":
    raise SystemExit(main())
