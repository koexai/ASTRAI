"""Compatibility wrapper; prefer ``astrai infer-split``."""

from astrai.cli.inference_split import main


if __name__ == "__main__":
    raise SystemExit(main())
