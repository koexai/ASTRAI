"""Compatibility wrapper; prefer ``astrai benchmark-inference``."""

from astrai.cli.benchmark_end_to_end_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
