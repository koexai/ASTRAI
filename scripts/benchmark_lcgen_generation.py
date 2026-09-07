"""Compatibility wrapper; prefer ``astrai benchmark-generation``."""

from astrai.cli.benchmark_lcgen_generation import main


if __name__ == "__main__":
    raise SystemExit(main())
