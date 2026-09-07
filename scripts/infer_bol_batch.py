"""Compatibility wrapper; prefer ``astrai infer-real batch``."""

import sys

from astrai.cli.infer_real import main


if __name__ == "__main__":
    raise SystemExit(main(["batch", *sys.argv[1:]]))
