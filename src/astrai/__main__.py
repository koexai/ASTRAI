"""Support ``python -m astrai`` as an alternative to the console script."""

from astrai.cli.dispatcher import main


if __name__ == "__main__":
    raise SystemExit(main())
