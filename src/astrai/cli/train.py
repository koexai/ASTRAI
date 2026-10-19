"""Compatibility entry point for the retired shared Unified training."""

import argparse


RETIREMENT_MESSAGE = (
    "Shared Unified training has been retired. Use 'astrai pipeline --config "
    "configs/default_split.yaml' (four parameters) or "
    "'astrai pipeline --config configs/7par_split.yaml' (seven parameters), "
    "or train-characterizer and train-generator with shared preprocessing. "
    "Historical Unified checkpoints remain supported for inference. "
    "Configurations are not converted automatically."
)


def run_unified_training(cfg, exp_dir=None, config_path=None):
    """Reject new shared training runs before creating any artefacts."""
    raise RuntimeError(RETIREMENT_MESSAGE)


def main(argv=None):
    """Keep discoverable CLI help and reject the retired training command."""
    parser = argparse.ArgumentParser(
        prog="astrai train", description=RETIREMENT_MESSAGE
    )
    parser.add_argument("--config", help="Historical Unified configuration")
    parser.parse_args(argv)
    parser.error(RETIREMENT_MESSAGE)


if __name__ == "__main__":
    main()
