"""
main.py - Full pipeline for training (preprocessing + characterizer + generator).

Creates separate experiment directories for each model, saves code snapshots
and config, then runs all three stages sequentially.

Usage::

    astrai pipeline
    astrai pipeline --config configs/default_split.yaml
    astrai pipeline --config configs/default_split.yaml --prep-out path/to/run
"""
import argparse
import yaml

from astrai.cli.preprocess import run_preprocessing
from astrai.cli.train_characterizer import run_characterizer_training
from astrai.cli.train_generator import run_generator_training
from astrai.paths import resolve_config_path
from astrai.utils.log_experiments import create_pipeline_run_id


def main(argv=None):
    """Main function to run the full split training pipeline.

    1. Preprocessing (PCA + scalers fitted once, shared by both models)
    2. Characterizer training (own experiment dir)
    3. Generator training (own experiment dir)
    """

    parser = argparse.ArgumentParser(
        prog="astrai pipeline",
        description="ASTRAI split training pipeline"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Split config YAML (default: packaged default_split.yaml)",
    )
    parser.add_argument(
        "--prep-out",
        default=None,
        help=(
            "Exact destination for the new preprocessing run. If omitted, "
            "a timestamped directory is created under preprocessed/."
        ),
    )
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config, "default_split.yaml")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipeline_run_id = create_pipeline_run_id()

    # 1. Preprocessing (PCA + scalers fitted once, shared by both models)
    print("=" * 50)
    print("STAGE 1: PREPROCESSING")
    print("=" * 50)
    prep_dir = run_preprocessing(
        cfg,
        out_dir=args.prep_out,
        config_path=str(config_path),
    )

    # 2. Characterizer training (own experiment dir)
    print("\n" + "=" * 50)
    print("STAGE 2: CHARACTERIZER TRAINING")
    print("=" * 50)
    char_exp = run_characterizer_training(
        cfg,
        prep_dir=prep_dir,
        config_path=str(config_path),
        pipeline_run_id=pipeline_run_id,
    )

    # 3. Generator training (own experiment dir)
    print("\n" + "=" * 50)
    print("STAGE 3: GENERATOR TRAINING")
    print("=" * 50)
    gen_exp = run_generator_training(
        cfg,
        prep_dir=prep_dir,
        config_path=str(config_path),
        pipeline_run_id=pipeline_run_id,
    )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print(f"  Preprocessing: {prep_dir}")
    print(f"  Characterizer: {char_exp}")
    print(f"  Generator:     {gen_exp}")
    print("=" * 50)
    print("\nInference command:")
    print(
        f"  astrai infer-split --exp-char {char_exp} --exp-gen {gen_exp}"
    )


if __name__ == "__main__":
    main()
