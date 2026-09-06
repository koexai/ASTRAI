"""
main.py - Full pipeline for training (preprocessing + characterizer + generator).

Creates separate experiment directories for each model, saves code snapshots
and config, then runs all three stages sequentially.

Usage::

    python main.py
    python main.py --config configs/default_split.yaml
    python main.py --config configs/default_split.yaml --prep-out path/to/run
"""
import argparse
import yaml

from preprocess import run_preprocessing
from train_characterizer import run_characterizer_training
from train_generator import run_generator_training
from utils.log_experiments import create_pipeline_run_id


def main():
    """Main function to run the full split training pipeline.

    1. Preprocessing (PCA + scalers fitted once, shared by both models)
    2. Characterizer training (own experiment dir)
    3. Generator training (own experiment dir)
    """

    parser = argparse.ArgumentParser(
        description="ASTRAI split training pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/default_split.yaml",
        help="Path to split config YAML",
    )
    parser.add_argument(
        "--prep-out",
        default=None,
        help=(
            "Exact destination for the new preprocessing run. If omitted, "
            "a timestamped directory is created under preprocessed/."
        ),
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipeline_run_id = create_pipeline_run_id()

    # 1. Preprocessing (PCA + scalers fitted once, shared by both models)
    print("=" * 50)
    print("STAGE 1: PREPROCESSING")
    print("=" * 50)
    prep_dir = run_preprocessing(
        cfg,
        out_dir=args.prep_out,
        config_path=args.config,
    )

    # 2. Characterizer training (own experiment dir)
    print("\n" + "=" * 50)
    print("STAGE 2: CHARACTERIZER TRAINING")
    print("=" * 50)
    char_exp = run_characterizer_training(
        cfg,
        prep_dir=prep_dir,
        config_path=args.config,
        pipeline_run_id=pipeline_run_id,
    )

    # 3. Generator training (own experiment dir)
    print("\n" + "=" * 50)
    print("STAGE 3: GENERATOR TRAINING")
    print("=" * 50)
    gen_exp = run_generator_training(
        cfg,
        prep_dir=prep_dir,
        config_path=args.config,
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
        f"  python inference_split.py --exp_char {char_exp} --exp_gen {gen_exp}"
    )


if __name__ == "__main__":
    main()
