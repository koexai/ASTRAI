"""Benchmark the ASTRAI end-to-end inference cycle.

The measured cycle is:

    LCObs -> PPReg -> predicted parameter values -> LCGen -> LCRec

This is a warm-inference benchmark for one light curve at a time. Model and
data loading, checkpoint deserialisation, process start-up and disk I/O are
excluded. StandardScaler/PCA transformations and their inverse transformations
are included.

Run from the repository root, for example:

    python -m scripts.benchmark_end_to_end_inference \
        --config path/to/config.yaml \
        --exp-char path/to/characterizer-experiment \
        --exp-gen path/to/generator-experiment
"""

import argparse
import platform
import time

import numpy as np
import torch

from scripts.inference_split import characterize, generate
from utils.checkpoints import (
    load_characterizer,
    load_config,
    load_data,
    load_generator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LCObs -> PPReg -> LCGen -> LCRec for one light curve."
        )
    )
    parser.add_argument("--config", required=True, help="Configuration YAML")
    parser.add_argument(
        "--exp-char",
        required=True,
        help="Characterizer experiment directory",
    )
    parser.add_argument(
        "--exp-gen",
        required=True,
        help="Generator experiment directory",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional data-path override supported by load_data",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the input light curve used for the benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="Number of unmeasured warm-up cycles",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2000,
        help="Number of measured cycles",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device; auto uses CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional PyTorch CPU thread count override",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def validate_args(args):
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or greater")
    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero")
    if args.threads is not None and args.threads <= 0:
        raise ValueError("--threads must be greater than zero")


def validate_sample_index(sample_index, n_samples):
    if not 0 <= sample_index < n_samples:
        raise IndexError(
            f"--sample-index must be between 0 and {n_samples - 1}"
        )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def report(name, values):
    values = np.asarray(values, dtype=np.float64)
    print(
        f"{name}: "
        f"median={np.median(values):.3f} ms | "
        f"mean={np.mean(values):.3f} ms | "
        f"p95={np.percentile(values, 95):.3f} ms"
    )


def main():
    args = parse_args()
    validate_args(args)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    device = select_device(args.device)

    char_model, char_x_scaler, char_y_scaler, char_pca = load_characterizer(
        cfg, device, args.exp_char
    )
    gen_model, gen_x_scaler, gen_y_scaler, gen_pca = load_generator(
        cfg, device, args.exp_gen
    )
    char_model.eval()
    gen_model.eval()

    x, _ = load_data(args.data, cfg)
    validate_sample_index(args.sample_index, len(x))
    x_obs = x[args.sample_index : args.sample_index + 1]

    def run_cycle():
        synchronize(device)
        start_ns = time.perf_counter_ns()

        pred_params = characterize(
            char_model,
            x_obs,
            char_x_scaler,
            char_y_scaler,
            char_pca,
            device,
        )

        synchronize(device)
        after_ppreg_ns = time.perf_counter_ns()

        lc_rec = generate(
            gen_model,
            pred_params,
            gen_x_scaler,
            gen_y_scaler,
            gen_pca,
            device,
        )

        synchronize(device)
        end_ns = time.perf_counter_ns()

        return (
            (after_ppreg_ns - start_ns) / 1e6,
            (end_ns - after_ppreg_ns) / 1e6,
            (end_ns - start_ns) / 1e6,
            lc_rec,
        )

    for _ in range(args.warmup):
        run_cycle()

    ppreg_times = []
    lcgen_times = []
    total_times = []
    lc_rec = None

    for _ in range(args.runs):
        ppreg_ms, lcgen_ms, total_ms, lc_rec = run_cycle()
        ppreg_times.append(ppreg_ms)
        lcgen_times.append(lcgen_ms)
        total_times.append(total_ms)

    print()
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Warm-up: {args.warmup} | Measured cycles: {args.runs}")
    print(f"Sample index: {args.sample_index}")
    print(f"Input shape: {x_obs.shape} | Output shape: {lc_rec.shape}")
    print(f"Finite output: {np.isfinite(lc_rec).all()}")
    report("PPReg", ppreg_times)
    report("LCGen", lcgen_times)
    report("Full cycle", total_times)


if __name__ == "__main__":
    main()
