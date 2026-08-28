"""Benchmark LCGen batch generation throughput.

The measured operation is the complete warm generation path:

    parameter values -> scaling -> LCGen -> inverse PCA -> light curves

Model and data loading, process start-up, input-batch construction, and disk
I/O are excluded. An optional semi-analytical reference time can be supplied
to produce an indicative speed comparison.

Run from the repository root, for example:

    python -m scripts.benchmark_lcgen_generation \
        --config path/to/config.yaml \
        --exp-gen path/to/generator-experiment \
        --device cpu \
        --sizes 1000 10000 100000 1000000 \
        --repeats 5
"""

import argparse
import platform
import time

import numpy as np
import torch

from scripts.inference_split import generate
from utils.checkpoints import load_config, load_data, load_generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark complete LCGen batch generation."
    )
    parser.add_argument("--config", required=True, help="Configuration YAML")
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
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help=(
            "Execution device; auto uses CUDA, then Apple MPS, when "
            "available, otherwise CPU"
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=(1_000, 10_000, 100_000, 1_000_000),
        help="Numbers of curves generated in each benchmark batch",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of unmeasured warm-up runs",
    )
    parser.add_argument(
        "--warmup-size",
        type=int,
        default=1_000,
        help="Batch size used for warm-up runs",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of measured repetitions for each batch size",
    )
    parser.add_argument(
        "--semi-analytic-seconds",
        type=float,
        default=None,
        help=(
            "Optional semi-analytical generation time per curve used for an "
            "indicative speed comparison"
        ),
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS requested but not available")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def validate_args(args):
    if any(size <= 0 for size in args.sizes):
        raise ValueError("Every value passed to --sizes must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or greater")
    if args.warmup_size <= 0:
        raise ValueError("--warmup-size must be greater than zero")
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than zero")
    if (
        args.semi_analytic_seconds is not None
        and args.semi_analytic_seconds <= 0
    ):
        raise ValueError("--semi-analytic-seconds must be greater than zero")
    if args.threads is not None and args.threads <= 0:
        raise ValueError("--threads must be greater than zero")


def validate_labels(labels):
    if labels is None or len(labels) == 0:
        raise ValueError("The configured dataset must contain parameter labels")


def repeat_rows(values, size):
    repetitions = (size + len(values) - 1) // len(values)
    return np.tile(values, (repetitions, 1))[:size]


def timed_generation(model_bundle, parameters, device):
    synchronize(device)
    start_ns = time.perf_counter_ns()
    generated = generate(model_bundle[0], parameters, *model_bundle[1:], device)
    synchronize(device)
    elapsed_seconds = (time.perf_counter_ns() - start_ns) / 1e9
    return elapsed_seconds, generated


def print_result(size, timings, semi_analytic_seconds):
    values = np.asarray(timings, dtype=np.float64)
    median_seconds = float(np.median(values))
    mean_seconds = float(np.mean(values))
    std_seconds = float(np.std(values))
    seconds_per_curve = median_seconds / size
    milliseconds_per_curve = seconds_per_curve * 1e3
    throughput = size / median_seconds
    speedup = None

    print(
        f"{size:>10,} curves | "
        f"median={median_seconds:.6f} s | "
        f"mean={mean_seconds:.6f} s | "
        f"std={std_seconds:.6f} s"
    )
    detail = (
        f"{'':>10}        | "
        f"per curve={milliseconds_per_curve:.9f} ms | "
        f"throughput={throughput:,.0f} curves/s"
    )
    if semi_analytic_seconds is not None:
        speedup = semi_analytic_seconds / seconds_per_curve
        detail += f" | speed-up={speedup:,.0f}x"
    print(detail)

    return {
        "size": size,
        "median_seconds": median_seconds,
        "milliseconds_per_curve": milliseconds_per_curve,
        "throughput": throughput,
        "speedup": speedup,
    }


def main():
    args = parse_args()
    validate_args(args)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    device = select_device(args.device)
    model_bundle = load_generator(cfg, device, args.exp_gen)
    model_bundle[0].eval()

    _, labels = load_data(args.data, cfg)
    validate_labels(labels)

    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Warm-up: {args.warmup} x {args.warmup_size:,} curves")
    print(f"Measured repetitions per size: {args.repeats}")
    if args.semi_analytic_seconds is not None:
        print(
            "Semi-analytical reference: "
            f"{args.semi_analytic_seconds:.6f} s per curve"
        )
        print(
            "Reference comparison is indicative only: hardware, software, "
            "and execution conditions may differ."
        )
    print(
        "Measured path: parameter scaling -> LCGen -> inverse PCA -> "
        f"{cfg['data']['n_days']}-point light curve"
    )

    if args.warmup:
        warmup_parameters = repeat_rows(labels, args.warmup_size)
        warmup_output = None
        for _ in range(args.warmup):
            _, warmup_output = timed_generation(
                model_bundle, warmup_parameters, device
            )
        if not np.isfinite(warmup_output).all():
            raise RuntimeError("Warm-up produced non-finite values")

    results = []
    for size in args.sizes:
        parameters = repeat_rows(labels, size)
        timings = []
        generated = None

        for _ in range(args.repeats):
            elapsed_seconds, generated = timed_generation(
                model_bundle, parameters, device
            )
            timings.append(elapsed_seconds)

        if generated.shape != (size, cfg["data"]["n_days"]):
            raise RuntimeError(
                f"Unexpected output shape {generated.shape} for batch {size}"
            )
        if not np.isfinite(generated[[0, -1]]).all():
            raise RuntimeError("Generation produced non-finite values")

        print()
        results.append(
            print_result(size, timings, args.semi_analytic_seconds)
        )

    largest = max(results, key=lambda result: result["size"])
    print()
    print("Reference result from the largest batch")
    print(f"Batch size: {largest['size']:,} curves")
    print(
        "LCGen time per curve: "
        f"{largest['milliseconds_per_curve']:.9f} ms"
    )
    print(f"LCGen throughput: {largest['throughput']:,.0f} curves/s")
    if largest["speedup"] is not None:
        print(
            "Indicative speed-up over the supplied semi-analytical reference: "
            f"{largest['speedup']:,.0f}x"
        )


if __name__ == "__main__":
    main()
