"""Top-level ASTRAI command dispatcher with lazy subcommand imports."""

import importlib
import sys


COMMANDS = {
    "pipeline": ("astrai.cli.main", "Run preprocessing and split training"),
    "preprocess": ("astrai.cli.preprocess", "Create preprocessing artefacts"),
    "train": ("astrai.cli.train", "Train the unified model"),
    "train-characterizer": (
        "astrai.cli.train_characterizer",
        "Train the split characterizer",
    ),
    "train-generator": (
        "astrai.cli.train_generator",
        "Train the split generator",
    ),
    "infer": ("astrai.cli.inference", "Run unified-model inference"),
    "infer-split": (
        "astrai.cli.inference_split",
        "Run split-model inference",
    ),
    "infer-real": (
        "astrai.cli.infer_real",
        "Run single or batch inference on real bolometric data",
    ),
    "benchmark-inference": (
        "astrai.cli.benchmark_end_to_end_inference",
        "Benchmark end-to-end inference",
    ),
    "benchmark-generation": (
        "astrai.cli.benchmark_lcgen_generation",
        "Benchmark batch generation",
    ),
    "plot-results": ("astrai.utils.plot_results", "Plot inference results"),
    "plot-curves": (
        "astrai.utils.plot_semi_analytical_curves",
        "Plot semi-analytical light curves",
    ),
    "visualize-reconstruction": (
        "astrai.utils.visualize_reconstruction",
        "Visualize reconstructed light curves",
    ),
}


def _print_help(stream=None):
    if stream is None:
        stream = sys.stdout
    stream.write("usage: astrai COMMAND [OPTIONS]\n\n")
    stream.write("ASTRAI command-line interface\n\ncommands:\n")
    width = max(map(len, COMMANDS))
    for name, (_, description) in COMMANDS.items():
        stream.write(f"  {name:<{width}}  {description}\n")
    stream.write("\nRun 'astrai COMMAND --help' for command-specific options.\n")


def main(argv=None):
    """Dispatch one ASTRAI subcommand without importing unrelated dependencies."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if arguments[0] in {"-V", "--version"}:
        from astrai import __version__

        print(f"astrai {__version__}")
        return 0

    command = arguments.pop(0)
    target = COMMANDS.get(command)
    if target is None:
        print(f"astrai: unknown command: {command}", file=sys.stderr)
        _print_help(sys.stderr)
        return 2

    module = importlib.import_module(target[0])
    result = module.main(arguments)
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
