# ASTRAI

Machine learning pipeline for astrophysical transient characterization and light curve generation.

ASTRAI provides two tasks:
- **Characterization**: predict physical parameters (mass, radius, energy, ...) from bolometric light curves.
- **Generation**: reconstruct bolometric light curves from physical parameters.

PNRR Project - Developed as part of the National Recovery and Resilience Plan at Koexai S.r.l.

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA following the [official instructions](https://pytorch.org/get-started/locally/).

## Quick Start

### Train (split pipeline, recommended)

Run preprocessing, characterizer, and generator training in sequence:

```bash
python main.py --config configs/default_split.yaml
```

This is equivalent to running the three stages separately:

```bash
python preprocess.py --config configs/default_split.yaml --out preprocessed
python train_characterizer.py --config configs/default_split.yaml --prep preprocessed
python train_generator.py --config configs/default_split.yaml --prep preprocessed
```

### Train (unified model)

Single model with both branches trained jointly:

```bash
python train.py --config configs/default.yaml
```

### Inference

```bash
# Split model
python inference_split.py \
    --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
    --exp_gen  experiments/generator/YYYYMMDD_HHMMSS

# Unified model
python inference.py --exp experiments/YYYYMMDD_HHMMSS
```

Save predictions to file:

```bash
python inference_split.py \
    --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
    --exp_gen  experiments/generator/YYYYMMDD_HHMMSS \
    --output predictions.parquet
```

## Performance Benchmarks

The benchmark commands below use placeholders deliberately. Replace them with
a compatible configuration and experiment directories containing the required
model checkpoints, fitted scalers and PCA artefacts. The reference runs used
local experiment artefacts that are not included in this repository, so a
clean checkout requires equivalent artefacts before the results can be
reproduced.

### End-to-End Inference Benchmark

The `scripts/benchmark_end_to_end_inference.py` script measures warm inference
latency for one light curve through the complete pipeline:

```text
LCObs -> PPReg -> predicted parameter values -> LCGen -> LCRec
```

The measured path includes input scaling, PCA transformations, the PPReg and
LCGen forward passes, and inverse transformations to obtain the reconstructed
light curve. Python start-up, disk I/O, data loading, checkpoint loading and
model initialisation are excluded.

Run the following command from the repository root:

```bash
/usr/bin/time -p python -m scripts.benchmark_end_to_end_inference \
  --config path/to/config.yaml \
  --exp-char path/to/characterizer-experiment \
  --exp-gen path/to/generator-experiment \
  --device cpu \
  --threads 12 \
  --warmup 200 \
  --runs 20000
```

The command loads the configuration, PPReg and LCGen checkpoints, preprocessing
artefacts and input data once, before timing begins. It selects one light curve
from the configured dataset and processes it individually during each cycle.

Each cycle reports the latency of PPReg, LCGen and the complete end-to-end
pipeline. The generated parameter values and reconstructed light curve remain
in memory and are discarded; the script does not create or modify files.

Use the `Full cycle` median as the representative warm inference time per
observation. The `real` value reported by `/usr/bin/time` also includes process
start-up, model and data loading, warm-up and all measured cycles.

Reference result on an Apple M4 Max CPU with 12 PyTorch threads, Python 3.12.2
and PyTorch 2.13.0:

| Metric | Result |
| --- | ---: |
| Input size | 1 light curve × 421 points |
| Warm-up cycles | 200 |
| Measured cycles | 20,000 |
| PPReg median latency | 0.225 ms |
| LCGen median latency | 0.152 ms |
| Full-cycle median latency | 0.377 ms |
| Full-cycle mean latency | 0.385 ms |
| Full-cycle 95th percentile | 0.405 ms |

The full-cycle median corresponds to approximately 2,650 observations per
second when curves are processed sequentially with the models already loaded
in memory.

### LCGen Batch Generation Benchmark

The `scripts/benchmark_lcgen_generation.py` script measures warm, vectorised
generation throughput:

```text
parameter values -> scaling -> LCGen -> inverse PCA -> light curves
```

Model and data loading, input-batch construction, process start-up and disk I/O
are excluded from the measurement.

Run the following command from the repository root:

```bash
/usr/bin/time -p python -u -m scripts.benchmark_lcgen_generation \
  --config path/to/config.yaml \
  --exp-gen path/to/generator-experiment \
  --device cpu \
  --threads 12 \
  --sizes 1000000 \
  --warmup 3 \
  --warmup-size 1000000 \
  --repeats 30
```

The command loads the configuration, LCGen checkpoint, preprocessing artefacts
and parameter values once, before timing begins. It constructs each requested
batch by repeating rows from the configured dataset.

Each measured run includes parameter scaling, the LCGen forward pass, inverse
PCA and inverse scaling to obtain the final light curves. The script reports
the median, mean, standard deviation, time per curve and throughput. Generated
curves remain in memory and are discarded. Terminal output is saved only if
the caller explicitly uses shell redirection or a command such as `tee`.

Reference result on an Apple M4 Max CPU with 12 PyTorch threads, Python 3.12.2
and PyTorch 2.13.0:

| Metric | Result |
| --- | ---: |
| Batch size | 1,000,000 curves |
| Warm-up runs | 3 × 1,000,000 curves |
| Measured repetitions | 30 |
| Median batch time | 0.720667 s |
| Median time per curve | 0.720667 µs |
| Mean time per curve | 0.724431 ± 0.021485 µs |
| Throughput | 1,387,604 curves/s |

An optional comparison with another implementation can be requested by adding
`--semi-analytic-seconds SECONDS_PER_CURVE`. The resulting speed-up is
indicative only unless both measurements use equivalent hardware, software and
execution conditions; the absolute LCGen time and throughput should therefore
remain the primary results.

## Pipeline Details

### Preprocessing (`preprocess.py`)

Fits scalers and PCA once on the full dataset, then creates K-Fold splits with LSST-augmented training data.

```bash
python preprocess.py --config configs/default_split.yaml --out preprocessed
```

Output structure:

```
preprocessed/
  x_scaler.pkl, y_scaler.pkl, pca.pkl    # Global artifacts
  x_raw.npy, y_raw.npy                   # Raw data (log-transformed params)
  fold_1/
    x_train_clean_pca.npy                 # Clean training curves (PCA space)
    x_train_aug_pca.npy                   # LSST-augmented training curves
    x_test_pca.npy                        # Test curves (PCA space)
    x_test_clean.npy                      # Test curves (original space)
    y_train_scaled.npy, y_test_scaled.npy # Scaled parameters
    y_test.npy                            # Original test parameters
    train_idx.npy, test_idx.npy           # Fold indices
  fold_2/
    ...
```

### Characterizer Training (`train_characterizer.py`)

Trains a `SplitMLPRegressor` (one independent MLP per physical parameter) on PCA-compressed curves.

```bash
python train_characterizer.py --config configs/default_split.yaml --prep preprocessed
```

### Generator Training (`train_generator.py`)

Trains a `MLPWithResiduals` to reconstruct PCA-compressed curves from physical parameters.

```bash
python train_generator.py --config configs/default_split.yaml --prep preprocessed
```

### Inference on Real Supernovae

Single supernova (SN2018hna):

```bash
python infer_sn2018hna.py \
    --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
    --exp_gen  experiments/generator/YYYYMMDD_HHMMSS \
    --csv SN2018hna.csv \
    --output sn2018hna_inference.pdf
```

Batch inference on all supernovae in the `bol/` directory:

```bash
python infer_bol_batch.py \
    --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
    --exp_gen  experiments/generator/YYYYMMDD_HHMMSS \
    --bol_dir bol \
    --output_dir plots/batch
```

### Visualization

Per-timestep reconstruction error and best-sample overlay:

```bash
python plot_results.py \
    --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
    --exp_gen  experiments/generator/YYYYMMDD_HHMMSS \
    --fold 1 \
    --output_dir plots/
```

3-panel reconstruction view for the unified model (original vs augmented vs reconstructed):

```bash
python visualize_reconstruction.py \
    --exp experiments/YYYYMMDD_HHMMSS \
    --top 5
```

Options: `--index N` for a specific sample, `--top N` for the N best by characterization RMSE.

## Configuration

All hyperparameters are set via YAML config files in `configs/`.

### `configs/default_split.yaml` (split pipeline)

| Section | Key Parameters |
|---------|---------------|
| `data` | `format`, `n_days`, `n_params`, `param_names`, `samples_per_day` |
| `preprocessing` | `pca_components` (32), `n_splits` (K-Fold), `random_seed` |
| `augmentation` | `noise_std` (0.05) |
| `characterizer` | `model` (width, depth, dropout), `training` (batch_size, epochs, lr) |
| `generator` | `model` (width, depth, dropout), `training` (batch_size, epochs, lr) |

### `configs/default.yaml` (unified model)

| Section | Key Parameters |
|---------|---------------|
| `data` | Same as above |
| `model` | `pca_components`, `width`, `depth`, `dropout` |
| `training` | `batch_size`, `epochs`, `learning_rate`, `n_splits` |
| `loss` | `alpha_char`, `alpha_gen` (loss weights) |

### Data Formats

- **parquet**: single file with curve columns (`"0"`, `"1"`, ..., `"n_days-1"`) and parameter columns.
- **npy_csv**: separate `.npy` for curves and `.csv` for parameters.

Set the format in the config under `data.format`.

## Experiment Tracking

Each training run creates a timestamped directory under `experiments/`:

```
experiments/
  characterizer/YYYYMMDD_HHMMSS/
    best_characterizer.pth       # Model weights (best fold by R2)
    best_char_x_scaler.pkl       # Feature scaler
    best_char_y_scaler.pkl       # Target scaler
    best_char_pca.pkl            # PCA transformer
    code.zip                     # Source code snapshot
    default_split.yaml           # Config used
  generator/YYYYMMDD_HHMMSS/
    ...
```

## Metrics

Evaluation reports R2, RMSE, RRMSE, and MAE:
- **Characterization**: per-parameter metrics averaged across physical quantities, with bootstrap confidence intervals.
- **Generation**: flattened across all time-steps and samples.

## Support

For questions or support, contact the development team at Koexai S.r.l.
