# ASTRAI

Machine learning pipeline for astrophysical transient characterization and light curve generation.

ASTRAI provides two tasks:
- **Characterization**: predict physical parameters (mass, radius, energy, ...) from bolometric light curves.
- **Generation**: reconstruct bolometric light curves from physical parameters.

PNRR Project - Developed as part of the National Recovery and Resilience Plan at Koexai S.r.l.

## Installation

```bash
python -m pip install --editable .
```

Use `python -m pip install .` for a standard installation. Both modes install
the `astrai` command; no `PYTHONPATH` changes are required. Runtime dependency
minimums are declared identically in `pyproject.toml` and `requirements.txt`.

For GPU support, install PyTorch with CUDA following the [official instructions](https://pytorch.org/get-started/locally/).

`requirements.txt` declares the runtime dependencies using minimum accepted
versions. The unit tests use Python's standard library and currently require no
additional third-party test dependencies. A single exact lock file is not
provided because PyTorch installations differ between CPU, CUDA and macOS
environments. Preprocessing and training metadata record the versions actually
installed for each run, allowing its software environment to be identified.

## Automated Validation

Every pull request and every push to `main` runs the full unit-test suite on
Python 3.12 and Ubuntu 24.04. The workflow verifies the installed dependency
set, compiles the Python sources and runs the tests with a headless Matplotlib
backend. GitHub Actions are pinned to complete commit SHAs. The workflow does
not train models or require external datasets or experiment artefacts.

Run the same checks locally from the repository root:

```bash
python -m pip check
python -m compileall -q src scripts tests
MPLBACKEND=Agg python -m unittest discover -s tests -v
astrai --help
```

## Quick Start

### Train (split pipeline, recommended)

Run preprocessing, characterizer, and generator training in sequence:

```bash
astrai pipeline --config configs/default_split.yaml
```

Preprocessing is written to a new timestamped run directory and that exact
directory is passed automatically to both training stages. To choose the new
directory explicitly, add `--prep-out path/to/new-run`; an existing directory
is rejected.

This is equivalent to running the three stages separately:

```bash
astrai preprocess --config configs/default_split.yaml

# Use the directory printed by preprocessing:
astrai train-characterizer \
  --config configs/default_split.yaml \
  --prep preprocessed/YYYYMMDD_HHMMSS_default_split
astrai train-generator \
  --config configs/default_split.yaml \
  --prep preprocessed/YYYYMMDD_HHMMSS_default_split
```

### Train (unified model)

Single model with both branches trained jointly:

```bash
astrai train --config configs/default.yaml
```

### Inference

```bash
# Split model
astrai infer-split \
    --exp-char experiments/characterizer/YYYYMMDD_HHMMSS_microseconds \
    --exp-gen  experiments/generator/YYYYMMDD_HHMMSS_microseconds

# Unified model
astrai infer --exp experiments/YYYYMMDD_HHMMSS_microseconds
```

Save predictions to file:

```bash
astrai infer-split \
    --exp-char experiments/characterizer/YYYYMMDD_HHMMSS_microseconds \
    --exp-gen  experiments/generator/YYYYMMDD_HHMMSS_microseconds \
    --output predictions.parquet
```

### CLI contract

`astrai --help` lists every supported command and each command provides its
own `--help`. Paths supplied by users are resolved from the current working
directory; packaged default YAML files are resolved from the installation and
therefore work outside the source checkout. Generated outputs are written only
to the explicit destination or to the documented current-directory default.

| Command | Required external inputs | Outputs |
| --- | --- | --- |
| `pipeline` | dataset from config | preprocessing and two experiment runs |
| `preprocess` | dataset from config | isolated preprocessing run |
| `train`, `train-characterizer`, `train-generator` | data/preprocessing and config | isolated experiment run |
| `infer`, `infer-split` | checkpoints and input dataset | terminal metrics; optional predictions Parquet |
| `infer-real single` | bol file, name, two experiment runs, explosion epoch or catalogue | one PDF |
| `infer-real batch` | bol directory, catalogue, two experiment runs | one PDF per object and a parameter CSV |
| `plot-results`, `plot-curves`, `visualize-reconstruction` | command-specific data/artefacts | plots in the requested directory |
| `benchmark-inference`, `benchmark-generation` | checkpoints, data and config | terminal timing report |

Successful commands return exit status `0`. Command-line usage errors,
including unknown commands or missing required options, return `2` through
`argparse`; missing files, invalid data and runtime/model errors return a
non-zero status and include the failure on standard error. Historical modules
under `scripts/` remain as thin source-checkout wrappers, but the `astrai`
commands above are canonical.

## Performance Benchmarks

The benchmark commands below use placeholders deliberately. Replace them with
a compatible configuration and experiment directories containing the required
model checkpoints, fitted scalers and PCA artefacts. The reference runs used
local experiment artefacts that are not included in this repository, so a
clean checkout requires equivalent artefacts before the results can be
reproduced.

### End-to-End Inference Benchmark

The `astrai benchmark-inference` command measures warm inference
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
/usr/bin/time -p astrai benchmark-inference \
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

The `astrai benchmark-generation` command measures warm, vectorised
generation throughput:

```text
parameter values -> scaling -> LCGen -> inverse PCA -> light curves
```

Model and data loading, input-batch construction, process start-up and disk I/O
are excluded from the measurement.

Run the following command from the repository root:

```bash
/usr/bin/time -p astrai benchmark-generation \
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

### Preprocessing (`astrai preprocess`)

Fits scalers and PCA once on the full dataset, then creates K-Fold splits with
LSST-augmented training data. Each invocation creates an isolated run; it does
not write directly into a shared `preprocessed/` directory.

```bash
astrai preprocess --config configs/default_split.yaml
```

The default destination is
`preprocessed/YYYYMMDD_HHMMSS_config-name/`. Use `--out path/to/new-run` to
choose an exact destination. In both cases the destination must not already
exist, preventing previous preprocessing artefacts from being overwritten.
The command prints the exact path to pass to subsequent `--prep` options.

Output structure:

```
preprocessed/
  YYYYMMDD_HHMMSS_config-name/
    config.yaml                            # Exact configuration snapshot
    code.zip                               # Python source snapshot
    metadata.yaml                          # Run, fold and Git metadata
    x_scaler.pkl, y_scaler.pkl, pca.pkl    # Global artefacts
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

`metadata.yaml` records the artefact schema version, UTC start and completion
times, run status, configured random seed, fold list, Git commit, branch and
whether the working tree was dirty. It also records the dtype and shape of
every NumPy artefact produced by a completed run. The current schema records
the seed-derivation scheme, the effective K-fold, PCA and per-fold augmentation
seeds, and a snapshot of the runtime environment. Its version history is
summarised under Experiment Tracking.

Persisted model arrays use `float32`, while fold indices use `int64`. The cast
is applied after augmentation, scaling and PCA, so it does not change those
intermediate calculations; it makes the saved representation match the
`float32` tensors already used by PyTorch. Existing preprocessing directories
containing legacy `float64` model arrays remain supported: training and
diagnostic loaders normalise them to `float32` in memory. Failed runs remain
marked as `failed` and are never silently reused.

### Reproducibility

The configured `random_seed` is the base for independent deterministic
streams. Stable NumPy `SeedSequence` namespaces derive separate seeds for PCA,
each fold's augmentation, model initialisation and DataLoader shuffling. The
K-fold splitter continues to use the configured base seed directly, preserving
the configured fold assignment.

To select a different set of random streams, change only `random_seed` in the
configuration file: `preprocessing.random_seed` for the split pipeline or
`training.random_seed` for the unified model. The value must be an integer from
0 to 4294967295. All stage-specific and fold-specific seeds are derived
automatically and should not be configured individually.

Preprocessing and diagnostic augmentation use explicit local NumPy generators
and do not depend on NumPy's process-global random state. Consequently, two
preprocessing runs with the same code, configuration and data produce identical
NumPy artefacts. `astrai plot-results` accepts `--lsst-seed` (and the legacy
spelling `--lsst_seed`); `astrai visualize-reconstruction` derives an independent
diagnostic stream for each selected sample.

Before every training fold, ASTRAI seeds Python, NumPy and PyTorch, enables
deterministic PyTorch algorithms, configures deterministic cuDNN behaviour and
uses an explicitly seeded DataLoader generator. Model and DataLoader seeds are
derived independently for the Characterizer, Generator and unified model. A
fold therefore has the same random streams whether it is trained alone or
after other stages or folds.

These controls target exact repetition with the same source, configuration,
data, dependency versions, device and execution environment. PyTorch does not
guarantee bitwise-identical results across releases, platforms or CPU and GPU
execution. Deterministic algorithms may also run more slowly and will raise an
error if an operation has no deterministic implementation.

### Characterizer Training (`astrai train-characterizer`)

Trains a `SplitMLPRegressor` (one independent MLP per physical parameter) on PCA-compressed curves.

```bash
astrai train-characterizer \
  --config configs/default_split.yaml \
  --prep preprocessed/YYYYMMDD_HHMMSS_default_split
```

After each evaluated fold, the report includes RMSE, RRMSE, MAE and R2 for
every entry in `data.param_names` alongside the existing aggregate R2. The
final report preserves all existing unweighted aggregate metrics and also
shows the mean and standard deviation of every per-parameter metric across
folds; a configured `held_out_fold` produces a single-fold report instead.

The per-parameter and aggregate values use the same evaluation-space targets
after reversing the target standardisation. This reporting does not introduce
an additional target transformation.

### Generator Training (`astrai train-generator`)

Trains a `MLPWithResiduals` to reconstruct PCA-compressed curves from physical parameters.

```bash
astrai train-generator \
  --config configs/default_split.yaml \
  --prep preprocessed/YYYYMMDD_HHMMSS_default_split
```

### Inference on Real Supernovae

The generic real-data command accepts the observed `bol` text format used by
the existing batch pipeline. Single and batch modes execute the same `L+BB`
loading, interpolation, missing-edge filling, scaling/PCA, characterization
and generation code.

For SN2018HNA, the catalogue explosion epoch is `58411.3`. The numeric epoch
and current unit convention are preserved as-is; no discovery epoch or unit
conversion is substituted.

```bash
astrai infer-real single \
  --name SN2018HNA \
  --bol-file data/real/87Alike_bolometric/bol_SN2018HNA_UBVRI.txt \
  --info data/real/metadata/info_87Alike.txt \
  --exp-char experiments/characterizer/YYYYMMDD_HHMMSS_microseconds \
  --exp-gen experiments/generator/YYYYMMDD_HHMMSS_microseconds \
  --output plots/SN2018HNA_inference.pdf
```

`--explosion-epoch 58411.3` may be supplied instead of looking the value up
in `--info`. Batch mode processes every catalogue entry with a matching file:

```bash
astrai infer-real batch \
  --exp-char experiments/characterizer/YYYYMMDD_HHMMSS_microseconds \
  --exp-gen experiments/generator/YYYYMMDD_HHMMSS_microseconds \
  --bol-dir data/real/87Alike_bolometric \
  --info data/real/metadata/info_87Alike.txt \
  --output-dir plots/batch
```

### Visualization

Per-timestep reconstruction error and best-sample overlay:

```bash
astrai plot-results \
    --exp-char experiments/characterizer/YYYYMMDD_HHMMSS_microseconds \
    --exp-gen  experiments/generator/YYYYMMDD_HHMMSS_microseconds \
    --fold 1 \
    --output-dir plots/
```

3-panel reconstruction view for the unified model (original vs augmented vs reconstructed):

```bash
astrai visualize-reconstruction \
    --exp experiments/YYYYMMDD_HHMMSS_microseconds \
    --top 5
```

Options: `--index N` for a specific sample, `--top N` for the N best by characterization RMSE.

### Semi-analytical Model Curves

`astrai plot-curves` compares clean curves already stored in a
configured semi-analytical dataset. It does not run the semi-analytical model
and does not use PPReg, LCGen or the augmentation pipeline. Relative dataset
paths are resolved from the current working directory, or from the explicit
`--data-root`. This keeps user data independent of the installation location.

The four-parameter configuration contains an exact one-at-a-time comparison:

```bash
astrai plot-curves \
  --config configs/4par.yaml \
  --output-dir plots/semi-analytical/4par
```

The reference values, levels and panel titles are declared under
`visualisation.one_at_a_time` in `configs/4par.yaml`. Every requested
combination must match exactly one row; zero or multiple matches produce an
explicit error.

The seven-parameter configuration produces a 3-by-3 layout with seven populated
panels, one for each parameter. Each panel contains three summary curves. To
construct them, the dataset rows are ordered by the parameter shown in that
panel and split according to the rank intervals configured under
`visualisation.quantile_summary.quantile_ranges`. The default intervals are the
lowest 20%, the central 20% (ranks 40--60%) and the highest 20%; the intervening
20--40% and 60--80% ranges are intentionally omitted.

Each displayed curve is the pointwise median of all clean light curves in its
group: every timestep is therefore the median luminosity across the selected
dataset rows. The other six parameters are not held fixed and group membership
is recomputed independently for every panel. The result is a marginal summary
of the available dataset, not an exact one-at-a-time comparison, an individual
semi-analytical model curve or an uncertainty interval. Each legend reports the
actual parameter range and number of rows contributing to the median:

```bash
astrai plot-curves \
  --config configs/default.yaml \
  --output-dir plots/semi-analytical/7par
```

Any configured dataset can also be inspected by selecting one or more existing
rows. This mode overlays complete sample curves in a single plot:

```bash
astrai plot-curves \
  --config configs/default.yaml \
  --output-dir plots/semi-analytical/7par-selected \
  --index 0 \
  --index 1 \
  --index 2
```

Rows may instead be selected by parameter values. A partial selection is
accepted only when it identifies one unique row:

```bash
astrai plot-curves \
  --config configs/4par.yaml \
  --output-dir plots/semi-analytical/selected \
  --parameters 'Radius=13,Mass=20,Energy=4.5,Nichel=0.05'
```

The current seven-parameter dataset is not a controlled Cartesian grid, so no
one-at-a-time effect is inferred from nearest neighbours. The quantile summary
aggregates many samples, while explicit selections compare actual rows and
print their complete parameter vectors. PDF and PNG are produced by default;
repeat `--format` to request selected formats, and use `--show` only when an
interactive window is wanted.

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
| `training` | `batch_size`, `epochs`, `learning_rate`, `n_splits`, `random_seed` |
| `loss` | `alpha_char`, `alpha_gen` (loss weights) |

### Data Formats

- **parquet**: single file with curve columns (`"0"`, `"1"`, ..., `"n_days-1"`) and parameter columns.
- **npy_csv**: separate `.npy` for curves and `.csv` for parameters.

Set the format in the config under `data.format`.

## Experiment Tracking

Each training invocation creates a new directory under `experiments/`. UTC
timestamps include microseconds and creation is atomic; if a name still
collides, a numeric suffix is added. Existing non-empty directories are
rejected rather than reused, so artefacts from separate runs cannot be mixed
or overwritten.

```
experiments/
  characterizer/YYYYMMDD_HHMMSS_microseconds/
    best_characterizer.pth       # Model weights (best fold by R2)
    best_char_x_scaler.pkl       # Feature scaler
    best_char_y_scaler.pkl       # Target scaler
    best_char_pca.pkl            # PCA transformer
    code.zip                     # Source code snapshot
    config.yaml                  # Effective configuration
    preprocessing_metadata.yaml  # Input preprocessing provenance
    metadata.yaml                # Lifecycle, results and artefact manifest
  generator/YYYYMMDD_HHMMSS_microseconds/
    ...
```

The experiment metadata is written when a run starts, after every completed
fold and whenever the best checkpoint changes. It records the run stage and
status, UTC times, Git revision and dirty state, effective configuration,
device, parameter order, dtype contract, selected folds, base and derived
seeds, runtime environment, per-fold and summary metrics, checkpoint selection
and SHA-256 digests for every persisted artefact. The runtime snapshot includes
Python, the operating-system platform, installed distribution versions,
PyTorch backends, thread counts and effective deterministic settings. Failures
retain their error and any partial fold records. Characterizer and Generator
runs started by `astrai pipeline` share a
`pipeline_run_id`, and each snapshots the metadata of its completed
preprocessing input. Older preprocessing directories without metadata remain
accepted and are explicitly marked as legacy inputs.

Source provenance follows the installed `astrai` package location rather than
the process working directory. An editable checkout records that checkout's
Git revision and archives its Python sources. A standard installation archives
the installed package sources and leaves Git fields null instead of
attributing the run to an unrelated repository from which the command happens
to be launched.

Checkpoint entries in configuration files are artefact names. When an older
configuration contains a complete historical path, only its final filename is
resolved inside the selected experiment directory. This preserves legacy
configuration compatibility while keeping all output within the current run.

### Metadata schema history

| Metadata | Version | Additions |
| --- | ---: | --- |
| Preprocessing artefacts | 1 | Run lifecycle, configuration snapshot, fold settings and Git provenance |
| Preprocessing artefacts | 2 | Array dtype contract and per-array dtype/shape manifest |
| Preprocessing artefacts | 3 | Seed-derivation scheme and effective preprocessing seed plan |
| Preprocessing artefacts | 4 | Python, platform, installed distributions and PyTorch runtime environment |
| Training experiments | 1 | Isolated lifecycle, preprocessing provenance, fold seeds and metrics, checkpoint selection and digest manifest |
| Training experiments | 2 | Runtime environment and effective deterministic execution settings |

## Metrics

Evaluation reports R2, RMSE, RRMSE, and MAE:
- **Characterization**: named per-parameter metrics and their unweighted
  aggregate. Characterizer training reports fold and cross-fold values;
  inference reports bootstrap confidence intervals.
- **Generation**: flattened across all time-steps and samples.

## Support

For questions or support, contact the development team at Koexai S.r.l.
