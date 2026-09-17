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

The Characterizer is trained on clean and augmented curves paired with the
same parameters. LCGen receives those duplicated parameters and is supervised
against the corresponding clean PCA-compressed curve in both cases.

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

Partitions original samples into outer-development/test and shared training/validation
pools before fitting scalers and PCA. Each invocation creates an isolated run; it does
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
    partitions.yaml                        # Shared sample assignments
    x_raw.npy                              # Original light curves
    y_physical.npy                         # Physical parameters
    y_transformed.npy                      # log1p physical parameters
    fold_1/
      bundle.yaml                          # Fit pool, policy and learned-state identity
      x_scaler.pkl, y_scaler.pkl, pca.pkl   # Fitted on this training subset only
      x_validation_pca.npy                 # Clean validation inputs
      y_validation_scaled.npy              # Validation parameters
      x_train_clean_pca.npy                 # Clean training curves (PCA space)
      x_train_aug_pca.npy                   # LSST-augmented training curves
      x_test_pca.npy                        # Test curves (PCA space)
      x_test_clean.npy                      # Test curves (original space)
      y_train_scaled.npy, y_test_scaled.npy # Scaled parameters
      y_test_transformed.npy                # Test parameters in model space
      y_test_physical.npy                   # Test parameters in physical space
      train_idx.npy, test_idx.npy           # Outer-development and outer-test
      training_idx.npy, validation_idx.npy  # Effective training and validation
    fold_2/
      ...
```

`metadata.yaml` records the artefact schema version, UTC start and completion
times, run status, configured random seed, fold list, Git commit, branch and
whether the working tree was dirty. It also records the digest, dtype and shape of
every NumPy artefact produced by a completed run. The current schema records
the seed-derivation scheme, the effective K-fold, PCA and per-fold augmentation
seeds, and a snapshot of the runtime environment. Its version history is
summarised under Experiment Tracking.

Persisted model arrays use `float32`; original-sample indices use `int64`.
Validation and test are clean. Failed runs remain marked as `failed` and are
never silently reused. A dataset digest identifies its contents and row order;
the canonical arrays are retained so the data remain recoverable.

Learned preprocessing must only use samples from the training pool of its
current role/fold. ASTRAI additionally chooses **clean-only fitting**: curve
scaling and PCA use clean training curves, and parameter scaling uses one
transformed parameter row per original training sample. Augmentations are
transformed afterwards and are additional Characterizer inputs; they never
contribute to the fit or become Generator targets. Clean-only fitting is an
ASTRAI policy, not a general requirement of cross-validation.

`partitioning.validation_fraction` (default `0.1`) controls the shared holdout.
Legacy training-section values remain accepted only when they agree with each
other and any shared value. At least two validation samples are reserved.
Both split stages use the same assignments even when run separately. The outer
K-fold assignment is unchanged; validation uses a new stage-independent stream.

The preprocessing interfaces also support these roles:

| Role | Fit pool | Excluded from fit |
| --- | --- | --- |
| Inner selection | Each K_select fold-training within outer-development | Its validation and the outer-test |
| Outer refit | Entire outer-development, with fresh objects | Outer-test |
| Final selection | Each K_select fold-training of the full admitted dataset | Its fold-validation |
| Final refit | Entire admitted dataset, with fresh objects | Any external data |

Optional `partitioning.selection_folds` defines the same K_select for outer
and final selection. When supplied, all assignments are recorded and PCA size
is checked on every selection training pool. No value is chosen implicitly.
These are partition and fit/transform interfaces: the executable trainers still
use holdout validation, epoch-based checkpoint selection and no outer/final
model refit. Their results do not estimate a selection-then-refit procedure.
Selection/refit preprocessing calls create fresh objects; no inner cache is
silently transferred. Callers can keep future inner bundles in memory. Current
precomputed holdout bundles remain in their isolated run for both stage consumers.

Each saved checkpoint carries its own fitted preprocessing. Independently
selected CV winners can have different parameter scalers and cannot then be
used for direct scaled handoff. For paired use, train both stages on the same
`test_fold` and shared preprocessing run. Compatibility checks reject mismatched
scaling and parameter semantics; no automatic conversion is performed. A CV
winner is not a final-refitted operational model.

### Phenomenological observation masking

`apply_lsst_pipeline` retains its historical API name, but now uses the
`phenomenological_masking_v1` recipe for incomplete bolometric observations.
It is not an LSST survey cadence, observing-geometry or multiband simulation.
Every clean-grid point is a candidate; no additional cadence or target number
of observations is imposed. The current historical noise kernel is unchanged.

The `augmentation.masking` mapping contains the parameters shown in the
supplied configurations. Missing fields use the same defaults in preprocessing,
Unified and diagnostics; unknown fields and invalid combinations are rejected.
Defaults are exploratory, not observationally calibrated:

- Daylight is an annual sinusoid of 12 +/- 2 equivalent hours, with random phase.
- A seasonal exclusion has a random centre and a duration drawn uniformly from
  0 to 90 days per curve. It repeats with a 365.25-day period.
- The lunar contribution follows the historical gated cosine, with period
  29.53 days and about 9.35 active days, removing up to four equivalent hours.
  The active window is a phenomenological duration, not a source-Moon angle.
- Weather alternates between exponentially distributed cloudy and clear spells.
  The mean cloudy duration is 2.3 days; 30% nominal cloudy occupancy implies a
  mean clear duration of about 5.37 days. The initial state is stationary.
  Individual curves need not lose exactly 30%, and spells are not all 2.3 days.

With daylight `H`, lunar contribution `M`, and clear solar/weather indicators
`S,C`, availability is `q = S*C*(24-H-moon_loss_hours*M)/24`. Solar and cloudy
intervals force zero availability. Availability is not renormalised over the
curve, so worse conditions can reduce the number of observations.

One uniform threshold is drawn per physical one-day interval, with a random
interval offset. A candidate at time `t` is retained exactly when `U(t) < q(t)`.
All draws occur when creating a temporal realisation; evaluating that realisation
on another grid consumes no random numbers. The public `generate_masking_realisation`
and `MaskingRealisation.evaluate` APIs allow the same path to be inspected at
multiple resolutions. On the same physical horizon and initial RNG state,
1/day masks equal every fourth element of the corresponding 4/day masks.
This does not guarantee equal observation counts, sampled gaps or interpolated
curves, nor invariance to changed horizons, batch regrouping or noise RNG use.

`data.n_days` is the number of samples. Time in days is `arange(n_days) /
samples_per_day`, starting at the first clean sample; the canonical default is
one sample/day. Declare the rate explicitly: 421 points at 1/day cover 420 days,
whereas 1,601 points at 4/day cover 400 days. Cloud durations and threshold
intervals never use a global digital sampling constant.

Only retained values contribute to interpolation, linearly in `log10(L_bol)`.
Missing edges use the nearest observed endpoint. One observation gives a
constant curve; zero observations raise an error identifying the batch row.
There is no automatic mask redraw or fallback to hidden clean/noisy values.
Short horizons can therefore be entirely unobserved. Model-facing arrays remain
float32 and returned observation masks are boolean. Clean-only preprocessing
fits, clean Generator targets and sample partitions are unchanged.

The component diagnostic uses the same recipe and actual retained points:

```bash
python -m astrai.utils.lsst --seed 42 --n-samples 421 --samples-per-day 1 --output masking.pdf
```

It retains the seven-panel structure of the published illustration, distinguishing
continuous availability from the realised boolean mask. Sampling gaps are measured
from retained candidates; neither 3-4-day gaps nor the paper's geometric thresholds
are enforced. The historical cumulative-budget sampling helpers are not called.
`plot-results` records the effective recipe and seed in `augmentation_metadata.json`;
its newly generated corruption is not a reproduction of a historical observing mask.

Preprocessing schema 7 identifies the effective masking parameters and interpolation
policy in `view_configuration`; training metadata version 6 records the same identity.
Precomputed training views without this identity, including schema 6 runs, must be
regenerated. Missing configuration fields use the new documented recipe, not the
historical masking. Historical target/checkpoint metadata remain readable under their
original contracts, without claiming that old augmentation used this recipe.

### Additional masking diagnostics

The optional report command evaluates many independent masking realisations,
without adding noise or running a model. It leaves the augmentation recipe
unchanged. For the usual candidate grids:

```bash
MPLBACKEND=Agg python -m astrai.utils.masking_diagnostics \
  --seed 42 --n-realisations 500 --n-samples 421 --samples-per-day 1 \
  --output-dir masking_diagnostics_421

MPLBACKEND=Agg python -m astrai.utils.masking_diagnostics \
  --seed 42 --n-realisations 500 --n-samples 1601 --samples-per-day 4 \
  --output-dir masking_diagnostics_1601
```

Each command creates three plots and their numerical data:

- `ensemble.pdf`: retained fraction, exact clouded physical-time fraction,
  maximum internal gaps, leading/trailing boundary gaps, cumulative component
  exclusions and interpolation error over missing samples.
- `resolution.pdf`: the **same physical paths** evaluated at 1/day and 4/day,
  with the number of disagreements at common times. Fractions of retained
  candidates can differ even when every common-time decision agrees.
- `interpolation.pdf`: the first seed's clean curve, retained points, filled
  curve and residuals, with constant boundary fills shaded separately.
- `realisations.csv`, `resolution.csv`, `example.csv` and `summary.json`:
  per-realisation metrics, plotted numerical values, seeds, resolved recipe,
  grid, curve identity and empty/single-observation counts.

Use a **new or empty** output directory. The default curve is an explicitly
labelled illustrative synthetic example, not a semi-analytical model or an
observational calibration. To assess an actual clean bolometric curve, append:

```bash
--curve-file path/to/x_raw.npy --curve-index 0
```

The file must contain unscaled `log10(L_bol)` values, either one 1D curve or a
2D `[curve, time]` array matching `--n-samples`. Do not supply PCA coefficients,
standardised values or linear luminosities. `--config path/to/config.yaml`
reads only `augmentation.masking`; grid settings remain explicit CLI options.
All realisations in a report mask the **same selected curve**: interpolation
error statistics measure its sensitivity to missing observations, not model
accuracy or population-wide performance. Seeds are `seed + i`, including the
unmodified first seed. The diagnostic does not consume the pipeline's noise
RNG, so it is not a reproduction of a training augmentation with the same seed.

Cloud occupancy is measured from physical episode durations, including the
parts at both boundaries; it is not estimated by counting cloudy candidates.
The nominal fraction is an ensemble expectation. The component chart adds
Daylight, Moon, Sun and Clouds in that order with shared thresholds. Its losses
are cumulative and order-dependent because exclusions overlap.

Zero-observation masks are counted and not redrawn; their interpolation and
associated error/gap statistics are undefined. A single observation produces
a constant fill with no internal-gap statistic. With no missing samples,
missing-point RMSE is undefined. Undefined CSV metrics are empty cells;
example interpolation uses NaN when unavailable and JSON summaries use null.
The core augmentation pipeline still raises on zero observations.

The 421/1 and 1601/4 reports span 420 and 400 days respectively and should not
be compared as paired realisations. Each report performs its own resolution
comparison within its physical horizon (421 vs 1681, or 401 vs 1601 points).
The number of physical model parameters does not enter the masking recipe.
These reports assess missing-data behaviour; they do not validate survey
cadence, observed bolometric population statistics or downstream model quality.

### Direct bolometric noise kernels

`astrai.utils.augmentation` exposes two standalone NumPy kernels. Both accept
a 1D curve or 2D batch in `log10(L_bol / [erg s^-1])` and return a new float64
array in the same space and shape:

* `add_iid_gaussian_noise_in_log10_luminosity(x, sigma_dex=..., rng=...)`
  adds independent zero-mean Gaussian noise in dex. This is an empirical
  robustness baseline and diagnostic tool, equivalent to multiplicative
  lognormal noise in luminosity. It preserves the log-space mean and linear
  median, not the linear mean.
* `add_heteroscedastic_noise_in_normalised_luminosity(x, a=..., b=...,
  x_ref=42.0, rng=...)` uses `f = 10**(x - x_ref)` and Gaussian proposals with
  mean `f` and variance `a*f + b`. Only non-positive proposals are resampled,
  giving a Gaussian conditioned on positivity, with no clipping floor.

`x_ref=42.0` is a **normalisation convention**, not a physically calibrated
value. The numerical coefficients `a` and `b` depend on that convention.
The source-dependent variance term `a*f` and constant term `b` form an
effective bolometric augmentation model, not a calibrated LSST detector
simulation. For exploratory use, `a=0.01`, `b=0.0004` at `x_ref=42` give a
nominal relative standard deviation of about 10.2% at `f=1`. These are
illustrative, uncalibrated values, not default amplitudes.

Positivity conditioning changes both mean and variance at low signal-to-noise:
`a*f+b` describes the proposal variance. As `f` tends to zero with `b>0`, the
output mean tends to `sqrt(b)*sqrt(2/pi)`. Very faint input values can therefore
be strongly perturbed. A numeric log10 value of zero is not treated as missing
data. Invalid or unrepresentable arithmetic raises an error; there is no
silent numerical floor. Positive sampling is bounded at 128 draws per element.

Pass a local `numpy.random.Generator` for repeatability with identical inputs,
order and parameters. Resampling means that changing batch partitioning need
not preserve individual draws. Zero amplitude returns an identical copy
without consuming the generator; neither kernel modifies NumPy's global RNG.

These kernels are direct APIs only: no YAML selector is provided and the
new noise kernels are not selected by the training or diagnostic pipeline. The three
older functions (`add_gaussian_noise`, `add_gaussian_noise_slow` and
`add_exp_gaussian_log_noise`) are deprecated for new use but retain their
signatures and numerical behaviour for historical reproduction and backwards
compatibility, without runtime deprecation warnings. The current pipeline
still calls `add_gaussian_noise`.

### Target representation

ASTRAI uses one explicit target contract throughout preprocessing, training,
inference, diagnostics and benchmarks:

```
physical (finite and non-negative) -> log1p -> transformed -> StandardScaler -> scaled
```

The inverse path removes target scaling and then applies `expm1`. Zero is a
valid physical parameter value. Model predictions are not clipped when they
are decoded, so extrapolation remains visible. New configurations record
`data.target_transform: log1p`; configurations from before this field was
introduced retain `log1p` as their compatibility default.

New preprocessing runs use artefact schema 7. New training requires these
pool-specific bundles and verified sample assignments: regenerate global or
metadata-free preprocessing before training. Historical experiments remain
readable with their original semantics; diagnostic target readers retain schema
5 and metadata-free compatibility. Schemas 1--4 have ambiguous target metadata.

### Reproducibility

The configured `random_seed` is the base for independent deterministic
streams. Stable NumPy `SeedSequence` namespaces derive separate seeds for PCA,
each fold's augmentation, model initialisation and DataLoader shuffling. The
validation split is shared by both model stages. Selection/refit roles have
separate derived streams, while existing model and DataLoader streams remain
unchanged. The
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
final report preserves all existing unweighted aggregate test metrics and also
shows the mean and standard deviation of every per-parameter test metric across
folds; a configured `test_fold` produces a single-fold report instead. The
legacy `held_out_fold` spelling remains accepted when `test_fold` is absent.

Checkpoint selection uses validation data only. Its default `R2` score remains
the historical unweighted aggregate in transformed target space after
reversing target standardisation. Per-parameter validation and test metrics are
reported both in transformed space and in physical units; physical RMSE and
MAE are not averaged across parameters with heterogeneous units. Test data is
reserved exclusively for final performance estimation and never participates
in epoch or fold selection.

### Generator Training (`astrai train-generator`)

Trains a `MLPWithResiduals` to reconstruct clean PCA-compressed curves from
scaled transformed physical parameters. Each training sample contributes one
parameter-to-clean-curve pair.

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

The generic `infer` and `infer-split` commands, as well as single and batch
real-data inference, display and persist parameter predictions in physical
space. Direct PPReg-to-LCGen handoff remains in scaled transformed space and
is accepted only when the two target scalers and transformation contracts are
compatible.

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
    --prep preprocessed/YYYYMMDD_HHMMSS_default_split \
    --fold 1 \
    --output-dir plots/
```

3-panel reconstruction view for the unified model (original vs augmented vs reconstructed):

```bash
astrai visualize-reconstruction \
    --exp experiments/YYYYMMDD_HHMMSS_microseconds \
    --top 5
```

Options: `--index N` for a specific sample, `--top N` for the N best by
transformed-space characterisation RMSE.

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
| `data` | `format`, `target_transform`, `n_days`, `n_params`, `param_names`, `samples_per_day` |
| `preprocessing` | `pca_components` (32), `n_splits` (K-Fold), `random_seed` |
| `augmentation` | `noise_std` (0.05), `masking` (phenomenological defaults above) |
| `characterizer` | `model` (width, depth, dropout), `training` (`test_fold`, `batch_size`, `epochs`, `learning_rate`, validation and selection controls) |
| `generator` | `model` (width, depth, dropout), `training` (`test_fold`, `batch_size`, `epochs`, `learning_rate`, validation and selection controls) |

### `configs/default.yaml` (unified model)

| Section | Key Parameters |
|---------|---------------|
| `data` | Same as above |
| `model` | `pca_components`, `width`, `depth`, `dropout` |
| `training` | `batch_size`, `epochs`, `learning_rate`, `n_splits`, `random_seed`, validation and selection controls |
| `loss` | `alpha_char`, `alpha_gen` (loss weights) |

`partitioning.validation_fraction` sets the shared validation split. Legacy
training-section fractions must agree. Every training section accepts
`checkpoint_selection.metric`. Supported selection metrics are `R2` (the
default, maximised), `RMSE`, `RRMSE` and `MAE` (minimised). The metric space is
fixed by the model contract: transformed aggregate characterisation for the
Characterizer and unified model, and flattened light-curve space for the
Generator.

`epochs` is always the maximum epoch count. `early_stopping.enabled` defaults
to `false`, so all configured epochs run while the best validation epoch is
still restored afterwards. When enabled, `patience` controls interruption and
`min_delta` controls only whether an improvement resets patience. Checkpoint
selection itself uses every strict improvement and retains the first epoch on
a tie.

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
    best_characterizer.pth       # Validation-selected model weights
    best_char_x_scaler.pkl       # Feature scaler
    best_char_y_scaler.pkl       # Target scaler
    best_char_pca.pkl            # PCA transformer
    code.zip                     # Source code snapshot
    config.yaml                  # Effective configuration
    preprocessing_metadata.yaml  # Input preprocessing provenance
    metadata.yaml                # Lifecycle, results and artefact manifest
    fold_1/
      training_indices.npy       # Rows used for optimisation
      validation_indices.npy     # Rows used for model selection
      test_indices.npy           # Rows used only for final estimation
      training_trace.csv         # Per-epoch loss and selection score
  generator/YYYYMMDD_HHMMSS_microseconds/
    ...
```

The experiment metadata is written when a run starts, after every completed
fold and whenever the best checkpoint changes. It records the run stage and
status, UTC times, Git revision and dirty state, effective configuration,
device, parameter order, dtype contract, selected folds, base and derived
seeds, explicit data indices, validation and test metrics, checkpoint selection
and SHA-256 digests for every persisted artefact. The runtime snapshot includes
Python, the operating-system platform, installed distribution versions,
PyTorch backends, thread counts and effective deterministic settings. Failures
retain their error and any partial fold records. Characterizer and Generator
runs started by `astrai pipeline` share a
`pipeline_run_id`, and each snapshots the metadata of its completed
preprocessing input. New split training validates canonical data identity, shared
partitions and each fold bundle before optimisation. Model reload validates the
selected checkpoint and its preprocessing association. Unified runs retain
canonical source arrays and their partition plan alongside the experiment.

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
| Preprocessing artefacts | 5 | Explicit physical, transformed and scaled target artefacts and target-transformation contract |
| Preprocessing artefacts | 6 | Shared partitions, training-only bundles, dataset identity, array digests and explicit fit policy |
| Preprocessing artefacts | 7 | Effective masking recipe, physical-time parameters and interpolation policy for precomputed views |
| Training experiments | 1 | Isolated lifecycle, preprocessing provenance, fold seeds and metrics, checkpoint selection and digest manifest |
| Training experiments | 2 | Runtime environment and effective deterministic execution settings |
| Training experiments | 3 | Target-transformation contract and explicit metric-space metadata |
| Training experiments | 4 | Explicit train/validation/test indices, validation-based epoch and fold selection, early-stopping evidence and unambiguous selected-checkpoint metadata |
| Training experiments | 5 | Fold preprocessing provenance, selected checkpoint association and recoverable unified preprocessing sources |
| Training experiments | 6 | Effective augmentation view configuration, including the masking recipe |

## Metrics

Evaluation reports R2, RMSE, RRMSE, and MAE:
- **Characterization**: named per-parameter validation and test metrics in
  transformed and physical spaces. The historical unweighted aggregate and
  validation-only checkpoint selection remain in transformed space; inference
  reports bootstrap confidence intervals.
- **Generation**: flattened across all time-steps and samples.

## Support

For questions or support, contact the development team at Koexai S.r.l.
