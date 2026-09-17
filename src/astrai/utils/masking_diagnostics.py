"""Ensemble, resolution and interpolation diagnostics for observation masking.

Run with ``python -m astrai.utils.masking_diagnostics --output-dir ...``.
No noise, training or survey calibration is performed. Realisations with no
observations are reported as failures, never redrawn or silently interpolated.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from astrai.utils.masking import (
    MaskingConfig, build_time_axis, generate_masking_realisation,
    interpolate_observations, resolve_masking_config,
)


STAGES = ('daylight', 'daylight_moon', 'daylight_moon_sun', 'all_components')


def observation_metrics(times, curve, mask):
    """Separate internal gaps, boundary fills and interpolation errors (dex)."""
    observed = times[mask]
    result = {'retained_count': int(mask.sum()), 'retained_fraction': float(mask.mean()),
              'max_internal_gap_days': np.nan, 'leading_gap_days': np.nan,
              'trailing_gap_days': np.nan, 'rmse_all_dex': np.nan,
              'rmse_missing_dex': np.nan, 'max_abs_error_dex': np.nan}
    if not observed.size:
        return result, np.full(curve.shape, np.nan)
    filled = interpolate_observations(curve, mask, times)
    error = filled - curve
    result.update(leading_gap_days=float(observed[0] - times[0]),
                  trailing_gap_days=float(times[-1] - observed[-1]),
                  rmse_all_dex=float(np.sqrt(np.mean(error**2))),
                  max_abs_error_dex=float(np.max(np.abs(error))))
    if (~mask).any():
        result['rmse_missing_dex'] = float(np.sqrt(np.mean(error[~mask]**2)))
    if observed.size >= 2:
        result['max_internal_gap_days'] = float(np.max(np.diff(observed)))
    return result, filled


def cloud_occupancy(realisation):
    """Exact fraction of physical time clouded, including boundary episodes."""
    boundaries = np.r_[0.0, realisation.cloud_transitions, realisation.horizon_days]
    cloudy = np.logical_xor(realisation.initially_cloudy,
                            np.arange(len(boundaries) - 1) % 2 == 1)
    return float(np.diff(boundaries)[cloudy].sum() / realisation.horizon_days)


def analyse(times, curve, *, seed=42, n_realisations=500, config=None):
    """Evaluate independent seeds and paired 1/day, 4/day grids on each path.

    Rows are independent realisations, not independent time samples. The
    component comparison adds exclusions in a fixed order using common
    thresholds; overlapping effects are not independent loss contributions.
    """
    times, curve = np.asarray(times), np.asarray(curve)
    if (times.ndim != 1 or len(times) < 2 or times.dtype.kind not in 'iuf'
            or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Times must start at zero and increase over a positive horizon')
    if curve.shape != times.shape or curve.dtype.kind not in 'iuf' or not np.isfinite(curve).all():
        raise ValueError('Curve must contain one finite real log10 luminosity per time sample')
    if isinstance(n_realisations, bool) or not isinstance(n_realisations, (int, np.integer)) or n_realisations < 1:
        raise ValueError('n_realisations must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    config = MaskingConfig.from_mapping(config)
    horizon = float(times[-1])
    coarse_times = np.arange(int(np.floor(horizon)) + 1, dtype=float)
    fine_times = np.arange(int(np.floor(4 * horizon)) + 1, dtype=float) / 4
    rows, resolutions, example = [], [], None
    for i in range(n_realisations):
        current_seed = int(seed) + i
        path = generate_masking_realisation(horizon, config, np.random.default_rng(current_seed))
        parts = path.evaluate(times)
        metrics, filled = observation_metrics(times, curve, parts['retained_mask'])
        row = {'seed': current_seed, **metrics,
               'mean_availability': float(parts['availability'].mean()),
               'cloud_time_fraction': cloud_occupancy(path),
               'cloud_sample_fraction': float(parts['cloud_blocked'].mean())}
        q = (24 - parts['daylight_hours']) / 24
        stages = [q, q - config.moon_loss_hours * parts['moon_contribution'] / 24]
        stages.append(stages[-1] * ~parts['sun_blocked'])
        stages.append(parts['availability'])
        for name, availability in zip(STAGES, stages):
            row[f'{name}_retained_fraction'] = float((parts['threshold'] < availability).mean())
        rows.append(row)
        coarse, fine = path.evaluate(coarse_times), path.evaluate(fine_times)
        resolutions.append({'seed': current_seed,
            'retained_fraction_1_per_day': float(coarse['retained_mask'].mean()),
            'retained_fraction_4_per_day': float(fine['retained_mask'].mean()),
            'common_time_mask_mismatches': int(np.count_nonzero(
                coarse['retained_mask'] != fine['retained_mask'][::4])),
            'common_time_max_availability_difference': float(np.max(np.abs(
                coarse['availability'] - fine['availability'][::4])))})
        if example is None:
            example = {'parts': parts, 'filled': filled, 'coarse': coarse, 'fine': fine}
    return rows, resolutions, example


def _hist(ax, values, label, **kwargs):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        ax.hist(values, bins=min(30, max(1, int(np.sqrt(values.size)))), label=label, **kwargs)
    else:
        ax.text(.5, .5, 'No defined values', ha='center', transform=ax.transAxes)
    ax.set_xlabel(label)
    ax.set_ylabel('Realisations')


def _save_plots(directory, times, curve, rows, resolutions, example, config, curve_label):
    column = lambda name: [row[name] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    _hist(axes[0, 0], column('retained_fraction'), 'Retained candidate fraction')
    axes[0, 0].axvline(np.mean(column('mean_availability')), color='black', ls='--', label='Mean availability')
    axes[0, 0].legend()
    _hist(axes[0, 1], column('cloud_time_fraction'), 'Clouded physical-time fraction')
    axes[0, 1].axvline(config.cloudy_fraction, color='black', ls='--', label='Nominal fraction')
    axes[0, 1].legend()
    _hist(axes[0, 2], column('max_internal_gap_days'), 'Largest internal gap (days)')
    boundaries = [np.asarray(column(name)) for name in ('leading_gap_days', 'trailing_gap_days')]
    boundaries = [values[np.isfinite(values)] for values in boundaries]
    if boundaries[0].size:
        limit = max(1.0, float(max(values.max() for values in boundaries)))
        bins = np.linspace(0, limit, 21)
        for values, label in zip(boundaries, ('Leading', 'Trailing')):
            axes[1, 0].hist(values, bins=bins, alpha=.6, label=label)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(.5, .5, 'No defined values', ha='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set(xlabel='Boundary gaps (days)', ylabel='Realisations')
    means = [np.mean(column(f'{name}_retained_fraction')) for name in STAGES]
    axes[1, 1].bar(range(4), means)
    axes[1, 1].set_xticks(range(4), ['Daylight', '+ Moon', '+ Sun', '+ Clouds'])
    axes[1, 1].set_ylabel('Mean retained fraction')
    axes[1, 1].set_title('Cumulative exclusions; shared thresholds')
    _hist(axes[1, 2], column('rmse_missing_dex'), 'Missing-point interpolation RMSE (dex)')
    zero = sum(row['retained_count'] == 0 for row in rows)
    fig.suptitle(f'{len(rows)} masking realisations; {zero} empty masks (not redrawn)\n'
                 f'{curve_label}; no added noise')
    fig.savefig(directory / 'ensemble.pdf', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
    parts, filled = example['parts'], example['filled']
    mask = parts['retained_mask']
    axes[0].plot(times, curve, label='Clean', color='black')
    axes[0].plot(times, filled, label='Interpolated', color='tab:orange')
    axes[0].scatter(times[mask], curve[mask], s=9, label='Retained', zorder=3)
    axes[0].set_ylabel('log10(L_bol / erg s^-1)')
    axes[0].legend()
    axes[1].plot(times, filled - curve)
    axes[1].axhline(0, color='black', lw=.5)
    axes[1].set_ylabel('Interpolation error (dex)')
    if mask.any():
        observed = times[mask]
        for ax in axes[:2]:
            ax.axvspan(times[0], observed[0], alpha=.12, color='red')
            ax.axvspan(observed[-1], times[-1], alpha=.12, color='red')
    else:
        axes[1].text(.5, .5, 'No observations: interpolation unavailable', ha='center', transform=axes[1].transAxes)
    axes[2].plot(times, parts['availability'], label='q(t)')
    axes[2].step(times, parts['threshold'], where='post', alpha=.6, label='U(t), evaluated at candidates')
    axes[2].scatter(times[mask], np.zeros(mask.sum()), marker='|', label='Retained')
    axes[2].set_ylabel('Availability / threshold')
    axes[2].set_xlabel('Days from first clean sample')
    axes[2].legend()
    fig.suptitle(f'{curve_label}; seed {rows[0]["seed"]}; no added noise\nRed shading: constant boundary fills')
    fig.savefig(directory / 'interpolation.pdf', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout='constrained')
    for parts, label, marker in [(example['fine'], '4/day', '.'), (example['coarse'], '1/day', 'o')]:
        select = parts['time_days'] <= min(30, times[-1])
        axes[0].plot(parts['time_days'][select], parts['retained_mask'][select].astype(float),
                     marker, label=label, alpha=.6, markerfacecolor='none' if marker == 'o' else None)
    axes[0].set(xlabel='Days from first clean sample (first 30 days)', ylabel='Retained mask', yticks=[0, 1])
    axes[0].legend()
    x = [row['retained_fraction_1_per_day'] for row in resolutions]
    y = [row['retained_fraction_4_per_day'] for row in resolutions]
    axes[1].scatter(x, y, s=10, alpha=.4)
    bounds = [min(x + y), max(x + y)]
    axes[1].plot(bounds, bounds, color='black', ls='--')
    axes[1].set(xlabel='Retained fraction at 1/day', ylabel='Retained fraction at 4/day')
    mismatch = sum(row['common_time_mask_mismatches'] for row in resolutions)
    fig.suptitle(f'Shared physical paths over {times[-1]:g} days; common-time mismatches: {mismatch}\n'
                 'Agreement at common times does not imply equal retained fractions')
    fig.savefig(directory / 'resolution.pdf', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def _write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: ('' if isinstance(value, float) and not np.isfinite(value) else value)
                         for key, value in row.items()} for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New or empty directory')
    parser.add_argument('--seed', type=int, default=42, help='First seed; subsequent realisations use seed + i')
    parser.add_argument('--n-realisations', type=int, default=500)
    parser.add_argument('--n-samples', type=int, default=421)
    parser.add_argument('--samples-per-day', type=float, default=1)
    parser.add_argument('--config', type=Path, help='ASTRAI YAML: read augmentation.masking only')
    parser.add_argument('--curve-file', type=Path, help='Clean log10 bolometric luminosity .npy, 1D or [curve, time]')
    parser.add_argument('--curve-index', type=int, default=0)
    args = parser.parse_args(argv)
    times = build_time_axis(args.n_samples, args.samples_per_day)
    config = resolve_masking_config(yaml.safe_load(args.config.read_text()) if args.config else {})
    if args.curve_file:
        raw = np.load(args.curve_file, allow_pickle=False, mmap_mode='r')
        if raw.ndim not in (1, 2):
            raise ValueError('Curve file must be a 1D curve or a 2D batch')
        count = 1 if raw.ndim == 1 else raw.shape[0]
        if not 0 <= args.curve_index < count:
            raise ValueError('curve-index is outside the curve file')
        curve = np.array(raw if raw.ndim == 1 else raw[args.curve_index], copy=True)
        source = {'kind': 'supplied_log10_curve', 'file': str(args.curve_file), 'index': args.curve_index}
        label = f'Input curve {args.curve_index}'
    else:
        if args.curve_index != 0:
            raise ValueError('curve-index requires curve-file')
        curve = 42 + 1.5 * np.exp(-.5 * ((times - 30) / 15)**2) - .002 * times
        source = {'kind': 'illustrative_synthetic_curve',
                  'formula': '42 + 1.5*exp(-0.5*((t-30)/15)**2) - 0.002*t'}
        label = 'Illustrative synthetic curve (not a fitted physical model)'
    source['float64_curve_sha256'] = hashlib.sha256(np.asarray(curve, dtype='<f8').tobytes()).hexdigest()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    rows, resolutions, example = analyse(times, curve, seed=args.seed,
        n_realisations=args.n_realisations, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / 'realisations.csv', rows)
    _write_rows(args.output_dir / 'resolution.csv', resolutions)
    parts = example['parts']
    names = ['time_days', 'clean_log10_luminosity', 'interpolated_log10_luminosity',
             'retained_mask', 'availability', 'threshold', 'daylight_hours',
             'moon_contribution', 'sun_blocked', 'cloud_blocked']
    arrays = [times, curve, example['filled']] + [parts[name] for name in names[3:]]
    np.savetxt(args.output_dir / 'example.csv', np.column_stack(arrays), delimiter=',',
               header=','.join(names), comments='')
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    rmse = np.array([row['rmse_missing_dex'] for row in rows])
    defined = np.isfinite(rmse)
    summary = {'diagnostic_version': 1, 'masking': config.record(), 'curve': source,
        'seed_start': args.seed, 'seed_policy': 'seed_plus_realisation_index',
        'n_realisations': args.n_realisations, 'n_samples': args.n_samples,
        'samples_per_day': args.samples_per_day, 'horizon_days': float(times[-1]),
        'noise_added': False, 'zero_observation_count': sum(row['retained_count'] == 0 for row in rows),
        'one_observation_count': sum(row['retained_count'] == 1 for row in rows),
        'mean_retained_fraction': mean('retained_fraction'),
        'mean_availability': mean('mean_availability'),
        'mean_cloud_time_fraction': mean('cloud_time_fraction'),
        'mean_cloud_sample_fraction': mean('cloud_sample_fraction'),
        'defined_missing_rmse_count': int(defined.sum()),
        'median_missing_rmse_dex': float(np.median(rmse[defined])) if defined.any() else None,
        'p90_missing_rmse_dex': float(np.quantile(rmse[defined], .9)) if defined.any() else None,
        'common_time_mask_mismatches': sum(row['common_time_mask_mismatches'] for row in resolutions),
        'common_time_max_availability_difference': max(row['common_time_max_availability_difference'] for row in resolutions),
        'undefined_csv_metrics': 'empty cell; example interpolation is NaN when no observations',
        'component_comparison': 'cumulative exclusions with shared thresholds; order dependent',
        'resolution_comparison': 'same path and horizon; integer and quarter-day candidates within horizon'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    _save_plots(args.output_dir, times, curve, rows, resolutions, example, config, label)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == '__main__':
    main()
