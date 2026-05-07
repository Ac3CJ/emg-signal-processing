"""
AblationMetricsCollator.py

Collects per-trial validation metrics and writes per-ablation summary CSVs.

Usage:
    python AblationMetricsCollator.py
    python AblationMetricsCollator.py --prefixes step window augment filter
    python AblationMetricsCollator.py --smooth 02

Inputs (raw, default):
    ./neural-network-models/<run>/validation/c_all_metrics.csv
    ./neural-network-models/<run>/validation/s_all_metrics.csv

Inputs (--smooth <tag>):
    ./neural-network-models/<run>/validation/smooth{tag}_c_all_metrics.csv
    ./neural-network-models/<run>/validation/smooth{tag}_s_all_metrics.csv

If the primary aggregate files are missing, the script falls back to individual
*_metrics.csv files — filtered to the same prefix (raw or smooth{tag}) to prevent
mixing.
"""

import argparse
import csv
import os
import re

import numpy as np

METRIC_COLUMNS = [
    'Yaw_r2',
    'Pitch_r2',
    'Yaw_rmse',
    'Pitch_rmse',
    'mean_abs_latency_ms',
]

DERIVED_COLUMNS = [
    'avg_r2',
    'avg_rmse',
]

CATEGORY_ORDER = [
    'collected',
    'secondary',
    'collected_benchmark',
    'secondary_benchmark',
    'overall',
]


def _resolve_root(path):
    return os.path.abspath(path)


def _validation_dir(root_dir, run_name):
    return os.path.join(root_dir, run_name, 'validation')


def _list_run_dirs(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )


def _parse_float(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == '' or text.lower() == 'nan':
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _load_metrics_file(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {'trial': row.get('trial', '')}
            for key in METRIC_COLUMNS:
                entry[key] = _parse_float(row.get(key))
            rows.append(entry)
    return rows


def _load_metrics_rows(validation_dir, smooth_tag=None):
    rows = []

    if smooth_tag is not None:
        primary_names = (
            f'smooth{smooth_tag}_c_all_metrics.csv',
            f'smooth{smooth_tag}_s_all_metrics.csv',
        )
    else:
        primary_names = ('c_all_metrics.csv', 's_all_metrics.csv')

    for name in primary_names:
        path = os.path.join(validation_dir, name)
        if os.path.exists(path):
            rows.extend(_load_metrics_file(path))

    if rows:
        return rows

    if not os.path.isdir(validation_dir):
        return []

    always_skip = {
        'c_all_metrics.csv',
        's_all_metrics.csv',
        'c_benchmark_metrics.csv',
        's_predefined_metrics.csv',
    }
    smooth_prefix = f'smooth{smooth_tag}_' if smooth_tag is not None else None

    for name in os.listdir(validation_dir):
        if name in always_skip:
            continue
        if not name.endswith('_metrics.csv'):
            continue
        if smooth_tag is not None:
            if not name.startswith(smooth_prefix):
                continue
        else:
            if name.startswith('smooth'):
                continue
        rows.extend(_load_metrics_file(os.path.join(validation_dir, name)))

    return rows


def _parse_trial_name(trial_name):
    match = re.match(r'([cs])P(\d+)M(\d+)', trial_name)
    if not match:
        return None
    data_type = match.group(1)
    participant = int(match.group(2))
    movement = int(match.group(3))
    return data_type, participant, movement


def _filter_rows(rows, data_type=None, participants=None):
    for row in rows:
        trial = row.get('trial', '')
        parsed = _parse_trial_name(trial)
        if parsed is None:
            continue
        dtype, participant, _ = parsed
        if data_type is not None and dtype != data_type:
            continue
        if participants is not None and participant not in participants:
            continue
        yield row


def _average_metrics(rows):
    if not rows:
        return {k: np.nan for k in METRIC_COLUMNS}

    averages = {}
    for key in METRIC_COLUMNS:
        values = [r.get(key, np.nan) for r in rows]
        mean_val = np.nanmean(values) if len(values) else np.nan
        averages[key] = float(mean_val) if not np.isnan(mean_val) else np.nan
    return averages


def _iqr_metrics(rows):
    if not rows:
        return {f'{k}_iqr': np.nan for k in METRIC_COLUMNS}

    iqrs = {}
    for key in METRIC_COLUMNS:
        values = [r.get(key, np.nan) for r in rows]
        valid = [v for v in values if not np.isnan(v)]
        if len(valid) >= 2:
            q75, q25 = np.percentile(valid, [75, 25])
            iqrs[f'{key}_iqr'] = float(q75 - q25)
        else:
            iqrs[f'{key}_iqr'] = np.nan
    return iqrs


def _format_value(key, value):
    if value is None or np.isnan(value):
        return 'NaN'
    if key.endswith('mean_abs_latency_ms'):
        return f'{value:.2f}'
    return f'{value:.4f}'


def _summarize_run(run_name, rows, collected_benchmark, secondary_benchmark):
    collected_rows = list(_filter_rows(rows, data_type='c'))
    secondary_rows = list(_filter_rows(rows, data_type='s'))
    collected_bench = list(_filter_rows(rows, data_type='c', participants=collected_benchmark))
    secondary_bench = list(_filter_rows(rows, data_type='s', participants=secondary_benchmark))
    overall_rows = collected_rows + secondary_rows

    category_rows = {
        'collected': collected_rows,
        'secondary': secondary_rows,
        'collected_benchmark': collected_bench,
        'secondary_benchmark': secondary_bench,
        'overall': overall_rows,
    }

    row = {'test_name': run_name}
    for category in CATEGORY_ORDER:
        cat_rows = category_rows[category]
        avgs = _average_metrics(cat_rows)
        iqrs = _iqr_metrics(cat_rows)

        for key in METRIC_COLUMNS:
            row[f'{category}_{key}'] = avgs.get(key, np.nan)

        yaw_r2 = avgs.get('Yaw_r2', np.nan)
        pitch_r2 = avgs.get('Pitch_r2', np.nan)
        yaw_rmse = avgs.get('Yaw_rmse', np.nan)
        pitch_rmse = avgs.get('Pitch_rmse', np.nan)
        row[f'{category}_avg_r2'] = float(np.nanmean([yaw_r2, pitch_r2])) if not np.isnan(yaw_r2) or not np.isnan(pitch_r2) else np.nan
        row[f'{category}_avg_rmse'] = float(np.nanmean([yaw_rmse, pitch_rmse])) if not np.isnan(yaw_rmse) or not np.isnan(pitch_rmse) else np.nan

        for key in METRIC_COLUMNS:
            row[f'{category}_{key}_iqr'] = iqrs.get(f'{key}_iqr', np.nan)

        if cat_rows:
            per_trial_avg_r2 = [float(np.nanmean([r.get('Yaw_r2', np.nan), r.get('Pitch_r2', np.nan)])) for r in cat_rows]
            per_trial_avg_rmse = [float(np.nanmean([r.get('Yaw_rmse', np.nan), r.get('Pitch_rmse', np.nan)])) for r in cat_rows]
            valid_r2 = [v for v in per_trial_avg_r2 if not np.isnan(v)]
            valid_rmse = [v for v in per_trial_avg_rmse if not np.isnan(v)]
            row[f'{category}_avg_r2_iqr'] = float(np.percentile(valid_r2, 75) - np.percentile(valid_r2, 25)) if len(valid_r2) >= 2 else np.nan
            row[f'{category}_avg_rmse_iqr'] = float(np.percentile(valid_rmse, 75) - np.percentile(valid_rmse, 25)) if len(valid_rmse) >= 2 else np.nan
        else:
            row[f'{category}_avg_r2_iqr'] = np.nan
            row[f'{category}_avg_rmse_iqr'] = np.nan

    return row


def _summarize_run_movement(run_name, rows, collected_benchmark, secondary_benchmark):
    by_movement = {}
    for row in rows:
        parsed = _parse_trial_name(row.get('trial', ''))
        if parsed is None:
            continue
        _, _, movement = parsed
        by_movement.setdefault(movement, []).append(row)

    result = []
    for movement in sorted(by_movement):
        movement_rows = by_movement[movement]
        collected_rows = list(_filter_rows(movement_rows, data_type='c'))
        secondary_rows = list(_filter_rows(movement_rows, data_type='s'))
        collected_bench = list(_filter_rows(movement_rows, data_type='c', participants=collected_benchmark))
        secondary_bench = list(_filter_rows(movement_rows, data_type='s', participants=secondary_benchmark))

        category_rows = {
            'collected': collected_rows,
            'secondary': secondary_rows,
            'collected_benchmark': collected_bench,
            'secondary_benchmark': secondary_bench,
            'overall': movement_rows,
        }

        mov_row = {'test_name': run_name, 'movement': movement}
        for category in CATEGORY_ORDER:
            cat_rows = category_rows[category]
            avgs = _average_metrics(cat_rows)
            iqrs = _iqr_metrics(cat_rows)

            for key in METRIC_COLUMNS:
                mov_row[f'{category}_{key}'] = avgs.get(key, np.nan)

            yaw_r2 = avgs.get('Yaw_r2', np.nan)
            pitch_r2 = avgs.get('Pitch_r2', np.nan)
            yaw_rmse = avgs.get('Yaw_rmse', np.nan)
            pitch_rmse = avgs.get('Pitch_rmse', np.nan)
            mov_row[f'{category}_avg_r2'] = float(np.nanmean([yaw_r2, pitch_r2])) if not np.isnan(yaw_r2) or not np.isnan(pitch_r2) else np.nan
            mov_row[f'{category}_avg_rmse'] = float(np.nanmean([yaw_rmse, pitch_rmse])) if not np.isnan(yaw_rmse) or not np.isnan(pitch_rmse) else np.nan

            for key in METRIC_COLUMNS:
                mov_row[f'{category}_{key}_iqr'] = iqrs.get(f'{key}_iqr', np.nan)

            if cat_rows:
                per_trial_avg_r2 = [float(np.nanmean([r.get('Yaw_r2', np.nan), r.get('Pitch_r2', np.nan)])) for r in cat_rows]
                per_trial_avg_rmse = [float(np.nanmean([r.get('Yaw_rmse', np.nan), r.get('Pitch_rmse', np.nan)])) for r in cat_rows]
                valid_r2 = [v for v in per_trial_avg_r2 if not np.isnan(v)]
                valid_rmse = [v for v in per_trial_avg_rmse if not np.isnan(v)]
                mov_row[f'{category}_avg_r2_iqr'] = float(np.percentile(valid_r2, 75) - np.percentile(valid_r2, 25)) if len(valid_r2) >= 2 else np.nan
                mov_row[f'{category}_avg_rmse_iqr'] = float(np.percentile(valid_rmse, 75) - np.percentile(valid_rmse, 25)) if len(valid_rmse) >= 2 else np.nan
            else:
                mov_row[f'{category}_avg_r2_iqr'] = np.nan
                mov_row[f'{category}_avg_rmse_iqr'] = np.nan

        result.append(mov_row)

    return result


def _group_by_prefix(run_names):
    grouped = {}
    for name in run_names:
        prefix = name.split('_', 1)[0] if '_' in name else name
        grouped.setdefault(prefix, []).append(name)
    for prefix in grouped:
        grouped[prefix].sort()
    return grouped


def _write_summary_csv(output_path, rows):
    header = ['test_name']
    for category in CATEGORY_ORDER:
        header.extend([f'{category}_{key}' for key in METRIC_COLUMNS])
        header.extend([f'{category}_{key}' for key in DERIVED_COLUMNS])
        header.extend([f'{category}_{key}_iqr' for key in METRIC_COLUMNS])
        header.extend([f'{category}_{key}_iqr' for key in DERIVED_COLUMNS])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row['test_name'],
                *[_format_value(key, row.get(key)) for key in header[1:]],
            ])


def _write_movement_summary_csv(output_path, rows):
    header = ['test_name', 'movement']
    for category in CATEGORY_ORDER:
        header.extend([f'{category}_{key}' for key in METRIC_COLUMNS])
        header.extend([f'{category}_{key}' for key in DERIVED_COLUMNS])
        header.extend([f'{category}_{key}_iqr' for key in METRIC_COLUMNS])
        header.extend([f'{category}_{key}_iqr' for key in DERIVED_COLUMNS])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row['test_name'],
                row['movement'],
                *[_format_value(key, row.get(key)) for key in header[2:]],
            ])


def main():
    parser = argparse.ArgumentParser(
        description='Collate validation metrics across ablation runs into summary CSVs.'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='./neural-network-models',
        help='Root folder containing run subdirectories (default: ./neural-network-models).',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to write summary CSVs (default: <root>/summary-csv).',
    )
    parser.add_argument(
        '--prefixes',
        nargs='*',
        default=None,
        help='Optional list of ablation prefixes to include (e.g., step window augment filter).',
    )
    parser.add_argument(
        '--collected_benchmark',
        nargs='*',
        type=int,
        default=[3, 6, 7, 9],
        help='Collected benchmark participants (default: 3 6 7 9).',
    )
    parser.add_argument(
        '--secondary_benchmark',
        nargs='*',
        type=int,
        default=[8],
        help='Secondary benchmark participants (default: 8).',
    )
    parser.add_argument(
        '--smooth',
        type=str,
        default=None,
        metavar='TAG',
        help='Smooth tag to collate (e.g. "02"). Uses smooth{TAG}_c/s_all_metrics.csv as primary source. Default: raw files only.',
    )
    args = parser.parse_args()

    root_dir = _resolve_root(args.root)
    output_dir = args.output_dir or os.path.join(root_dir, 'summary-csv')
    os.makedirs(output_dir, exist_ok=True)

    smooth_tag = args.smooth

    run_names = _list_run_dirs(root_dir)
    grouped = _group_by_prefix(run_names)

    if args.prefixes:
        grouped = {k: v for k, v in grouped.items() if k in set(args.prefixes)}

    if not grouped:
        print(f'No runs found under {root_dir}.')
        return

    for prefix, runs in grouped.items():
        summary_rows = []
        movement_rows = []
        for run_name in runs:
            validation_dir = _validation_dir(root_dir, run_name)
            rows = _load_metrics_rows(validation_dir, smooth_tag=smooth_tag)
            if not rows:
                print(f'[WARN] No metrics found for {run_name} in {validation_dir}')
                continue
            collected_bench = set(args.collected_benchmark)
            secondary_bench = set(args.secondary_benchmark)
            summary_rows.append(
                _summarize_run(run_name, rows, collected_bench, secondary_bench)
            )
            movement_rows.extend(
                _summarize_run_movement(run_name, rows, collected_bench, secondary_bench)
            )

        if not summary_rows:
            print(f'[WARN] No summary rows generated for prefix {prefix}.')
            continue

        output_path = os.path.join(output_dir, f'{prefix}_summary.csv')
        _write_summary_csv(output_path, summary_rows)
        print(f'[OK] Wrote {output_path}')

        if movement_rows:
            movement_path = os.path.join(output_dir, f'{prefix}_movement_summary.csv')
            _write_movement_summary_csv(movement_path, movement_rows)
            print(f'[OK] Wrote {movement_path}')


if __name__ == '__main__':
    main()
