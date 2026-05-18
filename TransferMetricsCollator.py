"""
TransferMetricsCollator.py

Collates per-trial validation metrics for transfer learning runs (tfFroz and
tfNotFroz) and writes two long-format summary CSVs for comparison.

Each run contributes only its own participant's rows (e.g. tfFroz_cP1 →
cP1M1–cP1M9 only). Movement M10 is excluded as a calibration test.

Outputs:
    <output_dir>/tf_participant_summary.csv   — mean + IQR per (participant × method)
    <output_dir>/tf_movement_summary.csv      — mean + IQR per (movement × method)

Usage:
    python TransferMetricsCollator.py
    python TransferMetricsCollator.py --root ./neural-network-models --output_dir ./neural-network-models/summary-csv
"""

import argparse
import csv
import os
import re
import warnings

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


# ---------------------------------------------------------------------------
# Helpers copied verbatim from AblationMetricsCollator.py
# ---------------------------------------------------------------------------

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


def _average_metrics(rows):
    if not rows:
        return {k: np.nan for k in METRIC_COLUMNS}

    averages = {}
    for key in METRIC_COLUMNS:
        values = [r.get(key, np.nan) for r in rows]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
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


def _parse_trial_name(trial_name):
    match = re.match(r'([cs])P(\d+)M(\d+)', trial_name)
    if not match:
        return None
    data_type = match.group(1)
    participant = int(match.group(2))
    movement = int(match.group(3))
    return data_type, participant, movement


def _format_value(key, value):
    if value is None or np.isnan(value):
        return 'NaN'
    if key.endswith('mean_abs_latency_ms'):
        return f'{value:.2f}'
    return f'{value:.4f}'


# ---------------------------------------------------------------------------
# Transfer-learning-specific logic
# ---------------------------------------------------------------------------

def _parse_run_name(dir_name):
    """Extract (method_tag, data_type, participant_id) from a tf run directory name.

    Examples:
        tfFroz_cP1   -> ('tfFroz', 'c', 1)
        tfNotFroz_sP8 -> ('tfNotFroz', 's', 8)

    Returns None if the name does not match the expected pattern.
    """
    match = re.match(r'^(tfFroz|tfNotFroz)_([cs])P(\d+)$', dir_name)
    if not match:
        return None
    method_tag = match.group(1)
    data_type = match.group(2)
    participant_id = int(match.group(3))
    return method_tag, data_type, participant_id


def _load_tf_run_rows(root_dir, run_name, participant_id, quiet=False):
    """Load and participant-filter rows from a tf run's validation directory.

    Only rows whose parsed participant_id matches the run's own participant are
    returned. M10 rows are excluded. The data_type in the trial filename is
    ignored for filtering — sP8 runs store their participant as cP8.

    Falls back to per-trial *_metrics.csv files if aggregate files are absent.
    """
    validation_dir = os.path.join(root_dir, run_name, 'validation')

    raw_rows = []
    for agg_name in ('c_all_metrics.csv', 's_all_metrics.csv'):
        path = os.path.join(validation_dir, agg_name)
        if os.path.exists(path):
            raw_rows.extend(_load_metrics_file(path))

    if not raw_rows:
        if not os.path.isdir(validation_dir):
            return []
        always_skip = {
            'c_all_metrics.csv',
            's_all_metrics.csv',
            'c_benchmark_metrics.csv',
            's_predefined_metrics.csv',
        }
        for name in os.listdir(validation_dir):
            if name in always_skip:
                continue
            if not name.endswith('_metrics.csv'):
                continue
            if name.startswith('smooth'):
                continue
            raw_rows.extend(_load_metrics_file(os.path.join(validation_dir, name)))

    filtered = []
    for row in raw_rows:
        parsed = _parse_trial_name(row.get('trial', ''))
        if parsed is None:
            continue
        _, p_id, movement = parsed
        if p_id != participant_id:
            continue
        if movement == 10:
            continue
        filtered.append(row)

    return filtered


def _derive_avg(avgs):
    """Compute avg_r2 and avg_rmse from an averages dict."""
    yaw_r2 = avgs.get('Yaw_r2', np.nan)
    pitch_r2 = avgs.get('Pitch_r2', np.nan)
    yaw_rmse = avgs.get('Yaw_rmse', np.nan)
    pitch_rmse = avgs.get('Pitch_rmse', np.nan)
    avg_r2 = float(np.nanmean([yaw_r2, pitch_r2])) if not (np.isnan(yaw_r2) and np.isnan(pitch_r2)) else np.nan
    avg_rmse = float(np.nanmean([yaw_rmse, pitch_rmse])) if not (np.isnan(yaw_rmse) and np.isnan(pitch_rmse)) else np.nan
    return avg_r2, avg_rmse


def _derive_avg_iqr(rows):
    """Compute IQR of per-trial avg_r2 and avg_rmse."""
    per_r2 = [float(np.nanmean([r.get('Yaw_r2', np.nan), r.get('Pitch_r2', np.nan)])) for r in rows]
    per_rmse = [float(np.nanmean([r.get('Yaw_rmse', np.nan), r.get('Pitch_rmse', np.nan)])) for r in rows]
    valid_r2 = [v for v in per_r2 if not np.isnan(v)]
    valid_rmse = [v for v in per_rmse if not np.isnan(v)]
    iqr_r2 = float(np.percentile(valid_r2, 75) - np.percentile(valid_r2, 25)) if len(valid_r2) >= 2 else np.nan
    iqr_rmse = float(np.percentile(valid_rmse, 75) - np.percentile(valid_rmse, 25)) if len(valid_rmse) >= 2 else np.nan
    return iqr_r2, iqr_rmse


def _build_metric_dict(rows):
    """Return a flat dict of mean + IQR for all METRIC_COLUMNS and derived columns."""
    avgs = _average_metrics(rows)
    iqrs = _iqr_metrics(rows)
    avg_r2, avg_rmse = _derive_avg(avgs)
    iqr_r2, iqr_rmse = _derive_avg_iqr(rows) if rows else (np.nan, np.nan)

    result = {}
    for key in METRIC_COLUMNS:
        result[key] = avgs.get(key, np.nan)
        result[f'{key}_iqr'] = iqrs.get(f'{key}_iqr', np.nan)
    result['avg_r2'] = avg_r2
    result['avg_rmse'] = avg_rmse
    result['avg_r2_iqr'] = iqr_r2
    result['avg_rmse_iqr'] = iqr_rmse
    return result


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

ALL_OUTPUT_COLUMNS = (
    [col for col in METRIC_COLUMNS]
    + ['avg_r2', 'avg_rmse']
    + [f'{col}_iqr' for col in METRIC_COLUMNS]
    + ['avg_r2_iqr', 'avg_rmse_iqr']
)


def _write_participant_summary(output_path, rows):
    header = ['method', 'data_type', 'participant_id'] + ALL_OUTPUT_COLUMNS
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row['method'],
                row['data_type'],
                row['participant_id'],
                *[_format_value(col, row.get(col)) for col in ALL_OUTPUT_COLUMNS],
            ])


def _write_movement_summary(output_path, rows):
    header = ['method', 'movement'] + ALL_OUTPUT_COLUMNS
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row['method'],
                row['movement'],
                *[_format_value(col, row.get(col)) for col in ALL_OUTPUT_COLUMNS],
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Collate transfer learning validation metrics for tfFroz vs tfNotFroz.'
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
        '--quiet',
        action='store_true',
        help='Suppress run discovery printout.',
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    output_dir = args.output_dir or os.path.join(root_dir, 'summary-csv')
    os.makedirs(output_dir, exist_ok=True)

    # Discover tf* runs
    if not os.path.isdir(root_dir):
        print(f'[ERROR] Root directory not found: {root_dir}')
        return

    tf_runs = []
    for name in sorted(os.listdir(root_dir)):
        if not os.path.isdir(os.path.join(root_dir, name)):
            continue
        parsed = _parse_run_name(name)
        if parsed is None:
            continue
        tf_runs.append((name, *parsed))

    if not tf_runs:
        print(f'[WARN] No tf* run directories found under {root_dir}.')
        return

    if not args.quiet:
        print(f'Discovered {len(tf_runs)} tf* run(s):')
        for run_name, method_tag, data_type, participant_id in tf_runs:
            print(f'  {run_name}  ->  method={method_tag}  data_type={data_type}  participant={participant_id}')

    # Collect per-participant rows keyed by (method, data_type, participant_id)
    participant_data = {}
    for run_name, method_tag, data_type, participant_id in tf_runs:
        rows = _load_tf_run_rows(root_dir, run_name, participant_id, quiet=args.quiet)
        if not rows:
            if not args.quiet:
                print(f'[SKIP] No valid rows for {run_name} (participant {participant_id})')
            continue
        key = (method_tag, data_type, participant_id)
        participant_data[key] = rows
        if not args.quiet:
            print(f'[OK]   {run_name}: {len(rows)} trial row(s) loaded')

    if not participant_data:
        print('[WARN] No data loaded. No output written.')
        return

    # Build participant summary (mean + IQR across movements per participant × method)
    participant_summary_rows = []
    for (method_tag, data_type, participant_id), rows in sorted(participant_data.items()):
        metrics = _build_metric_dict(rows)
        out_row = {
            'method': method_tag,
            'data_type': data_type,
            'participant_id': participant_id,
            **metrics,
        }
        participant_summary_rows.append(out_row)

    # Build movement summary (mean + IQR across participants per movement × method)
    # Collect rows grouped by (method, movement)
    movement_groups = {}
    for (method_tag, data_type, participant_id), rows in participant_data.items():
        for row in rows:
            parsed = _parse_trial_name(row.get('trial', ''))
            if parsed is None:
                continue
            _, _, movement = parsed
            movement_groups.setdefault((method_tag, movement), []).append(row)

    movement_summary_rows = []
    for (method_tag, movement) in sorted(movement_groups.keys()):
        rows = movement_groups[(method_tag, movement)]
        metrics = _build_metric_dict(rows)
        out_row = {
            'method': method_tag,
            'movement': movement,
            **metrics,
        }
        movement_summary_rows.append(out_row)

    # Write outputs
    participant_path = os.path.join(output_dir, 'tf_participant_summary.csv')
    _write_participant_summary(participant_path, participant_summary_rows)
    print(f'[OK] Wrote {participant_path}')

    movement_path = os.path.join(output_dir, 'tf_movement_summary.csv')
    _write_movement_summary(movement_path, movement_summary_rows)
    print(f'[OK] Wrote {movement_path}')


if __name__ == '__main__':
    main()
