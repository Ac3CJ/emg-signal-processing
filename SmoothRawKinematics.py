"""
SmoothRawKinematics.py

Post-hoc EMA smoothing of ModelValidator _predictions.csv outputs.

For each raw *_predictions.csv found under ./neural-network-models/<run>/validation/:
  - Applies exponential moving average (EMA) to Yaw_pred and Pitch_pred only.
  - Copies window_idx, time_s, Yaw_gt, Pitch_gt verbatim.
  - Writes smooth{tag}_{trial}_predictions.csv to the same validation/ directory.
  - If ground-truth columns are present, recomputes R², RMSE, and latency and writes
    smooth{tag}_{trial}_metrics.csv alongside.

After processing all trials in a run, writes combined aggregate CSVs:
  - smooth{tag}_c_all_metrics.csv  (collected trials, i.e. stem starts with 'c')
  - smooth{tag}_s_all_metrics.csv  (secondary trials, i.e. stem starts with 's')
These mirror the column structure of c_all_metrics.csv / s_all_metrics.csv so that
AblationMetricsCollator can consume them via --smooth <tag> without format changes.

Originals are never modified. Existing smooth_ files are overwritten silently.

Usage:
    python SmoothRawKinematics.py
    python SmoothRawKinematics.py --alpha 0.1
    python SmoothRawKinematics.py --alpha 0.05 --prefixes window step

Alpha tag encoding: "0" + decimal digits of alpha string
    0.2  -> "02"
    0.05 -> "005"
    0.1  -> "01"
"""

import argparse
import csv
import os
import re

import numpy as np
import scipy.ndimage
from sklearn.metrics import r2_score, mean_squared_error

DOF_NAMES = ['Yaw', 'Pitch']


# ---------------------------------------------------------------------------
# Alpha → tag
# ---------------------------------------------------------------------------

def _alpha_tag(alpha: float) -> str:
    """Convert alpha float to compact string tag (strip '0.' prefix)."""
    s = f'{alpha}'
    if s.startswith('0.'):
        return '0' + s[2:]
    return s.replace('.', '')

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def _apply_ema(series: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out

# ---------------------------------------------------------------------------
# Metric helpers (reimplemented inline — avoids importing ModelValidator and
# its heavy ML dependencies: torch, scipy.io, project-specific modules)
# ---------------------------------------------------------------------------

def _compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Per-DOF R² and RMSE."""
    metrics = {}
    for i, name in enumerate(DOF_NAMES):
        pred_col = predictions[:, i]
        gt_col = ground_truth[:, i]
        metrics[f'{name}_r2'] = float(r2_score(gt_col, pred_col))
        metrics[f'{name}_rmse'] = float(np.sqrt(mean_squared_error(gt_col, pred_col)))
    return metrics


def _detect_rising_edge_tops(sig_norm: np.ndarray) -> np.ndarray:
    """Schmitt-trigger rising-edge detector (mirrors ModelValidator._detect_rising_edge_tops)."""
    low_thresh, high_thresh = 0.2, 0.7
    if len(sig_norm) == 0:
        return np.array([], dtype=int)
    edges = []
    state = 'high' if sig_norm[0] >= low_thresh else 'low'
    for i in range(1, len(sig_norm)):
        v = sig_norm[i]
        if state == 'low' and v >= high_thresh:
            edges.append(i)
            state = 'high'
        elif state == 'high' and v < low_thresh:
            state = 'low'
    return np.array(edges, dtype=int)


def _compute_latency_ms(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    movement_class: int,
    step_ms: float,
) -> float:
    """Mean absolute kinematic latency in ms, or np.nan (mirrors ModelValidator._compute_latency_ms).

    step_ms is inferred from the time_s column so Config.INCREMENT / Config.FS are not needed.
    """
    if movement_class in (9, 10):
        return np.nan

    gt_signal = np.abs(ground_truth[:, 0]) + np.abs(ground_truth[:, 1])
    pred_signal = np.abs(predictions[:, 0]) + np.abs(predictions[:, 1])

    smoothing_window = max(3, int(round(200.0 / step_ms)))
    gt_smoothed = scipy.ndimage.uniform_filter1d(gt_signal, size=smoothing_window, mode='nearest')
    pred_smoothed = scipy.ndimage.uniform_filter1d(pred_signal, size=smoothing_window, mode='nearest')

    def _norm(sig):
        lo, hi = sig.min(), sig.max()
        span = hi - lo
        if span < 1e-9:
            return np.zeros_like(sig)
        return (sig - lo) / span

    gt_peaks = _detect_rising_edge_tops(_norm(gt_smoothed))
    pred_peaks = _detect_rising_edge_tops(_norm(pred_smoothed))

    if len(gt_peaks) == 0 or len(pred_peaks) == 0:
        return np.nan

    latencies = [
        abs(int(pred_peaks[np.argmin(np.abs(pred_peaks - g))]) - int(g)) * step_ms
        for g in gt_peaks
    ]
    return float(np.mean(latencies))


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _read_predictions_csv(path: str):
    """Returns (header list, list-of-rows-as-dicts)."""
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def _write_smoothed_predictions(out_path: str, rows: list, smoothed_yaw: np.ndarray, smoothed_pitch: np.ndarray, has_gt: bool) -> None:
    header = ['window_idx', 'time_s', 'Yaw_pred', 'Pitch_pred']
    if has_gt:
        header += ['Yaw_gt', 'Pitch_gt']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(rows):
            out_row = [
                row['window_idx'],
                row['time_s'],
                f'{smoothed_yaw[i]:.4f}',
                f'{smoothed_pitch[i]:.4f}',
            ]
            if has_gt:
                out_row += [row['Yaw_gt'], row['Pitch_gt']]
            writer.writerow(out_row)


def _write_metrics_csv(out_path: str, trial_name: str, metrics: dict, latency: float) -> None:
    header = ['trial'] + [f'{d}_r2' for d in DOF_NAMES] + [f'{d}_rmse' for d in DOF_NAMES] + ['mean_abs_latency_ms']
    latency_str = f'{latency:.2f}' if not np.isnan(latency) else 'NaN'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(
            [trial_name]
            + [f'{metrics[f"{d}_r2"]:.4f}' for d in DOF_NAMES]
            + [f'{metrics[f"{d}_rmse"]:.4f}' for d in DOF_NAMES]
            + [latency_str]
        )


def _write_aggregate_metrics_csv(out_path: str, trial_rows: list) -> None:
    """Write a multi-row aggregate metrics CSV mirroring c_all_metrics.csv format.

    Each entry in trial_rows is {'trial': str, 'metrics': dict, 'latency': float}.
    Trial names are written as bare stems (e.g. 'cP1M1') so AblationMetricsCollator
    can parse them without modification.
    """
    header = ['trial'] + [f'{d}_r2' for d in DOF_NAMES] + [f'{d}_rmse' for d in DOF_NAMES] + ['mean_abs_latency_ms']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in trial_rows:
            latency = entry['latency']
            latency_str = f'{latency:.2f}' if not np.isnan(latency) else 'NaN'
            metrics = entry['metrics']
            writer.writerow(
                [entry['trial']]
                + [f'{metrics[f"{d}_r2"]:.4f}' for d in DOF_NAMES]
                + [f'{metrics[f"{d}_rmse"]:.4f}' for d in DOF_NAMES]
                + [latency_str]
            )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _list_run_dirs(root_dir: str):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )


def _find_raw_prediction_files(validation_dir: str):
    """Returns sorted list of raw (non-smooth) *_predictions.csv paths in validation_dir."""
    if not os.path.isdir(validation_dir):
        return []
    return sorted(
        os.path.join(validation_dir, name)
        for name in os.listdir(validation_dir)
        if name.endswith('_predictions.csv') and not name.startswith('smooth')
    )


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(pred_path: str, alpha: float, tag: str):
    """Process one raw predictions file.

    Returns a dict {'trial': stem, 'metrics': dict, 'latency': float} when
    ground-truth is available (for aggregate CSV assembly), or None otherwise.
    """
    header, rows = _read_predictions_csv(pred_path)
    if not rows:
        print(f'[WARN] Empty predictions file, skipping: {pred_path}')
        return None

    has_gt = 'Yaw_gt' in header and 'Pitch_gt' in header

    yaw_pred = np.array([float(r['Yaw_pred']) for r in rows])
    pitch_pred = np.array([float(r['Pitch_pred']) for r in rows])

    smoothed_yaw = _apply_ema(yaw_pred, alpha)
    smoothed_pitch = _apply_ema(pitch_pred, alpha)

    # Trial name is the filename stem before _predictions.csv
    stem = os.path.basename(pred_path)[: -len('_predictions.csv')]
    validation_dir = os.path.dirname(pred_path)
    out_pred_name = f'smooth{tag}_{stem}_predictions.csv'
    out_pred_path = os.path.join(validation_dir, out_pred_name)

    _write_smoothed_predictions(out_pred_path, rows, smoothed_yaw, smoothed_pitch, has_gt)
    print(f'[OK] {out_pred_name}')

    if not has_gt:
        return None

    # Recover step_ms from the time_s column; fall back to 1 ms if only one window.
    time_vals = np.array([float(r['time_s']) for r in rows])
    step_ms = float((time_vals[1] - time_vals[0]) * 1000.0) if len(time_vals) > 1 else 1.0

    smoothed_preds = np.column_stack([smoothed_yaw, smoothed_pitch])
    gt_arr = np.column_stack([
        [float(r['Yaw_gt']) for r in rows],
        [float(r['Pitch_gt']) for r in rows],
    ])

    metrics = _compute_metrics(smoothed_preds, gt_arr)

    movement_class = None
    m_match = re.search(r'M(\d+)', stem)
    if m_match:
        movement_class = int(m_match.group(1))

    latency = (
        _compute_latency_ms(smoothed_preds, gt_arr, movement_class, step_ms)
        if movement_class is not None
        else np.nan
    )

    out_metrics_name = f'smooth{tag}_{stem}_metrics.csv'
    out_metrics_path = os.path.join(validation_dir, out_metrics_name)
    _write_metrics_csv(out_metrics_path, stem, metrics, latency)
    print(f'[OK] {out_metrics_name}')

    return {'trial': stem, 'metrics': metrics, 'latency': latency}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Apply post-hoc EMA smoothing to ModelValidator _predictions.csv files.'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='EMA smoothing factor (0 < alpha <= 1). Default: 0.2.',
    )
    parser.add_argument(
        '--root',
        type=str,
        default='./neural-network-models',
        help='Root folder containing run subdirectories (default: ./neural-network-models).',
    )
    parser.add_argument(
        '--prefixes',
        nargs='*',
        default=None,
        help='Optional list of run-name prefixes to process (e.g., window step).',
    )
    args = parser.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        parser.error('--alpha must be in (0, 1].')

    alpha = args.alpha
    tag = _alpha_tag(alpha)
    root_dir = os.path.abspath(args.root)

    run_names = _list_run_dirs(root_dir)
    if args.prefixes:
        prefix_set = set(args.prefixes)
        run_names = [n for n in run_names if n.split('_', 1)[0] in prefix_set]

    if not run_names:
        print(f'No runs found under {root_dir}.')
        return

    for run_name in run_names:
        validation_dir = os.path.join(root_dir, run_name, 'validation')
        pred_files = _find_raw_prediction_files(validation_dir)
        if not pred_files:
            print(f'[WARN] No raw _predictions.csv files in {validation_dir}')
            continue

        print(f'\n--- {run_name} ({len(pred_files)} trials) ---')
        collected_rows = []
        secondary_rows = []
        for pred_path in pred_files:
            result = _process_file(pred_path, alpha, tag)
            if result is None:
                continue
            stem = result['trial']
            if stem.startswith('c'):
                collected_rows.append(result)
            elif stem.startswith('s'):
                secondary_rows.append(result)

        if collected_rows:
            agg_c_path = os.path.join(validation_dir, f'smooth{tag}_c_all_metrics.csv')
            _write_aggregate_metrics_csv(agg_c_path, collected_rows)
            print(f'[OK] smooth{tag}_c_all_metrics.csv ({len(collected_rows)} rows)')

        if secondary_rows:
            agg_s_path = os.path.join(validation_dir, f'smooth{tag}_s_all_metrics.csv')
            _write_aggregate_metrics_csv(agg_s_path, secondary_rows)
            print(f'[OK] smooth{tag}_s_all_metrics.csv ({len(secondary_rows)} rows)')


if __name__ == '__main__':
    main()
