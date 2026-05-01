import argparse
import csv
import os
import re

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import ControllerConfiguration as Config

DOF_NAMES = ['Yaw', 'Pitch']


def _resolve_validation_dir(run_name):
    """Resolves ./neural-network-models/<name>/validation/ for a given run name."""
    sanitized = str(run_name).strip()
    if not sanitized:
        raise ValueError("Run name (--name) must be a non-empty string.")
    return os.path.join('./neural-network-models', sanitized, 'validation')


def _list_prediction_files(validation_dir):
    if not os.path.isdir(validation_dir):
        return []
    return sorted(
        f for f in os.listdir(validation_dir)
        if f.endswith('_predictions.csv')
    )


def _load_predictions_csv(path):
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    time_axis = np.array([float(r['time_s']) for r in rows], dtype=np.float32)
    predictions = np.stack([
        np.array([float(r[f'{d}_pred']) for r in rows], dtype=np.float32)
        for d in DOF_NAMES
    ], axis=1)

    has_gt = all(f'{d}_gt' in rows[0] for d in DOF_NAMES)
    ground_truth = None
    if has_gt:
        ground_truth = np.stack([
            np.array([float(r[f'{d}_gt']) for r in rows], dtype=np.float32)
            for d in DOF_NAMES
        ], axis=1)

    return time_axis, predictions, ground_truth


def _read_metrics_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics = {}
            for k, v in row.items():
                if k == 'trial':
                    metrics[k] = v
                else:
                    try:
                        metrics[k] = float(v)
                    except (TypeError, ValueError):
                        metrics[k] = np.nan
            return metrics
    return None


def _r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _compute_metrics(predictions, ground_truth):
    metrics = {}
    for i, name in enumerate(DOF_NAMES):
        metrics[f'{name}_r2'] = float(_r2_score(ground_truth[:, i], predictions[:, i]))
        metrics[f'{name}_rmse'] = float(_rmse(ground_truth[:, i], predictions[:, i]))
    return metrics


def _detect_rising_edge_tops(sig_norm, low_thresh=0.2, high_thresh=0.7):
    """Schmitt-trigger rising-edge detector: indices where `sig_norm` first crosses
    high_thresh after having been below low_thresh (the top of each rising edge)."""
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


def _detect_peaks(predictions, ground_truth):
    gt_signal = np.abs(ground_truth[:, 0]) + np.abs(ground_truth[:, 1])
    pred_signal = np.abs(predictions[:, 0]) + np.abs(predictions[:, 1])

    step_ms = (Config.INCREMENT / float(Config.FS)) * 1000.0
    smoothing_window = max(3, int(round(200.0 / step_ms)))
    gt_smoothed = scipy.ndimage.uniform_filter1d(gt_signal, size=smoothing_window, mode='nearest')
    pred_smoothed = scipy.ndimage.uniform_filter1d(pred_signal, size=smoothing_window, mode='nearest')

    def _normalise(sig):
        sig_min, sig_max = sig.min(), sig.max()
        span = sig_max - sig_min
        if span < 1e-9:
            return np.zeros_like(sig)
        return (sig - sig_min) / span

    gt_norm = _normalise(gt_smoothed)
    pred_norm = _normalise(pred_smoothed)

    return _detect_rising_edge_tops(gt_norm), _detect_rising_edge_tops(pred_norm)


def _compute_latency_ms(predictions, ground_truth, movement_class, fs=None):
    if movement_class in (9, 10):
        return np.nan

    if fs is None:
        fs = float(Config.FS)

    step_ms = (Config.INCREMENT / fs) * 1000.0

    gt_peaks, pred_peaks = _detect_peaks(predictions, ground_truth)

    if len(gt_peaks) == 0 or len(pred_peaks) == 0:
        return np.nan

    latencies = []
    for gt_idx in gt_peaks:
        nearest = pred_peaks[np.argmin(np.abs(pred_peaks - gt_idx))]
        latencies.append(abs(int(nearest) - int(gt_idx)) * step_ms)

    return float(np.mean(latencies))


def _format_metrics(metrics):
    if metrics is None:
        return "Metrics: N/A"

    def _fmt(value, digits=3):
        if value is None or np.isnan(value):
            return 'NaN'
        return f"{value:.{digits}f}"

    yaw_r2 = _fmt(metrics.get('Yaw_r2'))
    yaw_rmse = _fmt(metrics.get('Yaw_rmse'))
    pitch_r2 = _fmt(metrics.get('Pitch_r2'))
    pitch_rmse = _fmt(metrics.get('Pitch_rmse'))
    latency = _fmt(metrics.get('mean_abs_latency_ms'), digits=2)
    return (f"Yaw R2 {yaw_r2} RMSE {yaw_rmse} | "
            f"Pitch R2 {pitch_r2} RMSE {pitch_rmse} | "
            f"Latency {latency} ms")


def _parse_movement(trial_name):
    match = re.search(r"M(\d+)", trial_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _plot_trial(axs, trial_name, time_axis, predictions, ground_truth, metrics):
    for ax in axs:
        ax.clear()

    if ground_truth is not None:
        gt_peaks, pred_peaks = _detect_peaks(predictions, ground_truth)
    else:
        gt_peaks, pred_peaks = np.array([], dtype=int), np.array([], dtype=int)

    for idx, name in enumerate(DOF_NAMES):
        ax = axs[idx]
        ax.plot(time_axis, predictions[:, idx], label=f'{name} pred', color='tab:blue')
        if ground_truth is not None:
            ax.plot(time_axis, ground_truth[:, idx], label=f'{name} gt', color='tab:orange')
        if len(pred_peaks):
            ax.scatter(time_axis[pred_peaks], predictions[pred_peaks, idx],
                       marker='x', color='tab:blue', s=40, label='pred peaks')
        if len(gt_peaks):
            ax.scatter(time_axis[gt_peaks], ground_truth[gt_peaks, idx],
                       marker='x', color='tab:orange', s=40, label='gt peaks')
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel('Time (s)')
    axs[0].legend(loc='upper right')
    title = f"{trial_name} | {_format_metrics(metrics)}"
    axs[0].figure.suptitle(title)


def main():
    parser = argparse.ArgumentParser(description='Visualise kinematic predictions with latency peak markers.')
    parser.add_argument('--name', type=str, required=True,
                        help='Run name (validation CSVs live in ./neural-network-models/<name>/validation/).')
    args = parser.parse_args()

    validation_dir = _resolve_validation_dir(args.name)
    files = _list_prediction_files(validation_dir)
    if not files:
        print(f"No *_predictions.csv files found in {validation_dir}")
        return

    trial_names = [f.replace('_predictions.csv', '') for f in files]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    fig.subplots_adjust(bottom=0.18)

    state = {'idx': 0}

    def _load_and_plot(idx):
        trial_name = trial_names[idx]
        pred_path = os.path.join(validation_dir, f"{trial_name}_predictions.csv")
        metrics_path = os.path.join(validation_dir, f"{trial_name}_metrics.csv")
        payload = _load_predictions_csv(pred_path)
        if payload is None:
            return
        time_axis, predictions, ground_truth = payload
        metrics = _read_metrics_csv(metrics_path)
        if metrics is None and ground_truth is not None:
            metrics = _compute_metrics(predictions, ground_truth)
            movement = _parse_movement(trial_name)
            metrics['mean_abs_latency_ms'] = _compute_latency_ms(
                predictions, ground_truth, movement
            )
        _plot_trial(axs, trial_name, time_axis, predictions, ground_truth, metrics)
        fig.canvas.draw_idle()

    def _next(_event):
        state['idx'] = (state['idx'] + 1) % len(trial_names)
        _load_and_plot(state['idx'])

    def _prev(_event):
        state['idx'] = (state['idx'] - 1) % len(trial_names)
        _load_and_plot(state['idx'])

    ax_prev = fig.add_axes([0.75, 0.05, 0.1, 0.06])
    ax_next = fig.add_axes([0.86, 0.05, 0.1, 0.06])
    btn_prev = Button(ax_prev, 'Prev')
    btn_next = Button(ax_next, 'Next')
    btn_prev.on_clicked(_prev)
    btn_next.on_clicked(_next)

    _load_and_plot(state['idx'])
    plt.show()


if __name__ == '__main__':
    main()
