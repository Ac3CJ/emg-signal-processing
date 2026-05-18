"""KinematicPlotter.py
Visualize predicted vs ground-truth kinematics for a single (model_name, participant, movement)
trial, reading from the CSVs produced by ModelValidator.py.

Usage:
    python KinematicPlotter.py <model_name> <participant> <movement>
    python KinematicPlotter.py prelim-v1 3 5
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

import ControllerConfiguration as Config

DOF_NAMES = ['Yaw', 'Pitch']
DOF_DESCRIPTIONS = ['Yaw (Flexion / Extension)', 'Pitch (Abduction / Adduction)']
DOF_COLORS = ['tab:blue', 'tab:orange']


_VALID_SOURCES = ('c', 's')


def _resolve_predictions_csv(model_name, participant, movement, source=None):
    """Returns (csv_path, prefix). prefix ∈ {'c', 's', ''}.

    If source is given ('c' or 's'), resolves directly. Otherwise auto-detects by
    checking which prefixed CSV exists in the validation dir; falls back to the
    legacy unprefixed name when neither prefix variant is present. Raises if both
    prefixed variants exist (ambiguous — caller should pass --source).
    """
    name = str(model_name).strip()
    if not name:
        raise ValueError("Model name must be a non-empty string.")
    p, m = int(participant), int(movement)
    base_dir = os.path.join('./neural-network-models', name, 'validation')

    if source is not None:
        prefix = str(source).lower()
        if prefix not in _VALID_SOURCES:
            raise ValueError(f"Unknown source '{source}'. Use 'c' (collected) or 's' (secondary).")
        return os.path.join(base_dir, f"{prefix}P{p}M{m}_predictions.csv"), prefix

    candidates = [(prefix, os.path.join(base_dir, f"{prefix}P{p}M{m}_predictions.csv"))
                  for prefix in _VALID_SOURCES]
    existing = [(prefix, path) for prefix, path in candidates if os.path.exists(path)]
    if len(existing) > 1:
        raise FileNotFoundError(
            f"Multiple prediction CSVs found for P{p}M{m} in {base_dir}: "
            f"{[path for _, path in existing]}. Pass --source c or --source s to disambiguate."
        )
    if existing:
        return existing[0][1], existing[0][0]

    legacy_path = os.path.join(base_dir, f"P{p}M{m}_predictions.csv")
    return legacy_path, ''


def _load_predictions_csv(csv_path):
    """Loads predictions and (optional) ground truth from a trial's CSV.

    Returns:
        time_s (np.ndarray, shape (n,))
        predictions (np.ndarray, shape (n, 4))
        ground_truth (np.ndarray | None, shape (n, 4))
    """
    times = []
    preds = []
    gts = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_gt = all(f"{d}_gt" in fieldnames for d in DOF_NAMES)
        for row in reader:
            times.append(float(row['time_s']))
            preds.append([float(row[f"{d}_pred"]) for d in DOF_NAMES])
            if has_gt:
                gts.append([float(row[f"{d}_gt"]) for d in DOF_NAMES])

    if not times:
        raise ValueError(f"No prediction rows found in {csv_path}")

    time_s = np.asarray(times, dtype=np.float32)
    predictions = np.asarray(preds, dtype=np.float32)
    ground_truth = np.asarray(gts, dtype=np.float32) if has_gt and gts else None
    return time_s, predictions, ground_truth


def _compute_per_dof_metrics(predictions, ground_truth):
    """Returns a list of {'r2', 'rmse'} dicts, one per DOF."""
    metrics = []
    for i in range(predictions.shape[1]):
        gt_col = ground_truth[:, i]
        pred_col = predictions[:, i]
        r2 = float(r2_score(gt_col, pred_col))
        rmse = float(np.sqrt(mean_squared_error(gt_col, pred_col)))
        metrics.append({'r2': r2, 'rmse': rmse})
    return metrics


def plot_kinematics(model_name, participant, movement, source=None, save=True, show=True):
    csv_path, prefix = _resolve_predictions_csv(model_name, participant, movement, source=source)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}\n"
            f"Run ModelValidator.py --name {model_name} ... first."
        )

    print(f"[KinematicPlotter] Loading: {csv_path}")
    time_s, predictions, ground_truth = _load_predictions_csv(csv_path)

    p, m = int(participant), int(movement)
    movement_name = Config.MOVEMENT_NAMES.get(m, f"Movement {m}")
    target_vec = list(Config.TARGET_MAPPING.get(m, [0.0] * Config.NUM_OUTPUTS))[:len(DOF_NAMES)]

    metrics = _compute_per_dof_metrics(predictions, ground_truth) if ground_truth is not None else None

    fig, axes = plt.subplots(len(DOF_NAMES), 1, figsize=(14, 10), sharex=True)
    fig.canvas.manager.set_window_title(f"P{p}M{m} - {model_name}")

    if metrics is not None:
        mean_r2 = float(np.mean([entry['r2'] for entry in metrics]))
        mean_rmse = float(np.mean([entry['rmse'] for entry in metrics]))
        suptitle = (
            f"Subject P{p}  |  Movement {m}: {movement_name}  |  Target: {target_vec}\n"
            f"Model: {model_name}  |  Mean R² = {mean_r2:.3f}  |  Mean RMSE = {mean_rmse:.2f}°"
        )
    else:
        suptitle = (
            f"Subject P{p}  |  Movement {m}: {movement_name}  |  Target: {target_vec}\n"
            f"Model: {model_name} (no ground truth available in CSV)"
        )
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    for i, (ax, dof_label) in enumerate(zip(axes, DOF_DESCRIPTIONS)):
        ax.plot(time_s, predictions[:, i], color=DOF_COLORS[i], linewidth=2, label='Predicted')
        if ground_truth is not None:
            ax.plot(time_s, ground_truth[:, i], color='black', linewidth=1.5,
                    linestyle='--', alpha=0.75, label='Ground Truth')

        if metrics is not None:
            ax.set_title(
                f"{dof_label}    R² = {metrics[i]['r2']:.3f}    RMSE = {metrics[i]['rmse']:.2f}°",
                fontsize=11,
            )
        else:
            ax.set_title(dof_label, fontsize=11)

        ax.set_ylabel("Angle (°)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        save_path = os.path.join(os.path.dirname(csv_path), f"{prefix}P{p}M{m}_kinematics.png")
        plt.savefig(save_path, dpi=150)
        print(f"[KinematicPlotter] Figure saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot predicted vs ground-truth kinematics for a single trial.",
    )
    parser.add_argument('--model_name', type=str,
                        help="Model run name (matches --name in ModelTraining/ModelValidator).")
    parser.add_argument('--participant', type=int, help="Participant ID, e.g. 3.")
    parser.add_argument('--movement', type=int, help="Movement ID (1-9).")
    parser.add_argument('--source', choices=['c', 's'], default=None,
                        help="Data source prefix: 'c' (collected) or 's' (secondary). "
                             "If omitted, auto-detects from existing CSVs in the validation dir.")
    parser.add_argument('--no-show', action='store_true',
                        help="Save the figure without opening a window (headless).")
    args = parser.parse_args()

    try:
        plot_kinematics(
            model_name=args.model_name,
            participant=args.participant,
            movement=args.movement,
            source=args.source,
            save=True,
            show=not args.no_show,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
