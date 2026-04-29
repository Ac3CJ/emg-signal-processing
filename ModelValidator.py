import numpy as np
import torch
import argparse
import os
import csv
import scipy.io
from sklearn.metrics import r2_score, mean_squared_error

import SignalProcessing
import NeuralNetworkModels as NNModels
import ControllerConfiguration as Config
from FileRepository import DataRepository, FileSelection
import KinematicPlotter

DOF_NAMES = ['Yaw', 'Pitch']


def _resolve_validation_dir(run_name):
    """Resolves ./neural-network-models/<name>/validation/ for a given run name."""
    sanitized = str(run_name).strip()
    if not sanitized:
        raise ValueError("Run name (--name) must be a non-empty string.")
    validation_dir = os.path.join('./neural-network-models', sanitized, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    return validation_dir


def _resolve_default_model_path(run_name):
    """Default model checkpoint location for a given run name."""
    return os.path.join('./neural-network-models', str(run_name).strip(), 'training', 'best_shoulder_rcnn.pth')


def _compute_window_starts(num_samples):
    """Window starts for prediction, honoring Config.WARMUP_SECONDS."""
    warmup_samples = int(getattr(Config, 'WARMUP_SECONDS', 0.0) * Config.FS)
    return list(range(warmup_samples, num_samples - Config.WINDOW_SIZE + 1, Config.INCREMENT))


def _extract_robust_minmax(mat_data):
    for key in ("MIN_MAX_ROBUST", "MIN_MAX"):
        if key not in mat_data:
            continue
        matrix = np.asarray(mat_data[key], dtype=np.float32)
        if matrix.ndim != 2:
            continue
        if matrix.shape == (Config.NUM_CHANNELS, 2):
            return matrix
        if matrix.shape == (2, Config.NUM_CHANNELS):
            return matrix.T
    return None


def _resolve_robust_minmax_for_file(file_path, mat_data):
    matrix = _extract_robust_minmax(mat_data)
    if matrix is not None:
        return matrix

    repository = DataRepository()
    labelled_path = repository.labelled_candidate_path(file_path)
    if os.path.normpath(labelled_path) == os.path.normpath(file_path):
        return None

    try:
        labelled_mat = scipy.io.loadmat(labelled_path)
        return _extract_robust_minmax(labelled_mat)
    except Exception:
        return None

    return None


def _load_collected_kinematics_for_path(file_path):
    """Returns (n_emg_samples, NUM_OUTPUTS) KINEMATICS for a collected trial, or None.

    Resolves the sibling 'P{p}M{m}Kinematic.mat' file under collected/edited/ and pulls
    its 'KINEMATICS' array. Used so collected validation can show the dotted GT line even
    when the labelled .mat file itself doesn't carry KINEMATICS.
    """
    repository = DataRepository()
    selection = repository.selection_from_path(file_path)
    if selection is None or selection.data_type != 'collected':
        return None
    kin_path = os.path.join(
        repository.edited_root('collected'),
        f'P{selection.participant}M{selection.movement}Kinematic.mat',
    )
    if not os.path.exists(kin_path):
        return None
    try:
        kin_data = scipy.io.loadmat(kin_path)
    except Exception:
        return None
    if 'KINEMATICS' not in kin_data:
        return None
    kin = np.asarray(kin_data['KINEMATICS'], dtype=np.float32)
    if kin.ndim != 2 or kin.shape[1] != Config.NUM_OUTPUTS:
        return None
    return kin


def _load_secondary_kinematics_for_path(file_path, n_emg_samples):
    """Returns 1-D angolospalla resampled to EMG sample rate (length n_emg_samples), or None.

    Secondary trials store kinematics in a sibling `MovimentoAngS{m}_edit.mat` (raw fallback:
    `MovimentoAngS{m}.mat`) under the participant's `Soggetto{p}/` directory, NOT inside the
    labelled .mat. Linear interp from kinematic-rate to EMG-rate so EMG-domain `window_ends`
    can index it directly. Mirrors `DataPreparation._load_secondary_kinematic_profile`.
    """
    repository = DataRepository()
    selection = repository.selection_from_path(file_path)
    if selection is None or selection.data_type != 'secondary':
        return None

    subject_root = os.path.join(
        repository.edited_root('secondary'),
        f'Soggetto{selection.participant}',
    )
    candidates = [
        os.path.join(subject_root, f'MovimentoAngS{selection.movement}_edit.mat'),
        repository.secondary_kinematics_file_path(selection.participant, selection.movement),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            mat = scipy.io.loadmat(path)
        except Exception:
            continue
        if 'angolospalla' not in mat:
            continue
        angolospalla = np.asarray(mat['angolospalla'], dtype=np.float64).flatten()
        if angolospalla.size == 0:
            continue
        t_kin = np.linspace(0.0, 1.0, angolospalla.size, dtype=np.float64)
        t_emg = np.linspace(0.0, 1.0, n_emg_samples, dtype=np.float64)
        return np.interp(t_emg, t_kin, angolospalla).astype(np.float32)

    return None


def _scale_angolospalla_to_4dof(angolospalla_emg, movement_class):
    """Projects 1-D angolospalla (at EMG rate) onto the 4-DOF target vector for a movement.

    Direction-aware clamp keeps the trace inside [0, signed_ref] so noise on the wrong side
    of zero doesn't translate into spurious DOF values — same approach DataPreparation uses.
    """
    target_vec = np.asarray(
        Config.TARGET_MAPPING.get(movement_class, [0] * Config.NUM_OUTPUTS),
        dtype=np.float32,
    )
    dominant_idx = int(np.argmax(np.abs(target_vec)))
    signed_ref = float(target_vec[dominant_idx])
    if abs(signed_ref) < 1e-6:
        return None
    lo = min(0.0, signed_ref)
    hi = max(0.0, signed_ref)
    clamped = np.clip(angolospalla_emg, lo, hi)
    scale = (clamped / signed_ref).astype(np.float32)
    return (scale[:, None] * target_vec[None, :]).astype(np.float32)


def _load_ground_truth(mat_data, window_ends, movement_class=None, file_path=None):
    """Returns (n_windows, NUM_OUTPUTS) ground truth aligned to window end positions, or None."""
    if 'KINEMATICS' in mat_data:
        gt = np.asarray(mat_data['KINEMATICS'], dtype=np.float32)
        if gt.ndim == 2 and gt.shape[1] == Config.NUM_OUTPUTS:
            valid_ends = np.clip(window_ends, 0, gt.shape[0] - 1)
            return gt[valid_ends, :]

    if file_path is None:
        return None

    sibling_kin = _load_collected_kinematics_for_path(file_path)
    if sibling_kin is not None:
        valid_ends = np.clip(window_ends, 0, sibling_kin.shape[0] - 1)
        return sibling_kin[valid_ends, :]

    if movement_class is not None and 'EMGDATA' in mat_data:
        n_emg_samples = int(np.asarray(mat_data['EMGDATA']).shape[1])
        angolospalla_emg = _load_secondary_kinematics_for_path(file_path, n_emg_samples)
        if angolospalla_emg is not None:
            gt_4dof = _scale_angolospalla_to_4dof(angolospalla_emg, movement_class)
            if gt_4dof is None:
                return None
            valid_ends = np.clip(window_ends, 0, gt_4dof.shape[0] - 1)
            return gt_4dof[valid_ends, :]

    return None


def _compute_metrics(predictions, ground_truth):
    """Returns per-DOF R² and RMSE as a dict."""
    metrics = {}
    for i, name in enumerate(DOF_NAMES):
        pred_col = predictions[:, i]
        gt_col = ground_truth[:, i]
        metrics[f'{name}_r2'] = float(r2_score(gt_col, pred_col))
        metrics[f'{name}_rmse'] = float(np.sqrt(mean_squared_error(gt_col, pred_col)))
    return metrics


def _save_predictions_csv(output_dir, trial_name, predictions, ground_truth, time_axis):
    """Saves per-window predictions (and ground truth when available) to a CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{trial_name}_predictions.csv")
    has_gt = ground_truth is not None
    header = ['window_idx', 'time_s'] + [f'{d}_pred' for d in DOF_NAMES]
    if has_gt:
        header += [f'{d}_gt' for d in DOF_NAMES]
    num_dofs = len(DOF_NAMES)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(predictions)):
            row = [i, f'{time_axis[i]:.4f}'] + [f'{predictions[i, idx]:.4f}' for idx in range(num_dofs)]
            if has_gt:
                row += [f'{ground_truth[i, idx]:.4f}' for idx in range(num_dofs)]
            writer.writerow(row)
    return path


_DATA_TYPE_PREFIX = {'collected': 'c', 'secondary': 's'}


def _prefix_for(data_type):
    """Returns 'c' / 's' for known data types, '' otherwise."""
    return _DATA_TYPE_PREFIX.get(data_type, '')


def _maybe_plot_trial(plot, model_name, participant, movement, data_type=None):
    """If plot is True and (participant, movement) are known, render a predicted-vs-GT PNG
    via KinematicPlotter. Saves alongside the predictions CSV in the validation dir."""
    if not plot:
        return
    if model_name is None or participant is None or movement is None:
        return
    try:
        KinematicPlotter.plot_kinematics(
            model_name=model_name,
            participant=int(participant),
            movement=int(movement),
            source=_prefix_for(data_type) or None,
            save=True,
            show=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"     [plot] Skipped: {exc}")


def _metrics_row(trial_name, metrics):
    return ([trial_name]
            + [f'{metrics[f"{d}_r2"]:.4f}' for d in DOF_NAMES]
            + [f'{metrics[f"{d}_rmse"]:.4f}' for d in DOF_NAMES])


def _save_per_trial_metrics_csv(output_dir, trial_name, metrics):
    """Writes a single-row R²/RMSE CSV for one trial: <trial_name>_metrics.csv."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{trial_name}_metrics.csv")
    header = ['trial'] + [f'{d}_r2' for d in DOF_NAMES] + [f'{d}_rmse' for d in DOF_NAMES]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(_metrics_row(trial_name, metrics))
    return path


def _save_metrics_csv(output_dir, file_stem, all_metrics_rows):
    """Writes accumulated per-trial metric rows to <file_stem>.csv."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{file_stem}.csv")
    header = ['trial'] + [f'{d}_r2' for d in DOF_NAMES] + [f'{d}_rmse' for d in DOF_NAMES]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in all_metrics_rows:
            writer.writerow(row)
    return path


def get_predictions_for_file(model, device, file_path, mat_data=None):
    if mat_data is None:
        mat_data = scipy.io.loadmat(file_path)
    raw_data = mat_data['EMGDATA']
    num_samples = raw_data.shape[1]

    robust_minmax = _resolve_robust_minmax_for_file(file_path, mat_data)
    if robust_minmax is None:
        print(f"[ModelValidator] WARNING: MIN_MAX_ROBUST not found for {file_path}. Falling back to per-window percentiles.")

    predictions = []
    smoothed_pred = np.zeros(Config.NUM_OUTPUTS)
    window_starts = _compute_window_starts(num_samples)

    with torch.no_grad():
        for start in window_starts:
            window_raw = raw_data[:, start:start+Config.WINDOW_SIZE]

            cleaned_window = np.zeros_like(window_raw, dtype=np.float32)
            for c in range(Config.NUM_CHANNELS):
                cleaned_window[c, :] = SignalProcessing.applyStandardSEMGProcessing(
                    window_raw[c, :], fs=Config.FS
                )

            if robust_minmax is not None:
                normalized_window = np.zeros_like(cleaned_window, dtype=np.float32)
                for c in range(Config.NUM_CHANNELS):
                    scale = SignalProcessing.get_rectified_scale_from_minmax(
                        robust_minmax[c, 0],
                        robust_minmax[c, 1],
                    )
                    normalized_window[c, :] = np.clip(cleaned_window[c, :] / scale, 0.0, 1.0)
            else:
                normalized_window = SignalProcessing.applyGlobalNormalization(cleaned_window, (1, 99))

            window_tensor = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(window_tensor).cpu().numpy()[0]
            smoothed_pred = (Config.SMOOTHING_ALPHA * pred) + ((1.0 - Config.SMOOTHING_ALPHA) * smoothed_pred)
            predictions.append(smoothed_pred.copy())

    return np.array(predictions), np.array(window_starts) / Config.FS


def run_collected_validation(model_path, participant_num, base_path=Config.COLLECTED_DATA_PATH, output_dir='validation-csv', plot=False, model_name=None):
    """Validates all movements (M1-M9) for a specific participant using collected data."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Collected Data Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NNModels.ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    repository = DataRepository.from_standard_path(base_path)

    print(f"[Collected Data Validation] Processing Participant P{participant_num}...")

    all_metrics_rows = []

    for m in range(1, 10):
        if repository is not None:
            file_path = repository.output_file_path(
                FileSelection(data_type="collected", participant=participant_num, movement=m),
                create_dirs=False,
            )
        else:
            file_path = os.path.join(base_path, f'P{participant_num}M{m}_labelled.mat')

        if not os.path.exists(file_path):
            print(f"  -> Skipping P{participant_num}M{m}: File not found at {file_path}")
            continue

        trial_name = f"cP{participant_num}M{m}"
        print(f"  -> Analyzing {trial_name}...")

        mat_data = scipy.io.loadmat(file_path)
        predictions, time_axis = get_predictions_for_file(model, device, file_path, mat_data=mat_data)

        num_samples = mat_data['EMGDATA'].shape[1]
        window_ends = np.array([s + Config.WINDOW_SIZE - 1 for s in _compute_window_starts(num_samples)])
        ground_truth = _load_ground_truth(mat_data, window_ends, movement_class=m, file_path=file_path)

        pred_path = _save_predictions_csv(output_dir, trial_name, predictions, ground_truth, time_axis)
        print(f"     Predictions saved: {pred_path}")

        if ground_truth is not None:
            metrics = _compute_metrics(predictions, ground_truth)
            all_metrics_rows.append(_metrics_row(trial_name, metrics))
            _save_per_trial_metrics_csv(output_dir, trial_name, metrics)
            print(f"     R² — " + "  ".join(f"{d}: {metrics[f'{d}_r2']:.3f}" for d in DOF_NAMES))

        _maybe_plot_trial(plot, model_name, participant_num, m, data_type='collected')

    if all_metrics_rows:
        metrics_path = _save_metrics_csv(output_dir, f"cP{participant_num}_metrics", all_metrics_rows)
        print(f"\n[Collected Data Validation] Metrics saved: {metrics_path}")

    print(f"\n[Collected Data Validation] Done. CSVs saved to {output_dir}/")


def run_benchmark_validation(model_path, output_dir, base_path=Config.COLLECTED_DATA_PATH,
                             test_participants=(3, 6, 7, 9), plot=False, model_name=None):
    """Validates the held-out collected benchmark participants and writes CSVs to output_dir.

    Mirrors the train/test split used by the benchmark training pipeline:
    train on collected [1, 2, 4, 8], evaluate on the participants listed here.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Benchmark Validation] Loading Model: {model_path}")
    print(f"[Benchmark Validation] Held-out participants: {list(test_participants)}")
    blacklist = set(getattr(Config, 'COLLECTED_BLACKLIST', []) or [])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NNModels.ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    repository = DataRepository.from_standard_path(base_path)
    all_metrics_rows = []

    for participant in test_participants:
        for m in range(1, 10):
            if (participant, m) in blacklist:
                print(f"  -> Skipping P{participant}M{m}: blacklisted in COLLECTED_BLACKLIST")
                continue

            if repository is not None:
                file_path = repository.output_file_path(
                    FileSelection(data_type="collected", participant=participant, movement=m),
                    create_dirs=False,
                )
            else:
                file_path = os.path.join(base_path, f'P{participant}M{m}_labelled.mat')

            if not os.path.exists(file_path):
                print(f"  -> Skipping P{participant}M{m}: file not found at {file_path}")
                continue

            trial_name = f"cP{participant}M{m}"
            print(f"  -> Analyzing {trial_name}...")
            mat_data = scipy.io.loadmat(file_path)
            predictions, time_axis = get_predictions_for_file(model, device, file_path, mat_data=mat_data)

            num_samples = mat_data['EMGDATA'].shape[1]
            window_ends = np.array([s + Config.WINDOW_SIZE - 1 for s in _compute_window_starts(num_samples)])
            ground_truth = _load_ground_truth(mat_data, window_ends, movement_class=m, file_path=file_path)

            pred_path = _save_predictions_csv(output_dir, trial_name, predictions, ground_truth, time_axis)
            print(f"     Predictions saved: {pred_path}")

            if ground_truth is not None:
                metrics = _compute_metrics(predictions, ground_truth)
                all_metrics_rows.append(_metrics_row(trial_name, metrics))
                _save_per_trial_metrics_csv(output_dir, trial_name, metrics)
                print(f"     R² — " + "  ".join(f"{d}: {metrics[f'{d}_r2']:.3f}" for d in DOF_NAMES))

            _maybe_plot_trial(plot, model_name, participant, m, data_type='collected')

    if all_metrics_rows:
        metrics_path = _save_metrics_csv(output_dir, "c_benchmark_metrics", all_metrics_rows)
        print(f"\n[Benchmark Validation] Metrics saved: {metrics_path}")

    print(f"\n[Benchmark Validation] Done. CSVs saved to {output_dir}/")


def run_fast_validation(model_path, sim_file=None, predefined=False, base_path='./secondary_data', output_dir='validation-csv', plot=False, model_name=None):
    repository = DataRepository.from_standard_path(base_path)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Fast Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NNModels.ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if predefined:
        print("[Fast Validation] Running predefined benchmark suite...")
        test_cases = [(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (6, 1), (7, 2), (2, 3), (6, 4), (7, 5), (5, 6), (6, 7)]
        if repository is not None:
            files_to_process = [
                (repository.output_file_path(
                    FileSelection(data_type='secondary', participant=p, movement=m),
                    create_dirs=False,
                ), m)
                for p, m in test_cases
            ]
        else:
            files_to_process = [
                (os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}_labelled.mat'), m)
                for p, m in test_cases
            ]
    else:
        if not sim_file:
            print("ERROR: No simulation file provided for validation.")
            return
        files_to_process = [(sim_file, None)]

    all_metrics_rows = []

    for file_path, movement_class in files_to_process:
        if not os.path.exists(file_path):
            print(f"  -> Skipping: File not found at {file_path}")
            continue

        selection = (repository or DataRepository()).selection_from_path(file_path)
        if selection is not None:
            data_type = selection.data_type
            trial_name = f"{_prefix_for(data_type)}P{selection.participant}M{selection.movement}"
        else:
            data_type = None
            trial_name = os.path.splitext(os.path.basename(file_path))[0]

        print(f"  -> Analyzing {trial_name}...")

        mat_data = scipy.io.loadmat(file_path)
        predictions, time_axis = get_predictions_for_file(model, device, file_path, mat_data=mat_data)

        num_samples = mat_data['EMGDATA'].shape[1]
        window_ends = np.array([s + Config.WINDOW_SIZE - 1 for s in _compute_window_starts(num_samples)])
        ground_truth = _load_ground_truth(mat_data, window_ends, movement_class=movement_class, file_path=file_path)

        pred_path = _save_predictions_csv(output_dir, trial_name, predictions, ground_truth, time_axis)
        print(f"     Predictions saved: {pred_path}")

        if ground_truth is not None:
            metrics = _compute_metrics(predictions, ground_truth)
            all_metrics_rows.append(_metrics_row(trial_name, metrics))
            _save_per_trial_metrics_csv(output_dir, trial_name, metrics)
            print(f"     R² — " + "  ".join(f"{d}: {metrics[f'{d}_r2']:.3f}" for d in DOF_NAMES))

        if selection is not None:
            _maybe_plot_trial(plot, model_name, selection.participant, selection.movement, data_type=data_type)

    if all_metrics_rows:
        run_name = 's_predefined_metrics' if predefined else 'fast_validation_metrics'
        metrics_path = _save_metrics_csv(output_dir, run_name, all_metrics_rows)
        print(f"\n[Fast Validation] Metrics saved: {metrics_path}")

    print(f"\n[Fast Validation] Done. CSVs saved to ./{output_dir}/")


def run_all_validation(model_path, output_dir='validation-csv', plot=False, model_name=None,
                       collected_base=Config.COLLECTED_DATA_PATH,
                       secondary_base=Config.SECONDARY_DATA_PATH):
    """Validates every available (P, M) trial across both collected and secondary datasets.

    Honors COLLECTED_BLACKLIST and SECONDARY_BLACKLIST. Predictions and per-trial metrics
    are written with `c` / `s` filename prefixes; combined per-source metrics CSVs land
    as `c_all_metrics.csv` and `s_all_metrics.csv`.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Validate All] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NNModels.ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sources = [
        ('collected', collected_base, set(getattr(Config, 'COLLECTED_BLACKLIST', []) or [])),
        ('secondary', secondary_base, set(getattr(Config, 'SECONDARY_BLACKLIST', []) or [])),
    ]

    for data_type, base_path, blacklist in sources:
        prefix = _prefix_for(data_type)
        repository = DataRepository.from_standard_path(base_path)
        if repository is None:
            print(f"[Validate All] Skipping {data_type}: cannot resolve repository from {base_path}")
            continue

        selections = repository.iter_file_selections(data_type)
        all_metrics_rows = []

        print(f"\n[Validate All] Processing {data_type} ({len(selections)} candidate trials)...")

        for selection in selections:
            key = (selection.participant, selection.movement)
            trial_name = f"{prefix}P{selection.participant}M{selection.movement}"

            if key in blacklist:
                print(f"  -> Skipping {trial_name}: blacklisted in {data_type.upper()}_BLACKLIST")
                continue

            file_path = repository.output_file_path(selection, create_dirs=False)
            if not os.path.exists(file_path):
                print(f"  -> Skipping {trial_name}: file not found at {file_path}")
                continue

            print(f"  -> Analyzing {trial_name}...")
            mat_data = scipy.io.loadmat(file_path)
            predictions, time_axis = get_predictions_for_file(model, device, file_path, mat_data=mat_data)

            num_samples = mat_data['EMGDATA'].shape[1]
            window_ends = np.array([s + Config.WINDOW_SIZE - 1 for s in _compute_window_starts(num_samples)])
            ground_truth = _load_ground_truth(mat_data, window_ends, movement_class=selection.movement, file_path=file_path)

            pred_path = _save_predictions_csv(output_dir, trial_name, predictions, ground_truth, time_axis)
            print(f"     Predictions saved: {pred_path}")

            if ground_truth is not None:
                metrics = _compute_metrics(predictions, ground_truth)
                all_metrics_rows.append(_metrics_row(trial_name, metrics))
                _save_per_trial_metrics_csv(output_dir, trial_name, metrics)
                print(f"     R² — " + "  ".join(f"{d}: {metrics[f'{d}_r2']:.3f}" for d in DOF_NAMES))

            _maybe_plot_trial(plot, model_name, selection.participant, selection.movement, data_type=data_type)

        if all_metrics_rows:
            metrics_path = _save_metrics_csv(output_dir, f"{prefix}_all_metrics", all_metrics_rows)
            print(f"\n[Validate All] {data_type} combined metrics saved: {metrics_path}")

    print(f"\n[Validate All] Done. CSVs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Validator for sEMG Shoulder Prosthetics")
    parser.add_argument('--name', type=str, required=True,
                        help='Run name. CSVs go to ./neural-network-models/<name>/validation/.')
    parser.add_argument('--validate', action='store_true', help='Run validation on a single sim_file')
    parser.add_argument('--validate_predefined', action='store_true', help='Run the full benchmark suite')
    parser.add_argument('--validate_benchmark', action='store_true',
                        help='Validate the held-out collected benchmark participants (P3, P6, P7, P9 by default).')
    parser.add_argument('--validate_all', action='store_true',
                        help='Validate every available (P, M) across both collected and secondary datasets, honoring blacklists.')
    parser.add_argument('--collected', type=int, help='Validate all movements for a participant using collected data (e.g., --collected 1)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained PyTorch weights. Defaults to ./neural-network-models/<name>/training/best_shoulder_rcnn.pth.')
    parser.add_argument('--sim_file', type=str, default='./secondary_data/Soggetto1/Movimento3.mat', help='Specific .mat file to validate')
    parser.add_argument('--plot', action='store_true',
                        help='Render predicted-vs-GT kinematics PNGs alongside the CSVs (off by default).')

    args = parser.parse_args()

    try:
        validation_dir = _resolve_validation_dir(args.name)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)

    if args.model is None:
        args.model = _resolve_default_model_path(args.name)

    for arg in vars(args):
        print(f"  -> {arg}: {getattr(args, arg)}")
    print(f"  -> validation output dir: {validation_dir}")

    if args.validate_predefined:
        run_fast_validation(model_path=args.model, predefined=True, base_path=Config.SECONDARY_DATA_PATH, output_dir=validation_dir, plot=args.plot, model_name=args.name)
    if args.validate:
        run_fast_validation(model_path=args.model, sim_file=args.sim_file, predefined=False, output_dir=validation_dir, plot=args.plot, model_name=args.name)
    if args.collected is not None:
        run_collected_validation(model_path=args.model, participant_num=args.collected, output_dir=validation_dir, plot=args.plot, model_name=args.name)
    if args.validate_benchmark:
        run_benchmark_validation(model_path=args.model, output_dir=validation_dir, base_path=Config.COLLECTED_DATA_PATH, plot=args.plot, model_name=args.name)
    if args.validate_all:
        run_all_validation(model_path=args.model, output_dir=validation_dir, plot=args.plot, model_name=args.name,
                           collected_base=Config.COLLECTED_DATA_PATH, secondary_base=Config.SECONDARY_DATA_PATH)
