"""
transfer_sweep.py

Standalone sweep: fine-tunes the canonical LOSO checkpoint on each participant's
own data using two freeze strategies, producing one model per (participant × method).

Run naming:
    tfFroz_sP8   / tfNotFroz_sP8   — secondary participant 8 (LOSO holdout)
    tfFroz_cP1   / tfNotFroz_cP1   — collected participant N (P5, P8 excluded)

Usage:
    python transfer_sweep.py --pretrained <path/to/loso_checkpoint.pth>

Each model is saved to:
    ./neural-network-models/<run_name>/training/best_shoulder_rcnn.pth

The _run_config.json sidecar is written automatically by train_transfer_learning(),
so batch_validate.py and ModelValidator.py resolve model_type='rcnn' without manual config.
"""

import argparse
import os
import random

import numpy as np

import ControllerConfiguration as Config
import DataPreparation
import ModelTraining
from FileRepository import DataRepository

REPOSITORY = DataRepository()

SECONDARY_TRANSFER_PARTICIPANTS = [8]
COLLECTED_TRANSFER_PARTICIPANTS = [1, 2, 3, 4, 6, 7, 9]

SEGMENT_SPLIT_SEED = 42
TRAIN_FRACTION = 0.60



def _extract_segments_as_continuous(X, y, bounds):
    """Extracts the sample ranges covered by bounds into a new contiguous array.

    Returns the new continuous X, y, and re-indexed bounds (all starting from 0).
    Segments are concatenated in the order they appear in bounds.
    """
    if len(bounds) == 0:
        empty_X = np.zeros((X.shape[0], 0), dtype=np.float32)
        empty_y = np.zeros((0, y.shape[1]), dtype=np.float32)
        return empty_X, empty_y, []

    chunks_X = []
    chunks_y = []
    new_bounds = []
    cursor = 0
    for start, end in bounds:
        length = end - start
        if length <= 0:
            continue
        chunks_X.append(X[:, start:end])
        chunks_y.append(y[start:end])
        new_bounds.append((cursor, cursor + length))
        cursor += length

    if not chunks_X:
        empty_X = np.zeros((X.shape[0], 0), dtype=np.float32)
        empty_y = np.zeros((0, y.shape[1]), dtype=np.float32)
        return empty_X, empty_y, []

    return np.concatenate(chunks_X, axis=1), np.concatenate(chunks_y, axis=0), new_bounds


def _load_secondary_participant(participant_id):
    """Loads secondary participant data with augmentation (for training split)."""
    return DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=[participant_id],
        include_noise_aug=True,
        augment=True,
    )


def _load_secondary_participant_noaug(participant_id):
    """Loads secondary participant data without augmentation (for val split)."""
    return DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=[participant_id],
        include_noise_aug=False,
        augment=False,
    )


def _load_collected_participant(participant_id):
    """Loads collected participant data with augmentation (for training split)."""
    return DataPreparation.load_collected_data(
        folder_path=REPOSITORY.raw_root('collected'),
        labelled_folder_path=REPOSITORY.edited_root('collected'),
        augment=True,
        include_noise_aug=True,
        include_participants=[participant_id],
    )


def _load_collected_participant_noaug(participant_id):
    """Loads collected participant data without augmentation (for val split)."""
    return DataPreparation.load_collected_data(
        folder_path=REPOSITORY.raw_root('collected'),
        labelled_folder_path=REPOSITORY.edited_root('collected'),
        augment=False,
        include_noise_aug=False,
        include_participants=[participant_id],
    )


def _run_transfer(run_name, pretrained_path, X_train, y_train, train_bounds, train_classes,
                  X_val, y_val, val_bounds, val_classes, freeze_layers):
    """Resolves paths and calls train_transfer_learning for one run."""
    run_paths = ModelTraining._resolve_run_paths(run_name)
    on_the_fly_window_size = int(getattr(Config, 'ON_THE_FLY_WINDOW_SIZE', Config.WINDOW_SIZE))
    on_the_fly_step_size = int(getattr(Config, 'ON_THE_FLY_STEP_SIZE', Config.INCREMENT))
    on_the_fly_active_channels_raw = getattr(Config, 'ON_THE_FLY_ACTIVE_CHANNELS', None)
    active_channels = ModelTraining._parse_channel_list(on_the_fly_active_channels_raw)

    print(f"\n{'='*60}")
    print(f"[Sweep] Run: {run_name}  |  freeze={freeze_layers}")
    print(f"[Sweep] Train segments: {len(train_bounds)} | Val segments: {len(val_bounds)}")
    print(f"{'='*60}")

    ModelTraining.train_transfer_learning(
        X_train, y_train, train_bounds,
        X_val, y_val, val_bounds,
        pretrained_model_path=pretrained_path,
        freeze_layers=freeze_layers,
        window_size=on_the_fly_window_size,
        step_size=on_the_fly_step_size,
        active_channels=active_channels,
        model_save_path=run_paths['model_path'],
        loss_curve_path=run_paths['loss_curve_path'],
        training_log_path=run_paths['training_log_path'],
        model_type='rcnn',
    )
    ModelTraining.save_dataset_distribution(
        train_classes,
        val_classes,
        output_file=run_paths['distribution_path'],
    )


def run_sweep(pretrained_path):
    rng = random.Random(SEGMENT_SPLIT_SEED)

    all_runs = []

    for p in SECONDARY_TRANSFER_PARTICIPANTS:
        prefix = f"sP{p}"
        all_runs.append(('secondary', p, prefix))

    for p in COLLECTED_TRANSFER_PARTICIPANTS:
        prefix = f"cP{p}"
        all_runs.append(('collected', p, prefix))

    print(f"\n[Transfer Sweep] Pretrained checkpoint: {pretrained_path}")
    print(f"[Transfer Sweep] Participants: secondary={SECONDARY_TRANSFER_PARTICIPANTS}, "
          f"collected={COLLECTED_TRANSFER_PARTICIPANTS}")
    print(f"[Transfer Sweep] Total runs: {len(all_runs) * 2} ({len(all_runs)} participants × 2 methods)")

    for source, participant_id, prefix in all_runs:
        print(f"\n{'#'*60}")
        print(f"[Sweep] Loading data for {source} participant {participant_id} ...")
        print(f"{'#'*60}")

        if source == 'secondary':
            X_aug, y_aug, bounds_aug, classes_aug = _load_secondary_participant(participant_id)
            X_raw, y_raw, bounds_raw, classes_raw = _load_secondary_participant_noaug(participant_id)
        else:
            X_aug, y_aug, bounds_aug, classes_aug = _load_collected_participant(participant_id)
            X_raw, y_raw, bounds_raw, classes_raw = _load_collected_participant_noaug(participant_id)

        if X_aug is None or len(bounds_aug) == 0:
            print(f"[Sweep] WARNING: No training data for {source} P{participant_id}. Skipping.")
            continue
        if X_raw is None or len(bounds_raw) == 0:
            print(f"[Sweep] WARNING: No validation data for {source} P{participant_id}. Skipping.")
            continue

        aug_indices = list(range(len(bounds_aug)))
        rng.shuffle(aug_indices)
        n_train = max(1, round(len(aug_indices) * TRAIN_FRACTION))
        train_seg_idx = sorted(aug_indices[:n_train])

        train_bounds_selected = [bounds_aug[i] for i in train_seg_idx]
        train_classes_selected = [classes_aug[i] for i in train_seg_idx]
        X_train, y_train, train_bounds_reindexed = _extract_segments_as_continuous(
            X_aug, y_aug, train_bounds_selected
        )

        raw_indices = list(range(len(bounds_raw)))
        rng.shuffle(raw_indices)
        n_val = max(1, round(len(raw_indices) * (1.0 - TRAIN_FRACTION)))
        val_seg_idx = sorted(raw_indices[:n_val])

        val_bounds_selected = [bounds_raw[i] for i in val_seg_idx]
        val_classes_selected = [classes_raw[i] for i in val_seg_idx]
        X_val, y_val, val_bounds_reindexed = _extract_segments_as_continuous(
            X_raw, y_raw, val_bounds_selected
        )

        for freeze_layers, method_tag in [(True, 'tfFroz'), (False, 'tfNotFroz')]:
            run_name = f"{method_tag}_{prefix}"
            _run_transfer(
                run_name=run_name,
                pretrained_path=pretrained_path,
                X_train=X_train, y_train=y_train, train_bounds=train_bounds_reindexed,
                train_classes=train_classes_selected,
                X_val=X_val, y_val=y_val, val_bounds=val_bounds_reindexed,
                val_classes=val_classes_selected,
                freeze_layers=freeze_layers,
            )

    print(f"\n{'='*60}")
    print("[Sweep] All transfer runs complete.")
    print(f"[Sweep] Validate with: python batch_validate.py")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning sweep: fine-tune LOSO checkpoint per participant × method.'
    )
    parser.add_argument(
        '--pretrained', type=str, required=True,
        help='Path to canonical LOSO checkpoint (best_shoulder_rcnn.pth from LOSO run).'
    )
    args = parser.parse_args()

    if not os.path.exists(args.pretrained):
        print(f"ERROR: Pretrained checkpoint not found: {args.pretrained}")
        raise SystemExit(1)

    run_sweep(args.pretrained)


if __name__ == '__main__':
    main()
