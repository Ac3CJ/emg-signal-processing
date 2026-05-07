"""
BatchSweep.py
Sequential one-parameter-at-a-time ablation sweep for FYP report results.

Usage:
    python BatchSweep.py                          # run all groups
    python BatchSweep.py --groups step window     # run specific groups

Ablation groups:
    step    — window step size (500 ms window, step varies)
    window  — window size (increment scaled by 0.124 factor from Rivela et al.)
    augment — data augmentation methods (additive from no-augmentation control)
    filter  — signal processing pipeline during training

Base configuration (Rivela et al.):
    WINDOW_SIZE=500 ms, INCREMENT=62 ms, notch+bandpass+rectify+percentile-minmax,
    all augmentations on (mixup + MW + noise).

Naming convention: {group}_{test_name}
Each experiment trains a fresh model from scratch.

NOTE on filter ablation: The TRAINING pipeline is varied per condition. The
VALIDATION pipeline (ModelValidator.run_all_validation) always applies the
standard notch+bandpass+rectify+percentile-minmax inference chain. Results
therefore show "how well does training with filter X transfer to standard
inference?" — a valid comparison that should be stated in the report methodology.

"""

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import ControllerConfiguration as Config
import DataPreparation
import SignalProcessing
import ModelTraining
import ModelValidator

# ====================================================================================
# BASE CONFIGURATION — Rivela et al. reference point
# ====================================================================================
_BASE_CONFIG = {
    'WINDOW_SIZE': 500,
    'INCREMENT': 62,
    'ON_THE_FLY_WINDOW_SIZE': 500,
    'ON_THE_FLY_STEP_SIZE': 62,
    'MIXUP_ALPHA': 0.2,
    'MIXUP_RATIO': 0.75,
    'REST_MIXUP_ALPHA': 0.2,
    'REST_MIXUP_RATIO': 0.5,
    'TRAINING_NOISE_MAGNITUDES': [0.000005, 0.00001],
    'ENABLE_MAGNITUDE_WARPING': True,
}

_ORIG_FILTER_FN = DataPreparation._filter_and_normalize_burst


def _reset_to_base():
    for k, v in _BASE_CONFIG.items():
        setattr(Config, k, v)
    DataPreparation._filter_and_normalize_burst = _ORIG_FILTER_FN


# ====================================================================================
# CUSTOM FILTER BURST FUNCTIONS
# These replace DataPreparation._filter_and_normalize_burst for filter ablation.
# Each receives (raw_burst: np.ndarray[C, T], channel_minmax: np.ndarray[C, 2])
# and must return a processed np.ndarray[C, T].
# ====================================================================================

def _filter_burst_notch_only(raw_burst, channel_minmax):
    raw = np.asarray(raw_burst, dtype=np.float32)
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        notched = SignalProcessing.notchFilter(raw[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        rectified = np.abs(notched)
        scale = SignalProcessing.get_rectified_scale_from_minmax(channel_minmax[c, 0], channel_minmax[c, 1])
        out[c, :] = np.clip(rectified / scale, 0.0, 1.0)
    return out


def _filter_burst_bandpass_only(raw_burst, channel_minmax):
    raw = np.asarray(raw_burst, dtype=np.float32)
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        band = SignalProcessing.bandpassFilter(
            raw[c, :], fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH
        )
        rectified = np.abs(band)
        scale = SignalProcessing.get_rectified_scale_from_minmax(channel_minmax[c, 0], channel_minmax[c, 1])
        out[c, :] = np.clip(rectified / scale, 0.0, 1.0)
    return out


def _filter_burst_naive_minmax(raw_burst, channel_minmax):
    """Notch + bandpass + rectify, but scale by the burst's own max instead of stored percentile minmax."""
    raw = np.asarray(raw_burst, dtype=np.float32)
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        notched = SignalProcessing.notchFilter(raw[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        band = SignalProcessing.bandpassFilter(
            notched, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH
        )
        rectified = np.abs(band)
        scale = max(float(rectified.max()), 1e-6)
        out[c, :] = np.clip(rectified / scale, 0.0, 1.0)
    return out


def _filter_burst_no_rectification(raw_burst, channel_minmax):
    """Notch + bandpass only; normalize to [-1, 1] using stored minmax bounds."""
    raw = np.asarray(raw_burst, dtype=np.float32)
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        notched = SignalProcessing.notchFilter(raw[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        band = SignalProcessing.bandpassFilter(
            notched, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH
        )
        scale = SignalProcessing.get_unrectified_scale_from_minmax(channel_minmax[c, 0], channel_minmax[c, 1])
        out[c, :] = np.clip(band / scale, -1.0, 1.0)
    return out


def _filter_burst_no_filtering(raw_burst, channel_minmax):
    """No filtering at all; normalize raw signal to [-1, 1] using stored minmax bounds."""
    raw = np.asarray(raw_burst, dtype=np.float32)
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        scale = SignalProcessing.get_unrectified_scale_from_minmax(channel_minmax[c, 0], channel_minmax[c, 1])
        out[c, :] = np.clip(raw[c, :] / scale, -1.0, 1.0)
    return out


def _filter_burst_wavelet(raw_burst, channel_minmax):
    """Notch → bandpass → wavelet denoise (sym4, 10 levels, Garrote) → rectify → percentile-minmax."""
    raw = np.asarray(raw_burst, dtype=np.float32)
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        notched = SignalProcessing.notchFilter(raw[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        band = SignalProcessing.bandpassFilter(
            notched, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH
        )
        denoised = SignalProcessing.wavelet_denoise(band)
        rectified = np.abs(denoised)
        scale = SignalProcessing.get_rectified_scale_from_minmax(channel_minmax[c, 0], channel_minmax[c, 1])
        out[c, :] = np.clip(rectified / scale, 0.0, 1.0)
    return out


# ====================================================================================
# EXPERIMENT RUNNER
# ====================================================================================

def _run_experiment(run_name, config_overrides=None, filter_fn_override=None, model_type='rcnn'):
    _reset_to_base()

    if config_overrides:
        for k, v in config_overrides.items():
            setattr(Config, k, v)

    if filter_fn_override is not None:
        DataPreparation._filter_and_normalize_burst = filter_fn_override

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {run_name}")
    print(f"  MODEL={model_type}")
    print(f"  WINDOW={Config.WINDOW_SIZE}  INCREMENT={Config.INCREMENT}")
    print(f"  MIXUP_RATIO={Config.MIXUP_RATIO}  "
          f"NOISE={Config.TRAINING_NOISE_MAGNITUDES}  "
          f"MW={getattr(Config, 'ENABLE_MAGNITUDE_WARPING', True)}")
    if filter_fn_override is not None:
        print(f"  FILTER_FN={filter_fn_override.__name__}")
    print(f"{'=' * 70}\n")

    run_paths = ModelTraining._resolve_run_paths(run_name)

    ModelTraining.loso_pipeline(
        selected_collected_for_main_modes=[],
        collected_raw_path=ModelTraining.REPOSITORY.raw_root('collected'),
        collected_edited_path=ModelTraining.REPOSITORY.edited_root('collected'),
        on_the_fly_window_size=Config.WINDOW_SIZE,
        on_the_fly_step_size=Config.INCREMENT,
        selected_active_channels=None,
        run_paths=run_paths,
        model_type=model_type,
    )

    ModelValidator.run_all_validation(
        model_path=run_paths['model_path'],
        output_dir=run_paths['validation_dir'],
        plot=False,
        model_name=run_name,
        collected_base=Config.COLLECTED_DATA_PATH,
        secondary_base=Config.SECONDARY_DATA_PATH,
        model_type=model_type,
    )

    print(f"\n[BatchSweep] Completed: {run_name}\n")


# ====================================================================================
# ABLATION GROUPS
# Each entry: (run_name, config_overrides_dict, filter_fn_or_None)
# ====================================================================================

# Window step ablation: 100 ms window fixed, step varies
STEP_ABLATION = [
    ('step_20ms',  {'INCREMENT': 20,  'ON_THE_FLY_STEP_SIZE': 20},                  None),
    ('step_10ms',  {'INCREMENT': 10,  'ON_THE_FLY_STEP_SIZE': 10},                      None),
    ('step_4ms', {'INCREMENT': 4, 'ON_THE_FLY_STEP_SIZE': 4},                     None),
    ('step_15ms',  {'INCREMENT': 15,  'ON_THE_FLY_STEP_SIZE': 15},                      None),
]

# Window size ablation: increment scaled by 0.124 (Rivela et al. factor), rounded
# 500 * 0.124 = 62, 250 * 0.124 = 31, 150 * 0.124 ≈ 19, 100 * 0.124 ≈ 12
WINDOW_ABLATION = [
    ('window_500ms',{'WINDOW_SIZE': 500, 'ON_THE_FLY_WINDOW_SIZE': 500,
                      'INCREMENT': 62,  'ON_THE_FLY_STEP_SIZE': 62},                                None),
    ('window_300ms', {'WINDOW_SIZE': 300, 'ON_THE_FLY_WINDOW_SIZE': 300,
                      'INCREMENT': 37,  'ON_THE_FLY_STEP_SIZE': 37},                                None),
    ('window_200ms', {'WINDOW_SIZE': 200, 'ON_THE_FLY_WINDOW_SIZE': 200,
                      'INCREMENT': 25,  'ON_THE_FLY_STEP_SIZE': 25},                                None),
    ('window_150ms', {'WINDOW_SIZE': 150, 'ON_THE_FLY_WINDOW_SIZE': 150,
                      'INCREMENT': 19,  'ON_THE_FLY_STEP_SIZE': 19},                                None),
    ('window_100ms', {'WINDOW_SIZE': 100, 'ON_THE_FLY_WINDOW_SIZE': 100,
                      'INCREMENT': 12,  'ON_THE_FLY_STEP_SIZE': 12},                                None),
]

# Augmentation ablation: additive from no-augmentation control
AUGMENTATION_ABLATION = [
    ('aug_none',       {'MIXUP_RATIO': 0.0, 'REST_MIXUP_RATIO': 0.0,
                        'TRAINING_NOISE_MAGNITUDES': [], 'ENABLE_MAGNITUDE_WARPING': False},         None),
    ('aug_mixup_only', {'TRAINING_NOISE_MAGNITUDES': [], 'ENABLE_MAGNITUDE_WARPING': False},         None),
    ('aug_mw_only',    {'MIXUP_RATIO': 0.0, 'REST_MIXUP_RATIO': 0.0,
                        'TRAINING_NOISE_MAGNITUDES': []},                                            None),
    ('aug_noise_only', {'MIXUP_RATIO': 0.0, 'REST_MIXUP_RATIO': 0.0,
                        'ENABLE_MAGNITUDE_WARPING': False},                                          None),
    ('aug_all',        {},                                                                           None),
]

# Filter ablation: training pipeline varies; validation always uses standard inference chain
FILTER_ABLATION = [
    ('filter_base',             {},   None),
    ('filter_notch_only',       {},   _filter_burst_notch_only),
    ('filter_bandpass_only',    {},   _filter_burst_bandpass_only),
    ('filter_wavelet',          {},   _filter_burst_wavelet),
    ('filter_naive_minmax',     {},   _filter_burst_naive_minmax),
    ('filter_no_rectification', {},   _filter_burst_no_rectification),
    ('filter_no_filtering',     {},   _filter_burst_no_filtering),
]

# Architecture ablation: base config (500 ms / 62 ms, full filter pipeline, all augs ON);
# only the model architecture varies. Run name encodes the registry key.
# Entries are 4-tuples: (run_name, config_overrides, filter_fn, model_type).
ARCH_ABLATION = [
    # ('arch_rcnn',      {}, None, 'rcnn'),
    # ('arch_rnn',       {}, None, 'rnn'),
    ('arch_lstm',      {}, None, 'lstm'),
    ('arch_1dcnn',     {}, None, '1dcnn'),
    ('arch_naive_ann', {}, None, 'naive_ann'),
]

ALL_GROUPS = {
    'step':    STEP_ABLATION,
    'window':  WINDOW_ABLATION,
    'augment': AUGMENTATION_ABLATION,
    'filter':  FILTER_ABLATION,
    'arch':    ARCH_ABLATION,
}

# ====================================================================================
# ENTRY POINT
# ====================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sequential ablation sweep for FYP report results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--groups', nargs='+',
        default=['step', 'window', 'augment', 'filter', 'arch'],
        choices=list(ALL_GROUPS.keys()),
        help='Ablation groups to run (default: all).',
    )
    args = parser.parse_args()

    total = sum(len(ALL_GROUPS[g]) for g in args.groups)
    done = 0

    for group_name in args.groups:
        experiments = ALL_GROUPS[group_name]
        print(f"\n{'#' * 70}")
        print(f"  GROUP: {group_name.upper()}  ({len(experiments)} experiments)")
        print(f"{'#' * 70}")
        for entry in experiments:
            if len(entry) == 4:
                run_name, config_overrides, filter_fn, model_type = entry
            else:
                run_name, config_overrides, filter_fn = entry
                model_type = 'rcnn'
            done += 1
            print(f"\n[BatchSweep] Progress: {done}/{total}")
            _run_experiment(run_name, config_overrides, filter_fn, model_type=model_type)

    print(f"\n{'#' * 70}")
    print(f"  [BatchSweep] All {done} experiments complete.")
    print(f"{'#' * 70}\n")
