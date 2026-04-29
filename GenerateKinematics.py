"""
GenerateKinematics.py
Ground-truth kinematic signal generation and range correction for NN training.

Collected data
--------------
Generates a synthetic 4-DOF kinematic profile from manual window labels using a
half-cosine ramp-up → hold → ramp-down shape.  Produced at 60 Hz (matching the
future camera pipeline) then ZOH-interpolated to 1000 Hz so sample indices align
with the EMG signal.  Output: biosignal_data/collected/edited/PxMyKinematic.mat

    KINEMATICS      (n_emg_samples, 4)  – 1000 Hz angles [Yaw, Pitch, Roll, Elbow]
    KINEMATICS_60HZ (n_kin_samples, 4)  – native 60 Hz before ZOH
    LABELS          (N, 2)              – source window labels (sample indices)
    FS_KINEMATICS   scalar              – native kinematic rate (60.0 Hz)
    FS_EMG          scalar              – EMG rate (1000.0 Hz)
    TARGET_VECTOR   (4,)                – target angles from ControllerConfiguration

Training use: for a 100 ms EMG window ending at sample t, the GT target is
KINEMATICS[t, :] — the angle at the leading edge of the window.

Secondary data
--------------
Range-corrects existing 'angolospalla' kinematic signals by clamping to the valid
range implied by TARGET_MAPPING.  Direction-aware: hyperextension (M4, negative
target) clamps to [target, 0] rather than [0, target].
Output: biosignal_data/secondary/edited/Soggetto{p}/MovimentoAngS{m}_edit.mat

    angolospalla          (N,) – clamped signal (same key as source)
    angolospalla_original (N,) – original unclamped signal (reference copy)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import ControllerConfiguration as Config
from FileRepository import DataRepository, FileSelection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO = DataRepository()

FS_EMG: float = float(Config.FS)          # 1000 Hz
FS_KINEMATIC: float = 60.0                # simulated camera rate

# Participants raised their arm roughly twice as fast as the 3 s elevation window,
# so the ramp occupies ~half that time on each side.
RAMP_DURATION_S: float = 0.5
RAMP_SAMPLES_KIN: int = int(RAMP_DURATION_S * FS_KINEMATIC)  # 90 samples at 60 Hz

DOF_NAMES = ["Yaw", "Pitch", "Roll", "Elbow"]
DOF_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _primary_scalar_target(movement: int) -> float:
    """Scalar clamping target for secondary 1-D kinematic (angolospalla).

    Returns the first non-zero DOF target value, preserving sign so that
    hyperextension (negative target) clamps correctly.
    """
    target_vec = Config.TARGET_MAPPING.get(movement, [0.0, 0.0, 0.0, 0.0])
    for v in target_vec:
        if v != 0.0:
            return float(v)
    return 0.0


def _half_cosine_ramp(n_samples: int, rising: bool) -> np.ndarray:
    """Half-cosine envelope over n_samples.  rising=True → 0 to 1, False → 1 to 0."""
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, np.pi, n_samples)
    return 0.5 * (1.0 - np.cos(t)) if rising else 0.5 * (1.0 + np.cos(t))


def _build_ramp_profile_60hz(
    n_kin_samples: int,
    windows_kin: List[Tuple[int, int]],
    target_vector: np.ndarray,
    ramp_samples: int,
) -> np.ndarray:
    """(n_kin_samples, 4) kinematic profile at 60 Hz.

    Each labelled window receives a half-cosine ramp-up, a flat hold at the
    target angle, and a half-cosine ramp-down.  When a window is shorter than
    two full ramps the ramp duration is scaled proportionally.
    """
    profile = np.zeros((n_kin_samples, 4), dtype=np.float64)
    for start, end in windows_kin:
        window_len = end - start
        if window_len <= 0:
            continue
        actual_ramp = min(ramp_samples, window_len // 2)
        hold_len = window_len - 2 * actual_ramp
        scalar = np.concatenate([
            _half_cosine_ramp(actual_ramp, rising=True),
            np.ones(hold_len, dtype=np.float64),
            _half_cosine_ramp(actual_ramp, rising=False),
        ])
        profile[start:end, :] = scalar[:, np.newaxis] * target_vector[np.newaxis, :]
    return profile


def _zoh_to_emg_rate(kin_60hz: np.ndarray, n_emg_samples: int) -> np.ndarray:
    """Vectorised zero-order hold from 60 Hz kinematic to 1000 Hz EMG rate.

    Each EMG sample t maps to kinematic frame floor(t * 60 / 1000), which is
    then held until the next kinematic frame — exactly what a real camera would
    produce when sampled at 1000 Hz via a slower acquisition.
    """
    n_kin = kin_60hz.shape[0]
    emg_indices = np.arange(n_emg_samples, dtype=np.float64)
    kin_indices = np.clip(
        (emg_indices * FS_KINEMATIC / FS_EMG).astype(np.int64), 0, n_kin - 1
    )
    return kin_60hz[kin_indices, :]


def _parse_labels(mat: dict, n_samples: int) -> Optional[np.ndarray]:
    """Extract and validate LABELS from a loaded .mat dict."""
    if "LABELS" not in mat:
        return None
    raw = np.asarray(mat["LABELS"])
    if raw.size == 0:
        return None
    if raw.ndim == 1:
        raw = raw.reshape(-1, 2)
    if raw.ndim != 2 or raw.shape[1] < 2:
        return None
    labels = raw[:, :2].astype(int)
    # Keep only valid rows
    labels[:, 0] = np.clip(labels[:, 0], 0, n_samples - 1)
    labels[:, 1] = np.clip(labels[:, 1], 0, n_samples - 1)
    valid = labels[:, 1] > labels[:, 0]
    return labels[valid] if valid.any() else None


# ---------------------------------------------------------------------------
# Collected — single-file generation
# ---------------------------------------------------------------------------

def generate_collected_kinematics(
    participant: int,
    movement: int,
    show_plot: bool = False,
    save_plot: Optional[str] = None,
) -> Optional[str]:
    """Generate GT kinematic file for one collected trial.

    Reads PxMy_labelled.mat, builds a ramp profile from LABELS at 60 Hz,
    ZOH-interpolates to 1000 Hz, and writes PxMyKinematic.mat.

    Returns the output path on success, None on skip/error.
    """
    selection = FileSelection(data_type="collected", participant=participant, movement=movement)
    labelled_path = REPO.output_file_path(selection, create_dirs=False)
    if not os.path.exists(labelled_path):
        print(f"  [SKIP] P{participant}M{movement}: labelled file not found ({labelled_path})")
        return None

    try:
        mat = scipy.io.loadmat(labelled_path)
    except Exception as exc:
        print(f"  [ERROR] P{participant}M{movement}: could not load labelled file — {exc}")
        return None

    if "EMGDATA" not in mat:
        print(f"  [SKIP] P{participant}M{movement}: no EMGDATA in labelled file")
        return None

    emg_data = np.asarray(mat["EMGDATA"], dtype=np.float32)
    n_emg_samples = emg_data.shape[1]

    if movement not in Config.TARGET_MAPPING:
        print(f"  [SKIP] P{participant}M{movement}: movement {movement} not in TARGET_MAPPING")
        return None
    target_vector = np.array(Config.TARGET_MAPPING[movement], dtype=np.float64)

    labels = _parse_labels(mat, n_emg_samples)
    if labels is None:
        print(f"  [SKIP] P{participant}M{movement}: no valid LABELS found")
        return None

    # Convert EMG-rate label boundaries to 60 Hz kinematic sample indices
    n_kin_samples = int(np.ceil(n_emg_samples * FS_KINEMATIC / FS_EMG)) + 1
    windows_kin: List[Tuple[int, int]] = []
    for row in labels:
        s = int(np.clip(int(row[0]) * FS_KINEMATIC / FS_EMG, 0, n_kin_samples - 1))
        e = int(np.clip(int(row[1]) * FS_KINEMATIC / FS_EMG, 0, n_kin_samples - 1))
        if e > s:
            windows_kin.append((s, e))

    kin_60hz = _build_ramp_profile_60hz(n_kin_samples, windows_kin, target_vector, RAMP_SAMPLES_KIN)
    kin_1000hz = _zoh_to_emg_rate(kin_60hz, n_emg_samples)

    out_dir = REPO.edited_root("collected")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"P{participant}M{movement}Kinematic.mat")

    scipy.io.savemat(out_path, {
        "KINEMATICS":      kin_1000hz,
        "KINEMATICS_60HZ": kin_60hz,
        "FS_KINEMATICS":   np.array([[FS_KINEMATIC]]),
        "FS_EMG":          np.array([[FS_EMG]]),
        "LABELS":          labels,
        "TARGET_VECTOR":   target_vector,
    })

    print(
        f"  [SAVED] P{participant}M{movement}Kinematic.mat — "
        f"{len(labels)} window(s), {n_emg_samples} EMG samples, "
        f"target {target_vector.tolist()}"
    )

    if show_plot or save_plot:
        _visualise_collected(emg_data, kin_1000hz, labels, participant, movement, show_plot, save_plot)

    return out_path


# ---------------------------------------------------------------------------
# Secondary — single-file range correction
# ---------------------------------------------------------------------------

def process_secondary_kinematics(
    participant: int,
    movement: int,
    show_plot: bool = False,
    save_plot: Optional[str] = None,
) -> Optional[str]:
    """Clamp secondary angolospalla to the valid range and save as _edit.mat.

    Returns the output path on success, None on skip/error.
    """
    kin_path = REPO.secondary_kinematics_file_path(participant, movement)
    if not os.path.exists(kin_path):
        print(f"  [SKIP] S{participant} M{movement}: kinematic file not found ({kin_path})")
        return None

    try:
        mat = scipy.io.loadmat(kin_path)
    except Exception as exc:
        print(f"  [ERROR] S{participant} M{movement}: could not load — {exc}")
        return None

    if "angolospalla" not in mat:
        print(f"  [SKIP] S{participant} M{movement}: no 'angolospalla' key")
        return None

    original = np.asarray(mat["angolospalla"], dtype=np.float64).flatten()

    scalar_target = _primary_scalar_target(movement)
    lo = min(0.0, scalar_target)
    hi = max(0.0, scalar_target)
    clamped = np.clip(original, lo, hi)

    out_dir = os.path.join(REPO.edited_root("secondary"), f"Soggetto{participant}")
    os.makedirs(out_dir, exist_ok=True)

    src_stem = os.path.splitext(os.path.basename(kin_path))[0]
    out_path = os.path.join(out_dir, f"{src_stem}_edit.mat")

    payload = {k: v for k, v in mat.items() if not str(k).startswith("__")}
    payload["angolospalla"] = clamped.reshape(-1, 1)
    payload["angolospalla_original"] = original.reshape(-1, 1)
    scipy.io.savemat(out_path, payload)

    n_clamped = int(np.sum(original != clamped))
    print(
        f"  [SAVED] S{participant} M{movement} → {os.path.basename(out_path)}  "
        f"original [{original.min():.1f}°, {original.max():.1f}°] → "
        f"clamped [{lo:.1f}°, {hi:.1f}°]  ({n_clamped} samples adjusted)"
    )

    if show_plot or save_plot:
        _visualise_secondary(original, clamped, participant, movement, lo, hi, show_plot, save_plot)

    return out_path


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------

def batch_generate_collected(
    participants: Optional[List[int]] = None,
    show_plots: bool = False,
) -> None:
    """Generate kinematic GT files for all available collected trials."""
    if participants is None:
        participants = REPO.discover_participants("collected")
    print(f"\n[Batch] Collected kinematics — participants: {participants}")

    n_ok = n_skip = 0
    for p in participants:
        print(f"\n  Participant P{p}:")
        for m in REPO.discover_movements("collected", p):
            if hasattr(Config, "COLLECTED_BLACKLIST") and (p, m) in Config.COLLECTED_BLACKLIST:
                print(f"  [SKIP] P{p}M{m}: blacklisted")
                n_skip += 1
                continue
            result = generate_collected_kinematics(p, m, show_plot=show_plots)
            if result:
                n_ok += 1
            else:
                n_skip += 1

    print(f"\n[Batch] Done. Generated: {n_ok}  Skipped/Error: {n_skip}")


def batch_process_secondary(
    participants: Optional[List[int]] = None,
    show_plots: bool = False,
) -> None:
    """Range-correct secondary kinematic files for all available subjects."""
    if participants is None:
        participants = REPO.discover_participants("secondary")
    print(f"\n[Batch] Secondary kinematics — participants: {participants}")

    n_ok = n_skip = 0
    for p in participants:
        print(f"\n  Soggetto{p}:")
        for m in REPO.discover_movements("secondary", p):
            if hasattr(Config, "SECONDARY_BLACKLIST") and (p, m) in Config.SECONDARY_BLACKLIST:
                print(f"  [SKIP] S{p} M{m}: blacklisted")
                n_skip += 1
                continue
            result = process_secondary_kinematics(p, m, show_plot=show_plots)
            if result:
                n_ok += 1
            else:
                n_skip += 1

    print(f"\n[Batch] Done. Processed: {n_ok}  Skipped/Error: {n_skip}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _visualise_collected(
    emg_data: np.ndarray,
    kin_1000hz: np.ndarray,
    labels: np.ndarray,
    participant: int,
    movement: int,
    show: bool,
    save_path: Optional[str],
) -> None:
    """Two-panel figure: EMG envelope + window overlays (top), 4-DOF kinematics (bottom)."""
    time_s = np.arange(emg_data.shape[1]) / FS_EMG

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"P{participant}M{movement} — GT Kinematic Verification  "
        f"({Config.MOVEMENT_NAMES.get(movement, f'M{movement}')})",
        fontsize=12,
    )

    # --- EMG envelope ---
    envelope = np.sum(np.abs(emg_data), axis=0)
    envelope_norm = envelope / (envelope.max() + 1e-9)
    axes[0].plot(time_s, envelope_norm, color="steelblue", linewidth=0.7, label="EMG envelope (norm)")
    for row in labels:
        axes[0].axvspan(row[0] / FS_EMG, row[1] / FS_EMG, alpha=0.25, color="orange")
    axes[0].set_ylabel("Normalised amplitude")
    axes[0].set_title("EMG envelope + labelled windows")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.4)

    # --- Generated kinematics ---
    active_dofs = [i for i in range(4) if np.any(kin_1000hz[:, i] != 0)]
    for dof in active_dofs:
        axes[1].plot(time_s, kin_1000hz[:, dof], color=DOF_COLOURS[dof], linewidth=1.0, label=DOF_NAMES[dof])
    for row in labels:
        axes[1].axvspan(row[0] / FS_EMG, row[1] / FS_EMG, alpha=0.12, color="orange")
    axes[1].set_ylabel("Angle (degrees)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Generated GT kinematics — 1000 Hz (ZOH from 60 Hz)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [PLOT] Saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _visualise_secondary(
    original: np.ndarray,
    clamped: np.ndarray,
    participant: int,
    movement: int,
    lo: float,
    hi: float,
    show: bool,
    save_path: Optional[str],
) -> None:
    """Single-panel figure showing original vs clamped secondary kinematic."""
    sample_axis = np.arange(len(original))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sample_axis, original, color="steelblue", linewidth=0.7, alpha=0.8, label="Original")
    ax.plot(sample_axis, clamped, color="tab:orange", linewidth=1.0, label=f"Clamped [{lo:.0f}°, {hi:.0f}°]")
    ax.axhline(lo, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label=f"Lo = {lo:.0f}°")
    ax.axhline(hi, color="green", linestyle="--", linewidth=0.8, alpha=0.6, label=f"Hi = {hi:.0f}°")
    ax.set_xlabel("Kinematic sample index")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(
        f"Soggetto{participant} M{movement} — Range correction  "
        f"({Config.MOVEMENT_NAMES.get(movement, f'M{movement}')})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [PLOT] Saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    MODE = "batch_secondary"   # Options: batch_collected | batch_secondary | single_collected | single_secondary

    if MODE == "batch_collected":
        batch_generate_collected(show_plots=False)

    elif MODE == "batch_secondary":
        batch_process_secondary(show_plots=False)

    elif MODE == "single_collected":
        # Quick QA: generate one file and show the verification plot
        P, M = 4, 2
        generate_collected_kinematics(P, M, show_plot=True)

    elif MODE == "single_secondary":
        # Quick QA: range-correct one file and show the comparison plot
        P, M = 1, 1
        process_secondary_kinematics(P, M, show_plot=True)

    else:
        print(f"Unknown MODE '{MODE}'. Set MODE at the top of __main__.", file=sys.stderr)
        sys.exit(1)
