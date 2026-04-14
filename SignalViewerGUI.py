"""
SignalViewerGUI.py

Interactive GUI for side-by-side EMG signal comparison with a shared processing
pipeline and synchronized time axis.

Key features:
- Primary (blue) signal + secondary (gray, behind) overlay in time domain
- Primary/secondary spectrograms stacked vertically
- Bottom minimap slider (SpanSelector) to control zoom window
- Strictly ordered filtering pipeline applied to BOTH signals
- Optional white-noise injection (selected channel only, primary signal)
- Live readouts: SNR, pre-normalization peak amplitude, MNF/MDF
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.io
import scipy.signal

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

import ControllerConfiguration as Config
import DataPreparation
import SignalProcessing
from FileRepository import DataRepository
from LabelSignalData import EMGSignalProcessor


EPS = 1e-12


@dataclass
class LoadedSignal:
    file_path: str
    raw_data: np.ndarray  # Shape: [num_channels, num_samples]
    robust_minmax: Optional[np.ndarray]


class SignalViewerGUI(QMainWindow):
    def __init__(self, primary_path: Optional[str] = None, secondary_path: Optional[str] = None) -> None:
        super().__init__()

        self.setWindowTitle("Signal Viewer GUI - Primary vs Secondary")
        self.resize(1600, 980)

        self.fs = float(getattr(Config, "FS", 1000.0))
        self.num_channels = int(getattr(Config, "NUM_CHANNELS", 8))
        self.channel_map = dict(getattr(Config, "CHANNEL_MAP", {i: f"Channel {i}" for i in range(self.num_channels)}))

        self.signal_processor = EMGSignalProcessor(fs=self.fs)
        self.repo = DataRepository()

        self.primary_signal: Optional[LoadedSignal] = None
        self.secondary_signal: Optional[LoadedSignal] = None

        self.current_xlim: Tuple[float, float] = (0.0, 100.0)

        self._build_ui()
        self._connect_signals()

        if primary_path:
            self._load_primary(primary_path)
        if secondary_path:
            self._load_secondary(secondary_path)

        self._refresh_view()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)

        controls_group = QGroupBox("Signal Controls")
        controls_layout = QGridLayout(controls_group)

        self.primary_path_edit = QLineEdit()
        self.primary_path_edit.setReadOnly(True)
        self.secondary_path_edit = QLineEdit()
        self.secondary_path_edit.setReadOnly(True)

        self.load_primary_btn = QPushButton("Load Primary")
        self.load_secondary_btn = QPushButton("Load Secondary")
        self.clear_secondary_btn = QPushButton("Clear Secondary")

        self.channel_combo = QComboBox()
        for c in range(self.num_channels):
            self.channel_combo.addItem(f"{c}: {self.channel_map.get(c, f'Channel {c}')}", c)

        self.toggle_notch = QCheckBox("Notch")
        self.toggle_notch.setChecked(True)
        self.toggle_bandpass = QCheckBox("Bandpass")
        self.toggle_bandpass.setChecked(True)
        self.toggle_rectify = QCheckBox("Rectify")
        self.toggle_rectify.setChecked(True)
        self.toggle_normalize = QCheckBox("Normalize")
        self.toggle_normalize.setChecked(True)

        self.toggle_show_windows = QCheckBox("Show Windows")
        self.toggle_show_windows.setChecked(False)

        self.toggle_lock_spec_colors = QCheckBox("Lock Spectrogram Colors")
        self.toggle_lock_spec_colors.setChecked(False)

        self.toggle_inject_noise = QCheckBox("Inject Noise (Primary)")
        self.toggle_inject_noise.setChecked(False)

        self.noise_magnitude_edit = QLineEdit("0.0")
        self.noise_magnitude_edit.setFixedWidth(110)
        self.noise_magnitude_edit.setToolTip(
            "White noise standard deviation for the selected channel. Applied before filtering."
        )

        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.refresh_btn = QPushButton("Refresh")

        controls_layout.addWidget(QLabel("Primary"), 0, 0)
        controls_layout.addWidget(self.primary_path_edit, 0, 1, 1, 4)
        controls_layout.addWidget(self.load_primary_btn, 0, 5)

        controls_layout.addWidget(QLabel("Secondary"), 1, 0)
        controls_layout.addWidget(self.secondary_path_edit, 1, 1, 1, 4)
        controls_layout.addWidget(self.load_secondary_btn, 1, 5)
        controls_layout.addWidget(self.clear_secondary_btn, 1, 6)

        controls_layout.addWidget(QLabel("Channel"), 2, 0)
        controls_layout.addWidget(self.channel_combo, 2, 1)

        controls_layout.addWidget(self.toggle_notch, 2, 2)
        controls_layout.addWidget(self.toggle_bandpass, 2, 3)
        controls_layout.addWidget(self.toggle_rectify, 2, 4)
        controls_layout.addWidget(self.toggle_normalize, 2, 5)

        controls_layout.addWidget(self.toggle_show_windows, 3, 1)
        controls_layout.addWidget(self.toggle_lock_spec_colors, 3, 2)

        controls_layout.addWidget(self.toggle_inject_noise, 3, 3)
        controls_layout.addWidget(QLabel("Noise Magnitude"), 3, 4)
        controls_layout.addWidget(self.noise_magnitude_edit, 3, 5)

        controls_layout.addWidget(self.reset_zoom_btn, 2, 6)
        controls_layout.addWidget(self.refresh_btn, 3, 6)

        root_layout.addWidget(controls_group)

        self.figure = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.figure)
        root_layout.addWidget(self.canvas, stretch=1)

        gs = self.figure.add_gridspec(4, 1, height_ratios=[2.1, 1.2, 1.2, 0.7], hspace=0.08)
        self.ax_main = self.figure.add_subplot(gs[0, 0])
        self.ax_spec_primary = self.figure.add_subplot(gs[1, 0], sharex=self.ax_main)
        self.ax_spec_secondary = self.figure.add_subplot(gs[2, 0], sharex=self.ax_main)
        self.ax_minimap = self.figure.add_subplot(gs[3, 0], sharex=self.ax_main)

        self.span_selector = SpanSelector(
            self.ax_minimap,
            self._on_minimap_select,
            "horizontal",
            useblit=True,
            props=dict(facecolor="tab:blue", alpha=0.18),
        )

        readout_group = QGroupBox("Live Readouts")
        readout_layout = QHBoxLayout(readout_group)

        self.snr_label = QLabel("Current SNR (dB): n/a")
        self.peak_label = QLabel("Peak Amplitude (pre-normalization): n/a")
        self.freq_label = QLabel("MNF/MDF (Hz): n/a")

        readout_layout.addWidget(self.snr_label)
        readout_layout.addStretch(1)
        readout_layout.addWidget(self.peak_label)
        readout_layout.addStretch(1)
        readout_layout.addWidget(self.freq_label)

        root_layout.addWidget(readout_group)

    def _connect_signals(self) -> None:
        self.load_primary_btn.clicked.connect(self._browse_primary)
        self.load_secondary_btn.clicked.connect(self._browse_secondary)
        self.clear_secondary_btn.clicked.connect(self._clear_secondary)

        self.channel_combo.currentIndexChanged.connect(lambda _=None: self._refresh_view())

        self.toggle_notch.toggled.connect(lambda _=None: self._refresh_view())
        self.toggle_bandpass.toggled.connect(lambda _=None: self._refresh_view())
        self.toggle_rectify.toggled.connect(lambda _=None: self._refresh_view())
        self.toggle_normalize.toggled.connect(lambda _=None: self._refresh_view())
        self.toggle_show_windows.toggled.connect(lambda _=None: self._refresh_view())
        self.toggle_lock_spec_colors.toggled.connect(lambda _=None: self._refresh_view())

        self.toggle_inject_noise.toggled.connect(lambda _=None: self._refresh_view())
        self.noise_magnitude_edit.editingFinished.connect(self._refresh_view)

        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        self.refresh_btn.clicked.connect(self._refresh_view)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _browse_primary(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Primary .mat File",
            getattr(Config, "BASE_DATA_PATH", "."),
            "MAT Files (*.mat)",
        )
        if file_path:
            self._load_primary(file_path)

    def _browse_secondary(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Secondary .mat File",
            getattr(Config, "BASE_DATA_PATH", "."),
            "MAT Files (*.mat)",
        )
        if file_path:
            self._load_secondary(file_path)

    def _clear_secondary(self) -> None:
        self.secondary_signal = None
        self.secondary_path_edit.clear()
        self._refresh_view()

    def _load_primary(self, file_path: str) -> None:
        try:
            self.primary_signal = self._load_signal(file_path)
            self.primary_path_edit.setText(file_path)
            self._refresh_view()
        except Exception as exc:
            QMessageBox.critical(self, "Primary Load Error", str(exc))

    def _load_secondary(self, file_path: str) -> None:
        try:
            self.secondary_signal = self._load_signal(file_path)
            self.secondary_path_edit.setText(file_path)
            self._refresh_view()
        except Exception as exc:
            QMessageBox.critical(self, "Secondary Load Error", str(exc))

    def _load_signal(self, file_path: str) -> LoadedSignal:
        mat = scipy.io.loadmat(file_path)
        if "EMGDATA" not in mat:
            raise ValueError(f"EMGDATA missing in {file_path}")

        raw = np.asarray(mat["EMGDATA"], dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"EMGDATA in {file_path} is not 2D (shape={raw.shape})")

        if raw.shape[0] != self.num_channels and raw.shape[1] == self.num_channels:
            raw = raw.T

        if raw.shape[0] != self.num_channels:
            raise ValueError(
                f"Unexpected channel count in {file_path}: got {raw.shape[0]}, expected {self.num_channels}"
            )

        robust = DataPreparation._extract_robust_minmax_matrix(mat)
        if robust is None:
            candidate = self.repo.labelled_candidate_path(file_path)
            if candidate and os.path.exists(candidate):
                try:
                    labelled_mat = scipy.io.loadmat(candidate)
                    robust = DataPreparation._extract_robust_minmax_matrix(labelled_mat)
                except Exception:
                    robust = None

        return LoadedSignal(file_path=file_path, raw_data=raw, robust_minmax=robust)

    # ------------------------------------------------------------------
    # Processing and plotting
    # ------------------------------------------------------------------
    def _current_channel(self) -> int:
        return int(self.channel_combo.currentData())

    def _noise_magnitude(self) -> float:
        text = self.noise_magnitude_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            self.noise_magnitude_edit.setStyleSheet("border: 1px solid red;")
            return 0.0

        self.noise_magnitude_edit.setStyleSheet("")
        return max(0.0, value)

    def _run_pipeline(
        self,
        loaded: LoadedSignal,
        inject_noise_on_selected_channel: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            normalized (np.ndarray)
            pre_normalization_processed (np.ndarray)
            clean_raw (np.ndarray)
            noisy_raw (np.ndarray)
        """
        clean_raw = np.asarray(loaded.raw_data, dtype=np.float32)
        noisy_raw = clean_raw.copy()

        # IMPORTANT: Noise is injected BEFORE any filtering.
        if inject_noise_on_selected_channel and self.toggle_inject_noise.isChecked():
            noise_mag = self._noise_magnitude()
            if noise_mag > 0.0:
                channel = self._current_channel()
                noisy_raw = SignalProcessing.inject_white_noise_to_channels(
                    noisy_raw,
                    target_channels=[channel],
                    noise_magnitudes=[noise_mag],
                )

        working = noisy_raw.copy()

        # IMPORTANT: Always apply the active filters in this exact order.
        # Order: Notch -> Bandpass -> Rectification
        for ch in range(working.shape[0]):
            sig = working[ch, :]

            if self.toggle_notch.isChecked():
                sig = SignalProcessing.notchFilter(
                    sig,
                    fs=self.fs,
                    notchFreq=float(getattr(Config, "NOTCH_FREQ", 50.0)),
                    qualityFactor=float(getattr(Config, "NOTCH_QUALITY", 30.0)),
                )

            if self.toggle_bandpass.isChecked():
                sig = SignalProcessing.bandpassFilter(
                    sig,
                    fs=self.fs,
                    lowCut=float(getattr(Config, "BANDPASS_LOW", 30.0)),
                    highCut=float(getattr(Config, "BANDPASS_HIGH", 450.0)),
                    order=int(getattr(Config, "FILTER_ORDER", 4)),
                )

            if self.toggle_rectify.isChecked():
                sig = SignalProcessing.rectifySignal(sig)

            working[ch, :] = np.asarray(sig, dtype=np.float32)

        pre_norm = working.copy()

        # Apply normalization only if toggle is enabled
        if self.toggle_normalize.isChecked():
            normalized = self._normalize_for_alignment(pre_norm, loaded.robust_minmax)
        else:
            normalized = pre_norm.copy()

        return normalized, pre_norm, clean_raw, noisy_raw

    def _normalize_for_alignment(self, signal: np.ndarray, robust_minmax: Optional[np.ndarray]) -> np.ndarray:
        """
        Enforces normalization on all signals so differing recordings can be aligned.
        """
        rectified = self.toggle_rectify.isChecked()

        if robust_minmax is not None:
            if rectified:
                return SignalProcessing.applyRobustRectifiedNormalization(signal, robust_minmax)
            return SignalProcessing.applyRobustUnrectifiedNormalization(signal, robust_minmax)

        # Fallback when robust bounds are unavailable.
        if rectified:
            return SignalProcessing.applyGlobalNormalization(signal, percentiles=(1.0, 99.0))

        normalized = np.zeros_like(signal, dtype=np.float32)
        for ch in range(signal.shape[0]):
            normalized[ch, :] = SignalProcessing.normaliseSignal(
                signal[ch, :],
                output_range=(-1.0, 1.0),
                percentiles=(1.0, 99.0),
            )
        return normalized

    @staticmethod
    def _percent_time_axis(num_samples: int) -> np.ndarray:
        if num_samples <= 1:
            return np.array([0.0], dtype=np.float32)
        return np.linspace(0.0, 100.0, num_samples, endpoint=True, dtype=np.float32)

    def _compute_spectrogram_db(self, signal_1d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if signal_1d.size < 16:
            return None

        try:
            times, freqs, power = self.signal_processor.compute_spectrogram(signal_1d, freq_max=150.0)
        except Exception:
            return None

        duration_sec = max(signal_1d.size / self.fs, EPS)
        times_pct = (times / duration_sec) * 100.0
        power_db = 10.0 * np.log10(np.maximum(power, EPS))
        return times_pct, freqs, power_db

    @staticmethod
    def _compute_frequency_features(signal_1d: np.ndarray, fs: float) -> Tuple[float, float]:
        if signal_1d.size < 16:
            return np.nan, np.nan

        nperseg = min(512, signal_1d.size)
        freqs, power = scipy.signal.welch(signal_1d, fs=fs, nperseg=nperseg)

        total_power = float(np.sum(power))
        if total_power <= EPS:
            return np.nan, np.nan

        mnf = float(np.sum(freqs * power) / total_power)

        cdf = np.cumsum(power)
        half_power = 0.5 * cdf[-1]
        idx = int(np.searchsorted(cdf, half_power))
        idx = max(0, min(idx, len(freqs) - 1))
        mdf = float(freqs[idx])

        return mnf, mdf

    @staticmethod
    def _compute_snr_db(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        noise = noisy_signal - clean_signal
        signal_power = float(np.mean(clean_signal ** 2))
        noise_power = float(np.mean(noise ** 2))

        signal_power = max(signal_power, EPS)
        noise_power = max(noise_power, EPS)
        return float(10.0 * np.log10(signal_power / noise_power))

    def _window_spans_pct(self) -> list:
        """Load windows from LABELS in mat file first, fall back to burst detection if missing."""
        if self.primary_signal is None:
            return []

        # Try to load LABELS from the mat file first
        try:
            mat = scipy.io.loadmat(self.primary_signal.file_path)
            parsed_windows = self.signal_processor._parse_labels(
                mat.get("LABELS"),
                n_samples=self.primary_signal.raw_data.shape[1]
            )
            if parsed_windows:
                n = max(1, self.primary_signal.raw_data.shape[1] - 1)
                spans = []
                for start, end in parsed_windows:
                    start_pct = (float(start) / n) * 100.0
                    end_pct = (float(end) / n) * 100.0
                    spans.append((start_pct, end_pct))
                return spans
        except Exception:
            pass

        # Fall back to burst detection if LABELS are missing or invalid
        windows = self.signal_processor.detect_burst_windows(self.primary_signal.raw_data)
        if not windows:
            return []

        n = max(1, self.primary_signal.raw_data.shape[1] - 1)
        spans = []
        for start, end in windows:
            start_pct = (float(start) / n) * 100.0
            end_pct = (float(end) / n) * 100.0
            spans.append((start_pct, end_pct))
        return spans

    def _refresh_view(self) -> None:
        self.ax_main.clear()
        self.ax_spec_primary.clear()
        self.ax_spec_secondary.clear()
        self.ax_minimap.clear()

        self.ax_main.set_title("Primary (blue) with Secondary (gray) overlay")
        self.ax_main.set_ylabel("Normalized Amplitude")
        self.ax_spec_primary.set_ylabel("Freq (Hz)")
        self.ax_spec_secondary.set_ylabel("Freq (Hz)")
        self.ax_minimap.set_ylabel("Overview")
        self.ax_minimap.set_xlabel("Normalized Time (%)")

        if self.primary_signal is None:
            self.ax_main.text(
                0.5,
                0.5,
                "Load a primary signal to begin.",
                transform=self.ax_main.transAxes,
                ha="center",
                va="center",
            )
            self.canvas.draw_idle()
            return

        channel = self._current_channel()

        primary_norm, primary_pre_norm, primary_clean_raw, primary_noisy_raw = self._run_pipeline(
            self.primary_signal,
            inject_noise_on_selected_channel=True,
        )

        secondary_norm = None
        secondary_pre_norm = None
        if self.secondary_signal is not None:
            secondary_norm, secondary_pre_norm, _, _ = self._run_pipeline(
                self.secondary_signal,
                inject_noise_on_selected_channel=False,
            )

        primary_ch = primary_norm[channel, :]
        primary_ch_pre_norm = primary_pre_norm[channel, :]
        x_primary = self._percent_time_axis(primary_ch.size)

        stack_for_ylim = [primary_ch]

        # Secondary signal is intentionally drawn first so it stays behind the primary.
        if secondary_norm is not None:
            secondary_ch = secondary_norm[channel, :]
            x_secondary = self._percent_time_axis(secondary_ch.size)

            self.ax_main.fill_between(
                x_secondary,
                secondary_ch,
                0.0,
                color="0.80",
                alpha=0.35,
                zorder=0,
            )
            self.ax_main.plot(
                x_secondary,
                secondary_ch,
                color="0.45",
                linewidth=1.2,
                alpha=0.95,
                zorder=1,
                label="Secondary",
            )
            stack_for_ylim.append(secondary_ch)

        self.ax_main.plot(
            x_primary,
            primary_ch,
            color="tab:blue",
            linewidth=1.6,
            zorder=2,
            label="Primary",
        )

        if self.toggle_show_windows.isChecked():
            for start_pct, end_pct in self._window_spans_pct():
                self.ax_main.axvspan(start_pct, end_pct, color="tab:blue", alpha=0.11, zorder=-1)

            if secondary_norm is not None:
                secondary_ch = secondary_norm[channel, :]
                # For secondary, fall back to burst detection since we don't load secondary file labels
                windows = self.signal_processor.detect_burst_windows(secondary_norm)
                if windows:
                    n = max(1, secondary_ch.shape[0] - 1)
                    for start, end in windows:
                        start_pct = (float(start) / n) * 100.0
                        end_pct = (float(end) / n) * 100.0
                        self.ax_main.axvspan(start_pct, end_pct, color="0.50", alpha=0.07, zorder=-2)

        y_min = min(float(np.min(arr)) for arr in stack_for_ylim)
        y_max = max(float(np.max(arr)) for arr in stack_for_ylim)
        y_pad = max((y_max - y_min) * 0.10, 0.05)

        self.ax_main.set_xlim(*self.current_xlim)
        self.ax_main.set_ylim(y_min - y_pad, y_max + y_pad)
        self.ax_main.grid(True, linestyle=":", alpha=0.6)
        self.ax_main.legend(loc="upper right")

        # Spectrograms
        primary_spec = self._compute_spectrogram_db(primary_ch)
        secondary_spec = self._compute_spectrogram_db(secondary_norm[channel, :]) if secondary_norm is not None else None

        lock_colors = self.toggle_lock_spec_colors.isChecked() and primary_spec is not None and secondary_spec is not None
        global_vmin = None
        global_vmax = None
        if lock_colors:
            global_vmin = float(min(np.min(primary_spec[2]), np.min(secondary_spec[2])))
            global_vmax = float(max(np.max(primary_spec[2]), np.max(secondary_spec[2])))

        self._draw_spec_axis(
            ax=self.ax_spec_primary,
            spec_tuple=primary_spec,
            title="Primary Spectrogram",
            vmin=global_vmin,
            vmax=global_vmax,
        )

        self._draw_spec_axis(
            ax=self.ax_spec_secondary,
            spec_tuple=secondary_spec,
            title="Secondary Spectrogram",
            vmin=global_vmin,
            vmax=global_vmax,
        )

        # Minimap
        self.ax_minimap.plot(x_primary, primary_ch, color="tab:blue", linewidth=0.9, alpha=0.9)
        if secondary_norm is not None:
            secondary_ch = secondary_norm[channel, :]
            x_secondary = self._percent_time_axis(secondary_ch.size)
            self.ax_minimap.plot(x_secondary, secondary_ch, color="0.50", linewidth=0.9, alpha=0.85)

        self.ax_minimap.set_xlim(0.0, 100.0)
        self.ax_minimap.set_ylim(y_min - y_pad, y_max + y_pad)
        self.ax_minimap.grid(True, linestyle=":", alpha=0.5)

        # Keep all x-axes aligned.
        self.ax_spec_primary.set_xlim(*self.current_xlim)
        self.ax_spec_secondary.set_xlim(*self.current_xlim)

        self._update_readouts(
            channel=channel,
            primary_pre_norm=primary_ch_pre_norm,
            secondary_pre_norm=secondary_pre_norm[channel, :] if secondary_pre_norm is not None else None,
            primary_clean_raw=primary_clean_raw[channel, :],
            primary_noisy_raw=primary_noisy_raw[channel, :],
        )

        self.canvas.draw_idle()

    def _draw_spec_axis(
        self,
        ax,
        spec_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        title: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> None:
        ax.set_title(title)
        ax.set_ylabel("Freq (Hz)")
        ax.grid(False)

        if spec_tuple is None:
            ax.text(0.5, 0.5, "No signal", transform=ax.transAxes, ha="center", va="center")
            return

        times_pct, freqs, power_db = spec_tuple
        ax.pcolormesh(
            times_pct,
            freqs,
            power_db,
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_ylim(0.0, 150.0)
        ax.set_xlim(*self.current_xlim)

    def _update_readouts(
        self,
        channel: int,
        primary_pre_norm: np.ndarray,
        secondary_pre_norm: Optional[np.ndarray],
        primary_clean_raw: np.ndarray,
        primary_noisy_raw: np.ndarray,
    ) -> None:
        # Current SNR (dB)
        if self.toggle_inject_noise.isChecked() and self._noise_magnitude() > 0.0:
            snr_db = self._compute_snr_db(primary_clean_raw, primary_noisy_raw)
            snr_text = f"Primary Ch{channel}: {snr_db:.2f} dB"
        else:
            snr_text = "Primary: noise injection OFF"
        self.snr_label.setText(f"Current SNR (dB): {snr_text}")

        # Peak amplitude before normalization
        primary_peak = float(np.max(np.abs(primary_pre_norm))) if primary_pre_norm.size else np.nan
        if secondary_pre_norm is not None and secondary_pre_norm.size:
            secondary_peak = float(np.max(np.abs(secondary_pre_norm)))
            peak_text = f"Primary {primary_peak:.6f} | Secondary {secondary_peak:.6f}"
        else:
            peak_text = f"Primary {primary_peak:.6f} | Secondary n/a"
        self.peak_label.setText(f"Peak Amplitude (pre-normalization): {peak_text}")

        # Mean/Median Frequency
        p_mnf, p_mdf = self._compute_frequency_features(primary_pre_norm, self.fs)

        if secondary_pre_norm is not None and secondary_pre_norm.size:
            s_mnf, s_mdf = self._compute_frequency_features(secondary_pre_norm, self.fs)
            freq_text = (
                f"Primary MNF/MDF {p_mnf:.1f}/{p_mdf:.1f} Hz | "
                f"Secondary MNF/MDF {s_mnf:.1f}/{s_mdf:.1f} Hz"
            )
        else:
            freq_text = f"Primary MNF/MDF {p_mnf:.1f}/{p_mdf:.1f} Hz | Secondary n/a"

        self.freq_label.setText(f"MNF/MDF (Hz): {freq_text}")

    # ------------------------------------------------------------------
    # Minimap slider
    # ------------------------------------------------------------------
    def _on_minimap_select(self, xmin: float, xmax: float) -> None:
        start = max(0.0, min(float(xmin), float(xmax)))
        end = min(100.0, max(float(xmin), float(xmax)))

        if end - start < 0.5:
            return

        self.current_xlim = (start, end)
        self.ax_main.set_xlim(*self.current_xlim)
        self.ax_spec_primary.set_xlim(*self.current_xlim)
        self.ax_spec_secondary.set_xlim(*self.current_xlim)
        self.canvas.draw_idle()

    def _reset_zoom(self) -> None:
        self.current_xlim = (0.0, 100.0)
        self._refresh_view()



def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-signal EMG viewer with synchronized processing")
    parser.add_argument("--primary", type=str, default=None, help="Optional path to primary .mat file")
    parser.add_argument("--secondary", type=str, default=None, help="Optional path to secondary .mat file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = SignalViewerGUI(primary_path=args.primary, secondary_path=args.secondary)
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
