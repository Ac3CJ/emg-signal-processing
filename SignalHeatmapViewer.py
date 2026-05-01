"""
SignalHeatmapViewer.py
Standalone viewer for sEMG recordings as 2D heatmap images.
Channels = rows, time = columns, amplitude = colour.
No dependency on SignalViewerGUI.
"""

import sys
import argparse
import numpy as np
import scipy.io as spio
import scipy.signal
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QButtonGroup, QRadioButton,
    QSizePolicy,
)
from PyQt5.QtCore import Qt

import ControllerConfiguration as Config
import SignalProcessing


# ─── Signal processing helpers ──────────────────────────────────────────────

def _compute_minmax(raw_data, mat_data):
    """Returns (C, 2) min/max matrix from MIN_MAX_ROBUST field or 1/99 percentiles."""
    for key in ("MIN_MAX_ROBUST", "MIN_MAX"):
        if key not in mat_data:
            continue
        m = np.asarray(mat_data[key], dtype=np.float32)
        if m.ndim != 2:
            continue
        if m.shape == (raw_data.shape[0], 2):
            return m
        if m.shape == (2, raw_data.shape[0]):
            return m.T
    # fallback: per-channel 1/99 percentile from this file
    mins = np.percentile(raw_data, 1.0, axis=1).astype(np.float32)
    maxs = np.percentile(raw_data, 99.0, axis=1).astype(np.float32)
    return np.column_stack((mins, maxs))


def _apply_filtered_mode(raw_data, minmax):
    """Notch → bandpass → rectify → [0,1] normalise per channel. Returns (C, T) float32."""
    num_channels = raw_data.shape[0]
    result = np.zeros_like(raw_data, dtype=np.float32)
    for c in range(num_channels):
        notched = SignalProcessing.notchFilter(raw_data[c], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        banded = SignalProcessing.bandpassFilter(
            notched, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH
        )
        rectified = np.abs(banded)
        scale = SignalProcessing.get_rectified_scale_from_minmax(minmax[c, 0], minmax[c, 1])
        result[c] = np.clip(rectified / scale, 0.0, 1.0)
    return result


def _apply_raw_mode(raw_data, minmax):
    """Symmetric [-1, 1] normalisation only — no filtering, no rectification."""
    num_channels = raw_data.shape[0]
    result = np.zeros_like(raw_data, dtype=np.float32)
    for c in range(num_channels):
        scale = SignalProcessing.get_unrectified_scale_from_minmax(minmax[c, 0], minmax[c, 1])
        result[c] = np.clip(raw_data[c] / scale, -1.0, 1.0)
    return result


# ─── Matplotlib canvas ───────────────────────────────────────────────────────

class HeatmapCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.cbar = None
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot(self, heatmap_data, mode, fs):
        """
        heatmap_data: (C, T) array
        mode: 'filtered' or 'raw'
        fs: sampling frequency for x-axis ticks
        """
        self.ax.clear()
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None

        num_channels, num_samples = heatmap_data.shape
        duration_s = num_samples / fs

        if mode == 'filtered':
            cmap = 'viridis'
            vmin, vmax = 0.0, 1.0
        else:
            cmap = 'RdBu_r'
            vmin, vmax = -1.0, 1.0

        im = self.ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, duration_s, num_channels - 0.5, -0.5],
            interpolation='nearest',
        )

        self.cbar = self.fig.colorbar(im, ax=self.ax)
        amplitude_label = 'Amplitude [0, 1]' if mode == 'filtered' else 'Amplitude [-1, 1]'
        self.cbar.set_label(amplitude_label)

        channel_labels = [Config.CHANNEL_MAP.get(i, f'Ch {i}') for i in range(num_channels)]
        self.ax.set_yticks(range(num_channels))
        self.ax.set_yticklabels(channel_labels, fontsize=8)
        self.ax.set_xlabel('Time (s)')
        mode_label = 'Filtered [0,1]' if mode == 'filtered' else 'Raw [-1,1]'
        self.ax.set_title(f'sEMG Heatmap — {mode_label}')

        self.fig.canvas.draw_idle()


# ─── Main window ─────────────────────────────────────────────────────────────

class SignalHeatmapViewer(QMainWindow):
    def __init__(self, preload_path=None):
        super().__init__()
        self.setWindowTitle('sEMG Heatmap Viewer')
        self.resize(1200, 500)

        self._raw_data = None
        self._minmax = None
        self._current_file = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── toolbar ──
        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)

        self.open_btn = QPushButton('Open .mat file…')
        self.open_btn.clicked.connect(self._open_file)
        toolbar.addWidget(self.open_btn)

        self.file_label = QLabel('No file loaded')
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(self.file_label)

        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel('Mode:'))

        self._mode_group = QButtonGroup(self)
        self.filtered_radio = QRadioButton('Filtered')
        self.raw_radio = QRadioButton('Raw')
        self.filtered_radio.setChecked(True)
        self._mode_group.addButton(self.filtered_radio, 0)
        self._mode_group.addButton(self.raw_radio, 1)
        toolbar.addWidget(self.filtered_radio)
        toolbar.addWidget(self.raw_radio)
        self._mode_group.buttonClicked.connect(self._on_mode_change)

        # ── canvas ──
        self.canvas = HeatmapCanvas(self)
        layout.addWidget(self.canvas)

        if preload_path:
            self._load_file(preload_path)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open sEMG .mat file', '', 'MAT files (*.mat)'
        )
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            mat = spio.loadmat(path)
        except Exception as e:
            self.file_label.setText(f'Error loading file: {e}')
            return

        if 'EMGDATA' not in mat:
            self.file_label.setText(f'No EMGDATA field in {Path(path).name}')
            return

        raw = np.asarray(mat['EMGDATA'], dtype=np.float32)
        if raw.ndim != 2:
            self.file_label.setText(f'EMGDATA must be 2D, got shape {raw.shape}')
            return

        self._raw_data = raw
        self._minmax = _compute_minmax(raw, mat)
        self._current_file = path
        self.file_label.setText(Path(path).name)
        self._render()

    def _on_mode_change(self, _btn):
        if self._raw_data is not None:
            self._render()

    def _render(self):
        if self._raw_data is None:
            return
        mode = 'filtered' if self.filtered_radio.isChecked() else 'raw'
        if mode == 'filtered':
            data = _apply_filtered_mode(self._raw_data, self._minmax)
        else:
            data = _apply_raw_mode(self._raw_data, self._minmax)
        self.canvas.plot(data, mode, Config.FS)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='sEMG Heatmap Viewer')
    parser.add_argument('--file', type=str, default=None, help='Path to .mat file to preload')
    args, qt_args = parser.parse_known_args()

    app = QApplication(sys.argv[:1] + qt_args)
    window = SignalHeatmapViewer(preload_path=args.file)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
