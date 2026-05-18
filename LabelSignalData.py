"""
LabelSignalData.py
Interactive GUI for manually annotating burst windows in EMG time-series data.

Features:
- Loads .mat files from biosignal_data/{secondary,collected}/raw
- Saves labels to biosignal_data/{secondary,collected}/edited as [filename]_labelled.mat
- Stacked raw signal + spectrogram (0-150 Hz) with locked X-axis
- Minimap with draggable highlighted span controlling zoom window
- Envelope-based ghost burst windows (editable by dragging edges)
- Keyboard: A (start), D (end), Ctrl+Z (undo), Space (accept window), Ctrl+Space (next file)
- Autosave whenever navigating to another file
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import scipy.signal

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
	QApplication,
	QComboBox,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QMessageBox,
	QPushButton,
	QShortcut,
	QVBoxLayout,
	QWidget,
)

from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector

import ControllerConfiguration as Config
import SignalProcessing
from FileRepository import DataRepository, FileSelection


FS = float(getattr(Config, "FS", 1000.0))
CHANNEL_MAP = dict(
	getattr(
		Config,
		"CHANNEL_MAP",
		{
			0: "Channel 0",
			1: "Channel 1",
			2: "Channel 2",
			3: "Channel 3",
			4: "Channel 4",
			5: "Channel 5",
			6: "Channel 6",
			7: "Channel 7",
		},
	)
)

ROBUST_MIN_MAX_KEYS = ("MIN_MAX", "MIN_MAX_ROBUST")


def _contains_none_value(obj: object, depth: int = 0) -> bool:
	"""Recursively checks for None values inside MATLAB-bound payload objects."""
	if obj is None:
		return True
	if depth > 8:
		return False

	if isinstance(obj, dict):
		return any(_contains_none_value(v, depth + 1) for v in obj.values())

	if isinstance(obj, (list, tuple)):
		return any(_contains_none_value(v, depth + 1) for v in obj)

	if isinstance(obj, np.ndarray) and obj.dtype == object:
		return any(_contains_none_value(v, depth + 1) for v in obj.flat)

	return False


def _sanitize_mat_payload(payload: Dict[str, object]) -> Dict[str, object]:
	"""
	Drops fields that contain None (which savemat cannot serialize) and rewrites
	CHANNEL_MAP to a stable canonical structure.
	"""
	sanitized: Dict[str, object] = {}
	for key, value in payload.items():
		if str(key).startswith("__"):
			continue
		if _contains_none_value(value):
			continue
		sanitized[key] = value

	sanitized["CHANNEL_MAP"] = {str(k): str(v) for k, v in CHANNEL_MAP.items()}
	return sanitized

class EMGSignalProcessor:
	"""Signal loading, display preprocessing, spectrogram generation, burst proposals."""

	def __init__(self, fs: float = FS) -> None:
		self.fs = fs

	def load_emg_and_labels(self, file_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
		mat = scipy.io.loadmat(file_path)
		if "EMGDATA" not in mat:
			raise KeyError(f"'EMGDATA' not found in {file_path}")
		
		raw = np.asarray(mat["EMGDATA"], dtype=np.float32)
		if raw.ndim != 2:
			raise ValueError(f"Expected 2D EMGDATA, got shape {raw.shape}")

		labels = self._parse_labels(mat.get("LABELS"), n_samples=raw.shape[1])
		return raw, labels

	def load_raw_emg(self, file_path: str) -> np.ndarray:
		raw, _ = self.load_emg_and_labels(file_path)
		return raw

	def compute_robust_minmax_matrix(
		self,
		file_paths: Sequence[str],
		percentiles: Tuple[float, float] = (1.0, 99.0),
		expected_channels: Optional[int] = None,
	) -> np.ndarray:
		"""
		Computes robust channel-wise min/max from a set of files.
		Returns an array of shape [num_channels, 2] where columns are [min, max].
		"""
		all_data: List[np.ndarray] = []
		num_channels: Optional[int] = expected_channels

		for file_path in file_paths:
			mat = scipy.io.loadmat(file_path)
			if "EMGDATA" not in mat:
				raise KeyError(f"'EMGDATA' not found in {file_path}")

			emg = np.asarray(mat["EMGDATA"], dtype=np.float32)
			if emg.ndim != 2:
				raise ValueError(f"Expected 2D EMGDATA, got shape {emg.shape}")

			if num_channels is None:
				num_channels = emg.shape[0]

			if emg.shape[0] != num_channels:
				raise ValueError(f"Channel count mismatch in {file_path}")

			all_data.append(emg)

		if not all_data:
			raise ValueError("No valid EMGDATA found across participant files.")

		massive_array = np.concatenate(all_data, axis=1)
		robust_mins = np.percentile(massive_array, percentiles[0], axis=1)
		robust_maxs = np.percentile(massive_array, percentiles[1], axis=1)

		return np.column_stack((robust_mins, robust_maxs)).astype(np.float32)

	@staticmethod
	def _parse_labels(label_data: object, n_samples: int) -> List[Tuple[int, int]]:
		if label_data is None:
			return []

		arr = np.asarray(label_data)
		if arr.size == 0:
			return []

		arr = np.squeeze(arr)
		if arr.ndim == 1:
			if arr.shape[0] == 2:
				arr = arr.reshape(1, 2)
			else:
				return []
		elif arr.ndim > 2:
			arr = arr.reshape(-1, arr.shape[-1])

		if arr.ndim != 2:
			return []

		if arr.shape[0] == 2 and arr.shape[1] != 2:
			arr = arr.T

		if arr.shape[1] < 2:
			return []

		arr = arr[:, :2]
		windows: List[Tuple[int, int]] = []
		for row in arr:
			start = int(np.clip(round(float(row[0])), 0, max(0, n_samples - 1)))
			end = int(np.clip(round(float(row[1])), 0, max(0, n_samples - 1)))
			if end > start:
				windows.append((start, end))

		return sorted(windows, key=lambda w: w[0])

	def preprocess_for_display(self, raw: np.ndarray) -> np.ndarray:
		processed = np.zeros_like(raw, dtype=np.float32)
		for ch in range(raw.shape[0]):
			try:
				processed[ch, :] = SignalProcessing.applyStandardSEMGProcessing(raw[ch, :], fs=self.fs)
			except Exception:
				processed[ch, :] = np.abs(raw[ch, :])
		return processed

	def compute_spectrogram(
		self,
		signal_1d: np.ndarray,
		freq_max: float = 150.0,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		freqs, times, power = scipy.signal.spectrogram(
			signal_1d,
			fs=self.fs,
			nperseg=256,
			noverlap=192,
			scaling="density",
			mode="psd",
		)
		mask = freqs <= freq_max
		return times, freqs[mask], power[mask, :]

	def detect_burst_windows(self, processed: np.ndarray) -> List[Tuple[int, int]]:
		"""
		Basic envelope-based burst detection used to generate editable ghost boxes.
		Returns sample-index windows [(start, end), ...].
		"""
		if processed.size == 0:
			return []

		energy = np.sum(np.abs(processed), axis=0)
		kernel = int(max(11, (0.08 * self.fs) // 2 * 2 + 1))  # odd kernel
		smoothed = scipy.signal.medfilt(energy, kernel_size=kernel)

		robust_peak = float(np.percentile(smoothed, 95))
		if robust_peak <= 1e-8:
			return []

		peaks, _ = scipy.signal.find_peaks(
			smoothed,
			height=0.55 * robust_peak,
			prominence=0.08 * robust_peak,
			distance=int(0.8 * self.fs),
		)

		if len(peaks) == 0:
			return []

		_, _, left_ips, right_ips = scipy.signal.peak_widths(smoothed, peaks, rel_height=0.80)
		pad = int(0.10 * self.fs)
		n = processed.shape[1]

		windows: List[Tuple[int, int]] = []
		min_width = int(0.08 * self.fs)
		for left, right in zip(left_ips, right_ips):
			start = max(0, int(left) - pad)
			end = min(n - 1, int(right) + pad)
			if end - start >= min_width:
				windows.append((start, end))

		return self._merge_windows(windows, max_gap=int(0.05 * self.fs))

	@staticmethod
	def _merge_windows(windows: Sequence[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
		if not windows:
			return []

		sorted_windows = sorted((int(s), int(e)) for s, e in windows)
		merged: List[Tuple[int, int]] = [sorted_windows[0]]

		for start, end in sorted_windows[1:]:
			prev_start, prev_end = merged[-1]
			if start <= prev_end + max_gap:
				merged[-1] = (prev_start, max(prev_end, end))
			else:
				merged.append((start, end))

		return merged


def compute_and_inject_robust_minmax(
	participant_id: int,
	data_type: str,
	repo_path: Optional[str] = None,
	percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[np.ndarray, List[str]]:
	"""
	Computes robust min/max for a participant and injects a [num_channels, 2]
	matrix into participant labelled files.

	For collected data, this is MVC-aware:
	- If PxM10.mat exists and is readable, min/max is computed from that MVC trial.
	- Otherwise, it falls back to the legacy participant-wide aggregation.

	Note:
		This never edits raw files in-place. It always writes to *_labelled.mat.

	Returns:
		Tuple[np.ndarray, List[str]]: (min_max_matrix, updated_labelled_file_paths)
	"""
	repo = DataRepository(base_path=repo_path)
	processor = EMGSignalProcessor(fs=FS)

	def _load_valid_payload(path: str) -> Optional[Dict[str, object]]:
		try:
			mat = scipy.io.loadmat(path)
		except Exception:
			return None

		if "EMGDATA" not in mat:
			return None

		emg = np.asarray(mat["EMGDATA"])
		if emg.ndim != 2:
			return None

		return {k: v for k, v in mat.items() if not str(k).startswith("__")}

	def _compute_windowed_minmax_from_entries(
		entries: Sequence[Tuple[FileSelection, str]],
		skip_rest_movement: bool = True,
	) -> Tuple[Optional[np.ndarray], int, int]:
		"""
		Computes robust min/max from labelled window segments only.

		Returns:
			(Optional[np.ndarray], int, int):
				(matrix or None, contributing_file_count, contributing_sample_count)
		"""
		windowed_blocks: List[np.ndarray] = []
		contributing_files = 0

		for selection, source_path in entries:
			if skip_rest_movement and selection.movement == 9:
				continue

			payload = _load_valid_payload(source_path)
			if payload is None:
				continue

			emg = np.asarray(payload.get("EMGDATA"), dtype=np.float32)
			if emg.ndim != 2:
				continue
			if emg.shape[0] != len(CHANNEL_MAP):
				raise ValueError(
					f"Channel count mismatch in {source_path}: "
					f"expected {len(CHANNEL_MAP)}, got {emg.shape[0]}"
				)

			windows = processor._parse_labels(payload.get("LABELS"), n_samples=emg.shape[1])
			if not windows:
				continue

			segments: List[np.ndarray] = []
			for start, end in windows:
				start_i = max(0, int(start))
				end_i = min(emg.shape[1], int(end))
				if end_i > start_i:
					segments.append(emg[:, start_i:end_i])

			if not segments:
				continue

			windowed_blocks.append(np.concatenate(segments, axis=1))
			contributing_files += 1

		if not windowed_blocks:
			return None, 0, 0

		stacked = np.concatenate(windowed_blocks, axis=1)
		robust_mins = np.percentile(stacked, percentiles[0], axis=1)
		robust_maxs = np.percentile(stacked, percentiles[1], axis=1)
		matrix = np.column_stack((robust_mins, robust_maxs)).astype(np.float32)
		return matrix, contributing_files, int(stacked.shape[1])

	source_entries: List[Tuple[FileSelection, str]] = []
	skipped_issues: List[str] = []
	for movement in repo.discover_movements(data_type=data_type, participant=participant_id):
		selection = FileSelection(data_type=data_type, participant=participant_id, movement=movement)
		edited_path = repo.output_file_path(selection, create_dirs=False)
		raw_path = repo.raw_file_path(selection)

		chosen_source: Optional[str] = None
		for candidate in (edited_path, raw_path):
			if not os.path.exists(candidate):
				continue
			if _load_valid_payload(candidate) is not None:
				chosen_source = candidate
				break

		if chosen_source is not None:
			source_entries.append((selection, chosen_source))
		else:
			skipped_issues.append(
				f"Movement {movement}: neither labelled nor raw file was readable ({edited_path} | {raw_path})"
			)

	if not source_entries:
		detail = skipped_issues[0] if skipped_issues else "No source files discovered."
		raise FileNotFoundError(
			f"No readable movement files found for participant {participant_id} ({data_type}).\n{detail}"
		)

	min_max_matrix: Optional[np.ndarray] = None

	# MVC-first behavior for collected data when PxM10.mat exists.
	if data_type == "collected":
		mvc_selection = FileSelection(data_type="collected", participant=participant_id, movement=10)
		mvc_candidates = [
			repo.raw_file_path(mvc_selection),
			repo.output_file_path(mvc_selection, create_dirs=False),
		]

		mvc_source: Optional[str] = None
		for candidate in mvc_candidates:
			if not os.path.exists(candidate):
				continue
			if _load_valid_payload(candidate) is not None:
				mvc_source = candidate
				break

		if mvc_source is not None:
			try:
				windowed_mvc_matrix, mvc_files, mvc_samples = _compute_windowed_minmax_from_entries(
					[(mvc_selection, mvc_source)],
					skip_rest_movement=False,
				)

				if windowed_mvc_matrix is not None:
					min_max_matrix = windowed_mvc_matrix
					print(
						f"[MinMax] Using labelled MVC windows for participant {participant_id}: "
						f"{mvc_files} file(s), {mvc_samples} samples"
					)
				else:
					baseline, max_vals = SignalProcessing.compute_participant_minmax(
						mvc_file_path=mvc_source,
						fs=FS,
						percentiles=percentiles,
						expected_channels=len(CHANNEL_MAP),
					)
					min_max_matrix = np.column_stack((baseline, max_vals)).astype(np.float32)
					print(f"[MinMax] Using MVC trial for participant {participant_id}: {mvc_source}")
			except Exception as exc:
				print(
					f"[MinMax] MVC trial found for participant {participant_id} but failed "
					f"({exc}). Falling back to participant-wide robust min/max."
				)

	if min_max_matrix is None:
		windowed_matrix, windowed_files, windowed_samples = _compute_windowed_minmax_from_entries(
			source_entries,
			skip_rest_movement=True,
		)

		if windowed_matrix is not None:
			min_max_matrix = windowed_matrix
			print(
				f"[MinMax] Using labelled windows for participant {participant_id}: "
				f"{windowed_files} file(s), {windowed_samples} samples (rest movement excluded)."
			)
		else:
			source_paths = [source_path for _, source_path in source_entries]
			min_max_matrix = processor.compute_robust_minmax_matrix(
				file_paths=source_paths,
				percentiles=percentiles,
				expected_channels=len(CHANNEL_MAP),
			)
			print(
				f"[MinMax] No labelled windows found for participant {participant_id}. "
				"Falling back to whole-trial robust min/max."
			)

	updated_files: List[str] = []
	for selection, source_path in source_entries:
		output_path = repo.output_file_path(selection, create_dirs=True)

		# Preserve existing labelled payload when it is readable; otherwise seed from source.
		payload = _load_valid_payload(output_path) if os.path.exists(output_path) else None
		if payload is None:
			payload = _load_valid_payload(source_path)
		if payload is None:
			continue

		payload = _sanitize_mat_payload(payload)

		for key in ROBUST_MIN_MAX_KEYS:
			payload[key] = min_max_matrix

		scipy.io.savemat(output_path, payload)
		updated_files.append(output_path)

	if not updated_files:
		raise ValueError(
			f"Failed to inject robust min/max for participant {participant_id} ({data_type}): no writable labelled files."
		)

	return min_max_matrix, updated_files


class AnnotationState:
	"""Holds editable windows + marker state and supports undo."""

	def __init__(self) -> None:
		self.windows: List[Tuple[int, int]] = []
		self.pending_start: Optional[int] = None
		self.pending_end: Optional[int] = None
		self._history: List[List[Tuple[int, int]]] = [[]]

	def set_windows(self, windows: Sequence[Tuple[int, int]]) -> None:
		self.windows = sorted((int(s), int(e)) for s, e in windows if int(e) > int(s))
		self.pending_start = None
		self.pending_end = None
		self._history = [self.windows.copy()]

	def begin_window_drag(self) -> None:
		self._history.append(self.windows.copy())

	def update_window_edge(self, index: int, edge: str, new_idx: int, n_samples: int) -> None:
		if index < 0 or index >= len(self.windows):
			return

		start, end = self.windows[index]
		new_idx = int(np.clip(new_idx, 0, max(0, n_samples - 1)))
		min_width = 5

		if edge == "left":
			start = min(new_idx, end - min_width)
		elif edge == "right":
			end = max(new_idx, start + min_width)
		else:
			return

		self.windows[index] = (start, end)

	def delete_window(self, index: int) -> bool:
		if index < 0 or index >= len(self.windows):
			return False
		self._history.append(self.windows.copy())
		self.windows.pop(index)
		return True

	def set_start_marker(self, sample_idx: int) -> None:
		self.pending_start = int(sample_idx)
		self.pending_end = None

	def set_end_marker(self, sample_idx: int) -> None:
		if self.pending_start is None:
			return
		self.pending_end = int(sample_idx)

	def accept_pending(self) -> bool:
		if self.pending_start is None or self.pending_end is None:
			return False

		start = min(self.pending_start, self.pending_end)
		end = max(self.pending_start, self.pending_end)
		if end <= start:
			return False

		self.windows.append((start, end))
		self.windows.sort(key=lambda w: w[0])
		self._history.append(self.windows.copy())
		self.pending_start = None
		self.pending_end = None
		return True

	def clear_windows(self) -> None:
		self.windows = []
		self.pending_start = None
		self.pending_end = None
		self._history.append(self.windows.copy())

	def undo(self) -> bool:
		if self.pending_end is not None:
			self.pending_end = None
			return True
		if self.pending_start is not None:
			self.pending_start = None
			return True

		if len(self._history) <= 1:
			return False

		self._history.pop()
		self.windows = self._history[-1].copy()
		return True

	def labels_array(self) -> np.ndarray:
		if not self.windows:
			return np.empty((0, 2), dtype=np.int32)
		return np.asarray(sorted(self.windows, key=lambda w: w[0]), dtype=np.int32)

class SignalCanvas(FigureCanvas):
	"""Matplotlib canvas with stacked plots, minimap span, and edge-drag interactions."""

	edge_drag_started = pyqtSignal(int, str)
	edge_drag_moved = pyqtSignal(int, str, int)
	edge_drag_finished = pyqtSignal()
	window_selected = pyqtSignal(int)

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		self.figure = Figure(figsize=(13, 8), dpi=100)
		super().__init__(self.figure)
		self.setParent(parent)

		grid = self.figure.add_gridspec(3, 1, height_ratios=[3.0, 2.0, 0.8], hspace=0.22)
		self.ax_raw = self.figure.add_subplot(grid[0])
		self.ax_spec = self.figure.add_subplot(grid[1], sharex=self.ax_raw)
		self.ax_minimap = self.figure.add_subplot(grid[2])

		self.raw_data: Optional[np.ndarray] = None
		self.processed_data: Optional[np.ndarray] = None
		self.fs: float = FS
		self.time_axis: Optional[np.ndarray] = None
		self.duration_sec: float = 0.0
		self.current_channel: int = 0
		self.global_envelope_norm: Optional[np.ndarray] = None
		self._processor: Optional[EMGSignalProcessor] = None
		self.windows: List[Tuple[int, int]] = []
		self.selected_window_index: Optional[int] = None
		self.pending_start: Optional[int] = None
		self.pending_end: Optional[int] = None
		self.last_cursor_sec: Optional[float] = None

		self._overlay_artists: List[Artist] = []
		self._span_selector: Optional[SpanSelector] = None
		self._syncing_span = False

		self._drag_window_index: Optional[int] = None
		self._drag_edge: Optional[str] = None

		self.ax_raw.callbacks.connect("xlim_changed", self._on_main_xlim_changed)
		self.mpl_connect("button_press_event", self._on_mouse_press)
		self.mpl_connect("button_release_event", self._on_mouse_release)
		self.mpl_connect("motion_notify_event", self._on_mouse_move)

		self._configure_axes()

	def _configure_axes(self) -> None:
		self._configure_main_axes()

		self.ax_minimap.set_ylabel("Mini", fontsize=8)
		self.ax_minimap.set_xlabel("Time (s)")
		self.ax_minimap.grid(True, linestyle=":", alpha=0.25)

	def _configure_main_axes(self) -> None:
		self.ax_raw.set_ylabel("Amplitude")
		self.ax_raw.grid(True, linestyle="--", alpha=0.25)

		self.ax_spec.set_ylabel("Frequency (Hz)")
		self.ax_spec.grid(True, linestyle=":", alpha=0.25)

	def set_data(
		self,
		raw_data: np.ndarray,
		processed_data: np.ndarray,
		fs: float,
		windows: Sequence[Tuple[int, int]],
		processor: EMGSignalProcessor,
	) -> None:
		self.raw_data = raw_data
		self.processed_data = processed_data
		self._processor = processor
		self.fs = float(fs)
		self.time_axis = np.arange(raw_data.shape[1], dtype=np.float64) / self.fs
		self.duration_sec = float(self.time_axis[-1]) if len(self.time_axis) > 0 else 0.0
		self.current_channel = 0
		self.windows = list(windows)
		self.selected_window_index = None
		self.pending_start = None
		self.pending_end = None
		self.last_cursor_sec = None

		self.ax_raw.clear()
		self.ax_spec.clear()
		self.ax_minimap.clear()
		self._configure_axes()

		# Minimap from multi-channel energy envelope.
		energy = np.sum(np.abs(processed_data), axis=0)
		energy = scipy.signal.medfilt(energy, kernel_size=max(11, int((0.05 * self.fs) // 2 * 2 + 1)))
		if np.max(energy) > 0:
			self.global_envelope_norm = energy / np.max(energy)
		else:
			self.global_envelope_norm = np.zeros_like(energy)
		self.ax_minimap.plot(self.time_axis, self.global_envelope_norm, color="0.35", linewidth=0.8)

		initial_width = min(10.0, max(1.0, self.duration_sec))
		initial_xlim = (0.0, initial_width)

		self._render_main_channel(channel_idx=0, target_xlim=initial_xlim)
		self._setup_minimap_selector()
		self.draw_idle()

	def _render_main_channel(self, channel_idx: int, target_xlim: Optional[Tuple[float, float]] = None) -> None:
		if self.raw_data is None or self.processed_data is None or self.time_axis is None:
			return

		n_channels = self.raw_data.shape[0]
		if n_channels <= 0:
			return

		self.current_channel = int(np.clip(channel_idx, 0, n_channels - 1))

		self.ax_raw.clear()
		self.ax_spec.clear()
		self._configure_main_axes()

		channel_wave = self.raw_data[self.current_channel, :]
		channel_name = CHANNEL_MAP.get(self.current_channel, f"Channel {self.current_channel}")
		self.ax_raw.plot(self.time_axis, channel_wave, color="tab:blue", linewidth=0.9, label="Raw")

		if self.global_envelope_norm is not None and len(self.global_envelope_norm) == len(channel_wave):
			y_min = float(np.min(channel_wave))
			y_max = float(np.max(channel_wave))
			if abs(y_max - y_min) < 1e-9:
				y_max = y_min + 1.0
			envelope_scaled = y_min + self.global_envelope_norm * (y_max - y_min)
			self.ax_raw.plot(self.time_axis, envelope_scaled, color="red", linewidth=1.4, alpha=0.85, label="Global Envelope")

		self.ax_raw.set_title(f"Raw Signal ({channel_name})")
		self.ax_raw.legend(loc="upper right", fontsize=8)

		if self._processor is not None:
			spec_t, spec_f, spec_p = self._processor.compute_spectrogram(
				self.processed_data[self.current_channel, :],
				freq_max=150.0,
			)
			self.ax_spec.pcolormesh(spec_t, spec_f, np.log10(spec_p + 1e-12), shading="auto", cmap="viridis")

		self.ax_spec.set_ylim(0, 150)

		if target_xlim is not None:
			self.ax_raw.set_xlim(target_xlim)

		self._redraw_overlays()

	def set_channel(self, channel_idx: int) -> None:
		if self.raw_data is None:
			return
		xlim = self.ax_raw.get_xlim()
		self._render_main_channel(channel_idx=channel_idx, target_xlim=(float(xlim[0]), float(xlim[1])))

	def get_current_channel(self) -> int:
		return self.current_channel

	def set_windows(self, windows: Sequence[Tuple[int, int]]) -> None:
		prev_selected = self.selected_window_index
		self.windows = sorted((int(s), int(e)) for s, e in windows if int(e) > int(s))
		if prev_selected is not None and 0 <= prev_selected < len(self.windows):
			self.selected_window_index = prev_selected
		else:
			self.selected_window_index = None
		self._redraw_overlays()

	def set_selected_window(self, index: Optional[int]) -> None:
		if index is None or index < 0 or index >= len(self.windows):
			self.selected_window_index = None
		else:
			self.selected_window_index = index
		self._redraw_overlays()

	def get_selected_window_index(self) -> Optional[int]:
		if self.selected_window_index is None:
			return None
		if 0 <= self.selected_window_index < len(self.windows):
			return self.selected_window_index
		return None

	def set_pending_markers(self, start_idx: Optional[int], end_idx: Optional[int]) -> None:
		self.pending_start = None if start_idx is None else int(start_idx)
		self.pending_end = None if end_idx is None else int(end_idx)
		self._redraw_overlays()

	def get_reference_sample_index(self) -> Optional[int]:
		if self.time_axis is None or self.raw_data is None:
			return None
		if self.raw_data.shape[1] == 0:
			return None

		if self.last_cursor_sec is None:
			x0, x1 = self.ax_raw.get_xlim()
			ref_sec = 0.5 * (x0 + x1)
		else:
			ref_sec = self.last_cursor_sec

		return self._sec_to_idx(ref_sec)

	def _sec_to_idx(self, sec: float) -> int:
		if self.raw_data is None:
			return 0
		n = self.raw_data.shape[1]
		return int(np.clip(round(sec * self.fs), 0, max(0, n - 1)))

	def _idx_to_sec(self, idx: int) -> float:
		return float(idx) / self.fs

	def _setup_minimap_selector(self) -> None:
		if self._span_selector is not None:
			self._span_selector.disconnect_events()
			self._span_selector = None

		self._span_selector = SpanSelector(
			self.ax_minimap,
			self._on_minimap_selected,
			"horizontal",
			useblit=True,
			interactive=True,
			drag_from_anywhere=True,
			props={"facecolor": "tab:blue", "alpha": 0.20},
		)

		x0, x1 = self.ax_raw.get_xlim()
		self._syncing_span = True
		try:
			self._span_selector.extents = (x0, x1)
		finally:
			self._syncing_span = False

	def _on_minimap_selected(self, xmin: float, xmax: float) -> None:
		if self._syncing_span or self.time_axis is None:
			return

		x0, x1 = sorted((float(xmin), float(xmax)))
		if x1 - x0 < 0.05:
			return

		x0 = max(0.0, x0)
		x1 = min(self.duration_sec, x1)
		self.ax_raw.set_xlim(x0, x1)
		self.draw_idle()

	def _on_main_xlim_changed(self, _ax) -> None:
		if self._span_selector is None or self._syncing_span:
			return
		x0, x1 = self.ax_raw.get_xlim()
		self._syncing_span = True
		try:
			self._span_selector.extents = (x0, x1)
		except Exception:
			pass
		finally:
			self._syncing_span = False

	def _redraw_overlays(self) -> None:
		for artist in self._overlay_artists:
			try:
				artist.remove()
			except Exception:
				pass
		self._overlay_artists.clear()

		if self.raw_data is None:
			self.draw_idle()
			return

		raw_y0, raw_y1 = self.ax_raw.get_ylim()
		spec_y0, spec_y1 = self.ax_spec.get_ylim()

		for idx, (start, end) in enumerate(self.windows):
			left = self._idx_to_sec(start)
			width = max(0.0, self._idx_to_sec(end) - left)
			is_selected = idx == self.selected_window_index
			face_color = "dodgerblue" if is_selected else "yellow"
			edge_color = "royalblue" if is_selected else "goldenrod"
			alpha = 0.32 if is_selected else 0.22

			rect_raw = Rectangle(
				(left, raw_y0),
				width,
				raw_y1 - raw_y0,
				facecolor=face_color,
				edgecolor=edge_color,
				linewidth=1.2,
				alpha=alpha,
			)
			rect_spec = Rectangle(
				(left, spec_y0),
				width,
				spec_y1 - spec_y0,
				facecolor=face_color,
				edgecolor=edge_color,
				linewidth=1.0,
				alpha=alpha,
			)

			left_line_raw = self.ax_raw.axvline(left, color=edge_color, linestyle="-", linewidth=1.2, alpha=0.9)
			right_line_raw = self.ax_raw.axvline(left + width, color=edge_color, linestyle="-", linewidth=1.2, alpha=0.9)

			self.ax_raw.add_patch(rect_raw)
			self.ax_spec.add_patch(rect_spec)
			self._overlay_artists.extend([rect_raw, rect_spec, left_line_raw, right_line_raw])

		if self.pending_start is not None:
			start_sec = self._idx_to_sec(self.pending_start)
			l1 = self.ax_raw.axvline(start_sec, color="red", linestyle="--", linewidth=1.8)
			l2 = self.ax_spec.axvline(start_sec, color="red", linestyle="--", linewidth=1.4)
			self._overlay_artists.extend([l1, l2])

		if self.pending_end is not None:
			end_sec = self._idx_to_sec(self.pending_end)
			l1 = self.ax_raw.axvline(end_sec, color="green", linestyle="--", linewidth=1.8)
			l2 = self.ax_spec.axvline(end_sec, color="green", linestyle="--", linewidth=1.4)
			self._overlay_artists.extend([l1, l2])

		self.draw_idle()

	def _on_mouse_press(self, event) -> None:
		if self.raw_data is None:
			return
		if event.button != 1:
			return
		if event.inaxes not in (self.ax_raw, self.ax_spec):
			return
		if event.xdata is None:
			return

		x_sec = float(event.xdata)
		x0, x1 = self.ax_raw.get_xlim()
		tolerance = max(0.02, 0.012 * max(0.1, x1 - x0))

		best_dist = float("inf")
		best_idx: Optional[int] = None
		best_edge: Optional[str] = None

		for idx, (start, end) in enumerate(self.windows):
			left = self._idx_to_sec(start)
			right = self._idx_to_sec(end)

			d_left = abs(x_sec - left)
			if d_left < best_dist and d_left <= tolerance:
				best_dist = d_left
				best_idx = idx
				best_edge = "left"

			d_right = abs(x_sec - right)
			if d_right < best_dist and d_right <= tolerance:
				best_dist = d_right
				best_idx = idx
				best_edge = "right"

		if best_idx is not None and best_edge is not None:
			self.set_selected_window(best_idx)
			self.window_selected.emit(best_idx)
			self._drag_window_index = best_idx
			self._drag_edge = best_edge
			self.edge_drag_started.emit(best_idx, best_edge)
			return

		for idx in range(len(self.windows) - 1, -1, -1):
			start, end = self.windows[idx]
			left = self._idx_to_sec(start)
			right = self._idx_to_sec(end)
			if left <= x_sec <= right:
				self.set_selected_window(idx)
				self.window_selected.emit(idx)
				return

		if self.selected_window_index is not None:
			self.set_selected_window(None)
			self.window_selected.emit(-1)

	def _on_mouse_release(self, _event) -> None:
		if self._drag_window_index is not None:
			self.edge_drag_finished.emit()
		self._drag_window_index = None
		self._drag_edge = None

	def _on_mouse_move(self, event) -> None:
		if event.inaxes in (self.ax_raw, self.ax_spec) and event.xdata is not None:
			self.last_cursor_sec = float(event.xdata)

		if self._drag_window_index is None or self._drag_edge is None:
			return
		if event.inaxes not in (self.ax_raw, self.ax_spec):
			return
		if event.xdata is None:
			return

		idx = self._sec_to_idx(float(event.xdata))
		self.edge_drag_moved.emit(self._drag_window_index, self._drag_edge, idx)


class LabelSignalDataApp(QMainWindow):
	"""Main GUI for loading files, annotating windows, and saving labels."""

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Label Signal Data - Burst Annotation")
		self.resize(1460, 920)

		self.repo = DataRepository()
		self.processor = EMGSignalProcessor(fs=FS)
		self.annotations = AnnotationState()

		self.current_selection: Optional[FileSelection] = None
		self.current_raw_data: Optional[np.ndarray] = None
		self.current_processed_data: Optional[np.ndarray] = None

		self._build_ui()
		self._connect_signals()
		self._setup_shortcuts()

		self._populate_participants()
		self._populate_movements()
		self._update_progress_label()

	def _build_ui(self) -> None:
		root = QWidget(self)
		self.setCentralWidget(root)
		main_layout = QVBoxLayout(root)

		top_row = QHBoxLayout()

		self.dataset_combo = QComboBox()
		self.dataset_combo.addItem("Secondary", "secondary")
		self.dataset_combo.addItem("Collected", "collected")

		self.participant_combo = QComboBox()
		self.movement_combo = QComboBox()

		self.load_button = QPushButton("Load File")
		self.save_button = QPushButton("Save")
		self.minmax_button = QPushButton("Compute Min/Max")
		self.prev_channel_button = QPushButton("Prev Channel")
		self.next_channel_button = QPushButton("Next Channel")
		self.channel_label = QLabel("Channel: 0")
		self.delete_button = QPushButton("Delete Selected")
		self.clear_button = QPushButton("Clear Windows")
		self.next_button = QPushButton("Next File")

		self.progress_label = QLabel("Data: Secondary | Participant 1 | Movement 1")
		self.progress_label.setFont(QFont("Consolas", 10))

		top_row.addWidget(QLabel("Dataset:"))
		top_row.addWidget(self.dataset_combo)
		top_row.addWidget(QLabel("Participant:"))
		top_row.addWidget(self.participant_combo)
		top_row.addWidget(QLabel("Movement:"))
		top_row.addWidget(self.movement_combo)
		top_row.addWidget(self.load_button)
		top_row.addWidget(self.save_button)
		top_row.addWidget(self.minmax_button)
		top_row.addWidget(self.prev_channel_button)
		top_row.addWidget(self.next_channel_button)
		top_row.addWidget(self.channel_label)
		top_row.addWidget(self.delete_button)
		top_row.addWidget(self.clear_button)
		top_row.addWidget(self.next_button)
		top_row.addStretch(1)
		top_row.addWidget(self.progress_label)

		self.canvas = SignalCanvas(self)
		self.toolbar = NavigationToolbar2QT(self.canvas, self)

		hint = QLabel("Shortcuts: A=start marker, D=end marker, Ctrl+Z=undo, Space=accept window + next, Ctrl+Space=next file, Ctrl+S=save")
		hint.setFont(QFont("Consolas", 9))

		main_layout.addLayout(top_row)
		main_layout.addWidget(self.toolbar)
		main_layout.addWidget(self.canvas)
		main_layout.addWidget(hint)

	def _connect_signals(self) -> None:
		self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
		self.participant_combo.currentIndexChanged.connect(self._on_participant_changed)

		self.load_button.clicked.connect(self.load_selected_file)
		self.save_button.clicked.connect(self.save_annotations)
		self.minmax_button.clicked.connect(self.compute_participant_minmax)
		self.prev_channel_button.clicked.connect(self.view_previous_channel)
		self.next_channel_button.clicked.connect(self.view_next_channel)
		self.delete_button.clicked.connect(self.delete_selected_window)
		self.clear_button.clicked.connect(self.clear_windows)
		self.next_button.clicked.connect(self.go_to_next_file)

		self.canvas.edge_drag_started.connect(self._on_edge_drag_started)
		self.canvas.edge_drag_moved.connect(self._on_edge_drag_moved)
		self.canvas.edge_drag_finished.connect(self._on_edge_drag_finished)

	def _setup_shortcuts(self) -> None:
		QShortcut(QKeySequence("A"), self, self.place_start_marker)
		QShortcut(QKeySequence("D"), self, self.place_end_marker)
		QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last)
		QShortcut(QKeySequence("Space"), self, self.accept_and_next)
		QShortcut(QKeySequence("Ctrl+Space"), self, self.go_to_next_signal)
		QShortcut(QKeySequence("Ctrl+S"), self, self.save_annotations)
		QShortcut(QKeySequence("["), self, self.view_previous_channel)
		QShortcut(QKeySequence("]"), self, self.view_next_channel)
		QShortcut(QKeySequence("Delete"), self, self.delete_selected_window)
		QShortcut(QKeySequence("Backspace"), self, self.delete_selected_window)

	def save_annotations(self) -> None:
		self._save_annotations(show_popup=True)

	def _current_data_type(self) -> str:
		data = self.dataset_combo.currentData()
		return str(data) if isinstance(data, str) else "secondary"

	def _current_participant(self) -> int:
		text = self.participant_combo.currentText().strip()
		return int(text) if text else 1

	def _current_movement(self) -> int:
		text = self.movement_combo.currentText().strip()
		return int(text) if text else 1

	def _selection_from_ui(self) -> FileSelection:
		return FileSelection(
			data_type=self._current_data_type(),
			participant=self._current_participant(),
			movement=self._current_movement(),
		)

	def _update_channel_label(self) -> None:
		if self.current_raw_data is None:
			self.channel_label.setText("Channel: -")
			return

		channel_idx = self.canvas.get_current_channel()
		channel_name = CHANNEL_MAP.get(channel_idx, f"Channel {channel_idx}")
		self.channel_label.setText(f"Channel: {channel_idx} ({channel_name})")

	def _populate_participants(self) -> None:
		data_type = self._current_data_type()
		participants = self.repo.discover_participants(data_type)

		self.participant_combo.blockSignals(True)
		self.participant_combo.clear()
		for participant in participants:
			self.participant_combo.addItem(str(participant))
		self.participant_combo.blockSignals(False)

		if self.participant_combo.count() == 0:
			self.participant_combo.addItem("1")

	def _populate_movements(self) -> None:
		data_type = self._current_data_type()
		participant = self._current_participant()
		movements = self.repo.discover_movements(data_type, participant)

		self.movement_combo.blockSignals(True)
		self.movement_combo.clear()
		for movement in movements:
			self.movement_combo.addItem(str(movement))
		self.movement_combo.blockSignals(False)

		if self.movement_combo.count() == 0:
			self.movement_combo.addItem("1")

	def _on_dataset_changed(self) -> None:
		self._populate_participants()
		self._populate_movements()
		self._update_progress_label()

	def _on_participant_changed(self) -> None:
		self._populate_movements()
		self._update_progress_label()

	def _update_progress_label(self) -> None:
		data_label = "Secondary" if self._current_data_type() == "secondary" else "Collected"
		self.progress_label.setText(
			f"Data: {data_label} | Participant {self._current_participant()} | Movement {self._current_movement()}"
		)

	def load_selected_file(self) -> None:
		selection = self._selection_from_ui()

		# Autosave when moving from one loaded file to another.
		if self.current_selection is not None and self.current_selection != selection:
			self._save_annotations(show_popup=False)

		try:
			load_path = self.repo.preferred_input_path(selection)
		except ValueError as exc:
			QMessageBox.warning(self, "Invalid Selection", str(exc))
			return

		if not os.path.exists(load_path):
			QMessageBox.warning(self, "File Not Found", f"Could not find:\n{load_path}")
			return

		try:
			raw_data, loaded_labels = self.processor.load_emg_and_labels(load_path)
		except Exception as primary_exc:
			raw_fallback_path = self.repo.raw_file_path(selection)
			can_fallback_to_raw = (
				os.path.normpath(load_path) != os.path.normpath(raw_fallback_path)
				and os.path.exists(raw_fallback_path)
			)
			if not can_fallback_to_raw:
				QMessageBox.critical(
					self,
					"Load Error",
					f"Failed to load file ({type(primary_exc).__name__}):\n{primary_exc}\n\nPath:\n{load_path}",
				)
				return

			try:
				raw_data, loaded_labels = self.processor.load_emg_and_labels(raw_fallback_path)
				load_path = raw_fallback_path
				QMessageBox.warning(
					self,
					"Edited File Unreadable",
					f"Edited file could not be loaded ({type(primary_exc).__name__}):\n{primary_exc}\n\n"
					f"Falling back to raw file:\n{raw_fallback_path}",
				)
			except Exception as fallback_exc:
				QMessageBox.critical(
					self,
					"Load Error",
					f"Failed loading edited file and raw fallback.\n\n"
					f"Edited error ({type(primary_exc).__name__}): {primary_exc}\n"
					f"Raw error ({type(fallback_exc).__name__}): {fallback_exc}",
				)
				return

		try:
			processed_data = self.processor.preprocess_for_display(raw_data)
		except Exception as exc:
			QMessageBox.critical(self, "Process Error", f"Failed to preprocess file:\n{exc}")
			return

		self.current_selection = selection
		self.current_raw_data = raw_data
		self.current_processed_data = processed_data

		if loaded_labels:
			windows_to_use = loaded_labels
		else:
			windows_to_use = self.processor.detect_burst_windows(processed_data)

		self.annotations.set_windows(windows_to_use)
		self.canvas.set_data(
			raw_data=raw_data,
			processed_data=processed_data,
			fs=FS,
			windows=self.annotations.windows,
			processor=self.processor,
		)
		self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)
		self._update_channel_label()
		self._update_progress_label()

	def _save_annotations(self, show_popup: bool) -> bool:
		if self.current_selection is None or self.current_raw_data is None:
			return False

		labels = self.annotations.labels_array()
		out_path = self.repo.output_file_path(self.current_selection, create_dirs=True)

		try:
			self._save_single_labelled_file(out_path, self.current_raw_data, labels, self.current_selection.data_type)
		except Exception as exc:
			if show_popup:
				QMessageBox.critical(self, "Save Error", f"Failed to save labels:\n{exc}")
			return False

		if show_popup:
			QMessageBox.information(self, "Saved", f"Saved labels to:\n{out_path}")
		return True

	@staticmethod
	def _save_single_labelled_file(out_path: str, emg_data: np.ndarray, labels: np.ndarray, data_type: str = "secondary") -> None:
		existing_payload: Dict[str, object] = {}
		if os.path.exists(out_path):
			try:
				existing_payload = {
					k: v for k, v in scipy.io.loadmat(out_path).items() if not str(k).startswith("__")
				}
			except Exception:
				existing_payload = {}

		existing_payload = _sanitize_mat_payload(existing_payload)

		# Center collected data around 0 by subtracting the mean per channel.
		if data_type == "collected":
			emg_data = emg_data.copy()  # Avoid modifying the original
			for ch in range(emg_data.shape[0]):
				channel_mean = np.mean(emg_data[ch, :])
				emg_data[ch, :] -= channel_mean
		
		existing_payload.update(
			{
				"EMGDATA": emg_data,
				"LABELS": labels,
				"FS": np.array([[FS]], dtype=np.float64),
				"CHANNEL_MAP": {str(k): str(v) for k, v in CHANNEL_MAP.items()},
			}
		)
		scipy.io.savemat(
			out_path,
			existing_payload,
		)

	def compute_participant_minmax(self) -> None:
		data_type = self._current_data_type()
		participant = self._current_participant()

		try:
			min_max_matrix, updated_files = compute_and_inject_robust_minmax(
				participant_id=participant,
				data_type=data_type,
				repo_path=self.repo.base_path,
				percentiles=(0.25, 99.75),
			)
		except FileNotFoundError as exc:
			QMessageBox.warning(self, "Min/Max", str(exc))
			return
		except ValueError as exc:
			QMessageBox.warning(self, "Min/Max", f"Could not compute robust min/max:\n{exc}")
			return
		except Exception as exc:
			QMessageBox.critical(
				self,
				"Min/Max Error",
				f"Failed to inject robust min/max ({type(exc).__name__}):\n{exc}",
			)
			return

		preview = "\n".join(
			f"Ch{idx}: min={row[0]:.4f}, max={row[1]:.4f}" for idx, row in enumerate(min_max_matrix)
		)
		QMessageBox.information(
			self,
			"Min/Max Injected",
			f"Injected robust min/max into {len(updated_files)} file(s) for participant {participant}.\n\n"
			f"Stored keys: {', '.join(ROBUST_MIN_MAX_KEYS)}\n\n{preview}",
		)

	def clear_windows(self) -> None:
		reply = QMessageBox.question(
			self,
			"Clear All Windows",
			"Remove all current annotation windows?",
			QMessageBox.Yes | QMessageBox.No,
			QMessageBox.No,
		)
		if reply != QMessageBox.Yes:
			return

		self.annotations.clear_windows()
		self.canvas.set_selected_window(None)
		self.canvas.set_windows(self.annotations.windows)
		self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)

	def delete_selected_window(self) -> None:
		selected_idx = self.canvas.get_selected_window_index()
		if selected_idx is None:
			QMessageBox.information(self, "Delete Window", "Select a window first, then delete it.")
			return

		if not self.annotations.delete_window(selected_idx):
			return

		self.canvas.set_selected_window(None)
		self.canvas.set_windows(self.annotations.windows)
		self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)

	def place_start_marker(self) -> None:
		ref_idx = self.canvas.get_reference_sample_index()
		if ref_idx is None:
			return

		self.annotations.set_start_marker(ref_idx)
		self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)

	def place_end_marker(self) -> None:
		if self.annotations.pending_start is None:
			QMessageBox.information(self, "Marker", "Place start marker first (A).")
			return

		ref_idx = self.canvas.get_reference_sample_index()
		if ref_idx is None:
			return

		self.annotations.set_end_marker(ref_idx)
		self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)

	def accept_and_next(self) -> None:
		if self.annotations.pending_start is not None and self.annotations.pending_end is not None:
			if not self.annotations.accept_pending():
				QMessageBox.warning(self, "Invalid Window", "End marker must be after start marker.")
				return
			self.canvas.set_windows(self.annotations.windows)
			self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)

		# self.go_to_next_file()

	def go_to_next_signal(self) -> None:
		self.go_to_next_file()

	def undo_last(self) -> None:
		changed = self.annotations.undo()
		if not changed:
			return

		self.canvas.set_selected_window(None)
		self.canvas.set_windows(self.annotations.windows)
		self.canvas.set_pending_markers(self.annotations.pending_start, self.annotations.pending_end)

	def view_next_channel(self) -> None:
		if self.current_raw_data is None:
			return
		n_channels = self.current_raw_data.shape[0]
		if n_channels <= 0:
			return
		next_idx = (self.canvas.get_current_channel() + 1) % n_channels
		self.canvas.set_channel(next_idx)
		self._update_channel_label()

	def view_previous_channel(self) -> None:
		if self.current_raw_data is None:
			return
		n_channels = self.current_raw_data.shape[0]
		if n_channels <= 0:
			return
		prev_idx = (self.canvas.get_current_channel() - 1) % n_channels
		self.canvas.set_channel(prev_idx)
		self._update_channel_label()

	def _set_combo_to_value(self, combo: QComboBox, value: int) -> bool:
		value_str = str(value)
		for i in range(combo.count()):
			if combo.itemText(i) == value_str:
				combo.setCurrentIndex(i)
				return True
		return False

	def go_to_next_file(self) -> None:
		if self.current_selection is None:
			self.load_selected_file()
			return

		# Always autosave before moving forward.
		self._save_annotations(show_popup=False)

		data_type = self.current_selection.data_type
		participants = self.repo.discover_participants(data_type)
		participants = sorted(participants)

		if not participants:
			return

		participant = self.current_selection.participant
		movement = self.current_selection.movement

		# Try next movement in current participant first.
		movements = self.repo.discover_movements(data_type, participant)
		movements = sorted(movements)
		if not movements:
			movements = list(range(1, 10))

		if movement in movements:
			m_idx = movements.index(movement)
		else:
			m_idx = -1

		if m_idx + 1 < len(movements):
			next_participant = participant
			next_movement = movements[m_idx + 1]
		else:
			# Advance participant, then take that participant's first movement.
			if participant in participants:
				p_idx = participants.index(participant)
			else:
				p_idx = -1

			if p_idx + 1 >= len(participants):
				QMessageBox.information(self, "End", "Reached the end of available files.")
				return

			next_participant = participants[p_idx + 1]
			next_movements = self.repo.discover_movements(data_type, next_participant)
			next_movement = sorted(next_movements)[0] if next_movements else 1

		# Update combos then load.
		self._set_combo_to_value(self.participant_combo, next_participant)
		self._populate_movements()
		self._set_combo_to_value(self.movement_combo, next_movement)
		self.load_selected_file()

	def _on_edge_drag_started(self, _index: int, _edge: str) -> None:
		self.annotations.begin_window_drag()

	def _on_edge_drag_moved(self, index: int, edge: str, sample_idx: int) -> None:
		if self.current_raw_data is None:
			return
		n_samples = self.current_raw_data.shape[1]
		self.annotations.update_window_edge(index, edge, sample_idx, n_samples)
		self.canvas.set_windows(self.annotations.windows)

	def _on_edge_drag_finished(self) -> None:
		# No extra finalization needed currently.
		pass


def main() -> int:
	app = QApplication(sys.argv)
	window = LabelSignalDataApp()
	window.show()
	return app.exec_()


if __name__ == "__main__":
	sys.exit(main())
