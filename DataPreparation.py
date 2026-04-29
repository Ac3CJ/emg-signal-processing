import os
import re
from collections import defaultdict

import numpy as np
import scipy.interpolate
import scipy.io
import scipy.signal

import ControllerConfiguration as Config
import SignalProcessing
from FileRepository import DataRepository, FileSelection

REPO = DataRepository()

# Pipeline order: file load -> contraction segmentation -> per-burst augmentation
# -> filter/normalize -> concat into continuous arrays for on-the-fly windowing.
# Bursts cover both the labelled contraction and its rising/falling kinematic edges.
BURST_LENGTH_SEC = 4.5
BURST_LENGTH_SAMPLES = int(BURST_LENGTH_SEC * Config.FS)
REST_VALLEY_GAP_SAMPLES = int(0.5 * Config.FS)
SYNTHETIC_RAMP_SEC = 0.5
SYNTHETIC_RAMP_SAMPLES = int(SYNTHETIC_RAMP_SEC * Config.FS)
REST_TARGET_VECTOR = np.array(Config.TARGET_MAPPING[9], dtype=np.float32)
NUM_OUTPUTS = int(Config.NUM_OUTPUTS)


# ====================================================================================
# =============================== ROBUST MIN/MAX UTILS ===============================
# ====================================================================================

def _extract_robust_minmax_matrix(mat_data):
	"""Extracts a [num_channels, 2] robust min/max matrix from a MAT payload."""
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


def _participant_percentile_minmax(movements_data):
	"""Falls back to participant-wide 1/99 percentiles when MIN_MAX_ROBUST is absent."""
	mins = np.zeros(Config.NUM_CHANNELS, dtype=np.float32)
	maxs = np.zeros(Config.NUM_CHANNELS, dtype=np.float32)
	for c in range(Config.NUM_CHANNELS):
		channel_values = []
		for raw in movements_data.values():
			channel_values.extend(np.asarray(raw, dtype=np.float32)[c, :].flatten())
		channel_values = np.asarray(channel_values, dtype=np.float32)
		mins[c] = np.percentile(channel_values, 1.0)
		maxs[c] = np.percentile(channel_values, 99.0)
	return np.column_stack((mins, maxs)).astype(np.float32)


def _file_percentile_minmax(raw_data):
	"""1/99 percentile bounds per channel for a single file's raw signal."""
	raw = np.asarray(raw_data, dtype=np.float32)
	mins = np.percentile(raw, 1.0, axis=1).astype(np.float32)
	maxs = np.percentile(raw, 99.0, axis=1).astype(np.float32)
	return np.column_stack((mins, maxs)).astype(np.float32)


# ====================================================================================
# =============================== SHUFFLING UTILITIES ================================
# ====================================================================================

def _resolve_training_noise_magnitudes():
	"""Returns valid positive training noise magnitudes from configuration."""
	configured = getattr(Config, "TRAINING_NOISE_MAGNITUDES", [0.000005, 0.00001])
	magnitudes = []
	for value in configured:
		try:
			mag = float(value)
		except (TypeError, ValueError):
			continue
		if mag > 0.0:
			magnitudes.append(mag)
	return magnitudes


def _resolve_contraction_block_shuffle_seed():
	"""Returns optional deterministic seed for contraction-block shuffling."""
	raw_seed = getattr(Config, "CONTRACTION_BLOCK_SHUFFLE_SEED", None)
	if raw_seed is None:
		return None
	try:
		return int(raw_seed)
	except (TypeError, ValueError):
		return None


def _resolve_segment_block_shuffle_enabled(explicit_flag, default_condition):
	"""Resolves whether contraction-window block shuffling should run."""
	if explicit_flag is not None:
		return bool(explicit_flag)
	config_enabled = bool(getattr(Config, "CONTRACTION_BLOCK_SHUFFLE", True))
	return config_enabled and bool(default_condition)


def _shuffle_segment_lists_in_place(emg_segments, kin_segments, classes, rng):
	"""Shuffles three parallel lists in lockstep using a numpy RNG."""
	if rng is None or len(emg_segments) <= 1:
		return
	if not (len(emg_segments) == len(kin_segments) == len(classes)):
		raise ValueError("Cannot shuffle segment lists with mismatched lengths.")
	order = rng.permutation(len(emg_segments))
	emg_segments[:] = [emg_segments[idx] for idx in order]
	kin_segments[:] = [kin_segments[idx] for idx in order]
	classes[:] = [classes[idx] for idx in order]


# ====================================================================================
# ============================== KINEMATIC LOADING ===================================
# ====================================================================================

def _signed_reference_target(target_vec):
	"""Returns the first non-zero DOF magnitude with sign — matches GenerateKinematics convention."""
	for v in target_vec:
		if v != 0.0:
			return float(v)
	return 0.0


def _build_ramp_kinematic_at_emg_rate(n_samples, labels, target_vec):
	"""Synthesises a half-cosine ramp-up/hold/ramp-down kinematic profile at EMG rate."""
	profile = np.zeros((n_samples, NUM_OUTPUTS), dtype=np.float32)
	if labels is None or labels.size == 0:
		return profile

	target_arr = np.asarray(target_vec, dtype=np.float32)
	for row in labels:
		start = int(np.clip(int(row[0]), 0, n_samples))
		end = int(np.clip(int(row[1]), 0, n_samples))
		window_len = end - start
		if window_len <= 0:
			continue

		actual_ramp = min(SYNTHETIC_RAMP_SAMPLES, window_len // 2)
		hold_len = window_len - 2 * actual_ramp

		t_up = np.linspace(0.0, np.pi, actual_ramp) if actual_ramp > 0 else np.empty(0)
		ramp_up = 0.5 * (1.0 - np.cos(t_up)) if actual_ramp > 0 else np.empty(0)
		ramp_down = 0.5 * (1.0 + np.cos(t_up)) if actual_ramp > 0 else np.empty(0)
		hold = np.ones(hold_len, dtype=np.float64)
		scalar = np.concatenate([ramp_up, hold, ramp_down]).astype(np.float32)
		profile[start:end, :] = scalar[:, np.newaxis] * target_arr[np.newaxis, :]

	return profile


def _load_collected_kinematic_profile(participant, movement, n_emg_samples, labels, target_vec):
	"""Reads PxMyKinematic.mat KINEMATICS (1000 Hz, n_emg_samples × 4) — falls back to synthetic ramp."""
	out_dir = REPO.edited_root("collected")
	kin_path = os.path.join(out_dir, f"P{participant}M{movement}Kinematic.mat")

	if os.path.exists(kin_path):
		try:
			mat = scipy.io.loadmat(kin_path)
			kin = np.asarray(mat["KINEMATICS"], dtype=np.float32)
			if kin.ndim == 2 and kin.shape[1] == NUM_OUTPUTS and kin.shape[0] >= n_emg_samples:
				return kin[:n_emg_samples, :].astype(np.float32)
			# Length mismatch: pad/truncate to EMG length
			if kin.ndim == 2 and kin.shape[1] == NUM_OUTPUTS:
				if kin.shape[0] < n_emg_samples:
					padded = np.zeros((n_emg_samples, NUM_OUTPUTS), dtype=np.float32)
					padded[:kin.shape[0], :] = kin
					return padded
				return kin[:n_emg_samples, :].astype(np.float32)
		except Exception as exc:
			print(f"  [WARNING] Could not load {os.path.basename(kin_path)}: {exc}")

	return _build_ramp_kinematic_at_emg_rate(n_emg_samples, labels, target_vec)


def _scale_secondary_to_4dof(angolospalla_at_emg_rate, target_vec):
	"""Maps 1-D angolospalla scalar (per EMG sample) into a 4-DOF profile by scaling each non-zero DOF.

	For target = [45, 45, 0, 0] and angolospalla[t] = 22.5 → per-sample = [22.5, 22.5, 0, 0].
	For target = [-30, 0, 0, 0] and angolospalla[t] = -15 → per-sample = [-15, 0, 0, 0].
	"""
	target_arr = np.asarray(target_vec, dtype=np.float32)
	reference = _signed_reference_target(target_arr.tolist())
	if reference == 0.0:
		return np.zeros((angolospalla_at_emg_rate.shape[0], NUM_OUTPUTS), dtype=np.float32)
	scale = (angolospalla_at_emg_rate / reference).astype(np.float32)
	return (scale[:, None] * target_arr[None, :]).astype(np.float32)


def _load_secondary_kinematic_profile(participant, movement, n_emg_samples, labels, target_vec):
	"""Loads MovimentoAngS{m}_edit.mat angolospalla, interp to EMG rate, project to 4-DOF.

	Falls back to MovimentoAngS{m}.mat (raw) if the edited variant is missing,
	then to a synthesised ramp from labels if neither exists.
	"""
	subject_root = os.path.join(REPO.edited_root("secondary"), f"Soggetto{participant}")
	candidates = [
		os.path.join(subject_root, f"MovimentoAngS{movement}_edit.mat"),
		REPO.secondary_kinematics_file_path(participant, movement),
	]

	angolospalla = None
	for path in candidates:
		if not os.path.exists(path):
			continue
		try:
			mat = scipy.io.loadmat(path)
			if "angolospalla" not in mat:
				continue
			angolospalla = np.asarray(mat["angolospalla"], dtype=np.float64).flatten()
			break
		except Exception as exc:
			print(f"  [WARNING] Could not load {os.path.basename(path)}: {exc}")

	if angolospalla is None or angolospalla.size == 0:
		return _build_ramp_kinematic_at_emg_rate(n_emg_samples, labels, target_vec)

	# angolospalla is sampled at fs_kin = len / duration_emg_sec; resample to EMG rate.
	t_kin = np.linspace(0.0, 1.0, angolospalla.size, dtype=np.float64)
	t_emg = np.linspace(0.0, 1.0, n_emg_samples, dtype=np.float64)
	angolospalla_emg = np.interp(t_emg, t_kin, angolospalla).astype(np.float32)

	# Direction-aware clamp to valid target range (mirrors GenerateKinematics.process_secondary_kinematics).
	reference = _signed_reference_target(np.asarray(target_vec, dtype=np.float32).tolist())
	if reference != 0.0:
		lo = min(0.0, reference)
		hi = max(0.0, reference)
		angolospalla_emg = np.clip(angolospalla_emg, lo, hi)

	return _scale_secondary_to_4dof(angolospalla_emg, target_vec)


# ====================================================================================
# ============================== CONTRACTION SEGMENTATION ============================
# ====================================================================================

def _normalise_label_array(labels, n_samples):
	"""Coerces LABELS from a MAT file into an Nx2 int array within [0, n_samples]."""
	if labels is None:
		return np.zeros((0, 2), dtype=np.int64)
	arr = np.asarray(labels)
	if arr.size == 0:
		return np.zeros((0, 2), dtype=np.int64)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 2)
	if arr.ndim != 2 or arr.shape[1] < 2:
		return np.zeros((0, 2), dtype=np.int64)
	clipped = arr[:, :2].astype(np.int64)
	clipped[:, 0] = np.clip(clipped[:, 0], 0, n_samples)
	clipped[:, 1] = np.clip(clipped[:, 1], 0, n_samples)
	valid = clipped[:, 1] > clipped[:, 0]
	return clipped[valid] if valid.any() else np.zeros((0, 2), dtype=np.int64)


def _extract_burst_pairs(raw_emg, kinematic_profile, labels, movement_class):
	"""Slices raw EMG and kinematic profile into burst pairs covering contractions and rest valleys.

	For non-rest movements, each labelled window becomes a fixed-length 4.5 s burst
	centred on the label, plus rest valleys between consecutive labels.
	For movement_class == 9 (rest), the full recording is split into back-to-back
	4.5 s rest bursts.

	Returns three parallel lists: (raw_emg_burst, kinematic_burst, movement_class).
	"""
	emg = np.asarray(raw_emg, dtype=np.float32)
	if emg.ndim != 2:
		return [], [], []

	num_samples = emg.shape[1]
	kin = np.asarray(kinematic_profile, dtype=np.float32)
	if kin.shape != (num_samples, NUM_OUTPUTS):
		# Pad / truncate kinematic to match EMG length for safety.
		fixed = np.zeros((num_samples, NUM_OUTPUTS), dtype=np.float32)
		copy_len = min(num_samples, kin.shape[0]) if kin.ndim == 2 else 0
		if copy_len > 0:
			fixed[:copy_len, :] = kin[:copy_len, :NUM_OUTPUTS]
		kin = fixed

	emg_bursts = []
	kin_bursts = []
	classes = []

	if movement_class == 9:
		for start in range(0, num_samples - BURST_LENGTH_SAMPLES + 1, BURST_LENGTH_SAMPLES):
			end = start + BURST_LENGTH_SAMPLES
			emg_bursts.append(emg[:, start:end])
			kin_bursts.append(kin[start:end, :])
			classes.append(9)
		return emg_bursts, kin_bursts, classes

	clean_labels = _normalise_label_array(labels, num_samples)
	if clean_labels.shape[0] == 0:
		return emg_bursts, kin_bursts, classes

	last_end_idx = 0
	for label_pair in clean_labels:
		start_idx, end_idx = int(label_pair[0]), int(label_pair[1])

		window_center = (start_idx + end_idx) // 2
		burst_start = max(0, window_center - BURST_LENGTH_SAMPLES // 2)
		burst_end = min(num_samples, burst_start + BURST_LENGTH_SAMPLES)
		if burst_end - burst_start < BURST_LENGTH_SAMPLES:
			burst_start = max(0, burst_end - BURST_LENGTH_SAMPLES)
			burst_end = burst_start + BURST_LENGTH_SAMPLES
		if burst_end - burst_start != BURST_LENGTH_SAMPLES:
			continue

		emg_bursts.append(emg[:, burst_start:burst_end])
		kin_bursts.append(kin[burst_start:burst_end, :])
		classes.append(int(movement_class))

		valley_start = last_end_idx + REST_VALLEY_GAP_SAMPLES
		valley_end = start_idx - REST_VALLEY_GAP_SAMPLES
		if valley_end - valley_start >= Config.WINDOW_SIZE:
			emg_bursts.append(emg[:, valley_start:valley_end])
			kin_bursts.append(kin[valley_start:valley_end, :])
			classes.append(9)

		last_end_idx = burst_end

	return emg_bursts, kin_bursts, classes


# ====================================================================================
# =============================== AUGMENTATION =======================================
# ====================================================================================

def apply_magnitude_warping(burst, sigma=0.2, knot=4):
	"""Applies a smooth random multiplier curve to a (channels, samples) burst."""
	burst_arr = np.asarray(burst, dtype=np.float32)
	warp_steps = np.linspace(0, burst_arr.shape[1], knot + 2)
	random_multipliers = np.random.normal(loc=1.0, scale=sigma, size=(burst_arr.shape[0], knot + 2))
	warped = np.zeros_like(burst_arr, dtype=np.float32)
	for c in range(burst_arr.shape[0]):
		interpolator = scipy.interpolate.CubicSpline(warp_steps, random_multipliers[c, :])
		smooth_curve = interpolator(np.arange(burst_arr.shape[1]))
		warped[c, :] = burst_arr[c, :] * smooth_curve.astype(np.float32)
	return warped


def _inject_white_noise_to_burst(burst, magnitudes):
	"""Adds independent gaussian noise per channel across the entire burst."""
	burst_arr = np.asarray(burst, dtype=np.float32)
	if not magnitudes:
		return burst_arr
	mag = float(np.random.choice(magnitudes))
	noise = np.random.normal(loc=0.0, scale=mag, size=burst_arr.shape).astype(np.float32)
	return burst_arr + noise


def _apply_within_class_mixup(emg_bursts, kin_bursts, classes, alpha, ratio):
	"""Generates same-class mixup pairs across full bursts. Blends both EMG and kinematics.

	Skipped when `ratio <= 0`. Pairs are drawn within each movement class so kinematic
	ramp shapes align temporally; cross-class mixing would produce nonsense per-sample
	angle blends.
	"""
	if ratio <= 0.0 or len(emg_bursts) < 2:
		return emg_bursts, kin_bursts, classes

	indices_by_class = defaultdict(list)
	for idx, cls in enumerate(classes):
		indices_by_class[cls].append(idx)

	num_mixups = int(len(emg_bursts) * ratio)
	if num_mixups <= 0:
		return emg_bursts, kin_bursts, classes

	mixable_classes = [cls for cls, idx_list in indices_by_class.items() if len(idx_list) >= 2]
	if len(mixable_classes) == 0:
		print("[Augmentation] Mixup skipped: no class has 2+ same-class bursts.")
		return emg_bursts, kin_bursts, classes

	print(f"\n[Augmentation] Generating {num_mixups} within-class mixup pairs across {len(mixable_classes)} classes...")

	mixed_emg = []
	mixed_kin = []
	mixed_cls = []
	for _ in range(num_mixups):
		cls = mixable_classes[np.random.randint(0, len(mixable_classes))]
		pool = indices_by_class[cls]
		idx1, idx2 = np.random.choice(pool, 2, replace=False)
		# Match shapes: only mix bursts that share the same sample length.
		if emg_bursts[idx1].shape != emg_bursts[idx2].shape:
			continue
		lam = np.random.beta(alpha, alpha)
		emg_mix = (lam * emg_bursts[idx1] + (1.0 - lam) * emg_bursts[idx2]).astype(np.float32)
		kin_mix = (lam * kin_bursts[idx1] + (1.0 - lam) * kin_bursts[idx2]).astype(np.float32)
		mixed_emg.append(emg_mix)
		mixed_kin.append(kin_mix)
		mixed_cls.append(cls)

	return emg_bursts + mixed_emg, kin_bursts + mixed_kin, classes + mixed_cls


# ====================================================================================
# ============================ FILTERING + NORMALISATION =============================
# ====================================================================================

def _filter_and_normalize_burst(raw_burst, channel_minmax):
	"""Notch + bandpass + rectify + per-channel normalize on a single (channels, samples) burst."""
	raw = np.asarray(raw_burst, dtype=np.float32)
	processed = np.zeros_like(raw, dtype=np.float32)
	for c in range(raw.shape[0]):
		notch = SignalProcessing.notchFilter(raw[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
		band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
		rectified = np.abs(band)
		scale = SignalProcessing.get_rectified_scale_from_minmax(
			channel_minmax[c, 0], channel_minmax[c, 1]
		)
		processed[c, :] = np.clip(rectified / scale, 0.0, 1.0)
	return processed


# ====================================================================================
# ============================== CONCATENATION =======================================
# ====================================================================================

def _concat_segments(emg_segments, kin_segments, classes):
	"""Concatenates parallel segment lists into continuous arrays + segment bounds + class list."""
	valid_emg, valid_kin, valid_cls = [], [], []
	for emg, kin, cls in zip(emg_segments, kin_segments, classes):
		if emg is None or kin is None:
			continue
		emg_arr = np.asarray(emg, dtype=np.float32)
		kin_arr = np.asarray(kin, dtype=np.float32)
		if emg_arr.ndim != 2 or emg_arr.shape[1] == 0:
			continue
		if kin_arr.ndim != 2 or kin_arr.shape[0] != emg_arr.shape[1]:
			continue
		valid_emg.append(emg_arr)
		valid_kin.append(kin_arr)
		valid_cls.append(int(cls))

	if len(valid_emg) == 0:
		return None, None, [], []

	bounds = []
	cursor = 0
	for emg in valid_emg:
		bounds.append((cursor, cursor + emg.shape[1]))
		cursor += emg.shape[1]

	continuous_X = np.concatenate(valid_emg, axis=1).astype(np.float32)
	continuous_y = np.concatenate(valid_kin, axis=0).astype(np.float32)
	return continuous_X, continuous_y, bounds, valid_cls


# ====================================================================================
# =================================== PIPELINES ======================================
# ====================================================================================

def load_and_prepare_dataset(
	base_path='./biosignal_data/secondary/raw',
	include_subjects=None,
	labelled_base_path=None,
	include_noise_aug=True,
	augment=True,
	shuffle_segment_blocks=None,
):
	"""Secondary-data pipeline: load → segment → augment → filter → concat.

	Pipeline order:
	  1. File loading (raw EMG + per-participant min/max + LABELS + kinematic profile).
	  2. Contraction segmentation: 4.5 s bursts spanning the rising/falling kinematic
	     edges, plus rest valleys between contractions.
	  3. Augmentation across the whole burst: within-class mixup → magnitude warping
	     → white-noise injection. Kinematic ramps are blended only for same-class mixup.
	  4. Filtering: notch → bandpass → rectify → per-channel normalize on each burst.
	  5. Concat into continuous arrays with explicit segment bounds for the on-the-fly
	     ContractionBlockSampler at training time.

	Returns:
	    (continuous_X, continuous_y, segment_bounds, segment_classes)
	      continuous_X       (num_channels, total_samples) float32
	      continuous_y       (total_samples, num_outputs) float32 — per-sample kinematic angles
	      segment_bounds     [(start_sample, end_sample), ...] for each contraction/rest segment
	      segment_classes    [movement_class, ...] aligned with segment_bounds
	"""
	if include_subjects is None:
		include_subjects = list(range(1, 9))

	if not (base_path.endswith('/raw') or base_path.endswith('/edited')):
		labelled_base_path = base_path + "/secondary/edited"
	else:
		labelled_base_path = base_path.replace('/raw', '/edited')

	repository = DataRepository.from_standard_path(labelled_base_path)
	if repository is None:
		repository = DataRepository.from_standard_path(base_path)

	should_shuffle_segment_blocks = _resolve_segment_block_shuffle_enabled(
		shuffle_segment_blocks, default_condition=augment
	)
	block_shuffle_rng = (
		np.random.default_rng(_resolve_contraction_block_shuffle_seed())
		if should_shuffle_segment_blocks
		else None
	)

	print("Beginning Label-Based data extraction...")
	print(f"Include subjects: {include_subjects}")
	print(f"Loading raw files from: {base_path}")
	print(f"Loading labelled windows from: {labelled_base_path}")
	if augment and include_noise_aug:
		print(f"Per-burst noise augmentation enabled: {_resolve_training_noise_magnitudes()}")
	else:
		print("Per-burst noise augmentation disabled")
	print(f"Within-class mixup ratio: {getattr(Config, 'MIXUP_RATIO', 0)}, alpha: {getattr(Config, 'MIXUP_ALPHA', 0)}")
	print(f"Contraction-window block shuffling: {'enabled' if should_shuffle_segment_blocks else 'disabled'}")
	print("NOTE: Filtering applied AFTER segmentation and augmentation (per-burst).\n")

	all_emg_bursts = []
	all_kin_bursts = []
	all_classes = []

	for p in include_subjects:
		print(f"Processing Subject {p}...")

		movements_data = {}
		movements_labels = {}
		participant_minmax = None

		for m in range(1, 10):
			if hasattr(Config, 'SECONDARY_BLACKLIST') and (p, m) in Config.SECONDARY_BLACKLIST:
				continue

			if repository is not None:
				label_path = repository.output_file_path(
					FileSelection(data_type="secondary", participant=p, movement=m),
					create_dirs=False,
				)
			else:
				label_path = os.path.join(labelled_base_path, f'Soggetto{p}', f'Movimento{m}_labelled.mat')
			if not os.path.exists(label_path):
				raise FileNotFoundError(f"Expected file not found: {label_path}")

			try:
				label_mat = scipy.io.loadmat(label_path)
				movements_data[m] = np.asarray(label_mat['EMGDATA'], dtype=np.float32)
				movements_labels[m] = label_mat.get('LABELS')
				if participant_minmax is None:
					participant_minmax = _extract_robust_minmax_matrix(label_mat)
			except Exception as e:
				print(f"  [WARNING] Could not load labels for Movimento{m}: {e}")

		if not movements_data:
			continue

		if participant_minmax is None:
			print("  [WARNING] MIN_MAX_ROBUST not found. Falling back to participant percentiles.")
			participant_minmax = _participant_percentile_minmax(movements_data)
		print("  Robust normalization ranges per channel loaded.")

		for m, raw_data in movements_data.items():
			target_vector = np.array(Config.TARGET_MAPPING[m], dtype=np.float32)
			n_emg_samples = raw_data.shape[1]
			labels = movements_labels.get(m)

			kinematic_profile = _load_secondary_kinematic_profile(
				participant=p,
				movement=m,
				n_emg_samples=n_emg_samples,
				labels=_normalise_label_array(labels, n_emg_samples),
				target_vec=target_vector,
			)

			emg_bursts, kin_bursts, classes = _extract_burst_pairs(
				raw_data, kinematic_profile, labels, movement_class=m
			)
			if not emg_bursts:
				continue

			# Filter+normalize is applied per burst AFTER augmentation; we tag bursts
			# with their participant min/max so each survives the per-burst processing pass.
			for emg_b, kin_b, cls in zip(emg_bursts, kin_bursts, classes):
				all_emg_bursts.append(emg_b)
				all_kin_bursts.append(kin_b)
				all_classes.append((cls, participant_minmax))

		print(f"  Subject {p} processed. Total bursts so far: {len(all_emg_bursts)}.")

	if len(all_emg_bursts) == 0:
		print("No valid bursts extracted.")
		return None, None, [], []

	# ============================== AUGMENTATION ==============================
	classes_only = [cls for cls, _ in all_classes]
	if augment and getattr(Config, 'MIXUP_RATIO', 0) > 0:
		# Mixup samples inherit the participant min/max of their first source burst.
		minmax_lookup = [minmax for _, minmax in all_classes]
		all_emg_bursts, all_kin_bursts, classes_only = _apply_within_class_mixup(
			all_emg_bursts, all_kin_bursts, classes_only,
			alpha=Config.MIXUP_ALPHA, ratio=Config.MIXUP_RATIO,
		)
		# Pad minmax_lookup with the first source burst's min/max for each new mixup item.
		while len(minmax_lookup) < len(classes_only):
			minmax_lookup.append(minmax_lookup[0])
		all_classes = list(zip(classes_only, minmax_lookup))

	if augment:
		minmax_lookup = [minmax for _, minmax in all_classes]
		expanded_emg = []
		expanded_kin = []
		expanded_classes = []
		expanded_minmax = []
		mw_sigmas = (0.25, 0.40)
		noise_magnitudes = _resolve_training_noise_magnitudes() if include_noise_aug else []

		for emg_b, kin_b, cls, mm in zip(all_emg_bursts, all_kin_bursts, classes_only, minmax_lookup):
			expanded_emg.append(np.asarray(emg_b, dtype=np.float32))
			expanded_kin.append(np.asarray(kin_b, dtype=np.float32))
			expanded_classes.append(cls)
			expanded_minmax.append(mm)

			for sigma in mw_sigmas:
				expanded_emg.append(apply_magnitude_warping(emg_b, sigma=sigma))
				expanded_kin.append(np.asarray(kin_b, dtype=np.float32))
				expanded_classes.append(cls)
				expanded_minmax.append(mm)

			for mag in noise_magnitudes:
				expanded_emg.append(_inject_white_noise_to_burst(emg_b, [mag]))
				expanded_kin.append(np.asarray(kin_b, dtype=np.float32))
				expanded_classes.append(cls)
				expanded_minmax.append(mm)

		all_emg_bursts = expanded_emg
		all_kin_bursts = expanded_kin
		classes_only = expanded_classes
		minmax_lookup = expanded_minmax
	else:
		minmax_lookup = [minmax for _, minmax in all_classes]

	# ============================== FILTERING ==============================
	print(f"\nFiltering and normalising {len(all_emg_bursts)} bursts (per-burst notch+bandpass+rectify+normalize)...")
	filtered_emg = [
		_filter_and_normalize_burst(emg_b, mm)
		for emg_b, mm in zip(all_emg_bursts, minmax_lookup)
	]

	if should_shuffle_segment_blocks:
		_shuffle_segment_lists_in_place(filtered_emg, all_kin_bursts, classes_only, block_shuffle_rng)

	continuous_X, continuous_y, segment_bounds, segment_classes = _concat_segments(
		filtered_emg, all_kin_bursts, classes_only
	)
	if continuous_X is None:
		print("No valid continuous data extracted.")
		return None, None, [], []

	print(f"Continuous dataset generated! Total samples: {continuous_X.shape[1]}, segments: {len(segment_bounds)}")
	print(f"Data range after normalization: [{continuous_X.min():.4f}, {continuous_X.max():.4f}] (expected: [0.0, 1.0])\n")
	return continuous_X, continuous_y, segment_bounds, segment_classes


def load_collected_data(
	folder_path,
	labelled_folder_path=None,
	augment=True,
	include_noise_aug=True,
	include_participants=None,
	shuffle_segment_blocks=None,
):
	"""Collected-data pipeline: load → segment → augment → filter → concat.

	Returns:
	    (continuous_X, continuous_y, segment_bounds, segment_classes)
	"""
	if not os.path.exists(folder_path):
		print(f"[WARNING] Folder does not exist: {folder_path}")
		return None, None, [], []

	repository = DataRepository.from_standard_path(folder_path)
	use_repository_paths = repository is not None and os.path.normpath(folder_path) == os.path.normpath(
		repository.raw_root("collected")
	)
	should_shuffle_segment_blocks = _resolve_segment_block_shuffle_enabled(
		shuffle_segment_blocks, default_condition=augment
	)
	block_shuffle_rng = (
		np.random.default_rng(_resolve_contraction_block_shuffle_seed())
		if should_shuffle_segment_blocks
		else None
	)

	if labelled_folder_path is None:
		labelled_folder_path = folder_path.replace('/raw', '/edited')
		if not os.path.exists(labelled_folder_path):
			labelled_folder_path = folder_path.replace('collected_data', 'biosignal_data/collected/edited')
	if use_repository_paths:
		labelled_folder_path = repository.edited_root("collected")

	if use_repository_paths:
		participants = repository.discover_participants("collected")
		if include_participants is not None:
			participant_filter = set()
			for participant in include_participants:
				try:
					participant_filter.add(int(participant))
				except (TypeError, ValueError):
					continue
			participants = [p for p in participants if p in participant_filter]
			print(f"[Collected Data] Participant filter enabled: {sorted(participant_filter)}")
			if len(participants) == 0:
				print(f"[WARNING] No files matched selected participants in {folder_path}")
				return None, None, [], []
		selection_entries = repository.iter_file_selections("collected", participants)
	else:
		mat_files = [
			f for f in os.listdir(folder_path)
			if f.endswith('.mat') and not f.endswith('_labelled.mat')
		]
		if len(mat_files) == 0:
			print(f"[WARNING] No .mat files found in {folder_path}")
			return None, None, [], []

		participant_filter = None
		if include_participants is not None:
			participant_filter = set()
			for participant in include_participants:
				try:
					participant_filter.add(int(participant))
				except (TypeError, ValueError):
					continue
			filtered_files = []
			skipped_non_matching = 0
			for file_name in mat_files:
				match = re.search(r'P(\d+)M(\d+)', os.path.splitext(file_name)[0], flags=re.IGNORECASE)
				if not match:
					skipped_non_matching += 1
					continue
				participant_id = int(match.group(1))
				if participant_id in participant_filter:
					filtered_files.append(file_name)
			mat_files = filtered_files
			print(f"[Collected Data] Participant filter enabled: {sorted(participant_filter)}")
			if skipped_non_matching > 0:
				print(f"[Collected Data] Skipped {skipped_non_matching} files that did not match P#M# naming.")
			if len(mat_files) == 0:
				print(f"[WARNING] No files matched selected participants in {folder_path}")
				return None, None, [], []

		selection_entries = [(None, mat_file) for mat_file in sorted(mat_files)]

	expected_file_count = len(selection_entries)
	print(f"[Collected Data] Found {expected_file_count} .mat files in {folder_path}")
	print(f"[Collected Data] Using labelled windows from: {labelled_folder_path}")
	if augment and include_noise_aug:
		print(f"[Collected Data] Per-burst noise augmentation enabled: {_resolve_training_noise_magnitudes()}")
	else:
		print("[Collected Data] Per-burst noise augmentation disabled")
	print(f"[Collected Data] Contraction-window block shuffling: {'enabled' if should_shuffle_segment_blocks else 'disabled'}")
	print("[Collected Data] Filtering applied AFTER segmentation and augmentation (per-burst).")

	all_emg_bursts = []
	all_kin_bursts = []
	all_classes = []
	per_burst_minmax = []

	collected_blacklist = set(getattr(Config, 'COLLECTED_BLACKLIST', []) or [])

	for entry in selection_entries:
		if use_repository_paths:
			selection = entry if isinstance(entry, FileSelection) else entry[1]
			participant = int(selection.participant)
			movement = int(selection.movement)
			file_path = repository.raw_file_path(selection)
			mat_file = f'P{participant}M{movement}.mat'
			labelled_file = repository.output_file_path(selection, create_dirs=False)
			if (participant, movement) in collected_blacklist:
				print(f"  [SKIP] {mat_file}: blacklisted in COLLECTED_BLACKLIST")
				continue
		else:
			mat_file = entry[1] if isinstance(entry, tuple) else entry
			file_path = os.path.join(folder_path, mat_file)
			labelled_file = os.path.join(labelled_folder_path, mat_file.replace('.mat', '_labelled.mat'))
			match_pre = re.search(r'P(\d+)M(\d+)', os.path.splitext(mat_file)[0], flags=re.IGNORECASE)
			if match_pre and (int(match_pre.group(1)), int(match_pre.group(2))) in collected_blacklist:
				print(f"  [SKIP] {mat_file}: blacklisted in COLLECTED_BLACKLIST")
				continue

		try:
			if use_repository_paths and not repository.is_readable_mat(file_path):
				print(f"  [SKIP] {mat_file}: No EMGDATA found or file unreadable")
				continue

			mat = scipy.io.loadmat(file_path)
			if 'EMGDATA' not in mat:
				print(f"  [SKIP] {mat_file}: No EMGDATA found")
				continue
			raw_data = np.asarray(mat['EMGDATA'], dtype=np.float32)
			n_emg_samples = raw_data.shape[1]

			file_minmax = None
			labels = None
			if os.path.exists(labelled_file):
				try:
					label_mat = scipy.io.loadmat(labelled_file)
					file_minmax = _extract_robust_minmax_matrix(label_mat)
					labels = label_mat.get('LABELS')
				except Exception as exc:
					print(f"  [WARNING] Could not load labelled file for robust min/max: {exc}")

			if file_minmax is None:
				file_minmax = _file_percentile_minmax(raw_data)

			filename_base = os.path.splitext(mat_file)[0]
			match = re.search(r'P(\d+)M(\d+)', filename_base)
			if not match:
				print(f"  [SKIP] {mat_file}: Could not parse movement from filename. Use format P#M#.mat")
				continue
			participant_id = int(match.group(1))
			movement_id = int(match.group(2))
			if movement_id not in Config.TARGET_MAPPING:
				print(f"  [SKIP] {mat_file}: Unknown movement ID {movement_id}")
				continue

			target_vector = np.array(Config.TARGET_MAPPING[movement_id], dtype=np.float32)

			kinematic_profile = _load_collected_kinematic_profile(
				participant=participant_id,
				movement=movement_id,
				n_emg_samples=n_emg_samples,
				labels=_normalise_label_array(labels, n_emg_samples),
				target_vec=target_vector,
			)

			emg_bursts, kin_bursts, classes = _extract_burst_pairs(
				raw_data, kinematic_profile, labels, movement_class=movement_id
			)

			if len(emg_bursts) == 0:
				print(f"  [SKIP] {mat_file}: No segments extracted")
				continue

			for emg_b, kin_b, cls in zip(emg_bursts, kin_bursts, classes):
				all_emg_bursts.append(emg_b)
				all_kin_bursts.append(kin_b)
				all_classes.append(cls)
				per_burst_minmax.append(file_minmax)

			print(f"  [LOADED] {mat_file} → {len(emg_bursts)} segments")

		except Exception as e:
			print(f"  [ERROR] {mat_file}: {e}")
			continue

	if len(all_emg_bursts) == 0:
		print(f"[Collected Data] No valid bursts extracted from {folder_path}")
		return None, None, [], []

	# ============================== AUGMENTATION ==============================
	if augment and getattr(Config, 'MIXUP_RATIO', 0) > 0:
		original_count = len(all_emg_bursts)
		all_emg_bursts, all_kin_bursts, all_classes = _apply_within_class_mixup(
			all_emg_bursts, all_kin_bursts, all_classes,
			alpha=Config.MIXUP_ALPHA, ratio=Config.MIXUP_RATIO,
		)
		while len(per_burst_minmax) < len(all_emg_bursts):
			per_burst_minmax.append(per_burst_minmax[0])
		print(f"  [Augmentation] Mixup: {original_count} → {len(all_emg_bursts)} bursts")

	if augment:
		mw_sigmas = (0.25, 0.40)
		noise_magnitudes = _resolve_training_noise_magnitudes() if include_noise_aug else []
		expanded_emg = []
		expanded_kin = []
		expanded_classes = []
		expanded_minmax = []

		for emg_b, kin_b, cls, mm in zip(all_emg_bursts, all_kin_bursts, all_classes, per_burst_minmax):
			expanded_emg.append(np.asarray(emg_b, dtype=np.float32))
			expanded_kin.append(np.asarray(kin_b, dtype=np.float32))
			expanded_classes.append(cls)
			expanded_minmax.append(mm)

			for sigma in mw_sigmas:
				expanded_emg.append(apply_magnitude_warping(emg_b, sigma=sigma))
				expanded_kin.append(np.asarray(kin_b, dtype=np.float32))
				expanded_classes.append(cls)
				expanded_minmax.append(mm)

			for mag in noise_magnitudes:
				expanded_emg.append(_inject_white_noise_to_burst(emg_b, [mag]))
				expanded_kin.append(np.asarray(kin_b, dtype=np.float32))
				expanded_classes.append(cls)
				expanded_minmax.append(mm)

		all_emg_bursts = expanded_emg
		all_kin_bursts = expanded_kin
		all_classes = expanded_classes
		per_burst_minmax = expanded_minmax

	# ============================== FILTERING ==============================
	print(f"\n[Collected Data] Filtering and normalising {len(all_emg_bursts)} bursts...")
	filtered_emg = [
		_filter_and_normalize_burst(emg_b, mm)
		for emg_b, mm in zip(all_emg_bursts, per_burst_minmax)
	]

	if should_shuffle_segment_blocks:
		_shuffle_segment_lists_in_place(filtered_emg, all_kin_bursts, all_classes, block_shuffle_rng)

	continuous_X, continuous_y, segment_bounds, segment_classes = _concat_segments(
		filtered_emg, all_kin_bursts, all_classes
	)
	if continuous_X is None:
		print(f"[Collected Data] No valid continuous data from {folder_path}")
		return None, None, [], []

	print(f"[Collected Data] Total continuous samples: {continuous_X.shape[1]}, segments: {len(segment_bounds)}")
	return continuous_X, continuous_y, segment_bounds, segment_classes


if __name__ == "__main__":
	X, y, bounds, classes = load_and_prepare_dataset()
	print(f"\nLoaded: X={None if X is None else X.shape}, y={None if y is None else y.shape}, "
	      f"segments={len(bounds)}, classes={set(classes) if classes else 'none'}")
