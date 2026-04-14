import os
import re
import scipy.io
import numpy as np
import scipy.signal
import scipy.interpolate

import SignalProcessing 
import ControllerConfiguration as Config


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

# ====================================================================================
# =========================== DATA AUGMENTATION TECHNIQUES =========================== 
# ====================================================================================

def apply_magnitude_warping(window, sigma=0.2, knot=4):
    """Applies a smooth, random multiplier curve to the 500ms window."""
    warp_steps = np.linspace(0, window.shape[1], knot+2)
    random_multipliers = np.random.normal(loc=1.0, scale=sigma, size=(window.shape[0], knot+2))
    warped_window = np.zeros_like(window, dtype=np.float32)
    
    for i in range(window.shape[0]):
        interpolator = scipy.interpolate.CubicSpline(warp_steps, random_multipliers[i, :])
        smooth_curve = interpolator(np.arange(window.shape[1]))
        warped_window[i, :] = window[i, :] * smooth_curve
        
    return warped_window

def apply_mixup(data_arrays, targets, alpha=0.2, mixup_ratio=0.5):
    """
    Applies Mixup augmentation. 
    Works on both full Phase-Aligned bursts AND 500ms Rest windows.
    """
    if mixup_ratio <= 0.0 or len(data_arrays) == 0:
        return data_arrays, targets

    num_mixups = int(len(data_arrays) * mixup_ratio)
    print(f"\n[Augmentation] Generating {num_mixups} Mixup arrays...")
    
    mixed_arrays = []
    mixed_targets = []
    
    for _ in range(num_mixups):
        idx1, idx2 = np.random.choice(len(data_arrays), 2, replace=False)
        lam = np.random.beta(alpha, alpha)
        
        new_array = (lam * data_arrays[idx1]) + ((1 - lam) * data_arrays[idx2])
        new_target = (lam * targets[idx1]) + ((1 - lam) * targets[idx2])
        
        mixed_arrays.append(new_array.astype(np.float32))
        mixed_targets.append(new_target.astype(np.float32))
        
    return data_arrays + mixed_arrays, targets + mixed_targets


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


def _generate_raw_training_variants(raw_data, include_noise_aug=True, exclude_windows=None):
	"""
	Builds raw-data variants for augmentation.

	Always includes the clean signal. If enabled, adds noisy copies with configured
	magnitudes across ALL channels. Noise is injected before filtering.
	
	Args:
		raw_data (np.ndarray): Raw signal data of shape (num_channels, num_samples).
		include_noise_aug (bool): Whether to include noise augmentation variants.
		exclude_windows (list): List of (start_idx, end_idx) tuples defining contraction windows.
                         If provided, noise will NOT be injected in these windows.
                         If None, noise is injected everywhere.
	"""
	base = np.asarray(raw_data, dtype=np.float32)
	variants = [base]

	if not include_noise_aug:
		return variants

	noise_magnitudes = _resolve_training_noise_magnitudes()
	if len(noise_magnitudes) == 0:
		return variants

	channels = list(range(base.shape[0]))
	for mag in noise_magnitudes:
		noisy = SignalProcessing.inject_white_noise_to_channels_excluding_windows(
			signal=base,
			target_channels=channels,
			noise_magnitudes=[mag] * len(channels),
			exclude_windows=exclude_windows,
		)
		variants.append(np.asarray(noisy, dtype=np.float32))

	return variants

# ====================================================================================
# ============================== EXTRACTION PIPELINE =================================
# ====================================================================================

def extract_bursts_from_labels(classic_data, labels, fs=Config.FS, window_length_sec=4.5):
	"""
	Extract bursts and valleys using pre-labelled contraction windows from annotations.
	
	Args:
		classic_data (np.ndarray): Preprocessed and normalized signal (num_channels, num_samples)
		labels (np.ndarray or None): Labelled windows as Nx2 array of (start_idx, end_idx) pairs.
		                             If None, returns empty bursts (no extraction).
		fs (float): Sampling frequency
		window_length_sec (float): Duration of fixed burst window in seconds
	
	Returns:
		tuple: (active_bursts, rest_valleys) - lists of 4.5-second windows
	"""
	num_samples = classic_data.shape[1]
	active_bursts = []
	rest_valleys = []
	fixed_window_samples = int(window_length_sec * fs)
	
	if labels is None or len(labels) == 0:
		return active_bursts, rest_valleys
	
	# Ensure labels is 2D
	if labels.ndim == 1:
		labels = labels.reshape(-1, 2)
	
	# Process each labelled contraction window
	last_end_idx = 0
	for window_idx, label_pair in enumerate(labels):
		start_idx = int(label_pair[0])
		end_idx = int(label_pair[1])
		
		# Clamp to valid range
		start_idx = max(0, start_idx)
		end_idx = min(num_samples, end_idx)
		
		if end_idx <= start_idx:
			continue
		
		# Extract fixed-length burst centered on the labelled window
		window_center = (start_idx + end_idx) // 2
		burst_start = max(0, window_center - fixed_window_samples // 2)
		burst_end = min(num_samples, burst_start + fixed_window_samples)
		
		# Adjust if burst extends beyond bounds
		if burst_end - burst_start < fixed_window_samples:
			burst_start = max(0, burst_end - fixed_window_samples)
		
		burst_data = classic_data[:, burst_start:burst_end]
		if burst_data.shape[1] == fixed_window_samples:
			active_bursts.append(burst_data)
		
		# Extract rest valleys between contractions
		valley_start = last_end_idx + int(0.5 * fs)  # 500ms gap
		valley_end = start_idx - int(0.5 * fs)        # 500ms before next
		
		if valley_end - valley_start > Config.WINDOW_SIZE:
			valley_data = classic_data[:, valley_start:valley_end]
			if valley_data.shape[1] >= Config.WINDOW_SIZE:
				rest_valleys.append(valley_data)
		
		last_end_idx = burst_end
	
	return active_bursts, rest_valleys

def extract_bursts_and_valleys(classic_data, tkeo_data, movement_class, fs=Config.FS, window_length_sec=4.5):
    """
    Split Pipeline Extraction:
    Runs the burst detection math on `tkeo_data`, but extracts the actual training 
    features and resting valleys from `classic_data`.
    """
    num_samples = classic_data.shape[1]
    active_bursts = []
    rest_valleys = []
    
    fixed_window_samples = int(window_length_sec * fs)
    
    # Rest Class handles fixed splitting natively
    if movement_class == 9:
        for start in range(0, num_samples - fixed_window_samples, fixed_window_samples):
            rest_valleys.append(classic_data[:, start:start+fixed_window_samples])
        return active_bursts, rest_valleys

    # ==================== TRACK A: MATH (TKEO DATA) ====================
    global_energy = np.sum(tkeo_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(global_energy, kernel_size=501)
    
    # Edge Clamping
    cutoff_samples = int(0.5 * fs)
    steady_state_value = smoothed_energy[cutoff_samples]
    
    modified_smoothed_energy = np.copy(smoothed_energy)
    modified_smoothed_energy[:cutoff_samples] = steady_state_value
    robust_max = np.percentile(modified_smoothed_energy, 95)
    
    peaks, _ = scipy.signal.find_peaks(
        modified_smoothed_energy, 
        distance=2000,
        prominence=robust_max * 0.05,
        height=robust_max * 0.60
    )
    
    valid_peaks = [p for p in peaks if p > int(1.5 * fs)]
    
    if len(valid_peaks) > 0:
        widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
            modified_smoothed_energy, valid_peaks, rel_height=0.90
        )
        
        buffer_samples = int(0.2 * fs)
        last_end_idx = 0
        
        for i in range(len(valid_peaks)):
            rising_edge_idx = int(left_ips[i])
            start_idx = max(0, rising_edge_idx - buffer_samples)
            
            # Prevent overlap collision
            if start_idx < last_end_idx:
                continue
                
            end_idx = min(num_samples, start_idx + fixed_window_samples)
            
            # ==================== TRACK B: PAYLOAD (CLASSIC DATA) ====================
            
            # 1. Extract the Rest Valley BEFORE this burst
            valley_start = last_end_idx + 500
            valley_end = start_idx - 500
            if valley_end - valley_start > Config.WINDOW_SIZE:
                rest_valleys.append(classic_data[:, valley_start:valley_end])
                
            # 2. Extract the Active Burst
            burst_data = classic_data[:, start_idx:end_idx]
            if burst_data.shape[1] == fixed_window_samples: 
                active_bursts.append(burst_data)
                
            last_end_idx = end_idx
            
    return active_bursts, rest_valleys

def slice_into_windows(data_array, increment, window_size):
    """Helper function to run the sliding window across a long array."""
    windows = []
    for step in range(0, data_array.shape[1] - window_size, increment):
        window = data_array[:, step:step+window_size]
        if window.shape[1] == window_size:
            windows.append(window.astype(np.float32))
    return windows


def concat_continuous_segments(segments, targets):
	"""
	Concatenates channel-first segments into one continuous array with per-sample targets.

	Args:
		segments (list[np.ndarray]): Each segment has shape (channels, samples).
		targets (list[np.ndarray]): Each target has shape (num_outputs,).

	Returns:
		tuple: (continuous_X, continuous_y) where
			continuous_X has shape (channels, total_samples)
			continuous_y has shape (total_samples, num_outputs)
	"""
	valid_segments = []
	valid_targets = []

	for seg, target in zip(segments, targets):
		if seg is None or target is None:
			continue

		seg_arr = np.asarray(seg, dtype=np.float32)
		target_arr = np.asarray(target, dtype=np.float32).reshape(-1)

		if seg_arr.ndim != 2 or seg_arr.shape[1] == 0:
			continue
		if target_arr.size == 0:
			continue

		valid_segments.append(seg_arr)
		valid_targets.append(target_arr)

	if len(valid_segments) == 0:
		return None, None

	continuous_X = np.concatenate(valid_segments, axis=1).astype(np.float32)
	continuous_y = np.concatenate(
		[
			np.repeat(target[np.newaxis, :], seg.shape[1], axis=0).astype(np.float32)
			for seg, target in zip(valid_segments, valid_targets)
		],
		axis=0,
	)

	return continuous_X, continuous_y

def load_and_prepare_dataset(
	base_path='./biosignal_data/secondary/raw',
	include_subjects=None,
	labelled_base_path=None,
	include_noise_aug=True,
	return_continuous=False,
):
	"""
	Load and prepare dataset using pre-labelled contraction windows when available.
	Global normalization is computed per participant across all movements.
	
	Args:
		base_path (str): Path to raw data directory (e.g., ./biosignal_data/secondary/raw)
		include_subjects (list): List of subject IDs to include. If None, includes all subjects (1-8).
		labelled_base_path (str): Path to edited/labelled data directory. If None, infers from base_path.
		include_noise_aug (bool): If True, adds configured noisy raw variants before filtering.
		return_continuous (bool): If True, returns continuous arrays instead of pre-windowed tensors.
	
	Returns:
		tuple: (X_data, y_targets) - preprocessed data
			- pre-windowed: X=(num_windows, channels, window_size), y=(num_windows, outputs)
			- continuous:  X=(channels, total_samples), y=(total_samples, outputs)
	"""
	if include_subjects is None:
		include_subjects = list(range(1, 9))  # All 8 subjects by default
	
	if not (base_path.endswith('/raw') or base_path.endswith('/edited')):
		labelled_base_path = base_path + "/secondary/edited"
	else:
		labelled_base_path = base_path.replace('/raw', '/edited')
	
	all_active_bursts = []
	all_active_targets = []
	all_rest_valleys = []
	REST_VECTOR = np.array(Config.TARGET_MAPPING[9], dtype=np.float32)

	print("Beginning Label-Based data extraction...")
	print(f"Include subjects: {include_subjects}")
	print(f"Loading raw files from: {base_path}")
	print(f"Loading labelled windows from: {labelled_base_path}")
	if include_noise_aug:
		print(f"Noise augmentation BEFORE filtering enabled: {_resolve_training_noise_magnitudes()}")
	else:
		print("Noise augmentation BEFORE filtering disabled")
	print("NOTE: Global normalization computed per participant across all movements.\n")

	# Process each participant separately using participant-level robust min/max.
	for p in include_subjects:
		print(f"Processing Subject {p}...")
		
		# First pass: Collect data for this participant and read robust min/max.
		movements_data = {}
		movements_labels = {}
		participant_minmax = None
		
		for m in range(1, 10):
			if hasattr(Config, 'SECONDARY_BLACKLIST') and (p, m) in Config.SECONDARY_BLACKLIST:
				continue
			
			# Try to load pre-labelled windows from the labelled directory
			label_path = os.path.join(labelled_base_path, f'Soggetto{p}', f'Movimento{m}_labelled.mat')
			if not os.path.exists(label_path):
				raise FileNotFoundError(f"Expected file not found: {label_path}")
			
			try:
				label_mat = scipy.io.loadmat(label_path)
				movements_data[m] = label_mat['EMGDATA']

				movements_labels[m] = label_mat['LABELS']

				if participant_minmax is None:
					participant_minmax = _extract_robust_minmax_matrix(label_mat)
			except Exception as e:
				print(f"  [WARNING] Could not load labels for Movimento{m}: {e}")
		
		if not movements_data:
			continue

		if participant_minmax is None:
			print("  [WARNING] MIN_MAX_ROBUST not found. Falling back to participant percentiles.")
			global_mins = np.zeros(Config.NUM_CHANNELS, dtype=np.float32)
			global_maxs = np.zeros(Config.NUM_CHANNELS, dtype=np.float32)

			for c in range(Config.NUM_CHANNELS):
				all_channel_values = []
				for _, raw_data in movements_data.items():
					all_channel_values.extend(raw_data[c, :].flatten())

				all_channel_values = np.array(all_channel_values)
				global_mins[c] = np.percentile(all_channel_values, 1.0)
				global_maxs[c] = np.percentile(all_channel_values, 99.0)

			participant_minmax = np.column_stack((global_mins, global_maxs)).astype(np.float32)

		print("  Robust normalization ranges per channel loaded.")
		
		# Second pass: Preprocess each movement with robust normalization.
		for m, raw_data in movements_data.items():
			# Extract labels for noise exclusion (skip for rest movement 9)
			exclude_windows = None
			if m != 9 and m in movements_labels and movements_labels[m] is not None:
				labels = movements_labels[m]
				if labels.ndim == 1:
					labels = labels.reshape(-1, 2)
				if labels.shape[0] > 0:
					exclude_windows = labels
			
			raw_variants = _generate_raw_training_variants(raw_data, include_noise_aug=include_noise_aug, exclude_windows=exclude_windows)

			for raw_variant in raw_variants:
				classic_data = np.zeros_like(raw_variant, dtype=np.float32)

				for c in range(Config.NUM_CHANNELS):
					# ===== PREPROCESSING PIPELINE FOR NN ANALYSIS =====
					# 1. Notch filter (remove 50Hz powerline noise)
					notch = SignalProcessing.notchFilter(raw_variant[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)

					# 2. Bandpass filter (remove movement artifacts and high-freq noise)
					band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)

					# 3. Rectify
					rectified = np.abs(band)

					# Normalize after rectification using larger absolute robust bound.
					scale = SignalProcessing.get_rectified_scale_from_minmax(
						participant_minmax[c, 0],
						participant_minmax[c, 1],
					)
					classic_data[c, :] = np.clip(rectified / scale, 0.0, 1.0)

				# Extract bursts using labelled windows or automatic fallback
				if m in movements_labels:
					active_bursts, rest_valleys = extract_bursts_from_labels(classic_data, movements_labels[m])
				else:
					# Fallback: Use automatic detection (requires TKEO pipeline)
					tkeo_data = np.zeros_like(raw_variant, dtype=np.float32)
					for c in range(Config.NUM_CHANNELS):
						notch = SignalProcessing.notchFilter(raw_variant[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
						band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
						teager = SignalProcessing.tkeo(band)
						rectified_teager = np.abs(teager)
						envelope = SignalProcessing.lowpassFilter(rectified_teager, fs=Config.FS, cutoff=5.0)
						tkeo_max = np.percentile(envelope, 99.9) + 1e-6
						tkeo_data[c, :] = np.clip(envelope / tkeo_max, 0.0, 1.0)

					active_bursts, rest_valleys = extract_bursts_and_valleys(classic_data, tkeo_data, movement_class=m)

				target_vector = np.array(Config.TARGET_MAPPING[m], dtype=np.float32)

				for b in active_bursts:
					all_active_bursts.append(b)
					all_active_targets.append(target_vector)
				all_rest_valleys.extend(rest_valleys)
		
		print(f"  Subject {p} processed. Bursts: {len(all_active_bursts)}, Rest valleys: {len(all_rest_valleys)}.")

	# 2. Apply Mixup to the full 4.5-second Active arrays
	if hasattr(Config, 'MIXUP_RATIO') and Config.MIXUP_RATIO > 0:
		all_active_bursts, all_active_targets = apply_mixup(
			all_active_bursts, all_active_targets,
			alpha=Config.MIXUP_ALPHA, mixup_ratio=Config.MIXUP_RATIO
		)

	if return_continuous:
		continuous_segments = []
		continuous_targets = []

		for burst, target in zip(all_active_bursts, all_active_targets):
			continuous_segments.append(np.asarray(burst, dtype=np.float32))
			continuous_targets.append(np.asarray(target, dtype=np.float32))

		for valley in all_rest_valleys:
			continuous_segments.append(np.asarray(valley, dtype=np.float32))
			continuous_targets.append(REST_VECTOR)

		X_data, y_targets = concat_continuous_segments(continuous_segments, continuous_targets)
		if X_data is None:
			print("No valid continuous data extracted.")
			return None, None

		print(f"Continuous dataset generated! Total Samples: {X_data.shape[1]}")
		print(f"Data range after normalization: [{X_data.min():.4f}, {X_data.max():.4f}] (expected: [0.0, 1.0])\n")
		return X_data, y_targets

	X_data = []
	y_targets = []

	# 3. Slice Active bursts into 500ms windows and apply Magnitude Warping
	print("Slicing Active arrays and applying Magnitude Warping...")
	for burst, target in zip(all_active_bursts, all_active_targets):
		windows = slice_into_windows(burst, Config.INCREMENT, Config.WINDOW_SIZE)
		for w in windows:
			X_data.append(w)
			y_targets.append(target)

			X_data.append(apply_magnitude_warping(w, sigma=0.25))
			y_targets.append(target)
			X_data.append(apply_magnitude_warping(w, sigma=0.40))
			y_targets.append(target)

	# 4. Process Rest Valleys: Slice first, then augment
	print("Slicing Rest Valleys and applying Augmentation...")
	rest_windows_unaugmented = []
	rest_targets_unaugmented = []

	for valley in all_rest_valleys:
		windows = slice_into_windows(valley, Config.INCREMENT, Config.WINDOW_SIZE)
		for w in windows:
			rest_windows_unaugmented.append(w)
			rest_targets_unaugmented.append(REST_VECTOR)

	# Mixup on Rest Windows
	if hasattr(Config, 'REST_MIXUP_RATIO') and Config.REST_MIXUP_RATIO > 0:
		all_rest_windows, all_rest_targets = apply_mixup(
			rest_windows_unaugmented, rest_targets_unaugmented,
			alpha=Config.REST_MIXUP_ALPHA, mixup_ratio=Config.REST_MIXUP_RATIO
		)
	else:
		all_rest_windows, all_rest_targets = rest_windows_unaugmented, rest_targets_unaugmented

	# Magnitude Warping on Rest Windows
	for w, target in zip(all_rest_windows, all_rest_targets):
		X_data.append(w)
		y_targets.append(target)

		X_data.append(apply_magnitude_warping(w, sigma=0.25))
		y_targets.append(target)
		X_data.append(apply_magnitude_warping(w, sigma=0.40))
		y_targets.append(target)

	X_data = np.array(X_data, dtype=np.float32)
	y_targets = np.array(y_targets, dtype=np.float32)

	print(f"Dataset generated! Total Windows: {X_data.shape[0]}")
	print(f"Data range after normalization: [{X_data.min():.4f}, {X_data.max():.4f}] (expected: [0.0, 1.0])\n")
	return X_data, y_targets

# ====================================================================================
# ============================== COLLECTED DATA LOADING ==============================
# ====================================================================================

def load_collected_data(
	folder_path,
	labelled_folder_path=None,
	augment=True,
	include_noise_aug=True,
	include_participants=None,
	return_continuous=False,
):
	"""
	Load and preprocess collected .mat files using pre-labelled windows when available.
	
	Args:
		folder_path (str): Path to folder containing raw .mat files
		labelled_folder_path (str): Path to folder containing labelled .mat files. If None, infers from folder_path.
		augment (bool): Whether to apply augmentation (magnitude warping)
		include_noise_aug (bool): Whether to add configured noisy raw variants before filtering.
		include_participants (list[int] | None): Optional participant IDs to include.
			Only files matching P#M#.mat for selected participants are loaded.
		return_continuous (bool): If True, returns continuous arrays instead of pre-windowed tensors.
	
	Returns:
		tuple: (X_data, y_data) or (None, None) if no files found
	"""
	if not os.path.exists(folder_path):
		print(f"[WARNING] Folder does not exist: {folder_path}")
		return None, None
	
	# Infer labelled folder path
	if labelled_folder_path is None:
		labelled_folder_path = folder_path.replace('/raw', '/edited')
		if not os.path.exists(labelled_folder_path):
			labelled_folder_path = folder_path.replace('collected_data', 'biosignal_data/collected/edited')
	
	mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and not f.endswith('_labelled.mat')]
	if len(mat_files) == 0:
		print(f"[WARNING] No .mat files found in {folder_path}")
		return None, None

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
			return None, None
	
	print(f"[Collected Data] Found {len(mat_files)} .mat files in {folder_path}")
	print(f"[Collected Data] Using labelled windows from: {labelled_folder_path}")
	if include_noise_aug and augment:
		print(f"[Collected Data] Noise augmentation BEFORE filtering enabled: {_resolve_training_noise_magnitudes()}")
	else:
		print("[Collected Data] Noise augmentation BEFORE filtering disabled")

	REST_VECTOR = np.array(Config.TARGET_MAPPING[9], dtype=np.float32)
	
	X_data = []
	y_data = []
	
	for mat_file in sorted(mat_files):
		file_path = os.path.join(folder_path, mat_file)
		
		try:
			mat = scipy.io.loadmat(file_path)
			if 'EMGDATA' not in mat:
				print(f"  [SKIP] {mat_file}: No EMGDATA found")
				continue
			
			raw_data = mat['EMGDATA']

			labelled_file = os.path.join(labelled_folder_path, mat_file.replace('.mat', '_labelled.mat'))
			label_mat = None
			robust_minmax = None
			if os.path.exists(labelled_file):
				try:
					label_mat = scipy.io.loadmat(labelled_file)
					robust_minmax = _extract_robust_minmax_matrix(label_mat)
				except Exception as e:
					print(f"  [WARNING] Could not load labelled file for robust min/max: {e}")
					label_mat = None
			
			# Extract target from filename (assumes format: "P1M1.mat")
			filename_base = os.path.splitext(mat_file)[0]  # Remove .mat
			match = re.search(r'P(\d+)M(\d+)', filename_base)
			if match:
				movement_id = int(match.group(2))
				if movement_id in Config.TARGET_MAPPING:
					target_vector = np.array(Config.TARGET_MAPPING[movement_id], dtype=np.float32)
				else:
					print(f"  [SKIP] {mat_file}: Unknown movement ID {movement_id}")
					continue
			else:
				print(f"  [SKIP] {mat_file}: Could not parse movement from filename. Use format P#M#.mat")
				continue
			
			# Extract labels for noise exclusion (skip for rest movement 9)
			exclude_windows = None
			if label_mat is not None and movement_id != 9:
				if 'LABELS' in label_mat:
					labels = label_mat['LABELS']
					if labels.ndim == 1:
						labels = labels.reshape(-1, 2)
					if labels.shape[0] > 0:
						exclude_windows = labels
			
			raw_variants = _generate_raw_training_variants(
				raw_data,
				include_noise_aug=(include_noise_aug and augment),
				exclude_windows=exclude_windows,
			)
			
			variant_units_total = 0
			for raw_variant in raw_variants:
				processed_data = np.zeros_like(raw_variant, dtype=np.float32)
				
				for c in range(Config.NUM_CHANNELS):
					# ===== PREPROCESSING PIPELINE FOR NN ANALYSIS =====
					# 1. Notch filter (remove 50Hz powerline noise)
					notch = SignalProcessing.notchFilter(raw_variant[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)

					# 2. Bandpass filter (remove movement artifacts and high-freq noise)
					band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)

					# 3. Rectify
					rectified = np.abs(band)

					# 4. Normalize using MIN_MAX_ROBUST when available.
					if robust_minmax is not None:
						scale = SignalProcessing.get_rectified_scale_from_minmax(
							robust_minmax[c, 0],
							robust_minmax[c, 1],
						)
						normalised = np.clip(rectified / scale, 0.0, 1.0)
					else:
						min_val = np.percentile(rectified, 1.0)
						max_val = np.percentile(rectified, 99.0)
						range_span = max_val - min_val
						if range_span < 1e-6:
							range_span = 1e-6
						normalised = np.clip((rectified - min_val) / range_span, 0.0, 1.0)
					processed_data[c, :] = normalised

				if return_continuous:
					segments = []
					segment_targets = []

					if label_mat is not None and 'LABELS' in label_mat:
						labels = label_mat['LABELS']
						bursts, valleys = extract_bursts_from_labels(processed_data, labels)

						for burst in bursts:
							segments.append(burst)
							segment_targets.append(target_vector)
						for valley in valleys:
							segments.append(valley)
							segment_targets.append(REST_VECTOR)
					else:
						segments = [processed_data]
						segment_targets = [target_vector]

					variant_units_total += len(segments)
					if len(segments) == 0:
						continue

					for seg, seg_target in zip(segments, segment_targets):
						X_data.append(np.asarray(seg, dtype=np.float32))
						y_data.append(np.asarray(seg_target, dtype=np.float32))

						if augment:
							X_data.append(apply_magnitude_warping(seg, sigma=0.25))
							y_data.append(np.asarray(seg_target, dtype=np.float32))
							X_data.append(apply_magnitude_warping(seg, sigma=0.40))
							y_data.append(np.asarray(seg_target, dtype=np.float32))
				else:
					# Try to use pre-labelled windows
					windows = []

					if label_mat is not None:
						try:
							if 'LABELS' in label_mat:
								labels = label_mat['LABELS']
								bursts, valleys = extract_bursts_from_labels(processed_data, labels)

								# Use bursts + valleys for windowing
								all_contraction_data = bursts + valleys
								for data in all_contraction_data:
									windows.extend(slice_into_windows(data, Config.INCREMENT, Config.WINDOW_SIZE))
							else:
								# No LABELS in file, fall back to slicing
								windows = slice_into_windows(processed_data, Config.INCREMENT, Config.WINDOW_SIZE)
						except Exception as e:
							print(f"  [WARNING] Could not use labelled file, falling back to slicing: {e}")
							windows = slice_into_windows(processed_data, Config.INCREMENT, Config.WINDOW_SIZE)
					else:
						# No labelled file found, slice entire signal
						windows = slice_into_windows(processed_data, Config.INCREMENT, Config.WINDOW_SIZE)

					variant_units_total += len(windows)
					if len(windows) == 0:
						continue

					for w in windows:
						X_data.append(w)
						y_data.append(target_vector)

						# Apply augmentation if requested
						if augment:
							X_data.append(apply_magnitude_warping(w, sigma=0.25))
							y_data.append(target_vector)
							X_data.append(apply_magnitude_warping(w, sigma=0.40))
							y_data.append(target_vector)

			if variant_units_total == 0:
				print(f"  [SKIP] {mat_file}: No segments/windows extracted")
				continue

			if label_mat is not None and 'LABELS' in label_mat:
				unit_label = 'segments' if return_continuous else 'samples'
				print(f"  [LOADED] {mat_file} (using labelled data across {len(raw_variants)} variants → {variant_units_total} {unit_label})")
			
			if not os.path.exists(labelled_file):
				unit_label = 'segments' if return_continuous else 'windows'
				aug_multiplier = 3 if augment else 1
				print(
					f"  [LOADED] {mat_file} "
					f"({variant_units_total} {unit_label} across {len(raw_variants)} variants "
					f"→ {variant_units_total * aug_multiplier} items)"
				)
		
		except Exception as e:
			print(f"  [ERROR] {mat_file}: {e}")
			continue

	if len(X_data) == 0:
		print(f"[Collected Data] No valid data extracted from {folder_path}")
		return None, None

	if return_continuous:
		continuous_X, continuous_y = concat_continuous_segments(X_data, y_data)
		if continuous_X is None:
			print(f"[Collected Data] No valid continuous data extracted from {folder_path}")
			return None, None

		print(f"[Collected Data] Total continuous samples from {folder_path}: {continuous_X.shape[1]}")
		return continuous_X, continuous_y
	
	X_data = np.array(X_data, dtype=np.float32)
	y_data = np.array(y_data, dtype=np.float32)
	
	print(f"[Collected Data] Total samples from {folder_path}: {X_data.shape[0]}")
	return X_data, y_data

if __name__ == "__main__":
    X, y = load_and_prepare_dataset()