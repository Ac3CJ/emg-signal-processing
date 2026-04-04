import os
import re
import scipy.io
import numpy as np
import scipy.signal
import scipy.interpolate

import SignalProcessing 
import ControllerConfiguration as Config

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

# ====================================================================================
# ============================== EXTRACTION PIPELINE =================================
# ====================================================================================

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

def load_and_prepare_dataset(base_path='./secondary_data', include_subjects=None):
    """
    Load and prepare dataset with optional subject filtering for Leave-One-Subject-Out validation.
    
    Args:
        base_path (str): Path to secondary_data directory
        include_subjects (list): List of subject IDs to include (e.g., [1,2,3,4,5,6,7] for training).
                               If None, includes all subjects (1-8).
    
    Returns:
        tuple: (X_data, y_targets) - preprocessed training data
    """
    if include_subjects is None:
        include_subjects = list(range(1, 9))  # All 8 subjects by default
    
    all_active_bursts = []
    all_active_targets = []
    all_rest_valleys = []
    REST_VECTOR = np.array(Config.TARGET_MAPPING[9], dtype=np.float32)

    print("Beginning Split-Pipeline data extraction...")
    print(f"Include subjects: {include_subjects}")
    print("NOTE: Raw data is recorded as-is. Preprocessing and normalization applied only for NN analysis.\n")

    # 1. Collect all raw bursts via the Split Pipeline
    for p in range(1, 9):
        if p not in include_subjects:
            continue
        for m in range(1, 10):
            if hasattr(Config, 'CORRUPTED_TRIALS') and (p, m) in Config.CORRUPTED_TRIALS:
                continue

            file_path = os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}.mat')
            if not os.path.exists(file_path): continue

            mat = scipy.io.loadmat(file_path)
            if 'EMGDATA' not in mat: continue
            raw_data = mat['EMGDATA']

            classic_data = np.zeros_like(raw_data, dtype=np.float32)
            tkeo_data = np.zeros_like(raw_data, dtype=np.float32)

            for c in range(Config.NUM_CHANNELS):
                # ===== PREPROCESSING PIPELINE FOR NN ANALYSIS =====
                # 1. Notch filter (remove 50Hz powerline noise)
                notch = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)

                # 2. Bandpass filter (remove movement artifacts and high-freq noise)
                band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)

                # 3. Rectify for classic pipeline
                rectified_classic = np.abs(band)

                classic_data[c, :] = rectified_classic

                # ===== TKEO PIPELINE (for burst detection) =====
                teager = SignalProcessing.tkeo(band)
                rectified_teager = np.abs(teager)
                envelope = SignalProcessing.lowpassFilter(rectified_teager, fs=Config.FS, cutoff=5.0)

                tkeo_max = np.percentile(envelope, 99.9) + 1e-6
                tkeo_data[c, :] = np.clip(envelope / tkeo_max, 0.0, 1.0)

            classic_data = SignalProcessing.applyGlobalNormalization(classic_data, percentiles=(1.0, 99.0))

            active_bursts, rest_valleys = extract_bursts_and_valleys(classic_data, tkeo_data, movement_class=m)
            target_vector = np.array(Config.TARGET_MAPPING[m], dtype=np.float32)

            for b in active_bursts:
                all_active_bursts.append(b)
                all_active_targets.append(target_vector)
            all_rest_valleys.extend(rest_valleys)

        print(f"Processed Subject {p}...")

    # 2. Apply Mixup to the full 4.5-second Active arrays
    if hasattr(Config, 'MIXUP_RATIO') and Config.MIXUP_RATIO > 0:
        all_active_bursts, all_active_targets = apply_mixup(
            all_active_bursts, all_active_targets,
            alpha=Config.MIXUP_ALPHA, mixup_ratio=Config.MIXUP_RATIO
        )

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
    print(f"Data range after normalization: [{X_data.min():.4f}, {X_data.max():.4f}] (expected: [-1.0, 1.0])\n")
    return X_data, y_targets

# ====================================================================================
# ============================== COLLECTED DATA LOADING ==============================
# ====================================================================================

def load_collected_data(folder_path, augment=True):
    """
    Load and preprocess collected .mat files from a folder.
    Applies the same preprocessing pipeline as secondary data.
    
    Args:
        folder_path (str): Path to folder containing .mat files
        augment (bool): Whether to apply augmentation (magnitude warping)
    
    Returns:
        tuple: (X_data, y_data) or (None, None) if no files found
    """
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder does not exist: {folder_path}")
        return None, None
    
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    if len(mat_files) == 0:
        print(f"[WARNING] No .mat files found in {folder_path}")
        return None, None
    
    print(f"[Collected Data] Found {len(mat_files)} .mat files in {folder_path}")
    
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
            
            # Preprocess each channel
            processed_data = np.zeros_like(raw_data, dtype=np.float32)
            for c in range(Config.NUM_CHANNELS):
                # 1. Notch filter (remove 50Hz powerline noise)
                notch = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)

                # 2. Bandpass filter
                band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)

                # 3. Rectify
                rectified = np.abs(band)

                # 4. Normalize to -1 to 1 range for NN input
                normalised = SignalProcessing.normaliseSignal(rectified, output_range=(-1.0, 1.0))
                processed_data[c, :] = normalised
            
            # Extract target from filename (assumes format: "P1M1.mat")
            # For collected data, default to guessing from filename if possible
            try:
                filename_base = os.path.splitext(mat_file)[0]  # Remove .mat
                # Try to extract P and M numbers from filename like "P1M3"
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
            except Exception as e:
                print(f"  [SKIP] {mat_file}: Error parsing filename - {e}")
                continue
            
            # Slice into windows (no phase-aligned burst detection for collected data)
            windows = slice_into_windows(processed_data, Config.INCREMENT, Config.WINDOW_SIZE)
            
            if len(windows) == 0:
                print(f"  [SKIP] {mat_file}: No windows extracted")
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
            
            print(f"  [LOADED] {mat_file} ({len(windows)} windows → {len(windows) * (3 if augment else 1)} samples)")
            
        except Exception as e:
            print(f"  [ERROR] {mat_file}: {e}")
            continue
    
    if len(X_data) == 0:
        print(f"[Collected Data] No valid data extracted from {folder_path}")
        return None, None
    
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    
    print(f"[Collected Data] Total samples from {folder_path}: {X_data.shape[0]}")
    return X_data, y_data

if __name__ == "__main__":
    X, y = load_and_prepare_dataset()