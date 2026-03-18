import os
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

def apply_phase_aligned_mixup(bursts, targets, alpha=0.2, mixup_ratio=0.5):
    """
    Applies Mixup to the FULL bursts before they are windowed.
    This ensures the ramp-up and ramp-down phases of different movements align perfectly.
    """
    if mixup_ratio <= 0.0 or len(bursts) == 0:
        return bursts, targets

    num_mixups = int(len(bursts) * mixup_ratio)
    print(f"\n[Augmentation] Generating {num_mixups} Phase-Aligned Mixup bursts...")
    
    mixed_bursts = []
    mixed_targets = []
    
    for _ in range(num_mixups):
        # Pick two completely random full bursts
        idx1, idx2 = np.random.choice(len(bursts), 2, replace=False)
        lam = np.random.beta(alpha, alpha)
        
        # Mix the entire arrays
        new_burst = (lam * bursts[idx1]) + ((1 - lam) * bursts[idx2])
        new_target = (lam * targets[idx1]) + ((1 - lam) * targets[idx2])
        
        mixed_bursts.append(new_burst.astype(np.float32))
        mixed_targets.append(new_target.astype(np.float32))
        
    # Append the synthetic bursts to the original lists
    return bursts + mixed_bursts, targets + mixed_targets

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
            # (Buffer by 500 samples on each side to ensure no overlap with active muscles)
            valley_start = last_end_idx + 500
            valley_end = start_idx - 500
            if valley_end - valley_start > Config.WINDOW_SIZE:
                rest_valleys.append(classic_data[:, valley_start:valley_end])
                
            # 2. Extract the Active Burst
            burst_data = classic_data[:, start_idx:end_idx]
            if burst_data.shape[1] == fixed_window_samples: # Ensure exactly 4.5s
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

def load_and_prepare_dataset(base_path='./secondary_data'):
    all_active_bursts = []
    all_active_targets = []
    all_rest_valleys = []
    REST_VECTOR = np.array(Config.TARGET_MAPPING[9], dtype=np.float32)
    
    print("Beginning Phase-Aligned data extraction...")
    
    # 1. Collect all raw bursts via the Split Pipeline
    for p in range(1, 9):       
        for m in range(1, 10):  
            # Handle Blacklist
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
                # Base Conditioning
                notch = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
                band = SignalProcessing.bandpassFilter(notch, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
                
                # Track B: Classic Features
                rectified_classic = np.abs(band)
                classic_data[c, :] = rectified_classic
                
                # Track A: TKEO Envelope Generation
                teager = SignalProcessing.tkeo(band)
                rectified_teager = np.abs(teager)
                envelope = SignalProcessing.lowpassFilter(rectified_teager, fs=Config.FS, cutoff=5.0)
                
                # Normalize TKEO so the math thresholding works correctly
                tkeo_max = np.percentile(envelope, 99.9) + 1e-6
                tkeo_data[c, :] = np.clip(envelope / tkeo_max, 0.0, 1.0)
            
            active_bursts, rest_valleys = extract_bursts_and_valleys(classic_data, tkeo_data, movement_class=m)
            target_vector = np.array(Config.TARGET_MAPPING[m], dtype=np.float32)
            
            for b in active_bursts:
                all_active_bursts.append(b)
                all_active_targets.append(target_vector)
            all_rest_valleys.extend(rest_valleys)
                
        print(f"Processed Subject {p}...")

    # 2. Apply Phase-Aligned Mixup on the full 4.5-second arrays
    if hasattr(Config, 'MIXUP_RATIO') and Config.MIXUP_RATIO > 0:
        all_active_bursts, all_active_targets = apply_phase_aligned_mixup(
            all_active_bursts, all_active_targets, 
            alpha=Config.MIXUP_ALPHA, mixup_ratio=Config.MIXUP_RATIO
        )
        
    X_data = []
    y_targets = []
    
    # 3. Slice the bursts into 500ms windows and apply Magnitude Warping
    print("Slicing arrays and applying Magnitude Warping...")
    for burst, target in zip(all_active_bursts, all_active_targets):
        windows = slice_into_windows(burst, Config.INCREMENT, Config.WINDOW_SIZE)
        for w in windows:
            X_data.append(w)
            y_targets.append(target)
            
            # Warp the windows
            X_data.append(apply_magnitude_warping(w, sigma=0.25))
            y_targets.append(target)
            X_data.append(apply_magnitude_warping(w, sigma=0.40))
            y_targets.append(target)
            
    # 4. Slice the inter-burst valleys (Rest states)
    for valley in all_rest_valleys:
        windows = slice_into_windows(valley, Config.INCREMENT, Config.WINDOW_SIZE)
        for w in windows:
            X_data.append(w)
            y_targets.append(REST_VECTOR)

    X_data = np.array(X_data, dtype=np.float32)
    y_targets = np.array(y_targets, dtype=np.float32)
    
    print(f"Dataset generated! Total Windows: {X_data.shape[0]}")
    return X_data, y_targets

if __name__ == "__main__":
    # Test the extraction
    X, y = load_and_prepare_dataset()