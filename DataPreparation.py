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
    Applies Mixup to the FULL 3-second bursts before they are windowed.
    This ensures the ramp-up and ramp-down phases of different movements align perfectly.
    """
    if mixup_ratio <= 0.0 or len(bursts) == 0:
        return bursts, targets

    num_mixups = int(len(bursts) * mixup_ratio)
    print(f"\n[Augmentation] Generating {num_mixups} Phase-Aligned 3-Second Mixup bursts...")
    
    mixed_bursts = []
    mixed_targets = []
    
    for _ in range(num_mixups):
        # Pick two completely random full bursts
        idx1, idx2 = np.random.choice(len(bursts), 2, replace=False)
        lam = np.random.beta(alpha, alpha)
        
        # Mix the entire 3-second arrays
        new_burst = (lam * bursts[idx1]) + ((1 - lam) * bursts[idx2])
        new_target = (lam * targets[idx1]) + ((1 - lam) * targets[idx2])
        
        mixed_bursts.append(new_burst.astype(np.float32))
        mixed_targets.append(new_target.astype(np.float32))
        
    # Append the synthetic bursts to the original lists
    return bursts + mixed_bursts, targets + mixed_targets

# ====================================================================================
# ============================== EXTRACTION PIPELINE =================================
# ====================================================================================

def extract_bursts_and_valleys(signal_data, movement_class):
    """
    Extracts the full 3-second active holds AND the resting valleys between them.
    Does NOT slice them into windows yet.
    """
    num_samples = signal_data.shape[1]
    active_bursts = []
    rest_valleys = []
    
    if movement_class == 9:
        for start in range(0, num_samples - 3000, 3000):
            rest_valleys.append(signal_data[:, start:start+3000])
        return active_bursts, rest_valleys

    summed_energy = np.sum(signal_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(summed_energy, kernel_size=1001)
    peaks, _ = scipy.signal.find_peaks(smoothed_energy, distance=4000, prominence=np.max(smoothed_energy)*0.2)
    
    half_hold = int(1.5 * Config.FS)
    last_end_idx = 0
    
    for peak in peaks:
        start_idx = peak - half_hold
        end_idx = peak + half_hold
        
        if start_idx < 0 or end_idx > num_samples:
            continue 
            
        valley_start = last_end_idx + 500
        valley_end = start_idx - 500
        if valley_end - valley_start > Config.WINDOW_SIZE:
            rest_valleys.append(signal_data[:, valley_start:valley_end])

        burst_data = signal_data[:, start_idx:end_idx]
        if burst_data.shape[1] == half_hold * 2:  # Ensure exactly 3000 samples
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
    
    # 1. Collect all raw 3-second bursts
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
            
            clean_data = np.zeros_like(raw_data, dtype=np.float32)
            for c in range(Config.NUM_CHANNELS):
                sig = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
                sig = SignalProcessing.bandpassFilter(sig, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
                clean_data[c, :] = np.abs(sig) 
            
            active_bursts, rest_valleys = extract_bursts_and_valleys(clean_data, movement_class=m)
            target_vector = np.array(Config.TARGET_MAPPING[m], dtype=np.float32)
            
            for b in active_bursts:
                all_active_bursts.append(b)
                all_active_targets.append(target_vector)
            all_rest_valleys.extend(rest_valleys)
                
        print(f"Processed Subject {p}...")

    # 2. Apply Phase-Aligned Mixup on the full 3-second arrays
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