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
    """
    Applies a smooth, random multiplier curve to the 500ms window.
    This simulates the natural variation in muscle force without changing the movement.
    """
    warp_steps = np.linspace(0, window.shape[1], knot+2)
    
    # Generate random multipliers around 1.0 (e.g., 0.8 to 1.2)
    random_multipliers = np.random.normal(loc=1.0, scale=sigma, size=(window.shape[0], knot+2))
    
    warped_window = np.zeros_like(window)
    for i in range(window.shape[0]):
        # Create a smooth cubic spline from the random multipliers
        interpolator = scipy.interpolate.CubicSpline(warp_steps, random_multipliers[i, :])
        smooth_curve = interpolator(np.arange(window.shape[1]))
        
        # Multiply the original signal by the smooth random curve
        warped_window[i, :] = window[i, :] * smooth_curve
        
    return warped_window

# =================================================================================
# =========================== DATA PREPARATION PIPELINE ===========================
# =================================================================================

def detect_bursts_and_extract(signal_data, movement_class):
    """
    Finds the 10 active muscle contractions in the data, extracts the 3-second
    isometric hold, AND extracts the resting valleys between them.
    Returns: (active_windows, rest_windows)
    """
    num_channels, num_samples = signal_data.shape
    active_windows = []
    rest_windows = []
    
    # Movement 9 is 'Rest'. Just sample it all as rest.
    if movement_class == 9:
        for start in range(0, num_samples - Config.WINDOW_SIZE, 3000):
            for step in range(start, start + 3000 - Config.WINDOW_SIZE, Config.INCREMENT):
                window = signal_data[:, step:step+Config.WINDOW_SIZE]
                if window.shape[1] == Config.WINDOW_SIZE:
                    rest_windows.append(window)
        return active_windows, rest_windows

    # Find the peaks
    summed_energy = np.sum(signal_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(summed_energy, kernel_size=1001)
    peaks, _ = scipy.signal.find_peaks(smoothed_energy, distance=4000, prominence=np.max(smoothed_energy)*0.2)
    
    half_hold = int(1.5 * Config.FS)
    last_end_idx = 0  # Keep track of where the last burst ended
    
    for peak in peaks:
        start_idx = peak - half_hold
        end_idx = peak + half_hold
        
        if start_idx < 0 or end_idx > num_samples:
            continue 
            
        # 1. EXTRACT THE RESTING VALLEY (From end of last burst to start of this one)
        # We leave a small 500ms buffer so we don't accidentally grab the ramp-up phase
        valley_start = last_end_idx + 500
        valley_end = start_idx - 500
        if valley_end - valley_start > Config.WINDOW_SIZE:
            rest_data = signal_data[:, valley_start:valley_end]
            for step in range(0, rest_data.shape[1] - Config.WINDOW_SIZE, Config.INCREMENT):
                window = rest_data[:, step:step+Config.WINDOW_SIZE]
                if window.shape[1] == Config.WINDOW_SIZE:
                    rest_windows.append(window)

        # 2. EXTRACT THE ACTIVE BURST
        burst_data = signal_data[:, start_idx:end_idx]
        for step in range(0, burst_data.shape[1] - Config.WINDOW_SIZE, Config.INCREMENT):
            window = burst_data[:, step:step+Config.WINDOW_SIZE]
            if window.shape[1] == Config.WINDOW_SIZE:
                active_windows.append(window)
                
        last_end_idx = end_idx
            
    return active_windows, rest_windows

def load_and_prepare_dataset(base_path='./secondary_data'):
    """
    Loops through the dataset, cleans the signals, extracts the sliding windows,
    and pairs them with the correct kinematic targets.
    """
    X_data = []
    y_targets = []
    
    print("Beginning automated data extraction and labelling...")
    REST_VECTOR = Config.TARGET_MAPPING[9] # [0.0, 0.0, 0.0, 0.0]
    
    for p in range(1, 9):       # Subjects 1 to 8
        for m in range(1, 10):  # Movements 1 to 9
            if (p, m) in Config.CORRUPTED_TRIALS:
                print(f"Skipping Blacklisted Trial: Subject {p}, Movement {m}")
                continue
            file_path = os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}.mat')
            
            if not os.path.exists(file_path):
                continue
                
            mat = scipy.io.loadmat(file_path)
            if 'EMGDATA' not in mat:
                continue
                
            raw_data = mat['EMGDATA']
            
            # 1. Clean the entire file first using your pipeline
            clean_data = np.zeros_like(raw_data)
            for c in range(Config.NUM_CHANNELS):
                # Using 30Hz highcut here for ECG removal as discussed
                sig = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
                sig = SignalProcessing.bandpassFilter(sig, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
                clean_data[c, :] = np.abs(sig) # Rectify
            
            # 2. Automatically find bursts and extract overlapping 500ms windows
            active_windows, rest_windows = detect_bursts_and_extract(clean_data, movement_class=m)
            target_vector = Config.TARGET_MAPPING[m]
            
            # 3. Append to our dataset
            for w in active_windows:
                X_data.append(w)
                y_targets.append(target_vector)

                X_data.append(apply_magnitude_warping(w, sigma=0.25))
                y_targets.append(target_vector)
                
                X_data.append(apply_magnitude_warping(w, sigma=0.40))
                y_targets.append(target_vector)
                
            # explicitly label the inter-burst valleys as absolute zero
            for w in rest_windows:
                X_data.append(w)
                y_targets.append(REST_VECTOR)
                
        print(f"Processed Subject {p}...")

    X_data = np.array(X_data)
    y_targets = np.array(y_targets)
    
    print(f"Dataset generated! Total Windows: {X_data.shape[0]}")
    print(f"X shape: {X_data.shape} | y shape: {y_targets.shape}")
    
    return X_data, y_targets

if __name__ == "__main__":
    # Test the extraction
    X, y = load_and_prepare_dataset()