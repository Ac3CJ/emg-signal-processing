import os
import scipy.io
import numpy as np
import scipy.signal

import SignalProcessing 
import ControllerConfiguration as Config

def detect_bursts_and_extract(signal_data, movement_class):
    """
    Finds the 10 active muscle contractions in the data, extracts the 3-second
    isometric hold, and slices it into 500ms windows.
    """
    num_channels, num_samples = signal_data.shape
    extracted_windows = []
    
    # Movement 1 is 'Rest'. There are no peaks to find, so we just blindly sample it.
    if movement_class == 1:
        for start in range(0, num_samples - Config.WINDOW_SIZE, 3000): # Jump by 3 seconds
            for step in range(start, start + 3000 - Config.WINDOW_SIZE, Config.INCREMENT):
                window = signal_data[:, step:step+Config.WINDOW_SIZE]
                if window.shape[1] == Config.WINDOW_SIZE:
                    extracted_windows.append(window)
        return extracted_windows

    # For active movements, calculate the total "energy" across all channels to find peaks
    # We use a simple moving average of the summed rectified signals
    summed_energy = np.sum(signal_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(summed_energy, kernel_size=1001) # 1-second smoothing
    
    # Find the peaks of these muscular bursts. 
    # distance=4000 ensures we don't double-count the same burst (4 seconds apart)
    peaks, _ = scipy.signal.find_peaks(smoothed_energy, distance=4000, prominence=np.max(smoothed_energy)*0.2)
    
    # Extract 1.5 seconds before and 1.5 seconds after each peak (3 seconds total)
    half_hold = int(1.5 * Config.FS)
    
    for peak in peaks:
        start_idx = peak - half_hold
        end_idx = peak + half_hold
        
        if start_idx < 0 or end_idx > num_samples:
            continue # Skip if burst is too close to the start/end of the file
            
        burst_data = signal_data[:, start_idx:end_idx]
        
        # Chop the 3-second hold into overlapping 500ms windows
        for step in range(0, burst_data.shape[1] - Config.WINDOW_SIZE, Config.INCREMENT):
            window = burst_data[:, step:step+Config.WINDOW_SIZE]
            
            # Ensure all windows are the same size
            if window.shape[1] == Config.WINDOW_SIZE:
                extracted_windows.append(window)
            
    return extracted_windows

def load_and_prepare_dataset(base_path='./secondary_data'):
    """
    Loops through the dataset, cleans the signals, extracts the sliding windows,
    and pairs them with the correct kinematic targets.
    """
    X_data = []
    y_targets = []
    
    print("Beginning automated data extraction and labelling...")
    
    for p in range(1, 9):       # Subjects 1 to 8
        for m in range(1, 10):  # Movements 1 to 9
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
                sig = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=50.0)
                sig = SignalProcessing.bandpassFilter(sig, fs=Config.FS, lowCut=30.0, highCut=450.0)
                clean_data[c, :] = np.abs(sig) # Rectify
            
            # 2. Automatically find bursts and extract overlapping 500ms windows
            windows = detect_bursts_and_extract(clean_data, movement_class=m)
            target_vector = Config.TARGET_MAPPING[m]
            
            # 3. Append to our dataset
            for w in windows:
                X_data.append(w)
                y_targets.append(target_vector)
                
        print(f"Processed Subject {p}...")

    X_data = np.array(X_data)
    y_targets = np.array(y_targets)
    
    print(f"Dataset generated! Total Windows: {X_data.shape[0]}")
    print(f"X shape: {X_data.shape} | y shape: {y_targets.shape}")
    
    return X_data, y_targets

if __name__ == "__main__":
    # Test the extraction
    X, y = load_and_prepare_dataset()