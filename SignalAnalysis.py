import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal
from matplotlib.widgets import CheckButtons, Button

import SignalProcessing
import ControllerConfiguration as Config

# Sampling rate from Rivela et al. (1.0 kHz)
FS = 1000.0 

CHANNEL_MAP = {
    0: "Pectoralis Major (Clavicular)",
    1: "Pectoralis Major (Sternal)",
    2: "Serratus Anterior",
    3: "Trapezius (Descendent)",
    4: "Trapezius (Transversalis)",
    5: "Trapezius (Ascendant)",
    6: "Infraspinatus",
    7: "Latissimus Dorsi"
}

# ====================================================================================
# ============================== BATCH IMAGE GENERATION ==============================
# ====================================================================================

def generate_all_signal_images(base_data_path='./secondary_data', save_path='./signal_plots'):
    """
    Loops through all subjects and movements, applies filters matching DataPreparation pipeline,
    and saves a PNG with the filtered signal + TKEO envelope overlay.
    
    Processing matches DataPreparation.py:
    - Notch filter (50 Hz powerline)
    - Bandpass filter (30-450 Hz)
    - Full-wave rectification
    - TKEO envelope with lowpass smoothing
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print(f"Starting batch generation of PNGs in '{save_path}'...")
    print(f"Using filtering parameters from Config:")
    print(f"  Notch: {Config.NOTCH_FREQ} Hz")
    print(f"  Bandpass: {Config.BANDPASS_LOW}-{Config.BANDPASS_HIGH} Hz")
    print(f"  TKEO lowpass cutoff: 5.0 Hz\n")
    
    # 8 Subjects, 9 Movements
    for p in range(1, 9):
        for m in range(1, 10):
            file_name = os.path.join(base_data_path, f'Soggetto{p}', f'Movimento{m}.mat')
            
            # Skip if the file is missing
            if not os.path.exists(file_name):
                print(f"Skipping missing file: {file_name}")
                continue
                
            try:
                # Load the data
                mat_contents = scipy.io.loadmat(file_name)
                if 'EMGDATA' not in mat_contents:
                    print(f"No EMGDATA in {file_name}")
                    continue
                    
                raw_data = mat_contents['EMGDATA']
                num_channels, num_samples = raw_data.shape
                time_axis = np.arange(num_samples) / FS
                
                # Plot each channel
                for c in range(num_channels):
                    raw_signal = raw_data[c, :]
                    
                    # === FILTERING PIPELINE (matches DataPreparation.py) ===
                    # Step 1: Notch filter (powerline noise)
                    notch_filtered = SignalProcessing.notchFilter(
                        raw_signal, fs=FS, 
                        notchFreq=Config.NOTCH_FREQ, 
                        qualityFactor=Config.NOTCH_QUALITY
                    )
                    
                    # Step 2: Bandpass filter (movement artifacts + HF noise)
                    band_filtered = SignalProcessing.bandpassFilter(
                        notch_filtered, fs=FS,
                        lowCut=Config.BANDPASS_LOW,
                        highCut=Config.BANDPASS_HIGH,
                        order=Config.FILTER_ORDER
                    )
                    
                    # Step 3: Full-wave rectification
                    rectified_signal = np.abs(band_filtered)
                    
                    # === TKEO ENVELOPE CALCULATION ===
                    # Calculate TKEO (Teager-Kaiser Energy Operator)
                    tkeo_signal = SignalProcessing.tkeo(band_filtered)
                    rectified_tkeo = np.abs(tkeo_signal)
                    
                    # Lowpass smooth the TKEO envelope
                    tkeo_envelope = SignalProcessing.lowpassFilter(
                        rectified_tkeo, fs=FS, cutoff=5.0, order=4
                    )
                    
                    # Normalize TKEO to match rectified signal scale for visualization
                    tkeo_max = np.percentile(tkeo_envelope, 99.9) + 1e-6
                    tkeo_normalized = (tkeo_envelope / tkeo_max) * np.max(rectified_signal)
                    
                    # Setup figure without buttons for max space
                    fig, (ax_raw, ax_filt) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)
                    ch_name = CHANNEL_MAP.get(c, f"Channel {c}")
                    fig.suptitle(f"Participant {p} | Movement {m} | {ch_name}", fontsize=16, fontweight='bold')
                    
                    # Top Plot: Raw Signal
                    ax_raw.plot(time_axis, raw_signal, color='tab:blue', linewidth=0.5)
                    ax_raw.set_title("Raw sEMG Signal")
                    ax_raw.set_ylabel("Amplitude (mV)")
                    ax_raw.grid(True, alpha=0.3)
                    
                    # Bottom Plot: Filtered Signal + TKEO Envelope Overlay
                    ax_filt.plot(time_axis, rectified_signal, color='tab:orange', linewidth=0.8, label='Filtered & Rectified')
                    ax_filt.plot(time_axis, tkeo_normalized, color='red', linewidth=1.5, label='TKEO Envelope', alpha=0.8)
                    ax_filt.set_title("Processed Signal (Notch + Bandpass + Rectified) with TKEO Envelope")
                    ax_filt.set_xlabel("Time (seconds)")
                    ax_filt.set_ylabel("Amplitude")
                    ax_filt.grid(True, alpha=0.3)
                    ax_filt.legend(loc='upper right', fontsize=10)
                    
                    plt.tight_layout()
                    
                    # Save and Close
                    save_name = f"p{p}_m{m}_channel{c}.png"
                    full_save_path = os.path.join(save_path, save_name)
                    plt.savefig(full_save_path, dpi=150)
                    plt.close(fig) # CRITICAL: Frees up memory so the script doesn't crash
                    
            except Exception as e:
                print(f"Error processing Participant {p}, Movement {m}: {e}")
                
    print("Batch generation complete! All images saved.")

# ====================================================================================
# ============================== EXECUTION ===========================================
# ====================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SignalAnalysis: Batch Signal Processing & Visualization")
    print("=" * 80)
    generate_all_signal_images(base_data_path=Config.BASE_DATA_PATH, save_path='./signal_plots')
    print("\nSignal plots are ready for use by ImageGridGenerator.py")