import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal
from matplotlib.widgets import CheckButtons, Button
import argparse

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
# ============================== MEDIAN ENSEMBLE GENERATION ==========================
# ====================================================================================

def generate_median_ensemble_plots(base_data_path='./secondary_data', save_path='./signal_plots/ensemble'):
    """
    Computes median waveforms and spectrograms across all participants.
    Generates two images per movement: standard time-series and spectrograms.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print(f"Starting median ensemble generation in '{save_path}'...")
    
    for m in range(1, 10):
        print(f"Processing Movement {m}...")
        movement_name = Config.MOVEMENT_NAMES.get(m, f"Movement {m}") if hasattr(Config, 'MOVEMENT_NAMES') else f"Movement {m}"
        
        all_raw_signals = {c: [] for c in range(8)} 
        all_filtered_signals = {c: [] for c in range(8)} 
        time_axis = None
        
        for p in range(1, 9):
            file_name = os.path.join(base_data_path, f'Soggetto{p}', f'Movimento{m}.mat')
            if not os.path.exists(file_name):
                continue
                
            try:
                mat_contents = scipy.io.loadmat(file_name)
                if 'EMGDATA' not in mat_contents: continue
                    
                raw_data = mat_contents['EMGDATA']
                num_channels, num_samples = raw_data.shape
                
                if time_axis is None or len(time_axis) < num_samples:
                    time_axis = np.arange(num_samples) / Config.FS
                
                for c in range(num_channels):
                    raw_signal = raw_data[c, :]
                    notch_filtered = SignalProcessing.notchFilter(raw_signal, fs=Config.FS, notchFreq=Config.NOTCH_FREQ, qualityFactor=Config.NOTCH_QUALITY)
                    band_filtered = SignalProcessing.bandpassFilter(notch_filtered, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH, order=4)
                    rectified_signal = np.abs(band_filtered)
                    
                    all_raw_signals[c].append(raw_signal)
                    all_filtered_signals[c].append(rectified_signal)
                    
            except Exception as e:
                print(f"  Error processing P{p} M{m}: {e}")
        
        num_valid_participants = len(all_raw_signals[0])
        if num_valid_participants == 0:
            print(f"  No valid data found for Movement {m}. Skipping.\n")
            continue
        
        min_length = min([len(sig) for sig in all_raw_signals[0]])
        for c in range(8):
            all_raw_signals[c] = [sig[:min_length] for sig in all_raw_signals[c]]
            all_filtered_signals[c] = [sig[:min_length] for sig in all_filtered_signals[c]]
        time_axis = time_axis[:min_length]

        median_raw = {}
        median_filtered = {}
        for c in range(8):
            if len(all_raw_signals[c]) > 0:
                median_raw[c] = np.median(np.array(all_raw_signals[c]), axis=0)
                median_filtered[c] = np.median(np.array(all_filtered_signals[c]), axis=0)
        
        # ====================================================================
        # PLOT 1: STANDARD TIME-SERIES (Raw vs Filtered)
        # ====================================================================
        fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(16, 16))
        fig.suptitle(f"Movement {m}: {movement_name} (Median across {num_valid_participants} participants)", fontsize=18, fontweight='bold', y=0.995)
        
        for c in range(8):
            ch_name = Config.CHANNEL_MAP.get(c, f"Channel {c}") if hasattr(Config, 'CHANNEL_MAP') else f"Channel {c}"
            
            axes[c, 0].plot(time_axis, median_raw[c], color='tab:blue', linewidth=1)
            axes[c, 0].set_title(f"{ch_name} - Raw", fontsize=11, fontweight='bold')
            axes[c, 0].set_ylabel("Amplitude", fontsize=10)
            
            axes[c, 1].plot(time_axis, median_filtered[c], color='tab:orange', linewidth=1)
            axes[c, 1].set_title(f"{ch_name} - Filtered & Rectified", fontsize=11, fontweight='bold')
            axes[c, 1].set_ylabel("Amplitude", fontsize=10)
        
        axes[7, 0].set_xlabel("Time (seconds)", fontsize=10)
        axes[7, 1].set_xlabel("Time (seconds)", fontsize=10)
        plt.tight_layout()
        
        save_name = f"Movement_{m:02d}_{movement_name.replace(' ', '_').replace('/', '-')}_median_ensemble.png"
        plt.savefig(os.path.join(save_path, save_name), dpi=150, bbox_inches='tight')
        plt.close(fig) 

        # ====================================================================
        # PLOT 2: SPECTROGRAMS (Frequency vs Time)
        # ====================================================================
        fig_spec, axes_spec = plt.subplots(nrows=8, ncols=1, figsize=(14, 18))
        fig_spec.suptitle(f"Movement {m}: {movement_name} (Median Spectrogram)", fontsize=18, fontweight='bold', y=0.995)
        
        for c in range(8):
            ch_name = Config.CHANNEL_MAP.get(c, f"Channel {c}") if hasattr(Config, 'CHANNEL_MAP') else f"Channel {c}"
            ax = axes_spec[c]
            
            # Plot the spectrogram using the raw median signal
            # NFFT=256 gives a good 256ms frequency window. 'magma' highlights intensity beautifully.
            Pxx, freqs, bins, im = ax.specgram(median_raw[c], NFFT=256, Fs=Config.FS, noverlap=128, cmap='magma')
            
            ax.set_title(f"{ch_name} - Spectrogram", fontsize=11, fontweight='bold')
            ax.set_ylabel("Frequency (Hz)", fontsize=10)
            
        axes_spec[7].set_xlabel("Time (seconds)", fontsize=10)
        plt.tight_layout()
        
        spec_save_name = f"Movement_{m:02d}_{movement_name.replace(' ', '_').replace('/', '-')}_median_spectrogram.png"
        plt.savefig(os.path.join(save_path, spec_save_name), dpi=150, bbox_inches='tight')
        plt.close(fig_spec)
                
    print(f"Generation complete! Time-series and Spectrograms saved to {save_path}/")

def generate_participant_plots(participant_id, base_data_path=r'.\collected_data', save_path='./signal_plots/participant', use_edit_suffix=False):
    """
    Generates time-series and spectrogram plots for a single, specific participant.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print(f"Starting plot generation for Participant {participant_id} in '{save_path}'...")
    
    for m in range(1, 10):
        movement_name = Config.MOVEMENT_NAMES.get(m, f"Movement {m}") if hasattr(Config, 'MOVEMENT_NAMES') else f"Movement {m}"
        
        suffix = "_edit" if use_edit_suffix else ""
        file_name = os.path.join(base_data_path, f'P{participant_id}M{m}{suffix}.mat')
        
        if not os.path.exists(file_name):
            print(f"  [-] Skipping P{participant_id}M{m}: File not found at {file_name}")
            continue
            
        print(f"  [+] Processing P{participant_id}M{m}...")
        
        try:
            mat_contents = scipy.io.loadmat(file_name)
            if 'EMGDATA' not in mat_contents: continue
                
            raw_data = mat_contents['EMGDATA']
            num_channels, num_samples = raw_data.shape
            time_axis = np.arange(num_samples) / Config.FS
            
            plot_raw = {}
            plot_filtered = {}
            
            for c in range(num_channels):
                raw_signal = raw_data[c, :]
                notch_filtered = SignalProcessing.notchFilter(raw_signal, fs=Config.FS, notchFreq=Config.NOTCH_FREQ, qualityFactor=Config.NOTCH_QUALITY)
                band_filtered = SignalProcessing.bandpassFilter(notch_filtered, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH, order=4)
                rectified_signal = np.abs(band_filtered)
                
                plot_raw[c] = raw_signal
                plot_filtered[c] = rectified_signal
                
            # ====================================================================
            # PLOT 1: STANDARD TIME-SERIES
            # ====================================================================
            fig, axes = plt.subplots(nrows=num_channels, ncols=2, figsize=(16, 16))
            fig.suptitle(f"Participant {participant_id} - Movement {m}: {movement_name}", fontsize=18, fontweight='bold', y=0.995)
            
            for c in range(num_channels):
                ch_name = Config.CHANNEL_MAP.get(c, f"Channel {c}") if hasattr(Config, 'CHANNEL_MAP') else f"Channel {c}"
                
                axes[c, 0].plot(time_axis, plot_raw[c], color='tab:blue', linewidth=1)
                axes[c, 0].set_title(f"{ch_name} - Raw", fontsize=11, fontweight='bold')
                axes[c, 0].set_ylabel("Amplitude", fontsize=10)
                
                axes[c, 1].plot(time_axis, plot_filtered[c], color='tab:orange', linewidth=1)
                axes[c, 1].set_title(f"{ch_name} - Filtered & Rectified", fontsize=11, fontweight='bold')
                axes[c, 1].set_ylabel("Amplitude", fontsize=10)
            
            axes[-1, 0].set_xlabel("Time (seconds)", fontsize=10)
            axes[-1, 1].set_xlabel("Time (seconds)", fontsize=10)
            plt.tight_layout()
            
            save_name = f"P{participant_id}_M{m:02d}_{movement_name.replace(' ', '_').replace('/', '-')}.png"
            plt.savefig(os.path.join(save_path, save_name), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # ====================================================================
            # PLOT 2: SPECTROGRAMS
            # ====================================================================
            fig_spec, axes_spec = plt.subplots(nrows=num_channels, ncols=1, figsize=(14, 18))
            fig_spec.suptitle(f"Participant {participant_id} - Movement {m}: {movement_name} (Spectrogram)", fontsize=18, fontweight='bold', y=0.995)
            
            for c in range(num_channels):
                ch_name = Config.CHANNEL_MAP.get(c, f"Channel {c}") if hasattr(Config, 'CHANNEL_MAP') else f"Channel {c}"
                ax = axes_spec[c]
                
                Pxx, freqs, bins, im = ax.specgram(plot_raw[c], NFFT=256, Fs=Config.FS, noverlap=128, cmap='magma')
                
                ax.set_title(f"{ch_name} - Spectrogram", fontsize=11, fontweight='bold')
                ax.set_ylabel("Frequency (Hz)", fontsize=10)
                
            axes_spec[-1].set_xlabel("Time (seconds)", fontsize=10)
            plt.tight_layout()
            
            spec_save_name = f"P{participant_id}_M{m:02d}_{movement_name.replace(' ', '_').replace('/', '-')}_spectrogram.png"
            plt.savefig(os.path.join(save_path, spec_save_name), dpi=150, bbox_inches='tight')
            plt.close(fig_spec)
                
        except Exception as e:
            print(f"  [!] Error processing P{participant_id} M{m}: {e}")
            
    print(f"\nPlot generation complete for Participant {participant_id}! Images saved to {save_path}/")

# ====================================================================================
# ============================== EXECUTION ===========================================
# ====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SignalAnalysis: Batch Signal Processing & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python SignalAnalysis.py --mode signal                    # Generate all signal images
  python SignalAnalysis.py --mode ensemble                  # Generate ensemble medians
  python SignalAnalysis.py --mode participant --pid 3       # Generate participant 3 plots
  python SignalAnalysis.py --mode all                       # Run all modes (default)
  python SignalAnalysis.py --mode signal ensemble --pid 5   # Run signal & ensemble, with participant 5 data available
        """
    )
    
    parser.add_argument(
        '--mode',
        nargs='+',
        choices=['signal', 'ensemble', 'participant', 'all'],
        default=['all'],
        help='Select which analysis modes to run (default: all)'
    )
    
    parser.add_argument(
        '--pid',
        type=int,
        default=1,
        metavar='ID',
        help='Participant ID for participant mode (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Determine which modes to run
    modes_to_run = args.mode
    if 'all' in modes_to_run:
        modes_to_run = ['signal', 'ensemble', 'participant']
    
    print("=" * 80)
    print("SignalAnalysis: Batch Signal Processing & Visualization")
    print("=" * 80)
    print(f"Running modes: {', '.join(modes_to_run)}")
    if 'participant' in modes_to_run:
        print(f"Participant ID: {args.pid}")
    print()
    
    if 'signal' in modes_to_run:
        print("Generating all signal images...")
        generate_all_signal_images(base_data_path=Config.BASE_DATA_PATH, save_path='./signal_plots')
        print()
    
    if 'ensemble' in modes_to_run:
        print("Generating median ensemble plots...")
        generate_median_ensemble_plots(base_data_path=Config.BASE_DATA_PATH, save_path='./signal_plots/ensemble')
        print()
    
    if 'participant' in modes_to_run:
        print(f"Generating participant {args.pid} plots...")
        generate_participant_plots(participant_id=args.pid, base_data_path='./collected_data/edit/', save_path='./signal_plots/participant', use_edit_suffix=True)
        print()
    
    print("\nSignal plots are ready for use by ImageGridGenerator.py")