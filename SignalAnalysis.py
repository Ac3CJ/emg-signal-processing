import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal
from matplotlib.widgets import CheckButtons, Button
import argparse
from PIL import Image

import SignalProcessing
import ControllerConfiguration as Config
from FileRepository import DataRepository

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

def _normalise_participant_ids(participant_ids):
    if participant_ids is None:
        return None

    normalised = set()
    for participant_id in participant_ids:
        try:
            normalised.add(int(participant_id))
        except (TypeError, ValueError):
            continue

    return sorted(normalised)

# ====================================================================================
# ============================== BATCH IMAGE GENERATION ==============================
# ====================================================================================

def _generate_channel_image(raw_signal, participant_id, movement_id, channel_id, save_path, time_axis=None):
    """
    Helper function: Generates a single channel image with standardized format.
    
    Creates a 2-subplot figure (Raw + Processed with TKEO) for a single EMG channel.
    Output filename: p{p}_m{m}_channel{c}.png
    """
    if time_axis is None:
        num_samples = len(raw_signal)
        time_axis = np.arange(num_samples) / FS
    
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
    ch_name = CHANNEL_MAP.get(channel_id, f"Channel {channel_id}")
    fig.suptitle(f"Participant {participant_id} | Movement {movement_id} | {ch_name}", fontsize=16, fontweight='bold')
    
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
    save_name = f"p{participant_id}_m{movement_id}_channel{channel_id}.png"
    full_save_path = os.path.join(save_path, save_name)
    plt.savefig(full_save_path, dpi=150)
    plt.close(fig)  # CRITICAL: Frees up memory so the script doesn't crash


def generate_all_signal_images(base_data_path=None, save_path=None, data_structure='secondary', participant_ids=None):
    """
    Generates individual channel images for all participants and movements.
    Unified for both secondary (Soggetto{p} structure) and collected (PM{x} flat) data.
    
    Args:
        base_data_path (str): Path to dataset directory
        save_path (str): Path to save generated PNGs
        data_structure (str): 'secondary' (Soggetto nested) or 'collected' (flat PM)
        participant_ids (list[int] | None): Optional collected participant IDs to process.
    
    Processing matches DataPreparation.py:
    - Notch filter (50 Hz powerline)
    - Bandpass filter (30-450 Hz)
    - Full-wave rectification
    - TKEO envelope with lowpass smoothing
    """
    if base_data_path is None:
        base_data_path = Config.SECONDARY_DATA_PATH if data_structure == 'secondary' else Config.COLLECTED_DATA_PATH
    if save_path is None:
        save_path = './signal_plots/secondary' if data_structure == 'secondary' else './signal_plots/collected'
    
    # Convert to absolute paths for robustness
    base_data_path = os.path.abspath(base_data_path)
    save_path = os.path.abspath(save_path)
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print(f"[INFO] Data structure: {data_structure}")
    print(f"[INFO] Base data path: {base_data_path}")
    print(f"[INFO] Output path: {save_path}")
    print(f"Starting batch generation of PNGs in '{save_path}'...")
    print(f"Using filtering parameters from Config:")
    print(f"  Notch: {Config.NOTCH_FREQ} Hz")
    print(f"  Bandpass: {Config.BANDPASS_LOW}-{Config.BANDPASS_HIGH} Hz")
    print(f"  TKEO lowpass cutoff: 5.0 Hz\n")

    repository = DataRepository.from_standard_path(base_data_path)
    
    if data_structure == 'secondary':
        # Secondary: Nested Soggetto{p} structure, 8 participants × 9 movements
        if repository is not None:
            trial_selections = repository.iter_file_selections('secondary')
            for selection in trial_selections:
                file_name = repository.output_file_path(selection, create_dirs=False)

                if not os.path.exists(file_name):
                    print(f"[-] Skipping missing file: {file_name}")
                    continue

                try:
                    mat_contents = scipy.io.loadmat(file_name)
                    if 'EMGDATA' not in mat_contents:
                        print(f"[-] No EMGDATA in {file_name}")
                        continue

                    raw_data = mat_contents['EMGDATA']
                    num_channels, num_samples = raw_data.shape
                    time_axis = np.arange(num_samples) / FS

                    for c in range(num_channels):
                        raw_signal = raw_data[c, :]
                        _generate_channel_image(raw_signal, selection.participant, selection.movement, c, save_path, time_axis)

                except Exception as e:
                    print(f"[!] Error processing Participant {selection.participant}, Movement {selection.movement}: {e}")
        else:
            for p in range(1, 9):
                for m in range(1, 10):
                    file_name = os.path.join(base_data_path, f'Soggetto{p}', f'Movimento{m}_labelled.mat')

                    if not os.path.exists(file_name):
                        print(f"[-] Skipping missing file: {file_name}")
                        continue

                    try:
                        mat_contents = scipy.io.loadmat(file_name)
                        if 'EMGDATA' not in mat_contents:
                            print(f"[-] No EMGDATA in {file_name}")
                            continue

                        raw_data = mat_contents['EMGDATA']
                        num_channels, num_samples = raw_data.shape
                        time_axis = np.arange(num_samples) / FS

                        for c in range(num_channels):
                            raw_signal = raw_data[c, :]
                            _generate_channel_image(raw_signal, p, m, c, save_path, time_axis)

                    except Exception as e:
                        print(f"[!] Error processing Participant {p}, Movement {m}: {e}")
    
    elif data_structure == 'collected':
        # Collected: Flat PM{x} structure, variable participants × 9 movements
        if repository is None:
            repository = DataRepository()

        available_participants = repository.discover_participants('collected')

        participants = _normalise_participant_ids(participant_ids)

        if participants is None:
            participants = available_participants
            print(f"[INFO] Found collected participants: {participants}\n")
        else:
            print(f"[INFO] Collected participant filter enabled: {participants}")
            print(f"[INFO] Available collected participants: {available_participants}\n")

        skipped_trials = []

        if len(participants) == 0:
            print("[WARNING] No collected participants were available to process.")
            print("Batch generation complete! All images saved.")
            return

        for selection in repository.iter_file_selections('collected', participants):
            file_name = repository.output_file_path(selection, create_dirs=False)

            if not os.path.exists(file_name):
                skipped_trials.append((selection.participant, selection.movement, 'missing file'))
                continue

            try:
                mat_contents = scipy.io.loadmat(file_name)
                if 'EMGDATA' not in mat_contents:
                    skipped_trials.append((selection.participant, selection.movement, 'missing EMGDATA'))
                    continue

                raw_data = mat_contents['EMGDATA']
                num_channels, num_samples = raw_data.shape
                time_axis = np.arange(num_samples) / FS

                for c in range(num_channels):
                    raw_signal = raw_data[c, :]
                    _generate_channel_image(raw_signal, selection.participant, selection.movement, c, save_path, time_axis)

            except Exception as e:
                skipped_trials.append((selection.participant, selection.movement, f'corrupted or unreadable ({e})'))

        if skipped_trials:
            print("\n[SKIP LOG] Collected trials skipped:")
            for p, m, reason in skipped_trials:
                print(f"  - P{p}M{m}: {reason}")
    
    print("Batch generation complete! All images saved.")

# ====================================================================================
# ============================== MEDIAN ENSEMBLE GENERATION ==========================
# ====================================================================================

def generate_median_ensemble_plots(base_data_path=None, save_path=None):
    """
    Computes median waveforms and spectrograms across all participants.
    Generates two images per movement: standard time-series and spectrograms.
    """
    if base_data_path is None:
        base_data_path = Config.SECONDARY_DATA_PATH
    if save_path is None:
        save_path = './signal_plots/secondary/ensemble'
    
    # Convert to absolute paths for robustness
    base_data_path = os.path.abspath(base_data_path)
    save_path = os.path.abspath(save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print(f"[INFO] Base data path: {base_data_path}")
    print(f"[INFO] Output path: {save_path}")
    print(f"Starting median ensemble generation in '{save_path}'...")
    
    for m in range(1, 10):
        print(f"Processing Movement {m}...")
        movement_name = Config.MOVEMENT_NAMES.get(m, f"Movement {m}") if hasattr(Config, 'MOVEMENT_NAMES') else f"Movement {m}"
        
        all_raw_signals = {c: [] for c in range(8)} 
        all_filtered_signals = {c: [] for c in range(8)} 
        time_axis = None
        
        for p in range(1, 9):
            file_name = os.path.join(base_data_path, f'Soggetto{p}', f'Movimento{m}_labelled.mat')
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

# ====================================================================================
# ========================= GRID GENERATION (WAVEFORM STITCHING) =====================
# ====================================================================================

def generate_participant_waveform_grids(source_dir=None, save_dir=None, scale_factor=0.5):
    """
    Generates 8x9 participant waveform grids (8 channels rows, 9 movements columns) for each participant.
    
    Args:
        source_dir (str): Directory containing individual channel PNG files
        save_dir (str): Directory where participant grids will be saved
        scale_factor (float): Scale factor for final image size
    """
    if source_dir is None:
        source_dir = './signal_plots/secondary'
    if save_dir is None:
        save_dir = './waveform_grids/secondary/participant'
    
    # Convert to absolute paths for robustness
    source_dir = os.path.abspath(source_dir)
    save_dir = os.path.abspath(save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n[GRID] Source directory: {source_dir}")
    print(f"[GRID] Output directory: {save_dir}")
    print(f"[GRID] Generating participant waveform grids in '{save_dir}'...")
    
    # Grab one image to determine cell dimensions
    sample_img_path = os.path.join(source_dir, 'p1_m1_channel0.png')
    if not os.path.exists(sample_img_path):
        print(f"[GRID] Error: Could not find '{sample_img_path}'. Did you generate signal images first?")
        return
    
    with Image.open(sample_img_path) as img:
        orig_width, orig_height = img.size
    
    cell_width = int(orig_width * scale_factor)
    cell_height = int(orig_height * scale_factor)
    
    num_channels = 8
    num_movements = 9
    
    grid_width = num_movements * cell_width
    grid_height = num_channels * cell_height
    
    print(f"[GRID] Grid resolution: {grid_width}x{grid_height} pixels (8 channels × 9 movements)")
    
    # Generate 1 grid per participant (Participants 1 through 8)
    for p in range(1, 9):
        print(f"[GRID] Stitching Participant {p}...")
        
        canvas = Image.new('RGB', (grid_width, grid_height), 'white')
        
        for c in range(num_channels):  # Rows = Channels
            for m in range(1, num_movements + 1):  # Columns = Movements
                img_name = f"p{p}_m{m}_channel{c}.png"
                img_path = os.path.join(source_dir, img_name)
                
                if os.path.exists(img_path):
                    with Image.open(img_path) as cell_img:
                        if scale_factor != 1.0:
                            cell_img = cell_img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                        
                        x_offset = (m - 1) * cell_width
                        y_offset = c * cell_height
                        canvas.paste(cell_img, (x_offset, y_offset))
                else:
                    print(f"[GRID]   Warning: Missing {img_name}, leaving blank space")
        
        save_path = os.path.join(save_dir, f'Participant_{p}_Waveforms_Grid.png')
        canvas.save(save_path)
        print(f"[GRID] Saved: {save_path}")
    
    print("[GRID] Participant waveform grids complete!")


def generate_movement_waveform_grids(source_dir=None, save_dir=None, scale_factor=0.5):
    """
    Generates 8x8 movement waveform grids (8 channels rows, 8 participants columns) for each movement.
    This creates Movement_1_Grid.png, Movement_2_Grid.png, etc.
    
    Args:
        source_dir (str): Directory containing individual channel PNG files
        save_dir (str): Directory where movement grids will be saved
        scale_factor (float): Scale factor for final image size
    """
    if source_dir is None:
        source_dir = './signal_plots/secondary'
    if save_dir is None:
        save_dir = './waveform_grids/secondary/movement'
    
    # Convert to absolute paths for robustness
    source_dir = os.path.abspath(source_dir)
    save_dir = os.path.abspath(save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n[GRID] Source directory: {source_dir}")
    print(f"[GRID] Output directory: {save_dir}")
    print(f"[GRID] Generating movement waveform grids in '{save_dir}'...")
    
    # Grab one image to determine cell dimensions
    sample_img_path = os.path.join(source_dir, 'p1_m1_channel0.png')
    if not os.path.exists(sample_img_path):
        print(f"[GRID] Error: Could not find '{sample_img_path}'. Did you generate signal images first?")
        return
    
    with Image.open(sample_img_path) as img:
        orig_width, orig_height = img.size
    
    cell_width = int(orig_width * scale_factor)
    cell_height = int(orig_height * scale_factor)
    
    num_participants = 8
    num_channels = 8
    
    grid_width = num_participants * cell_width
    grid_height = num_channels * cell_height
    
    print(f"[GRID] Grid resolution: {grid_width}x{grid_height} pixels (8 channels × 8 participants)")
    
    # Generate 1 grid per movement (Movements 1 through 9)
    for m in range(1, 10):
        print(f"[GRID] Stitching Movement {m}...")
        
        canvas = Image.new('RGB', (grid_width, grid_height), 'white')
        
        for c in range(num_channels):  # Rows = Channels
            for p in range(1, num_participants + 1):  # Columns = Participants
                img_name = f"p{p}_m{m}_channel{c}.png"
                img_path = os.path.join(source_dir, img_name)
                
                if os.path.exists(img_path):
                    with Image.open(img_path) as cell_img:
                        if scale_factor != 1.0:
                            cell_img = cell_img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                        
                        x_offset = (p - 1) * cell_width
                        y_offset = c * cell_height
                        canvas.paste(cell_img, (x_offset, y_offset))
                else:
                    print(f"[GRID]   Warning: Missing {img_name}, leaving blank space")
        
        save_path = os.path.join(save_dir, f'Movement_{m:02d}_Waveforms_Grid.png')
        canvas.save(save_path)
        print(f"[GRID] Saved: {save_path}")
    
    print("[GRID] Movement waveform grids complete!")


def generate_all_images_and_grids(data_type='secondary', participant_ids=None):
    """
    Master function that generates all signal images and waveform grids in one workflow.
    
    Args:
        data_type (str): 'secondary', 'collected', or 'all'
        participant_ids (list[int] | None): Optional collected participant IDs to process.
    """
    data_configs = {
        'secondary': {
            'base_path': Config.SECONDARY_DATA_PATH,
            'signal_plots_dir': './signal_plots/secondary',
            'participant_grid_dir': './waveform_grids/secondary/participant',
            'movement_grid_dir': './waveform_grids/secondary/movement',
        },
        'collected': {
            'base_path': Config.COLLECTED_DATA_PATH,
            'signal_plots_dir': './signal_plots/collected',
            'participant_grid_dir': './waveform_grids/collected/participant',
            'movement_grid_dir': './waveform_grids/collected/movement',
        }
    }
    
    datasets_to_process = []
    if data_type == 'all':
        datasets_to_process = ['secondary', 'collected']
    elif data_type in data_configs:
        datasets_to_process = [data_type]
    
    for dataset in datasets_to_process:
        config = data_configs[dataset]
        
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()} dataset")
        print(f"{'='*80}")
        
        # Step 1: Generate individual signal waveforms
        print(f"\n[STEP 1] Generating individual signal waveforms...")
        generate_all_signal_images(
            base_data_path=config['base_path'],
            save_path=config['signal_plots_dir'],
            data_structure=dataset,
            participant_ids=participant_ids
        )
        
        # Step 2: Generate participant waveform grids (8x9)
        print(f"\n[STEP 2] Generating participant waveform grids (8 channels × 9 movements)...")
        generate_participant_waveform_grids(
            source_dir=config['signal_plots_dir'],
            save_dir=config['participant_grid_dir'],
            scale_factor=0.5
        )
        
        # Step 3: Generate movement waveform grids (8x8)
        print(f"\n[STEP 3] Generating movement waveform grids (8 channels × 8 participants)...")
        generate_movement_waveform_grids(
            source_dir=config['signal_plots_dir'],
            save_dir=config['movement_grid_dir'],
            scale_factor=0.5
        )
    
    print(f"\n{'='*80}")
    print("ALL IMAGE AND GRID GENERATION COMPLETE!")
    print(f"{'='*80}")

# ====================================================================================
# ============================== EXECUTION ===========================================
# ====================================================================================

if __name__ == "__main__":
    # Ensure we're running from the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(
        description="SignalAnalysis: Batch Signal Processing & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python SignalAnalysis.py --mode signal                    # Generate secondary data signal images
  python SignalAnalysis.py --mode participant               # Generate collected data signal images
    python SignalAnalysis.py --mode participant --pid 1 3 6  # Generate collected data for specific participants
  python SignalAnalysis.py --mode grids                     # Generate all grids (comprehensive workflow)
  python SignalAnalysis.py --mode grids --dataset secondary # Generate grids for secondary data only
  python SignalAnalysis.py --mode grids --dataset collected # Generate grids for collected data only
    python SignalAnalysis.py --mode grids --dataset collected --pid 1 3 6 # Generate collected grids using selected participants
  python SignalAnalysis.py --mode ensemble                  # Generate ensemble medians
  python SignalAnalysis.py --mode all                       # Run all modes (default)
        """
    )
    
    parser.add_argument(
        '--mode',
        nargs='+',
        choices=['signal', 'ensemble', 'participant', 'grids', 'all'],
        default=['all'],
        help='Select which analysis modes to run (default: all)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['secondary', 'collected', 'all'],
        default='all',
        help='Dataset to process for grids mode (default: all)'
    )

    parser.add_argument(
        '--pid',
        type=int,
        nargs='+',
        default=None,
        help='Collected participant IDs to process (e.g. --pid 1 3 6). If omitted, all detected participants are used.'
    )
    
    args = parser.parse_args()
    
    # Determine which modes to run
    modes_to_run = args.mode
    if 'all' in modes_to_run:
        modes_to_run = ['signal', 'ensemble', 'participant', 'grids']
    
    print("=" * 80)
    print("SignalAnalysis: Batch Signal Processing & Visualization")
    print("=" * 80)
    print(f"Running modes: {', '.join(modes_to_run)}")
    if 'grids' in modes_to_run:
        print(f"Dataset: {args.dataset}")
    print()
    
    if 'signal' in modes_to_run:
        print("Generating secondary data signal images...")
        generate_all_signal_images(base_data_path=Config.SECONDARY_DATA_PATH, save_path='./signal_plots/secondary', data_structure='secondary')
        print()
    
    if 'ensemble' in modes_to_run:
        print("Generating median ensemble plots...")
        generate_median_ensemble_plots(base_data_path=Config.SECONDARY_DATA_PATH, save_path='./signal_plots/secondary/ensemble')
        print()
    
    if 'participant' in modes_to_run:
        print("Generating collected data signal images...")
        generate_all_signal_images(base_data_path=Config.COLLECTED_DATA_PATH, save_path='./signal_plots/collected', data_structure='collected', participant_ids=args.pid)
        print()
    
    if 'grids' in modes_to_run:
        print("Generating comprehensive waveform grids (images + grids)...")
        generate_all_images_and_grids(data_type=args.dataset, participant_ids=args.pid)
        print()
    
    print("\n" + "=" * 80)
    print("SignalAnalysis: All requested visualizations complete!")
    print("=" * 80)