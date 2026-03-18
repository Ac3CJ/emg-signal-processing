import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np

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
# ============================== BURST DETECTION LOGIC ===============================
# ====================================================================================

# def get_burst_time_windows(signal_data, fs=1000.0):
#     """
#     Mimics the exact extraction logic from DataPreparation.py.
#     Returns a list of (start_time, end_time) tuples in seconds.
#     """
#     # Sum across all 8 channels
#     summed_energy = np.sum(signal_data, axis=0)
#     # Smooth with heavy median filter
#     smoothed_energy = scipy.signal.medfilt(summed_energy, kernel_size=1001)
#     # Find peaks
#     peaks, _ = scipy.signal.find_peaks(smoothed_energy, distance=4000, prominence=np.max(smoothed_energy)*0.2)
    
#     burst_windows = []
#     half_hold = int(2.0 * fs) # 2.0 seconds
#     num_samples = signal_data.shape[1]
    
#     for peak in peaks:
#         start_idx = max(0, peak - half_hold)
#         end_idx = min(num_samples, peak + half_hold)
#         # Convert indices to seconds for the plot
#         burst_windows.append((start_idx / fs, end_idx / fs))
        
#     return burst_windows

def get_burst_time_windows(signal_data, fs=1000.0, window_length_sec=4.5):
    """
    SciPy Peak-Width Onset Detection.
    Finds peaks, uses SciPy to trace down to the exact rising edge, 
    and projects a fixed rigid window forward.
    """
    
    # 1. Root Sum Square for global energy
    global_energy = np.sqrt(np.sum(signal_data ** 2, axis=0))
    # global_energy = np.sum(signal_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(global_energy, kernel_size=1001)
    
    # 2. Find peaks (Standard logic)
    peaks, _ = scipy.signal.find_peaks(
        smoothed_energy, 
        distance=4000, 
        prominence=np.max(smoothed_energy) * 0.10
    )
    
    # FIX: Ignore startup spikes by just throwing out peaks found in the first 1.5 seconds,
    # rather than creating artificial cliffs by zeroing the data!
    valid_peaks = [p for p in peaks if p > int(1.5 * fs)]
    
    burst_windows = []
    
    if len(valid_peaks) > 0:
        # 3. Use SciPy to find the exact rising edge!
        # rel_height=0.90 tells it to trace 90% of the way down the left slope
        widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
            smoothed_energy, valid_peaks, rel_height=0.90
        )
        
        fixed_window_samples = int(window_length_sec * fs)
        buffer_samples = int(0.2 * fs) # Pull back 200ms to grab the baseline just before the flex
        
        for i in range(len(valid_peaks)):
            # left_ips[i] holds the exact index of the rising edge for this peak
            rising_edge_idx = int(left_ips[i])
            
            # Anchor to the rising edge, pull back slightly, and rigidly project forward
            start_idx = max(0, rising_edge_idx - buffer_samples)
            end_idx = min(signal_data.shape[1], start_idx + fixed_window_samples)
            
            burst_windows.append((start_idx / fs, end_idx / fs))
            
    return burst_windows

# ====================================================================================
# ============================== INTERACTIVE PLOTTER =================================
# ====================================================================================

class EMGInteractivePlotter:
    def __init__(self, raw_data, fs, channel_map):
        self.raw_data = raw_data
        self.fs = fs
        self.channel_map = channel_map
        self.num_channels, self.num_samples = raw_data.shape
        self.time_axis = np.arange(self.num_samples) / self.fs
        self.current_idx = 0
        
        print("Pre-computing Filter Pipelines & Burst Zones... Please wait.")
        self.classic_data = np.zeros_like(self.raw_data)
        self.tkeo_data = np.zeros_like(self.raw_data)

        # Pre-process all channels with both pipelines
        for c in range(self.num_channels):
            # Base Conditioning
            notch = SignalProcessing.notchFilter(self.raw_data[c, :], fs=self.fs, notchFreq=Config.NOTCH_FREQ)
            band = SignalProcessing.bandpassFilter(notch, fs=self.fs, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)

            # PIPELINE 1: Classic (Rectify only)
            rectified_classic = np.abs(band)
            
            # PIPELINE 2: TKEO Envelope (Root-TKEO -> Rectify -> 5Hz Low-Pass)
            teager = SignalProcessing.tkeo(band)
            rectified_teager = np.abs(teager)
            envelope = SignalProcessing.lowpassFilter(rectified_teager, fs=self.fs, cutoff=5.0)

            # Robust Percentile Normalization (99.9th percentile)
            classic_max = np.percentile(rectified_classic, 99.9) + 1e-6
            tkeo_max = np.percentile(envelope, 99.9) + 1e-6

            self.classic_data[c, :] = np.clip(rectified_classic / classic_max, 0.0, 1.0)
            self.tkeo_data[c, :] = np.clip(envelope / tkeo_max, 0.0, 1.0)

        # Find Burst Zones mathematically
        self.classic_bursts = get_burst_time_windows(self.classic_data, self.fs)
        self.tkeo_bursts = get_burst_time_windows(self.tkeo_data, self.fs)

        # --- SETUP FIGURE ---
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        self.fig.canvas.manager.set_window_title('Preprocessing Pipeline Comparison')
        plt.subplots_adjust(bottom=0.25, hspace=0.3)

        # Plot Initial Lines
        self.line_raw, = self.ax1.plot(self.time_axis, self.raw_data[0, :], color='tab:gray', linewidth=1)
        self.line_classic, = self.ax2.plot(self.time_axis, self.classic_data[0, :], color='tab:orange', label='Classic Signal', alpha=0.8)
        self.line_tkeo, = self.ax2.plot(self.time_axis, self.tkeo_data[0, :], color='tab:blue', label='TKEO Envelope', linewidth=2)

        # Draw Background Burst Highlights
        self.classic_spans = []
        for start_t, end_t in self.classic_bursts:
            span = self.ax2.axvspan(start_t, end_t, color='lightcoral', alpha=0.25, lw=0)
            self.classic_spans.append(span)

        self.tkeo_spans = []
        for start_t, end_t in self.tkeo_bursts:
            span = self.ax2.axvspan(start_t, end_t, color='lightblue', alpha=0.35, lw=0)
            self.tkeo_spans.append(span)

        self.ax1.set_ylabel("Amplitude (mV)")
        self.ax1.grid(True, linestyle=':', alpha=0.6)
        
        self.ax2.set_ylabel("Normalized Power")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.grid(True, linestyle='--', alpha=0.6)
        self.ax2.legend(loc='upper right')

        # --- GUI BUTTONS ---
        axprev = plt.axes([0.35, 0.05, 0.1, 0.05])
        self.bprev = Button(axprev, 'Previous Ch')
        self.bprev.on_clicked(self.prev_ch)

        axnext = plt.axes([0.55, 0.05, 0.1, 0.05])
        self.bnext = Button(axnext, 'Next Ch')
        self.bnext.on_clicked(self.next_ch)

        # --- CHECKBOXES FOR VISIBILITY ---
        ax_check = plt.axes([0.75, 0.02, 0.2, 0.1])
        self.check = CheckButtons(ax_check, ['Classic Pipeline (Red Zones)', 'TKEO Pipeline (Blue Zones)'], [True, True])
        self.check.on_clicked(self.toggle_visibility)
        
        self.check.labels[0].set_color('tab:orange')
        self.check.labels[0].set_fontweight('bold')
        self.check.labels[1].set_color('tab:blue')
        self.check.labels[1].set_fontweight('bold')

        self.draw_plot()
        plt.show()

    def toggle_visibility(self, label):
        """Toggles the visibility of the lines AND their corresponding highlight zones."""
        if 'Classic' in label:
            vis = not self.line_classic.get_visible()
            self.line_classic.set_visible(vis)
            for span in self.classic_spans: span.set_visible(vis)
        elif 'TKEO' in label:
            vis = not self.line_tkeo.get_visible()
            self.line_tkeo.set_visible(vis)
            for span in self.tkeo_spans: span.set_visible(vis)
            
        self.fig.canvas.draw_idle()

    def draw_plot(self):
        muscle_name = self.channel_map.get(self.current_idx, f"Channel {self.current_idx}")
        self.fig.suptitle(f"Channel {self.current_idx}: {muscle_name}", fontsize=16, fontweight='bold')
        
        self.line_raw.set_ydata(self.raw_data[self.current_idx, :])
        self.line_classic.set_ydata(self.classic_data[self.current_idx, :])
        self.line_tkeo.set_ydata(self.tkeo_data[self.current_idx, :])
        
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw_idle()

    def next_ch(self, event):
        self.current_idx = (self.current_idx + 1) % self.num_channels
        self.draw_plot()

    def prev_ch(self, event):
        self.current_idx = (self.current_idx - 1) % self.num_channels
        self.draw_plot()

# ====================================================================================
# ============================== DATA LOADING ========================================
# ====================================================================================

def get_mat_headers(file_path):
    try:
        mat_contents = scipy.io.loadmat(file_path)
        headers = [key for key in mat_contents.keys() if not key.startswith('__')]
        print(f"Successfully loaded: {file_path}")
        return mat_contents, headers
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

if __name__ == "__main__":
    file_location = './secondary_data/Soggetto3/'
    file_name = file_location + 'Movimento1.mat' 
    contents, keys = get_mat_headers(file_name)

    if contents and 'EMGDATA' in contents:
        raw_emg_data = contents['EMGDATA']
        plotter = EMGInteractivePlotter(raw_emg_data, FS, CHANNEL_MAP)