import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np
import os

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

def get_burst_time_windows(signal_data, fs=1000.0, window_length_sec=4.5):
    """
    Sum with Noise Gating, SciPy Rising-Edge Detection, 
    and optimized peak-distance for rushed movements.
    """
    # 2. Sum for global energy
    global_energy = np.sum(signal_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(global_energy, kernel_size=501)
    
    # --- NEW: EDGE CLAMPING TO REMOVE STARTUP SPIKE ---
    # Cut off the first 0.5s and stretch the steady-state value backwards to t=0
    cutoff_samples = int(0.5 * fs)
    steady_state_value = smoothed_energy[cutoff_samples]
    
    modified_smoothed_energy = np.copy(smoothed_energy)
    modified_smoothed_energy[:cutoff_samples] = steady_state_value
    robust_max = np.percentile(modified_smoothed_energy, 95)
    
    # 3. Find Peaks (using the clamped signal)
    peaks, _ = scipy.signal.find_peaks(
        modified_smoothed_energy, 
        distance=2000,
        prominence=robust_max*0.05,
        height=robust_max * 0.60
    )
    
    # Ignore startup hardware spikes
    valid_peaks = [p for p in peaks if p > int(1.5 * fs)]
    
    burst_windows = []
    
    if len(valid_peaks) > 0:
        # 4. Trace down 90% of the left slope to find the exact rising edge
        # (Using the clamped signal so it doesn't accidentally trace into the old spike)
        widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
            modified_smoothed_energy, valid_peaks, rel_height=0.90
        )
        
        fixed_window_samples = int(window_length_sec * fs)
        buffer_samples = int(0.2 * fs) # Pull back 200ms
        
        last_end_idx = 0

        for i in range(len(valid_peaks)):
            rising_edge_idx = int(left_ips[i])
            
            # Anchor to the rising edge, pull back slightly, and rigidly project forward
            start_idx = max(0, rising_edge_idx - buffer_samples)

            if start_idx < last_end_idx:
                continue

            end_idx = min(signal_data.shape[1], start_idx + fixed_window_samples)
            
            burst_windows.append((start_idx / fs, end_idx / fs))
            last_end_idx = end_idx
            
    # Return the modified curve so you can visually verify the flat plateau via the red line
    return burst_windows, modified_smoothed_energy

# ====================================================================================
# ============================== INTERACTIVE PLOTTER =================================
# ====================================================================================

class EMGInteractivePlotter:
    def __init__(self, raw_data, fs, channel_map, generate_burst_plots=False):
        self.raw_data = raw_data
        self.fs = fs
        self.channel_map = channel_map
        self.num_channels, self.num_samples = raw_data.shape
        self.time_axis = np.arange(self.num_samples) / self.fs
        self.current_idx = 0
        
        print("Pre-computing Filter Pipelines & Burst Zones... Please wait.")
        self.classic_data = np.zeros_like(self.raw_data)
        self.tkeo_data = np.zeros_like(self.raw_data)

        for c in range(self.num_channels):
            median = scipy.signal.medfilt(self.raw_data[c, :], kernel_size=11)
            notch = SignalProcessing.notchFilter(median, fs=self.fs, notchFreq=Config.NOTCH_FREQ)
            band = SignalProcessing.bandpassFilter(notch, fs=self.fs, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)

            rectified_classic = np.abs(band)
            
            teager = SignalProcessing.tkeo(band)
            rectified_teager = np.abs(teager)
            envelope = SignalProcessing.lowpassFilter(rectified_teager, fs=self.fs, cutoff=5.0)

            classic_max = np.percentile(rectified_classic, 99.9) + 1e-6
            tkeo_max = np.percentile(envelope, 99.9) + 1e-6

            self.classic_data[c, :] = np.clip(rectified_classic / classic_max, 0.0, 1.0)
            self.tkeo_data[c, :] = np.clip(envelope / tkeo_max, 0.0, 1.0)

        # Get windows AND the global curve
        self.classic_bursts, classic_global_curve = get_burst_time_windows(self.classic_data, self.fs)
        self.tkeo_bursts, tkeo_global_curve = get_burst_time_windows(self.tkeo_data, self.fs)

        # Normalize the global curves so they fit on the 0-1 graph
        self.norm_classic_global = classic_global_curve / (np.max(classic_global_curve) + 1e-6)
        self.norm_tkeo_global = tkeo_global_curve / (np.max(tkeo_global_curve) + 1e-6)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        self.fig.canvas.manager.set_window_title('Preprocessing Pipeline Comparison')
        plt.subplots_adjust(bottom=0.25, hspace=0.3)

        self.line_raw, = self.ax1.plot(self.time_axis, self.raw_data[0, :], color='tab:gray', linewidth=1)
        
        # Plot Pipeline Data
        self.line_classic, = self.ax2.plot(self.time_axis, self.classic_data[0, :], color='tab:orange', label='Classic Signal', alpha=0.8)
        self.line_tkeo, = self.ax2.plot(self.time_axis, self.tkeo_data[0, :], color='tab:blue', label='TKEO Envelope', linewidth=2)
        
        # Plot the Global Math Curves (Red)
        self.line_classic_math, = self.ax2.plot(self.time_axis, self.norm_classic_global, color='tab:red', linestyle='-', linewidth=2, alpha=0.3, label='Global Math Curve')
        self.line_tkeo_math, = self.ax2.plot(self.time_axis, self.norm_tkeo_global, color='tab:red', linestyle='-', linewidth=2, alpha=0.8)

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

        axprev = plt.axes([0.35, 0.05, 0.1, 0.05])
        self.bprev = Button(axprev, 'Previous Ch')
        self.bprev.on_clicked(self.prev_ch)

        axnext = plt.axes([0.55, 0.05, 0.1, 0.05])
        self.bnext = Button(axnext, 'Next Ch')
        self.bnext.on_clicked(self.next_ch)

        ax_check = plt.axes([0.75, 0.02, 0.2, 0.1])
        self.check = CheckButtons(ax_check, ['Classic Pipeline', 'TKEO Pipeline'], [True, True])
        self.check.on_clicked(self.toggle_visibility)
        
        self.check.labels[0].set_color('tab:orange')
        self.check.labels[0].set_fontweight('bold')
        self.check.labels[1].set_color('tab:blue')
        self.check.labels[1].set_fontweight('bold')

        self.draw_plot()
        plt.show()

    def toggle_visibility(self, label):
        if 'Classic' in label:
            vis = not self.line_classic.get_visible()
            self.line_classic.set_visible(vis)
            self.line_classic_math.set_visible(vis)
            for span in self.classic_spans: span.set_visible(vis)
        elif 'TKEO' in label:
            vis = not self.line_tkeo.get_visible()
            self.line_tkeo.set_visible(vis)
            self.line_tkeo_math.set_visible(vis)
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
# ============================== FULL DATASET BATCH PROCESSING =======================
# ====================================================================================

def batch_generate_all_plots(base_dir='./secondary_data/', output_dir='./burstwindow-plots/'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting FULL BATCH GENERATION. Images will be saved to: {output_dir}")

    for p in range(1, 9):
        for m in range(1, 9):
            file_path = os.path.join(base_dir, f'Soggetto{p}', f'Movimento{m}.mat')
            
            if not os.path.exists(file_path):
                continue
                
            try:
                mat_contents = scipy.io.loadmat(file_path)
                if 'EMGDATA' not in mat_contents:
                    continue
                raw_data = mat_contents['EMGDATA']
            except Exception as e:
                print(f"Skipping {file_path} (Read Error)")
                continue

            num_channels = raw_data.shape[0]
            time_axis = np.arange(raw_data.shape[1]) / FS
            tkeo_data = np.zeros_like(raw_data)

            for c in range(num_channels):
                notch = SignalProcessing.notchFilter(raw_data[c, :], fs=FS, notchFreq=Config.NOTCH_FREQ)
                band = SignalProcessing.bandpassFilter(notch, fs=FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
                teager = SignalProcessing.tkeo(band)
                rectified_teager = np.abs(teager)
                envelope = SignalProcessing.lowpassFilter(rectified_teager, fs=FS, cutoff=5.0)

                tkeo_max = np.percentile(envelope, 99.9) + 1e-6
                tkeo_data[c, :] = np.clip(envelope / tkeo_max, 0.0, 1.0)

            # Get bursts and the global curve used to find them
            tkeo_bursts, tkeo_global_curve = get_burst_time_windows(tkeo_data, FS)
            
            # Normalize global curve for plotting
            norm_global = tkeo_global_curve / (np.max(tkeo_global_curve) + 1e-6)

            fig, axes = plt.subplots(num_channels, 1, figsize=(16, 12), sharex=True)
            fig.suptitle(f"Burst Windows: Participant {p} - Movement {m}", fontsize=18, fontweight='bold')

            for c in range(num_channels):
                ax = axes[c]
                muscle_name = CHANNEL_MAP.get(c, f"Ch {c}")
                
                raw_max = np.max(np.abs(raw_data[c, :])) + 1e-6
                norm_raw = raw_data[c, :] / raw_max
                ax.plot(time_axis, norm_raw, color='tab:gray', linewidth=0.5, alpha=0.6, label='Raw Signal')
                
                # Plot TKEO
                ax.plot(time_axis, tkeo_data[c, :], color='tab:blue', linewidth=1.5, label='TKEO')
                
                # Plot Global Math Curve (Red overlay)
                ax.plot(time_axis, norm_global, color='tab:red', linestyle='-', linewidth=2, alpha=0.5, label='Math Curve')

                for start_t, end_t in tkeo_bursts:
                    ax.axvspan(start_t, end_t, color='lightblue', alpha=0.4, lw=0)

                ax.set_ylabel(f"Ch {c}", fontsize=10)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                ax.text(1.01, 0.5, muscle_name, transform=ax.transAxes, va='center', fontsize=10, fontweight='bold')

            axes[-1].set_xlabel("Time (seconds)", fontsize=12)
            plt.tight_layout(rect=[0, 0, 0.9, 0.97])

            filename = f"P{p}M{m}-burst-windows.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150)
            plt.close(fig)

    print("Batch Processing Complete!")

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
    RUN_BATCH_GENERATION = False

    if RUN_BATCH_GENERATION:
        batch_generate_all_plots()
    else:
        # file_location = './secondary_data/Soggetto5/'
        # file_name = file_location + 'Movimento5.mat' 
        file_location = './collected_data/'
        file_name = file_location + 'P1M4_edit.mat' 
        contents, keys = get_mat_headers(file_name)

        if contents and 'EMGDATA' in contents:
            raw_emg_data = contents['EMGDATA']
            plotter = EMGInteractivePlotter(raw_emg_data, FS, CHANNEL_MAP, generate_burst_plots=False)