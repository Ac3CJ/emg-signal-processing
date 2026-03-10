import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import SignalProcessing

# Sampling rate from Rivela et al. (1.0 kHz) [cite: 82]
FS = 1000.0 

# Verified Channel Map from Table I in the research paper [cite: 81]
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
        
        # Pre-compute filtered data so switching pages is instant
        print("Applying filters to all channels, please wait...")
        self.filtered_data = np.zeros_like(self.raw_data)
        for i in range(self.num_channels):
            self.filtered_data[i, :] = SignalProcessing.applyStandardSEMGProcessing(self.raw_data[i, :], fs=self.fs)
        
        # Setup Figure and Axes
        self.fig, (self.ax_raw, self.ax_filt) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
        plt.subplots_adjust(bottom=0.2, hspace=0.3) # Leave room for buttons at the bottom
        
        # Setup Buttons
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_ch)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_ch)
        
        # Draw the initial plot
        self.draw_plot()
        plt.show()

    def draw_plot(self):
        """Clears axes and redraws the selected channel."""
        self.ax_raw.clear()
        self.ax_filt.clear()
        
        ch_name = self.channel_map.get(self.current_idx, f"Channel {self.current_idx+1}")
        self.fig.suptitle(f"Channel {self.current_idx}: {ch_name}", fontsize=16, fontweight='bold')
        
        # Top Plot: Raw Signal
        self.ax_raw.plot(self.time_axis, self.raw_data[self.current_idx, :], color='tab:blue', linewidth=0.5)
        self.ax_raw.set_title(f"Raw sEMG Signal")
        self.ax_raw.set_ylabel("Amplitude")
        self.ax_raw.grid(True, alpha=0.3)
        
        # Bottom Plot: Filtered Signal
        self.ax_filt.plot(self.time_axis, self.filtered_data[self.current_idx, :], color='tab:orange', linewidth=0.5)
        self.ax_filt.set_title(f"Processed Signal (Notch + Bandpass + Rectified)")
        self.ax_filt.set_xlabel("Time (seconds)")
        self.ax_filt.set_ylabel("Amplitude")
        self.ax_filt.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
        
    def next_ch(self, event):
        """Cycles to the next channel."""
        self.current_idx = (self.current_idx + 1) % self.num_channels
        self.draw_plot()

    def prev_ch(self, event):
        """Cycles to the previous channel."""
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

# Usage
if __name__ == "__main__":
    file_location = './secondary_data/Soggetto1/'
    file_name = file_location + 'Movimento1.mat' 
    contents, keys = get_mat_headers(file_name)

    if contents and 'EMGDATA' in contents:
        raw_emg_data = contents['EMGDATA']
        # Launch the interactive plotter
        plotter = EMGInteractivePlotter(raw_emg_data, FS, CHANNEL_MAP)
    else:
        print("Could not find 'EMGDATA' in the provided file.")