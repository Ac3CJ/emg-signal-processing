import argparse
import os
import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

import DataPreparation
import SignalProcessing
import ControllerConfiguration as Config

# ====================================================================================
# ============================== MODE 1: WINDOW VIEWER ===============================
# ====================================================================================
class WindowViewer:
    def __init__(self, movement_id, windows, cols=5):
        """Creates an interactive Matplotlib viewer to page through signal windows."""
        self.movement_id = movement_id
        self.windows = windows  
        self.cols = cols
        self.page = 0
        self.total_pages = int(np.ceil(len(windows) / cols))
        
        self.fig, self.axes = plt.subplots(Config.NUM_CHANNELS, cols, figsize=(16, 9), sharex=True, sharey='row')
        self.fig.canvas.manager.set_window_title(f'Validation - Movement {movement_id}')
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.3, wspace=0.1)
        
        axprev = plt.axes([0.35, 0.05, 0.1, 0.05])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_page)
        
        axnext = plt.axes([0.55, 0.05, 0.1, 0.05])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_page)
        
        self.draw_page()
        plt.show() 

    def draw_page(self):
        self.fig.suptitle(f'Movement {self.movement_id} | Page {self.page + 1} of {self.total_pages}\nClose window to proceed to next movement.', fontsize=14, fontweight='bold')
        start_idx = self.page * self.cols
        
        for col in range(self.cols):
            window_idx = start_idx + col
            for row in range(Config.NUM_CHANNELS):
                ax = self.axes[row, col]
                ax.clear()
                
                if window_idx < len(self.windows):
                    signal = self.windows[window_idx, row, :]
                    ax.plot(signal, color='tab:blue', linewidth=1)
                    
                    if row == 0: ax.set_title(f"Window {window_idx + 1}", fontsize=10)
                    if col == 0:
                        muscle_name = Config.CHANNEL_MAP.get(row, f"Ch {row}").split('(')[0].strip()
                        ax.set_ylabel(muscle_name, rotation=90, size=8)
                        
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.set_xticks([]) 
                else:
                    ax.axis('off')

        self.fig.canvas.draw_idle()

    def next_page(self, event):
        if self.page < self.total_pages - 1:
            self.page += 1
            self.draw_page()

    def prev_page(self, event):
        if self.page > 0:
            self.page -= 1
            self.draw_page()

# ====================================================================================
# ============================== MODE 2: OVERLAY VIEWER ==============================
# ====================================================================================
class ParticipantOverlayViewer:
    def __init__(self, participant_id, movement_data_dict):
        """Overlays two consecutive bursts (and the rest between) of all 8 movements."""
        self.fig, self.axes = plt.subplots(Config.NUM_CHANNELS, 1, figsize=(14, 9), sharex=True, sharey='row')
        self.fig.canvas.manager.set_window_title(f'Overlay Validation - Subject {participant_id}')
        
        # Make room for the checkboxes on the bottom right
        plt.subplots_adjust(bottom=0.1, right=0.85, top=0.95, hspace=0.1)
        self.fig.suptitle(f'Cross-Talk Investigation (2 Bursts): Subject {participant_id}', fontsize=16, fontweight='bold')

        self.lines_by_movement = {} 
        # Distinct colors for M1 through M8
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        self.mov_labels = [f"M{m}" for m in range(1, 9)]

        # Setup Y-Axis labels
        for ch in range(Config.NUM_CHANNELS):
            ax = self.axes[ch]
            muscle_name = Config.CHANNEL_MAP.get(ch, f"Ch {ch}")
            ax.set_ylabel(muscle_name, rotation=0, labelpad=60, ha='center', size=8, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            if ch < Config.NUM_CHANNELS - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')

        # Plot the data
        for idx, m in enumerate(range(1, 9)):
            self.lines_by_movement[m] = []
            if m in movement_data_dict and movement_data_dict[m] is not None:
                data = movement_data_dict[m] # Shape (8, dynamic_length)
                
                # Create a proper time axis in seconds so varying lengths align perfectly at t=0
                time_axis = np.arange(data.shape[1]) / Config.FS
                
                for ch in range(Config.NUM_CHANNELS):
                    line, = self.axes[ch].plot(time_axis, data[ch], color=colors[idx], label=f"M{m}", alpha=0.85, linewidth=1.5)
                    self.lines_by_movement[m].append(line)

        # Create the CheckButtons at the bottom right
        ax_check = plt.axes([0.87, 0.1, 0.1, 0.3])
        self.check = CheckButtons(ax_check, self.mov_labels, [True]*8)
        
        # Match checkbox text colors to the lines for readability
        for idx, label in enumerate(self.check.labels):
            label.set_color(colors[idx])
            label.set_fontweight('bold')

        self.check.on_clicked(self.toggle_visibility)
        plt.show()

    def toggle_visibility(self, label):
        """Toggles the visibility of all lines associated with a movement."""
        m_id = int(label.replace('M', ''))
        if m_id in self.lines_by_movement:
            for line in self.lines_by_movement[m_id]:
                line.set_visible(not line.get_visible())
        self.fig.canvas.draw_idle()

def extract_representative_burst(p, m):
    """Hunts down the first TWO bursts of a specific trial, including the rest period between them."""
    file_path = os.path.join(Config.BASE_DATA_PATH, f'Soggetto{p}', f'Movimento{m}.mat')
    if not os.path.exists(file_path): return None
    if (p, m) in Config.CORRUPTED_TRIALS: return None
    
    mat = scipy.io.loadmat(file_path)
    raw_data = mat['EMGDATA']
    
    clean_data = np.zeros_like(raw_data)
    for c in range(Config.NUM_CHANNELS):
        sig = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        sig = SignalProcessing.bandpassFilter(sig, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
        clean_data[c, :] = np.abs(sig)
        
    summed_energy = np.sum(clean_data, axis=0)
    smoothed_energy = scipy.signal.medfilt(summed_energy, kernel_size=1001)
    peaks, _ = scipy.signal.find_peaks(smoothed_energy, distance=4000, prominence=np.max(smoothed_energy)*0.2)
    
    if len(peaks) == 0: return None
    
    # Grab 1.5s before the FIRST peak, and 1.5s after the SECOND peak
    first_peak = peaks[0]
    
    # If the file somehow only has 1 peak, just use the first peak for both
    second_peak = peaks[1] if len(peaks) >= 2 else peaks[0] 
    
    half_hold = int(1.5 * Config.FS)
    start_idx = max(0, first_peak - half_hold)
    end_idx = min(clean_data.shape[1], second_peak + half_hold)
    
    burst = clean_data[:, start_idx:end_idx]
    return burst

# ====================================================================================
# ================================== EXECUTION =======================================
# ====================================================================================
def main():
    parser = argparse.ArgumentParser(description="Data Validation Viewer for sEMG Signals")
    parser.add_argument('--mode', choices=['windows', 'overlay'], default='windows', help='Choose the viewing mode.')
    parser.add_argument('--subject', type=int, default=1, help='Subject ID (1-8) to view in overlay mode.')
    args = parser.parse_args()

    if args.mode == 'windows':
        print("Loading full dataset for Window Pagination Mode...")
        X_full, y_full = DataPreparation.load_and_prepare_dataset(base_path=Config.BASE_DATA_PATH)
        if len(X_full) == 0: return
        reverse_mapping = {tuple(v): k for k, v in Config.TARGET_MAPPING.items()}
        grouped_windows = {m: [] for m in range(1, 10)}
        for i in range(len(X_full)):
            target_tuple = tuple(y_full[i])
            if target_tuple in reverse_mapping:
                movement_id = reverse_mapping[target_tuple]
                grouped_windows[movement_id].append(X_full[i])
                
        for m in range(1, 10):
            windows = np.array(grouped_windows[m])
            if len(windows) > 0:
                print(f"--- Opening Viewer for Movement {m} ---")
                viewer = WindowViewer(movement_id=m, windows=windows, cols=5)

    elif args.mode == 'overlay':
        print(f"Loading data for Subject {args.subject} Overlay Mode...")
        movement_data_dict = {}
        for m in range(1, 9): # Ignore M9 (Rest)
            burst = extract_representative_burst(args.subject, m)
            movement_data_dict[m] = burst
            if burst is not None:
                print(f"Successfully extracted representative burst for M{m}")
            else:
                print(f"Skipped M{m} (File missing or blacklisted in Config)")
                
        viewer = ParticipantOverlayViewer(args.subject, movement_data_dict)

if __name__ == "__main__":
    main()