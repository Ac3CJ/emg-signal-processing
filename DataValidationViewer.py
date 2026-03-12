import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import DataPreparation
import ControllerConfiguration as Config

class WindowViewer:
    def __init__(self, movement_id, windows, cols=5):
        """
        Creates an interactive Matplotlib viewer to page through signal windows.
        """
        self.movement_id = movement_id
        self.windows = windows  # Shape: (Num_Windows, 8, 500)
        self.cols = cols
        self.page = 0
        self.total_pages = int(np.ceil(len(windows) / cols))
        
        # Setup the figure and grid (8 rows for channels, 'cols' columns for windows)
        self.fig, self.axes = plt.subplots(Config.NUM_CHANNELS, cols, figsize=(16, 9), sharex=True, sharey='row')
        self.fig.canvas.manager.set_window_title(f'Validation - Movement {movement_id}')
        
        # Adjust layout to make room for buttons at the bottom
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.3, wspace=0.1)
        
        # --- Pagination Buttons ---
        axprev = plt.axes([0.35, 0.05, 0.1, 0.05])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_page)
        
        axnext = plt.axes([0.55, 0.05, 0.1, 0.05])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_page)
        
        self.draw_page()
        plt.show() # Blocks execution until the window is closed

    def draw_page(self):
        """Draws the current page of windows onto the grid."""
        self.fig.suptitle(f'Movement {self.movement_id} | Page {self.page + 1} of {self.total_pages}\nClose window to proceed to next movement.', fontsize=14, fontweight='bold')
        
        start_idx = self.page * self.cols
        
        for col in range(self.cols):
            window_idx = start_idx + col
            
            for row in range(Config.NUM_CHANNELS):
                ax = self.axes[row, col]
                ax.clear()
                
                # Check if we have run out of windows on the last page
                if window_idx < len(self.windows):
                    # Plot the 500ms window for this specific channel
                    signal = self.windows[window_idx, row, :]
                    ax.plot(signal, color='tab:blue', linewidth=1)
                    
                    # Formatting
                    if row == 0:
                        ax.set_title(f"Window {window_idx + 1}", fontsize=10)
                    if col == 0:
                        muscle_name = Config.CHANNEL_MAP.get(row, f"Ch {row}").split('(')[0].strip()
                        ax.set_ylabel(muscle_name, rotation=90, size=8)
                        
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.set_xticks([]) # Hide X ticks to reduce visual clutter
                else:
                    # Hide empty axes on the last page
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

def main():
    # 1. Generate the dataset using your established pipeline
    X_full, y_full = DataPreparation.load_and_prepare_dataset(base_path=Config.BASE_DATA_PATH)
    
    if len(X_full) == 0:
        print("No data extracted. Please check your data paths and blacklist.")
        return

    # 2. Reverse Map the Target Vectors back to Movement IDs
    # Because y_full contains floats like [45.0, 0.0, 0.0, 0.0], we use the Config target mapping
    reverse_mapping = {tuple(v): k for k, v in Config.TARGET_MAPPING.items()}
    
    # 3. Group the windows by their movement ID
    grouped_windows = {m: [] for m in range(1, 10)}
    
    for i in range(len(X_full)):
        target_tuple = tuple(y_full[i])
        if target_tuple in reverse_mapping:
            movement_id = reverse_mapping[target_tuple]
            grouped_windows[movement_id].append(X_full[i])
            
    # 4. Launch the viewer for each movement sequentially
    for m in range(1, 10):
        windows = np.array(grouped_windows[m])
        if len(windows) > 0:
            print(f"\n--- Opening Viewer for Movement {m} ---")
            print(f"Total extracted windows: {len(windows)}")
            
            # This will pause the script until you close the Matplotlib window
            viewer = WindowViewer(movement_id=m, windows=windows, cols=5)
        else:
            print(f"\nSkipping Movement {m} (No data extracted or all blacklisted).")
            
    print("\nValidation complete! All movements reviewed.")

if __name__ == "__main__":
    main()