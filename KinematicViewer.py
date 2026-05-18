"""KinematicViewer.py
Interactive viewer for ground truth kinematics from secondary data.
Allows selecting a participant and cycling through movements with arrow buttons.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import use
use('TkAgg')  # Use TkAgg backend for better button responsiveness

import ControllerConfiguration as Config

# Configuration
DATA_DIR = os.path.join('biosignal_data', 'secondary', 'edited')
DOF_NAMES = ['Yaw (Flexion/Extension)', 'Pitch (Abduction/Adduction)']
DOF_COLORS = ['tab:blue', 'tab:orange']
FS = 1000.0  # Sampling rate in Hz


class KinematicViewer:
    def __init__(self):
        self.current_participant = None
        self.current_movement_idx = 0
        self.movements = []
        self.current_kinematics = None
        self.current_original = None
        self.current_time = None

        # Set up figure and axes
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title("Kinematic Viewer")

        # Create button axes
        ax_prev = self.fig.add_axes([0.35, 0.05, 0.08, 0.075])
        ax_next = self.fig.add_axes([0.57, 0.05, 0.08, 0.075])

        self.btn_prev = Button(ax_prev, '← Previous')
        self.btn_next = Button(ax_next, 'Next →')

        self.btn_prev.on_clicked(self.on_prev_clicked)
        self.btn_next.on_clicked(self.on_next_clicked)

        # Initial participant selection
        self.select_participant()

    def get_available_participants(self):
        """Get list of available participant folders."""
        if not os.path.exists(DATA_DIR):
            print(f"Data directory not found: {DATA_DIR}")
            return []

        participants = sorted([d for d in os.listdir(DATA_DIR)
                              if os.path.isdir(os.path.join(DATA_DIR, d))])
        return participants

    def get_available_movements(self, participant):
        """Get list of available edited kinematic files for a participant."""
        participant_dir = os.path.join(DATA_DIR, participant)
        kinematic_files = glob.glob(os.path.join(participant_dir, 'MovimentoAngS*_edit.mat'))

        movements = []
        for f in kinematic_files:
            basename = os.path.basename(f)
            try:
                num_str = basename.replace('MovimentoAngS', '').replace('_edit.mat', '')
                num = int(num_str)
                movements.append((num, f))
            except ValueError:
                continue

        movements.sort(key=lambda x: x[0])
        return movements

    def select_participant(self):
        """Interactively select a participant."""
        participants = self.get_available_participants()

        if not participants:
            print("No participants found!")
            return False

        print("\nAvailable Participants:")
        for i, p in enumerate(participants, 1):
            print(f"  {i}. {p}")

        while True:
            try:
                choice = int(input(f"Select participant (1-{len(participants)}): "))
                if 1 <= choice <= len(participants):
                    self.current_participant = participants[choice - 1]
                    self.movements = self.get_available_movements(self.current_participant)

                    if not self.movements:
                        print(f"No edited kinematic files found for {self.current_participant}")
                        continue

                    self.current_movement_idx = 0
                    print(f"\nSelected: {self.current_participant}")
                    print(f"Found {len(self.movements)} movements")
                    return True
                else:
                    print(f"Please enter a number between 1 and {len(participants)}")
            except ValueError:
                print("Please enter a valid number")

    def load_kinematic_file(self, filepath):
        """Load rescaled and original kinematic signals from an _edit.mat file.

        Returns (time, rescaled, original) where original may be None if the key
        is absent (e.g. older files written before angolospalla_original was added).
        """
        try:
            mat_data = sio.loadmat(filepath)

            if 'angolospalla' not in mat_data:
                print(f"Warning: no 'angolospalla' key in {filepath}")
                print(f"Available keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")
                return None, None, None

            rescaled = np.asarray(mat_data['angolospalla'], dtype=np.float64).flatten()
            original = (
                np.asarray(mat_data['angolospalla_original'], dtype=np.float64).flatten()
                if 'angolospalla_original' in mat_data else None
            )

            time = np.arange(len(rescaled)) / FS
            return time, rescaled, original

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None, None

    def plot_kinematic(self):
        """Plot current kinematic data."""
        if not self.movements:
            print("No movements available")
            return

        movement_num, filepath = self.movements[self.current_movement_idx]

        time, rescaled, original = self.load_kinematic_file(filepath)
        if time is None:
            print(f"Failed to load kinematic data for movement {movement_num}")
            return

        self.current_kinematics = rescaled
        self.current_original = original
        self.current_time = time

        self.ax.clear()

        # Route signal to correct DOF(s) based on TARGET_MAPPING
        target_vec = Config.TARGET_MAPPING.get(movement_num, [0.0, 0.0, 0.0, 0.0])
        active_dofs = [i for i, v in enumerate(target_vec) if v != 0.0 and i < len(DOF_NAMES)]
        if not active_dofs:
            active_dofs = [0]

        for dof_idx in active_dofs:
            color = DOF_COLORS[dof_idx] if dof_idx < len(DOF_COLORS) else f'C{dof_idx}'
            label = DOF_NAMES[dof_idx] if dof_idx < len(DOF_NAMES) else f'DOF {dof_idx}'

            self.ax.plot(time, rescaled, label=label, color=color, linewidth=2)
            if original is not None:
                self.ax.plot(time, original, label=f'{label} (original)',
                             color=color, linewidth=1.5, linestyle='--', alpha=0.6)

        # Format plot
        self.ax.set_xlabel('Time (s)', fontsize=12)
        self.ax.set_ylabel('Angle (degrees)', fontsize=12)
        self.ax.set_title(
            f'{self.current_participant} - Movement {movement_num} '
            f'({self.current_movement_idx + 1}/{len(self.movements)})',
            fontsize=14, fontweight='bold'
        )
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, alpha=0.3)

        plt.subplots_adjust(bottom=0.2)

        self.fig.canvas.draw()

    def on_prev_clicked(self, event):
        """Handle previous button click."""
        self.current_movement_idx = (self.current_movement_idx - 1) % len(self.movements)
        self.plot_kinematic()

    def on_next_clicked(self, event):
        """Handle next button click."""
        self.current_movement_idx = (self.current_movement_idx + 1) % len(self.movements)
        self.plot_kinematic()

    def run(self):
        """Start the viewer."""
        if self.current_participant and self.movements:
            self.plot_kinematic()
            plt.show()
        else:
            print("No participant or movements selected")


def main():
    viewer = KinematicViewer()
    viewer.run()


if __name__ == '__main__':
    main()
