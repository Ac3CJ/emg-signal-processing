import sys
import time
import socket
import numpy as np
import torch
import argparse
import re
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import scipy.signal

import SignalProcessing
from ModelTraining import ShoulderRCNN 
import ControllerConfiguration as Config

class RealTimeProstheticController:
    def __init__(self, model_path='best_shoulder_rcnn.pth', simulate_data=False, sim_file=None):
        self.simulate_data = simulate_data
        
        # --- PARSE TRIAL NAME FOR TITLES ---
        self.trial_name = "Live Stream"
        if sim_file:
            match = re.search(r'Soggetto(\d+).*Movimento(\d+)', sim_file)
            if match:
                self.trial_name = f"P{match.group(1)}M{match.group(2)}"

        # --- LOAD SIMULATION DATA ---
        if self.simulate_data and sim_file:
            import scipy.io
            import os
            if os.path.exists(sim_file):
                print(f"Loading simulation stream from: {sim_file} ({self.trial_name})")
                mat = scipy.io.loadmat(sim_file)
                self.sim_data_stream = mat['EMGDATA'] 
                self.sim_playback_idx = 0
            else:
                print(f"ERROR: Could not find {sim_file}. Falling back to random noise.")
                self.sim_data_stream = None
        else:
            self.sim_data_stream = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP Socket initialized. Target: {Config.UDP_IP}:{Config.UDP_PORT}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"WARNING: '{model_path}' not found. Using untrained weights.")
        
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, Config.WINDOW_SIZE))
        
        self.vis_window_size = Config.WINDOW_SIZE * 3 
        self.vis_buffer = np.zeros((Config.NUM_CHANNELS, self.vis_window_size))
        
        self.kinematic_history_len = 300
        self.yaw_history = np.zeros(self.kinematic_history_len)
        self.pitch_history = np.zeros(self.kinematic_history_len)

        self.alpha = Config.SMOOTHING_ALPHA
        self.smoothed_output = np.zeros(Config.NUM_OUTPUTS)

        self.setup_gui()

    def setup_gui(self):
        self.app = QtWidgets.QApplication(sys.argv)
        
        self.win_emg = pg.GraphicsLayoutWidget(show=True, title=f"sEMG Stream - {self.trial_name}")
        self.win_emg.resize(1000, 800)
        self.win_emg.setWindowTitle(f'EMG Feed ({self.trial_name})')
        
        self.plots_emg = []
        self.curves_emg = []
        
        Y_MIN = -0.0002
        Y_MAX = 0.0004

        for i in range(Config.NUM_CHANNELS):
            p = self.win_emg.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', Config.CHANNEL_MAP.get(i, f"Ch {i}"))
            
            p.setYRange(Y_MIN, Y_MAX, padding=0)
            p.disableAutoRange(axis=pg.ViewBox.YAxis)

            if i < Config.NUM_CHANNELS - 1:
                p.hideAxis('bottom')
            
            vLine = pg.InfiniteLine(angle=90, movable=False, pos=self.vis_window_size - Config.WINDOW_SIZE)
            
            try:
                dash_style = QtCore.Qt.PenStyle.DashLine
            except AttributeError:
                dash_style = QtCore.Qt.DashLine
                
            vLine.setPen(pg.mkPen(color='r', style=dash_style, width=2))
            p.addItem(vLine)
            
            curve = p.plot(pen=pg.mkPen(color=(50, 150, 255), width=1))
            self.plots_emg.append(p)
            self.curves_emg.append(curve)
        
        self.win_kin = pg.GraphicsLayoutWidget(show=True, title=f"Kinematics - {self.trial_name}")
        self.win_kin.resize(800, 400)
        self.win_kin.setWindowTitle(f'Predicted Output ({self.trial_name})')
        
        self.plot_kin = self.win_kin.addPlot(title=f"Joint Angles: {self.trial_name}")
        self.plot_kin.showGrid(x=True, y=True, alpha=0.5)
        
        self.plot_kin.setYRange(-40, 130, padding=0)
        self.plot_kin.disableAutoRange(axis=pg.ViewBox.YAxis)
        
        self.plot_kin.setLabel('left', 'Angle (Degrees)')
        self.plot_kin.setLabel('bottom', 'Time Steps (62ms)')
        
        self.plot_kin.addLegend(offset=(10, 10))
        
        self.curve_yaw = self.plot_kin.plot(pen=pg.mkPen('r', width=3), name='Yaw (Flexion)')
        self.curve_pitch = self.plot_kin.plot(pen=pg.mkPen('c', width=3), name='Pitch (Abduction)')

    def read_new_samples(self, num_samples):
        if self.simulate_data:
            if self.sim_data_stream is not None:
                end_idx = self.sim_playback_idx + num_samples
                if end_idx > self.sim_data_stream.shape[1]:
                    self.sim_playback_idx = 0
                    end_idx = num_samples
                chunk = self.sim_data_stream[:, self.sim_playback_idx:end_idx]
                self.sim_playback_idx = end_idx
                return chunk
            else:
                return np.random.randn(Config.NUM_CHANNELS, num_samples) * 0.1
        else:
            pass

    def control_step(self):
        new_data = self.read_new_samples(Config.INCREMENT)
        
        self.data_buffer = np.roll(self.data_buffer, -Config.INCREMENT, axis=1)
        self.data_buffer[:, -Config.INCREMENT:] = new_data
        
        self.vis_buffer = np.roll(self.vis_buffer, -Config.INCREMENT, axis=1)
        self.vis_buffer[:, -Config.INCREMENT:] = new_data
        
        for i in range(Config.NUM_CHANNELS):
            self.curves_emg[i].setData(self.vis_buffer[i])
        
        cleaned_window = np.zeros_like(self.data_buffer)
        for i in range(Config.NUM_CHANNELS):
            cleaned_window[i, :] = SignalProcessing.applyStandardSEMGProcessing(self.data_buffer[i, :], fs=Config.FS)
        
        input_tensor = torch.tensor(cleaned_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_predictions = self.model(input_tensor).cpu().numpy()[0]
        
        self.smoothed_output = (self.alpha * raw_predictions) + ((1 - self.alpha) * self.smoothed_output)
        yaw, pitch, roll, elbow = self.smoothed_output
        
        self.yaw_history = np.roll(self.yaw_history, -1)
        self.yaw_history[-1] = yaw
        
        self.pitch_history = np.roll(self.pitch_history, -1)
        self.pitch_history[-1] = pitch
        
        self.curve_yaw.setData(self.yaw_history)
        self.curve_pitch.setData(self.pitch_history)

        packet_string = f"{yaw:.2f},{pitch:.2f},{roll:.2f},{elbow:.2f}"
        print(f"Sending Telemetry: {packet_string}")
        try:
            self.sock.sendto(packet_string.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
        except Exception as e:
            pass

    def run(self):
        print("\nStarting Real-Time Control GUI. Close the window to stop.")
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.control_step)
        self.timer.start(Config.INCREMENT)
        
        try:
            sys.exit(self.app.exec())
        finally:
            self.sock.close()
            print("UDP Socket closed safely.")

def get_predictions_for_file(model, device, file_path):
    import scipy.io
    mat = scipy.io.loadmat(file_path)
    raw_data = mat['EMGDATA']
    num_samples = raw_data.shape[1]

    clean_data = np.zeros_like(raw_data, dtype=np.float32)
    for c in range(Config.NUM_CHANNELS):
        sig = SignalProcessing.notchFilter(raw_data[c, :], fs=Config.FS, notchFreq=Config.NOTCH_FREQ)
        sig = SignalProcessing.bandpassFilter(sig, fs=Config.FS, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
        clean_data[c, :] = np.abs(sig)

    predictions = []
    smoothed_pred = np.zeros(Config.NUM_OUTPUTS)
    window_starts = list(range(0, num_samples - Config.WINDOW_SIZE + 1, Config.INCREMENT))
    
    with torch.no_grad():
        for start in window_starts:
            window = clean_data[:, start:start+Config.WINDOW_SIZE]
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred = model(window_tensor).cpu().numpy()[0]
            smoothed_pred = (Config.SMOOTHING_ALPHA * pred) + ((1.0 - Config.SMOOTHING_ALPHA) * smoothed_pred)
            predictions.append(smoothed_pred.copy())
            
    return np.array(predictions), np.array(window_starts) / Config.FS

def run_fast_validation(model_path, sim_file=None, predefined=False, base_path='./secondary_data'):
    import matplotlib.pyplot as plt
    import os
    import re
    
    output_dir = 'kinematic-plots'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Fast Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if predefined:
        print("[Fast Validation] Running predefined benchmark suite...")
        test_cases = [(6, 1), (7, 2), (2, 3), (6, 4), (7, 5), (5, 6), (6, 7), (8, 8)]
        files_to_process = [os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}.mat') for p, m in test_cases]
    else:
        if not sim_file:
            print("ERROR: No simulation file provided for validation.")
            return
        files_to_process = [sim_file]

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"  -> Skipping: File not found at {file_path}")
            continue

        match = re.search(r'Soggetto(\d+).*Movimento(\d+)', file_path.replace('\\', '/'))
        if match:
            subject_id = match.group(1)
            m = match.group(2)
            trial_name = f"{m}-kinematics-P{subject_id}M{m}"
        else:
            trial_name = "Unknown_Trial"

        print(f"  -> Analyzing {trial_name}...")
        predictions, time_axis = get_predictions_for_file(model, device, file_path)

        plt.figure(figsize=(18, 5))
        plt.plot(time_axis, predictions[:, 0], label='Yaw (Flex/Ext)', color='tab:blue', linewidth=2)
        plt.plot(time_axis, predictions[:, 1], label='Pitch (Abd/Add)', color='tab:orange', linewidth=2)
        plt.plot(time_axis, predictions[:, 2], label='Roll (Int/Ext Rot)', color='tab:green', linewidth=2)
        plt.plot(time_axis, predictions[:, 3], label='Elbow (Flex)', color='tab:red', linewidth=2)
        
        # --- DYNAMIC Y-AXIS LIMITS ---
        y_min = np.min(predictions) - 10
        y_max = np.max(predictions) + 10
        plt.ylim(y_min, y_max)
        
        plt.title(trial_name, fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Predicted Angle (Degrees)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"{trial_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"\n[Fast Validation] Success! Plots saved to ./{output_dir}/")

def run_ensemble_validation(model_path, base_path='./secondary_data'):
    """
    Overlays the FULL prediction timeline of all valid participants for each movement,
    calculating the median response across the entire trial.
    """
    import matplotlib.pyplot as plt
    import os
    
    output_dir = 'ensemble-plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[Ensemble Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for m in range(1, 9):
        print(f"Aggregating full trials for Movement {m}...")
        all_subject_preds = []
        all_time_axes = []
        
        target_vec = Config.TARGET_MAPPING[m]
        
        for p in range(1, 9):
            if hasattr(Config, 'CORRUPTED_TRIALS') and (p, m) in Config.CORRUPTED_TRIALS:
                continue
                
            file_path = os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}.mat')
            if not os.path.exists(file_path): continue
            
            predictions, time_axis = get_predictions_for_file(model, device, file_path)
            all_subject_preds.append(predictions)
            all_time_axes.append(time_axis)
                    
        if len(all_subject_preds) == 0:
            print(f"  -> Skipping M{m}: No valid trials found.")
            continue
            
        # Truncate all trials to the length of the shortest trial so they can be stacked
        min_len = min(len(p) for p in all_subject_preds)
        stacked_trials = np.array([p[:min_len] for p in all_subject_preds])
        
        # Calculate the median across subjects for every time step
        median_trial = np.median(stacked_trials, axis=0)
        time_axis = all_time_axes[0][:min_len]
        
        plt.figure(figsize=(18, 5)) # Wider figure to fit the full 100+ seconds
        plt.plot(time_axis, median_trial[:, 0], label='Yaw (Flex/Ext)', color='tab:blue', linewidth=2)
        plt.plot(time_axis, median_trial[:, 1], label='Pitch (Abd/Add)', color='tab:orange', linewidth=2)
        plt.plot(time_axis, median_trial[:, 2], label='Roll (Int/Ext Rot)', color='tab:green', linewidth=2)
        plt.plot(time_axis, median_trial[:, 3], label='Elbow (Flex)', color='tab:red', linewidth=2)
        
        # --- DYNAMIC Y-AXIS LIMITS ---
        y_min = np.min(median_trial) - 10
        y_max = np.max(median_trial) + 10
        plt.ylim(y_min, y_max)
        
        plt.title(f"Full-Sequence Ensemble Average: Movement {m} (Target: {target_vec})\nMedian of {len(stacked_trials)} valid subjects across entire timeline", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Predicted Angle (Degrees)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"M{m}-Full-Ensemble.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"\n[Ensemble Validation] Success! Plots saved to ./{output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time EMG Prosthetic Controller")
    parser.add_argument('--simulate', action='store_true', help='Generate dummy sEMG data instead of reading hardware')
    parser.add_argument('--validate', action='store_true', help='Fast-forward process a single sim_file')
    parser.add_argument('--validate_predefined', action='store_true', help='Run the entire benchmark suite of 8 files')
    parser.add_argument('--validate_ensemble', action='store_true', help='Generate aligned median average plots across all participants')
    parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH, help='Path to the trained PyTorch weights')
    parser.add_argument('--sim_file', type=str, default='./secondary_data/Soggetto1/Movimento3.mat', help='Specific .mat file to stream')
    
    args = parser.parse_args()
    
    if args.validate_ensemble:
        run_ensemble_validation(model_path=args.model, base_path=Config.BASE_DATA_PATH)
    if args.validate_predefined:
        run_fast_validation(model_path=args.model, predefined=True, base_path=Config.BASE_DATA_PATH)
    if args.validate:
        run_fast_validation(model_path=args.model, sim_file=args.sim_file, predefined=False)
        
    if not (args.validate or args.validate_predefined or args.validate_ensemble):
        controller = RealTimeProstheticController(
            model_path=args.model, 
            simulate_data=args.simulate, 
            sim_file=args.sim_file
        )
        controller.run()