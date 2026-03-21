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
import scipy.io
import os
import signal

import SignalProcessing
from ModelTraining import ShoulderRCNN 
import ControllerConfiguration as Config
from SignalReading import ContinuousReadingMode, DataCollectionMode

class RealTimeProstheticController:
    def __init__(self, model_path='best_shoulder_rcnn.pth', reading_mode=None, simulate_data=False, sim_file=None):
        """
        Initialize the Real-Time Prosthetic Controller.
        
        Args:
            model_path (str): Path to trained model weights
            reading_mode: SignalReadingMode object (ContinuousReadingMode or DataCollectionMode).
                        If None and not simulating, defaults to ContinuousReadingMode.
                        If simulating, hardware reading is disabled.
            simulate_data (bool): Use simulation mode instead of hardware
            sim_file (str): Path to .mat file for simulation playback
        """
        self.reading_mode = reading_mode
        self.simulate_data = simulate_data
        self.use_hardware = (reading_mode is not None) and (not simulate_data)
        
        # --- PARSE TRIAL NAME FOR TITLES ---
        self.trial_name = "Live Stream"
        if sim_file:
            match = re.search(r'Soggetto(\d+).*Movimento(\d+)', sim_file)
            if match:
                self.trial_name = f"P{match.group(1)}M{match.group(2)}"

        # --- LOAD SIMULATION DATA ---
        if self.simulate_data and sim_file:
            if os.path.exists(sim_file):
                print(f"[Controller] Loading simulation stream from: {sim_file} ({self.trial_name})")
                mat = scipy.io.loadmat(sim_file)
                self.sim_data_stream = mat['EMGDATA'] 
                self.sim_playback_idx = 0
            else:
                print(f"[Controller] ERROR: Could not find {sim_file}. Falling back to random noise.")
                self.sim_data_stream = None
        else:
            self.sim_data_stream = None

        # --- INITIALIZE DEFAULT HARDWARE MODE IF PROVIDED ---
        if self.use_hardware and self.reading_mode is None:
            print("[Controller] No reading_mode provided. Defaulting to ContinuousReadingMode.")
            self.reading_mode = ContinuousReadingMode()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[Controller] UDP Socket initialized. Target: {Config.UDP_IP}:{Config.UDP_PORT}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[Controller] Successfully loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"[Controller] WARNING: '{model_path}' not found. Using untrained weights.")
        
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
        """
        Read new samples from hardware or simulation.
        
        Args:
            num_samples (int): Number of samples to read (Config.INCREMENT)
            
        Returns:
            np.ndarray: Shape (Config.NUM_CHANNELS, num_samples) array of ADC values
        """
        if self.use_hardware:
            # Read from hardware via reading mode
            chunk = self.reading_mode.read_sample_chunk()
            return chunk
        elif self.simulate_data:
            # Read from simulation
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
            # Fallback to random noise
            return np.random.randn(Config.NUM_CHANNELS, num_samples) * 0.1

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
        """Start the real-time control GUI and event loop."""
        mode_desc = "Hardware" if self.use_hardware else "Simulation" if self.simulate_data else "Random Noise"
        print(f"\n[Controller] Starting Real-Time Control GUI ({mode_desc} mode). Close the window to stop.")
        print(f"[Controller] To shutdown: press Ctrl+C or close the GUI window")
        
        self.running = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.control_step)
        self.timer.start(Config.INCREMENT)
        
        # --- Graceful Shutdown Handlers ---
        def handle_sigint(sig, frame):
            """Handle Ctrl+C (SIGINT) signal."""
            print("\n[Controller] SIGINT received. Initiating graceful shutdown...")
            self.app.quit()
        
        def handle_sigterm(sig, frame):
            """Handle SIGTERM signal (used by systemd/SSH)."""
            print("\n[Controller] SIGTERM received. Initiating graceful shutdown...")
            self.app.quit()

        # Register signal handlers
        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigterm)

        try:
            print("[Controller] Event loop running. Press Ctrl+C to stop.\n")
            exit_code = self.app.exec()
            return exit_code
        except KeyboardInterrupt:
            print("\n[Controller] Keyboard interrupt in main thread")
            self.app.quit()
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time EMG Prosthetic Controller")
    
    # Mode selection
    parser.add_argument('--hardware', action='store_true', help='Use hardware (MCP3008) for real-time reading')
    parser.add_argument('--continuous', action='store_true', help='Continuous reading mode (hardware only)')
    parser.add_argument('--collect', action='store_true', help='Data collection mode (hardware only) - stores data for later')
    parser.add_argument('--collection_name', type=str, default='hardware_trial', help='Name for collected data (used with --collect)')
    parser.add_argument('--simulate', action='store_true', help='Use simulation mode with sample .mat file')
    
    # Model and file paths
    parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH, help='Path to trained PyTorch weights')
    parser.add_argument('--sim_file', type=str, default='./secondary_data/Soggetto1/Movimento3.mat', help='Specific .mat file to stream (simulation mode)')
    
    args = parser.parse_args()
    
    # === CONFIGURE READING MODE ===
    reading_mode = None
    
    if args.hardware:
        if args.collect:
            print("[Main] Initializing Data Collection Mode (hardware will store readings for later saving)")
            reading_mode = DataCollectionMode(collection_name=args.collection_name)
        else:
            print("[Main] Initializing Continuous Reading Mode (hardware reads without persistent storage)")
            reading_mode = ContinuousReadingMode()
    elif args.simulate:
        print("[Main] Initializing Simulation Mode (reading from .mat file)")
    else:
        print("[Main] No mode specified. Using Simulation Mode by default.")
        args.simulate = True
    
    # === START CONTROLLER ===
    controller = RealTimeProstheticController(
        model_path=args.model, 
        reading_mode=reading_mode,
        simulate_data=args.simulate, 
        sim_file=args.sim_file
    )
    
    exit_code = 0
    
    try:
        controller.run()
    except Exception as e:
        print(f"\n[Main] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # === CLEANUP RESOURCES ===
        print("\n[Main] Cleaning up resources...")
        
        try:
            controller.sock.close()
            print("[Main] ✓ UDP Socket closed")
        except Exception as e:
            print(f"[Main] Warning: Could not close socket: {e}")
        
        try:
            if controller.use_hardware and controller.reading_mode:
                controller.reading_mode.cleanup()
                print("[Main] ✓ Hardware resources cleaned up")
        except Exception as e:
            print(f"[Main] Warning: Could not cleanup hardware: {e}")
        
        # === SAVE DATA IF IN COLLECTION MODE ===
        if args.hardware and args.collect and reading_mode:
            try:
                stats = reading_mode.get_collection_stats()
                print(f"\n[Main] Collection Statistics:")
                print(f"       Samples: {stats['sample_count']}")
                print(f"       Duration: {stats['duration_seconds']:.2f} seconds")
                print(f"       Memory Used: {stats['memory_mb']:.2f} MB")
                
                # In headless mode (SSH), auto-save to default location
                # In interactive mode, prompt the user
                import sys
                if sys.stdin.isatty():
                    save_prompt = input("[Main] Save collected data? (y/n): ").lower().strip()
                    if save_prompt == 'y':
                        output_path = input("[Main] Enter output path (or press Enter for default): ").strip()
                        if not output_path:
                            output_path = None
                        reading_mode.save_collection(output_path=output_path)
                        print("[Main] ✓ Data saved")
                else:
                    # Headless/SSH mode: auto-save to default
                    print("[Main] Headless mode detected. Auto-saving collected data...")
                    reading_mode.save_collection()
                    print("[Main] ✓ Data auto-saved")
            except EOFError:
                print("[Main] EOF detected (headless mode). Skipping save prompt.")
            except Exception as e:
                print(f"[Main] Error during data saving: {e}")
        
        print("[Main] Shutdown complete\n")
        sys.exit(exit_code)
