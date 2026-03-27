import sys
import time
import socket
import threading
import subprocess
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

        # --- UDP Communication Setup ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[Controller] UDP Socket initialized. Target: {Config.UDP_IP}:{Config.UDP_PORT}")
        
        # --- Recording State & STOP Command Handling ---
        self.stop_recording_requested = False
        self.recording_active = False
        self.input_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.input_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.input_sock.bind(('0.0.0.0', Config.UDP_PORT))
            self.input_sock.setblocking(False)
            print(f"[Controller] UDP Listener initialized on port {Config.UDP_PORT}")
        except Exception as e:
            print(f"[Controller] WARNING: Could not bind input socket: {e}")
            self.input_sock = None
        
        # Start UDP listener thread
        self.listener_thread = threading.Thread(target=self._listen_for_udp_commands, daemon=True)
        self.listener_thread.start()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[Controller] Successfully loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"[Controller] WARNING: '{model_path}' not found. Using untrained weights.")
        
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, Config.WINDOW_SIZE))
        
        # --- Calculate visualization buffer size based on VIS_WINDOW_SIZE_MS ---
        self.vis_window_size = int(Config.VIS_WINDOW_SIZE_MS / 1000.0 * Config.FS)  # Convert ms to samples
        self.vis_buffer = np.zeros((Config.NUM_CHANNELS, self.vis_window_size))
        
        # --- Kinematics uses independent 1500ms window (for local trend display) ---
        self.kin_window_ms = 1500
        self.kinematic_history_len = int(self.kin_window_ms / 1000.0 * Config.FS / Config.INCREMENT)  # 1500ms of samples at update rate
        self.yaw_history = np.zeros(self.kinematic_history_len)
        self.pitch_history = np.zeros(self.kinematic_history_len)

        self.alpha = Config.SMOOTHING_ALPHA
        self.smoothed_output = np.zeros(Config.NUM_OUTPUTS)
        
        # --- Dynamic Y-axis scaling for EMG plots ---
        self.channel_mins = np.full(Config.NUM_CHANNELS, np.inf)  # Track minimum value per channel
        self.channel_maxs = np.full(Config.NUM_CHANNELS, -np.inf)  # Track maximum value per channel
        self.plot_y_ranges = {}  # Store current Y-range for each plot to avoid redundant updates

        self.setup_gui()

    def setup_gui(self):
        """
        Create a unified window with EMG plots above and Kinematics plot below.
        EMG and Kinematics share the same X-axis (synchronized time).
        8:2 split with EMG taking 80% of the screen height.
        """
        self.app = QtWidgets.QApplication(sys.argv)
        
        # --- Create main window (fullscreen) ---
        self.main_window = pg.GraphicsLayoutWidget(show=True, title=f"EMG & Kinematics - {self.trial_name}")
        self.main_window.setWindowTitle(f'EMG & Kinematics Feed ({self.trial_name})')
        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.main_window.showFullScreen()
        
        self.plots_emg = []
        self.curves_emg = []
        
        # Initial Y-range for EMG (will auto-adjust based on incoming data)
        Y_MIN = -0.0001
        Y_MAX = 0.0001

        # --- Create reference ViewBox for X-axis linking ---
        self.ref_viewbox = None
        
        for i in range(Config.NUM_CHANNELS):
            p = self.main_window.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', Config.CHANNEL_MAP.get(i, f"Ch {i}"), width=80)
            
            # Set initial Y-range
            p.setYRange(Y_MIN, Y_MAX, padding=0)
            p.disableAutoRange(axis=pg.ViewBox.YAxis)

            # Link X-axes: first plot is reference, others link to it
            if self.ref_viewbox is None:
                self.ref_viewbox = p.getViewBox()
            else:
                p.setXLink(self.ref_viewbox)
            
            # Hide X-axis labels for all but the top EMG plot (will show below)
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
        
        # --- Create Kinematics plot below EMG (row=8, col=0) ---
        # This will be below all 8 EMG channels with independent 1500ms window
        self.plot_kin = self.main_window.addPlot(row=8, col=0)
        self.plot_kin.showGrid(x=True, y=True, alpha=0.5)
        self.plot_kin.setLabel('left', 'Angle (Degrees)', width=80)
        self.plot_kin.setLabel('bottom', f'Time (ms)')
        
        # Kinematics has independent X-axis: fixed 1500ms window, not linked to EMG
        self.plot_kin.setXRange(0, 1500, padding=0)
        self.plot_kin.disableAutoRange(axis=pg.ViewBox.XAxis)
        
        self.plot_kin.setYRange(-40, 130, padding=0)
        self.plot_kin.disableAutoRange(axis=pg.ViewBox.YAxis)
        
        self.plot_kin.addLegend(offset=(10, 10))
        
        self.curve_yaw = self.plot_kin.plot(pen=pg.mkPen('r', width=3), name='Yaw (Flexion)')
        self.curve_pitch = self.plot_kin.plot(pen=pg.mkPen('c', width=3), name='Pitch (Abduction)')
        
        # Note: PyQtGraph GraphicsLayout automatically distributes space proportionally
        # EMG plots (rows 0-7) will take ~80% of screen, Kinematics (row 8) will take ~20%

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
        
        # Apply filtering pipeline to visualization data to see actual muscle activity
        filtered_new_data = np.zeros_like(new_data)
        for i in range(Config.NUM_CHANNELS):
            # filtered_new_data[i, :] = SignalProcessing.applyStandardSEMGProcessing(
            #     new_data[i, :], fs=Config.FS
            # )
            filtered_new_data[i, :] = new_data[i, :]
        
        self.vis_buffer = np.roll(self.vis_buffer, -Config.INCREMENT, axis=1)
        self.vis_buffer[:, -Config.INCREMENT:] = filtered_new_data
        
        for i in range(Config.NUM_CHANNELS):
            self.curves_emg[i].setData(self.vis_buffer[i])
        
        # --- Update dynamic Y-axis scaling based on incoming data ---
        y_range_updated = False
        for i in range(Config.NUM_CHANNELS):
            channel_data = self.vis_buffer[i]
            channel_min = np.min(channel_data)
            channel_max = np.max(channel_data)
            
            # Track overall min/max for this channel
            if channel_min < self.channel_mins[i]:
                self.channel_mins[i] = channel_min
                y_range_updated = True
            if channel_max > self.channel_maxs[i]:
                self.channel_maxs[i] = channel_max
                y_range_updated = True
        
        # Update plot Y-ranges if new extremes were found
        if y_range_updated:
            for i in range(Config.NUM_CHANNELS):
                # Calculate range with 10% padding
                range_span = self.channel_maxs[i] - self.channel_mins[i]
                padding = max(range_span * 0.1, 1e-5)  # At least 10% padding or small epsilon
                
                new_y_min = self.channel_mins[i] - padding
                new_y_max = self.channel_maxs[i] + padding
                
                # Only update if significantly different from current range (avoid jitter)
                if i not in self.plot_y_ranges or \
                   abs(self.plot_y_ranges[i][0] - new_y_min) > 1e-6 or \
                   abs(self.plot_y_ranges[i][1] - new_y_max) > 1e-6:
                    self.plots_emg[i].setYRange(new_y_min, new_y_max, padding=0)
                    self.plot_y_ranges[i] = (new_y_min, new_y_max)
        
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
        
        # Calculate time values in milliseconds for kinematics plot
        # Each data point is Config.INCREMENT ms apart (one per control_step call)
        kin_time_values = np.arange(len(self.yaw_history)) * Config.INCREMENT
        
        # Plot with time axis (scrolling right-to-left like EMG plots)
        self.curve_yaw.setData(kin_time_values, self.yaw_history)
        self.curve_pitch.setData(kin_time_values, self.pitch_history)

        packet_string = f"{yaw:.2f},{pitch:.2f},{roll:.2f},{elbow:.2f}"
        print(f"Sending Telemetry: {packet_string}")
        try:
            self.sock.sendto(packet_string.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
        except Exception as e:
            pass
        
        # --- Check for STOP command if actively recording ---
        if self.recording_active and self.stop_recording_requested:
            print("\n[Controller] STOP command received from Virtual Environment!")
            self._save_and_transfer_collection()
            self.stop_recording_requested = False
            self.recording_active = False
            self.timer.stop()
            self.app.quit()

    def _listen_for_udp_commands(self):
        """
        Background thread that listens for incoming UDP commands from Virtual Environment.
        Specifically looks for CMD:STOP to cease recording.
        """
        print("[Controller] UDP listener thread started (daemon)")
        while True:
            try:
                if self.input_sock is None:
                    time.sleep(0.1)
                    continue
                
                data, addr = self.input_sock.recvfrom(1024)
                command = data.decode('utf-8').strip()
                print(f"[UDP Listener] Received command: {command} from {addr}")
                
                if "CMD:STOP" in command:
                    print("[UDP Listener] STOP command detected!")
                    self.stop_recording_requested = True
                
            except BlockingIOError:
                # No data available, continue checking
                time.sleep(0.01)
            except Exception as e:
                print(f"[UDP Listener] Error receiving command: {e}")
                time.sleep(0.1)
    
    def _save_and_transfer_collection(self):
        """
        Save the collected EMG data locally on the Pi.
        The Windows deployment script will automatically pull it upon exit.
        """
        if not isinstance(self.reading_mode, DataCollectionMode):
            print("[Controller] Not in data collection mode. Skipping save.")
            return
        
        try:
            # Save .mat file locally on Raspberry Pi
            output_path = self.reading_mode.save_collection()
            print(f"[Controller] Data saved locally: {output_path}")
            print(f"[Controller] Windows batch script will automatically retrieve this file upon exit.")
            
        except Exception as e:
            print(f"[Controller] ERROR during save: {e}")


    def run(self):
        """Start the real-time control GUI and event loop."""
        mode_desc = "Hardware" if self.use_hardware else "Simulation" if self.simulate_data else "Random Noise"
        print(f"\n[Controller] Starting Real-Time Control GUI ({mode_desc} mode). Close the window to stop.")
        print(f"[Controller] To shutdown: press Ctrl+C or close the GUI window")
        
        # === DATA COLLECTION MODE: MOVEMENT SELECTION & START TRIGGER ===
        is_collecting = self.use_hardware and isinstance(self.reading_mode, DataCollectionMode)
        selected_movement = None
        
        if is_collecting:
            print("\n" + "="*70)
            print("DATA COLLECTION MODE - SELECT MOVEMENT")
            print("="*70)
            print(f"Collection name: {self.reading_mode.collection_name}\n")
            
            # Display movement table
            print("Available Movements:")
            print("-" * 70)
            print(f"{'ID':<5} {'Movement':<30} {'Kinematics':<35}")
            print("-" * 70)
            for movement_id in sorted(Config.MOVEMENT_NAMES.keys()):
                movement_name = Config.MOVEMENT_NAMES[movement_id]
                kinematics = Config.TARGET_MAPPING[movement_id]
                kinematics_str = f"[Yaw={kinematics[0]:.0f}°, Pitch={kinematics[1]:.0f}°, Roll={kinematics[2]:.0f}°, Elbow={kinematics[3]:.0f}°]"
                print(f"{movement_id:<5} {movement_name:<30} {kinematics_str:<35}")
            print("-" * 70)
            print()
            
            # Prompt for movement selection
            while selected_movement is None:
                try:
                    user_input = input("Enter movement ID (1-9): ").strip()
                    movement_id = int(user_input)
                    
                    if movement_id < 1 or movement_id > 9:
                        print(f"[ERROR] Invalid movement ID: {movement_id}. Please enter a number between 1 and 9.")
                        continue
                    
                    selected_movement = movement_id
                    selected_name = Config.MOVEMENT_NAMES[movement_id]
                    print(f"\n[Controller] Selected Movement {movement_id}: {selected_name}")
                    
                except ValueError:
                    print(f"[ERROR] Invalid input: '{user_input}'. Please enter a valid number (1-9).")
                except EOFError:
                    # Handle SSH headless mode
                    print("[Controller] Running in headless mode. Using default (Movement 9: Rest).")
                    selected_movement = 9
                    selected_name = Config.MOVEMENT_NAMES[9]
                except KeyboardInterrupt:
                    print("\n[Controller] Interrupted during movement selection.")
                    return 1
            
            # === SEND UDP GO COMMAND WITH MOVEMENT CLASS ===
            print(f"\n[Controller] Sending GO command to Unity with Movement Class {selected_movement}...")
            try:
                go_cmd = f"CMD:GO_RECORDING:{selected_movement}"
                self.sock.sendto(go_cmd.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
                print(f"[Controller] UDP command sent: {go_cmd}")
            except Exception as e:
                print(f"[Controller] WARNING: Could not send UDP command: {e}")
            
            # === 3-SECOND COUNTDOWN BEFORE RECORDING STARTS ===
            print("\n[Controller] Starting in...")
            for countdown in range(3, 0, -1):
                print(f"  {countdown}...", end=" ", flush=True)
                time.sleep(1)
            print("GO!")
            print("[Controller] Recording started. Streaming EMG data to Unity.\n")
            print("[Controller] Waiting for STOP command from Virtual Environment...")
            self.recording_active = True
        
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
            if not is_collecting:
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
