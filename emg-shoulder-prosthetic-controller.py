import sys
import time
import socket
import threading
import subprocess
import queue
import numpy as np
import torch
import argparse
import re
import scipy.signal
import scipy.io
import os
import signal

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except Exception:
    pg = None
    QtCore = None
    QtWidgets = None

import SignalProcessing
import NeuralNetworkModels as NNModels
import ControllerConfiguration as Config
from FileRepository import DataRepository, FileSelection

_QMainWindowBase = QtWidgets.QMainWindow if QtWidgets is not None else object

class RealTimeProstheticController:
    def __init__(
        self,
        model_path='best_shoulder_rcnn.pth',
        reading_mode=None,
        simulate_data=False,
        sim_file=None,
        data_queue=None,
        participant_id=None,
    ):
        """
        Initialize the Real-Time Prosthetic Controller.
        
        Args:
            model_path (str): Path to trained model weights
            reading_mode: SignalReadingMode object (ContinuousReadingMode or DataCollectionMode).
                        If None and not simulating, defaults to ContinuousReadingMode.
                        If simulating, hardware reading is disabled.
            simulate_data (bool): Use simulation mode instead of hardware
            sim_file (str): Path to .mat file for simulation playback
            data_queue (queue.Queue): Optional queue for Producer-Consumer pattern. If provided,
                                     predictions are put on the queue. If None, runs headless.
            participant_id (str): Optional participant ID (e.g., P1 or 1). If omitted,
                                  resolved from current_participant.txt or defaults to P1.
        """
        self.data_queue = data_queue
        self.reading_mode = reading_mode
        self.simulate_data = simulate_data
        self.use_hardware = (reading_mode is not None) and (not simulate_data)
        self.repo = DataRepository()
        
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

        # --- BOOT-UP STATE MACHINE: PARTICIPANT + CALIBRATION ---
        self.participant_id = self._resolve_active_participant(participant_id)
        self.is_calibrated = False
        self.norm_baseline = np.zeros(Config.NUM_CHANNELS, dtype=np.float32)
        self.norm_max = np.ones(Config.NUM_CHANNELS, dtype=np.float32)
        self._initialize_calibration_state()

        # --- INITIALIZE DEFAULT HARDWARE MODE IF PROVIDED ---
        if self.use_hardware and self.reading_mode is None:
            print("[Controller] No reading_mode provided. Defaulting to ContinuousReadingMode.")
            self.reading_mode = ContinuousReadingMode()

        # --- UDP Communication Setup ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[Controller] UDP Socket initialized. Target: {Config.UDP_IP}:{Config.UDP_PORT}")
        
        # --- Recording State & Inter-Process Command Handling ---
        self.stop_recording_requested = False
        self.recording_active = False
        self.ack_received = threading.Event()  # Set when Unity replies CMD:ACK
        self.exit_requested = False  # Set by stdin watcher on 'finish'/'exit'
        self.session_save_paths = []  # Tracks files saved during the current session
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
        self.model = NNModels.ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[Controller] Successfully loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"[Controller] WARNING: '{model_path}' not found. Using untrained weights.")
        
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, Config.WINDOW_SIZE))
        self.warmup_samples_required = int(getattr(Config, 'WARMUP_SECONDS', 0.0) * Config.FS)
        self.real_samples_seen = 0

        self.alpha = Config.SMOOTHING_ALPHA
        self.smoothed_output = np.zeros(Config.NUM_OUTPUTS)
        
        # Kinematics history for telemetry
        self.kin_window_ms = 1500
        self.kinematic_history_len = int(self.kin_window_ms / 1000.0 * Config.FS / Config.INCREMENT)
        self.yaw_history = np.zeros(self.kinematic_history_len)
        self.pitch_history = np.zeros(self.kinematic_history_len)

    @staticmethod
    def _normalize_participant_token(participant_token):
        if participant_token is None:
            return None

        token = str(participant_token).strip().upper()
        if not token:
            return None
        if token.startswith("P"):
            token = token[1:]
        if not token.isdigit():
            return None
        return f"P{int(token)}"

    @staticmethod
    def _participant_token_to_int(participant_token):
        normalised = RealTimeProstheticController._normalize_participant_token(participant_token)
        if normalised is None:
            return None
        return int(normalised[1:])

    def _resolve_active_participant(self, participant_id):
        cli_participant = self._normalize_participant_token(participant_id)
        if cli_participant is not None:
            print(f"[Controller] Participant resolved from CLI: {cli_participant}")
            return cli_participant

        participant_file_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_participant.txt"),
            os.path.join(os.getcwd(), "current_participant.txt"),
        ]

        for participant_file_path in participant_file_candidates:
            if not os.path.exists(participant_file_path):
                continue

            try:
                with open(participant_file_path, "r", encoding="utf-8") as handle:
                    first_line = handle.readline().strip()
                file_participant = self._normalize_participant_token(first_line)
                if file_participant is not None:
                    print(f"[Controller] Participant resolved from current_participant.txt: {file_participant}")
                    return file_participant
                print("[Controller] current_participant.txt is invalid. Falling back to default P1.")
            except Exception as e:
                print(f"[Controller] Could not read current_participant.txt: {e}. Falling back to default P1.")

        print("[Controller] Participant defaulting to P1.")
        return "P1"

    def _initialize_calibration_state(self):
        participant_number = self._participant_token_to_int(self.participant_id)
        mvc_selection = FileSelection(data_type="collected", participant=participant_number, movement=10)
        self.mvc_file_path = self.repo.raw_file_path(mvc_selection)

        if not os.path.exists(self.mvc_file_path):
            self.is_calibrated = False
            print(f"[Controller] State: UNCALIBRATED (missing MVC file: {self.mvc_file_path})")
            return

        try:
            baseline, max_vals = SignalProcessing.compute_participant_minmax(
                mvc_file_path=self.mvc_file_path,
                fs=Config.FS,
                percentiles=(1.0, 99.0),
                expected_channels=Config.NUM_CHANNELS,
            )
        except Exception as e:
            self.is_calibrated = False
            print(f"[Controller] MVC file found but calibration failed: {e}")
            print("[Controller] State: UNCALIBRATED")
            return

        self.norm_baseline = baseline.astype(np.float32)
        self.norm_max = max_vals.astype(np.float32)
        self.is_calibrated = True
        print(f"[Controller] State: CALIBRATED using MVC file: {self.mvc_file_path}")

    def read_new_samples(self, num_samples):
        """
        Read new samples from hardware or simulation.
        
        Args:
            num_samples (int): Number of samples to read (Config.INCREMENT)
            
        Returns:
            np.ndarray: Shape (Config.NUM_CHANNELS, num_samples) array of ADC values
        """
        
        if not (self.use_hardware or self.simulate_data):
            return np.random.randn(Config.NUM_CHANNELS, num_samples) * 0.1
        
        if self.use_hardware:
            # Read from hardware via reading mode
            chunk = self.reading_mode.read_sample_chunk()
            return chunk

        # If simulating, read from pre-loaded .mat data stream
        if self.sim_data_stream is None:
            return np.random.randn(Config.NUM_CHANNELS, num_samples) * 0.1
        
        end_idx = self.sim_playback_idx + num_samples

        if end_idx > self.sim_data_stream.shape[1]:
            self.sim_playback_idx = 0
            end_idx = num_samples
        chunk = self.sim_data_stream[:, self.sim_playback_idx:end_idx]
        self.sim_playback_idx = end_idx
        return chunk

    @staticmethod
    def _extract_robust_minmax(mat_data):
        for key in ("MIN_MAX_ROBUST", "MIN_MAX"):
            if key not in mat_data:
                continue
            matrix = np.asarray(mat_data[key], dtype=np.float32)
            if matrix.ndim != 2:
                continue
            if matrix.shape == (Config.NUM_CHANNELS, 2):
                return matrix
            if matrix.shape == (2, Config.NUM_CHANNELS):
                return matrix.T
        return None

    @staticmethod
    def _to_labelled_candidate(path):
        norm_path = os.path.normpath(path)
        if not norm_path.lower().endswith(".mat"):
            return norm_path
        if norm_path.lower().endswith("_labelled.mat"):
            return norm_path

        base, _ = os.path.splitext(norm_path)
        labelled = base + "_labelled.mat"
        raw_segment = f"{os.sep}raw{os.sep}"
        edited_segment = f"{os.sep}edited{os.sep}"
        if raw_segment in labelled:
            labelled = labelled.replace(raw_segment, edited_segment)
        return labelled

    def _try_load_minmax_from_path(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return None
        try:
            mat = scipy.io.loadmat(file_path)
        except Exception:
            return None
        return self._extract_robust_minmax(mat)

    def _load_runtime_minmax(self, sim_file=None):
        candidates = []

        if sim_file:
            candidates.append(os.path.normpath(sim_file))
            candidates.append(self.repo.labelled_candidate_path(sim_file))

        candidates.extend(
            [
                self.repo.output_file_path(FileSelection(data_type="collected", participant=1, movement=1), create_dirs=False),
                self.repo.output_file_path(FileSelection(data_type="secondary", participant=1, movement=1), create_dirs=False),
            ]
        )

        seen = set()
        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            matrix = self._try_load_minmax_from_path(candidate)
            if matrix is not None:
                print(f"[Controller] Loaded MIN_MAX_ROBUST from: {candidate}")
                return matrix

        for edited_root in (self.repo.edited_root("secondary"), self.repo.edited_root("collected")):
            if not os.path.isdir(edited_root):
                continue

            for root, _, files in os.walk(edited_root):
                for name in files:
                    if not name.lower().endswith("_labelled.mat"):
                        continue
                    candidate = os.path.join(root, name)
                    matrix = self._try_load_minmax_from_path(candidate)
                    if matrix is not None:
                        print(f"[Controller] Loaded MIN_MAX_ROBUST from: {candidate}")
                        return matrix

        raise FileNotFoundError(
            "No readable MIN_MAX_ROBUST found in labelled files. "
            "Run the Compute Min/Max action in LabelSignalData first."
        )

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
                elif "CMD:ACK" in command:
                    print("[UDP Listener] Unity ACK received.")
                    self.ack_received.set()

            except BlockingIOError:
                # No data available, continue checking
                time.sleep(0.01)
            except Exception as e:
                print(f"[UDP Listener] Error receiving command: {e}")
                time.sleep(0.1)

    def _send_command(self, command):
        """Sends a UDP command to Unity (Config.UDP_IP:Config.UDP_PORT)."""
        try:
            self.sock.sendto(command.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
            print(f"[Controller] UDP command sent: {command}")
        except Exception as e:
            print(f"[Controller] WARNING: Could not send UDP command: {e}")

    def _is_collection_mode(self):
        """Duck-type check that avoids importing DataCollectionMode at module load."""
        return self.use_hardware and self.reading_mode is not None and hasattr(self.reading_mode, 'save_collection')

    def _start_stdin_watcher(self):
        """Background thread reads stdin; sets self.exit_requested on 'finish'/'exit'."""
        def watch_stdin():
            print("[Controller] Stdin watcher active. Type 'finish' or 'exit' to stop.")
            while self.running:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    return
                if not line:
                    return
                cmd = line.strip().lower()
                if cmd in ('finish', 'finished', 'exit', 'quit'):
                    print(f"[Controller] '{cmd}' received. Will exit at next safe checkpoint.")
                    self.exit_requested = True
                    return

        thread = threading.Thread(target=watch_stdin, daemon=True)
        thread.start()

    def _save_trial(self, movement_id):
        """Saves the current trial buffer to biosignal_data/collected/raw/P{id}M{m}.mat."""
        if not self._is_collection_mode():
            return None

        participant_num = self._participant_token_to_int(self.participant_id)
        if participant_num is None:
            print("[Controller] WARNING: Could not resolve participant ID for save. Falling back to default name.")
            return self.reading_mode.save_collection()

        selection = FileSelection(data_type="collected", participant=participant_num, movement=movement_id)
        save_path = self.repo.raw_file_path(selection)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return self.reading_mode.save_collection(output_path=save_path)

    def _prompt_movement_or_finish(self):
        """Displays the movement menu and reads a selection. Returns movement_id or None for 'finish'."""
        print("\n" + "=" * 70)
        print("DATA COLLECTION SESSION — SELECT MOVEMENT")
        print("=" * 70)
        print(f"Participant: {self.participant_id}")
        print(f"Calibration: {'OK' if self.is_calibrated else 'UNCALIBRATED — collect M10 to calibrate.'}")
        print("-" * 70)
        print(f"{'ID':<5} {'Movement':<30} {'Kinematics':<35}")
        print("-" * 70)
        for movement_id in sorted(Config.MOVEMENT_NAMES.keys()):
            movement_name = Config.MOVEMENT_NAMES[movement_id]
            kinematics = Config.TARGET_MAPPING[movement_id]
            kinematics_str = f"[Yaw={kinematics[0]:.0f}°, Pitch={kinematics[1]:.0f}°, Roll={kinematics[2]:.0f}°, Elbow={kinematics[3]:.0f}°]"
            print(f"{movement_id:<5} {movement_name:<30} {kinematics_str:<35}")
        print(f"{10:<5} {'MVC Trial (auto-recalibrate)':<30}")
        print("-" * 70)

        while True:
            try:
                user_input = input("Enter movement ID (1-10) or 'finish': ").strip().lower()
            except EOFError:
                print("[Controller] EOF on stdin. Ending session.")
                return None
            except KeyboardInterrupt:
                print("\n[Controller] Interrupted during selection. Ending session.")
                return None

            if user_input in ('finish', 'finished', 'exit', 'quit'):
                return None
            try:
                movement_id = int(user_input)
            except ValueError:
                print(f"[ERROR] Invalid input: '{user_input}'. Try again.")
                continue
            if movement_id == 10 or movement_id in Config.MOVEMENT_NAMES:
                return movement_id
            print(f"[ERROR] Movement ID {movement_id} not recognized.")

    def _stream_one_iteration(self):
        """Reads one INCREMENT chunk, runs the model (after warmup), pushes to UDP/queue.

        Returns True if a prediction was produced this iteration, False during warmup.
        Used by both continuous and recording loops.
        """
        new_data = self.read_new_samples(Config.INCREMENT)

        self.data_buffer = np.roll(self.data_buffer, -Config.INCREMENT, axis=1)
        self.data_buffer[:, -Config.INCREMENT:] = new_data
        self.real_samples_seen += Config.INCREMENT

        if self.real_samples_seen < self.warmup_samples_required:
            return False

        rectified_window = np.zeros_like(self.data_buffer)
        for i in range(Config.NUM_CHANNELS):
            rectified_window[i, :] = SignalProcessing.applyStandardSEMGProcessing(
                self.data_buffer[i, :],
                fs=Config.FS,
            )

        if self.is_calibrated:
            denominator = (self.norm_max - self.norm_baseline)[:, np.newaxis] + 1e-8
            emg_window = (rectified_window - self.norm_baseline[:, np.newaxis]) / denominator
        else:
            emg_window = rectified_window

        input_tensor = torch.tensor(emg_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_predictions = self.model(input_tensor).cpu().numpy()[0]

        self.smoothed_output = (self.alpha * raw_predictions) + ((1 - self.alpha) * self.smoothed_output)
        yaw, pitch, roll, elbow = self.smoothed_output

        self.yaw_history = np.roll(self.yaw_history, -1)
        self.yaw_history[-1] = yaw
        self.pitch_history = np.roll(self.pitch_history, -1)
        self.pitch_history[-1] = pitch

        packet_string = f"{yaw:.2f},{pitch:.2f},{roll:.2f},{elbow:.2f}"
        try:
            self.sock.sendto(packet_string.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
        except Exception:
            pass

        if self.data_queue is not None:
            try:
                self.data_queue.put_nowait({
                    'emg': emg_window.copy(),
                    'pred': self.smoothed_output.copy(),
                    'yaw_history': self.yaw_history.copy(),
                    'pitch_history': self.pitch_history.copy(),
                })
            except queue.Full:
                pass

        return True

    def _run_continuous_loop(self):
        """Streams predictions to Unity continuously until 'finish'/'exit' is typed (or signal)."""
        print("[Continuous] Streaming. Type 'finish' or 'exit' (then Enter) to stop.\n")
        increment_seconds = Config.INCREMENT / 1000.0
        warmup_announced = self.warmup_samples_required <= 0

        while self.running and not self.exit_requested:
            had_prediction = self._stream_one_iteration()
            if had_prediction and not warmup_announced:
                print(f"[Controller] Warmup complete ({self.warmup_samples_required} samples). Predictions live.")
                warmup_announced = True
            time.sleep(increment_seconds)

        print("[Continuous] Stream stopped.")

    def _run_recording_loop(self):
        """Streams during a single trial until Unity sends CMD:STOP (or session exit)."""
        print("[Controller] Recording. Streaming EMG to Unity. Awaiting CMD:STOP from Unity...")
        self.recording_active = True
        self.stop_recording_requested = False
        self.real_samples_seen = 0  # Restart warmup per trial so settling is per-trial
        self.data_buffer[:] = 0.0   # Clear stale samples between trials
        increment_seconds = Config.INCREMENT / 1000.0
        warmup_announced = self.warmup_samples_required <= 0

        while self.running and self.recording_active and not self.exit_requested:
            if self.stop_recording_requested:
                print("[Controller] STOP received. Ending recording.")
                break

            had_prediction = self._stream_one_iteration()
            if had_prediction and not warmup_announced:
                print(f"[Controller] Warmup complete ({self.warmup_samples_required} samples). Predictions live.")
                warmup_announced = True
            time.sleep(increment_seconds)

        self.recording_active = False
        self.stop_recording_requested = False

    def _run_collection_session(self):
        """Outer session loop: prompt → ACK handshake → countdown → record → save → repeat."""
        print("\n" + "=" * 70)
        print(f"DATA COLLECTION SESSION — Participant {self.participant_id}")
        print("=" * 70)
        print("Pick a movement per trial. Type 'finish' to end the session.\n")

        while self.running and not self.exit_requested:
            movement_id = self._prompt_movement_or_finish()
            if movement_id is None:
                print("[Session] Finish requested. Ending session.")
                break

            # Send GO and wait for Unity ACK so countdowns line up
            self.ack_received.clear()
            self._send_command(f"CMD:GO_RECORDING:{movement_id}")
            print("[Controller] Waiting for Unity ACK (timeout 5s)...")
            ack_ok = self.ack_received.wait(timeout=5.0)
            if not ack_ok:
                print("[WARNING] No ACK from Unity. Proceeding without sync — check that Unity is running.")
            else:
                print("[Controller] ACK received. Starting synchronized countdown.")

            # 3s countdown — Unity runs its own 3s 'Syncing...' loop in parallel
            for countdown in range(3, 0, -1):
                print(f"  {countdown}...", end=" ", flush=True)
                time.sleep(1.0)
            print("GO!")

            # Stream until Unity STOP (or session exit)
            self.reading_mode.clear_collection()
            self._run_recording_loop()

            # Save the trial buffer to P{id}M{m}.mat
            try:
                save_path = self._save_trial(movement_id)
                if save_path:
                    self.session_save_paths.append(save_path)
                    print(f"[Controller] Trial saved: {save_path}")
            except Exception as e:
                print(f"[Controller] ERROR saving trial: {e}")

            # If MVC trial just finished, refresh calibration from the new file
            if movement_id == 10:
                print("[Controller] M10 saved — re-initializing calibration with the fresh MVC.")
                self._initialize_calibration_state()

            print("[Session] Returning to menu.\n")

        # Session over — summary
        print("\n" + "=" * 70)
        print("SESSION ENDED")
        print(f"Trials saved this session: {len(self.session_save_paths)}")
        for path in self.session_save_paths:
            print(f"  - {path}")
        print("=" * 70)

    def run_forever(self):
        """Entry point: dispatches to continuous loop or collection session loop."""
        mode_desc = "Hardware" if self.use_hardware else "Simulation" if self.simulate_data else "Random Noise"
        print(f"\n[Controller] Starting Real-Time Control ({mode_desc} mode).")

        self.running = True

        def handle_sigint(sig, frame):
            print("\n[Controller] SIGINT received. Initiating graceful shutdown...")
            self.running = False
            self.exit_requested = True

        def handle_sigterm(sig, frame):
            print("\n[Controller] SIGTERM received. Initiating graceful shutdown...")
            self.running = False
            self.exit_requested = True

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, handle_sigint)
            signal.signal(signal.SIGTERM, handle_sigterm)

        # Stdin watcher works for both continuous and collection modes
        self._start_stdin_watcher()

        try:
            if self._is_collection_mode():
                self._run_collection_session()
            else:
                self._run_continuous_loop()
            return 0
        except KeyboardInterrupt:
            print("\n[Controller] Keyboard interrupt")
            return 0
        except Exception as e:
            print(f"\n[Controller] Error in main loop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.running = False

    def run(self):
        """Backward-compatible wrapper."""
        return self.run_forever()


class SignalViewerGUI(_QMainWindowBase):
    """
    Standalone GUI consumer for EMG + Kinematics visualization.
    Receives data from Controller via a thread-safe queue.Queue.
    Runs on the main thread with PyQt event loop.
    """
    
    def __init__(self, data_queue):
        """
        Initialize the GUI consumer.
        
        Args:
            data_queue (queue.Queue): Thread-safe queue with maxsize=2 containing dicts:
                                     {'emg': ndarray, 'pred': ndarray, 'yaw_history': ndarray, 'pitch_history': ndarray}
        """
        if QtWidgets is None or QtCore is None or pg is None:
            raise RuntimeError("GUI dependencies are unavailable. Install PyQtGraph/Qt or run without --gui.")

        super().__init__()
        self.data_queue = data_queue
        
        # Visualization buffers
        self.vis_window_size = int(Config.VIS_WINDOW_SIZE_MS / 1000.0 * Config.FS)
        self.vis_buffer = np.zeros((Config.NUM_CHANNELS, self.vis_window_size))
        
        # Y-axis scaling tracking
        self.channel_mins = np.full(Config.NUM_CHANNELS, np.inf)
        self.channel_maxs = np.full(Config.NUM_CHANNELS, -np.inf)
        self.plot_y_ranges = {}
        
        # GUI setup
        self.setup_ui()
        
        # Timer to poll queue at 33ms (~30 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_and_update)
        self.timer.start(33)
    
    def setup_ui(self):
        """Create PyQtGraph plots for EMG and Kinematics."""
        self.setWindowTitle("EMG & Kinematics Viewer")
        self.showFullScreen()
        
        # Central widget with GraphicsLayout
        self.central_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.central_widget)
        
        self.plots_emg = []
        self.curves_emg = []
        
        Y_MIN = -0.0001
        Y_MAX = 0.0001
        self.ref_viewbox = None
        
        # Create EMG plots (8 rows)
        for i in range(Config.NUM_CHANNELS):
            p = self.central_widget.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', Config.CHANNEL_MAP.get(i, f"Ch {i}"), width=80)
            
            p.setYRange(Y_MIN, Y_MAX, padding=0)
            p.disableAutoRange(axis=pg.ViewBox.YAxis)
            
            if self.ref_viewbox is None:
                self.ref_viewbox = p.getViewBox()
            else:
                p.setXLink(self.ref_viewbox)
            
            if i < Config.NUM_CHANNELS - 1:
                p.hideAxis('bottom')
            
            curve = p.plot(pen=pg.mkPen(color=(50, 150, 255), width=1))
            self.plots_emg.append(p)
            self.curves_emg.append(curve)
        
        # Create Kinematics plot (row 8)
        self.plot_kin = self.central_widget.addPlot(row=8, col=0)
        self.plot_kin.showGrid(x=True, y=True, alpha=0.5)
        self.plot_kin.setLabel('left', 'Angle (Degrees)', width=80)
        self.plot_kin.setLabel('bottom', 'Time (ms)')
        
        self.plot_kin.setXRange(0, 1500, padding=0)
        self.plot_kin.disableAutoRange(axis=pg.ViewBox.XAxis)
        
        self.plot_kin.setYRange(-40, 130, padding=0)
        self.plot_kin.disableAutoRange(axis=pg.ViewBox.YAxis)
        
        self.plot_kin.addLegend(offset=(10, 10))
        
        self.curve_yaw = self.plot_kin.plot(pen=pg.mkPen('r', width=3), name='Yaw (Flexion)')
        self.curve_pitch = self.plot_kin.plot(pen=pg.mkPen('c', width=3), name='Pitch (Abduction)')
    
    def poll_and_update(self):
        """
        Poll the queue for producer data and update all plots.
        Safely drains queue, keeping only the newest data.
        """
        latest_data = None
        
        # Drain queue, keeping only newest data
        try:
            while True:
                latest_data = self.data_queue.get_nowait()
        except queue.Empty:
            pass
        
        # If we got any data, update plots
        if latest_data is not None:
            self.update_visualization(latest_data)
    
    def update_visualization(self, data_dict):
        """Update all plots with latest data from producer."""
        emg_window = data_dict['emg']
        smoothed_output = data_dict['pred']
        yaw_history = data_dict['yaw_history']
        pitch_history = data_dict['pitch_history']
        
        # Update visualization buffer
        self.vis_buffer = np.roll(self.vis_buffer, -Config.INCREMENT, axis=1)
        new_data = emg_window[:, -Config.INCREMENT:]
        self.vis_buffer[:, -Config.INCREMENT:] = new_data
        
        # Update EMG curves
        for i in range(Config.NUM_CHANNELS):
            self.curves_emg[i].setData(self.vis_buffer[i])
        
        # Update Y-axis scaling
        y_range_updated = False
        for i in range(Config.NUM_CHANNELS):
            channel_data = self.vis_buffer[i]
            channel_min = np.min(channel_data)
            channel_max = np.max(channel_data)
            
            if channel_min < self.channel_mins[i]:
                self.channel_mins[i] = channel_min
                y_range_updated = True
            if channel_max > self.channel_maxs[i]:
                self.channel_maxs[i] = channel_max
                y_range_updated = True
        
        if y_range_updated:
            for i in range(Config.NUM_CHANNELS):
                range_span = self.channel_maxs[i] - self.channel_mins[i]
                padding = max(range_span * 0.1, 1e-5)
                
                new_y_min = self.channel_mins[i] - padding
                new_y_max = self.channel_maxs[i] + padding
                
                if i not in self.plot_y_ranges or \
                   abs(self.plot_y_ranges[i][0] - new_y_min) > 1e-6 or \
                   abs(self.plot_y_ranges[i][1] - new_y_max) > 1e-6:
                    self.plots_emg[i].setYRange(new_y_min, new_y_max, padding=0)
                    self.plot_y_ranges[i] = (new_y_min, new_y_max)
        
        # Update kinematics plots
        kin_time_values = np.arange(len(yaw_history)) * Config.INCREMENT
        self.curve_yaw.setData(kin_time_values, yaw_history)
        self.curve_pitch.setData(kin_time_values, pitch_history)

    def closeEvent(self, event):
        """Stop timer when window closes."""
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time EMG Prosthetic Controller")
    
    # Mode selection
    parser.add_argument('--hardware', action='store_true', help='Use hardware (MCP3008) for real-time reading')
    parser.add_argument('--continuous', action='store_true', help='Continuous reading mode (hardware only)')
    parser.add_argument('--collect', action='store_true', help='Data collection mode (hardware only) - stores data for later')
    parser.add_argument('--collection_name', type=str, default='hardware_trial', help='Name for collected data (used with --collect)')
    parser.add_argument('--simulate', action='store_true', help='Use simulation mode with sample .mat file')
    parser.add_argument('--gui', action='store_true', default=False, help='Enable GUI visualization (default: False for headless)')
    parser.add_argument('--participant', type=str, default=None, help='Participant ID (e.g., P1 or 1). Overrides current_participant.txt.')
    
    # Model and file paths
    parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH, help='Path to trained PyTorch weights')
    parser.add_argument('--sim_file', type=str, default='./secondary_data/Soggetto1/Movimento3.mat', help='Specific .mat file to stream (simulation mode)')
    
    args = parser.parse_args()
    
    # === CONFIGURE READING MODE ===
    reading_mode = None
    
    if args.hardware:
        from SignalReading import ContinuousReadingMode, DataCollectionMode
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
    
    exit_code = 0
    controller = None
    
    try:
        if args.gui:
            # === GUI MODE: Producer-Consumer Pattern ===
            print("[Main] GUI mode enabled. Starting Controller in background thread...")

            if QtWidgets is None or QtCore is None or pg is None:
                raise RuntimeError("GUI mode requested but PyQtGraph/Qt is not available in this environment.")
            
            # Create thread-safe queue
            data_queue = queue.Queue(maxsize=2)
            
            # Create controller with queue
            controller = RealTimeProstheticController(
                model_path=args.model, 
                reading_mode=reading_mode,
                simulate_data=args.simulate, 
                sim_file=args.sim_file,
                data_queue=data_queue,  # Pass queue to producer
                participant_id=args.participant,
            )
            
            # Start controller in daemon thread
            controller_thread = threading.Thread(target=controller.run_forever, daemon=True)
            controller_thread.start()
            
            # Create GUI on main thread
            app = QtWidgets.QApplication(sys.argv)
            gui = SignalViewerGUI(data_queue)
            gui.show()
            
            # Run Qt event loop on main thread
            exit_code = app.exec_()
            
        else:
            # === HEADLESS MODE: Direct Controller ===
            print("[Main] Headless mode (no GUI).")
            
            # Create controller without queue
            controller = RealTimeProstheticController(
                model_path=args.model, 
                reading_mode=reading_mode,
                simulate_data=args.simulate, 
                sim_file=args.sim_file,
                data_queue=None,  # No queue in headless mode
                participant_id=args.participant,
            )
            
            # Run on main thread
            exit_code = controller.run_forever()
    
    except Exception as e:
        print(f"\n[Main] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    finally:
        # === CLEANUP RESOURCES ===
        print("\n[Main] Cleaning up resources...")
        
        if controller is not None:
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
        
        # Per-trial saves are handled inside the session loop now; no end-of-run save.
        if args.hardware and args.collect and controller is not None:
            saved = getattr(controller, 'session_save_paths', [])
            print(f"\n[Main] Session Summary: {len(saved)} trial file(s) saved this session.")
            for path in saved:
                print(f"   - {path}")

        print("[Main] Shutdown complete\n")
        sys.exit(exit_code)
