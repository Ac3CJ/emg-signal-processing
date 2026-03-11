import sys
import time
import socket
import numpy as np
import torch
import argparse
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import SignalProcessing
from ModelTraining import ShoulderRCNN 
import ControllerConfiguration as Config

class RealTimeProstheticController:
    def __init__(self, model_path='best_shoulder_rcnn.pth', simulate_data=False, sim_file=None):
        self.simulate_data = simulate_data
        
        # --- LOAD SIMULATION DATA ---
        if self.simulate_data and sim_file:
            import scipy.io
            import os
            if os.path.exists(sim_file):
                print(f"Loading simulation stream from: {sim_file}")
                mat = scipy.io.loadmat(sim_file)
                self.sim_data_stream = mat['EMGDATA'] # Shape: (8, total_samples)
                self.sim_playback_idx = 0
            else:
                print(f"ERROR: Could not find {sim_file}. Falling back to random noise.")
                self.sim_data_stream = None
        else:
            self.sim_data_stream = None

        # 1. Setup UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP Socket initialized. Target: {Config.UDP_IP}:{Config.UDP_PORT}")
        
        # 2. Setup Device & Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"WARNING: '{model_path}' not found. Using untrained weights.")
        
        # 3. Initialize Data Buffers
        # data_buffer is exactly 500ms long for the Neural Network
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, Config.WINDOW_SIZE))
        
        # vis_buffer is 3 windows long (1500ms) for the GUI
        self.vis_window_size = Config.WINDOW_SIZE * 3 
        self.vis_buffer = np.zeros((Config.NUM_CHANNELS, self.vis_window_size))
        
        self.alpha = Config.SMOOTHING_ALPHA
        self.smoothed_output = np.zeros(Config.NUM_OUTPUTS)

        # 4. Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Initializes the PyQtGraph visualizer."""
        self.app = QtWidgets.QApplication(sys.argv)
        
        # Create the main window
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time sEMG Stream")
        self.win.resize(1000, 800)
        self.win.setWindowTitle('Non-Invasive Prosthetic Controller - Live Feed')
        
        self.plots = []
        self.curves = []
        
        # Create 8 stacked plots
        for i in range(Config.NUM_CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', Config.CHANNEL_MAP.get(i, f"Ch {i}"))
            
            # Hide X-axis on all but the bottom plot for a cleaner look
            if i < Config.NUM_CHANNELS - 1:
                p.hideAxis('bottom')
            
            # Add the red dotted line 500ms from the right edge
            vLine = pg.InfiniteLine(angle=90, movable=False, pos=self.vis_window_size - Config.WINDOW_SIZE)
            
            # Cross-compatible Qt DashLine check (PyQt6 vs PyQt5)
            try:
                dash_style = QtCore.Qt.PenStyle.DashLine
            except AttributeError:
                dash_style = QtCore.Qt.DashLine
                
            vLine.setPen(pg.mkPen(color='r', style=dash_style, width=2))
            p.addItem(vLine)
            
            # Create the curve object we will update rapidly
            curve = p.plot(pen=pg.mkPen(color=(50, 150, 255), width=1))
            
            self.plots.append(p)
            self.curves.append(curve)

    def read_new_samples(self, num_samples):
        """
        Reads new data from the hardware sensors. 
        If simulating, generates random noise.
        """
        if self.simulate_data:
            if self.sim_data_stream is not None:
                # Stream the real .mat file
                end_idx = self.sim_playback_idx + num_samples
                
                # Loop back to the start if we hit the end of the file
                if end_idx > self.sim_data_stream.shape[1]:
                    self.sim_playback_idx = 0
                    end_idx = num_samples
                    
                chunk = self.sim_data_stream[:, self.sim_playback_idx:end_idx]
                self.sim_playback_idx = end_idx
                return chunk
            else:
                # Fallback random noise
                return np.random.randn(Config.NUM_CHANNELS, num_samples) * 0.1
        else:
            # TODO: Hardware code goes here later
            pass

    def control_step(self):
        """
        This replaces the while loop. It is triggered every 62ms by the QTimer.
        """
        # 1. Fetch new data
        new_data = self.read_new_samples(Config.INCREMENT)
        
        # 2. Update Neural Network Buffer (500 samples)
        self.data_buffer = np.roll(self.data_buffer, -Config.INCREMENT, axis=1)
        self.data_buffer[:, -Config.INCREMENT:] = new_data
        
        # 3. Update Visual Buffer (1500 samples)
        self.vis_buffer = np.roll(self.vis_buffer, -Config.INCREMENT, axis=1)
        self.vis_buffer[:, -Config.INCREMENT:] = new_data
        
        # 4. Update the GUI Curves
        for i in range(Config.NUM_CHANNELS):
            self.curves[i].setData(self.vis_buffer[i])
        
        # 5. Clean the 500ms window for the NN
        cleaned_window = np.zeros_like(self.data_buffer)
        for i in range(Config.NUM_CHANNELS):
            cleaned_window[i, :] = SignalProcessing.applyStandardSEMGProcessing(self.data_buffer[i, :], fs=Config.FS)
        
        # 6. Neural Network Inference
        input_tensor = torch.tensor(cleaned_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_predictions = self.model(input_tensor).cpu().numpy()[0]
        
        # 7. Kinematic Smoothing
        self.smoothed_output = (self.alpha * raw_predictions) + ((1 - self.alpha) * self.smoothed_output)
        yaw, pitch, roll, elbow = self.smoothed_output
        
        # 8. Send Telemetry
        packet_string = f"{yaw:.2f},{pitch:.2f},{roll:.2f},{elbow:.2f}"
        print(f"Sending Telemetry: {packet_string}")
        try:
            self.sock.sendto(packet_string.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
        except Exception as e:
            pass # Ignore UDP errors so it doesn't crash the loop

    def run(self):
        """Starts the Qt Event Loop and the hardware timer."""
        print("\nStarting Real-Time Control GUI. Close the window to stop.")
        
        # Create a QTimer that triggers control_step every 62ms
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.control_step)
        self.timer.start(Config.INCREMENT)
        
        # Start the GUI event loop
        try:
            sys.exit(self.app.exec())
        finally:
            self.sock.close()
            print("UDP Socket closed safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time EMG Prosthetic Controller")
    parser.add_argument('--simulate', action='store_true', help='Generate dummy sEMG data instead of reading hardware')
    parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH, help='Path to the trained PyTorch weights')
    
    # NEW ARGUMENT: Allows you to pick exactly which file to test!
    parser.add_argument('--sim_file', type=str, default='./secondary_data/Soggetto1/Movimento3.mat', help='Specific .mat file to stream during simulation')
    
    args = parser.parse_args()
    
    controller = RealTimeProstheticController(
        model_path=args.model, 
        simulate_data=args.simulate,
        sim_file=args.sim_file
    )
    controller.run()