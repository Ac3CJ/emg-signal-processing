import time
import socket
import numpy as np
import torch
import argparse

import SignalProcessing
from ModelTraining import ShoulderRCNN  # Importing the PyTorch model we created
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
        
        # 3. Initialize the Sliding Window Buffer
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, Config.WINDOW_SIZE))
        self.alpha = Config.SMOOTHING_ALPHA
        self.smoothed_output = np.zeros(Config.NUM_OUTPUTS)

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

    def run_control_loop(self):
        """
        The main loop that updates the buffer, processes the signal, 
        runs the inference, and sends telemetry.
        """
        print("\nStarting Real-Time Control Loop. Press Ctrl+C to stop.")
        
        try:
            while True:
                loop_start_time = time.time()
                
                # 1. Fetch new data (62 samples for the 62ms increment)
                new_data = self.read_new_samples(Config.INCREMENT)
                
                # 2. Shift the buffer left and append new data to the right
                self.data_buffer = np.roll(self.data_buffer, -Config.INCREMENT, axis=1)
                self.data_buffer[:, -Config.INCREMENT:] = new_data
                
                # 3. Clean the 500ms window using your SignalProcessing.py pipeline
                cleaned_window = np.zeros_like(self.data_buffer)
                for i in range(Config.NUM_CHANNELS):
                    cleaned_window[i, :] = SignalProcessing.applyStandardSEMGProcessing(self.data_buffer[i, :], fs=Config.FS)
                
                # 4. Neural Network Inference
                # Convert (8, 500) numpy array to PyTorch Tensor shape (1, 8, 500) [Batch=1]
                input_tensor = torch.tensor(cleaned_window, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Output shape will be (1, 4) -> [Yaw, Pitch, Roll, Elbow]
                    raw_predictions = self.model(input_tensor).cpu().numpy()[0]
                
                # 5. Kinematic Smoothing (Exponential Moving Average)
                self.smoothed_output = (self.alpha * raw_predictions) + ((1 - self.alpha) * self.smoothed_output)
                
                # Extract values
                yaw, pitch, roll, elbow = self.smoothed_output
                
                # 6. Send Telemetry to Virtual Environment
                packet_string = f"{yaw:.2f},{pitch:.2f},{roll:.2f},{elbow:.2f}"
                print(f"Sending Telemetry: {packet_string}")

                self.sock.sendto(packet_string.encode('utf-8'), (Config.UDP_IP, Config.UDP_PORT))
                
                # 7. Maintain Real-Time constraints (Wait until 62ms have passed)
                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0, (Config.INCREMENT / 1000.0) - elapsed_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nControl loop stopped by user.")
        finally:
            self.sock.close()
            print("UDP Socket closed.")

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
    controller.run_control_loop()