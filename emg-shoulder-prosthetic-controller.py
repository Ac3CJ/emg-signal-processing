import time
import socket
import numpy as np
import torch
import argparse

import SignalProcessing
from ModelTraining import ShoulderRCNN  # Importing the PyTorch model we created
import ControllerConfiguration as Config

# --- Network Configuration (Matches your virt-env-tester.py) ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

class RealTimeProstheticController:
    def __init__(self, model_path='best_shoulder_rcnn.pth', simulate_data=False):
        """
        Initializes the real-time controller, loads the PyTorch model, 
        and sets up the UDP telemetry socket.
        """
        self.simulate_data = simulate_data
        
        # 1. Setup UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP Socket initialized. Target: {UDP_IP}:{UDP_PORT}")
        
        # 2. Setup Device & Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading PyTorch model on: {self.device}")
        
        # Initialize model (Note: Ensure ModelTraining.py is updated to output 4 features)
        self.model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Set to evaluation mode (disables dropout, etc.)
            print(f"Successfully loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"WARNING: '{model_path}' not found. Using untrained initialized weights for debugging.")
        
        # 3. Initialize the Sliding Window Buffer
        # Shape: (8 channels, 500 samples)
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, Config.WINDOW_SIZE))
        
        # Smoothing filter for output kinematics (Moving average to prevent jitter)
        self.alpha = 0.3  # Smoothing factor (0.0 to 1.0)
        self.smoothed_output = np.zeros(Config.NUM_OUTPUTS)

    def read_new_samples(self, num_samples):
        """
        Reads new data from the hardware sensors. 
        If simulating, generates random noise.
        """
        if self.simulate_data:
            # Generate dummy sEMG data for all 8 channels
            return np.random.randn(Config.NUM_CHANNELS, num_samples) * 0.1
        else:
            # TODO: Phase 3 - Replace this with your actual SensorReader.py logic
            # e.g., return SensorReader.get_latest_samples(num_samples)
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
                self.sock.sendto(packet_string.encode('utf-8'), (UDP_IP, UDP_PORT))
                
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
    parser.add_argument('--model', type=str, default='best_shoulder_rcnn.pth', help='Path to the trained PyTorch weights')
    
    args = parser.parse_args()
    
    # Instantiate and run the controller
    controller = RealTimeProstheticController(model_path=args.model, simulate_data=args.simulate)
    controller.run_control_loop()