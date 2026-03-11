"""
ControllerConfiguration.py
Centralized configuration file for the Non-Invasive Shoulder Prosthetic Project.
All hyperparameters, network settings, and hardware constraints are defined here.
"""

# ====================================================================================
# 1. HARDWARE & ACQUISITION SETTINGS
# ====================================================================================
FS = 1000.0                 # Sampling rate in Hz (1.0 kHz based on Rivela et al.)
NUM_CHANNELS = 8            # Number of MyoWare/sEMG sensors
NUM_OUTPUTS = 4             # Output DOFs: [Yaw, Pitch, Roll, Elbow]

# ====================================================================================
# 2. SLIDING WINDOW & REAL-TIME CONSTRAINTS
# ====================================================================================
WINDOW_SIZE = 500           # 500 ms window size
INCREMENT = 62              # 62 ms step size (~16 Hz update rate)
SMOOTHING_ALPHA = 0.3       # Exponential moving average factor for kinematic output (0.0 to 1.0)

# ====================================================================================
# 3. SIGNAL PROCESSING (FILTERING)
# ====================================================================================
NOTCH_FREQ = 50.0           # UK Powerline noise frequency
NOTCH_QUALITY = 30.0        # Quality factor for the notch filter
BANDPASS_LOW = 30.0         # Lower cutoff (30Hz recommended for ECG artifact removal)
BANDPASS_HIGH = 450.0       # Upper cutoff (Nyquist is 500Hz)
FILTER_ORDER = 4            # Butterworth filter order

# ====================================================================================
# 4. NEURAL NETWORK TRAINING PARAMETERS
# ====================================================================================
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10               # Early stopping patience
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2            # 20% of data used for validation

# ====================================================================================
# 5. NETWORKING & TELEMETRY
# ====================================================================================
UDP_IP = "127.0.0.1"        # Localhost for virtual environment testing
UDP_PORT = 5005             # Port listening on the Unity/Virtual Environment side

# ====================================================================================
# 6. FILE PATHS
# ====================================================================================
BASE_DATA_PATH = './secondary_data'
MODEL_SAVE_PATH = 'best_shoulder_rcnn.pth'

# ====================================================================================
# 7. MAPPINGS
# ====================================================================================
# Maps hardware channel index to the physical muscle
CHANNEL_MAP = {
    0: "Pectoralis Major (Clavicular)",
    1: "Pectoralis Major (Sternal)",
    2: "Serratus Anterior",
    3: "Trapezius (Descendent)",
    4: "Trapezius (Transversalis)",
    5: "Trapezius (Ascendant)",
    6: "Infraspinatus",
    7: "Latissimus Dorsi"
}

# Maps Rivela et al. movement classes to target Kinematics: [Yaw, Pitch, Roll, Elbow]
TARGET_MAPPING = {
    1: [0.0, 0.0, 0.0, 0.0],     # Rest
    2: [0.0, 45.0, 0.0, 0.0],    # Flexion 45
    3: [0.0, 90.0, 0.0, 0.0],    # Flexion 90
    4: [0.0, 110.0, 0.0, 0.0],   # Flexion 110
    5: [0.0, -30.0, 0.0, 0.0],   # Hyperextension -30
    6: [0.0, 0.0, 45.0, 0.0],    # Abduction 45
    7: [0.0, 0.0, 90.0, 0.0],    # Abduction 90
    8: [0.0, 45.0, 45.0, 0.0],   # Elevation 45
    9: [0.0, 90.0, 90.0, 0.0]    # Elevation 90
}