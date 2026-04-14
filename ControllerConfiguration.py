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
WINDOW_SIZE = 100           # 100 ms window size
INCREMENT = 20              # 20 ms step size
SMOOTHING_ALPHA = 0.1       # Exponential moving average factor for kinematic output (0.0 to 1.0)

# Optional training mode: slice windows dynamically from continuous EMG arrays.
# If disabled, training uses pre-windowed tensors (legacy behavior).
ON_THE_FLY_WINDOW_SLICING = True
ON_THE_FLY_WINDOW_SIZE = WINDOW_SIZE
ON_THE_FLY_STEP_SIZE = INCREMENT
ON_THE_FLY_ACTIVE_CHANNELS = None  # Example: [0, 1, 2, 3]

# ====================================================================================
# 2b. VISUALIZATION SETTINGS
# ====================================================================================
VIS_WINDOW_SIZE_MS = 4500   # Visualization window duration in milliseconds (4.5 seconds)
PLOT_EMG_RATIO = 0.8        # Fraction of screen for EMG plots (0.8 = 80% EMG, 20% Kinematics)

# ====================================================================================
# 3. SIGNAL PROCESSING (FILTERING)
# ====================================================================================
NOTCH_FREQ = 50.0           # UK Powerline noise frequency
NOTCH_QUALITY = 30.0        # Quality factor for the notch filter
BANDPASS_LOW = 30.0         # Lower cutoff (30Hz recommended for ECG artifact removal)
BANDPASS_HIGH = 450.0       # Upper cutoff (Nyquist is 500Hz)
FILTER_ORDER = 4            # Butterworth filter order

# ====================================================================================
# 4. DATA AUGMENTATION HYPERPARAMETERS
# ====================================================================================
MIXUP_ALPHA = 0.2           # Alpha parameter for the Beta distribution (controls blend intensity)
MIXUP_RATIO = 0.75           # Ratio of new mixup samples to generate (0.5 = dataset increases by 50%)

# White-noise augmentation applied BEFORE filtering in data-preparation pipelines.
# Each value creates an additional full noisy copy of every trial across all channels.
TRAINING_NOISE_MAGNITUDES = [0.000005, 0.00001]

REST_MIXUP_ALPHA = 0.2           # Alpha parameter for the Beta distribution (controls blend intensity)
REST_MIXUP_RATIO = 0.1           # Ratio of new mixup samples to generate (0.5 = dataset increases by 50%)

# ====================================================================================
# 5. NEURAL NETWORK TRAINING PARAMETERS
# ====================================================================================
EPOCHS = 150
BATCH_SIZE = 256            # Reduced for 16GB RAM constraint (was 1280)
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate 2 batches = effective batch of 1024 without RAM spike
NUM_DATA_WORKERS = 1        # Reduced for RAM (was 4)
PATIENCE = 20              # Early stopping patience
LEARNING_RATE = 0.001
LR_SCHEDULER_FACTOR = 0.5  # Reduce LR by this factor when plateau detected
LR_SCHEDULER_PATIENCE = 5  # Wait this many epochs before reducing LR
TEST_SPLIT = 0.2            # 20% of data used for validation
PREFETCH_FACTOR = 1         # Reduced for RAM (was 2)

# ====================================================================================
# 5b. TRANSFER LEARNING PARAMETERS
# ====================================================================================
TRANSFER_LEARNING_EPOCHS = 75       # Fewer epochs needed for fine-tuning
TRANSFER_LEARNING_LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
TRANSFER_LEARNING_BATCH_SIZE = 128  # Smaller batch size for collected data
TRANSFER_LEARNING_PATIENCE = 20     # Early stopping patience for transfer learning
FREEZE_BACKBONE_LAYERS = True       # Freeze convolutional layers
NUM_LAYERS_TO_UNFREEZE = 2          # Number of final layers to unfreeze for training
TRANSFER_LEARNING_MODEL_SAVE_PATH = 'best_shoulder_rcnn_transfer.pth'

# ====================================================================================
# 6. NETWORKING & TELEMETRY
# ====================================================================================
# IP from Phone's Mobile Network "10.161.179.75", "172.26.212.181", "169.254.128.156"
UDP_IP = "10.161.179.75"
UDP_PORT = 5005             # Port listening on the Unity/Virtual Environment side

# ====================================================================================
# 7. FILE PATHS
# ====================================================================================
BASE_DATA_PATH = './biosignal_data'
SECONDARY_DATA_PATH = f'{BASE_DATA_PATH}/secondary/edited'
COLLECTED_DATA_PATH = f'{BASE_DATA_PATH}/collected/edited'
MODEL_SAVE_PATH = 'best_shoulder_rcnn.pth'

# ====================================================================================
# 8. MAPPINGS
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
# Yaw   = Flexion / Extension
# Pitch = Abduction / Adduction
# Roll  = Internal / External Rotation
# Elbow = Elbow Flexion
TARGET_MAPPING = {
    1: [45.0, 0.0, 0.0, 0.0],      # Flexion 45 (Yaw)
    2: [90.0, 0.0, 0.0, 0.0],      # Flexion 90 (Yaw)
    3: [110.0, 0.0, 0.0, 0.0],     # Flexion 110 (Yaw)
    4: [-30.0, 0.0, 0.0, 0.0],     # Hyperextension -30 (Yaw)
    5: [0.0, 45.0, 0.0, 0.0],      # Abduction 45 (Pitch)
    6: [0.0, 90.0, 0.0, 0.0],      # Abduction 90 (Pitch)
    7: [45.0, 45.0, 0.0, 0.0],     # Elevation 45 (Yaw + Pitch)
    8: [90.0, 90.0, 0.0, 0.0],     # Elevation 90 (Yaw + Pitch)
    9: [0.0, 0.0, 0.0, 0.0]        # Rest
}

# Not Indexed at 0 for easier human readability (Subject 1 = Soggetto1, Movement 1 = Movimento1)


# Movement names for user selection (1-9)
MOVEMENT_NAMES = {
    1: "Flexion 45°",
    2: "Flexion 90°",
    3: "Flexion 110°",
    4: "Hyperextension -30°",
    5: "Abduction 45°",
    6: "Abduction 90°",
    7: "Elevation 45°",
    8: "Elevation 90°",
    9: "Rest"
}

SECONDARY_BLACKLIST = [
    (1, 1),  # P1, M1: Defective Pectoralis Major (Sternal)
    (5, 1),  # P5, M1: Defective Trapezius Ascendant
    (7, 1),  # P7, M1: Defective Latissimus Dorsi
    # (8, 2),  # P8, M2: Defective Latissimus Dorsi
    # (8, 3),  # P8, M3: Defective Latissimus Dorsi
    (3, 4),  # P3, M4: Defective Trapezius Ascendant
    (4, 4),  # P4, M4: Noisy Everything
    (1, 5),  # P1, M5: Defective Serratus Anterior
    (6, 5),  # P6, M5: Defective Trapezius Ascendant
    (3, 6),  # P3, M6: FATAL - Flatline Trapezius Ascendant
    (4, 6),  # P4, M6: Noisy Serratus Anterior
    (7, 6),  # P7, M6: FATAL - Flatline Trapezius Ascendant
    (3, 7),  # P3, M7: FATAL - Flatline Trapezius Ascendant
    (7, 7),  # P7, M7: FATAL - Flatline Trapezius Ascendant
    (7, 8),  # P7, M8: FATAL - Flatline Trapezius Ascendant
]

COLLECTED_BLACKLIST = [
    (6, 1),     # Completely Corrupted
]

# New CORRUPTED TRIALS identified during validation (added 2024-06-15, much worse than before)
# SECONDARY_BLACKLIST = [
#     (1, 1),  # P1, M1: Defective Pectoralis Major (Sternal)
#     (5, 1),  # P5, M1: Defective Trapezius Ascendant
#     (7, 1),  # P7, M1: Defective Latissimus Dorsi
#     (8, 1),  # P8, M1: Defective Trapezius Descendent
#     (2, 2),  # P2, M2: Noisy Latissimus Dorsi
#     (4, 2),  # P4, M2: Noisy Serratus Anterior
#     (6, 2),  # P6, M2: Noisy Latissimus Dorsi
#     (8, 2),  # P8, M2: Defective Latissimus Dorsi
#     (7, 3),  # P7, M3: Noisy Trapezius Ascendant
#     (8, 3),  # P8, M3: Defective Latissimus Dorsi
#     (3, 4),  # P3, M4: Noisy Channel 6
#     (4, 4),  # P4, M4: Noisy Everything
#     (7, 4),  # P7, M4: Noisy Channel 6
#     (1, 5),  # P1, M5: Defective Serratus Anterior
#     (6, 5),  # P6, M5: Defective Trapezius Ascendant
#     (3, 6),  # P3, M6: Defective Trapezius Ascendant
#     (4, 6),  # P4, M6: Noisy Serratus Anterior
#     (6, 6),  # P6, M6: Noisy Trapezius Ascendant
#     (7, 6),  # P7, M6: Defective Trapezius Ascendant
#     (5, 7),  # P5, M7: Noisy Pectoralis Major (Clavicular)
#     (7, 7),  # P7, M7: Defective Trapezius Ascendant
#     (4, 8),  # P4, M8: Noisy Serratus Anterior
#     (7, 8),  # P7, M8: Defective Trapezius Ascendant
# ]