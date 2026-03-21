# Hardware Integration Guide

## Overview

The signal reading system has been refactored to provide a clean, decoupled architecture for hardware integration. The system supports:

1. **Continuous Reading Mode** - Real-time streaming without memory storage
2. **Data Collection Mode** - Real-time streaming with persistent data storage
3. **Simulation Mode** - Testing with recorded .mat files

## Architecture

### Core Components

```
SignalReading.py
├── SignalReadingMode (ABC)              [Abstract base class]
│   ├── _initialize_spi()                [Shared SPI setup]
│   ├── _read_adc_channel()              [Shared ADC read logic]
│   └── read_sample_chunk() [abstract]   [Mode-specific implementation]
│
├── ContinuousReadingMode (SignalReadingMode)
│   └── read_sample_chunk()              [Reads without storage]
│
├── DataCollectionMode (SignalReadingMode)
│   ├── read_sample_chunk()              [Reads with buffering]
│   ├── save_collection()                [Save to .mat file]
│   ├── get_collection_stats()           [Display statistics]
│   └── clear_collection()               [Clear buffer]
│
└── HardwareSignalVisualizer (Optional)
    └── update_visualization()           [PyQtGraph display, decoupled]


emg-shoulder-prosthetic-controller.py
├── RealTimeProstheticController
│   ├── __init__(reading_mode=None)      [Inject reading mode]
│   ├── read_new_samples()               [Abstracted data source]
│   ├── control_step()                   [Core prosthetic control loop]
│   └── run()                            [Qt event loop]
```

## Usage Examples

### 1. Continuous Reading Mode (Hardware)

Real-time streaming for live prosthetic control - data flows through the system but is **not stored**:

```bash
python emg-shoulder-prosthetic-controller.py --hardware --continuous --model best_shoulder_rcnn.pth
```

This will:
- Initialize hardware (MCP3008) via Continuous Reading Mode
- Read 8 channels at 8kHz effective rate (1kHz × 8 channels multiplexed)
- Process through the trained model
- Send kinematics via UDP to the virtual environment
- Close cleanly when you exit the GUI

### 2. Data Collection Mode (Hardware)

Real-time streaming with data storage for later analysis:

```bash
python emg-shoulder-prosthetic-controller.py --hardware --collect --collection_name "movement_5_trial_1" --model best_shoulder_rcnn.pth
```

This will:
- Initialize hardware with Data Collection Mode
- Read and process data live (same as Continuous Reading)
- **Additionally store all data in memory** for later saving
- At end of session, prompt you to save to `./hardware_collections/movement_5_trial_1.mat`
- Data saved in same format as `secondary_data/` files (EMGDATA key)

### 3. Simulation Mode (Default)

Testing with existing recorded datasets:

```bash
python emg-shoulder-prosthetic-controller.py --simulate --sim_file ./secondary_data/Soggetto1/Movimento3.mat
```

Or simply:

```bash
python emg-shoulder-prosthetic-controller.py --simulate
```

## Programmatic Usage

### Continuous Reading Only (No Controller)

```python
from SignalReading import ContinuousReadingMode
import ControllerConfiguration as Config
import numpy as np

# Initialize hardware
reader = ContinuousReadingMode()

# Read data continuously
try:
    while True:
        chunk = reader.read_sample_chunk()  # Returns (8, 62) array at 8kHz
        print(f"Read {chunk.shape[1]} samples")
        # Process chunk as needed
except KeyboardInterrupt:
    reader.cleanup()
```

### Data Collection (Stand-alone)

```python
from SignalReading import DataCollectionMode
import ControllerConfiguration as Config

# Initialize with custom name
collector = DataCollectionMode(collection_name="test_movement")

# Read and accumulate
for i in range(100):
    chunk = collector.read_sample_chunk()
    stats = collector.get_collection_stats()
    print(f"Collected: {stats['duration_seconds']:.2f}s")

# Save to file
collector.save_collection(output_path="./my_data.mat")
collector.cleanup()
```

### Integrated with Controller

```python
from SignalReading import DataCollectionMode
from emg_shoulder_prosthetic_controller import RealTimeProstheticController

# Setup data collection
collection = DataCollectionMode(collection_name="live_trial")

# Create controller with the collection mode
controller = RealTimeProstheticController(
    model_path='best_shoulder_rcnn.pth',
    reading_mode=collection,
    simulate_data=False
)

# Run live control + data collection simultaneously
controller.run()

# Save afterwards
if collection.sample_count > 0:
    collection.save_collection()
```

## Signal Reading Speed

The hardware reads at **8x the base sampling frequency**:

- Base frequency: `Config.FS = 1000 Hz` (1 kHz)
- Number of channels: `Config.NUM_CHANNELS = 8`
- **Effective reading rate: 8 kHz**

This is because the MCP3008 reads one channel at a time via SPI. With 8 channels being multiplexed, we must clock 8x faster to maintain 1 kHz per-channel sampling.

**Sample timing calculation:**
```
sample_delay = 1.0 / (Config.FS * Config.NUM_CHANNELS)
            = 1.0 / (1000 * 8)
            = 125 microseconds per channel sample
```

## Data Format

Both modes return data in a format compatible with `secondary_data/` files:

**Shape:** `(8, Config.INCREMENT)` where increment = 62 by default
- 8 rows = 8 channels (Pectoralis Major Clav, Pectoralis Major Sternal, etc.)
- 62 columns = 62 ms of data at 1 kHz

**When saved as .mat file:**
```python
{
    'EMGDATA': np.array of shape (8, total_samples)
}
```

This matches the format expected by `DataPreparation.py` and `ModelValidator.py`.

## Desirable Modes

### Hardware Modes

| Mode | Storage | Use Case | Command |
|------|---------|----------|---------|
| Continuous | No | Live prosthetic control, real-time feedback | `--hardware --continuous` |
| Data Collection | Yes (in-memory then file) | Recording training data, trials | `--hardware --collect` |

### Simulation/Testing Modes

| Mode | Source | Use Case | Command |
|------|--------|----------|---------|
| Simulation | .mat file | Testing without hardware, validation | `--simulate --sim_file <path>` |
| Visualization | Hardware + Qt | Hardware verification, signal inspection | `python SignalReading.py` (standalone) |

## Separation of Concerns

The architecture maintains clean separation:

- **SignalReading.py**: Pure hardware abstraction
  - No GUI dependencies
  - No model/control logic
  - Reusable across different applications

- **emg-shoulder-prosthetic-controller.py**: Control logic
  - Agnostic to data source (hardware/simulation)
  - Focuses on prosthetic inference and UDP transmission
  - Accepts reading_mode as a dependency injection parameter

- **Visualization (HardwareSignalVisualizer)**: Optional decoration
  - Completely separate from reading logic
  - Can be used standalone or with any reading mode
  - Does not affect hardware timing or data integrity

## Cleanup & Exception Handling

Both modes properly clean up resources:

```python
reading_mode.cleanup()  # Closes SPI bus safely
```

The controller automatically calls this when exiting:

```python
finally:
    if self.use_hardware and self.reading_mode:
        self.reading_mode.cleanup()
```

## Troubleshooting

### "SPI Bus Open Failed"
- Ensure you're running on Raspberry Pi with SPI enabled
- Check: `raspi-config` → Interface Options → SPI
- Verify MCP3008 wiring (CLK, DOUT, DIN, CS pins)

### "ADC reads all zeros or noise"
- Check VREF connection (should be 3.3V)
- Verify signal conditioning on inputs
- Check ground connections

### Data Collection Memory Issues
- Memory per chunk: `8 channels × 62 samples × 4 bytes = 1984 bytes`
- For 1 hour collection at 1 kHz: ~7.2 GB
- Recommend saving periodically in long sessions

### Hardware Timing Variability
- Software sleep-based timing may drift slightly
- For tighter synchronization, consider timer-based reading on next iteration
- Current 125 µs sampling gives ~8 kHz ±2%, acceptable for real-time control

## Configuration

All hardware parameters are centralized in `ControllerConfiguration.py`:

```python
FS = 1000.0                 # Sampling frequency
NUM_CHANNELS = 8            # Number of channels
INCREMENT = 62              # Samples per window (62 ms at 1 kHz)

NOTCH_FREQ = 50.0           # Powerline noise
BANDPASS_LOW = 30.0         # Signal conditioning
BANDPASS_HIGH = 450.0
```

These are used by both signal reading and signal processing pipelines.
