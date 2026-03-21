# Raspberry Pi 5 Deployment Guide

## Overview

This guide covers deployment of the EMG Shoulder Prosthetic Controller to a Raspberry Pi 5 using automated batch scripts for a terminal-based workflow in VSCode.

## Quick Start

### 1. Initial Setup (First Time Only)

```bash
.\pi-setup.bat
```

This script will:
- ✅ Test SSH connection to Pi
- ✅ Create project directory and Python virtual environment
- ✅ Install all required dependencies:
  - NumPy, SciPy, scikit-learn
  - PyTorch (most time spent here)
  - PyQtGraph, PyQt5
  - spidev for SPI communication

**Expected runtime:** 20-30 minutes (PyTorch installation is the longest step)

After setup completes, reboot the Pi:
```bash
.\pi-deploy.bat --reboot
```

### 2. Run Controller (Subsequent Uses)

**Simulation Mode** (test software):
```bash
.\pi-deploy.bat --simulate
```

**Continuous Reading** (live prosthetic control):
```bash
.\pi-deploy.bat --hardware --continuous
```

**Data Collection** (record trial data):
```bash
.\pi-deploy.bat --hardware --collect --collection_name "movement_5_trial1"
```

## Shutdown

### From Terminal (Graceful)
```bash
.\pi-deploy.bat --shutdown
```

### From SSH Session (Ctrl+C)
Press `Ctrl+C` while the controller is running. This triggers:
1. SIGINT signal handler
2. Qt event loop shutdown
3. Proper cleanup of hardware resources
4. Data saving prompt (if in collection mode)
5. Clean exit with status code

### Emergency Shutdown
If graceful shutdown fails, the batch scripts kill any remaining processes automatically on next run.

## Deployment Script Reference

### Command Syntax

```bash
pi-deploy.bat [--mode] [--options] [--model PATH]
```

### Modes

#### Hardware Modes

**Continuous Reading** (No Storage)
```bash
.\pi-deploy.bat --hardware --continuous
```
- Real-time prosthetic control
- Data flows through model but is NOT saved
- Suitable for live testing and operation
- Best for responsive, low-latency control

**Data Collection** (With Storage)
```bash
.\pi-deploy.bat --hardware --collect --collection_name "trial_name"
```
- Real-time prosthetic control + data recording
- All data buffered in memory during execution
- Saved to `hardware_collections/trial_name.mat` after execution
- Prompts user to save on exit (auto-saves in headless mode)

#### Simulation Mode

```bash
.\pi-deploy.bat --simulate --sim_file path/to/data.mat
```
- Test software without hardware
- Uses recorded .mat file data (defaults to Soggetto1/Movimento3.mat)
- No GPIO/SPI access required
- Perfect for debugging and code testing

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--hardware` | Enable physical hardware (MCP3008) | `--hardware` |
| `--continuous` | Continuous mode (no storage) | `--hardware --continuous` |
| `--collect` | Data collection mode | `--hardware --collect` |
| `--collection_name` | Name for data file | `--collection_name "move_5_t1"` |
| `--simulate` | Simulation mode | `--simulate` |
| `--model` | Model weights path | `--model /path/to/model.pth` |
| `--sim_file` | Simulation data file | `--sim_file ./data.mat` |

### Special Commands

**Sync Code Only** (Don't Execute)
```bash
.\pi-deploy.bat --sync
```
- Transfers code files to Pi without running the controller
- Useful for updating code mid-session

**Reboot Pi**
```bash
pi-deploy.bat --reboot
```

**Test Hardware Setup**
```bash
pi-deploy.bat --test-hardware
```

**Show Help**
```bash
pi-deploy.bat --help
```

## File Structure on Raspberry Pi

```
/home/realcj/emg-controller/
├── venv/                          # Python virtual environment
├── ControllerConfiguration.py      # Hyperparameters
├── SignalReading.py               # Hardware abstraction
├── SignalProcessing.py            # Signal filtering
├── ModelTraining.py               # Neural network definition
├── emg-shoulder-prosthetic-controller.py  # Main controller
├── activate_env.sh                # Convenience script
├── logs/                          # Execution logs
├── hardware_collections/          # Saved data from collection mode
├── signal_plots/                  # Generated signal plots
└── movement_grids/                # Stitched movement grids
```

## Troubleshooting

### Cannot Connect to Pi

**Error:** `Cannot connect to Raspberry Pi at realcj-pi5.local`

**Solutions:**
1. Check Pi is powered on:
   ```bash
   ping realcj-pi5.local
   ```

2. Verify SSH is enabled:
   ```bash
   ssh realcj@realcj-pi5.local "echo 'SSH works'"
   ```

3. Update hostname in batch scripts if different:
   - Edit `pi-setup.bat` and `pi-deploy.bat`
   - Change `set PI_HOST=...` to your Pi's hostname

### Package Installation Failures

**During setup, if PyTorch installation fails:**

The Pi has limited bandwidth and memory. PyTorch installation can take 15+ minutes. If it times out:

```bash
REM On the Pi directly via SSH:
ssh realcj@realcj-pi5.local
cd ~/emg-controller
source venv/bin/activate
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### SPI Not Working

**Error:** `OSError: [Errno 2] No such file or directory: '/dev/spidev0.0'`

**Solutions:**
1. Verify SPI is enabled:
   ```bash
   pi-deploy.bat --test-hardware
   ```

2. Check `/boot/firmware/config.txt` has `dtparam=spi=on`:
   ```bash
   ssh realcj@realcj-pi5.local "grep spi /boot/firmware/config.txt"
   ```

3. If not present, re-run setup:
   ```bash
   pi-deploy.bat --reboot
   ```

### Hardware Reads All Zeros

**Check wiring:**
1. MCP3008 VDD → 3.3V
2. MCP3008 VREF → 3.3V
3. MCP3008 AGND → GND
4. MCP3008 DGND → GND
5. CLK → Pi GPIO 11 (SCLK)
6. DOUT → Pi GPIO 9 (MISO)
7. DIN → Pi GPIO 10 (MOSI)
8. CS → Pi GPIO 8 (CE0)

**Verify connection:**
```bash
pi-deploy.bat --test-hardware
```

### Process Won't Shutdown

**Manual cleanup:**
```bash
ssh realcj@realcj-pi5.local "pkill -9 -f emg-shoulder"
```

## Development Workflow

### Typical Session

1. **Edit code locally in VSCode**

2. **Sync and run:**
   ```bash
   pi-deploy.bat --hardware --continuous
   ```

3. **Monitor output in terminal:**
   - Real-time telemetry output
   - Signal processing status
   - Model predictions

4. **Stop with Ctrl+C:**
   - Graceful shutdown
   - Automatic resource cleanup

5. **Commit changes:**
   ```bash
   git add *.py
   git commit -m "Description of changes"
   git push
   ```

### Data Collection Workflow

1. **Start collection:**
   ```bash
   pi-deploy.bat --hardware --collect --collection_name "movement_5_trial_1"
   ```

2. **Run test in physical environment:**
   - Prosthetic performing movement
   - ~4.5 seconds per trial
   - Can repeat multiple times before stopping

3. **Stop with Ctrl+C:**
   - Displays collection statistics
   - Prompts to save (or auto-saves in headless mode)
   - Data saved to `hardware_collections/movement_5_trial_1.mat`

4. **Retrieve data:**
   ```bash
   scp realcj@realcj-pi5.local:/home/realcj/emg-controller/hardware_collections/*.mat ./
   ```

5. **Process with DataPreparation.py:**
   ```bash
   python DataPreparation.py --hardware_collection hardware_collections/movement_5_trial_1.mat
   ```

## Performance Tuning

### Real-Time Latency
The target latency is ~62ms (Config.INCREMENT):
- Best achieved with **Continuous Reading Mode**
- Data Collection Mode adds minimal overhead (~<1ms)

### Memory Usage
- **Continuous Mode:** ~150 MB (stable)
- **Data Collection:** ~150 MB + collected data
  - Example: 1-hour collection = ~7.2 GB
  - Recommendation: Save every 10-20 minutes in long sessions

### CPU Usage
- Expected: ~30-40% utilization during normal operation
- Monitor with: `ssh pi 'top'"` from another terminal

## Advanced Options

### Custom Model

```bash
pi-deploy.bat --hardware --continuous --model ./models/custom_v2.pth
```

### Custom Data File

```bash
pi-deploy.bat --simulate --sim_file ./secondary_data/Soggetto6/Movimento7.mat
```

### Headless Operation (No GUI)

The controller automatically detects headless mode (SSH without X11 display) and:
- Skips PyQtGraph GUI initialization
- Auto-saves collected data instead of prompting
- Uses text-only telemetry output

## Logging

Execution logs are automatically saved to `~/emg-controller/logs/`. To view:

```bash
ssh realcj@realcj-pi5.local "tail -f ~/emg-controller/logs/*.log"
```

## Security Notes

- SSH access via `realcj@realcj-pi5.local`
- No hardcoded credentials in scripts (uses SSH key auth)
- Data files stored in user home directory (not world-readable)
- Telemetry sent only to `127.0.0.1:5005` (localhost)

## Recovery

### Full Reset

If something breaks and you want a clean slate:

```bash
REM Delete all data and recreate environment
ssh realcj@realcj-pi5.local "rm -rf ~/emg-controller"

REM Run setup again
pi-setup.bat
```

### Partial Recovery (Keep Data)

```bash
REM Just update code and venv, keep hardware_collections
pi-deploy.bat --sync
pi-deploy.bat --reboot
```

## Windows SSH Setup

If you don't have an SSH client configured:

1. Install OpenSSH Client (Windows 10+):
   ```powershell
   Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
   ```

2. Generate SSH key (if not already done):
   ```bash
   ssh-keygen -t ed25519
   ```

3. Copy key to Pi:
   ```bash
   scp %USERPROFILE%\.ssh\id_ed25519.pub realcj@realcj-pi5.local:
   SSH realcj@realcj-pi5.local "cat id_ed25519.pub >> .ssh/authorized_keys"
   ```

## Getting Help

- **Batch script issues:** Run with `--help`:
  ```bash
  pi-deploy.bat --help
  pi-setup.bat --help
  ```

- **Hardware issues:** Run diagnostic:
  ```bash
  pi-deploy.bat --test-hardware
  ```

- **Code issues:** Check controller output for `[ERROR]` and `[WARNING]` tags
