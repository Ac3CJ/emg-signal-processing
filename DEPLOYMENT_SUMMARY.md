# Deployment Scripts Summary

## What Was Created

Three comprehensive batch scripts for Raspberry Pi 5 deployment using only terminal commands in VSCode:

### 1. **pi-setup.bat** - Initial Environment Setup
**Purpose:** One-time complete environment configuration (30-45 minutes)

**What it does:**
- ✅ Tests SSH connection to Pi
- ✅ Updates system packages (apt-get)
- ✅ Creates Python virtual environment
- ✅ Installs ALL dependencies:
  - NumPy, SciPy, scikit-learn
  - PyTorch (CPU optimized for Pi)
  - PyQtGraph, PyQt5
  - SPI device drivers (spidev)
  - Build tools and libraries
- ✅ Enables SPI/GPIO interfaces in firmware
- ✅ Transfers code files to Pi
- ✅ Verifies successful installation

**Usage:**
```bash
.\pi-setup.bat
```

**Then reboot:**
```bash
.\pi-deploy.bat --reboot
```

### 2. **pi-deploy.bat** - Execution and Control
**Purpose:** Run controller in various modes with full shutdown capability

**Modes:**
```bash
# Test with recorded data (no hardware needed)
.\pi-deploy.bat --simulate

# Live control (hardware, no storage)
.\pi-deploy.bat --hardware --continuous

# Record data (hardware + save)
.\pi-deploy.bat --hardware --collect --collection_name "trial_name"
```

**Control:**
- Press **Ctrl+C** during execution → Graceful shutdown with resource cleanup
- Run `--shutdown` → Graceful remote shutdown
- Run `--reboot` → Reboot Pi
- Run `--sync` → Just push code, don't execute

**Features:**
- ✅ Automatic code synchronization
- ✅ Previous process cleanup
- ✅ Graceful shutdown with signal handlers
- ✅ Data auto-save (collection mode)
- ✅ Exit code tracking
- ✅ Help system

### 3. **Documentation Files**
- **PI_DEPLOYMENT.md** - Complete reference guide with troubleshooting
- **QUICK_REFERENCE.txt** - Quick command cheat sheet for terminal

---

## Key Enhancements Made to Code

### emg-shoulder-prosthetic-controller.py

**Added Signal Handlers:**
```python
# SIGINT (Ctrl+C) handler
signal.signal(signal.SIGINT, handle_sigint)

# SIGTERM handler (from systemd/SSH kill)
signal.signal(signal.SIGTERM, handle_sigterm)
```

**Enhanced Cleanup (finally block):**
- UDP socket closure with error handling
- Hardware resource cleanup (SPI, GPIO)
- Data saving with headless mode detection
- Comprehensive error logging
- Exit code tracking

**Headless Mode Support:**
- Detects SSH without GUI (`sys.stdin.isatty()`)
- Auto-saves data instead of prompting
- Works perfectly with batch scripts

---

## Shutdown Verification

When you press **Ctrl+C**, you'll see:

```
[Controller] SIGINT received. Initiating graceful shutdown...
[Main] Cleaning up resources...
[Main] ✓ UDP Socket closed
[Main] ✓ Hardware resources cleaned up
[Main] Shutdown complete

=========================================
Execution Finished (Exit Code: 0)
=========================================
```

All of this happens **automatically** - no manual commands needed!

---

## Typical Workflow Example

```bash
# First time setup (20-30 min)
C:\project> .\pi-setup.bat
... waits for completion ...

C:\project> .\pi-deploy.bat --reboot
... waits for reboot ...

# Later: Quick hardware test
C:\project> pi-deploy.bat --hardware --continuous
[Hardware runs, streams telemetry]
Press Ctrl+C
[Automatic cleanup, exit]

# Collect data for training
C:\project> pi-deploy.bat --hardware --collect --collection_name "abduction_trial1"
[Hardware runs]
Ctrl+C
[Data saved to hardware_collections/abduction_trial1.mat]

# Commit changes
C:\project> git add *.py
C:\project> git commit -m "Improved filter cutoff"
C:\project> git push
```

---

## File Organization

**On Windows (local machine):**
```
python-signal-processing/
├── pi-setup.bat              (← Run first time)
├── pi-deploy.bat             (← Run every time after)
├── PI_DEPLOYMENT.md          (← Reference guide)
├── QUICK_REFERENCE.txt       (← Quick commands)
├── emg-shoulder-prosthetic-controller.py  (enhanced with shutdown)
├── SignalReading.py          (hardware abstraction)
├── *.py                      (other modules)
└── [source code files]
```

**On Raspberry Pi:**
```
~/emg-controller/
├── venv/                     (Python environment, created by setup)
├── *.py                      (your code, synced by deploy)
├── hardware_collections/     (data saved here)
├── signal_plots/
├── movement_grids/
└── logs/
```

---

## Terminal-Only Workflow

Everything works from the VSCode terminal - no SSH GUI needed:

```bash
# All these work from VSCode terminal:
pi-setup.bat
pi-deploy.bat --hardware --continuous
Ctrl+C
pi-deploy.bat --shutdown
pi-deploy.bat --reboot
pi-deploy.bat --test-hardware
```

No clicking, no windows - pure terminal-based as requested!

---

## Shutdown Capability Checklist

✅ **Graceful Shutdown:**
- Ctrl+C during execution
- Triggers signal handlers (SIGINT, SIGTERM)
- Closes all resources cleanly
- Saves data (if collection mode)
- Exit code 0

✅ **Remote Shutdown:**
- `pi-deploy.bat --shutdown` command
- Kills process on Pi remotely
- No need for separate SSH session

✅ **Automatic Cleanup:**
- Previous processes killed before new run
- No leftover Python instances
- SPI/GPIO properly released

✅ **Headless Mode:**
- Auto-detects SSH environment
- Skips interactive prompts
- Auto-saves data

---

## What You Can Do Now

1. **Deploy with any mode:** Hardware continuous, hardware collection, or simulation
2. **Stop cleanly:** Just Ctrl+C - everything cleans up automatically
3. **Reboot Pi:** `pi-deploy.bat --reboot`
4. **Test hardware:** `pi-deploy.bat --test-hardware`
5. **Push code only:** `pi-deploy.bat --sync` (no execution)
6. **Everything from terminal:** No GUI needed, all batch commands

---

## Commands at a Glance

```bash
# SETUP (first time only)
pi-setup.bat                          # 30-45 min setup

# DEPLOYMENT (subsequent runs)
pi-deploy.bat                         # Simulate (default)
pi-deploy.bat --hardware --continuous # Live control
pi-deploy.bat --hardware --collect --collection_name "trial"  # Record

# CONTROL
Ctrl+C                                # Graceful stop
pi-deploy.bat --shutdown              # Remote shutdown
pi-deploy.bat --reboot                # Reboot Pi
pi-deploy.bat --sync                  # Code push only

# VERIFICATION
pi-deploy.bat --test-hardware         # Hardware check
pi-deploy.bat --help                  # Show all options
```

---

## Next Steps

1. **Review the scripts** in VSCode to understand the flow
2. **Test locally first** with `pi-deploy.bat --simulate`
3. **When ready for hardware:**
   - `pi-setup.bat` (one time only)
   - `pi-deploy.bat --reboot` 
   - `pi-deploy.bat --hardware --continuous` (live test)
4. **Collect data:** `pi-deploy.bat --hardware --collect --collection_name "movement_5"`

Everything is ready to go! The scripts handle all the complexity - you just run them from the terminal.
