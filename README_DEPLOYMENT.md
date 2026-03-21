# Raspberry Pi Deployment - Complete Documentation Index

## 📚 Documentation Guide

### For First-Time Setup
Start here in this order:

1. **[PRE_DEPLOYMENT_CHECKLIST.md](PRE_DEPLOYMENT_CHECKLIST.md)** ⭐ START HERE
   - Hardware wiring verification
   - Software prerequisites
   - Step-by-step setup sequence
   - Troubleshooting common issues

2. **[pi-setup.bat](pi-setup.bat)** - Run This First
   - Automated 30-45 minute initial setup
   ```bash
   pi-setup.bat
   ```

3. **[pi-deploy.bat](pi-deploy.bat)** - Run for Everything After
   - Multiple operating modes
   - Graceful shutdown capability
   ```bash
   pi-deploy.bat --hardware --continuous
   ```

### For Daily Usage

4. **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** ⭐ MOST USEFUL
   - Quick command reference for terminal
   - Common workflows
   - Fast lookup of commands

5. **[DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)**
   - Overview of all scripts
   - What was created and enhanced
   - Typical workflow example

### For Detailed Information

6. **[PI_DEPLOYMENT.md](PI_DEPLOYMENT.md)**
   - Complete reference documentation
   - All command options explained
   - Development workflow guide
   - Advanced configuration

7. **[HARDWARE_INTEGRATION_GUIDE.md](HARDWARE_INTEGRATION_GUIDE.md)**
   - Hardware abstraction architecture
   - ContinuousReadingMode vs DataCollectionMode
   - Programmatic usage examples

---

## 🚀 Quick Start (TL;DR)

### First Time
```bash
# Setup (20-30 min)
.\pi-setup.bat

# Reboot
.\pi-deploy.bat --reboot
```

### Every Time After
```bash
# Simulate (test code)
.\pi-deploy.bat --simulate

# Hardware live control
.\pi-deploy.bat --hardware --continuous

# Collect data
.\pi-deploy.bat --hardware --collect --collection_name "trial1"

# Stop (from any mode)
Ctrl+C
```

---

## 📋 What Each Document Contains

### PRE_DEPLOYMENT_CHECKLIST.md
```
✓ Hardware wiring checklist (MCP3008 pinout)
✓ Software prerequisites verification
✓ Phase 1: Initial Setup (step-by-step)
✓ Success verification
```
**When to use:** First time deploying, before running any scripts

### QUICK_REFERENCE.txt
```
✓ Terminal commands at a glance
✓ Common workflows (test → collect → save)
✓ Shutdown procedures and behavior table
✓ All available command options
```
**When to use:** Every time—quick reference for command syntax

### DEPLOYMENT_SUMMARY.md
```
✓ What was created (3 batch scripts)
✓ Code enhancements made
✓ Shutdown capability details
✓ File organization on Pi
✓ Typical workflow example
✓ Command reference
```
**When to use:** Understanding the overall approach

### PI_DEPLOYMENT.md
```
✓ Complete reference guide
✓ All command-line options documented
✓ Development workflow
✓ Data collection workflow
✓ Performance tuning
✓ Advanced configuration
✓ Security notes
✓ Recovery procedures
```
**When to use:** Deep dive into any topic, troubleshooting

### HARDWARE_INTEGRATION_GUIDE.md
```
✓ Signal reading architecture
✓ ContinuousReadingMode details
✓ DataCollectionMode details
✓ Visualization system
✓ Data format specifications
```
**When to use:** Understanding the hardware abstraction layer

### HARDWARE_INTEGRATION_GUIDE.md (from Previous Session)
```
✓ Signal processing pipeline
✓ TKEO envelope visualization
✓ Format compatibility with training
```
**When to use:** Understanding signal processing

---

## 🔧 Batch Scripts Overview

### pi-setup.bat (Initial Setup)
**Run once, takes 20-30 minutes**

Does:
- Tests SSH connection
- Creates project directory and virtual environment
- Installs all dependencies (NumPy, SciPy, PyTorch, etc.)

Command:
```bash
.\\pi-setup.bat
```

### pi-deploy.bat (Run Anywhere)
**Run every time you want to execute the controller**

Modes:
```bash
.\\pi-deploy.bat --simulate                      # Test code
.\\pi-deploy.bat --hardware --continuous         # Live control
.\\pi-deploy.bat --hardware --collect --collection_name "trial1"  # Record data
```

Control:
```bash
Ctrl+C                      # Graceful stop (resource cleanup)
.\\pi-deploy.bat --shutdown    # Remote shutdown
.\\pi-deploy.bat --reboot      # Reboot Pi
.\\pi-deploy.bat --sync        # Code push only
```

---

## 🎯 Shutdown Capability Summary

### How Shutdown Works
1. **Ctrl+C during execution:**
   - Triggers SIGINT/SIGTERM signal handlers
   - Qt event loop quits
   - Socket closes
   - Hardware resources released
   - Data saved (if collection mode)
   - Cleanup prints to console
   - Exit code 0 (success)

2. **Remote shutdown from another terminal:**
   ```bash
   pi-deploy.bat --shutdown
   ```

3. **Functionality tested:**
   - ✅ Graceful signal handling (SIGINT, SIGTERM)
   - ✅ Proper resource cleanup
   - ✅ Data persistence
   - ✅ Headless mode support (SSH)

---

## 📊 Typical Session

```bash
# Windows Command Prompt (VSCode Terminal)

# Session 1: Software setup
C:\project> pi-setup.bat
[30-45 min of installation]
[Reboot Pi]
C:\project> pi-deploy.bat --reboot
.\\pi-setup.bat
[20-30 min of installation]

C:\project> .\\pi-deploy.bat --reboot
[Waits for Pi to come online]

# Session 2: Test and deploy
C:\project> .\\pi-deploy.bat --simulate
[GUI opens, watch output]
C:\project> [Ctrl+C]
[Graceful shutdown, resources cleaned]

# Session 3: Live hardware test
C:\project> .\\pi-deploy.bat --hardware --continuous
[Watch telemetry output]
C:\project> [Ctrl+C]
[Graceful shutdown]

# Session 4: Collect data
C:\project> .\\pi-deploy.bat --hardware --collect --collection_name "abduction_trial1"
[Perform movement for 5 seconds]
C:\project> [Ctrl+C]
[Data saved to hardware_collections/abduction_trial1.mat]
```

---

## ✅ Verification Checklist After Setup

- [ ] `.\\pi-setup.bat` runs to completion
- [ ] `.\\pi-deploy.bat --reboot` succeeds and Pi comes back online
- [ ] `.\\pi-deploy.bat --simulate` runs and shows GUI
- [ ] `Ctrl+C` cleanly stops simulation
- [ ] `.\\pi-deploy.bat --hardware --continuous` reads and displays data
- [ ] `Ctrl+C` shows clean shutdown messages
- [ ] `.\\pi-deploy.bat --hardware --collect --collection_name "test"` saves .mat file

**Windows (Development Machine):**
```
C:\Users\YourName\...\python-signal-processing\
├── pi-setup.bat              ← Run once
├── pi-deploy.bat             ← Run always
├── PRE_DEPLOYMENT_CHECKLIST.md
├── QUICK_REFERENCE.txt
├── DEPLOYMENT_SUMMARY.md
├── PI_DEPLOYMENT.md
├── HARDWARE_INTEGRATION_GUIDE.md
├── *.py files
└── secondary_data/ (for simulation)
```

**Raspberry Pi:**
```
/home/realcj/emg-controller/
├── venv/                     (created by setup)
├── *.py files                (synced by deploy)
├── hardware_collections/     (data saved here)
├── signal_plots/
├── movement_grids/
└── logs/
```

---

## 🆘 Help & Troubleshooting

**Can't connect to Pi:**
```bash
ping realcj-pi5.local
# If fails, may need to edit hostname in batch scripts
```

**Setup fails:**
See PRE_DEPLOYMENT_CHECKLIST.md "Troubleshooting During Deployment"

**Hardware not working:**
```bash
pi-deploy.bat --test-hardware
# Should show: ✓ SPI interface OK
```

**Can't stop with Ctrl+C:**
```bash
# Run in another terminal
pi-deploy.bat --shutdown
```

**All issues:**
Read [PI_DEPLOYMENT.md](PI_DEPLOYMENT.md) "Troubleshooting" section

---

## 🎓 Learning Path

If you're new to these scripts:

1. **Read:** `PRE_DEPLOYMENT_CHECKLIST.md` (5 min)
2. **Read:** `QUICK_REFERENCE.txt` (2 min)
3. **Run:** `pi-setup.bat` (45 min, mostly waiting)
4. **Run:** `pi-deploy.bat --reboot` (wait for Pi)
5. **Keep:** `QUICK_REFERENCE.txt` visible for every session
6. **Reference:** `PI_DEPLOYMENT.md` when you have questions
7. **Advanced:** `HARDWARE_INTEGRATION_GUIDE.md` for deep understanding

---

## 🏁 Summary

You now have:
- ✅ **Automated setup script** - One-time 45-minute environment setup
- ✅ **Multi-mode deployment** - Simulate, continuous, or data collection
- ✅ **Graceful shutdown** - Ctrl+C properly cleans everything
- ✅ **Complete documentation** - From checklist to reference guides
- ✅ **Terminal-based workflow** - Everything from VSCode terminal
- ✅ **Data collection infrastructure** - Auto-saves to .mat format

Everything is ready to deploy. Start with `PRE_DEPLOYMENT_CHECKLIST.md` and follow the steps!
