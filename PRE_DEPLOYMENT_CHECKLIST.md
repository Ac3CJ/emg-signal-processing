# Pre-Deployment Checklist

## Before You Deploy

### Hardware Setup
- [ ] **MCP3008 ADC Wired:**
  - [ ] Pin 16 (VDD) → 3.3V power
  - [ ] Pin 15 (VREF) → 3.3V power  
  - [ ] Pin 14 (AGND) → Ground
  - [ ] Pin 13 (CLK) → Pi GPIO 11 (SCLK)
  - [ ] Pin 12 (DOUT) → Pi GPIO 9 (MISO)
  - [ ] Pin 11 (DIN) → Pi GPIO 10 (MOSI)
  - [ ] Pin 10 (CS) → Pi GPIO 8 (CE0)
  - [ ] Pin 9 (DGND) → Ground

- [ ] **Sensors Connected:**
  - [ ] All 8 channels connected to ADC analog inputs
  - [ ] Signal conditioning circuit working
  - [ ] Ground connections solid

- [ ] **Raspberry Pi 5:**
  - [ ] Power supply connected (Overrides enabled if using 3A)
  - [ ] Network connection (Ethernet or Mobile Hotspot 2.4GHz mode)
  - [ ] SSH enabled

### Software Prerequisites
- [ ] Windows machine with OpenSSH client installed
- [ ] Pi hostname resolves correctly (`ping realcj-pi5.local`)

---

## Step-by-Step Deployment

### Phase 1: Initial Setup (One-Time Only)

**1. Open VSCode Terminal**
Navigate to your project directory.

**2. Run Setup**
```bash
.\pi-setup.bat
```

The script will:
- Test SSH connection
- Create directory and Python virtual environment
- Install all dependencies (numpy, scipy, torch, pyqtgraph, spidev)
- Completion message appears when done

**3. Reboot Pi**

```bash
.\pi-deploy.bat --reboot
```

Wait 30-45 seconds for the Pi to come back online.