@echo off
echo =========================================
echo Raspberry Pi 5 - Automated Setup
echo =========================================
echo.

set PI_HOST=realcj-pi5.local
set PI_USER=realcj

echo [1/3] Testing Connection...
ssh %PI_USER%@%PI_HOST% "echo 'Connected!'" > nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot connect to Raspberry Pi.
    pause
    exit /b 1
)

echo [2/3] Sending Setup Commands...
ssh %PI_USER%@%PI_HOST% "mkdir -p ~/emg-controller && sudo apt update && sudo apt install -y python3-pyqt5 python3-venv && python3 -m venv ~/emg-controller/venv --system-site-packages"

echo [3/3] Installing Python Packages (This will take a few minutes)...
ssh %PI_USER%@%PI_HOST% "source ~/emg-controller/venv/bin/activate && pip install spidev numpy scipy scikit-learn matplotlib pyqtgraph torch torchvision torchaudio"

echo.
echo =========================================
echo Setup Complete! 
echo You can now run pi-deploy.bat
echo =========================================
pause