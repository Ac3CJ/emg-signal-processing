@echo off
REM ================================================================================
REM Raspberry Pi 5 Deployment and Control Script
REM ================================================================================

setlocal enabledelayedexpansion

set PI_HOST=realcj-pi5.local
set PI_USER=realcj
set PI_PROJECT_DIR=/home/realcj/emg-controller
set PI_VENV=%PI_PROJECT_DIR%/venv
set PI_PIDFILE=%PI_PROJECT_DIR%/controller.pid

REM Parse command-line arguments
set MODE=simulate
set HARDWARE=
set CONTINUOUS=
set COLLECT=
set COLLECTION_NAME=hardware_trial
set MODEL=best_shoulder_rcnn.pth
set SIM_FILE=./secondary_data/Soggetto1/Movimento3.mat

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--hardware" set HARDWARE=1& shift& goto parse_args
if "%~1"=="--continuous" set MODE=hardware& set CONTINUOUS=1& shift& goto parse_args
if "%~1"=="--collect" set MODE=hardware& set COLLECT=1& shift& goto parse_args
if "%~1"=="--collection_name" set COLLECTION_NAME=%~2& shift& shift& goto parse_args
if "%~1"=="--simulate" set MODE=simulate& shift& goto parse_args
if "%~1"=="--model" set MODEL=%~2& shift& shift& goto parse_args
if "%~1"=="--sim_file" set SIM_FILE=%~2& shift& shift& goto parse_args
if "%~1"=="--shutdown" goto do_shutdown
if "%~1"=="--reboot" goto do_reboot
if "%~1"=="--test-hardware" goto test_hardware
if "%~1"=="--sync" goto sync_only
if "%~1"=="--help" goto show_help
shift
goto parse_args

:args_done

echo ================================================================================
echo Raspberry Pi 5 - Deployment and Control
echo ================================================================================
echo.
echo Target Host: %PI_HOST%
echo Target User: %PI_USER%
echo Project Directory: %PI_PROJECT_DIR%
echo Mode: %MODE%
if "%HARDWARE%"=="1" (
    echo Hardware: Enabled
    if "%CONTINUOUS%"=="1" echo  ^|-- Reading Mode: Continuous
    if "%COLLECT%"=="1" echo  ^|-- Reading Mode: Data Collection (stores to: %COLLECTION_NAME%)
)
echo Model: %MODEL%
echo.

echo [STEP 1] Testing SSH Connection...
echo =========================================
ssh %PI_USER%@%PI_HOST% "echo 'Connected' && pwd" > nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot connect to Raspberry Pi at %PI_HOST%
    exit /b 1
)
echo [OK] Connected to %PI_HOST%
echo.

echo [STEP 2] Syncing Code Files...
echo =========================================
scp ControllerConfiguration.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
scp SignalReading.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
scp SignalProcessing.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
scp emg-shoulder-prosthetic-controller.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
scp ModelTraining.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
scp DataPreparation.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
scp best_shoulder_rcnn.pth %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
echo [OK] Code files synchronized
echo.

if "%1"=="--sync" (
    echo Sync complete. Exiting.
    exit /b 0
)

echo [STEP 3] Cleaning Up Previous Processes...
echo =========================================
REM Using standard SSH quotes instead of bash Here-Doc
ssh %PI_USER%@%PI_HOST% "if [ -f %PI_PIDFILE% ]; then PID=$(cat %PI_PIDFILE%); if kill -0 $PID 2>/dev/null; then echo '[Pi] Killing previous process...'; kill -15 $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null; fi; rm -f %PI_PIDFILE%; fi; pkill -f 'python.*emg-shoulder-prosthetic-controller' 2>/dev/null || true; echo '[Pi] Cleanup complete'"
echo [OK] Previous processes cleaned
echo.

echo [STEP 4] Executing Controller...
echo =========================================
set "CMD_BASE=source %PI_VENV%/bin/activate && export DISPLAY=:0 && python3 -u %PI_PROJECT_DIR%/emg-shoulder-prosthetic-controller.py"

if "%HARDWARE%"=="1" (
    set "CMD_ARGS=--hardware"
    if "%CONTINUOUS%"=="1" set "CMD_ARGS=!CMD_ARGS! --continuous"
    if "%COLLECT%"=="1" set "CMD_ARGS=!CMD_ARGS! --collect --collection_name %COLLECTION_NAME%"
) else (
    set "CMD_ARGS=--simulate --sim_file %SIM_FILE%"
)

set "CMD_ARGS=!CMD_ARGS! --model %PI_PROJECT_DIR%/%MODEL%"

echo Command: "%CMD_BASE% !CMD_ARGS!"
echo Controller is running. Press Ctrl+C to stop gracefully.
echo.

ssh %PI_USER%@%PI_HOST% "cd %PI_PROJECT_DIR% && source %PI_VENV%/bin/activate && export DISPLAY=:0 && python3 -u %PI_PROJECT_DIR%/emg-shoulder-prosthetic-controller.py !CMD_ARGS!"
set EXIT_CODE=%errorlevel%

echo.
echo [STEP 5] Retrieving Collected Data...
echo =========================================
if "%COLLECT%"=="1" (
    echo Pulling %COLLECTION_NAME%.mat from Raspberry Pi...
    scp %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/hardware_collections/%COLLECTION_NAME%.mat "%USERPROFILE%\Documents"
    
    if errorlevel 0 (
        echo [OK] Successfully saved to your Windows Desktop!
    ) else (
        echo [WARNING] Failed to pull the .mat file. You may need to transfer it manually.
    )
) else (
    echo No data collection requested. Skipping transfer.
)

echo.
echo =========================================
echo Execution Finished (Exit Code: !EXIT_CODE!)
echo =========================================
exit /b !EXIT_CODE!


:do_shutdown
echo =========================================
echo Gracefully Shutting Down Remote Process
echo =========================================
ssh %PI_USER%@%PI_HOST% "if [ -f %PI_PIDFILE% ]; then PID=$(cat %PI_PIDFILE%); if kill -0 $PID 2>/dev/null; then kill -15 $PID; sleep 3; kill -9 $PID 2>/dev/null; fi; rm -f %PI_PIDFILE%; else pkill -9 -f 'python.*emg-shoulder' 2>/dev/null; fi; echo '[OK] Shutdown complete'"
exit /b 0


:do_reboot
echo =========================================
echo Rebooting Raspberry Pi
echo =========================================
ssh %PI_USER%@%PI_HOST% "pkill -9 -f 'python.*emg-shoulder' 2>/dev/null || true; echo '[Pi] Initiating reboot...'; sudo reboot"
echo [OK] Reboot command sent. Waiting for Pi to come back online...
timeout /t 30 /nobreak > nul
call :wait_for_pi
echo Raspberry Pi is back online!
exit /b 0

:wait_for_pi
setlocal
for /L %%i in (1,1,10) do (
    ssh %PI_USER%@%PI_HOST% "echo ready" > nul 2>&1
    if errorlevel 0 exit /b 0
    timeout /t 5 /nobreak
)
exit /b 1