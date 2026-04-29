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
set GUI=
set PARTICIPANT=
set DISPLAY_VALUE=:0
set SYNC_ONLY=0

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--hardware" set HARDWARE=1& shift& goto parse_args
if "%~1"=="--continuous" set MODE=hardware& set CONTINUOUS=1& shift& goto parse_args
if "%~1"=="--collect" set MODE=hardware& set COLLECT=1& shift& goto parse_args
if "%~1"=="--collection_name" set COLLECTION_NAME=%~2& shift& shift& goto parse_args
if "%~1"=="--simulate" set MODE=simulate& shift& goto parse_args
if "%~1"=="--gui" set GUI=1& shift& goto parse_args
if "%~1"=="--participant" set PARTICIPANT=%~2& shift& shift& goto parse_args
if "%~1"=="--display" set DISPLAY_VALUE=%~2& shift& shift& goto parse_args
if "%~1"=="--model" set MODEL=%~2& shift& shift& goto parse_args
if "%~1"=="--sim_file" set SIM_FILE=%~2& shift& shift& goto parse_args
if "%~1"=="--shutdown" goto do_shutdown
if "%~1"=="--reboot" goto do_reboot
if "%~1"=="--test-hardware" set MODE=hardware& set HARDWARE=1& set CONTINUOUS=1& shift& goto parse_args
if "%~1"=="--sync" set SYNC_ONLY=1& shift& goto parse_args
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
if "%GUI%"=="1" (
    echo GUI: Enabled (DISPLAY=%DISPLAY_VALUE%)
) else (
    echo GUI: Disabled ^(headless^)
)
if not "%PARTICIPANT%"=="" echo Participant Override: %PARTICIPANT%
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

REM ---- Resolve participant token up front (used by sync AND end-of-session pull) ----
if not "%PARTICIPANT%"=="" (
    set PARTICIPANT_TOKEN=%PARTICIPANT%
    if /I "!PARTICIPANT_TOKEN:~0,1!"=="P" (
        set PARTICIPANT_NUM=!PARTICIPANT_TOKEN:~1!
    ) else (
        set PARTICIPANT_NUM=!PARTICIPANT_TOKEN!
        set PARTICIPANT_TOKEN=P!PARTICIPANT_TOKEN!
    )
)

if "%SYNC_ONLY%"=="1" (
    echo [STEP 2] Syncing Code Files...
    echo =========================================
    scp ControllerConfiguration.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    scp SignalReading.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    scp SignalProcessing.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    scp FileRepository.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    scp emg-shoulder-prosthetic-controller.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    scp NeuralNetworkModels.py %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    scp "%MODEL%" %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul

    if exist current_participant.txt (
        scp current_participant.txt %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/ 2>nul
    )

    ssh %PI_USER%@%PI_HOST% "mkdir -p %PI_PROJECT_DIR%/biosignal_data/collected/raw %PI_PROJECT_DIR%/biosignal_data/collected/edited" > nul 2>&1

    if not "%PARTICIPANT%"=="" (
        ssh %PI_USER%@%PI_HOST% "printf '%s\n' '!PARTICIPANT_TOKEN!' > %PI_PROJECT_DIR%/current_participant.txt" > nul 2>&1

        if exist "biosignal_data\collected\raw\!PARTICIPANT_TOKEN!M10.mat" (
            scp "biosignal_data\collected\raw\!PARTICIPANT_TOKEN!M10.mat" %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/biosignal_data/collected/raw/ 2>nul
        )
        if exist "biosignal_data\collected\edited\!PARTICIPANT_TOKEN!M10_labelled.mat" (
            scp "biosignal_data\collected\edited\!PARTICIPANT_TOKEN!M10_labelled.mat" %PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/biosignal_data/collected/edited/ 2>nul
        )
    )
    echo [OK] Code files synchronized
    echo.
    echo Sync complete. Run pi-deploy.bat without --sync to start a session.
    exit /b 0
)
echo [SKIP STEP 2] Code sync skipped (use --sync after code changes).
echo.

echo [STEP 3] Cleaning Up Previous Processes...
echo =========================================
REM Using standard SSH quotes instead of bash Here-Doc
ssh %PI_USER%@%PI_HOST% "if [ -f %PI_PIDFILE% ]; then PID=$(cat %PI_PIDFILE%); if kill -0 $PID 2>/dev/null; then echo '[Pi] Killing previous process...'; kill -15 $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null; fi; rm -f %PI_PIDFILE%; fi; pkill -f 'python.*emg-shoulder-prosthetic-controller' 2>/dev/null || true; echo '[Pi] Cleanup complete'"
echo [OK] Previous processes cleaned
echo.

echo [STEP 4] Executing Controller...
echo =========================================
if "%GUI%"=="1" (
    set "CMD_BASE=source %PI_VENV%/bin/activate && export DISPLAY=%DISPLAY_VALUE% && python3 -u %PI_PROJECT_DIR%/emg-shoulder-prosthetic-controller.py"
) else (
    set "CMD_BASE=source %PI_VENV%/bin/activate && python3 -u %PI_PROJECT_DIR%/emg-shoulder-prosthetic-controller.py"
)

if "%HARDWARE%"=="1" (
    set "CMD_ARGS=--hardware"
    if "%CONTINUOUS%"=="1" set "CMD_ARGS=!CMD_ARGS! --continuous"
    if "%COLLECT%"=="1" set "CMD_ARGS=!CMD_ARGS! --collect --collection_name %COLLECTION_NAME%"
) else (
    set "CMD_ARGS=--simulate --sim_file %SIM_FILE%"
)

if "%GUI%"=="1" set "CMD_ARGS=!CMD_ARGS! --gui"
if not "%PARTICIPANT%"=="" set "CMD_ARGS=!CMD_ARGS! --participant %PARTICIPANT%"

set "CMD_ARGS=!CMD_ARGS! --model %PI_PROJECT_DIR%/%MODEL%"

echo Command: "%CMD_BASE% !CMD_ARGS!"
echo Controller is running. Type 'finish' (then Enter) in this window to end the session.
echo.

REM -t allocates a pseudo-TTY so the local user can type 'finish' into the Pi's stdin
ssh -t %PI_USER%@%PI_HOST% "cd %PI_PROJECT_DIR% && !CMD_BASE! !CMD_ARGS!"
set EXIT_CODE=%errorlevel%

echo.
echo [STEP 5] Retrieving Collected Data...
echo =========================================
if "%COLLECT%"=="1" (
    if "%PARTICIPANT_TOKEN%"=="" (
        echo [WARNING] No --participant supplied; cannot determine which P*M*.mat files to pull.
        echo            Pass --participant ^<id^> to enable end-of-session retrieval.
    ) else (
        echo Pulling !PARTICIPANT_TOKEN!M*.mat from Raspberry Pi to Desktop...
        if not exist "%USERPROFILE%\Desktop\emg-collected" mkdir "%USERPROFILE%\Desktop\emg-collected" 2>nul
        scp "%PI_USER%@%PI_HOST%:%PI_PROJECT_DIR%/biosignal_data/collected/raw/!PARTICIPANT_TOKEN!M*.mat" "%USERPROFILE%\Desktop\emg-collected\" 2>nul
        if errorlevel 0 (
            echo [OK] Files saved to %USERPROFILE%\Desktop\emg-collected\
        ) else (
            echo [WARNING] No files matched !PARTICIPANT_TOKEN!M*.mat or transfer failed.
        )
    )
) else (
    echo No data collection requested. Skipping transfer.
)

echo.
echo [STEP 6] Closing Leftover Pi Processes (incl. PyQt windows)...
echo =========================================
ssh %PI_USER%@%PI_HOST% "pkill -f 'python.*emg-shoulder-prosthetic-controller' 2>/dev/null || true; pkill -f 'pyqtgraph' 2>/dev/null || true; pkill -f 'QApplication' 2>/dev/null || true; echo '[Pi] Cleanup OK'"

echo.
echo =========================================
echo Execution Finished (Exit Code: !EXIT_CODE!)
echo =========================================
exit /b !EXIT_CODE!


:sync_only
set SYNC_ONLY=1
goto args_done


:test_hardware
set MODE=hardware
set HARDWARE=1
set CONTINUOUS=1
goto args_done


:show_help
echo.
echo Usage: pi-deploy.bat [options]
echo.
echo Core options:
echo   --simulate                       Run simulated stream mode (default)
echo   --hardware --continuous          Run live hardware continuous mode
echo   --hardware --collect             Run hardware collection mode
echo   --collection_name NAME           Set saved collection name
echo   --model FILE                     Model file to deploy/run
echo   --sim_file FILE                  Simulation .mat file path on Pi
echo   --participant ID                 Participant override (e.g. P1 or 1)
echo.
echo GUI and display:
echo   --gui                            Enable GUI mode on Pi display
echo   --display VALUE                  DISPLAY value for GUI mode (default: :0)
echo.
echo Utility options:
echo   --sync                           Push code to Pi (run once after code changes), then exit
echo                                    NOTE: Sessions no longer auto-sync. Run --sync only after
echo                                          you change code locally; otherwise just launch a session.
echo   --test-hardware                  Shortcut for --hardware --continuous
echo   --shutdown                       Stop running remote controller
echo   --reboot                         Reboot Raspberry Pi
echo   --help                           Show this help message
echo.
echo Typical workflow:
echo   pi-deploy.bat --sync                                  ^(once after code change^)
echo   pi-deploy.bat --hardware --collect --participant P5   ^(many times during session^)
echo.
exit /b 0


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