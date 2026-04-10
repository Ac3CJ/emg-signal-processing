@echo off
setlocal enabledelayedexpansion

:: Parse --name
set MODEL_NAME=
:arg_loop
if "%~1"=="" goto arg_done
if /I "%~1"=="--name" (
	set MODEL_NAME=%~2
	shift
	shift
	goto arg_loop
)
echo Error: Unknown argument %~1
echo Usage: model-validation.bat --name your_run_name
exit /b 1

:arg_done
if "%MODEL_NAME%"=="" (
	echo Error: Please provide a name using --name.
	echo Example: model-validation.bat --name my_run
	exit /b 1
)

set RUN_ROOT=.\old-models\%MODEL_NAME%
set TRAIN_ROOT=%RUN_ROOT%\training
set VALIDATION_ROOT=%RUN_ROOT%\validation

if not exist "%TRAIN_ROOT%" (
	echo Error: Training outputs not found: %TRAIN_ROOT%
	echo Run batch-model-training.bat first with --name %MODEL_NAME%
	exit /b 1
)

if not exist "%VALIDATION_ROOT%" mkdir "%VALIDATION_ROOT%"

echo =========================================
echo Automatic Validation for: %MODEL_NAME%
echo =========================================
echo.

call :validate_model "loso-colearning" "%TRAIN_ROOT%\loso-colearning\best_shoulder_rcnn_loso_colearning.pth"
if errorlevel 1 goto validation_failed

call :validate_model "loso-secondary" "%TRAIN_ROOT%\loso-secondary\best_shoulder_rcnn_loso_secondary.pth"
if errorlevel 1 goto validation_failed

call :validate_model "transfer" "%TRAIN_ROOT%\transfer\best_shoulder_rcnn_transfer_from_loso_secondary.pth"
if errorlevel 1 goto validation_failed

echo.
echo =========================================
echo Validation Complete!
echo =========================================
exit /b 0

:validate_model
set MODEL_LABEL=%~1
set MODEL_PATH=%~2

if not exist "%MODEL_PATH%" (
	echo [SKIP] Model not found for !MODEL_LABEL!: %MODEL_PATH%
	exit /b 0
)

echo -----------------------------------------
echo Validating !MODEL_LABEL!
echo Model: %MODEL_PATH%
echo -----------------------------------------

call :clear_live_outputs

python .\ModelValidator.py --model "%MODEL_PATH%" --validate_predefined --validate_ensemble
if errorlevel 1 (
	echo ERROR: Predefined/ensemble validation failed for !MODEL_LABEL!
	exit /b 1
)

for %%P in (1 2 3 4) do (
	python .\ModelValidator.py --model "%MODEL_PATH%" --collected %%P
	if errorlevel 1 (
		echo ERROR: Collected validation failed for participant %%P on !MODEL_LABEL!
		exit /b 1
	)
)

set DEST_ROOT=%VALIDATION_ROOT%\!MODEL_LABEL!
if not exist "!DEST_ROOT!\kinematic-plots" mkdir "!DEST_ROOT!\kinematic-plots"
if not exist "!DEST_ROOT!\ensemble-plots" mkdir "!DEST_ROOT!\ensemble-plots"

if exist ".\kinematic-plots\*.png" move /Y ".\kinematic-plots\*.png" "!DEST_ROOT!\kinematic-plots\" >nul
if exist ".\ensemble-plots\*.png" move /Y ".\ensemble-plots\*.png" "!DEST_ROOT!\ensemble-plots\" >nul

echo [DONE] Stored validation outputs in !DEST_ROOT!
echo.
exit /b 0

:clear_live_outputs
if exist ".\kinematic-plots\*.png" del /q ".\kinematic-plots\*.png"
if exist ".\ensemble-plots\*.png" del /q ".\ensemble-plots\*.png"
exit /b 0

:validation_failed
echo.
echo =========================================
echo Validation Failed!
echo =========================================
exit /b 1