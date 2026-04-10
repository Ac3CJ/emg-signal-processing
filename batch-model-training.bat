@echo off
setlocal enabledelayedexpansion

:: 1. Parse arguments
set MODEL_NAME=
set RUN_VALIDATE=0
set COLLECTED_PARTICIPANTS=[1,3,4]

:arg_loop
if "%~1"=="" goto arg_done
if /I "%~1"=="--name" (
    set MODEL_NAME=%~2
    shift
    shift
    goto arg_loop
)
if /I "%~1"=="--validate" (
    set RUN_VALIDATE=1
    shift
    goto arg_loop
)
echo Error: Unknown argument %~1
echo Usage: batch-model-training.bat --name your_run_name [--validate]
exit /b 1

shift
goto arg_loop
:arg_done

:: Check if the name was actually provided
if "%MODEL_NAME%"=="" (
    echo Error: Please provide a name using the --name flag.
    echo Example: batch-model-training.bat --name my_new_model [--validate]
    exit /b 1
)

set RUN_ROOT=.\old-models\%MODEL_NAME%
set TRAIN_ROOT=%RUN_ROOT%\training

echo ========================================================
echo Starting batch training run for: %MODEL_NAME%
echo ========================================================
if "%RUN_VALIDATE%"=="1" (
    echo Validation after training: ENABLED
) else (
    echo Validation after training: DISABLED
)

:: 2. Create destination directories
if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%TRAIN_ROOT%" mkdir "%TRAIN_ROOT%"
if not exist "%TRAIN_ROOT%\loso-colearning" mkdir "%TRAIN_ROOT%\loso-colearning"
if not exist "%TRAIN_ROOT%\loso-secondary" mkdir "%TRAIN_ROOT%\loso-secondary"
if not exist "%TRAIN_ROOT%\transfer" mkdir "%TRAIN_ROOT%\transfer"

:: Keep workspace root tidy for this batch run
if exist "best_shoulder_rcnn_loso_secondary.pth" del /q "best_shoulder_rcnn_loso_secondary.pth"

:: 3. LOSO + Colearning (secondary + selected collected participants)
echo.
echo --- Running LOSO + Colearning (Collected: %COLLECTED_PARTICIPANTS%) ---
call :clean_training_outputs
python ModelTraining.py --mode loso --include_collected --collected_participants "%COLLECTED_PARTICIPANTS%"
if errorlevel 1 goto training_failed

if exist "%TRAIN_ROOT%\loso-colearning\loso-fold-results" rmdir /s /q "%TRAIN_ROOT%\loso-colearning\loso-fold-results"
if exist "loso-fold-results" move "loso-fold-results" "%TRAIN_ROOT%\loso-colearning\" >nul
if exist "best_shoulder_rcnn.pth" move /Y "best_shoulder_rcnn.pth" "%TRAIN_ROOT%\loso-colearning\best_shoulder_rcnn_loso_colearning.pth" >nul
if exist "training_loss_curve.png" move /Y "training_loss_curve.png" "%TRAIN_ROOT%\loso-colearning\training_loss_curve_loso_colearning.png" >nul
if exist "training_dataset_distribution.txt" move /Y "training_dataset_distribution.txt" "%TRAIN_ROOT%\loso-colearning\training_dataset_distribution_loso_colearning.txt" >nul

:: 4. LOSO (secondary only)
echo.
echo --- Running LOSO (Secondary Only) ---
call :clean_training_outputs
python ModelTraining.py --mode loso
if errorlevel 1 goto training_failed

if exist "best_shoulder_rcnn.pth" copy /Y "best_shoulder_rcnn.pth" "best_shoulder_rcnn_loso_secondary.pth" >nul
if exist "%TRAIN_ROOT%\loso-secondary\loso-fold-results" rmdir /s /q "%TRAIN_ROOT%\loso-secondary\loso-fold-results"
if exist "loso-fold-results" move "loso-fold-results" "%TRAIN_ROOT%\loso-secondary\" >nul
if exist "best_shoulder_rcnn.pth" move /Y "best_shoulder_rcnn.pth" "%TRAIN_ROOT%\loso-secondary\best_shoulder_rcnn_loso_secondary.pth" >nul
if exist "training_loss_curve.png" move /Y "training_loss_curve.png" "%TRAIN_ROOT%\loso-secondary\training_loss_curve_loso_secondary.png" >nul
if exist "training_dataset_distribution.txt" move /Y "training_dataset_distribution.txt" "%TRAIN_ROOT%\loso-secondary\training_dataset_distribution_loso_secondary.txt" >nul

if not exist "best_shoulder_rcnn_loso_secondary.pth" (
    echo ERROR: Missing pretrained LOSO-secondary model for transfer learning.
    goto training_failed
)

:: 5. Transfer learning from LOSO secondary model
echo.
echo --- Running Transfer Learning (Collected Train Participants: %COLLECTED_PARTICIPANTS%) ---
call :clean_training_outputs
python ModelTraining.py --mode transfer --pretrained "best_shoulder_rcnn_loso_secondary.pth" --collected_train_participants "%COLLECTED_PARTICIPANTS%"
if errorlevel 1 goto training_failed

if exist "best_shoulder_rcnn_transfer.pth" move /Y "best_shoulder_rcnn_transfer.pth" "%TRAIN_ROOT%\transfer\best_shoulder_rcnn_transfer_from_loso_secondary.pth" >nul
if exist "training_loss_curve.png" move /Y "training_loss_curve.png" "%TRAIN_ROOT%\transfer\training_loss_curve_transfer.png" >nul
if exist "transfer_learning_dataset_distribution.txt" move /Y "transfer_learning_dataset_distribution.txt" "%TRAIN_ROOT%\transfer\transfer_learning_dataset_distribution.txt" >nul

:: 6. Optional validation
if "%RUN_VALIDATE%"=="1" (
    echo.
    echo --- Running Validation Batch ---
    call .\model-validation.bat --name "%MODEL_NAME%"
    if errorlevel 1 goto validation_failed
)

echo.
echo ========================================================
echo Batch training and file organization complete!
echo ========================================================
exit /b 0

:clean_training_outputs
if exist "loso-fold-results" rmdir /s /q "loso-fold-results"
if exist "best_shoulder_rcnn.pth" del /q "best_shoulder_rcnn.pth"
if exist "best_shoulder_rcnn_transfer.pth" del /q "best_shoulder_rcnn_transfer.pth"
if exist "training_loss_curve.png" del /q "training_loss_curve.png"
if exist "training_dataset_distribution.txt" del /q "training_dataset_distribution.txt"
if exist "transfer_learning_dataset_distribution.txt" del /q "transfer_learning_dataset_distribution.txt"
exit /b 0

:training_failed
echo.
echo ========================================================
echo ERROR: Training batch failed.
echo ========================================================
exit /b 1

:validation_failed
echo.
echo ========================================================
echo ERROR: Training finished, but validation batch failed.
echo ========================================================
exit /b 1