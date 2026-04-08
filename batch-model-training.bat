@echo off
setlocal enabledelayedexpansion

:: 1. Parse the --name argument
set MODEL_NAME=
:arg_loop
if "%~1"=="" goto arg_done
if "%~1"=="--name" (
    set MODEL_NAME=%~2
    shift
    shift
    goto arg_loop
)
shift
goto arg_loop
:arg_done

:: Check if the name was actually provided
if "%MODEL_NAME%"=="" (
    echo Error: Please provide a name using the --name flag.
    echo Example: run_training_batch.bat --name my_new_model
    exit /b 1
)

echo ========================================================
echo Starting batch training run for: %MODEL_NAME%
echo ========================================================

:: 2. Create the empty old-models folder
echo Creating directory: .\old-models\%MODEL_NAME%
if not exist ".\old-models\%MODEL_NAME%" mkdir ".\old-models\%MODEL_NAME%"

:: 3. Create the required subfolders for the plots
for %%d in ("training-loss" "ensemble-plots" "kinematic-plots") do (
    if not exist ".\%%~d\%MODEL_NAME%_80-20" mkdir ".\%%~d\%MODEL_NAME%_80-20"
    if not exist ".\%%~d\%MODEL_NAME%_loso" mkdir ".\%%~d\%MODEL_NAME%_loso"
)

:: 4. Run Standard Mode (80-20)
echo.
echo --- Running STANDARD Mode (80-20) ---
python ModelTraining.py --mode standard

:: Rename Standard Mode outputs
echo Renaming 80-20 outputs...
if exist "training_loss_curve.png" ren "training_loss_curve.png" "training_loss_curve_80-20.png"
if exist "best_shoulder_rcnn.pth" ren "best_shoulder_rcnn.pth" "best_shoulder_rcnn_80-20.pth"
if exist "training_dataset_distribution.txt" ren "training_dataset_distribution.txt" "training_dataset_distribution_80-20.txt"

:: 5. Run LOSO Mode
echo.
echo --- Running LOSO Mode ---
python ModelTraining.py --mode loso

:: Rename LOSO Mode outputs
echo Renaming LOSO outputs...
if exist "training_loss_curve.png" ren "training_loss_curve.png" "training_loss_curve_loso.png"
if exist "best_shoulder_rcnn.pth" ren "best_shoulder_rcnn.pth" "best_shoulder_rcnn_loso.pth"
if exist "training_dataset_distribution.txt" ren "training_dataset_distribution.txt" "training_dataset_distribution_loso.txt"

echo.
echo ========================================================
echo Batch training and file organization complete!
echo ========================================================