@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

set "MODEL_ROOT=%CD%\neural-network-models"
set "SMOOTH_ALPHA=0.05"
set "RUN_NAMES=tfFroz_sP8 tfNotFroz_sP8"

if not exist "%MODEL_ROOT%" (
	echo Error: model root not found: "%MODEL_ROOT%"
	popd
	exit /b 1
)

echo =========================================
echo Validating secondary sP8 runs
echo =========================================

for %%R in (%RUN_NAMES%) do (
	set "RUN_NAME=%%R"
	if exist "%MODEL_ROOT%\!RUN_NAME!\training\best_shoulder_rcnn.pth" (
		echo.
		echo -----------------------------------------
		echo Validating !RUN_NAME! using secondary benchmark data
		echo -----------------------------------------
		python .\ModelValidator.py --name "!RUN_NAME!" --model "%MODEL_ROOT%\!RUN_NAME!\training\best_shoulder_rcnn.pth" --validate_predefined --arch rcnn
		if errorlevel 1 (
			echo Error: validation failed for !RUN_NAME!
			popd
			exit /b 1
		)
	) else (
		echo Skipping !RUN_NAME!: missing training\best_shoulder_rcnn.pth
	)
)

echo.
echo =========================================
echo Applying EMA smoothing with alpha %SMOOTH_ALPHA%
echo =========================================

python .\SmoothRawKinematics.py --alpha %SMOOTH_ALPHA% --root "%MODEL_ROOT%" --runs %RUN_NAMES%
if errorlevel 1 (
	echo Error: smoothing failed.
	popd
	exit /b 1
)

echo.
echo Done.
popd
exit /b 0