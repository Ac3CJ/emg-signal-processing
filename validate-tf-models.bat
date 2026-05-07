@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

set "MODEL_ROOT=%CD%\neural-network-models"
set "SMOOTH_ALPHA=0.05"
set "RUN_PREFIXES="

if not exist "%MODEL_ROOT%" (
	echo Error: model root not found: "%MODEL_ROOT%"
	popd
	exit /b 1
)

echo =========================================
echo Validating all tf-prefixed runs
echo =========================================

for /d %%D in ("%MODEL_ROOT%\tf*") do (
	set "RUN_NAME=%%~nxD"
	for /f "tokens=1 delims=_" %%P in ("!RUN_NAME!") do set "RUN_PREFIX=%%P"

	if exist "%%D\training\best_shoulder_rcnn.pth" (
		echo.
		echo -----------------------------------------
		echo Validating !RUN_NAME!
		echo -----------------------------------------
		python .\ModelValidator.py --name "!RUN_NAME!" --model "%%D\training\best_shoulder_rcnn.pth" --validate_all --arch rcnn
		if errorlevel 1 (
			echo Error: validation failed for !RUN_NAME!
			popd
			exit /b 1
		)

		if defined RUN_PREFIXES (
			set "RUN_PREFIXES=!RUN_PREFIXES! !RUN_PREFIX!"
		) else (
			set "RUN_PREFIXES=!RUN_PREFIX!"
		)
	) else (
		echo Skipping !RUN_NAME!: missing training\best_shoulder_rcnn.pth
	)
)

if not defined RUN_PREFIXES (
	echo No tf-prefixed runs were found.
	popd
	exit /b 0
)

echo.
echo =========================================
echo Applying EMA smoothing with alpha %SMOOTH_ALPHA%
echo =========================================

python .\SmoothRawKinematics.py --alpha %SMOOTH_ALPHA% --root "%MODEL_ROOT%" --prefixes %RUN_PREFIXES%
if errorlevel 1 (
	echo Error: smoothing failed.
	popd
	exit /b 1
)

echo.
echo Done.
popd
exit /b 0