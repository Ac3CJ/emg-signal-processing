@echo off
echo =========================================
echo Automatic Collection Test Validation
echo =========================================
echo.

python .\ModelValidator.py --model best_shoulder_rcnn.pth --collected 1
python .\ModelValidator.py --model best_shoulder_rcnn.pth --collected 2
python .\ModelValidator.py --model best_shoulder_rcnn.pth --collected 3

echo.
echo =========================================
echo Validation Complete!
echo =========================================
pause