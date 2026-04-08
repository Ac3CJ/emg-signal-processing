@echo off
echo =========================================
echo Automatic Collection Test Validation
echo =========================================
echo.

python .\ModelValidator.py --model best_shoulder_rcnn_loso.pth --validate_predefined --validate_ensemble
python .\ModelValidator.py --model best_shoulder_rcnn_loso.pth --collected 1
python .\ModelValidator.py --model best_shoulder_rcnn_loso.pth --collected 2
python .\ModelValidator.py --model best_shoulder_rcnn_loso.pth --collected 3

echo.
echo =========================================
echo Validation Complete!
echo =========================================
pause