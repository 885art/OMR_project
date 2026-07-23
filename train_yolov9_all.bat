@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe"
set "YOLOV9_ROOT=C:\Cheewai\OMR_work\yolov9"
set "WEIGHTS=%~dp0articulation_experiments\outputs\pretrained\yolov9-s.pt"
set "SYMBOL_DATA=%~dp0articulation_experiments\outputs\yolo_dataset_extended\dataset.yaml"
set "CURVE_DATA=%~dp0slur_tie_experiments\outputs\yolo_dataset_curves\dataset.yaml"

echo [1/3] Checking official YOLOv9, CUDA, and datasets...
"%PYTHON%" articulation_experiments\train\check_yolov9_setup.py ^
  --yolov9-root "%YOLOV9_ROOT%" ^
  --weights "%WEIGHTS%" ^
  --symbol-data "%SYMBOL_DATA%" ^
  --curve-data "%CURVE_DATA%"
if errorlevel 1 goto :failed

set "OMR_YOLOV9_CHAIN=1"
echo [2/3] Training 40-class YOLOv9 symbol detector...
call train_yolov9_symbols.bat
if errorlevel 1 goto :failed

echo [3/3] Training YOLOv9 slur/tie detector...
call train_yolov9_curves.bat
if errorlevel 1 goto :failed

echo.
echo Both YOLOv9 models completed successfully.
pause
exit /b 0

:failed
echo.
echo YOLOv9 training stopped with an error. Copy the last error message.
pause
exit /b 1
