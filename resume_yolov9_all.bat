@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe"
set "YOLOV9_ROOT=C:\Cheewai\OMR_work\yolov9"
set "YOLOV5_CONFIG_DIR=%~dp0articulation_experiments\outputs\yolov9_config"
set "SYMBOL_LAST=%~dp0articulation_experiments\outputs\runs\yolov9_symbols_v1\weights\last.pt"
set "CURVE_LAST=%~dp0slur_tie_experiments\outputs\runs\yolov9_curves_v1\weights\last.pt"

if exist "%SYMBOL_LAST%" (
  echo Resuming YOLOv9 symbol model...
  "%PYTHON%" articulation_experiments\train\yolov9_compat_launcher.py ^
    --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py ^
    --resume "%SYMBOL_LAST%"
  if errorlevel 1 goto :failed
)

if exist "%CURVE_LAST%" (
  echo Resuming YOLOv9 curve model...
  "%PYTHON%" articulation_experiments\train\yolov9_compat_launcher.py ^
    --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py ^
    --resume "%CURVE_LAST%"
  if errorlevel 1 goto :failed
)

if not exist "%SYMBOL_LAST%" if not exist "%CURVE_LAST%" (
  echo [ERROR] No YOLOv9 last.pt checkpoint was found.
  goto :failed
)

echo Resume command completed.
pause
exit /b 0

:failed
echo Resume stopped with an error.
pause
exit /b 1
