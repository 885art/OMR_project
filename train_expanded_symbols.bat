@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] Python environment not found: %PYTHON%
  pause
  exit /b 1
)

echo [1/2] Checking the 17-class dataset...
"%PYTHON%" articulation_experiments\train\train_baseline.py ^
  --config articulation_experiments\configs\expanded_symbols.yaml ^
  --preflight-only
if errorlevel 1 goto :failed

echo [2/2] Starting YOLO11n expanded-symbol training...
"%PYTHON%" articulation_experiments\train\train_baseline.py ^
  --config articulation_experiments\configs\expanded_symbols.yaml
if errorlevel 1 goto :failed

echo.
echo Training completed successfully.
echo Weights: articulation_experiments\outputs\runs\expanded_symbols_v1\weights\best.pt
pause
exit /b 0

:failed
echo.
echo Training stopped with an error. Copy the last error message for diagnosis.
pause
exit /b 1
