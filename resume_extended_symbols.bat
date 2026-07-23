@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe"
set "CHECKPOINT=%~dp0articulation_experiments\outputs\runs\extended_symbols_v2\weights\last.pt"
if not exist "%PYTHON%" (
  echo [ERROR] Python environment not found: %PYTHON%
  pause
  exit /b 1
)
if not exist "%CHECKPOINT%" (
  echo [ERROR] Resume checkpoint not found: %CHECKPOINT%
  echo Run train_extended_symbols.bat first.
  pause
  exit /b 1
)

echo Resuming 40-class training from last.pt...
"%PYTHON%" articulation_experiments\train\train_baseline.py ^
  --config articulation_experiments\configs\extended_symbols.yaml ^
  --resume "%CHECKPOINT%"
if errorlevel 1 goto :failed

echo.
echo Training completed successfully.
pause
exit /b 0

:failed
echo.
echo Resume stopped with an error. Copy the last error message for diagnosis.
pause
exit /b 1
