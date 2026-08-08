@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=C:\Users\minemine\miniconda3\envs\omr\python.exe"
set "YOLOV9_ROOT=C:\OMR_work\yolov9"
set "YOLOV5_CONFIG_DIR=%CD%\articulation_experiments\outputs\yolov9_config"
set "WEIGHTS=C:\OMR_work\weights\yolov9-e.pt"
set "DATA=C:\OMR_work\experiments\datasets\curve_v2_fullbbox_2048\dataset.yaml"
set "SYMBOL_DATA=C:\OMR_work\experiments\datasets\symbols_tiny_v2_full\dataset.yaml"
set "RUNS_DIR=C:\OMR_work\experiments\runs"
set "RUN_NAME=yolov9_e_curve_v2_fullbbox_2048_30ep_3090"
set "RUN_DIR=%RUNS_DIR%\%RUN_NAME%"
set "EPOCHS=30"
set "BATCH_SIZE=3"

if defined OMR_PYTHON set "PYTHON=%OMR_PYTHON%"
if defined OMR_EPOCHS set "EPOCHS=%OMR_EPOCHS%"
if defined OMR_BATCH_SIZE set "BATCH_SIZE=%OMR_BATCH_SIZE%"
if defined OMR_RUN_NAME set "RUN_NAME=%OMR_RUN_NAME%"
set "RUN_DIR=%RUNS_DIR%\%RUN_NAME%"

if not exist "%PYTHON%" (echo [ERROR] Missing Python: %PYTHON% & goto :failed)
if not exist "%YOLOV9_ROOT%\train_dual.py" (echo [ERROR] Missing YOLOv9: %YOLOV9_ROOT% & goto :failed)
if not exist "%WEIGHTS%" (echo [ERROR] Missing pretrained weights: %WEIGHTS% & goto :failed)
if not exist "%DATA%" (echo [ERROR] Missing curve v2 dataset: %DATA% & goto :failed)
if exist "%RUN_DIR%" (
  echo [ERROR] Run already exists: %RUN_DIR%
  echo Set another OMR_RUN_NAME if you intentionally want a separate run.
  goto :failed
)

"%PYTHON%" articulation_experiments\train\check_yolov9_setup.py ^
  --yolov9-root "%YOLOV9_ROOT%" ^
  --weights "%WEIGHTS%" ^
  --symbol-data "%SYMBOL_DATA%" ^
  --curve-data "%DATA%" ^
  --expected-symbol-classes 40 ^
  --expected-curve-classes 1 ^
  --model-config "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml"
if errorlevel 1 goto :failed
if /I "%~1"=="preflight" (
  echo YOLOv9-E curve v2 preflight passed.
  exit /b 0
)

echo Starting YOLOv9-E curve v2: epochs=%EPOCHS% batch=%BATCH_SIZE% imgsz=1280 device=0
"%PYTHON%" articulation_experiments\train\yolov9_compat_launcher.py ^
  --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py ^
  --workers 4 --device 0 --batch-size %BATCH_SIZE% ^
  --data "%DATA%" --imgsz 1280 ^
  --cfg "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml" ^
  --weights "%WEIGHTS%" ^
  --hyp "%CD%\articulation_experiments\configs\yolov9_score_hyp.yaml" ^
  --optimizer AdamW --epochs %EPOCHS% --patience 10 --save-period 5 ^
  --project "%RUNS_DIR%" ^
  --name "%RUN_NAME%" --seed 20260809
if errorlevel 1 goto :failed

echo YOLOv9 curve v2 training completed.
pause
exit /b 0

:failed
pause
exit /b 1
