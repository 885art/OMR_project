@echo off
setlocal

set "REPO_ROOT=C:\OMR_work\25-omr"
set "YOLOV9_ROOT=C:\OMR_work\yolov9"
set "DATASET_ROOT=C:\OMR_work\experiments\datasets\symbols_tiny_v2_full"
set "RUNS_DIR=C:\OMR_work\experiments\runs"
set "WEIGHTS=C:\OMR_work\weights\yolov9-e.pt"
set "PYTHON_EXE=C:\Users\minemine\miniconda3\envs\omr\python.exe"
set "RUN_NAME=yolov9_e_symbols_tiny_v2_pilot_3090"
set "EPOCHS=30"
set "BATCH_SIZE=2"

if defined OMR_PYTHON set "PYTHON_EXE=%OMR_PYTHON%"
if defined OMR_EPOCHS set "EPOCHS=%OMR_EPOCHS%"
if defined OMR_BATCH_SIZE set "BATCH_SIZE=%OMR_BATCH_SIZE%"
if defined OMR_RUN_NAME set "RUN_NAME=%OMR_RUN_NAME%"
if not exist "%PYTHON_EXE%" (echo Missing Python: %PYTHON_EXE% & exit /b 1)
if not exist "%DATASET_ROOT%\dataset.yaml" (echo Missing full dataset: %DATASET_ROOT% & exit /b 1)
if not exist "%WEIGHTS%" (echo Missing weights: %WEIGHTS% & exit /b 1)
if exist "%RUNS_DIR%\%RUN_NAME%" (echo Run already exists: %RUNS_DIR%\%RUN_NAME% & exit /b 1)
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\check_yolov9_setup.py" --yolov9-root "%YOLOV9_ROOT%" --weights "%WEIGHTS%" --symbol-data "%DATASET_ROOT%\dataset.yaml" --expected-symbol-classes 40 --model-config "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml"
if errorlevel 1 exit /b %errorlevel%
if /I "%~1"=="preflight" (
  echo YOLOv9-E pilot preflight passed.
  exit /b 0
)

echo Starting YOLOv9-E pilot: epochs=%EPOCHS% batch=%BATCH_SIZE% imgsz=1024 device=0
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\yolov9_compat_launcher.py" --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py --workers 4 --device 0 --batch-size %BATCH_SIZE% --data "%DATASET_ROOT%\dataset.yaml" --imgsz 1024 --cfg "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml" --weights "%WEIGHTS%" --hyp "%REPO_ROOT%\articulation_experiments\configs\yolov9_score_hyp.yaml" --optimizer AdamW --epochs %EPOCHS% --patience 20 --save-period 10 --project "%RUNS_DIR%" --name "%RUN_NAME%" --seed 20260730
exit /b %errorlevel%
