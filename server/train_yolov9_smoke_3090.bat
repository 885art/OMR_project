@echo off
setlocal

set "REPO_ROOT=C:\OMR_work\25-omr"
set "YOLOV9_ROOT=C:\OMR_work\yolov9"
set "DATASET_ROOT=C:\OMR_work\experiments\datasets\symbols_tiny_v2_smoke"
set "RUNS_DIR=C:\OMR_work\experiments\runs"
set "WEIGHTS=C:\OMR_work\weights\yolov9-e.pt"
set "PYTHON_EXE=C:\Users\minemine\miniconda3\envs\omr\python.exe"
set "RUN_NAME=yolov9_e_symbols_tiny_v2_smoke_3090"

if defined OMR_PYTHON set "PYTHON_EXE=%OMR_PYTHON%"
if not exist "%PYTHON_EXE%" (echo Missing Python: %PYTHON_EXE% & exit /b 1)
if not exist "%YOLOV9_ROOT%\train_dual.py" (echo Missing train_dual.py & exit /b 1)
if not exist "%YOLOV9_ROOT%\val_dual.py" (echo Missing val_dual.py & exit /b 1)
if not exist "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml" (echo Missing yolov9-e.yaml & exit /b 1)
if not exist "%WEIGHTS%" (echo Missing weights: %WEIGHTS% & exit /b 1)
if not exist "%DATASET_ROOT%\dataset.yaml" (echo Missing dataset.yaml & exit /b 1)
if exist "%RUNS_DIR%\%RUN_NAME%" (echo Run already exists: %RUNS_DIR%\%RUN_NAME% & exit /b 1)

if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\check_yolov9_setup.py" ^
  --yolov9-root "%YOLOV9_ROOT%" ^
  --weights "%WEIGHTS%" ^
  --symbol-data "%DATASET_ROOT%\dataset.yaml" ^
  --expected-symbol-classes 40 ^
  --model-config "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml"
if errorlevel 1 exit /b %errorlevel%
if /I "%~1"=="preflight" (
  echo YOLOv9-E smoke preflight passed.
  exit /b 0
)

echo Starting YOLOv9-E smoke: epochs=1 batch=2 imgsz=1024 device=0 workers=4
echo Watch CUDA utilization and dedicated memory with: nvidia-smi -l 1
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\yolov9_compat_launcher.py" ^
  --yolov9-root "%YOLOV9_ROOT%" ^
  --script train_dual.py ^
  --workers 4 ^
  --device 0 ^
  --batch-size 2 ^
  --data "%DATASET_ROOT%\dataset.yaml" ^
  --imgsz 1024 ^
  --cfg "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml" ^
  --weights "%WEIGHTS%" ^
  --hyp "%REPO_ROOT%\articulation_experiments\configs\yolov9_score_hyp.yaml" ^
  --optimizer AdamW ^
  --epochs 1 ^
  --patience 20 ^
  --save-period 10 ^
  --project "%RUNS_DIR%" ^
  --name "%RUN_NAME%" ^
  --seed 20260730
if errorlevel 1 exit /b %errorlevel%

if not exist "%RUNS_DIR%\%RUN_NAME%\weights\best.pt" (echo Missing best.pt after training & exit /b 1)
if not exist "%RUNS_DIR%\%RUN_NAME%\weights\last.pt" (echo Missing last.pt after training & exit /b 1)

rem Official train_dual.py uses 2x the training batch for its integrated validation.
rem Run the requested independent validation explicitly with batch 2 as well.
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\yolov9_compat_launcher.py" ^
  --yolov9-root "%YOLOV9_ROOT%" ^
  --script val_dual.py ^
  --data "%DATASET_ROOT%\dataset.yaml" ^
  --weights "%RUNS_DIR%\%RUN_NAME%\weights\best.pt" ^
  --imgsz 1024 ^
  --batch-size 2 ^
  --device 0 ^
  --workers 4 ^
  --project "%RUNS_DIR%" ^
  --name "%RUN_NAME%_validation_batch2" ^
  --exist-ok
exit /b %errorlevel%
