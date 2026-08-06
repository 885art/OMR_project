@echo off
setlocal

set "REPO_ROOT=C:\OMR_work\25-omr"
set "RUNS_DIR=C:\OMR_work\experiments\runs"
set "WEIGHTS=C:\OMR_work\weights\yolo11n.pt"
set "PYTHON_EXE=C:\Users\minemine\miniconda3\envs\omr\python.exe"
set "DATASET_ROOT=C:\OMR_work\experiments\datasets\symbols_tiny_v2_full"
set "RUN_NAME=yolo11n_symbols_tiny_v2_compare_3090"
set "EPOCHS=30"
set "BATCH_SIZE=8"

if /I "%~1"=="smoke" (
  set "DATASET_ROOT=C:\OMR_work\experiments\datasets\symbols_tiny_v2_smoke"
  set "RUN_NAME=yolo11n_symbols_tiny_v2_smoke_3090"
  set "EPOCHS=1"
)
if defined OMR_PYTHON set "PYTHON_EXE=%OMR_PYTHON%"
if defined OMR_BATCH_SIZE set "BATCH_SIZE=%OMR_BATCH_SIZE%"
if not exist "%PYTHON_EXE%" (echo Missing Python: %PYTHON_EXE% & exit /b 1)
if not exist "%DATASET_ROOT%\dataset.yaml" (echo Missing dataset: %DATASET_ROOT% & exit /b 1)
if not exist "%WEIGHTS%" (echo Missing weights: %WEIGHTS% & exit /b 1)
if exist "%RUNS_DIR%\%RUN_NAME%" (echo Run already exists: %RUNS_DIR%\%RUN_NAME% & exit /b 1)
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

set "YOLO_CONFIG_DIR=C:\OMR_work\experiments\ultralytics_config"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
pushd "%YOLO_CONFIG_DIR%"

echo Starting YOLO11n: epochs=%EPOCHS% batch=%BATCH_SIZE% imgsz=1024 device=0
if /I "%~2"=="preflight" (
  "%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\train_baseline.py" --config "%REPO_ROOT%\articulation_experiments\configs\yolo11_tiny_symbols_v2.yaml" --dataset-root "%DATASET_ROOT%" --runs-dir "%RUNS_DIR%" --run-name "%RUN_NAME%" --epochs %EPOCHS% --batch %BATCH_SIZE% --device 0 --initial-weights "%WEIGHTS%" --preflight-only
  if errorlevel 1 (
    popd
    exit /b 1
  )
  popd
  exit /b 0
)
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\train_baseline.py" --config "%REPO_ROOT%\articulation_experiments\configs\yolo11_tiny_symbols_v2.yaml" --dataset-root "%DATASET_ROOT%" --runs-dir "%RUNS_DIR%" --run-name "%RUN_NAME%" --epochs %EPOCHS% --batch %BATCH_SIZE% --device 0 --initial-weights "%WEIGHTS%"
set "TRAIN_RC=%errorlevel%"
popd
exit /b %TRAIN_RC%
