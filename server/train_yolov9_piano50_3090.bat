@echo off
setlocal

set "REPO_ROOT=C:\OMR_work\25-omr"
set "YOLOV9_ROOT=C:\OMR_work\yolov9"
set "DATASET_ROOT=C:\OMR_work\experiments\datasets\piano50_dense_parentheses"
set "RUNS_DIR=C:\OMR_work\experiments\runs"
set "WEIGHTS=C:\OMR_work\weights\yolov9-e.pt"
set "PYTHON_EXE=C:\Users\minemine\miniconda3\envs\omr\python.exe"
set "RUN_NAME=yolov9_e_piano50_dense_parentheses_3090"
set "EPOCHS=20"
set "BATCH_SIZE=4"

if defined OMR_PYTHON set "PYTHON_EXE=%OMR_PYTHON%"
if defined OMR_DATASET_ROOT set "DATASET_ROOT=%OMR_DATASET_ROOT%"
if defined OMR_EPOCHS set "EPOCHS=%OMR_EPOCHS%"
if defined OMR_BATCH_SIZE set "BATCH_SIZE=%OMR_BATCH_SIZE%"
if defined OMR_RUN_NAME set "RUN_NAME=%OMR_RUN_NAME%"
if not exist "%PYTHON_EXE%" (echo Missing Python: %PYTHON_EXE% & exit /b 1)
if not exist "%DATASET_ROOT%\dataset.yaml" (echo Run prepare_piano50_dense_3090.ps1 first. & exit /b 1)
if not exist "%WEIGHTS%" (echo Missing weights: %WEIGHTS% & exit /b 1)
if exist "%RUNS_DIR%\%RUN_NAME%" (echo Run already exists: %RUNS_DIR%\%RUN_NAME% & exit /b 1)
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\check_yolov9_setup.py" --yolov9-root "%YOLOV9_ROOT%" --weights "%WEIGHTS%" --symbol-data "%DATASET_ROOT%\dataset.yaml" --expected-symbol-classes 50 --model-config "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml"
if errorlevel 1 exit /b %errorlevel%
if /I "%~1"=="preflight" (echo Piano-50 preflight passed. & exit /b 0)

echo Starting YOLOv9-E piano-50: epochs=%EPOCHS% batch=%BATCH_SIZE% imgsz=1024 device=0
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\yolov9_compat_launcher.py" --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py --workers 4 --device 0 --batch-size %BATCH_SIZE% --data "%DATASET_ROOT%\dataset.yaml" --imgsz 1024 --cfg "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml" --weights "%WEIGHTS%" --hyp "%REPO_ROOT%\articulation_experiments\configs\yolov9_score_hyp.yaml" --optimizer AdamW --epochs %EPOCHS% --patience 8 --save-period 5 --project "%RUNS_DIR%" --name "%RUN_NAME%" --seed 20260805
if errorlevel 1 exit /b %errorlevel%
set "BEST_WEIGHTS=%RUNS_DIR%\%RUN_NAME%\weights\best.pt"
if not exist "%BEST_WEIGHTS%" (echo Training ended without best.pt: %BEST_WEIGHTS% & exit /b 1)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\server\activate_piano50_model.ps1" -Weights "%BEST_WEIGHTS%" -DataYaml "%DATASET_ROOT%\dataset.yaml"
if errorlevel 1 exit /b %errorlevel%
echo Training complete and piano-50 best.pt is now active.
exit /b 0
