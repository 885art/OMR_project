@echo off
setlocal

set "REPO_ROOT=C:\OMR_work\25-omr"
set "YOLOV9_ROOT=C:\OMR_work\yolov9"
set "DATASET_ROOT=C:\OMR_work\experiments\datasets\bps_piano50_dense_replay_v1"
set "RUNS_DIR=C:\OMR_work\experiments\runs"
set "WEIGHTS=C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_30ep_b4_3090\weights\best.pt"
set "PYTHON_EXE=C:\Users\minemine\miniconda3\envs\omr\python.exe"
set "RUN_NAME=yolov9_e_bps_piano50_finetune_v1_3090"
set "EPOCHS=30"
set "BATCH_SIZE=4"
set "RESUME_CHECKPOINT="

if defined OMR_PYTHON set "PYTHON_EXE=%OMR_PYTHON%"
if defined OMR_DATASET_ROOT set "DATASET_ROOT=%OMR_DATASET_ROOT%"
if defined OMR_WEIGHTS set "WEIGHTS=%OMR_WEIGHTS%"
if defined OMR_EPOCHS set "EPOCHS=%OMR_EPOCHS%"
if defined OMR_BATCH_SIZE set "BATCH_SIZE=%OMR_BATCH_SIZE%"
if defined OMR_RUN_NAME set "RUN_NAME=%OMR_RUN_NAME%"
if defined OMR_RESUME_CHECKPOINT set "RESUME_CHECKPOINT=%OMR_RESUME_CHECKPOINT%"
if not defined RESUME_CHECKPOINT set "RESUME_CHECKPOINT=%RUNS_DIR%\%RUN_NAME%\weights\last.pt"

if not exist "%PYTHON_EXE%" (echo [ERROR] Missing Python: %PYTHON_EXE% & exit /b 1)
if not exist "%YOLOV9_ROOT%\train_dual.py" (echo [ERROR] Missing YOLOv9: %YOLOV9_ROOT% & exit /b 1)
if not exist "%DATASET_ROOT%\dataset.yaml" (echo [ERROR] Run prepare_bps_finetune_3090.ps1 first. & exit /b 1)
if not exist "%DATASET_ROOT%\validation_report.json" (echo [ERROR] Missing validation report. & exit /b 1)
if not exist "%WEIGHTS%" (echo [ERROR] Missing Dense Piano50 checkpoint: %WEIGHTS% & exit /b 1)
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\check_yolov9_setup.py" --yolov9-root "%YOLOV9_ROOT%" --weights "%WEIGHTS%" --symbol-data "%DATASET_ROOT%\dataset.yaml" --expected-symbol-classes 50 --model-config "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml"
if errorlevel 1 exit /b %errorlevel%
if /I "%~1"=="preflight" (echo BPS Piano50 fine-tune preflight passed. & exit /b 0)
if /I "%~1"=="resume" (
  if not exist "%RESUME_CHECKPOINT%" (echo [ERROR] Missing resume checkpoint: %RESUME_CHECKPOINT% & exit /b 1)
  "%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\yolov9_compat_launcher.py" --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py --resume "%RESUME_CHECKPOINT%"
  if errorlevel 1 exit /b 1
  exit /b 0
)
if exist "%RUNS_DIR%\%RUN_NAME%" (echo [ERROR] Run already exists: %RUNS_DIR%\%RUN_NAME% & exit /b 1)

echo Starting BPS Piano50 fine-tuning: epochs=%EPOCHS% batch=%BATCH_SIZE% imgsz=1024 device=0
"%PYTHON_EXE%" "%REPO_ROOT%\articulation_experiments\train\yolov9_compat_launcher.py" --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py --workers 4 --device 0 --batch-size %BATCH_SIZE% --data "%DATASET_ROOT%\dataset.yaml" --imgsz 1024 --cfg "%YOLOV9_ROOT%\models\detect\yolov9-e.yaml" --weights "%WEIGHTS%" --hyp "%REPO_ROOT%\articulation_experiments\configs\yolov9_bps_finetune_hyp.yaml" --optimizer AdamW --epochs %EPOCHS% --patience 8 --save-period 5 --project "%RUNS_DIR%" --name "%RUN_NAME%" --seed 20260811
exit /b %errorlevel%
