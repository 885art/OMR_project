@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe"
set "YOLOV9_ROOT=C:\Cheewai\OMR_work\yolov9"
set "YOLOV5_CONFIG_DIR=%~dp0articulation_experiments\outputs\yolov9_config"
set "WEIGHTS=%~dp0articulation_experiments\outputs\pretrained\yolov9-s.pt"
set "DATA=%~dp0slur_tie_experiments\outputs\yolo_dataset_curves\dataset.yaml"
set "RUN_DIR=%~dp0slur_tie_experiments\outputs\runs\yolov9_curves_v1"

if not exist "%PYTHON%" goto :missing
if not exist "%YOLOV9_ROOT%\train_dual.py" goto :missing
if not exist "%WEIGHTS%" goto :missing
if exist "%RUN_DIR%" (
  echo [ERROR] Run already exists: %RUN_DIR%
  echo Use resume_yolov9_all.bat if training was interrupted.
  goto :failed
)

"%PYTHON%" articulation_experiments\train\yolov9_compat_launcher.py ^
  --yolov9-root "%YOLOV9_ROOT%" --script train_dual.py ^
  --workers 2 --device 0 --batch-size 2 ^
  --data "%DATA%" --imgsz 1024 ^
  --cfg "%YOLOV9_ROOT%\models\detect\yolov9-s.yaml" ^
  --weights "%WEIGHTS%" ^
  --hyp "%~dp0articulation_experiments\configs\yolov9_score_hyp.yaml" ^
  --optimizer AdamW --epochs 60 --patience 18 --save-period 10 ^
  --project "%~dp0slur_tie_experiments\outputs\runs" ^
  --name yolov9_curves_v1 --seed 20260723
if errorlevel 1 goto :failed

echo YOLOv9 slur/tie training completed.
exit /b 0

:missing
echo [ERROR] Python, official YOLOv9, or pretrained weights are missing.
:failed
if not defined OMR_YOLOV9_CHAIN pause
exit /b 1
