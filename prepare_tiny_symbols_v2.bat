@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe"
set "SOURCE=C:\Cheewai\OMR_work\ds2_dense"
set "OUTPUT=%~dp0articulation_experiments\outputs\yolo_dataset_extended_tiny_v2"
set "MAPPING=%~dp0articulation_experiments\dataset\class_mapping_extended.json"

if not exist "%PYTHON%" goto :missing
if not exist "%SOURCE%\deepscores_train.json" goto :missing

"%PYTHON%" articulation_experiments\dataset\convert_deepscores_to_yolo.py ^
  --dataset-root "%SOURCE%" ^
  --output-dir "%OUTPUT%" ^
  --class-mapping "%MAPPING%" ^
  --tile-size 512 --overlap 128 --edge-policy shift ^
  --minimum-intersection-ratio 0.6 ^
  --minimum-tenuto-bbox-height-pixels 8 ^
  --negative-ratio 0.25 --png-compress-level 1 ^
  --overwrite
if errorlevel 1 goto :failed

"%PYTHON%" articulation_experiments\dataset\validate_yolo_dataset.py ^
  --dataset-root "%OUTPUT%" ^
  --source-dataset-root "%SOURCE%" ^
  --class-mapping "%MAPPING%"
if errorlevel 1 goto :failed

echo Tiny-object v2 dataset is ready: %OUTPUT%
pause
exit /b 0

:missing
echo [ERROR] Python or ds2_dense is missing.
:failed
pause
exit /b 1
