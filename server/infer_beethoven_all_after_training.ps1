[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$ErrorActionPreference = "Stop"
$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe"
$RepoRoot = "C:\OMR_work\25-omr"
$InputDir = "C:\OMR_work\experiments\inference\beethoven1_300dpi_all\pages"
$OutputRoot = "C:\OMR_work\experiments\inference\beethoven1_300dpi_all"
$SymbolOutputDir = Join-Path $OutputRoot "yolov9_dense"
$CurveOutputDir = Join-Path $OutputRoot "yolov9_curves"
$CombinedOutputDir = Join-Path $OutputRoot "combined"
$LogPath = "C:\OMR_work\experiments\inference\beethoven1_300dpi_all\after_training.log"

"WAITING for training PID $TrainingPid at $(Get-Date -Format o)" | Set-Content -LiteralPath $LogPath
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
"TRAINING PID EXITED at $(Get-Date -Format o)" | Add-Content -LiteralPath $LogPath
Start-Sleep -Seconds 60

& $PythonExe "$RepoRoot\articulation_experiments\inference\infer_yolov9_pages.py" `
    --input-dir $InputDir `
    --output-dir $SymbolOutputDir `
    --weights "C:\OMR_work\experiments\runs\yolov9_e_dense_20ep_b4_3090\weights\best.pt" `
    --data-yaml "C:\OMR_work\experiments\datasets\symbols_tiny_v2_full\dataset.yaml" `
    --class-mapping "$RepoRoot\articulation_experiments\dataset\class_mapping_extended.json" `
    --yolov9-root "C:\OMR_work\yolov9" `
    --tile-size 512 `
    --overlap 128 `
    --model-input-size 1024 `
    --edge-policy shift `
    --confidence 0.25 `
    --nms-iou 0.5 `
    --batch 4 `
    --device 0 `
    --resume *>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$CurveWeights = "$RepoRoot\slur_tie_experiments\outputs\runs\yolov9_curves_v1\weights\best.pt"
if (-not (Test-Path -LiteralPath $CurveWeights -PathType Leaf)) {
    "CURVE WEIGHTS MISSING: $CurveWeights" | Add-Content -LiteralPath $LogPath
    exit 1
}
& $PythonExe "$RepoRoot\articulation_experiments\inference\infer_yolov9_pages.py" `
    --input-dir $InputDir `
    --output-dir $CurveOutputDir `
    --weights $CurveWeights `
    --data-yaml "$RepoRoot\slur_tie_experiments\outputs\yolo_dataset_curves\dataset.yaml" `
    --class-mapping "$RepoRoot\slur_tie_experiments\dataset\class_mapping_curves.json" `
    --yolov9-root "C:\OMR_work\yolov9" `
    --tile-size 1024 `
    --overlap 256 `
    --model-input-size 1024 `
    --edge-policy pad `
    --confidence 0.25 `
    --nms-iou 0.5 `
    --batch 4 `
    --device 0 `
    --resume *>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonExe "$RepoRoot\articulation_experiments\inference\combine_page_detections.py" `
    --input-dir $InputDir `
    --symbol-dir $SymbolOutputDir `
    --curve-dir $CurveOutputDir `
    --output-dir $CombinedOutputDir *>&1 | Tee-Object -FilePath $LogPath -Append
exit $LASTEXITCODE
