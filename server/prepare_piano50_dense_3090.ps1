[CmdletBinding()]
param(
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [double]$ParenthesisFraction = 0.20,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$Repo = "C:\OMR_work\25-omr"
$Source = "C:\OMR_work\data\ds2_dense"
$Base = if ($Smoke) {
    "C:\OMR_work\experiments\datasets\piano50_dense_smoke"
} else {
    "C:\OMR_work\experiments\datasets\piano50_dense"
}
$Combined = if ($Smoke) {
    "C:\OMR_work\experiments\datasets\piano50_dense_parentheses_smoke"
} else {
    "C:\OMR_work\experiments\datasets\piano50_dense_parentheses"
}
$Mapping = Join-Path $Repo "articulation_experiments\dataset\class_mapping_piano.json"
$Convert = Join-Path $Repo "articulation_experiments\dataset\convert_deepscores_to_yolo.py"
$Validate = Join-Path $Repo "articulation_experiments\dataset\validate_yolo_dataset.py"
$Augment = Join-Path $Repo "articulation_experiments\dataset\augment_parentheses.py"

foreach ($Path in @($PythonExe, $Mapping, $Convert, $Validate, $Augment)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing: $Path" }
}
if (Test-Path -LiteralPath $Base) { throw "Dataset already exists: $Base" }
if (Test-Path -LiteralPath $Combined) { throw "Dataset already exists: $Combined" }

$ConvertArgs = @(
    $Convert,
    "--dataset-root", $Source,
    "--images-dir", (Join-Path $Source "images"),
    "--output-dir", $Base,
    "--class-mapping", $Mapping,
    "--tile-size", "512",
    "--overlap", "128",
    "--edge-policy", "shift",
    "--minimum-intersection-ratio", "0.6",
    "--minimum-tenuto-bbox-height-pixels", "8",
    "--negative-ratio", "0.25",
    "--seed", "20260805",
    "--png-compress-level", "1",
    "--progress-every", "100"
)
if ($Smoke) { $ConvertArgs += @("--max-images-per-split", "10") }

& $PythonExe @ConvertArgs
if ($LASTEXITCODE -ne 0) { throw "Dense conversion failed: $LASTEXITCODE" }
& $PythonExe $Validate --dataset-root $Base --source-dataset-root $Source --class-mapping $Mapping
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed: $LASTEXITCODE" }
& $PythonExe $Augment --dataset-root $Base --output-dir $Combined `
    --fraction $ParenthesisFraction --copies-per-image 1 --include-originals
if ($LASTEXITCODE -ne 0) { throw "Parenthesis augmentation failed: $LASTEXITCODE" }

Write-Host "PIANO-50 DATASET READY: $Combined"
Write-Host "Includes Dense originals, tuplet classes, and parenthesized views."
