[CmdletBinding()]
param(
    [ValidateSet("Smoke10", "Full")]
    [string]$Mode = "Full",
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [switch]$Overwrite,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if ($Overwrite -and $Resume) {
    throw "-Overwrite and -Resume are mutually exclusive"
}

$RepoRoot = "C:\OMR_work\25-omr"
$SourceRoot = "C:\OMR_work\data\ds2_dense"
$Mapping = Join-Path $RepoRoot "slur_tie_experiments\dataset\class_mapping_curve_v2.json"
$Converter = Join-Path $RepoRoot "articulation_experiments\dataset\convert_deepscores_to_yolo.py"
$Validator = Join-Path $RepoRoot "articulation_experiments\dataset\validate_yolo_dataset.py"
$DatasetRoot = if ($Mode -eq "Full") {
    "C:\OMR_work\experiments\datasets\curve_v2_fullbbox_2048"
} else {
    "C:\OMR_work\experiments\datasets\curve_v2_smoke10"
}

foreach ($Path in @($PythonExe, $Mapping, $Converter, $Validator)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing required file: $Path"
    }
}
foreach ($Path in @(
    $SourceRoot,
    (Join-Path $SourceRoot "images"),
    (Join-Path $SourceRoot "deepscores_train.json"),
    (Join-Path $SourceRoot "deepscores_test.json")
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing DeepScores Dense input: $Path"
    }
}

$ConvertArgs = @(
    $Converter,
    "--dataset-root", $SourceRoot,
    "--images-dir", (Join-Path $SourceRoot "images"),
    "--output-dir", $DatasetRoot,
    "--class-mapping", $Mapping,
    "--tile-size", "2048",
    "--overlap", "1024",
    "--edge-policy", "shift",
    "--minimum-intersection-ratio", "0.6",
    "--minimum-tenuto-bbox-height-pixels", "0",
    "--negative-ratio", "0.20",
    "--require-full-bbox",
    "--add-target-centered-windows",
    "--seed", "20260809",
    "--png-compress-level", "1",
    "--progress-every", "100"
)
if ($Mode -eq "Smoke10") {
    $ConvertArgs += @("--max-images-per-split", "10")
}
if ($Overwrite) {
    $ConvertArgs += "--overwrite"
}
if ($Resume) {
    $ConvertArgs += "--resume"
}

Write-Host "Preparing one-class full-curve dataset ($Mode): $DatasetRoot"
& $PythonExe @ConvertArgs
if ($LASTEXITCODE -ne 0) {
    throw "Curve v2 dataset conversion failed with exit code $LASTEXITCODE"
}

& $PythonExe $Validator `
    --dataset-root $DatasetRoot `
    --source-dataset-root $SourceRoot `
    --class-mapping $Mapping
if ($LASTEXITCODE -ne 0) {
    throw "Curve v2 dataset validation failed with exit code $LASTEXITCODE"
}

Write-Host "CURVE V2 DATASET READY: $DatasetRoot"
Write-Host "Next: double-click C:\OMR_work\25-omr\START_CURVE_V2_TRAIN_30EP.bat"
