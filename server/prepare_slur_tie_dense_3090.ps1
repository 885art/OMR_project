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
$Mapping = Join-Path $RepoRoot "slur_tie_experiments\dataset\class_mapping_curves.json"
$Converter = Join-Path $RepoRoot "articulation_experiments\dataset\convert_deepscores_to_yolo.py"
$Validator = Join-Path $RepoRoot "articulation_experiments\dataset\validate_yolo_dataset.py"

if ($Mode -eq "Full") {
    # Keep this path aligned with omr/slur_tie.py and train_yolov9_curves.bat.
    $DatasetRoot = Join-Path $RepoRoot "slur_tie_experiments\outputs\yolo_dataset_curves"
} else {
    $DatasetRoot = "C:\OMR_work\experiments\datasets\slur_tie_dense_smoke10"
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
    "--tile-size", "1024",
    "--overlap", "256",
    "--edge-policy", "pad",
    "--minimum-intersection-ratio", "0.6",
    "--minimum-tenuto-bbox-height-pixels", "0",
    "--negative-ratio", "0.25",
    "--seed", "20260723",
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

Write-Host "Preparing 2-class slur/tie dataset ($Mode): $DatasetRoot"
& $PythonExe @ConvertArgs
if ($LASTEXITCODE -ne 0) {
    throw "Curve dataset conversion failed with exit code $LASTEXITCODE"
}

& $PythonExe $Validator `
    --dataset-root $DatasetRoot `
    --source-dataset-root $SourceRoot `
    --class-mapping $Mapping
if ($LASTEXITCODE -ne 0) {
    throw "Curve dataset validation failed with exit code $LASTEXITCODE"
}

Write-Host "SLUR/TIE DATASET READY: $DatasetRoot"
Write-Host "Next: double-click C:\OMR_work\25-omr\train_yolov9_curves.bat"
