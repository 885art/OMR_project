[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Full",
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [string]$DenseRoot = "C:\OMR_work\data\ds2_dense",
    [string]$WorkRoot = "C:\OMR_work\experiments",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DatasetTools = Join-Path $RepoRoot "articulation_experiments\dataset"
$MappingGenerator = Join-Path $DatasetTools "generate_deepscores_all_mapping.py"
$Converter = Join-Path $DatasetTools "convert_deepscores_to_yolo.py"
$Validator = Join-Path $DatasetTools "validate_yolo_dataset.py"
$Mapping = Join-Path $WorkRoot "mappings\class_mapping_deepscores_all136.json"
$DatasetName = if ($Mode -eq "Smoke") { "deepscores_dense_all136_1024_smoke10" } else { "deepscores_dense_all136_1024" }
$DatasetRoot = Join-Path $WorkRoot "datasets\$DatasetName"

foreach ($Path in @(
    $PythonExe, $MappingGenerator, $Converter, $Validator, $DenseRoot,
    (Join-Path $DenseRoot "images"),
    (Join-Path $DenseRoot "deepscores_train.json"),
    (Join-Path $DenseRoot "deepscores_test.json")
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing Dense all-class preparation input: $Path"
    }
}

& $PythonExe $MappingGenerator `
    --source-json (Join-Path $DenseRoot "deepscores_train.json") `
    --output $Mapping `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "All-class mapping generation failed" }

$DatasetComplete =
    (Test-Path -LiteralPath (Join-Path $DatasetRoot "dataset.yaml") -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $DatasetRoot "statistics.json") -PathType Leaf)
if ($DatasetComplete -and -not $Overwrite) {
    Write-Host "Reusing converted Dense all136 dataset: $DatasetRoot"
} else {
    $Arguments = @(
        $Converter,
        "--dataset-root", $DenseRoot,
        "--images-dir", (Join-Path $DenseRoot "images"),
        "--output-dir", $DatasetRoot,
        "--class-mapping", $Mapping,
        "--tile-size", "1024",
        "--overlap", "256",
        "--edge-policy", "shift",
        "--minimum-intersection-ratio", "0.6",
        "--minimum-tenuto-bbox-height-pixels", "8",
        "--negative-ratio", "0.05",
        "--seed", "20260811",
        "--png-compress-level", "1",
        "--progress-every", "20"
    )
    if ($Mode -eq "Smoke") {
        $Arguments += @("--max-images-per-split", "10")
    }
    if ($Overwrite) {
        $Arguments += "--overwrite"
    } elseif (Test-Path -LiteralPath $DatasetRoot -PathType Container) {
        $Arguments += "--resume"
        Write-Host "Resuming interrupted Dense all136 conversion: $DatasetRoot"
    }
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dense all136 conversion failed with exit code $LASTEXITCODE"
    }
}

& $PythonExe $Validator `
    --dataset-root $DatasetRoot `
    --source-dataset-root $DenseRoot `
    --class-mapping $Mapping
if ($LASTEXITCODE -ne 0) {
    throw "Dense all136 validation failed with exit code $LASTEXITCODE"
}

Write-Host "DENSE ALL136 DATASET READY: $DatasetRoot"
Write-Host "Mapping: $Mapping"
Write-Host "No training was started."
