[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Smoke",
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [string]$CompleteRoot = "C:\OMR_work\data\ds2_complete",
    [string]$WorkRoot = "C:\OMR_work\experiments",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Mapping = Join-Path $RepoRoot "articulation_experiments\dataset\class_mapping_piano.json"
$Merger = Join-Path $RepoRoot "articulation_experiments\dataset\merge_deepscores_complete.py"
$Converter = Join-Path $RepoRoot "articulation_experiments\dataset\convert_deepscores_to_yolo.py"
$Validator = Join-Path $RepoRoot "articulation_experiments\dataset\validate_yolo_dataset.py"
$Suffix = if ($Mode -eq "Smoke") { "smoke10" } else { "full" }
$MergedRoot = Join-Path $WorkRoot "datasets\deepscores_complete_piano50_${Suffix}_merged"
$DatasetRoot = Join-Path $WorkRoot "datasets\piano50_complete_$Suffix"

foreach ($Path in @(
    $PythonExe, $Mapping, $Merger, $Converter, $Validator,
    $CompleteRoot, (Join-Path $CompleteRoot "images")
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing Complete preparation input: $Path"
    }
}

$MergedFiles = @(
    (Join-Path $MergedRoot "deepscores_train.json"),
    (Join-Path $MergedRoot "deepscores_test.json"),
    (Join-Path $MergedRoot "merge_statistics.json")
)
$MergedReady = @($MergedFiles | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }).Count -eq 3
if ($MergedReady -and -not $Overwrite) {
    Write-Host "Reusing merged Complete Piano50 source: $MergedRoot"
} else {
    $MergeArgs = @(
        $Merger,
        "--complete-root", $CompleteRoot,
        "--output-dir", $MergedRoot,
        "--class-mapping", $Mapping,
        "--cross-split-policy", "train",
        "--all-available-shards"
    )
    if ($Mode -eq "Smoke") {
        $MergeArgs += @("--max-images-per-shard", "10")
    }
    if ($Overwrite) {
        $MergeArgs += "--overwrite"
    }
    & $PythonExe @MergeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Complete Piano50 merge failed with exit code $LASTEXITCODE"
    }
}

$DatasetComplete =
    (Test-Path -LiteralPath (Join-Path $DatasetRoot "dataset.yaml") -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $DatasetRoot "statistics.json") -PathType Leaf)
if ($DatasetComplete -and -not $Overwrite) {
    Write-Host "Reusing converted Complete Piano50 dataset: $DatasetRoot"
} else {
    $ConvertArgs = @(
        $Converter,
        "--dataset-root", $MergedRoot,
        "--images-dir", (Join-Path $CompleteRoot "images"),
        "--output-dir", $DatasetRoot,
        "--class-mapping", $Mapping,
        "--tile-size", "512",
        "--overlap", "128",
        "--edge-policy", "shift",
        "--minimum-intersection-ratio", "0.6",
        "--minimum-tenuto-bbox-height-pixels", "8",
        "--negative-ratio", "0.25",
        "--seed", "20260811",
        "--png-compress-level", "1",
        "--progress-every", "100"
    )
    if ($Overwrite) {
        $ConvertArgs += "--overwrite"
    } elseif (Test-Path -LiteralPath $DatasetRoot -PathType Container) {
        $ConvertArgs += "--resume"
        Write-Host "Resuming incomplete Complete conversion: $DatasetRoot"
    }
    & $PythonExe @ConvertArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Complete Piano50 conversion failed with exit code $LASTEXITCODE"
    }
}

& $PythonExe $Validator `
    --dataset-root $DatasetRoot `
    --source-dataset-root $MergedRoot `
    --class-mapping $Mapping
if ($LASTEXITCODE -ne 0) {
    throw "Complete Piano50 validation failed with exit code $LASTEXITCODE"
}

Write-Host "COMPLETE PIANO50 DATASET READY: $DatasetRoot"
Write-Host "No training was started."
