[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Full",
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [string]$SourceRoot = "C:\OMR_work\BPS-OMRv01",
    [string]$WorkRoot = "C:\OMR_work\experiments",
    [double]$ReplayRatio = 1.0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DatasetDir = Join-Path $RepoRoot "articulation_experiments\dataset"
$Converter = Join-Path $DatasetDir "convert_bps_yolo_finetune.py"
$Composer = Join-Path $DatasetDir "compose_finetune_replay.py"
$Validator = Join-Path $DatasetDir "validate_finetune_yolo_dataset.py"
$SplitManifest = Join-Path $DatasetDir "bps_work_split_v1.json"
$DenseSymbols = Join-Path $WorkRoot "datasets\piano50_dense_parentheses"
$DenseCurves = Join-Path $WorkRoot "datasets\curve_v2_fullbbox_2048"
if ($Mode -eq "Smoke") {
    $SymbolTargetName = "bps_piano50_smoke"
    $CurveTargetName = "bps_curve_v2_smoke"
    $SymbolMixedName = "bps_piano50_dense_replay_smoke"
    $CurveMixedName = "bps_curve_v2_dense_replay_smoke"
} else {
    $SymbolTargetName = "bps_piano50_worksplit_v1"
    $CurveTargetName = "bps_curve_v2_worksplit_v1"
    $SymbolMixedName = "bps_piano50_dense_replay_v1"
    $CurveMixedName = "bps_curve_v2_dense_replay_v1"
}
$SymbolTarget = Join-Path $WorkRoot "datasets\$SymbolTargetName"
$CurveTarget = Join-Path $WorkRoot "datasets\$CurveTargetName"
$SymbolMixed = Join-Path $WorkRoot "datasets\$SymbolMixedName"
$CurveMixed = Join-Path $WorkRoot "datasets\$CurveMixedName"

foreach ($Path in @(
    $PythonExe, $Converter, $Composer, $Validator, $SplitManifest,
    (Join-Path $SourceRoot "classes.txt"),
    (Join-Path $DenseSymbols "dataset.yaml"),
    (Join-Path $DenseCurves "dataset.yaml")
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required BPS preparation input: $Path"
    }
}
if ($ReplayRatio -lt 0) {
    throw "ReplayRatio must be non-negative"
}

function Invoke-CheckedPython([object[]]$Arguments) {
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Prepare-TargetDataset(
    [string]$Task,
    [string]$Output,
    [int]$ExpectedClasses
) {
    if ((Test-Path -LiteralPath $Output) -and -not $Overwrite) {
        Write-Host "Reusing existing target dataset: $Output"
    } else {
        $Arguments = @(
            $Converter,
            "--source-root", $SourceRoot,
            "--output-dir", $Output,
            "--task", $Task,
            "--split-manifest", $SplitManifest
        )
        if ($Mode -eq "Smoke") {
            $Arguments += @("--max-pages-per-split", "1")
        }
        if ($Overwrite) {
            $Arguments += "--overwrite"
        }
        Invoke-CheckedPython $Arguments
    }
    Invoke-CheckedPython @(
        $Validator, "--dataset-root", $Output,
        "--expected-classes", $ExpectedClasses
    )
}

function Prepare-MixedDataset(
    [string]$Target,
    [string]$Replay,
    [string]$Output,
    [int]$ExpectedClasses
) {
    if ((Test-Path -LiteralPath $Output) -and -not $Overwrite) {
        Write-Host "Reusing existing replay dataset: $Output"
    } else {
        $Arguments = @(
            $Composer,
            "--target-dataset", $Target,
            "--replay-dataset", $Replay,
            "--output-dir", $Output,
            "--replay-ratio", $ReplayRatio,
            "--link-mode", "auto"
        )
        if ($Overwrite) {
            $Arguments += "--overwrite"
        }
        Invoke-CheckedPython $Arguments
    }
    Invoke-CheckedPython @(
        $Validator, "--dataset-root", $Output,
        "--expected-classes", $ExpectedClasses
    )
}

Prepare-TargetDataset "symbols" $SymbolTarget 50
Prepare-TargetDataset "curves" $CurveTarget 1
Prepare-MixedDataset $SymbolTarget $DenseSymbols $SymbolMixed 50
Prepare-MixedDataset $CurveTarget $DenseCurves $CurveMixed 1

Write-Host "BPS FINE-TUNING DATA READY"
Write-Host "Symbols target only: $SymbolTarget"
Write-Host "Symbols + replay:    $SymbolMixed"
Write-Host "Curves target only:  $CurveTarget"
Write-Host "Curves + replay:     $CurveMixed"
Write-Host "No training was started."
