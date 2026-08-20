[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("Symbols", "Curves")]
    [string]$Task,
    [Parameter(Mandatory)]
    [string]$Weights,
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [string]$YoloV9Root = "C:\OMR_work\yolov9",
    [string]$RunsDir = "C:\OMR_work\experiments\runs",
    [switch]$AllowFinalTest
)

$ErrorActionPreference = "Stop"
if (-not $AllowFinalTest) {
    throw "Final BPS test is locked. Re-run with -AllowFinalTest only after the training recipe is selected from validation results."
}
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $RepoRoot "articulation_experiments\train\yolov9_compat_launcher.py"
if ($Task -eq "Symbols") {
    $Data = "C:\OMR_work\experiments\datasets\bps_piano50_dense_replay_v1\dataset.yaml"
    $ImageSize = 1024
    $BatchSize = 8
} else {
    $Data = "C:\OMR_work\experiments\datasets\bps_curve_v2_dense_replay_v1\dataset.yaml"
    $ImageSize = 1280
    $BatchSize = 4
}
foreach ($Path in @($PythonExe, $Launcher, $Data, $Weights)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing test-evaluation input: $Path"
    }
}
$RunName = "$(Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent $Weights)))_bps_test"
& $PythonExe $Launcher `
    --yolov9-root $YoloV9Root `
    --script val_dual.py `
    --data $Data `
    --weights $Weights `
    --task test `
    --imgsz $ImageSize `
    --batch-size $BatchSize `
    --device 0 `
    --workers 4 `
    --project $RunsDir `
    --name $RunName
if ($LASTEXITCODE -ne 0) {
    throw "BPS final test evaluation failed with exit code $LASTEXITCODE"
}
