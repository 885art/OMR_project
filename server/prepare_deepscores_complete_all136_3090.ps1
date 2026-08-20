[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Smoke",
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [string]$CompleteRoot = "C:\OMR_work\data\ds2_complete",
    [string]$WorkRoot = "C:\OMR_work\experiments",
    [switch]$OverwriteChunks
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DatasetTools = Join-Path $RepoRoot "articulation_experiments\dataset"
$MappingGenerator = Join-Path $DatasetTools "generate_deepscores_all_mapping.py"
$Converter = Join-Path $DatasetTools "convert_deepscores_complete_sharded.py"
$Mapping = Join-Path $WorkRoot "mappings\class_mapping_deepscores_all136.json"
$FirstShard = Get-ChildItem -LiteralPath $CompleteRoot -Filter "deepscores-complete-*_train.json" -File |
    Sort-Object Name | Select-Object -First 1
if ($null -eq $FirstShard) { throw "No Complete train shard found in $CompleteRoot" }
$DatasetName = if ($Mode -eq "Smoke") {
    "deepscores_complete_all136_sharded_smoke10"
} else {
    "deepscores_complete_all136_sharded"
}
$DatasetRoot = Join-Path $WorkRoot "datasets\$DatasetName"

& $PythonExe $MappingGenerator --source-json $FirstShard.FullName --output $Mapping --overwrite
if ($LASTEXITCODE -ne 0) { throw "All-class mapping generation failed" }

$Arguments = @(
    $Converter,
    "--complete-root", $CompleteRoot,
    "--output-dir", $DatasetRoot,
    "--class-mapping", $Mapping,
    "--tile-size", "1024",
    "--overlap", "256",
    "--minimum-intersection-ratio", "0.6",
    "--negative-ratio", "0.05",
    "--png-compress-level", "1",
    "--resume"
)
if ($Mode -eq "Smoke") {
    $Arguments += @("--max-shards-per-split", "1", "--max-images-per-shard", "10")
}
if ($OverwriteChunks) {
    $Arguments += "--overwrite-chunks"
    Write-Warning "Explicit chunk overwrite enabled."
}
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Complete all136 sharded conversion failed with exit code $LASTEXITCODE"
}
Write-Host "COMPLETE ALL136 DATASET READY: $DatasetRoot"
Write-Host "No training was started."
