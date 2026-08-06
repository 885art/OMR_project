[CmdletBinding()]
param(
    [ValidateSet("Dense", "Complete")]
    [string]$SourceFormat = "Dense",
    [ValidateSet("Smoke40", "Full")]
    [string]$Mode = "Smoke40",
    [string]$PythonExe = "C:\Users\minemine\miniconda3\envs\omr\python.exe",
    [switch]$Overwrite,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if ($Overwrite -and $Resume) {
    throw "-Overwrite and -Resume are mutually exclusive"
}

$RepoRoot = "C:\OMR_work\25-omr"
$WorkRoot = "C:\OMR_work\experiments"
$Mapping = Join-Path $RepoRoot "articulation_experiments\dataset\class_mapping_extended.json"
$Merger = Join-Path $RepoRoot "articulation_experiments\dataset\merge_deepscores_complete.py"
$Converter = Join-Path $RepoRoot "articulation_experiments\dataset\convert_deepscores_to_yolo.py"
$Validator = Join-Path $RepoRoot "articulation_experiments\dataset\validate_yolo_dataset.py"

if ($SourceFormat -eq "Dense") {
    $DeepScoresRoot = "C:\OMR_work\data\ds2_dense"
    $SourceJsonRoot = $DeepScoresRoot
    if ($Mode -eq "Smoke40") {
        $DatasetRoot = Join-Path $WorkRoot "datasets\symbols_tiny_v2_dense_smoke40"
    } else {
        $DatasetRoot = Join-Path $WorkRoot "datasets\symbols_tiny_v2_full"
    }
} else {
    $DeepScoresRoot = "C:\OMR_work\data\ds2_complete"
    if ($Mode -eq "Smoke40") {
        $DatasetRoot = Join-Path $WorkRoot "datasets\symbols_tiny_v2_complete_smoke40"
        $SourceJsonRoot = Join-Path $WorkRoot "datasets\deepscores_complete_smoke40_merged"
    } else {
        $DatasetRoot = Join-Path $WorkRoot "datasets\symbols_tiny_v2_complete_full"
        $SourceJsonRoot = Join-Path $WorkRoot "datasets\deepscores_complete_full_merged"
    }
}

$RequiredFiles = @($PythonExe, $Mapping, $Converter, $Validator)
if ($SourceFormat -eq "Complete") {
    $RequiredFiles += $Merger
}
foreach ($Path in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing required file: $Path"
    }
}
if (-not (Test-Path -LiteralPath $DeepScoresRoot -PathType Container)) {
    throw "DeepScores $SourceFormat is not ready: $DeepScoresRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $DeepScoresRoot "images") -PathType Container)) {
    throw "DeepScores complete images directory is missing"
}

New-Item -ItemType Directory -Path (Join-Path $WorkRoot "datasets") -Force | Out-Null

if ($SourceFormat -eq "Complete") {
    $MergedFiles = @(
        (Join-Path $SourceJsonRoot "deepscores_train.json"),
        (Join-Path $SourceJsonRoot "deepscores_test.json"),
        (Join-Path $SourceJsonRoot "merge_statistics.json")
    )
    if ($Resume) {
        $MissingMergedFiles = @($MergedFiles | Where-Object {
            -not (Test-Path -LiteralPath $_ -PathType Leaf)
        })
        if ($MissingMergedFiles.Count -gt 0) {
            throw "Cannot resume because merged Complete files are missing: $($MissingMergedFiles -join ', ')"
        }
        Write-Host "Reusing merged Complete source: $SourceJsonRoot"
    } else {
        $MergeArgs = @(
            $Merger,
            "--complete-root", $DeepScoresRoot,
            "--output-dir", $SourceJsonRoot,
            "--class-mapping", $Mapping,
            "--cross-split-policy", "train",
            "--all-available-shards"
        )
        if ($Mode -eq "Smoke40") {
            # Process every available shard, but only 10 pages from each shard.
            $MergeArgs += @("--max-images-per-shard", "10")
        }
        if ($Overwrite) {
            $MergeArgs += "--overwrite"
        }

        Write-Host "Merging DeepScores complete ($Mode) into $SourceJsonRoot"
        & $PythonExe @MergeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "DeepScores merge failed with exit code $LASTEXITCODE"
        }
    }
}

$ConvertArgs = @(
    $Converter,
    "--dataset-root", $SourceJsonRoot,
    "--images-dir", (Join-Path $DeepScoresRoot "images"),
    "--output-dir", $DatasetRoot,
    "--class-mapping", $Mapping,
    "--tile-size", "512",
    "--overlap", "128",
    "--edge-policy", "shift",
    "--minimum-intersection-ratio", "0.6",
    "--minimum-tenuto-bbox-height-pixels", "8",
    "--negative-ratio", "0.25",
    "--seed", "20260730",
    "--png-compress-level", "1",
    "--progress-every", "100"
)
if ($SourceFormat -eq "Dense" -and $Mode -eq "Smoke40") {
    $ConvertArgs += @("--max-images-per-split", "10")
}
if ($Overwrite) {
    $ConvertArgs += "--overwrite"
}
if ($Resume) {
    $ConvertArgs += "--resume"
}

Write-Host "Converting tiny-object v2 dataset into $DatasetRoot"
& $PythonExe @ConvertArgs
if ($LASTEXITCODE -ne 0) {
    throw "Dataset conversion failed with exit code $LASTEXITCODE"
}

& $PythonExe $Validator `
    --dataset-root $DatasetRoot `
    --source-dataset-root $SourceJsonRoot `
    --class-mapping $Mapping
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed with exit code $LASTEXITCODE"
}

Write-Host "DATASET READY: $DatasetRoot"
Write-Host "SOURCE JSON ROOT: $SourceJsonRoot"
