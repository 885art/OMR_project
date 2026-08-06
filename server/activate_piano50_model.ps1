[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Weights,
    [string]$DataYaml = "C:\OMR_work\experiments\datasets\piano50_dense_parentheses\dataset.yaml",
    [string]$Config = "C:\OMR_work\25-omr\jsonTemplate.json"
)

$ErrorActionPreference = "Stop"
foreach ($Path in @($Weights, $DataYaml, $Config)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing: $Path" }
}
$Document = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$Document.articulation.backend = "yolov9"
$Document.articulation.weights = (Resolve-Path -LiteralPath $Weights).Path.Replace("\", "/")
$Document.articulation.data_yaml = (Resolve-Path -LiteralPath $DataYaml).Path.Replace("\", "/")
$Document.articulation.yolov9_root = "C:/OMR_work/yolov9"
if ($Document.articulation.PSObject.Properties.Name -contains "mapping") {
    $Document.articulation.mapping = "C:/OMR_work/25-omr/articulation_experiments/dataset/class_mapping_piano.json"
} else {
    $Document.articulation | Add-Member -NotePropertyName mapping -NotePropertyValue "C:/OMR_work/25-omr/articulation_experiments/dataset/class_mapping_piano.json"
}
$Document | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $Config -Encoding utf8
Write-Host "ACTIVE PIANO-50 MODEL: $Weights"
