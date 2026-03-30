param(
    [string]$RunId = "trial-20260321-cleandata1"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Archive = Join-Path $RootDir "weights\best_checkpoint.tar"
$EpochDir = Join-Path $RootDir ("checkpoints\" + $RunId + "\joint\epoch_038")
$BestDir = Join-Path $RootDir ("checkpoints\" + $RunId + "\joint\best")

if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Checkpoint archive not found: $Archive"
}

New-Item -ItemType Directory -Force -Path $EpochDir | Out-Null
New-Item -ItemType Directory -Force -Path $BestDir | Out-Null

tar -xf $Archive -C $EpochDir --strip-components=1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract checkpoint into $EpochDir"
}

tar -xf $Archive -C $BestDir --strip-components=1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract checkpoint into $BestDir"
}

Write-Host "Restored best checkpoint to:"
Write-Host "- $EpochDir"
Write-Host "- $BestDir"
