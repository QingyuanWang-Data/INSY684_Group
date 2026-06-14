$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    $localUv = Join-Path $HOME ".local\bin\uv.exe"
    if (Test-Path $localUv) {
        $uv = $localUv
    }
    else {
        throw "uv was not found on PATH or at $localUv"
    }
}
else {
    $uv = $uvCommand.Source
}

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
$pytestTempRoot = Join-Path $repoRoot ".pytest_tmp"
$pytestBaseTemp = Join-Path $pytestTempRoot "run-$PID"
New-Item -ItemType Directory -Force -Path $pytestTempRoot | Out-Null

function Invoke-QualityStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $uv @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: uv $($Arguments -join ' ')"
    }
}

Invoke-QualityStep @("run", "--no-sync", "ruff", "check", ".")
Invoke-QualityStep @("run", "--no-sync", "ty", "check", "src", "tests")
Invoke-QualityStep @("run", "--no-sync", "pytest", "--basetemp", $pytestBaseTemp)
