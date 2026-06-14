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

& $uv run ruff check .
& $uv run ty check src tests
& $uv run pytest
