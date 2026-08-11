<#
.SYNOPSIS
  One-command local demo (Windows): creates the venv, installs deps, then
  starts the API and UI via scripts/demo.py.

.EXAMPLE
  .\demo.ps1
  .\demo.ps1 -Setup      # download corpus + build index, then start
  .\demo.ps1 -Check      # preflight only
  .\demo.ps1 -NoUi       # API only
#>
param(
    [switch]$Check,
    [switch]$Setup,
    [switch]$Yes,
    [switch]$NoUi,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py    = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$stamp = Join-Path $PSScriptRoot "venv\.requirements.sha"
$reqs  = Join-Path $PSScriptRoot "requirements.txt"

# --- venv ------------------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "Creating venv..." -ForegroundColor Cyan
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv venv --python 3.11
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.11 -m venv venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv venv
    } else {
        Write-Host "No Python found. Install Python 3.11+ or uv (https://astral.sh/uv)." -ForegroundColor Red
        exit 1
    }
}

# --- dependencies (only when requirements.txt changed) ---------------------
$want = (Get-FileHash $reqs -Algorithm SHA256).Hash
$have = ""
if (Test-Path $stamp) { $have = (Get-Content $stamp -Raw).Trim() }

if ($want -ne $have) {
    Write-Host "Installing dependencies (first run, or requirements.txt changed)..." -ForegroundColor Cyan
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv pip install --python $py -r $reqs
    } else {
        & $py -m pip install --upgrade pip
        & $py -m pip install -r $reqs
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "Dependency install failed." -ForegroundColor Red; exit 1 }
    Set-Content -Path $stamp -Value $want -Encoding utf8
}

# --- run -------------------------------------------------------------------
$demoArgs = @()
if ($Check)     { $demoArgs += "--check" }
if ($Setup)     { $demoArgs += "--setup" }
if ($Yes)       { $demoArgs += "--yes" }
if ($NoUi)      { $demoArgs += "--no-ui" }
if ($NoBrowser) { $demoArgs += "--no-browser" }

& $py (Join-Path $PSScriptRoot "scripts\demo.py") @demoArgs
exit $LASTEXITCODE
