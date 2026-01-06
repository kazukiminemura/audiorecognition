$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root "venv\\Scripts\\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "venv not found at $venvPython"
    Write-Host "Create it first: python -m venv .\\venv"
    exit 1
}

$requirements = Join-Path $root "requirements.txt"
$spec = Join-Path $PSScriptRoot "audiorecognition.spec"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $requirements
& $venvPython -m pip install pyinstaller

$env:AUDIOREC_ROOT = $root
$env:PYTHONWARNINGS = "ignore:.*torch\\.distributed\\._sharding_spec.*:DeprecationWarning,ignore:.*torch\\.distributed\\._sharded_tensor.*:DeprecationWarning,ignore:.*torch\\.distributed\\._shard\\.checkpoint.*:DeprecationWarning"
& $venvPython -m PyInstaller $spec
