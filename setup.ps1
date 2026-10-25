<#
.SYNOPSIS
    Create a dedicated .venv with CUDA torch and the server dependencies.

.DESCRIPTION
    Optional. run.ps1 works out of the box with the existing ai-toolkit venv on
    this machine; use this script when you want an isolated environment or are
    setting the project up on another machine.  Requires uv on PATH
    (winget install astral-sh.uv).
#>
[CmdletBinding()]
param(
    [string] $Python = "3.12",
    [string] $TorchIndex = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw "uv is required: winget install astral-sh.uv" }

Write-Host "==> Creating .venv (Python $Python)" -ForegroundColor Cyan
& $uv venv --python $Python .venv
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

$venvPython = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "==> Installing torch from $TorchIndex" -ForegroundColor Cyan
& $uv pip install --python $venvPython --index-url $TorchIndex torch torchvision
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "==> Installing server dependencies" -ForegroundColor Cyan
& $uv pip install --python $venvPython -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

Write-Host "==> Verifying" -ForegroundColor Cyan
& $venvPython -c "import torch, diffusers, transformers, fastapi; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__)"
Write-Host "Done. Start the server with .\run.ps1" -ForegroundColor Green
