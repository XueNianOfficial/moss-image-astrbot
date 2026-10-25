<#
.SYNOPSIS
    Start the MOSS-Image API server in the foreground.

.DESCRIPTION
    Picks a Python interpreter (project .venv, then the ai-toolkit venv, then
    python from PATH) and forwards every argument to ``python -m moss_server``.
    The model is served by ai-toolkit's inference engine (int8 convrot8, layer
    offloading, hook-mode LoRA).
    For an unattended, hidden start use start-server.ps1 instead.

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 --config config.toml --idle-unload 1200

.EXAMPLE
    .\run.ps1 --lora D:\loras\my_lora.safetensors --lora-scale 0.8
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ServerArgs
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
. (Join-Path $root "python-runtime.ps1")

$python = Get-MossPython -Root $root

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
Write-Host "Python : $python" -ForegroundColor DarkGray
Write-Host "Project: $root" -ForegroundColor DarkGray

# Keep going on stderr output: Python logs to stderr constantly and with
# $ErrorActionPreference = "Stop" Windows PowerShell would turn the first log
# line into a terminating NativeCommandError.
$ErrorActionPreference = "Continue"
& $python -m moss_server @ServerArgs
exit $LASTEXITCODE
