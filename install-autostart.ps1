<#
.SYNOPSIS
    Install (or remove) autostart for the MOSS-Image server.

.DESCRIPTION
    Preferred: a per-user scheduled task that starts the server at logon.
    If task creation is denied (it needs elevation on some machines), the
    script falls back to a hidden launcher in the user's Startup folder.

    The server itself is lazy: it holds no VRAM until the first request and
    unloads again after idle_unload_seconds of inactivity.

.PARAMETER Remove
    Remove both the scheduled task and the Startup entry.

.PARAMETER ForceStartupFolder
    Skip the scheduled task and use the Startup folder directly.

.EXAMPLE
    .\install-autostart.ps1

.EXAMPLE
    .\install-autostart.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch] $Remove,
    [switch] $ForceStartupFolder,
    [string] $TaskName = "MOSS-Image Server"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $root "start-server.ps1"
$startupDir = [Environment]::GetFolderPath("Startup")
$startupFile = Join-Path $startupDir "MOSS-Image Server.vbs"
$powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

function Invoke-Schtasks {
    # Native stderr becomes an error record when $ErrorActionPreference is Stop,
    # so run these quietly and report the exit code instead.
    param([string[]] $Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & schtasks @Arguments 2>&1 | Out-Null
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Get-TaskPresent {
    param([string] $Name)
    return (Invoke-Schtasks @("/Query", "/TN", $Name)) -eq 0
}

function Remove-Autostart {
    if (Get-TaskPresent $TaskName) {
        Invoke-Schtasks @("/Delete", "/TN", $TaskName, "/F") | Out-Null
        Write-Host "Removed scheduled task: $TaskName" -ForegroundColor Yellow
    }
    if (Test-Path $startupFile) {
        Remove-Item $startupFile -Force
        Write-Host "Removed startup entry: $startupFile" -ForegroundColor Yellow
    }
}

if ($Remove) {
    Remove-Autostart
    return
}

if (-not (Test-Path $launcher)) {
    throw "Missing launcher: $launcher"
}

$method = $null
if (-not $ForceStartupFolder) {
    $action = '"{0}" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{1}"' -f $powershell, $launcher
    if ((Invoke-Schtasks @("/Create", "/TN", $TaskName, "/TR", $action, "/SC", "ONLOGON", "/RL", "LIMITED", "/F")) -eq 0) {
        $method = "scheduled task"
    } else {
        Write-Host "Scheduled task creation was denied; falling back to the Startup folder." -ForegroundColor DarkYellow
    }
}

if (-not $method) {
    # Startup folder: a tiny VBScript runs the launcher with no console window.
    # VBS escapes a literal quote inside a string by doubling it.
    $command = '"' + $powershell + '" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $launcher + '"'
    $escaped = $command -replace '"', '""'
    $vbs = 'Set shell = CreateObject("WScript.Shell")' + "`r`n" +
           'shell.Run "' + $escaped + '", 0, False' + "`r`n"
    New-Item -ItemType Directory -Force -Path $startupDir | Out-Null
    Set-Content -Path $startupFile -Value $vbs -Encoding ASCII
    $method = "Startup folder"
}

Write-Host "Autostart installed via $method" -ForegroundColor Green
if ($method -eq "scheduled task") {
    Write-Host "  task    : $TaskName (at user logon)"
} else {
    Write-Host "  shortcut: $startupFile"
}
Write-Host "  command : $powershell -File `"$launcher`""
Write-Host "  config  : $(Join-Path $root 'config.toml')"
Write-Host "  logs    : $(Join-Path $root 'logs')"
Write-Host ""
Write-Host "Notes:" -ForegroundColor DarkGray
Write-Host "  - starts at the next user logon; start it now with:" -ForegroundColor DarkGray
Write-Host "      $powershell -File `"$launcher`"" -ForegroundColor DarkGray
Write-Host "  - remove with: .\install-autostart.ps1 -Remove" -ForegroundColor DarkGray
