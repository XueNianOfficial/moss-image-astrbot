<#
.SYNOPSIS
    Hidden launcher used by the autostart entry.

.DESCRIPTION
    Starts ``python -m moss_server --config config.toml`` as a detached,
    windowless process with stdout/stderr redirected to timestamped log files,
    then exits.  Nothing already listening on the health endpoint is started
    twice.

    Using Start-Process with native redirection (instead of PowerShell's *>)
    keeps Windows PowerShell 5.1 from turning Python's stderr log lines into
    terminating errors, and lets the launcher exit so the server survives.
#>
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
. (Join-Path $root "python-runtime.ps1")

$configPath = Join-Path $root "config.toml"
$port = 8000
if (Test-Path $configPath) {
    $match = Select-String -Path $configPath -Pattern '^\s*port\s*=\s*(\d+)' | Select-Object -First 1
    if ($match) { $port = [int]$match.Matches[0].Groups[1].Value }
}

# Another instance already serving: nothing to do.
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing
    if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { exit 0 }
} catch {
    # Nothing listening, continue.
}

$python = Get-MossPython -Root $root
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $logDir "server-$stamp.out.log"
$stderr = Join-Path $logDir "server-$stamp.err.log"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$arguments = @("-m", "moss_server", "--config", $configPath)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -PassThru

Write-Output ("started pid {0} -> {1}" -f $process.Id, $stdout)
