# Shared interpreter discovery for run.ps1 / start-server.ps1 / install-autostart.ps1.
#
# Order:
#   1. the repo's own .venv
#   2. <ai-toolkit>\.venv, where <ai-toolkit> comes from (-AiToolkitDir),
#      $env:AI_TOOLKIT_DIR, or the .install.json written by install.ps1
#   3. whatever `python` is on PATH

function Get-MossAiToolkitDir {
    param([string] $Root, [string] $Override)

    if ($Override) { return $Override }
    if ($env:AI_TOOLKIT_DIR) { return $env:AI_TOOLKIT_DIR }
    $manifest = Join-Path $Root ".install.json"
    if (Test-Path $manifest) {
        try {
            $data = Get-Content $manifest -Raw | ConvertFrom-Json
            if ($data.ai_toolkit) { return [string]$data.ai_toolkit }
        } catch {
            Write-Verbose "Could not read $manifest : $_"
        }
    }
    return $null
}

function Get-MossPython {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [string] $AiToolkitDir
    )

    $candidates = @((Join-Path $Root ".venv\Scripts\python.exe"))
    $toolkit = Get-MossAiToolkitDir -Root $Root -Override $AiToolkitDir
    if ($toolkit) { $candidates += (Join-Path $toolkit ".venv\Scripts\python.exe") }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "No Python interpreter found. Run .\install.ps1 (or .\setup.ps1) first."
}
