#Requires -Version 5.1
<#
.SYNOPSIS
    One-click installer for the MOSS-Image API server + AstrBot plugin (Windows).

.DESCRIPTION
    Prepares everything needed to run MOSS-Image locally in low-VRAM mode:

      1. checks the prerequisites (Windows, NVIDIA driver, git, Python)
      2. clones ai-toolkit at a pinned commit (it provides the inference engine)
      3. creates a virtualenv and installs CUDA torch + the dependencies
      4. downloads the MOSS-Image-SFT diffusers bundle from Hugging Face
      5. writes config.toml with the resolved paths
      6. copies the AstrBot plugin into AstrBot's data directory
      7. optionally registers the server for auto-start at logon

    Re-running is safe: existing steps are skipped, config.toml is backed up.

.PARAMETER AiToolkitDir
    Where to clone ai-toolkit. Default: <repo>\.ai-toolkit

.PARAMETER ModelDir
    Where to download the model bundle. Default: <repo>\.models\MOSS-Image-SFT

.PARAMETER ModelRepo
    Hugging Face repo of the diffusers bundle. Default: OpenMOSS-Team/MOSS-Image-SFT

.PARAMETER PluginDir
    AstrBot plugin directory. Default: %USERPROFILE%\.astrbot\data\plugins

.PARAMETER Python
    Python 3.10-3.13 interpreter to build the venv with. Default: auto-detect

.PARAMETER TorchIndex
    PyTorch wheel index. Default: https://download.pytorch.org/whl/cu128
    Use -Cpu to install CPU-only torch instead (very slow, for smoke testing).

.PARAMETER SkipModel
    Do not download the model bundle (you already have one, see -ModelDir).

.PARAMETER SkipPlugin
    Do not install the AstrBot plugin.

.PARAMETER SkipDeps
    Do not create the venv / install Python packages.

.PARAMETER Autostart
    Also register the server to start at logon (hidden window).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1

.EXAMPLE
    .\install.ps1 -ModelDir D:\models\MOSS-Image-SFT -SkipModel -Autostart
#>
[CmdletBinding()]
param(
    [string] $AiToolkitDir,
    [string] $ModelDir,
    [string] $ModelRepo = "OpenMOSS-Team/MOSS-Image-SFT",
    [string] $PluginDir,
    [string] $Python,
    [string] $TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch] $SkipModel,
    [switch] $SkipPlugin,
    [switch] $SkipDeps,
    [switch] $Autostart,
    [switch] $Cpu
)

$ErrorActionPreference = "Stop"

# ai-toolkit revision this release was tested against. It ships the MOSS-Image
# inference engine (int8 convrot8 quantisation, layer offloading, hook LoRA).
$AiToolkitRepo = "https://github.com/ostris/ai-toolkit.git"
$AiToolkitCommit = "04733873cf1da8060fce01316ec45daedbd4a918"

$Root = $PSScriptRoot
if (-not $AiToolkitDir) { $AiToolkitDir = Join-Path $Root ".ai-toolkit" }
if (-not $ModelDir) { $ModelDir = Join-Path $Root ".models\MOSS-Image-SFT" }
$AiToolkitDir = [System.IO.Path]::GetFullPath($AiToolkitDir)
$ModelDir = [System.IO.Path]::GetFullPath($ModelDir)

function Write-Step([int] $n, [int] $total, [string] $message) {
    Write-Host ""
    Write-Host "[$n/$total] $message" -ForegroundColor Cyan
}

function Write-Ok([string] $message) { Write-Host "      $message" -ForegroundColor Green }
function Write-Warn([string] $message) { Write-Host "      $message" -ForegroundColor Yellow }
function Fail([string] $message) {
    Write-Host ""
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 1. checks
Write-Step 1 7 "Checking prerequisites"

if (-not $IsWindows -and $PSVersionTable.PSVersion.Major -ge 6) {
    Fail "This script is for Windows. Use ./install.sh on Linux/macOS."
}

if (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Fail "nvidia-smi not found. Install the NVIDIA driver, or pass -Cpu for a CPU-only install (not practical for this model)."
    }
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join ""
    Write-Ok "GPU: $gpu"
    $freeGiB = [math]::Round((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) / 1024, 1)
    Write-Ok "VRAM: $freeGiB GiB (12 GiB is enough with the default low-VRAM settings)"
}

$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) { Fail "git not found. Install Git for Windows: https://git-scm.com/download/win" }
Write-Ok "git: $($git.Source)"

function Resolve-Python([string] $requested) {
    if ($requested) {
        if (-not (Test-Path $requested)) { Fail "Python not found: $requested" }
        return $requested
    }
    foreach ($candidate in @("py -3.12", "py -3.11", "py -3.10", "python")) {
        $parts = $candidate.Split(" ")
        $exe = Get-Command $parts[0] -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try {
            $version = & $parts[0] @($parts[1..($parts.Length - 1)] | Where-Object { $_ }) -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version) {
                $major, $minor = $version.Trim().Split(".")
                if ([int]$major -eq 3 -and [int]$minor -ge 10 -and [int]$minor -le 13) {
                    $path = & $parts[0] @($parts[1..($parts.Length - 1)] | Where-Object { $_ }) -c "import sys;print(sys.executable)" 2>$null
                    return $path.Trim()
                }
            }
        } catch { }
    }
    Fail "No Python 3.10-3.13 found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run."
}

if (-not $SkipDeps) {
    $Python = Resolve-Python $Python
    Write-Ok "python: $Python"
}

# ------------------------------------------------------------ 2. ai-toolkit
Write-Step 2 7 "Fetching ai-toolkit (inference engine)"
if (Test-Path (Join-Path $AiToolkitDir "run.py")) {
    Write-Ok "already present: $AiToolkitDir"
} else {
    & git clone $AiToolkitRepo $AiToolkitDir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed: $AiToolkitRepo" }
}
Push-Location $AiToolkitDir
try {
    & git fetch --depth 1 origin $AiToolkitCommit 2>$null
    if ($LASTEXITCODE -eq 0) {
        & git checkout --quiet FETCH_HEAD
        Write-Ok "checked out $($AiToolkitCommit.Substring(0,8))"
    } else {
        Write-Warn "could not fetch the pinned commit, keeping the current checkout"
    }
} finally {
    Pop-Location
}

# ----------------------------------------------------------------- 3. venv
Write-Step 3 7 "Installing Python dependencies"
$VenvDir = Join-Path $AiToolkitDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if ($SkipDeps) {
    Write-Warn "skipped (-SkipDeps)"
} else {
    if (-not (Test-Path $VenvPython)) {
        Write-Ok "creating venv: $VenvDir"
        & $Python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { Fail "could not create the virtualenv" }
    }
    & $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
    if ($LASTEXITCODE -ne 0) { Fail "pip bootstrap failed" }

    if ($Cpu) {
        Write-Ok "installing CPU torch (slow; only for testing the wiring)"
        & $VenvPython -m pip install torch torchvision --quiet
    } else {
        Write-Ok "installing CUDA torch from $TorchIndex (large download)"
        & $VenvPython -m pip install torch torchvision --index-url $TorchIndex
    }
    if ($LASTEXITCODE -ne 0) { Fail "torch install failed" }

    $toolkitReq = Join-Path $AiToolkitDir "requirements.txt"
    if (Test-Path $toolkitReq) {
        Write-Ok "installing ai-toolkit requirements"
        & $VenvPython -m pip install -r $toolkitReq --quiet
        if ($LASTEXITCODE -ne 0) { Write-Warn "ai-toolkit requirements reported errors; continuing" }
    }

    Write-Ok "installing server requirements"
    & $VenvPython -m pip install -r (Join-Path $Root "requirements.txt") --quiet
    if ($LASTEXITCODE -ne 0) { Fail "server requirements failed" }

    & $VenvPython -m pip install huggingface_hub --quiet
}

# ----------------------------------------------------------------- 4. model
Write-Step 4 7 "Preparing the model bundle"
if ($SkipModel) {
    if (-not (Test-Path (Join-Path $ModelDir "model_index.json"))) {
        Write-Warn "-SkipModel given but $ModelDir has no model_index.json"
    } else {
        Write-Ok "using existing bundle: $ModelDir"
    }
} else {
    if (Test-Path (Join-Path $ModelDir "model_index.json")) {
        Write-Ok "already downloaded: $ModelDir"
    } else {
        Write-Ok "downloading $ModelRepo (~30 GB, resumable)"
        $code = "from huggingface_hub import snapshot_download; snapshot_download('$ModelRepo', local_dir=r'$ModelDir')"
        & $VenvPython -c $code
        if ($LASTEXITCODE -ne 0) {
            Fail "model download failed. Retry with: hf download $ModelRepo --local-dir `"$ModelDir`""
        }
    }
}

# ---------------------------------------------------------------- 5. config
Write-Step 5 7 "Writing config.toml"
$configPath = Join-Path $Root "config.toml"
if (Test-Path $configPath) {
    $backup = "$configPath.bak-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    Copy-Item $configPath $backup
    Write-Warn "existing config backed up to $(Split-Path $backup -Leaf)"
}
$template = Get-Content (Join-Path $Root "config.example.toml") -Raw
$template = $template.Replace('aitk_root = "G:/ai-toolkit"', ('aitk_root = "' + ($AiToolkitDir -replace '\\','/') + '"'))
$template = $template.Replace('model_dir = "G:/model"', ('model_dir = "' + ($ModelDir -replace '\\','/') + '"'))
$template = $template.Replace('output_dir = "./outputs"', ('output_dir = "' + ((Join-Path $Root "outputs") -replace '\\','/') + '"'))
Set-Content -Path $configPath -Value $template -Encoding UTF8
Write-Ok "config: $configPath"

$manifest = @{
    ai_toolkit = $AiToolkitDir
    model_dir  = $ModelDir
    installed  = (Get-Date).ToString("s")
} | ConvertTo-Json
Set-Content -Path (Join-Path $Root ".install.json") -Value $manifest -Encoding UTF8

# ---------------------------------------------------------------- 6. plugin
Write-Step 6 7 "Installing the AstrBot plugin"
if ($SkipPlugin) {
    Write-Warn "skipped (-SkipPlugin)"
} else {
    if (-not $PluginDir) { $PluginDir = Join-Path $env:USERPROFILE ".astrbot\data\plugins" }
    $target = Join-Path $PluginDir "astrbot_plugin_moss_image"
    if (-not (Test-Path (Split-Path $PluginDir -Parent))) {
        Write-Warn "AstrBot data directory not found, skipping (install AstrBot first, then re-run)"
    } else {
        if (Test-Path $target) { Remove-Item $target -Recurse -Force }
        New-Item -ItemType Directory -Path $PluginDir -Force | Out-Null
        Copy-Item (Join-Path $Root "plugin\astrbot_plugin_moss_image") $target -Recurse
        Write-Ok "plugin: $target"
        Write-Warn "reload the plugin from the AstrBot WebUI (Plugins -> reload) to pick it up"
    }
}

# ------------------------------------------------------------- 7. autostart
Write-Step 7 7 "Auto-start"
if ($Autostart) {
    & (Join-Path $Root "install-autostart.ps1")
    Write-Ok "registered"
} else {
    Write-Warn "not registered (pass -Autostart to start the server at logon)"
}

# ------------------------------------------------------------------ summary
Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "  ai-toolkit : $AiToolkitDir"
Write-Host "  model      : $ModelDir"
Write-Host "  config     : $configPath"
Write-Host ""
Write-Host "Start the server (first request loads the model, ~20 s):"
Write-Host "  .\run.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Check it:"
Write-Host "  .\run.ps1 --help"
Write-Host "  python scripts\smoke_test.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "AstrBot: reload the plugin, then try  /draw a red fox in snow"
Write-Host "Docs: README.md, docs\configuration.md, docs\faq.md"
