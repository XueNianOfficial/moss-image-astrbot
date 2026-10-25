#!/usr/bin/env bash
# One-click installer for the MOSS-Image API server + AstrBot plugin (Linux/macOS).
#
#   ./install.sh                 # install into <repo>/.ai-toolkit and <repo>/.models
#   AI_TOOLKIT_DIR=~/ai-toolkit MODEL_DIR=/data/MOSS-Image-SFT ./install.sh
#   SKIP_MODEL=1 ./install.sh    # you already downloaded the bundle
#
# Environment variables:
#   AI_TOOLKIT_DIR   where to clone ai-toolkit      (default: <repo>/.ai-toolkit)
#   MODEL_DIR        where to put the model bundle  (default: <repo>/.models/MOSS-Image-SFT)
#   MODEL_REPO       Hugging Face repo id           (default: OpenMOSS-Team/MOSS-Image-SFT)
#   PLUGIN_DIR       AstrBot plugin directory       (default: ~/.astrbot/data/plugins)
#   PYTHON           interpreter for the venv       (default: python3)
#   TORCH_INDEX      torch wheel index              (default: https://download.pytorch.org/whl/cu128)
#   SKIP_MODEL=1     skip the ~30 GB download
#   SKIP_DEPS=1      skip venv + pip
#   SKIP_PLUGIN=1    skip the AstrBot plugin
#
# Tested on Windows (install.ps1) and Linux with an NVIDIA GPU. CPU-only runs
# are possible with TORCH_INDEX=https://download.pytorch.org/whl/cpu but are
# impractically slow for this model.
set -euo pipefail

AI_TOOLKIT_REPO="https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_COMMIT="04733873cf1da8060fce01316ec45daedbd4a918"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITK="${AI_TOOLKIT_DIR:-$ROOT/.ai-toolkit}"
MODEL="${MODEL_DIR:-$ROOT/.models/MOSS-Image-SFT}"
MODEL_REPO="${MODEL_REPO:-OpenMOSS-Team/MOSS-Image-SFT}"
PLUGIN_DIR="${PLUGIN_DIR:-$HOME/.astrbot/data/plugins}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

step() { printf '\n[%s] %s\n' "$1" "$2"; }
ok()   { printf '      %s\n' "$1"; }
warn() { printf '      %s\n' "$1"; }

step 1/7 "Checking prerequisites"
command -v git >/dev/null || { echo "ERROR: git is required"; exit 1; }
ok "git: $(git --version)"
if [ "${SKIP_DEPS:-0}" != "1" ]; then
    command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not found (pass PYTHON=...)"; exit 1; }
    ok "python: $($PYTHON -V 2>&1)"
fi
if command -v nvidia-smi >/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
else
    warn "nvidia-smi not found - only CPU torch will work (very slow)"
fi

step 2/7 "Fetching ai-toolkit (inference engine)"
if [ -f "$AITK/run.py" ]; then
    ok "already present: $AITK"
else
    git clone "$AI_TOOLKIT_REPO" "$AITK"
fi
if git -C "$AITK" fetch --depth 1 origin "$AI_TOOLKIT_COMMIT" 2>/dev/null; then
    git -C "$AITK" checkout --quiet FETCH_HEAD
    ok "checked out ${AI_TOOLKIT_COMMIT:0:8}"
else
    warn "could not fetch the pinned commit, keeping the current checkout"
fi

step 3/7 "Installing Python dependencies"
if [ "${SKIP_DEPS:-0}" = "1" ]; then
    warn "skipped (SKIP_DEPS=1)"
else
    [ -d "$AITK/.venv" ] || "$PYTHON" -m venv "$AITK/.venv"
    # shellcheck disable=SC1091
    . "$AITK/.venv/bin/activate"
    python -m pip install --upgrade pip setuptools wheel --quiet
    pip install torch torchvision --index-url "$TORCH_INDEX"
    [ -f "$AITK/requirements.txt" ] && pip install -r "$AITK/requirements.txt" --quiet || true
    pip install -r "$ROOT/requirements.txt" --quiet
    pip install huggingface_hub --quiet
    ok "venv: $AITK/.venv"
fi

step 4/7 "Preparing the model bundle"
if [ "${SKIP_MODEL:-0}" = "1" ]; then
    warn "skipped (SKIP_MODEL=1): $MODEL"
elif [ -f "$MODEL/model_index.json" ]; then
    ok "already downloaded: $MODEL"
else
    ok "downloading $MODEL_REPO (~30 GB, resumable)"
    "$AITK/.venv/bin/python" - "$MODEL_REPO" "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi

step 5/7 "Writing config.toml"
CONFIG="$ROOT/config.toml"
[ -f "$CONFIG" ] && cp "$CONFIG" "$CONFIG.bak-$(date +%Y%m%d-%H%M%S)" && warn "previous config backed up"
sed -e "s|aitk_root = \"G:/ai-toolkit\"|aitk_root = \"$AITK\"|" \
    -e "s|model_dir = \"G:/model\"|model_dir = \"$MODEL\"|" \
    -e "s|output_dir = \"./outputs\"|output_dir = \"$ROOT/outputs\"|" \
    "$ROOT/config.example.toml" > "$CONFIG"
ok "config: $CONFIG"
printf '{\n  "ai_toolkit": "%s",\n  "model_dir": "%s"\n}\n' "$AITK" "$MODEL" > "$ROOT/.install.json"

step 6/7 "Installing the AstrBot plugin"
if [ "${SKIP_PLUGIN:-0}" = "1" ]; then
    warn "skipped (SKIP_PLUGIN=1)"
elif [ ! -d "$(dirname "$PLUGIN_DIR")" ]; then
    warn "AstrBot data directory not found, skipping (install AstrBot, then re-run)"
else
    rm -rf "$PLUGIN_DIR/astrbot_plugin_moss_image"
    mkdir -p "$PLUGIN_DIR"
    cp -r "$ROOT/plugin/astrbot_plugin_moss_image" "$PLUGIN_DIR/"
    ok "plugin: $PLUGIN_DIR/astrbot_plugin_moss_image"
    warn "reload the plugin from the AstrBot WebUI to pick it up"
fi

step 7/7 "Auto-start"
warn "not configured; see docs/configuration.md (systemd unit example)"

cat <<EOF

Done.

  ai-toolkit : $AITK
  model      : $MODEL
  config     : $CONFIG

Start the server (first request loads the model, ~20 s):
  $AITK/.venv/bin/python -m moss_server --config $CONFIG

Check it:
  $AITK/.venv/bin/python scripts/smoke_test.py

AstrBot: reload the plugin, then try  /draw a red fox in snow
EOF
