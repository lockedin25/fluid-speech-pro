#!/bin/zsh
set -euo pipefail

echo "=========================================="
echo "  Fluid Dictation — Local Setup (macOS)"
echo "=========================================="

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Error: Fluid Dictation requires an Apple Silicon (arm64) Mac." >&2
  exit 1
fi

SCRIPT_DIR="${0:A:h}"
if [[ -d "$SCRIPT_DIR/payload" && -f "$SCRIPT_DIR/payload/fluid_speech_mlx.py" ]]; then
  SRC_DIR="$SCRIPT_DIR/payload"
else
  SRC_DIR="$SCRIPT_DIR"
fi

INSTALL_DIR="$HOME/Library/Application Support/FluidDictation"
APP_DEST="$HOME/Applications/Fluid Dictation.app"
PLIST_PATH="$HOME/Library/LaunchAgents/local.shubham.fluid-dictation.plist"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

echo "[1/7] Stopping any running Fluid Dictation instance..."
pkill -x FluidDictation 2>/dev/null || true
sleep 0.5

echo "[2/7] Installing runtime files to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/hotkey" "$INSTALL_DIR/models/fluid-speech-pro-mlx" "$HOME/Applications" "$HOME/Library/LaunchAgents"

cp "$SRC_DIR/fluid_speech_mlx.py" "$INSTALL_DIR/"
cp "$SRC_DIR/transcribe.py" "$INSTALL_DIR/"
cp "$SRC_DIR/transcribe" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/transcribe"
cp "$SRC_DIR/requirements.lock.txt" "$INSTALL_DIR/"
cp "$SRC_DIR/SHA256SUMS" "$INSTALL_DIR/"
[[ -f "$SRC_DIR/MODEL_CARD.md" ]] && cp "$SRC_DIR/MODEL_CARD.md" "$INSTALL_DIR/"
[[ -f "$SRC_DIR/README.md" ]] && cp "$SRC_DIR/README.md" "$INSTALL_DIR/"

cp "$SRC_DIR/hotkey/worker.py" "$INSTALL_DIR/hotkey/"
cp "$SRC_DIR/hotkey/inference.py" "$INSTALL_DIR/hotkey/"
cp "$SRC_DIR/hotkey/vocabulary_boost.py" "$INSTALL_DIR/hotkey/"
cp "$SRC_DIR/hotkey/FluidDictation.swift" "$INSTALL_DIR/hotkey/"
cp "$SRC_DIR/hotkey/build.sh" "$INSTALL_DIR/hotkey/"
chmod +x "$INSTALL_DIR/hotkey/build.sh"

if [[ ! -f "$INSTALL_DIR/hotkey/config.json" ]]; then
  cat > "$INSTALL_DIR/hotkey/config.json" <<'JSON'
{
  "idle_unload_seconds": 10,
  "vocabulary_boost": {
    "enabled": false,
    "beam_size": 2,
    "terms_file": "vocabulary/active-boost.json"
  }
}
JSON
fi

echo "[3/7] Checking uv and ffmpeg..."
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
  else
    echo "Warning: ffmpeg not found and Homebrew is not installed."
    echo "Install Homebrew (https://brew.sh) and run: brew install ffmpeg" >&2
  fi
fi

echo "[4/7] Setting up Python 3.12 environment..."
if [[ ! -x "$INSTALL_DIR/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$INSTALL_DIR/.venv"
fi
uv pip install --python "$INSTALL_DIR/.venv/bin/python" -r "$INSTALL_DIR/requirements.lock.txt"

echo "[5/7] Checking local 8-bit MLX model weights (~750 MB)..."
if ! (cd "$INSTALL_DIR" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1); then
  if [[ -f "$SRC_DIR/models/fluid-speech-pro-mlx/model.8bit.safetensors" ]]; then
    echo "Copying bundled model weights..."
    cp -R "$SRC_DIR/models/fluid-speech-pro-mlx/." "$INSTALL_DIR/models/fluid-speech-pro-mlx/"
  else
    echo "Downloading Fluid Speech Pro (8-bit MLX) from Hugging Face..."
    "$INSTALL_DIR/.venv/bin/hf" download FluidVoice-Labs/fluid-speech-pro-mlx \
      config.json model.8bit.safetensors clean-mode.safetensors LICENSE NOTICE \
      --revision aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0 \
      --local-dir "$INSTALL_DIR/models/fluid-speech-pro-mlx"
  fi
  (cd "$INSTALL_DIR" && shasum -a 256 -c SHA256SUMS)
else
  echo "Model weights verified."
fi

echo "[6/7] Installing Fluid Dictation.app..."
if [[ -d "$SCRIPT_DIR/Fluid Dictation.app" ]]; then
  rm -rf "$APP_DEST"
  ditto "$SCRIPT_DIR/Fluid Dictation.app" "$APP_DEST"
  xattr -cr "$APP_DEST" 2>/dev/null || true
  /usr/bin/codesign --force --sign - --identifier local.shubham.fluid-dictation "$APP_DEST"
else
  "$INSTALL_DIR/hotkey/build.sh" "$APP_DEST"
fi

echo "[7/7] Configuring Fn key & launch-at-login..."
defaults write com.apple.HIToolbox AppleFnUsageType -int 0
if [[ -x "/System/Library/PrivateFrameworks/SystemAdministration.framework/Resources/activateSettings" ]]; then
  /System/Library/PrivateFrameworks/SystemAdministration.framework/Resources/activateSettings -u || true
fi

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>local.shubham.fluid-dictation</string>
	<key>ProgramArguments</key>
	<array>
		<string>/usr/bin/open</string>
		<string>-g</string>
		<string>-a</string>
		<string>$APP_DEST</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/local.shubham.fluid-dictation" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || true

open -a "$APP_DEST" --args --setup

echo ""
echo "Setup complete! Fluid Dictation is now running in your menu bar."
echo "Next:"
echo "  1. Allow Microphone and Accessibility permissions when prompted."
echo "  2. Tap Fn (toggle) or hold Fn (push-to-talk) anywhere to dictate."
echo "     Backup shortcut: Control + Option + Space"
