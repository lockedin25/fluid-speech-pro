#!/bin/zsh
set -eu
ROOT="${0:A:h:h}"
APP="${1:-$HOME/Applications/Fluid Dictation.app}"
mkdir -p "$APP/Contents/MacOS"
/usr/bin/swiftc -O "$ROOT/hotkey/FluidDictation.swift" -o "$APP/Contents/MacOS/FluidDictation" -framework AppKit -framework AVFoundation -framework Carbon -framework ApplicationServices
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>local.shubham.fluid-dictation</string>
<key>CFBundleName</key><string>Fluid Dictation</string>
<key>CFBundleExecutable</key><string>FluidDictation</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>LSUIElement</key><true/>
<key>NSMicrophoneUsageDescription</key><string>Record speech when you press your dictation shortcut, then transcribe locally on your Mac.</string>
</dict></plist>
PLIST
/usr/bin/codesign --force --sign - --identifier local.shubham.fluid-dictation "$APP"
