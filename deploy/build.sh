#!/bin/bash
#
# build.sh — produce the Athena release binary.
#
# This is the one deploy step the binary cannot self-host: MLX's Metal
# shaders are NOT built by `swift build` (documented mlx-swift limitation) —
# a swift-built binary aborts at first inference with "Failed to load the
# default metallib". The build MUST go through xcodebuild. Requires a full
# Xcode + the Metal toolchain component (one-time:
# `xcodebuild -downloadComponent MetalToolchain`).
#
# Then install with the binary itself:
#   sudo .build/xcode/Build/Products/<config>/athena install --config deploy/athena.toml
#
# Usage:  ./deploy/build.sh [Release|Debug]   (default: Release)
#
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-Release}"
command -v xcodebuild >/dev/null || {
  echo "error: xcodebuild not found — a full Xcode install is required " \
       "(Command Line Tools cannot build MLX Metal shaders)" >&2
  exit 1
}

xcodebuild -scheme athena -configuration "$CONFIG" \
  -destination 'platform=macOS' -derivedDataPath .build/xcode \
  -skipMacroValidation -skipPackagePluginValidation build

echo
echo "built: .build/xcode/Build/Products/${CONFIG}/athena"
echo "install: sudo .build/xcode/Build/Products/${CONFIG}/athena install \\"
echo "           --config deploy/athena.toml"
