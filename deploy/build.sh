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

# xcodebuild needs FULL Xcode, not the Command Line Tools. `command -v
# xcodebuild` succeeds even under CLT, but the build then fails with
# "tool 'xcodebuild' requires Xcode". Honor an explicit DEVELOPER_DIR;
# else if the active dir is CLT, auto-select /Applications/Xcode.app.
if [ -z "${DEVELOPER_DIR:-}" ]; then
  active="$(xcode-select -p 2>/dev/null || true)"
  case "$active" in
    *Xcode.app*) : ;;  # already full Xcode
    *)
      if [ -d /Applications/Xcode.app/Contents/Developer ]; then
        export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
        echo "build.sh: using DEVELOPER_DIR=$DEVELOPER_DIR" \
             "(active xcode-select was '$active')"
      else
        echo "error: full Xcode required but not found. Install" \
             "Xcode, or run: sudo xcode-select -s" \
             "/Applications/Xcode.app, or export DEVELOPER_DIR." >&2
        exit 1
      fi
      ;;
  esac
fi

# The `athena-Package` aggregate scheme builds the `athena` binary
# plus every `*.bundle` resource Products/<config>/ ends up holding.
# `athena install` copies all of them (InstallPlan.artifactNames).
# M43.3 removed the M14.2d `athenad` launcher (its bare argv[0] execv
# broke MLX's metallib-bundle lookup under hardened-runtime spawn —
# `athena start` and the LaunchDaemon now point at `athena` directly).
# `CLANG_ENABLE_CODE_COVERAGE=NO` suppresses C-side coverage
# instrumentation. The Swift-side `-profile-generate
# -profile-coverage-mapping` flags come from the SwiftPM-generated
# scheme's `codeCoverageEnabled=YES` and can only be overridden by
# committing an explicit xcscheme — deferred (M43 fragility #5).
xcodebuild -scheme athena-Package -configuration "$CONFIG" \
  -destination 'platform=macOS' -derivedDataPath .build/xcode \
  CLANG_ENABLE_CODE_COVERAGE=NO \
  -skipMacroValidation -skipPackagePluginValidation build

echo
echo "built: .build/xcode/Build/Products/${CONFIG}/athena"
echo "install: sudo .build/xcode/Build/Products/${CONFIG}/athena install \\"
echo "           --config deploy/athena.toml"
