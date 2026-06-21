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
# `CLANG_ENABLE_CODE_COVERAGE=NO` suppresses C-side instrumentation;
# the Swift-side `-profile-generate -profile-coverage-mapping` flags
# are killed by the committed
# `.swiftpm/xcode/xcshareddata/xcschemes/athena-Package.xcscheme`
# (codeCoverageEnabled="NO") which overrides the auto-generated one.
# `ONLY_ACTIVE_ARCH=YES` keeps the build arm64-only — the appliance
# is Apple-Silicon-only (README "Requirements") and the auto-scheme
# previously used the same default; the committed scheme doesn't
# pin architectures so we set it here.
xcodebuild -scheme athena-Package -configuration "$CONFIG" \
  -destination 'platform=macOS' -derivedDataPath .build/xcode \
  CLANG_ENABLE_CODE_COVERAGE=NO ONLY_ACTIVE_ARCH=YES \
  -skipMacroValidation -skipPackagePluginValidation build

BIN=".build/xcode/Build/Products/${CONFIG}/athena"

# ── ADR 024 Tier 1 — process lockdown (Hardened Runtime, no get-task-allow) ──
#
# A dev/adhoc binary effectively carries `get-task-allow`, so any process that
# can call task_for_pid (root trivially) can attach and scrape the address
# space — KV cache, resident weights, decrypted secrets. We re-sign WITH the
# Hardened Runtime and WITHOUT get-task-allow (deploy/athena.entitlements) so
# AMFI denies the task port to a co-resident scraper. Phase-0 Spike B confirmed
# MLX's metallib still loads under the Hardened Runtime (the M43.3 risk), and
# that this signature flips a debugger attach from "reaches the task" to "Not
# allowed to attach."
#
# Signing identity:
#   * CODESIGN_IDENTITY set        → use it (a "Developer ID Application: …"
#                                     identity enables notarization below).
#   * unset                        → adhoc ("-"). Adhoc + Hardened Runtime is
#                                     a fully valid local-dev posture: it still
#                                     drops get-task-allow and locks the task
#                                     port. It is NOT notarizable / Gatekeeper-
#                                     distributable — that needs Developer ID.
IDENTITY="${CODESIGN_IDENTITY:--}"
echo
echo "build.sh: applying Hardened Runtime signature (identity: ${IDENTITY})"
codesign --force --timestamp --options runtime \
  --entitlements deploy/athena.entitlements \
  --sign "$IDENTITY" "$BIN"

# Fail-closed regression gate: the binary must be hardened + no get-task-allow.
./deploy/verify-hardening.sh "$BIN"

# ── Notarization (release only; gated on a Developer ID identity) ────────────
# Notarization is a Gatekeeper/distribution concern, orthogonal to the task-port
# lockdown above. It requires a Developer ID identity AND notarytool credentials
# (a stored keychain profile or Apple-ID app-specific password). Run it only
# when explicitly requested AND credentialed; otherwise skip with a note.
#   NOTARIZE=1  CODESIGN_IDENTITY="Developer ID Application: …"  \
#   NOTARYTOOL_PROFILE="athena-notary"  ./deploy/build.sh Release
if [ "${NOTARIZE:-0}" = "1" ]; then
  if [ "$IDENTITY" = "-" ]; then
    echo "build.sh: NOTARIZE=1 but no CODESIGN_IDENTITY (adhoc) — cannot notarize." >&2
    exit 1
  fi
  prof="${NOTARYTOOL_PROFILE:-}"
  [ -n "$prof" ] || { echo "build.sh: NOTARIZE=1 needs NOTARYTOOL_PROFILE (xcrun notarytool store-credentials)." >&2; exit 1; }
  zip=".build/xcode/athena-notarize.zip"
  /usr/bin/ditto -c -k --keepParent "$BIN" "$zip"
  echo "build.sh: submitting $zip to notarytool (profile: $prof)…"
  xcrun notarytool submit "$zip" --keychain-profile "$prof" --wait
  # CLI tools cannot be `stapler staple`d (no bundle); the notarization ticket
  # is published to Apple and validated online by Gatekeeper on first run.
  echo "build.sh: notarization complete (CLI binary — ticket is served online, no stapling)."
else
  echo "build.sh: notarization skipped (set NOTARIZE=1 + CODESIGN_IDENTITY + NOTARYTOOL_PROFILE for a release build)."
fi

echo
echo "built: ${BIN}"
echo "install: sudo ${BIN} install \\"
echo "           --config deploy/athena.toml"
