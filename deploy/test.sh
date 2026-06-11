#!/bin/bash
#
# test.sh — run the Athena test suite via SwiftPM.
#
# Why this script exists: `swift test` fails out of the box on a machine
# whose active developer dir is the Command Line Tools — CLT ships no
# XCTest module ("no such module 'XCTest'"). The suite needs the FULL
# Xcode toolchain. This mirrors deploy/build.sh's toolchain selection so
# the tests are reproducibly runnable without remembering DEVELOPER_DIR.
#
# Tiers:
#   - default        — the fast logic tier: every non-model test. The
#                      heavy inference/transcription tests self-skip
#                      unless ATHENA_RUN_MODEL_TESTS=1 (they need real
#                      model weights on disk + the metallib that only
#                      `xcodebuild` bundles).
#   - with a model   — ATHENA_RUN_MODEL_TESTS=1 ./deploy/test.sh
#                      runs the gated end-to-end invariants too. Those
#                      execute MLX kernels, so prefer running them under
#                      a binary built by deploy/build.sh on a Metal host.
#
# Usage:
#   ./deploy/test.sh                          # whole logic tier
#   ./deploy/test.sh --filter RBACTests       # one suite (args pass through)
#   ATHENA_RUN_MODEL_TESTS=1 ./deploy/test.sh # + gated model tests
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Full Xcode is required for XCTest. Honor an explicit DEVELOPER_DIR;
# else if the active dir is CLT, auto-select /Applications/Xcode.app.
if [ -z "${DEVELOPER_DIR:-}" ]; then
  active="$(xcode-select -p 2>/dev/null || true)"
  case "$active" in
    *Xcode.app*) : ;;  # already full Xcode
    *)
      if [ -d /Applications/Xcode.app/Contents/Developer ]; then
        export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
        echo "test.sh: using DEVELOPER_DIR=$DEVELOPER_DIR" \
             "(active xcode-select was '$active')"
      else
        echo "error: full Xcode required for XCTest but not found." \
             "Install Xcode, or run: sudo xcode-select -s" \
             "/Applications/Xcode.app, or export DEVELOPER_DIR." >&2
        exit 1
      fi
      ;;
  esac
fi

# The AthenaStructured module links the Rust structured-output staticlib;
# build it first if it's missing so a fresh checkout can test offline.
if [ ! -f rust-shim/target/release/libathena_structured_shim.a ]; then
  echo "test.sh: building rust-shim staticlib (one-time)…"
  ./rust-shim/build.sh
fi

echo "test.sh: swift test ${ATHENA_RUN_MODEL_TESTS:+(ATHENA_RUN_MODEL_TESTS=$ATHENA_RUN_MODEL_TESTS) }$*"
exec swift test "$@"
