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
#   ./deploy/test.sh --print-toolchain        # identify the compiler, run nothing
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
        # stderr: with --print-toolchain, stdout is a machine-read value.
        echo "test.sh: using DEVELOPER_DIR=$DEVELOPER_DIR" \
             "(active xcode-select was '$active')" >&2
      else
        echo "error: full Xcode required for XCTest but not found." \
             "Install Xcode, or run: sudo xcode-select -s" \
             "/Applications/Xcode.app, or export DEVELOPER_DIR." >&2
        exit 1
      fi
      ;;
  esac
fi

# --print-toolchain: emit the identity of the compiler THIS SCRIPT would
# build with, then exit. Exists so CI can key its SPM cache on the compiler
# that actually performs the build instead of on the Xcode it selected (#59).
#
# It lives here rather than in the workflow so the reported compiler is the
# one THIS script would build with, whatever the resolution above decides.
# In CI that is currently belt-and-braces — the workflow asserts
# DEVELOPER_DIR is set, so the `case` never runs and an inline
# `swift --version` would agree — but it keeps the two tied together for any
# caller that has not made that guarantee.
#
# Placed BEFORE the rust-shim build so identifying the toolchain never
# triggers a cargo build as a side effect.
if [ "${1:-}" = "--print-toolchain" ]; then
  # `exec`, and NOT `2>&1`: a broken DEVELOPER_DIR makes xcrun write to stderr
  # and exit non-zero. Folding that onto stdout would let a caller hash the
  # ERROR TEXT into a plausible-looking cache key while the step still passed.
  exec swift --version
fi

# The AthenaStructured module links the Rust structured-output staticlib;
# build it first if it's missing so a fresh checkout can test offline.
if [ ! -f rust-shim/target/release/libathena_structured_shim.a ]; then
  echo "test.sh: building rust-shim staticlib (one-time)…"
  ./rust-shim/build.sh
fi

echo "test.sh: swift test ${ATHENA_RUN_MODEL_TESTS:+(ATHENA_RUN_MODEL_TESTS=$ATHENA_RUN_MODEL_TESTS) }$*"
exec swift test "$@"
