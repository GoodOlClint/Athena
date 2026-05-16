#!/bin/bash
#
# build.sh — build the Athena structured-output Rust staticlib.
#
# host == target == arm64 macOS, so a plain release build produces the
# linkable .a; no cross-compile / xcframework. SwiftPM links it via the
# CAthenaStructured system-library module map (rust-shim/include header +
# -L this target/release). Run before `swift build` / deploy/build.sh.
#
set -euo pipefail
cd "$(dirname "$0")"

command -v cargo >/dev/null || { echo "error: cargo not on PATH" >&2; exit 1; }

cargo build --release
LIB="target/release/libathena_structured_shim.a"
[[ -f "$LIB" ]] || { echo "error: $LIB not produced" >&2; exit 1; }

echo "built: $(cd "$(dirname "$LIB")" && pwd)/$(basename "$LIB")"
echo "header: $(pwd)/include/athena_structured.h"
