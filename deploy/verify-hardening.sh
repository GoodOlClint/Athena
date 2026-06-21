#!/bin/bash
#
# verify-hardening.sh — assert the Athena binary carries the ADR-024 Tier-1
# process-lockdown posture: Hardened Runtime ON and `get-task-allow` NOT granted.
#
# This is the deterministic, MLX-free regression gate for T1. It does NOT load a
# model — the "metallib still loads under the Hardened Runtime" regression
# (M43.3) is exercised by the real-inference e2e path (deploy/e2e-rbac.sh) and
# was validated by the Phase-0 Spike B. Here we pin the signing posture so a
# future build that silently drops `--options runtime` or re-injects
# get-task-allow (Xcode dev signing does) fails loudly.
#
# Usage:  ./deploy/verify-hardening.sh [path/to/athena]
# Exit:   0 = hardened + no get-task-allow; non-zero = posture regressed.
#
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
if [ ! -x "$BIN" ]; then
  echo "verify-hardening: binary not found at '$BIN'" >&2
  exit 2
fi

fail=0

# 1) Hardened Runtime flag (0x10000) must be set. `codesign -dv` prints e.g.
#    `flags=0x10002(adhoc,runtime)` — require the `runtime` token.
flags="$(codesign -dv "$BIN" 2>&1 | grep -iE '^CodeDirectory' | grep -oiE 'flags=0x[0-9a-f]+\([^)]*\)' || true)"
if echo "$flags" | grep -qi 'runtime'; then
  echo "✓ Hardened Runtime ON         ($flags)"
else
  echo "✗ Hardened Runtime MISSING    (${flags:-<no flags>}) — build.sh must sign with --options runtime" >&2
  fail=1
fi

# 2) get-task-allow must NOT be granted. Accept three shapes: entitlement
#    absent entirely, present-but-false, or present-but-empty (codesign renders
#    <false/> as an empty value). A `true` value is the regression.
ents="$(codesign -d --entitlements - --xml "$BIN" 2>/dev/null || codesign -d --entitlements - "$BIN" 2>/dev/null || true)"
if echo "$ents" | grep -qi 'get-task-allow'; then
  # The key is present — make sure it is not true.
  # `plutil` parse of the embedded entitlements if it is well-formed XML.
  if echo "$ents" | grep -A1 -i 'get-task-allow' | grep -qi '<true'; then
    echo "✗ get-task-allow = TRUE       — the task port is open to scrapers (ADR 024 T1 regression)" >&2
    fail=1
  else
    echo "✓ get-task-allow not granted  (present, false/empty)"
  fi
else
  echo "✓ get-task-allow not granted  (entitlement absent)"
fi

# 3) Advisory: report notarization / signer. Adhoc is fine for local dev; a
#    release build SHOULD be Developer ID + notarized (operator-side, gated on a
#    signing identity). Warn-only so local adhoc builds still pass the gate.
signer="$(codesign -dv "$BIN" 2>&1 | grep -iE 'Authority|Signature' | head -3 || true)"
if echo "$signer" | grep -qi 'Developer ID'; then
  echo "✓ signed with Developer ID    (release posture)"
else
  echo "ℹ adhoc/dev signature         — release builds should be Developer ID + notarized (see deploy/build.sh)"
fi

if [ "$fail" -ne 0 ]; then
  echo "verify-hardening: FAILED — process-lockdown posture regressed." >&2
  exit 1
fi
echo "verify-hardening: OK"
