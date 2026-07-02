#!/bin/bash
#
# e2e-embedding-select.sh — real-model end-to-end proof of per-request
# embedding model selection (M39). The stub e2e (deploy/e2e-rbac.sh)
# structurally CANNOT cover this: only a real model gives the
# dimension difference (bge-small 384 vs bge-large 1024) that proves
# the `model` field actually selected the served model rather than
# being echoed back unused.
#
# It brings up a loopback, auth-disabled daemon with TWO embedding
# models declared and asserts, on the sole embedding surface:
#   • /v1/embeddings  (OpenAI)      — M39.1
# (native /api/embed + queued embeddings removed — ADR 031/013 + ADR 025)
# that the requested model selects the right dimension AND the response
# reports the model ACTUALLY served (truthful), and that an unknown
# model is a 400 `model_not_available` — never a silent wrong-dim
# fallback, never an on-request download.
#
# Usage:
#   deploy/integration/e2e-embedding-select.sh            # all paths
#   ATHENA_BIN=/path/to/athena deploy/integration/e2e-embedding-select.sh
#
# Tunables (env):
#   ATHENA_BIN   athena binary (default: auto-resolve xcodebuild output)
#   PORT         listen port (default: 7491)
#   SMALL        small model HF id (default: BAAI/bge-small-en-v1.5, 384)
#   LARGE        large model HF id (default: BAAI/bge-large-en-v1.5, 1024)
#   LLM_MODEL    an LLM in the store so `load` starts (embeddings never
#                touch it; default: first store entry)
#   PHASE        1 = /v1 only (M39.1 gate); 2 = + native + queued
#                (M39.2 gate). Default: 2.
#
# NOTE: first run downloads the two embedding models (~130 MB + ~1.3 GB)
# into the HF cache — they are in the DECLARED set, so loading them is
# sanctioned (the "no on-request download" rule is about UNDECLARED
# models, which this script proves stay a 400).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PORT="${PORT:-7491}"
SMALL="${SMALL:-BAAI/bge-small-en-v1.5}"
LARGE="${LARGE:-BAAI/bge-large-en-v1.5}"
PHASE="${PHASE:-2}"
DATA_DIR="$(mktemp -d)"
base="http://127.0.0.1:$PORT"

pass=0; fail=0
ok()  { echo "  ok  — $*"; pass=$((pass + 1)); }
bad() { echo "  BAD — $*"; fail=$((fail + 1)); }

# Resolve the binary: explicit env > PATH > xcodebuild output.
if [ -z "${ATHENA_BIN:-}" ]; then
  if command -v athena >/dev/null 2>&1; then
    ATHENA_BIN="$(command -v athena)"
  else
    for c in \
      .build/xcode/Build/Products/Release/athena \
      .build/xcode/Build/Products/Debug/athena; do
      [ -x "$c" ] && ATHENA_BIN="$c" && break
    done
  fi
fi
[ -x "${ATHENA_BIN:-/nonexistent}" ] || {
  echo "no athena binary — set ATHENA_BIN or run deploy/build.sh" >&2
  exit 2
}
echo "using: $ATHENA_BIN"

# An LLM only needs to exist so `load` starts; embeddings never load it.
if [ -z "${LLM_MODEL:-}" ]; then
  LLM_MODEL="$(/bin/ls -1 "$HOME/.athena/models" 2>/dev/null | head -1)"
fi
[ -n "${LLM_MODEL:-}" ] || {
  echo "no LLM in ~/.athena/models — set LLM_MODEL" >&2; exit 2; }

echo "════════════════════════════════════════════════════════════"
echo " M39 embedding-selection e2e (real mlx engine, loopback)"
echo "   port      : $PORT"
echo "   small/384 : $SMALL"
echo "   large/1024: $LARGE"
echo "   llm (idle): $LLM_MODEL"
echo "   phase     : $PHASE"
echo "════════════════════════════════════════════════════════════"

cleanup() {
  [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null || true
  wait "${SRV_PID:-}" 2>/dev/null || true
  rm -rf "$DATA_DIR"
}
trap cleanup EXIT

# Loopback + no creds ⇒ auth disabled ⇒ curl needs no token.
"$ATHENA_BIN" load --engine mlx --host 127.0.0.1 --port "$PORT" \
  --model "$LLM_MODEL" --data-dir "$DATA_DIR" \
  --embedding-model "$SMALL" --embedding-model "$LARGE" \
  >"$DATA_DIR/daemon.log" 2>&1 &
SRV_PID=$!

printf 'waiting for /healthz '
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "$base/healthz" 2>/dev/null; then
    echo " up"; break; fi
  printf '.'; sleep 1
done
curl -fsS -o /dev/null "$base/healthz" 2>/dev/null || {
  echo; echo "daemon did not become healthy:"; tail -30 "$DATA_DIR/daemon.log"
  exit 1; }

# dim <json> — length of data[0].embedding
v1dim()  { python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["data"][0]["embedding"]))'; }
v1model(){ python3 -c 'import sys,json;print(json.load(sys.stdin)["model"])'; }

echo "== /v1/embeddings — model selection (M39.1) =="

# bge-large ⇒ 1024 + truthful model
R="$(curl -fsS -H 'content-type: application/json' \
  -d "{\"model\":\"$LARGE\",\"input\":\"hello\"}" "$base/v1/embeddings")"
[ "$(printf '%s' "$R" | v1dim)" = 1024 ] \
  && ok "large ⇒ 1024-dim" || bad "large dim != 1024 ($R)"
[ "$(printf '%s' "$R" | v1model)" = "$LARGE" ] \
  && ok "large ⇒ truthful model=$LARGE" || bad "model not truthful ($R)"

# bge-small ⇒ 384 + truthful model
R="$(curl -fsS -H 'content-type: application/json' \
  -d "{\"model\":\"$SMALL\",\"input\":\"hello\"}" "$base/v1/embeddings")"
[ "$(printf '%s' "$R" | v1dim)" = 384 ] \
  && ok "small ⇒ 384-dim" || bad "small dim != 384 ($R)"
[ "$(printf '%s' "$R" | v1model)" = "$SMALL" ] \
  && ok "small ⇒ truthful model=$SMALL" || bad "model not truthful ($R)"

# absent model ⇒ default (first declared = small) + truthful
R="$(curl -fsS -H 'content-type: application/json' \
  -d '{"input":"hello"}' "$base/v1/embeddings")"
[ "$(printf '%s' "$R" | v1dim)" = 384 ] \
  && ok "absent ⇒ default 384-dim" || bad "absent dim != 384 ($R)"
[ "$(printf '%s' "$R" | v1model)" = "$SMALL" ] \
  && ok "absent ⇒ truthful default model=$SMALL" || bad "default not truthful ($R)"

# unknown model ⇒ 400 model_not_available (NOT a silent 384)
CODE="$(curl -s -o "$DATA_DIR/nope.json" -w '%{http_code}' \
  -H 'content-type: application/json' \
  -d '{"model":"nope/not-loaded","input":"hello"}' "$base/v1/embeddings")"
[ "$CODE" = 400 ] \
  && ok "unknown ⇒ HTTP 400" || bad "unknown not 400 (got $CODE)"
grep -q 'model_not_available' "$DATA_DIR/nope.json" \
  && ok "unknown ⇒ code model_not_available" \
  || bad "missing model_not_available ($(cat "$DATA_DIR/nope.json"))"

# /api/embed (native) + /v1/queue/embeddings blocks removed — both routes
# are gone (ADR 031/013 native-embed removal; ADR 025 queue removal). /v1
# is the single inference surface, exercised above.

echo "════════════════════════════════════════════════════════════"
echo " M39 e2e: $pass passed, $fail failed"
echo "════════════════════════════════════════════════════════════"
[ "$fail" = 0 ]
