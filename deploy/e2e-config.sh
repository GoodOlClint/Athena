#!/bin/bash
#
# e2e-config.sh — end-to-end DoD for ADR 037 (daemon-mediated config + sudoless
# restart). Runs a loopback dev daemon (no auth, stub engine — no model needed)
# and proves, all as the CURRENT (non-root) user:
#
#   1) GET /api/config             → 200, projects the TOML + readonly_keys.
#   2) `athena config set` (API)   → writes the daemon's own TOML with NO sudo
#                                     (the file is the one the daemon reads).
#   3) deny-listed key             → 400 config_key_readonly (never written).
#   4) unknown key                 → 400.
#   5) `athena restart` (API)      → 200; the daemon drains + exit(0)s with NO
#                                     sudo (launchd KeepAlive relaunch is the
#                                     installed-host tier — not exercised here).
#
# The static-plist + full-TOML-at-boot enabler (slice 1) is unit-pinned in
# AthenaDeployTests; this is the control-plane + CLI-repoint DoD.
#
# Usage: ./deploy/e2e-config.sh [binary]
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
PORT=7791
DATA="$(mktemp -d)"
CFG="$(mktemp -d)/athena.toml"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$DATA" "$(dirname "$CFG")"' EXIT

[ -x "$BIN" ] || { echo "error: no binary at $BIN (build it first)"; exit 1; }

# A minimal TOML the daemon reads via ATHENA_CONFIG (full-TOML-at-boot, slice 1).
cat >"$CFG" <<TOML
listen_host = "127.0.0.1"
listen_port = $PORT
engine = "stub"
max_tokens = 1024
TOML

pass=0; fail=0
ok() { echo "  ok   $1"; pass=$((pass+1)); }
no() { echo "  FAIL $1"; fail=$((fail+1)); }
URL="http://127.0.0.1:$PORT"
OPTS=(--host 127.0.0.1 --port "$PORT")

echo "== starting loopback dev daemon on :$PORT (stub engine, no auth) =="
ATHENA_CONFIG="$CFG" "$BIN" load "${OPTS[@]}" --engine stub \
  --data-dir "$DATA" >"$DATA/daemon.log" 2>&1 &
DPID=$!
for _ in $(seq 1 60); do
  curl -sf "$URL/healthz" >/dev/null 2>&1 && break; sleep 1
done
curl -sf "$URL/healthz" >/dev/null 2>&1 || { echo "daemon never came up"; cat "$DATA/daemon.log"; exit 1; }

# 1) GET /api/config
GET="$(curl -s "$URL/api/config")"
echo "$GET" | grep -q '"readonly_keys"' && echo "$GET" | grep -q "tls_key" \
  && ok "GET /api/config projects config + readonly_keys" \
  || no "GET /api/config missing readonly_keys/tls_key"

# 2) athena config set via the API — NO sudo. Writes the daemon's own TOML.
if "$BIN" config set "${OPTS[@]}" max_tokens 4096 >/dev/null 2>&1; then
  ok "non-root \`athena config set max_tokens 4096\` exit 0 (via daemon API)"
else
  no "config set exited non-zero"
fi
grep -qE "^max_tokens = 4096" "$CFG" \
  && ok "daemon rewrote its own TOML (max_tokens = 4096) with no sudo" \
  || no "TOML not updated ($(grep max_tokens "$CFG"))"

# 3) deny-listed key → 400 config_key_readonly
CODE="$(curl -s -o "$DATA/deny.json" -w '%{http_code}' -X PUT "$URL/api/config" \
  -H 'Content-Type: application/json' -d '{"key":"tls_key","value":"/x/y.pem"}')"
if [ "$CODE" = "400" ] && grep -q "config_key_readonly" "$DATA/deny.json"; then
  ok "deny-listed key tls_key → 400 config_key_readonly"
else
  no "deny-listed key gave $CODE ($(cat "$DATA/deny.json"))"
fi
grep -q "tls_key" "$CFG" && no "denied key was written to TOML!" \
  || ok "denied key never written to TOML"

# 4) unknown key → 400
CODE="$(curl -s -o /dev/null -w '%{http_code}' -X PUT "$URL/api/config" \
  -H 'Content-Type: application/json' -d '{"key":"not_a_real_key","value":"1"}')"
[ "$CODE" = "400" ] && ok "unknown key → 400" || no "unknown key gave $CODE"

# 5) athena restart via the API — NO sudo (daemon drains + exit(0)s).
"$BIN" restart "${OPTS[@]}" >/dev/null 2>&1 \
  && ok "non-root \`athena restart\` exit 0 (via daemon API)" \
  || no "restart exited non-zero"
# The foreground daemon has no launchd to relaunch it, so it should now be gone.
for _ in $(seq 1 10); do curl -sf "$URL/healthz" >/dev/null 2>&1 || break; sleep 1; done
curl -sf "$URL/healthz" >/dev/null 2>&1 \
  && no "daemon still up after restart (exit(0) did not fire)" \
  || ok "daemon exited on restart (launchd relaunch is the installed-host tier)"

echo "════════════════════════════════════════"
echo "  PASS=$pass  FAIL=$fail"
echo "════════════════════════════════════════"
[ "$fail" -eq 0 ]
