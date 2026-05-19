#!/bin/bash
#
# studio-setup.sh — prepare the Mac Studio as the real-model Athena
# daemon host for the MANUAL integration tier (see RUNBOOK.md). This
# is the "automate the boring part" half: it makes the environment
# reproducible and idempotent; a human then runs the action steps in
# RUNBOOK.md from the MacBook.
#
# It runs the REAL engine (`--engine mlx`) against a pinned small
# model on the external SSD — the thing the stub e2e (deploy/
# e2e-rbac.sh) structurally cannot cover.
#
# Usage:
#   deploy/integration/studio-setup.sh            # set up + start
#   deploy/integration/studio-setup.sh --teardown # stop the daemon
#   deploy/integration/studio-setup.sh --teardown --purge   # + creds
#
# Tunables (env, with defaults):
#   ATHENA_BIN    path to the athena binary (default: auto-resolve)
#   BIND_HOST     listen host (default: 0.0.0.0 — LAN reachable)
#   PORT          listen port (default: 7447 — Athena's own port)
#   MODEL_STORE   model store root (default: ~/.athena/models;
#                 SET THIS to the external SSD path on the Studio)
#   DATA_DIR      runtime/data dir (default: ~/.athena)
#   TEST_MODEL    pinned real model (HF id or store name) — REQUIRED
#   ADMIN_USER    seeded admin account (default: admin)
#   ADMIN_PASS    admin password (default: prompted, no echo)
#
# Security: fail-safe refuses a non-loopback bind with no creds, so
# the admin account is seeded BEFORE the daemon starts. The minted
# bearer token is printed ONCE to stdout and never written to disk
# (matches Athena's no-secret-at-rest posture) — copy it to the
# MacBook (macbook-env.sh) by hand.
#
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-7447}"
DATA_DIR="${DATA_DIR:-$HOME/.athena}"
MODEL_STORE="${MODEL_STORE:-$HOME/.athena/models}"
ADMIN_USER="${ADMIN_USER:-admin}"
TEARDOWN=0
PURGE=0
for a in "$@"; do
  case "$a" in
    --teardown) TEARDOWN=1 ;;
    --purge)    PURGE=1 ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

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
  echo "no athena binary — set ATHENA_BIN or build first" >&2
  exit 2
}
echo "using: $ATHENA_BIN"

base="http://127.0.0.1:$PORT"

healthz() { curl -fsS -o /dev/null "$base/healthz" 2>/dev/null; }

stop_daemon() {
  echo "== stopping daemon =="
  "$ATHENA_BIN" stop --data-dir "$DATA_DIR" 2>/dev/null || true
}

if [ "$TEARDOWN" = 1 ]; then
  stop_daemon
  if [ "$PURGE" = 1 ]; then
    # Destructive: only on explicit --purge. Cascades the test
    # admin's roles + tokens (auth user rm).
    read -r -p "PURGE: delete user '$ADMIN_USER' + its tokens? [y/N] " c
    if [ "$c" = y ] || [ "$c" = Y ]; then
      "$ATHENA_BIN" auth user rm "$ADMIN_USER" \
        --data-dir "$DATA_DIR" 2>/dev/null || true
      echo "purged $ADMIN_USER"
    else
      echo "purge skipped"
    fi
  fi
  echo "teardown done."
  exit 0
fi

if [ -z "${TEST_MODEL:-}" ]; then
  cat >&2 <<'EOF'
ERROR: TEST_MODEL is required — pin the small real model the manual
tier exercises, e.g.:
  TEST_MODEL=mlx-community/<small-model> \
  MODEL_STORE=/Volumes/<SSD>/athena/models \
  deploy/integration/studio-setup.sh
Skipping silently would let the suite "pass" without ever loading a
real model — refused on purpose.
EOF
  exit 2
fi

echo "════════════════════════════════════════════════════════════"
echo " Athena integration host setup (Mac Studio)"
echo "   bind        : $BIND_HOST:$PORT"
echo "   data dir    : $DATA_DIR"
echo "   model store : $MODEL_STORE"
echo "   test model  : $TEST_MODEL  (engine: mlx — REAL)"
echo "════════════════════════════════════════════════════════════"
case "$MODEL_STORE" in
  /Volumes/*) : ;;
  *) echo "WARN: MODEL_STORE is not on /Volumes/* — point it at the" \
          "external SSD on the Studio." ;;
esac

# 1. Persist config (in-place TOML edit; keeps comments/layout).
echo "== writing config =="
"$ATHENA_BIN" config set listen_host "$BIND_HOST"
"$ATHENA_BIN" config set listen_port "$PORT"
"$ATHENA_BIN" config set model_store "$MODEL_STORE"
"$ATHENA_BIN" config set data_dir    "$DATA_DIR"
"$ATHENA_BIN" config set engine      mlx
"$ATHENA_BIN" config set model       "$TEST_MODEL"
"$ATHENA_BIN" config path

# 2. Seed the admin account BEFORE start (fail-safe needs creds for a
#    non-loopback bind). `auth user add` create/replaces idempotently.
echo "== seeding admin account =="
if [ -z "${ADMIN_PASS:-}" ]; then
  read -r -s -p "admin password (>= 8 chars): " ADMIN_PASS; echo
fi
"$ATHENA_BIN" auth user add "$ADMIN_USER" \
  --password "$ADMIN_PASS" --role admin --data-dir "$DATA_DIR"

# 3. Mint a bearer token — printed ONCE, never persisted here.
echo "== minting admin bearer token (shown once) =="
TOK="$("$ATHENA_BIN" auth token add --user "$ADMIN_USER" \
  --data-dir "$DATA_DIR" 2>/dev/null \
  | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
[ -n "$TOK" ] || { echo "FAILED to mint token" >&2; exit 1; }

# 4. Start the daemon (background; non-launchd quick path — the
#    launchd install/start path is exercised by RUNBOOK scenario D).
echo "== starting daemon =="
"$ATHENA_BIN" start --host "$BIND_HOST" --port "$PORT" \
  --engine mlx --model "$TEST_MODEL" --data-dir "$DATA_DIR"

printf 'waiting for /healthz '
for _ in $(seq 1 60); do
  if healthz; then echo " up"; break; fi
  printf '.'; sleep 1
done
healthz || { echo; echo "daemon did not become healthy" >&2
  "$ATHENA_BIN" logs --source start --data-dir "$DATA_DIR" -n 40 \
    2>/dev/null || true; exit 1; }

# 5. Ensure the pinned model is in the store (local pull on the box).
echo "== ensuring model in store =="
if ! "$ATHENA_BIN" show "$TEST_MODEL" \
     --model-store "$MODEL_STORE" >/dev/null 2>&1; then
  "$ATHENA_BIN" pull "$TEST_MODEL" --model-store "$MODEL_STORE"
fi

LANIP="$(ipconfig getifaddr en0 2>/dev/null \
  || ipconfig getifaddr en1 2>/dev/null || echo "<studio-lan-ip>")"
echo
echo "════════════════════════════════════════════════════════════"
echo " READY"
echo "   health  : curl http://$LANIP:$PORT/healthz"
echo "   console : http://$LANIP:$PORT/ui   (sign in as $ADMIN_USER)"
echo "   admin bearer token (copy to the MacBook, shown ONCE):"
echo "     $TOK"
echo
echo " On the MacBook:"
echo "   source deploy/integration/macbook-env.sh $LANIP $PORT"
echo "   (paste the token when prompted), then follow RUNBOOK.md"
echo
echo " NOTE: no TLS — token/cookie cross the LAN in plaintext."
echo "       Use a trusted LAN / VPN, or a TLS reverse proxy."
echo "════════════════════════════════════════════════════════════"
