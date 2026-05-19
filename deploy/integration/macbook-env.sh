# shellcheck shell=bash
#
# macbook-env.sh — set up the local MacBook as the second node /
# operator console for the MANUAL integration tier (RUNBOOK.md).
# SOURCE this (don't execute it) so the helpers + env land in your
# shell:
#
#   source deploy/integration/macbook-env.sh <studio-host> [port]
#
# It exports ATHENA_KEY (so the bearer token never appears in argv /
# process list) and defines:
#   ath        — the portable client pointed at the Studio daemon
#   ath_ping   — health + status sanity check
#   ath_web    — print the WebUI console URL
#
# No `set -e` here: this file is sourced; a failed command must not
# kill the operator's interactive shell.

# Refuse to run as a subprocess (env would not persist). Detect
# "sourced" portably for both bash and zsh (operator shells differ).
_ath_sourced=1
if [ -n "${BASH_SOURCE:-}" ]; then
  [ "${BASH_SOURCE[0]}" = "${0}" ] && _ath_sourced=0
elif [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in
    *:file*) _ath_sourced=1 ;;
    *)       _ath_sourced=0 ;;
  esac
fi
if [ "$_ath_sourced" = 0 ]; then
  echo "source this file, do not execute it:" >&2
  echo "  source deploy/integration/macbook-env.sh <studio> [port]" \
    >&2
  exit 2
fi
unset _ath_sourced

ATHENA_STUDIO_HOST="${1:-${ATHENA_STUDIO_HOST:-}}"
ATHENA_STUDIO_PORT="${2:-${ATHENA_STUDIO_PORT:-7447}}"

if [ -z "$ATHENA_STUDIO_HOST" ]; then
  echo "usage: source macbook-env.sh <studio-host> [port]" >&2
  return 2 2>/dev/null || exit 2
fi

# Resolve the portable client binary.
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
if [ ! -x "${ATHENA_BIN:-/nonexistent}" ]; then
  echo "no athena binary — set ATHENA_BIN or build first" >&2
  return 2 2>/dev/null || exit 2
fi

# Bearer token via env (NOT argv). Prompt once if unset; the client
# reads ATHENA_KEY when --key is omitted.
if [ -z "${ATHENA_KEY:-}" ]; then
  read -r -s -p "Studio admin bearer token (sk-athena-…): " ATHENA_KEY
  echo
fi
export ATHENA_KEY
export ATHENA_BIN ATHENA_STUDIO_HOST ATHENA_STUDIO_PORT

# The portable client, always pointed off-box at the Studio daemon
# (DaemonOptions.isRemote is true for a non-loopback --host, so every
# verb drives the remote /api over HTTP).
ath() {
  "$ATHENA_BIN" "$@" \
    --host "$ATHENA_STUDIO_HOST" --port "$ATHENA_STUDIO_PORT"
}

ath_ping() {
  local b="http://$ATHENA_STUDIO_HOST:$ATHENA_STUDIO_PORT"
  echo "GET $b/healthz"
  curl -fsS "$b/healthz" && echo
  echo "--- athena status ---"
  ath status
}

ath_web() {
  echo "http://$ATHENA_STUDIO_HOST:$ATHENA_STUDIO_PORT/ui"
}

cat <<EOF
════════════════════════════════════════════════════════════
 Athena integration — MacBook node ready
   studio  : $ATHENA_STUDIO_HOST:$ATHENA_STUDIO_PORT
   client  : $ATHENA_BIN  (use the 'ath' wrapper)
   console : $(ath_web)

   ath <verb> …   e.g.  ath status | ath list | ath run <model> hi
   ath_ping       health + governor sanity
   ath_web        print the console URL (open in a real browser)

 NOTE: no TLS — the bearer token + the browser session cookie
 cross the LAN in plaintext. Trusted LAN / VPN only, or front it
 with a TLS reverse proxy. Do not run this over untrusted Wi-Fi.
════════════════════════════════════════════════════════════
EOF
