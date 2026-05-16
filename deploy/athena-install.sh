#!/bin/bash
#
# athena-install.sh — install Athena as a boot-time launchd system daemon.
#
# Athena is macOS/launchd (the the platform orchestrator is Linux/systemd on
# Crete); this mirrors the the platform deploy idiom (committed deploy/<svc> config
# + rendered unit + load) but runs LOCALLY on the Mac Studio — there is no
# SSH to a Crete guest. Run it on the Mac itself, as root.
#
# Prereqts: macOS, sudo, swift (Command Line Tools is enough — the daemon
# binary needs no test frameworks). No jq/TOML CLI needed: the flat config
# is parsed with sed.
#
# Env (override before invoking):
#   ATHENA_USER   service account (default: ${SUDO_USER:-$(whoami)})
#   ATHENA_LABEL  launchd label   (default: me.goodolclint.athena)
#
# Usage:  sudo ./deploy/athena-install.sh
#
set -euo pipefail

LABEL="${ATHENA_LABEL:-me.goodolclint.athena}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "error: must run as root (sudo $0)" >&2
  exit 1
fi

SERVICE_USER="${ATHENA_USER:-${SUDO_USER:-$(whoami)}}"

# --- preflight ---------------------------------------------------------
command -v swift >/dev/null || { echo "error: swift not on PATH" >&2; exit 1; }

CFG="deploy/athena.toml"
if [[ ! -f "$CFG" ]]; then
  echo "error: $CFG missing — copy deploy/templates/athena.toml.example" >&2
  exit 1
fi
TEMPLATE="deploy/templates/athena.plist.template"
[[ -f "$TEMPLATE" ]] || { echo "error: $TEMPLATE missing" >&2; exit 1; }

# Minimal TOML reader for our flat scalar schema: first uncommented
# `key = value` line; strips quotes, inline comments, surrounding space.
# Commented keys (e.g. `# budget_bytes = ...`) intentionally yield empty.
toml_get() {
  grep -E "^[[:space:]]*$1[[:space:]]*=" "$CFG" 2>/dev/null | head -1 \
    | sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]+$//; s/^"(.*)"$/\1/'
}

LISTEN_HOST="$(toml_get listen_host)"
LISTEN_PORT="$(toml_get listen_port)"
LOG_DIR="$(toml_get log_dir)"
BUDGET_BYTES="$(toml_get budget_bytes)"

[[ -n "$LISTEN_HOST" && -n "$LISTEN_PORT" && -n "$LOG_DIR" ]] || {
  echo "error: $CFG missing listen_host/listen_port/log_dir" >&2; exit 1; }

# --- build (release; CLT swift is sufficient for the executable) -------
echo "==> building athena (release)"
swift build -c release
BIN_SRC="$REPO_ROOT/.build/release/athena"
[[ -x "$BIN_SRC" ]] || { echo "error: build produced no binary" >&2; exit 1; }

# --- install binary, config, dirs -------------------------------------
install -d -m 0755 /usr/local/bin
install -m 0755 "$BIN_SRC" /usr/local/bin/athena

install -d -m 0755 /usr/local/etc/athena
install -m 0644 "$CFG" /usr/local/etc/athena/athena.toml

install -d -o "$SERVICE_USER" -m 0755 /usr/local/var/athena
install -d -o "$SERVICE_USER" -m 0755 "$LOG_DIR"

# --- render the launchd plist -----------------------------------------
# The budget arg lives on one sentinel-tagged line: substitute its value and
# strip the sentinel when a budget is configured, else delete the line so the
# binary falls back to its 75%-of-RAM default. Pure sed — no multi-line awk.
if [[ -n "$BUDGET_BYTES" ]]; then
  BUDGET_SED=(-e "s#@BUDGET_VALUE@#${BUDGET_BYTES}#g" -e "s#<!--BUDGET-->##g")
else
  BUDGET_SED=(-e "/<!--BUDGET-->/d")
fi

PLIST="/Library/LaunchDaemons/${LABEL}.plist"
TMP_PLIST="$(mktemp)"
trap 'rm -f "$TMP_PLIST"' EXIT

sed "${BUDGET_SED[@]}" \
  -e "s#@ATHENA_BIN@#/usr/local/bin/athena#g" \
  -e "s#@ATHENA_USER@#${SERVICE_USER}#g" \
  -e "s#@LISTEN_HOST@#${LISTEN_HOST}#g" \
  -e "s#@LISTEN_PORT@#${LISTEN_PORT}#g" \
  -e "s#@LOG_DIR@#${LOG_DIR}#g" \
  "$TEMPLATE" >"$TMP_PLIST"

install -m 0644 -o root -g wheel "$TMP_PLIST" "$PLIST"

# --- (re)load the daemon ----------------------------------------------
echo "==> bootstrapping ${LABEL}"
launchctl bootout system "$PLIST" 2>/dev/null || true
launchctl bootstrap system "$PLIST"
launchctl enable "system/${LABEL}"
launchctl kickstart -k "system/${LABEL}"

echo
echo "installed. status:"
launchctl print "system/${LABEL}" 2>/dev/null | sed -n '1,12p' || true
echo
echo "logs:  tail -f ${LOG_DIR}/athena.err.log ${LOG_DIR}/athena.out.log"
echo "health: curl -s http://${LISTEN_HOST}:${LISTEN_PORT}/healthz"
