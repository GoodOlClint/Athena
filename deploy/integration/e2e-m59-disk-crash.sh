#!/bin/bash
#
# e2e-m59-disk-crash.sh — host-bound proof that the ADR 027 S4 EAGER/frontier
# spill survives a HARD CRASH (SIGKILL), not just a graceful shutdown.
#
# S3b's restart gate spills on a clean flush/shutdown. S4 adds eager spill at the
# store seam (prompt_cache_persist_eager). This gate proves that path:
#
#   • Daemon #A: persist ON + eager ON. Run A (a long prompt). The eager spill
#     writes A's blobs to <D>/prompt-cache DURING store() — no flush, no
#     shutdown. Then `kill -9` (SIGKILL): no graceful drain, no shutdown spill.
#   • Daemon #B: a FRESH process, SAME data-dir. Run B (shares A's prefix) →
#     DISK HIT ⇒ the eager spill persisted A before the crash and survived it.
#
# Bit-identicality of the disk restore is already proven by e2e-m59-disk-restart;
# this gate asserts only that the EAGER spill wrote-before-crash and restores.
#
# Usage: MODEL=Qwen3.5-27B-4bit-mtp deploy/integration/e2e-m59-disk-crash.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:?set MODEL to a resident MTP model id}"
PORT="${PORT:-7481}"
MAXTOK="${MAXTOK:-32}"
DOCWORDS="${DOCWORDS:-2400}"

if [ -z "${ATHENA_BIN:-}" ]; then
  for c in .build/xcode/Build/Products/Release/athena \
           .build/xcode/Build/Products/Debug/athena; do
    [ -x "$c" ] && ATHENA_BIN="$c" && break
  done
fi
[ -x "${ATHENA_BIN:-/nonexistent}" ] || { echo "no athena binary" >&2; exit 2; }

pass=0; fail=0
ok()  { echo "  ok  — $*"; pass=$((pass + 1)); }
bad() { echo "  BAD — $*"; fail=$((fail + 1)); }

D="$(mktemp -d)"; KF="$(mktemp)"; head -c 64 /dev/urandom > "$KF"
PA="$PORT"; PB="$((PORT + 1))"; PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
            cp -f "$D/a.log" /tmp/m59crash-a.log 2>/dev/null || true
            cp -f "$D/b.log" /tmp/m59crash-b.log 2>/dev/null || true
            rm -rf "$D" "$KF"; }
trap cleanup EXIT

DOC="$(python3 - "$DOCWORDS" <<'PY'
import sys
n = int(sys.argv[1])
w = ("the quick brown fox jumps over the lazy dog while the auditor records "
     "every transaction and the clerk files the report").split()
out, i = [], 0
while len(" ".join(out).split()) < n:
    i += 1; out.append(f"Section {i}: " + " ".join(w) + ".")
print(" ".join(out))
PY
)"
SYS="You are a meticulous information-extraction engine."
INSTR_A='List every distinct PERSON mentioned, as a JSON array of strings.'
INSTR_B='List every distinct ORGANIZATION mentioned, as a JSON array of strings.'

start_daemon() {  # <data-dir> <port> <logfile>
  env ATHENA_PROMPT_CACHE=1 ATHENA_PROMPT_CACHE_PERSIST=1 \
      ATHENA_PROMPT_CACHE_PERSIST_EAGER=1 ATHENA_PROMPT_CACHE_PERSIST_KEYFILE="$KF" \
    "$ATHENA_BIN" load --engine mlx --host 127.0.0.1 --port "$2" \
    --model "$MODEL" --data-dir "$1" >"$3" 2>&1 &
  PIDS+=($!); LAST_PID=$!
  for _ in $(seq 1 120); do
    curl -fsS -o /dev/null "http://127.0.0.1:$2/healthz" 2>/dev/null && break; sleep 1; done
  printf 'warming :%s ' "$2"
  for _ in $(seq 1 300); do
    code="$(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"temperature\":0,\"speculative\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
      "http://127.0.0.1:$2/v1/chat/completions" 2>/dev/null || echo 000)"
    [ "$code" = 200 ] && { echo " ready"; return 0; }; printf '.'; sleep 2; done
  echo " NOT-READY"; tail -20 "$3"; return 1
}
req() {  # <port> <instr>
  python3 - "$1" "$MODEL" "$MAXTOK" "$SYS" "$DOC" "$2" <<'PY'
import json, sys, urllib.request
port, model, maxtok, sysmsg, doc, instr = sys.argv[1:7]
body = {"model": model, "temperature": 0, "speculative": True, "max_tokens": int(maxtok),
        "messages": [{"role": "system", "content": sysmsg},
                     {"role": "user", "content": doc + "\n\n"},
                     {"role": "user", "content": instr}]}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json"})
with urllib.request.urlopen(r, timeout=2400) as resp:
    sys.stdout.write(json.load(resp)["choices"][0]["message"]["content"])
PY
}

echo "═══ ADR 027 S4 — eager-spill crash-survival gate ($MODEL) ═══"
echo "== daemon #A: eager ON, prime A, then SIGKILL (no flush/shutdown) =="
start_daemon "$D" "$PA" "$D/a.log" || exit 1
_=$(req "$PA" "$INSTR_A"); echo "   A primed"
nb="$(find "$D/prompt-cache" -name '*.kvs' 2>/dev/null | wc -l | tr -d ' ')"
[ "${nb:-0}" -gt 0 ] && ok "eager spill wrote $nb .kvs blob(s) at store time (no flush)" \
                      || bad "no blobs after A — eager spill did not fire"
kill -9 "$LAST_PID" 2>/dev/null || true
printf 'hard-killed #A; waiting '; for _ in $(seq 1 30); do
  kill -0 "$LAST_PID" 2>/dev/null || { echo gone; break; }; printf .; sleep 1; done

echo "== daemon #B: FRESH process, SAME data-dir → B restores from disk =="
start_daemon "$D" "$PB" "$D/b.log" || exit 1
_=$(req "$PB" "$INSTR_B"); echo "   B done"
if grep -qE 'prefix-cache DISK HIT' "$D/b.log"; then
  ok "B restored from disk after crash: $(grep -E 'DISK HIT' "$D/b.log" | tail -1 | sed 's/.*prefix-cache/prefix-cache/')"
else
  bad "no DISK HIT on #B — eager-spilled blobs did not survive the crash"; tail -20 "$D/b.log"
fi

echo "─── gate: $pass ok, $fail bad ───"
[ "$fail" = 0 ] && echo " RESULT: PASS — ADR 027 eager-spill survives SIGKILL" \
                || echo " RESULT: FAIL"
[ "$fail" = 0 ]
