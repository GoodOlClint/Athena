#!/bin/bash
#
# e2e-m59-disk-restart.sh — host-bound proof that the ADR 027 disk KV tier
# resumes a session ACROSS A PROCESS RESTART and is still BIT-IDENTICAL to a
# cold prefill.
#
# This is the S3b acceptance gate. It extends e2e-m59-prefix-cache.sh from
# "warm reuse within one process" to "reuse from an ENCRYPTED disk blob written
# by a DIFFERENT process":
#
#   • Daemon #1 (cold ref): persist OFF, run B first on an empty cache →
#     genuine COLD prefill → B_cold.
#   • Daemon #2a (prime+spill): persist ON (keyfile KEK), data-dir D2. Run A
#     (primes the shared [system+doc] prefix), then DELETE /api/cache/prompt →
#     flushIdle spills A to <D2>/prompt-cache as AES-256-GCM blobs and drops it
#     from RAM. SIGTERM #2a and wait for it to exit.
#   • Daemon #2b (restore): a FRESH process, persist ON, SAME data-dir D2. Run
#     B → an in-RAM miss falls through to the disk tier, the prompt probes
#     descending 512-boundaries, finds A's spilled boundary, decrypts + restores
#     → WARM-FROM-DISK reuse → B_disk.
#   • B_disk == B_cold (byte-for-byte) ⇒ PASS. Because the restore happens in a
#     process that never saw A, this proves the on-disk serialize→encrypt→
#     restart→decrypt→decode round-trip is bit-preserving (not just in-RAM).
#
# Usage:
#   MODEL=Qwen3.5-27B-4bit-mtp deploy/integration/e2e-m59-disk-restart.sh
#
# Tunables (env): MODEL (required MTP id), ATHENA_BIN, PORT (default 7471),
#   MAXTOK (200), DOCWORDS (2400).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:?set MODEL to a resident MTP model id (e.g. Qwen3.5-27B-4bit-mtp)}"
PORT="${PORT:-7471}"
MAXTOK="${MAXTOK:-200}"
DOCWORDS="${DOCWORDS:-2400}"

if [ -z "${ATHENA_BIN:-}" ]; then
  for c in .build/xcode/Build/Products/Release/athena \
           .build/xcode/Build/Products/Debug/athena; do
    [ -x "$c" ] && ATHENA_BIN="$c" && break
  done
fi
[ -x "${ATHENA_BIN:-/nonexistent}" ] || {
  echo "no athena binary — set ATHENA_BIN or run deploy/build.sh Release" >&2; exit 2; }

pass=0; fail=0
ok()  { echo "  ok  — $*"; pass=$((pass + 1)); }
bad() { echo "  BAD — $*"; fail=$((fail + 1)); }

D1="$(mktemp -d)"; D2="$(mktemp -d)"; KF="$(mktemp)"
head -c 64 /dev/urandom > "$KF"          # the keyfile KEK (≥32 bytes)
P1="$PORT"; P2="$((PORT + 1))"; P3="$((PORT + 2))"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
            cp -f "$D1/daemon.log" /tmp/m59disk-d1.log 2>/dev/null || true
            cp -f "$D2/d2a.log" /tmp/m59disk-d2a.log 2>/dev/null || true
            cp -f "$D2/d2b.log" /tmp/m59disk-d2b.log 2>/dev/null || true
            rm -rf "$D1" "$D2" "$KF"; }
trap cleanup EXIT

DOC="$(python3 - "$DOCWORDS" <<'PY'
import sys
n = int(sys.argv[1])
words = ("the quick brown fox jumps over the lazy dog while the auditor "
         "records every transaction and the clerk files the report").split()
out, i = [], 0
while len(" ".join(out).split()) < n:
    i += 1
    out.append(f"Section {i}: " + " ".join(words) + ".")
print(" ".join(out))
PY
)"
SYS="You are a meticulous information-extraction engine. Read the document and answer only with the requested JSON."
INSTR_A='List every distinct PERSON mentioned, as a JSON array of strings.'
INSTR_B='List every distinct ORGANIZATION mentioned, as a JSON array of strings.'

# start_daemon <data-dir> <port> <logfile> <persist 0|1>
start_daemon() {
  # Word-split into separate KEY=VAL args for `env` (KF is a space-free mktemp
  # path). Avoids bash-3.2's empty-array-under-`set -u` error.
  local persist_env=""
  [ "$4" = 1 ] && persist_env="ATHENA_PROMPT_CACHE_PERSIST=1 ATHENA_PROMPT_CACHE_PERSIST_KEYFILE=$KF"
  env ATHENA_PROMPT_CACHE=1 $persist_env \
    "$ATHENA_BIN" load --engine mlx \
    --host 127.0.0.1 --port "$2" --model "$MODEL" --data-dir "$1" \
    >"$3" 2>&1 &
  PIDS+=($!)
  LAST_PID=$!
  printf 'waiting for :%s ' "$2"
  for _ in $(seq 1 120); do
    curl -fsS -o /dev/null "http://127.0.0.1:$2/healthz" 2>/dev/null && { echo " up"; break; }
    printf '.'; sleep 1
  done
  printf 'warming :%s ' "$2"
  for _ in $(seq 1 300); do
    code="$(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"temperature\":0,\"speculative\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
      "http://127.0.0.1:$2/v1/chat/completions" 2>/dev/null || echo 000)"
    [ "$code" = 200 ] && { echo " ready"; return 0; }
    printf '.'; sleep 2
  done
  echo " MODEL-NOT-READY (last=$code)"; tail -30 "$3"; return 1
}

req() {
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

echo "════════════════════════════════════════════════════════════"
echo " ADR 027 S3b — disk KV restart gate (real MTP)"
echo "   binary : $ATHENA_BIN"
echo "   model  : $MODEL   maxtok=$MAXTOK   doc≈${DOCWORDS}w"
echo "════════════════════════════════════════════════════════════"

echo "== daemon #1 (cold reference): B first, persist OFF =="
start_daemon "$D1" "$P1" "$D1/daemon.log" 0 || exit 1
BCOLD="$(req "$P1" "$INSTR_B")"; echo "   B(cold) bytes=${#BCOLD}"

echo "== daemon #2a: persist ON, prime A, flush to disk, then SIGTERM =="
start_daemon "$D2" "$P2" "$D2/d2a.log" 1 || exit 1
D2A_PID=$LAST_PID
A="$(req "$P2" "$INSTR_A")"; echo "   A(prime) bytes=${#A}"
curl -fsS -X DELETE "http://127.0.0.1:$P2/api/cache/prompt" >/dev/null 2>&1 \
  && ok "flushed prompt cache → spill to disk" || bad "flush request failed"
nblobs="$(find "$D2/prompt-cache" -name '*.kvs' 2>/dev/null | wc -l | tr -d ' ')"
[ "${nblobs:-0}" -gt 0 ] && ok "disk blobs written: $nblobs .kvs file(s)" \
                          || bad "no .kvs blobs on disk under $D2/prompt-cache"
# Confirm at-rest ciphertext: the verbatim doc text must NOT appear in any blob.
if grep -rqa "the quick brown fox jumps over the lazy dog" "$D2/prompt-cache" 2>/dev/null; then
  bad "plaintext prompt found in a disk blob"
else
  ok "disk blobs are ciphertext (no plaintext prompt on disk)"
fi
kill "$D2A_PID" 2>/dev/null || true
printf 'waiting for #2a to exit '; for _ in $(seq 1 30); do
  kill -0 "$D2A_PID" 2>/dev/null || { echo " gone"; break; }; printf '.'; sleep 1; done

echo "== daemon #2b (FRESH process, SAME data-dir): B restores from disk =="
start_daemon "$D2" "$P3" "$D2/d2b.log" 1 || exit 1
BDISK="$(req "$P3" "$INSTR_B")"; echo "   B(disk) bytes=${#BDISK}"
if grep -qE 'prefix-cache DISK HIT' "$D2/d2b.log"; then
  ok "B restored from disk on #2b: $(grep -E 'prefix-cache DISK HIT' "$D2/d2b.log" | tail -1 | sed 's/.*prefix-cache/prefix-cache/')"
else
  bad "no DISK HIT on #2b — disk restore did not engage"; tail -20 "$D2/d2b.log"
fi

echo "== compare =="
if [ "$BDISK" = "$BCOLD" ]; then
  ok "BIT-IDENTICAL across restart: disk-restored B == cold-prefill B"
else
  bad "DIVERGED: disk B != cold B (cross-restart restore is NOT bit-identical)"
  diff <(printf '%s' "$BDISK") <(printf '%s' "$BCOLD") || true
fi

echo "──────────────────────────────────────────"
echo " gate: $pass ok, $fail bad"
[ "$fail" = 0 ] && echo " RESULT: PASS — ADR 027 disk restart gate holds" \
                || echo " RESULT: FAIL — do NOT flip prompt_cache_persist_to_disk on"
echo "──────────────────────────────────────────"
[ "$fail" = 0 ]
