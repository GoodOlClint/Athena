#!/bin/bash
#
# e2e-m59-prefix-cache.sh — host-bound proof that cross-request prompt-prefix
# KV reuse (M59.1) is BIT-IDENTICAL to a cold prefill on a real MTP model.
#
# This is the slice-1 acceptance gate, same discipline as M20 TurboQuant /
# M21 TriAttention: prefix-reuse + suffix-prefill MUST produce byte-identical
# greedy output to a cold full prefill. It needs a real Qwen3.5-*-mtp model
# (the stub/CI path has no MTP backbone and no recurrent state), so it stands
# up loopback daemons itself using the built binary.
#
# WHY the shape it has:
#   • Greedy (temperature 0) + speculative + a FIXED max_tokens makes the
#     output a deterministic function of the prompt — so a byte diff is a
#     real correctness signal.
#   • Request A and request B share a long bit-identical prefix
#     [system + verbatim doc] and DIVERGE only in the trailing instruction.
#     The doc is sized so the shared prefix spans several 512-token prefill
#     chunks AND the divergence lands mid-chunk — the hard path (recurrent
#     checkpoint restore at B=floor(L/512)*512 + sub-chunk re-run [B:L] +
#     suffix prefill). A short/aligned prefix would NOT exercise the replay.
#
# HOW it isolates cold vs warm WITHOUT a config flip (caching is ON for both,
# via ATHENA_PROMPT_CACHE=1) — it exploits "first request on a fresh daemon
# hits an EMPTY cache":
#   • Daemon #1: run B FIRST → empty cache → genuine COLD prefill → B_cold.
#   • Daemon #2 (fresh): run A (primes the shared [system+doc] prefix), THEN
#     run B → B hits A's entry at L=len(system+doc) and resumes from a 512
#     boundary inside it → WARM divergent reuse → B_warm.
#   • B_cold == B_warm (byte-for-byte) ⇒ PASS.
# Each daemon starts with an empty cache, so B is cold on #1 and warm on #2
# without ever needing to disable the feature or flush.
#
# cached_tokens is NOT asserted here (that surface is M59.3); this gate is
# correctness only. The warm HIT is independently confirmed from daemon #2's
# log line "prefix-cache HIT ... B=<512-multiple> suffix=<small>".
#
# Usage:
#   MODEL=Qwen3.5-2B-4bit-mtp deploy/integration/e2e-m59-prefix-cache.sh
#   ATHENA_BIN=... MODEL=Qwen3.5-27B-8bit-mtp PORT=7461 ... (env below)
#
# Tunables (env):
#   MODEL     resident MTP model id/dir (required; MUST have an MTP head)
#   ATHENA_BIN athena binary            (default: auto-resolve xcode Release)
#   PORT      base listen port          (default: 7461; #2 uses PORT+1)
#   MAXTOK    fixed completion cap      (default: 200)
#   DOCWORDS  approx doc length (words) (default: 2400 ⇒ ~3k tokens, >5 chunks)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

MODEL="${MODEL:?set MODEL to a resident MTP model id (e.g. Qwen3.5-2B-4bit-mtp)}"
PORT="${PORT:-7461}"
MAXTOK="${MAXTOK:-200}"
DOCWORDS="${DOCWORDS:-2400}"

if [ -z "${ATHENA_BIN:-}" ]; then
  for c in .build/xcode/Build/Products/Release/athena \
           .build/xcode/Build/Products/Debug/athena; do
    [ -x "$c" ] && ATHENA_BIN="$c" && break
  done
fi
[ -x "${ATHENA_BIN:-/nonexistent}" ] || {
  echo "no athena binary — set ATHENA_BIN or run deploy/build.sh" >&2; exit 2; }

pass=0; fail=0
ok()  { echo "  ok  — $*"; pass=$((pass + 1)); }
bad() { echo "  BAD — $*"; fail=$((fail + 1)); }

D1="$(mktemp -d)"; D2="$(mktemp -d)"
P1="$PORT"; P2="$((PORT + 1))"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
            # preserve daemon logs for post-mortem before removing temp dirs
            cp -f "$D1/daemon.log" /tmp/m59-d1.log 2>/dev/null || true
            cp -f "$D2/daemon.log" /tmp/m59-d2.log 2>/dev/null || true
            rm -rf "$D1" "$D2"; }
trap cleanup EXIT

# A long, deterministic, verbatim "document" — identical every run so the
# shared prefix is bit-identical across A and B.
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

# start_daemon <data-dir> <port> <logfile>
start_daemon() {
  ATHENA_PROMPT_CACHE=1 "$ATHENA_BIN" load --engine mlx \
    --host 127.0.0.1 --port "$2" --model "$MODEL" --data-dir "$1" \
    >"$3" 2>&1 &
  PIDS+=($!)
  printf 'waiting for :%s /healthz ' "$2"
  for _ in $(seq 1 120); do
    curl -fsS -o /dev/null "http://127.0.0.1:$2/healthz" 2>/dev/null && { echo " up"; break; }
    printf '.'; sleep 1
  done
  curl -fsS -o /dev/null "http://127.0.0.1:$2/healthz" 2>/dev/null || {
    echo " TIMEOUT"; tail -30 "$3"; return 1; }
  # The model loads lazily/in-background; first requests get a cold-load 503
  # (M43.2). Warm it with a tiny request until it answers 200 before we start
  # timing the real prompts. Generous budget: a 27B can take minutes to map.
  printf 'warming model on :%s ' "$2"
  for _ in $(seq 1 300); do
    code="$(curl -s -o /dev/null -w '%{http_code}' \
      -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"temperature\":0,\"speculative\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
      "http://127.0.0.1:$2/v1/chat/completions" 2>/dev/null || echo 000)"
    [ "$code" = 200 ] && { echo " ready"; return 0; }
    printf '.'; sleep 2
  done
  echo " MODEL-NOT-READY (last=$code)"; tail -30 "$3"; return 1
}

# req <port> <instruction> → assistant message content
req() {
  python3 - "$1" "$MODEL" "$MAXTOK" "$SYS" "$DOC" "$2" <<'PY'
import json, sys, urllib.request
port, model, maxtok, sysmsg, doc, instr = sys.argv[1:7]
body = {"model": model, "temperature": 0, "speculative": True,
        "max_tokens": int(maxtok),
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
echo " M59.1 prompt-prefix cache — bit-identical gate (real MTP)"
echo "   binary : $ATHENA_BIN"
echo "   model  : $MODEL   maxtok=$MAXTOK   doc≈${DOCWORDS}w"
echo "   ports  : cold=$P1  warm=$P2   (caching ON for both)"
echo "════════════════════════════════════════════════════════════"

echo "== daemon #1 (cold reference): B first on an empty cache =="
start_daemon "$D1" "$P1" "$D1/daemon.log" || exit 1
BCOLD="$(req "$P1" "$INSTR_B")"
echo "   B(cold) bytes=${#BCOLD}"
grep -qE 'prefix-cache MISS' "$D1/daemon.log" \
  && ok "B was a cold MISS on daemon #1" \
  || echo "  (note: no MISS log line seen — check $D1/daemon.log)"

echo "== daemon #2 (warm): prime A, then B reuses the shared prefix =="
start_daemon "$D2" "$P2" "$D2/daemon.log" || exit 1
A="$(req "$P2" "$INSTR_A")"; echo "   A(prime) bytes=${#A}"
BWARM="$(req "$P2" "$INSTR_B")"; echo "   B(warm) bytes=${#BWARM}"
if grep -qE 'prefix-cache HIT' "$D2/daemon.log"; then
  ok "B hit the cache on daemon #2: $(grep -E 'prefix-cache HIT' "$D2/daemon.log" | tail -1 | sed 's/.*prefix-cache/prefix-cache/')"
else
  bad "no prefix-cache HIT on daemon #2 — warm path did not engage"
  tail -20 "$D2/daemon.log"
fi

echo "== compare =="
if [ "$BWARM" = "$BCOLD" ]; then
  ok "BIT-IDENTICAL: warm-prefix-reuse B == cold-prefill B"
else
  bad "DIVERGED: warm B != cold B (prefix reuse is NOT bit-identical)"
  echo "----- diff (warm ↓ / cold ↑) -----"
  diff <(printf '%s' "$BWARM") <(printf '%s' "$BCOLD") || true
  echo "----------------------------------"
fi

echo "──────────────────────────────────────────"
echo " gate: $pass ok, $fail bad"
[ "$fail" = 0 ] && echo " RESULT: PASS — M59.1 bit-identical gate holds" \
                || echo " RESULT: FAIL — do NOT flip prompt_cache on"
echo "──────────────────────────────────────────"
[ "$fail" = 0 ]
