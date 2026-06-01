#!/bin/bash
#
# e2e-m59-governor.sh — M59.2 acceptance: the prompt-prefix KV pool is GOVERNED
# (bounded by entry count + bytes) and never drives the box to OOM under a
# sustained multi-pass loop, and /healthz reports pool bytes + entries.
#
# Brings up ONE loopback daemon with the pool enabled (ATHENA_PROMPT_CACHE=1)
# and fires a stream of DISTINCT long-prefix requests (each shares [SYS+DOC]
# and stores its own entry). With ROUNDS > the built-in entry cap (4), the
# pool MUST stay bounded rather than growing once per request. Asserts:
#   • /healthz.promptCachePoolEntries  in (0, 4]            — bounded by count
#   • /healthz.promptCachePoolBytes     > 0                  — reported
#   • promptCachePoolBytes <= promptCacheCapBytes            — within the cap
#   • daemon still healthy after the loop                    — no OOM / crash
#
# NOTE on caps: only the ENABLE flag is env-backed (ATHENA_PROMPT_CACHE); the
# numeric caps (prompt_cache_max_entries / _max_bytes / _idle_ttl_secs) come
# from TOML, which the daemon reads from its installed config path. This
# self-contained run therefore exercises the BUILT-IN defaults (entries=4,
# bytes=governor cap, idle=600s). The count-bound at 4 despite ROUNDS=7 is the
# headline M59.2 proof. To exercise a custom byte cap or the idle-TTL drain,
# set the keys in the daemon's config and raise ROUNDS / lower the TTL.
#
# Usage:
#   MODEL=Qwen3.5-27B-4bit-mtp deploy/integration/e2e-m59-governor.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:?set MODEL to a resident MTP model id (e.g. Qwen3.5-27B-4bit-mtp)}"
PORT="${PORT:-7471}"
ROUNDS="${ROUNDS:-7}"
DOCWORDS="${DOCWORDS:-1600}"
MAXTOK="${MAXTOK:-32}"
ENTRY_CAP=4   # built-in default (prompt_cache_max_entries)

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

DATA="$(mktemp -d)"; base="http://127.0.0.1:$PORT"
cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null || true
            cp -f "$DATA/daemon.log" /tmp/m59-gov.log 2>/dev/null || true
            rm -rf "$DATA"; }
trap cleanup EXIT

SYS="You extract structured facts from the document below."

hz() { curl -fsS "$base/healthz" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('$1',0))"; }

# Each round uses a DISTINCT document (its first token differs), so every
# request is a fresh COLD MISS that stores its OWN entry — that's what forces
# the pool past the entry cap and exercises LRU eviction. (The warm-reuse path
# is covered by the M59.1 bit-identical gate.)
fire() {  # $1 = round number → unique [SYS+DOC_n]
  python3 - "$base" "$MODEL" "$MAXTOK" "$SYS" "$DOCWORDS" "$1" >/dev/null 2>&1 <<'PY' || true
import json,sys,urllib.request
base,model,mt,sysmsg,docwords,n=sys.argv[1:7]
w=("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
   "omicron pi rho sigma tau").split()
o,i=[],0
while len(" ".join(o).split())<int(docwords):
    i+=1; o.append(f"Clause {i}: "+" ".join(w)+".")
doc=f"Document {n} (unique corpus {n}). "+" ".join(o)   # unique leading tokens
body={"model":model,"temperature":0,"speculative":True,"max_tokens":int(mt),
 "messages":[{"role":"system","content":sysmsg},{"role":"user","content":doc+"\n\n"},
  {"role":"user","content":"List distinct entities as a JSON array."}]}
try:
  urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",
    data=json.dumps(body).encode(),headers={"content-type":"application/json"}),timeout=2400).read()
except Exception: pass
PY
}

echo "════════════════════════════════════════════════════════════"
echo " M59.2 prompt-cache governor accounting + bounding"
echo "   model=$MODEL  rounds=$ROUNDS  built-in entry cap=$ENTRY_CAP"
echo "════════════════════════════════════════════════════════════"

ATHENA_PROMPT_CACHE=1 "$ATHENA_BIN" load --engine mlx --host 127.0.0.1 \
  --port "$PORT" --model "$MODEL" --data-dir "$DATA" >"$DATA/daemon.log" 2>&1 &
PID=$!
printf 'waiting for /healthz '
for _ in $(seq 1 120); do curl -fsS -o /dev/null "$base/healthz" 2>/dev/null && { echo up; break; }; printf .; sleep 1; done
printf 'warming '
for _ in $(seq 1 300); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"temperature\":0,\"speculative\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
    "$base/v1/chat/completions" 2>/dev/null||echo 000)"
  [ "$code" = 200 ] && { echo ready; break; }; printf .; sleep 2
done

echo "== sustained loop: $ROUNDS distinct long-prefix requests =="
for n in $(seq 1 "$ROUNDS"); do
  fire "$n"
  echo "   round $n → entries=$(hz promptCachePoolEntries) bytes=$(hz promptCachePoolBytes)"
done

ENTRIES="$(hz promptCachePoolEntries)"; BYTES="$(hz promptCachePoolBytes)"; CAPN="$(hz promptCacheCapBytes)"
echo "== after loop: entries=$ENTRIES bytes=$BYTES cap=$CAPN =="

[ "$ENTRIES" -gt 0 ] 2>/dev/null && ok "pool reports >0 entries ($ENTRIES)" || bad "entries not >0 ($ENTRIES)"
# $ROUNDS(>cap) distinct cold misses must settle at EXACTLY the cap — proving
# the pool both grew AND evicted LRU rather than growing unbounded.
[ "$ENTRIES" = "$ENTRY_CAP" ] 2>/dev/null \
  && ok "evicted to entry cap (==$ENTRY_CAP from $ROUNDS distinct prompts)" \
  || bad "expected $ENTRY_CAP entries after $ROUNDS distinct prompts, got $ENTRIES"
[ "$BYTES" -gt 0 ] 2>/dev/null && ok "/healthz reports pool bytes ($BYTES)" || bad "pool bytes not reported"
[ "$CAPN" -gt 0 ] 2>/dev/null && [ "$BYTES" -le "$CAPN" ] 2>/dev/null \
  && ok "pool bytes within governor cap ($BYTES ≤ $CAPN)" || bad "pool bytes exceed cap ($BYTES vs $CAPN)"
curl -fsS -o /dev/null "$base/healthz" 2>/dev/null \
  && ok "daemon still healthy after sustained loop (no OOM/crash)" || bad "daemon unhealthy after loop"

echo "──────────────────────────────────────────"
echo " gate: $pass ok, $fail bad"
[ "$fail" = 0 ] && echo " RESULT: PASS — M59.2 pool is governed + bounded" || echo " RESULT: FAIL"
echo "──────────────────────────────────────────"
[ "$fail" = 0 ]
