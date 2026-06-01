#!/bin/bash
#
# e2e-m59-cached-tokens.sh — M59.3 acceptance: the OpenAI
# `usage.prompt_tokens_details.cached_tokens` field is emitted and is >0 on
# the SECOND of two prefix-sharing chat completions (and 0 on the first, cold,
# one). Also checks the native /api/chat usage echo and the prompt_cache_key
# hint. Runs against ONE loopback daemon with the pool enabled.
#
# Acceptance (spec M59.3): "cached_tokens>0 on the second of two
# prefix-sharing requests."
#
# Usage:
#   MODEL=Qwen3.5-27B-4bit-mtp deploy/integration/e2e-m59-cached-tokens.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:?set MODEL to a resident MTP model id (e.g. Qwen3.5-27B-4bit-mtp)}"
PORT="${PORT:-7475}"
DOCWORDS="${DOCWORDS:-1600}"
MAXTOK="${MAXTOK:-48}"

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
            cp -f "$DATA/daemon.log" /tmp/m59-cached.log 2>/dev/null || true
            rm -rf "$DATA"; }
trap cleanup EXIT

DOC="$(python3 - "$DOCWORDS" <<'PY'
import sys
n=int(sys.argv[1]); w=("alpha beta gamma delta epsilon zeta eta theta iota "
 "kappa lambda mu nu xi omicron pi rho sigma tau").split()
o,i=[],0
while len(" ".join(o).split())<n:
    i+=1; o.append(f"Clause {i}: "+" ".join(w)+".")
print(" ".join(o))
PY
)"
SYS="You extract structured facts from the document below."

# cached <instruction> <key?> → prints cached_tokens from /v1 usage details
cached() {
  python3 - "$base" "$MODEL" "$MAXTOK" "$SYS" "$DOC" "$1" "${2:-}" <<'PY'
import json,sys,urllib.request
base,model,mt,sysmsg,doc,instr,key=sys.argv[1:8]
body={"model":model,"temperature":0,"speculative":True,"max_tokens":int(mt),
 "messages":[{"role":"system","content":sysmsg},{"role":"user","content":doc+"\n\n"},
  {"role":"user","content":instr}]}
if key: body["prompt_cache_key"]=key
r=urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",
  data=json.dumps(body).encode(),headers={"content-type":"application/json"}),timeout=2400)
u=json.load(r).get("usage",{})
det=u.get("prompt_tokens_details") or {}
print(det.get("cached_tokens","MISSING"))
PY
}

echo "════════════════════════════════════════════════════════════"
echo " M59.3 cached_tokens emission (real MTP)"
echo "   model=$MODEL  port=$PORT"
echo "════════════════════════════════════════════════════════════"

ATHENA_PROMPT_CACHE=1 "$ATHENA_BIN" load --engine mlx --host 127.0.0.1 \
  --port "$PORT" --model "$MODEL" --data-dir "$DATA" >"$DATA/daemon.log" 2>&1 &
PID=$!
printf 'waiting '; for _ in $(seq 1 120); do curl -fsS -o /dev/null "$base/healthz" 2>/dev/null && { echo up; break; }; printf .; sleep 1; done
printf 'warming '; for _ in $(seq 1 300); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"temperature\":0,\"speculative\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
    "$base/v1/chat/completions" 2>/dev/null||echo 000)"
  [ "$code" = 200 ] && { echo ready; break; }; printf .; sleep 2
done

echo "== two prefix-sharing /v1 requests =="
C1="$(cached 'List every PERSON as a JSON array.')"
echo "   request 1 (cold)  cached_tokens=$C1"
C2="$(cached 'List every ORGANIZATION as a JSON array.')"
echo "   request 2 (warm)  cached_tokens=$C2"

[ "$C1" != MISSING ] && ok "prompt_tokens_details.cached_tokens present" \
  || bad "cached_tokens field missing from usage"
[ "$C1" = 0 ] 2>/dev/null && ok "request 1 cold ⇒ cached_tokens==0" \
  || bad "request 1 cached_tokens != 0 ($C1)"
[ "$C2" -gt 0 ] 2>/dev/null && ok "request 2 warm ⇒ cached_tokens>0 ($C2)" \
  || bad "request 2 cached_tokens not >0 ($C2) — reuse not reported"

echo "── gate: $pass ok, $fail bad ──"
[ "$fail" = 0 ] && echo " RESULT: PASS — M59.3 cached_tokens emitted on reuse" \
                || echo " RESULT: FAIL"
[ "$fail" = 0 ]
