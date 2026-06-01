#!/bin/bash
#
# e2e-m59-cache-api.sh — M59.4 acceptance: the operator surface for the
# prompt-prefix KV pool. Brings up ONE loopback daemon (auth OFF, so the
# admin-gated routes are open to the loopback operator) with the pool enabled
# and asserts:
#   • GET  /api/cache/prompt          → {enabled:true, ...} and reflects a
#                                        populated pool after a request
#   • DELETE /api/cache/prompt        → flushes (flushed>0), GET then shows 0
#   • the flush is recorded in the audit trail (prompt_cache.flush)
#   • the OpenAPI drift-guard: GET /openapi.json documents both methods of
#     /api/cache/prompt, and there is no spec↔route drift for it
#
# Usage:
#   MODEL=Qwen3.5-27B-4bit-mtp deploy/integration/e2e-m59-cache-api.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:?set MODEL to a resident MTP model id (e.g. Qwen3.5-27B-4bit-mtp)}"
PORT="${PORT:-7479}"
DOCWORDS="${DOCWORDS:-1600}"

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
            cp -f "$DATA/daemon.log" /tmp/m59-cacheapi.log 2>/dev/null || true
            rm -rf "$DATA"; }
trap cleanup EXIT

jget() { python3 -c "import sys,json;print(json.load(sys.stdin).get('$1',''))"; }

DOC="$(python3 - "$DOCWORDS" <<'PY'
import sys
n=int(sys.argv[1]); w="alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
o,i=[],0
while len(" ".join(o).split())<n:
    i+=1; o.append(f"Clause {i}: "+" ".join(w)+".")
print(" ".join(o))
PY
)"
fire() {  # one populating request (≥512-token prompt ⇒ stores an entry)
  python3 - "$base" "$MODEL" "$DOC" >/dev/null 2>&1 <<'PY' || true
import json,sys,urllib.request
base,model,doc=sys.argv[1:4]
body={"model":model,"temperature":0,"speculative":True,"max_tokens":16,
 "messages":[{"role":"user","content":doc+"\n\nList entities as JSON array."}]}
try:
  urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",
    data=json.dumps(body).encode(),headers={"content-type":"application/json"}),timeout=2400).read()
except Exception: pass
PY
}

echo "════════════════════════════════════════════════════════════"
echo " M59.4 prompt-cache operator API + drift-guard"
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

echo "== GET /api/cache/prompt (enabled) =="
S="$(curl -fsS "$base/api/cache/prompt")"
[ "$(printf '%s' "$S" | jget enabled)" = True ] && ok "stats report enabled=true" || bad "not enabled ($S)"

echo "== populate then re-check occupancy =="
fire
S="$(curl -fsS "$base/api/cache/prompt")"
EN="$(printf '%s' "$S" | jget entries)"; BY="$(printf '%s' "$S" | jget bytes)"
echo "   entries=$EN bytes=$BY"
[ "$EN" -ge 1 ] 2>/dev/null && ok "pool populated (entries=$EN)" || bad "pool not populated ($S)"

echo "== DELETE /api/cache/prompt (flush) =="
F="$(curl -fsS -X DELETE "$base/api/cache/prompt")"
FL="$(printf '%s' "$F" | jget flushed)"
echo "   flushed=$FL"
[ "$FL" -ge 1 ] 2>/dev/null && ok "flush freed $FL entr(y/ies)" || bad "flush freed nothing ($F)"
S="$(curl -fsS "$base/api/cache/prompt")"
[ "$(printf '%s' "$S" | jget entries)" = 0 ] && ok "pool empty after flush" || bad "pool not empty after flush ($S)"

echo "== flush audited =="
A="$(curl -fsS "$base/api/audit?action=prompt_cache.flush")"
echo "$A" | grep -q 'prompt_cache.flush' && ok "flush recorded in audit trail" || bad "flush not audited ($A)"

echo "== OpenAPI drift-guard for /api/cache/prompt =="
curl -fsS "$base/openapi.json" > "$DATA/spec.json"
python3 - "$DATA/spec.json" <<'PY' && ok "spec documents GET+DELETE /api/cache/prompt" || bad "spec missing /api/cache/prompt methods"
import json,sys
ops=json.load(open(sys.argv[1])).get("paths",{}).get("/api/cache/prompt",{})
sys.exit(0 if ("get" in ops and "delete" in ops) else 1)
PY
# Full bidirectional drift-guard (every /v1+/api route documented, no stale).
if python3 - "$DATA/spec.json" Sources/athena/Server/AthenaServer.swift <<'PY'
import json,re,sys
spec=json.load(open(sys.argv[1])); paths=spec.get("paths",{})
src=open(sys.argv[2]).read()
routes=re.findall(r'router\.(get|post|put|delete|patch)\("([^"]+)"',src)
op={"/healthz","/metrics","/openapi.json"}; HTTP={"get","post","put","delete","patch"}
ins=lambda p:p.startswith("/v1/") or p.startswith("/api/") or p in op
norm=lambda p:re.sub(r":([A-Za-z_]+)",r"{\1}",p)
live={(m,norm(p)) for m,p in routes if ins(p)}
doc={(m,path) for path,ops in paths.items() for m in ops if m in HTTP}
miss=sorted(m.upper()+" "+p for m,p in live-doc); stale=sorted(m.upper()+" "+p for m,p in doc-live)
if miss: print("UNDOCUMENTED: "+"; ".join(miss)); sys.exit(1)
if stale: print("STALE: "+"; ".join(stale)); sys.exit(1)
print("spec ↔ routes exact (%d routes)"%len(live)); sys.exit(0)
PY
then ok "spec ↔ routes exact (no drift)"; else bad "OpenAPI drift"; fi

echo "──────────────────────────────────────────"
echo " gate: $pass ok, $fail bad"
[ "$fail" = 0 ] && echo " RESULT: PASS — M59.4 operator surface + drift-guard" || echo " RESULT: FAIL"
[ "$fail" = 0 ]
