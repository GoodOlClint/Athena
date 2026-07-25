#!/bin/bash
#
# e2e-count-tokens.sh — end-to-end DoD for ADR 042 Track B (context discovery +
# exact pre-flight token counting). Runs a dev-mode daemon against the real
# model store and asserts:
#
#   1) GET /v1/models/{id} carries BOTH `context_length` and
#      `max_prompt_tokens`  (pre-change: neither key exists).
#   2) POST /v1/chat/completions/count_tokens returns `prompt_tokens` EXACTLY
#      equal to the `usage.prompt_tokens` /v1/chat/completions reports for the
#      IDENTICAL multi-turn body carrying a system prompt AND a tool
#      definition  (pre-change: the route 404s). Equality, not proximity — that
#      is the whole value of routing both through `container.prepare`, and the
#      tool definition is what proves tool-schema tokens are counted.
#   3) The count is CHEAP when the engine is idle (tens of ms — no prefill, no
#      eval), and a count issued while a long decode is in flight still returns
#      the same number with a 200.
#
#      NOT asserted: that a mid-decode count returns *promptly*. It does not.
#      Measured 2026-07-25: 34 ms idle, but 8.3 s when issued 3 s into an 11.3 s
#      decode — i.e. it waits out the decode. The route takes no ADR 029
#      execution gate, but the substrate's `ModelContainer` guards ALL context
#      access with an async mutex held for the whole decode
#      (`SerialAccessContainer.read`), and `prepare` needs that mutex to reach
#      the processor. ADR 042 §4(b) is amended accordingly — the "cheap every
#      turn even under load" claim is retracted, and the script prints the
#      measured latency as evidence rather than asserting a bound it cannot
#      honestly hold.
#
# MLX numerics can't run under `swift test` (ADR 009), so this is the heavy DoD
# — run it on a host with the model resident.
#
# Requires: a built Release binary + an LLM model in the store.
# Usage: ./deploy/e2e-count-tokens.sh [binary] [model-id]
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
MODEL="${2:-${ATHENA_LLM_MODEL:-gemma-4-26b-a4b-it-8bit}}"
PORT=7791
STORE="$HOME/.athena/models"
WORK="$(mktemp -d)"
DATA="$(mktemp -d)"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$WORK" "$DATA"' EXIT

[ -x "$BIN" ] || { echo "error: no binary at $BIN (build it first)"; exit 1; }
[ -d "$STORE/$MODEL" ] || { echo "SKIP: model '$MODEL' not in $STORE"; exit 0; }

echo "== starting dev-mode daemon on :$PORT (model=$MODEL) =="
"$BIN" load --port "$PORT" --data-dir "$DATA" --model-store "$STORE" \
  --llm-model "$MODEL" >"$WORK/daemon.log" 2>&1 &
DPID=$!
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done

BASE="http://127.0.0.1:$PORT"
fail=0

echo "== 1) GET /v1/models/{id} publishes both limits =="
curl -s -o "$WORK/model.json" "$BASE/v1/models/$MODEL"
CTX=$(python3 -c "import json;print(json.load(open('$WORK/model.json')).get('context_length','MISSING'))")
CAP=$(python3 -c "import json;print(json.load(open('$WORK/model.json')).get('max_prompt_tokens','MISSING'))")
if [ "$CTX" != "MISSING" ] && [ "$CAP" != "MISSING" ]; then
  echo "  ok: context_length=$CTX max_prompt_tokens=$CAP"
  # Report the gap the operator has to decide on (ADR 042 closing consequence).
  python3 -c "
c,m=$CTX,$CAP
print('  note: effective budget = min(%d, %d) = %d tokens (%.0fx below the advertised window)' % (c,m,min(c,m), (c/m if m else 0)))"
else
  echo "  FAIL context_length=$CTX max_prompt_tokens=$CAP (body: $(cat "$WORK/model.json"))"; fail=1
fi

# One fixed body, reused verbatim for both calls: system + multi-turn + a tool.
BODY=$(python3 - "$MODEL" <<'PY'
import json, sys
print(json.dumps({
  "model": sys.argv[1],
  "messages": [
    {"role": "system", "content": "You are a terse assistant. Answer in one short sentence."},
    {"role": "user", "content": "What is the weather in Chicago right now?"},
    {"role": "assistant", "content": "I would need to look that up."},
    {"role": "user", "content": "Then look it up and tell me whether I need a coat."},
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "searchWeb",
      "description": "Search the public web for a short factual answer.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "The search query."},
          "max_results": {"type": "integer", "description": "How many hits to return."},
        },
        "required": ["query"],
      },
    },
  }],
  "max_tokens": 16,
  "temperature": 0,
}))
PY
)

echo "== 2) count_tokens == usage.prompt_tokens for the identical body (with a tool) =="
CH=$(curl -s -o "$WORK/count.json" -w "%{http_code}" -X POST \
  "$BASE/v1/chat/completions/count_tokens" \
  -H 'Content-Type: application/json' -d "$BODY")
COUNT=$(python3 -c "import json;print(json.load(open('$WORK/count.json')).get('prompt_tokens','MISSING'))" 2>/dev/null)
RH=$(curl -s -o "$WORK/chat.json" -w "%{http_code}" -X POST \
  "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY")
USED=$(python3 -c "import json;print(json.load(open('$WORK/chat.json'))['usage']['prompt_tokens'])" 2>/dev/null)
if [ "$CH" = "200" ] && [ "$RH" = "200" ] && [ -n "$COUNT" ] && [ "$COUNT" = "$USED" ]; then
  echo "  ok: count=$COUNT == usage.prompt_tokens=$USED (exact)"
else
  echo "  FAIL count_http=$CH chat_http=$RH count=$COUNT usage=$USED"
  echo "       count body: $(cat "$WORK/count.json")"; fail=1
fi

echo "== 2b) image content parts are refused, not under-counted =="
IMGBODY="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what is this\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==\"}}]}]}"
IH=$(curl -s -o "$WORK/img.json" -w "%{http_code}" -X POST \
  "$BASE/v1/chat/completions/count_tokens" \
  -H 'Content-Type: application/json' -d "$IMGBODY")
ICODE=$(python3 -c "import json;print(json.load(open('$WORK/img.json'))['error'].get('code',''))" 2>/dev/null)
if [ "$IH" = "400" ] && [ "$ICODE" = "image_count_unsupported" ]; then
  echo "  ok: 400 image_count_unsupported"
else
  echo "  FAIL http=$IH code=$ICODE body=$(cat "$WORK/img.json")"; fail=1
fi

echo "== 3a) count is cheap on an idle engine =="
IDLE=$(curl -s -o "$WORK/count_idle.json" -w "%{time_total}" -X POST \
  "$BASE/v1/chat/completions/count_tokens" \
  -H 'Content-Type: application/json' -d "$BODY")
if python3 -c "exit(0 if $IDLE < 1.0 else 1)"; then
  echo "  ok: ${IDLE}s idle (no prefill, no eval)"
else
  echo "  FAIL idle count took ${IDLE}s — expected well under 1s"; fail=1
fi

echo "== 3b) count still answers correctly with a long decode in flight =="
# Start a long generation in the background, wait for it to actually be
# decoding, then time a count. Latency is REPORTED, not asserted (see header):
# the substrate's container mutex makes the count wait out the decode.
curl -s -o "$WORK/long.json" -X POST "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed 900-word essay about the history of maritime navigation.\"}],\"max_tokens\":900,\"temperature\":0}" &
LPID=$!
# Wait until the gate is actually held (the decode has started), max ~20s.
for i in $(seq 1 40); do
  held=$(curl -s "$BASE/healthz" | python3 -c "import json,sys;print(json.load(sys.stdin).get('gateHeld',False))" 2>/dev/null)
  [ "$held" = "True" ] && break
  sleep 0.5
done
T0=$(python3 -c 'import time;print(time.time())')
CH2=$(curl -s -o "$WORK/count2.json" -w "%{http_code}" -X POST \
  "$BASE/v1/chat/completions/count_tokens" \
  -H 'Content-Type: application/json' -d "$BODY")
ELAPSED=$(python3 -c "import time;print('%.2f' % (time.time()-$T0))")
COUNT2=$(python3 -c "import json;print(json.load(open('$WORK/count2.json')).get('prompt_tokens','MISSING'))" 2>/dev/null)
if [ "$CH2" = "200" ] && [ "$COUNT2" = "$COUNT" ]; then
  echo "  ok: same count=$COUNT2 under contention (gateHeld=$held at issue time)"
  echo "  measured: ${ELAPSED}s mid-decode vs ${IDLE}s idle — the substrate's"
  echo "            ModelContainer mutex, held for the whole decode, is what"
  echo "            this waits on (ADR 042 §4(b) amendment)"
else
  echo "  FAIL http=$CH2 count=$COUNT2 elapsed=${ELAPSED}s gateHeld=$held"; fail=1
fi
wait $LPID 2>/dev/null

echo
if [ "$fail" = "0" ]; then echo "ALL CHECKS PASSED"; else echo "FAILURES (see above)"; fi
exit "$fail"
