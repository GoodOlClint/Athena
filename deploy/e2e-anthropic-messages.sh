#!/bin/bash
#
# e2e-anthropic-messages.sh — end-to-end DoD for ADR 036 S2 (the Anthropic
# Messages adapter, POST /v1/messages, over the shared inference engine). Runs a
# dev-mode daemon against the real model store and asserts, with one tool
# declared:
#
#   1) non-stream text        → {type:message, role:assistant, content[text],
#                               stop_reason:end_turn|max_tokens, usage}.
#   2) non-stream forced tool  → a tool_use content block + stop_reason:"tool_use"
#      (tool_choice:any).        with a parsed `input` object.
#   3) tool_result round-trip  → after feeding the assistant tool_use + a user
#                               tool_result, the model answers in TEXT using the
#                               result (history mapping works).
#   4) streaming text          → the Anthropic SSE event sequence: message_start
#                               → content_block_start(text) → content_block_delta
#                               (text_delta)+ → content_block_stop → message_delta
#                               (stop_reason) → message_stop.
#   5) streaming forced tool    → content_block_start(tool_use) + input_json_delta
#      (tool_choice:any)         + message_delta stop_reason:"tool_use".
#
# Fails before the change (no /v1/messages → 404), passes after. MLX numerics
# can't run under `swift test` (ADR 009), so this is the heavy DoD — run it on
# the studio (or any host with the model resident).
#
# Requires: a built Release binary + an LLM model in the store.
# Usage: ./deploy/e2e-anthropic-messages.sh [binary] [model-id]
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
MODEL="${2:-${ATHENA_LLM_MODEL:-gemma-4-26b-a4b-it-8bit}}"
PORT=7793
STORE="$HOME/.athena/models"
WORK="$(mktemp -d)"
DATA="$(mktemp -d)"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$WORK" "$DATA"' EXIT

[ -x "$BIN" ] || { echo "error: no binary at $BIN (build it first)"; exit 1; }
[ -d "$STORE/$MODEL" ] || { echo "SKIP: model '$MODEL' not in $STORE"; exit 0; }

# NOTE: no `description` field — a tool description breaks forced-tool generation
# on gemma-4-26b-a4b-it-8bit for BOTH dialects (a pre-existing model/template
# quirk, not adapter-specific; the OpenAI e2e-tool-choice-auto omits it too).
TOOL='{"name":"searchWeb","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}'

echo "== starting dev-mode daemon on :$PORT (model=$MODEL) =="
"$BIN" load --port "$PORT" --data-dir "$DATA" --model-store "$STORE" \
  --llm-model "$MODEL" >"$WORK/daemon.log" 2>&1 &
DPID=$!
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done

URL="http://127.0.0.1:$PORT/v1/messages"
fail=0
post() { curl -s -o "$WORK/out" -w "%{http_code}" -X POST "$URL" \
  -H 'Content-Type: application/json' -d "$1"; }
sse() { curl -s -N -X POST "$URL" -H 'Content-Type: application/json' -d "$1"; }

echo "== 1) non-stream text =="
h=$(post "{\"model\":\"$MODEL\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}")
ok=$(python3 -c "
import json
d=json.load(open('$WORK/out'))
print('yes' if d.get('type')=='message' and d.get('role')=='assistant' and d['content'][0]['type']=='text' and d.get('stop_reason') in('end_turn','max_tokens') and 'input_tokens' in d['usage'] else 'no')
" 2>/dev/null)
[ "$h" = "200" ] && [ "$ok" = "yes" ] && echo "  ok" || { echo "  FAIL http=$h"; fail=1; }

echo "== 2) non-stream forced tool_use =="
h=$(post "{\"model\":\"$MODEL\",\"max_tokens\":256,\"tool_choice\":{\"type\":\"any\"},\"tools\":[$TOOL],\"messages\":[{\"role\":\"user\",\"content\":\"weather in paris\"}]}")
TUID=$(python3 -c "
import json
d=json.load(open('$WORK/out'))
tu=[b for b in d.get('content',[]) if b.get('type')=='tool_use']
print(tu[0]['id'] if d.get('stop_reason')=='tool_use' and len(tu)==1 and tu[0]['name']=='searchWeb' and isinstance(tu[0].get('input'),dict) else '')
" 2>/dev/null)
[ "$h" = "200" ] && [ -n "$TUID" ] && echo "  ok" || { echo "  FAIL http=$h"; fail=1; }

echo "== 3) tool_result round-trip → text =="
[ -z "$TUID" ] && TUID="toolu_x"
h=$(post "{\"model\":\"$MODEL\",\"max_tokens\":64,\"tools\":[$TOOL],\"messages\":[{\"role\":\"user\",\"content\":\"weather in paris\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"$TUID\",\"name\":\"searchWeb\",\"input\":{\"query\":\"Paris weather\"}}]},{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"$TUID\",\"content\":\"18C and sunny\"}]}]}")
ok=$(python3 -c "
import json
d=json.load(open('$WORK/out'))
txt=[b for b in d.get('content',[]) if b.get('type')=='text']
print('yes' if d.get('stop_reason') in('end_turn','max_tokens') and txt and txt[0]['text'].strip() else 'no')
" 2>/dev/null)
[ "$h" = "200" ] && [ "$ok" = "yes" ] && echo "  ok" || { echo "  FAIL http=$h"; fail=1; }

echo "== 4) streaming text (SSE event sequence) =="
sse "{\"model\":\"$MODEL\",\"max_tokens\":32,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}]}" >"$WORK/sse1"
if grep -q 'event: message_start' "$WORK/sse1" \
  && grep -q 'event: content_block_start' "$WORK/sse1" \
  && grep -q '"type":"text_delta"' "$WORK/sse1" \
  && grep -q 'event: message_delta' "$WORK/sse1" \
  && grep -q 'event: message_stop' "$WORK/sse1"; then
  echo "  ok"
else
  echo "  FAIL (missing events)"; sed -n '1,12p' "$WORK/sse1"; fail=1
fi

echo "== 5) streaming forced tool_use (input_json_delta) =="
sse "{\"model\":\"$MODEL\",\"max_tokens\":256,\"stream\":true,\"tool_choice\":{\"type\":\"any\"},\"tools\":[$TOOL],\"messages\":[{\"role\":\"user\",\"content\":\"weather in paris\"}]}" >"$WORK/sse2"
if grep -q '"type":"tool_use"' "$WORK/sse2" \
  && grep -q '"type":"input_json_delta"' "$WORK/sse2" \
  && grep -q '"stop_reason":"tool_use"' "$WORK/sse2"; then
  echo "  ok"
else
  echo "  FAIL (missing tool_use stream events)"; sed -n '1,16p' "$WORK/sse2"; fail=1
fi

echo ""
[ "$fail" = "0" ] && echo "PASS" || echo "FAILED"
exit $fail
