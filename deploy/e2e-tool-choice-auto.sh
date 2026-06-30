#!/bin/bash
#
# e2e-tool-choice-auto.sh — end-to-end DoD for ADR 034 (tool_choice:auto must
# not force a tool call). Runs a dev-mode daemon against the real model store
# and asserts, with one `searchWeb` tool declared:
#
#   1) auto + greeting        → TEXT answer (finish_reason:"stop", content set,
#                               NO tool_calls). Before the fix this was a forced
#                               tool_call — the root of the spiral.
#   2) multi-turn (the spiral guard): [user, assistant(tool_call), tool(result)]
#      with auto                → TEXT answer, NOT another tool_call. This is the
#                               discriminating check: pre-fix the model was
#                               grammar-forced to re-call forever.
#   3) required + tools        → STILL a tool_call (forced path regression guard).
#   4) plain chat, no tools    → TEXT answer (untouched).
#
# Fails before the change, passes after. MLX numerics can't run under
# `swift test` (ADR 009), so this is the heavy DoD — run it on the studio (or
# any host with the model resident).
#
# Requires: a built Release binary + an LLM model in the store.
# Usage: ./deploy/e2e-tool-choice-auto.sh [binary] [model-id]
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
MODEL="${2:-${ATHENA_LLM_MODEL:-gemma-4-26b-a4b-it-8bit}}"
PORT=7790
STORE="$HOME/.athena/models"
WORK="$(mktemp -d)"
DATA="$(mktemp -d)"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$WORK" "$DATA"' EXIT

[ -x "$BIN" ] || { echo "error: no binary at $BIN (build it first)"; exit 1; }
[ -d "$STORE/$MODEL" ] || { echo "SKIP: model '$MODEL' not in $STORE"; exit 0; }

TOOL='{"type":"function","function":{"name":"searchWeb","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}'

echo "== starting dev-mode daemon on :$PORT (model=$MODEL) =="
"$BIN" load --port "$PORT" --data-dir "$DATA" --model-store "$STORE" \
  --llm-model "$MODEL" >"$WORK/daemon.log" 2>&1 &
DPID=$!
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done

URL="http://127.0.0.1:$PORT/v1/chat/completions"
fail=0
# Extract a field from choices[0] of the saved response.
fr() { python3 -c "import json;print(json.load(open('$WORK/out'))['choices'][0].get('finish_reason',''))" 2>/dev/null; }
content() { python3 -c "import json;c=json.load(open('$WORK/out'))['choices'][0]['message'].get('content') or '';print(c.strip())" 2>/dev/null; }
has_tc() { python3 -c "import json;m=json.load(open('$WORK/out'))['choices'][0]['message'];print('yes' if m.get('tool_calls') else 'no')" 2>/dev/null; }
post() { curl -s -o "$WORK/out" -w "%{http_code}" -X POST "$URL" -H 'Content-Type: application/json' -d "$1"; }

echo "== 1) auto + greeting → text answer, no tool_call =="
h=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello there, how are you?\"}],\"max_tokens\":128,\"tools\":[$TOOL]}")
if [ "$h" = "200" ] && [ "$(has_tc)" = "no" ] && [ -n "$(content)" ] && [ "$(fr)" = "stop" ]; then
  echo "  ok: text answer, finish=stop, no tool_call"
else
  echo "  FAIL http=$h finish=$(fr) tool_calls=$(has_tc) content='$(content)'"; fail=1
fi

echo "== 2) multi-turn after a tool result → text answer, NOT another call (spiral guard) =="
MT="{\"model\":\"$MODEL\",\"max_tokens\":128,\"tools\":[$TOOL],\"messages\":[\
{\"role\":\"user\",\"content\":\"What is the weather?\"},\
{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"searchWeb\",\"arguments\":\"{\\\"query\\\":\\\"weather\\\"}\"}}]},\
{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"It is 72F and sunny.\"}]}"
h=$(post "$MT")
# ADR 035 — content must not leak channel-reasoning markers. Trivially true for
# non-channel models; the real gate on gemma-4-class channel-format models.
leak=$(content | grep -c -E '<\|channel>|<channel\|>' || true)
if [ "$h" = "200" ] && [ "$(has_tc)" = "no" ] && [ -n "$(content)" ] && [ "$leak" = "0" ]; then
  echo "  ok: answered in text after the tool result (no re-call, no channel leak)"
else
  echo "  FAIL http=$h finish=$(fr) tool_calls=$(has_tc) leak=$leak content='$(content)'"; fail=1
fi

echo "== 3) required + tools → forced tool_call (regression guard) =="
# 256-token cap: under the forcing Guide a small/quantized model can emit a
# verbose object; too low a cap truncates it (parse fails → no tool_call).
h=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"weather in paris\"}],\"max_tokens\":256,\"tools\":[$TOOL],\"tool_choice\":\"required\"}")
if [ "$h" = "200" ] && [ "$(has_tc)" = "yes" ] && [ "$(fr)" = "tool_calls" ]; then
  echo "  ok: forced a tool_call"
else
  echo "  FAIL http=$h finish=$(fr) tool_calls=$(has_tc)"; fail=1
fi

echo "== 4) plain chat, no tools → text answer (untouched) =="
h=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}],\"max_tokens\":32}")
if [ "$h" = "200" ] && [ "$(has_tc)" = "no" ] && [ -n "$(content)" ]; then
  echo "  ok: text answer"
else
  echo "  FAIL http=$h finish=$(fr) content='$(content)'"; fail=1
fi

if [ "$fail" = "0" ]; then echo "PASS"; else echo "FAIL ($fail check(s))"; fi
exit $fail
