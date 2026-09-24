#!/bin/bash
#
# e2e-disconnect-cancel.sh — #209: a client that disconnects mid-request stops
# its decode and releases the ADR 029 inference gate within a bound.
#
# For each of POST /v1/chat/completions and POST /v1/messages, non-streaming
# and streaming: a raw-socket client starts a long generation, waits until the
# gate is held, closes the socket, then polls /healthz. PASS when `gateHeld`
# goes false within $BOUND seconds of the close.
#
# Fails before the #209 fix on both non-streaming cases (the decode runs to
# completion, holding the gate for tens of seconds); streaming passed before
# and must keep passing. Needs real MLX (ADR 009), so it is not in `swift test`.
#
# Requires: a built Release binary + a small LLM in the store.
# Usage: ./deploy/e2e-disconnect-cancel.sh [binary] [model-id]
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
MODEL="${2:-${ATHENA_LLM_MODEL:-Llama-3.2-3B-Instruct-8bit}}"
PORT="${ATHENA_E2E_PORT:-7450}"
BOUND="${ATHENA_E2E_CANCEL_BOUND:-5}"
STORE="${ATHENA_MODEL_STORE:-$HOME/.athena/models}"
WORK="$(mktemp -d)"
DATA="$(mktemp -d)"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$WORK" "$DATA"' EXIT

[ -x "$BIN" ] || { echo "error: no binary at $BIN (build it first)"; exit 1; }
[ -d "$STORE/$MODEL" ] || { echo "SKIP: model '$MODEL' not in $STORE"; exit 0; }

echo "== starting dev-mode daemon on :$PORT (model=$MODEL) =="
# A missing ATHENA_CONFIG path keeps an installed /usr/local/etc config out.
ATHENA_CONFIG="$WORK/none.toml" "$BIN" load --port "$PORT" --data-dir "$DATA" \
  --model-store "$STORE" --model "$MODEL" --cold-load-wait-secs 600 \
  >"$WORK/daemon.log" 2>&1 &
DPID=$!
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done

echo "== warming $MODEL =="
curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"

fail=0
for path in /v1/chat/completions /v1/messages; do
  for stream in 0 1; do
    out="$(python3 - "$PORT" "$path" "$MODEL" "$stream" "$BOUND" <<'PY'
import json, socket, sys, time, urllib.request
port, path, model, stream, bound = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", float(sys.argv[5])
body = json.dumps({"model": model, "max_tokens": 4000, "stream": stream, "messages": [{"role": "user",
    "content": "Write an extremely long, detailed story of at least 6000 words about a lighthouse keeper and her family across three generations. Do not stop early."}]}).encode()
def gate():
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
        return json.load(r).get("gateHeld")
s = socket.create_connection(("127.0.0.1", int(port)))
s.sendall((f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
s.settimeout(0.1)
deadline = time.time() + 60
while not gate():
    if time.time() > deadline:
        print("NOGATE"); sys.exit()
    try: s.recv(65536)
    except socket.timeout: pass
time.sleep(1.0)
s.close()
t = time.time()
while time.time() - t < bound:
    if not gate():
        print(f"RELEASED {time.time() - t:.2f}"); sys.exit()
    time.sleep(0.1)
print(f"HELD {bound}")
PY
)"
    case "$out" in
      RELEASED*) echo "PASS $path stream=$stream: gate released ${out#RELEASED }s after close" ;;
      *) echo "FAIL $path stream=$stream: $out (bound ${BOUND}s)"; fail=1 ;;
    esac
    # Let a decode that did not cancel finish before the next case.
    for _ in $(seq 1 120); do
      curl -s "http://127.0.0.1:$PORT/healthz" | grep -q '"gateHeld":false' && break
      sleep 1
    done
  done
done

[ "$fail" = 0 ] && echo "ALL PASS" || { echo "FAILED — daemon log: $WORK/daemon.log"; tail -20 "$WORK/daemon.log"; }
exit "$fail"
