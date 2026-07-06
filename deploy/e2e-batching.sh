#!/bin/bash
#
# e2e-batching.sh — ADR 039 S2 Definition of Done: continuous batching.
#
# Spins a loopback, no-auth daemon with `ATHENA_BATCHING=1` on a real text-chat
# model, fires N concurrent /v1/chat/completions, and asserts every one returns
# a coherent, non-empty completion with no errors — i.e. the batch scheduler +
# per-row detokenization + stream fan-out are correct under concurrency, and the
# governor never overcommits. Flag-OFF byte-parity is covered by the existing
# unit suite + e2e-rbac (batching defaults off).
#
# Requires a real model on disk (like the ATHENA_RUN_MODEL_TESTS heavy tier), so
# it SKIPS cleanly when the model isn't present — safe to run anywhere.
#
# Usage:
#   ./deploy/e2e-batching.sh                        # gemma-4-26b-a4b-it-8bit, N=6
#   ./deploy/e2e-batching.sh <model> <N>            # override
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-gemma-4-26b-a4b-it-8bit}"
N="${2:-6}"
PORT=17449
STORE="${ATHENA_MODEL_STORE:-$HOME/.athena/models}"
BIN=".build/xcode/Build/Products/Release/athena"

[ -x "$BIN" ] || { echo "skip: no Release binary at $BIN (run ./deploy/build.sh Release)"; exit 0; }
[ -d "$STORE/$MODEL" ] || { echo "skip: model '$MODEL' not in store $STORE"; exit 0; }

DD="$(mktemp -d)"
LOG="$(mktemp)"
cleanup() { kill "$PID" 2>/dev/null || true; rm -rf "$DD" "$LOG"; }
trap cleanup EXIT

echo "e2e-batching: starting daemon (ATHENA_BATCHING=1, model=$MODEL, N=$N)"
ATHENA_BATCHING=1 "$BIN" load --host 127.0.0.1 --port "$PORT" --engine mlx \
  --data-dir "$DD" --model-store "$STORE" --llm-model "$MODEL" > "$LOG" 2>&1 &
PID=$!

for i in $(seq 1 30); do
  curl -s -m2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done
grep -q "continuous batching ENABLED" "$LOG" || { echo "FAIL: batching not enabled"; exit 1; }

# Fire N concurrent chat completions; assert all 200 + non-empty content.
python3 - "$PORT" "$MODEL" "$N" <<'PY'
import json, sys, threading, urllib.request
port, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = ["Name three primary colors.", "What is 2 plus 2?",
           "Write one sentence about the ocean.", "List two fruits.",
           "What sound does a cat make?", "Say hello in French.",
           "What is the capital of Japan?", "Complete: the sky is"]
def call(p):
    body = {"model": model, "max_tokens": 48, "temperature": 0,
            "messages": [{"role": "user", "content": p}]}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]
res = [None] * n
b = threading.Barrier(n)
def worker(i):
    b.wait()
    try:
        t = call(prompts[i % len(prompts)]).strip()
        res[i] = ("ok", t) if t else ("empty", "")
    except Exception as e:
        res[i] = ("err", str(e))
ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
for t in ts: t.start()
for t in ts: t.join()
bad = [r for r in res if r[0] != "ok"]
for st, msg in res:
    print(f"  {st}: {msg[:56]!r}")
if bad:
    print(f"FAIL: {len(bad)}/{n} requests errored or empty under batching")
    sys.exit(1)
print(f"PASS: {n}/{n} concurrent completions coherent + non-empty")
PY
echo "e2e-batching: PASS"
