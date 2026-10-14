#!/bin/bash
#
# e2e-embedding-parity.sh — cosine parity of /v1/embeddings against a
# reference vector set produced by another implementation of the same model.
#
# The reference is a JSON file whose `queries` array holds entries with
# `prefixed_input` (the exact string to embed, task prefix included) and
# `vector` (the reference embedding). Each prefixed_input is POSTed to a
# running daemon; the script prints one cosine per query and exits non-zero if
# any is below the bar.
#
# Requires: a running daemon with the model in its store.
# Usage: ./deploy/e2e-embedding-parity.sh <reference.json> [model-id]
# Env:   ATHENA_URL (default http://127.0.0.1:7447), ATHENA_TOKEN (bearer, if
#        auth is on), PARITY_MIN (default 0.99).
set -euo pipefail

ref="${1:?usage: $0 <reference.json> [model-id]}"
model="${2:-embeddinggemma-2}"
url="${ATHENA_URL:-http://127.0.0.1:7447}"
min="${PARITY_MIN:-0.99}"

exec python3 -I - "$ref" "$model" "$url" "$min" "${ATHENA_TOKEN:-}" <<'PY'
import json, math, sys, urllib.error, urllib.request

ref_path, model, url, bar, token = sys.argv[1:6]
bar = float(bar)
queries = json.load(open(ref_path))["queries"]
headers = {"Content-Type": "application/json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

def embed(text):
    body = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(f"{url}/v1/embeddings", body, headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["data"][0]["embedding"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} for {text!r}: {e.read().decode(errors='replace')}")

def cosine(a, b):
    if len(a) != len(b):
        raise SystemExit(f"dimension mismatch: got {len(a)}, reference {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

worst = 1.0
for q in queries:
    c = cosine(embed(q["prefixed_input"]), q["vector"])
    worst = min(worst, c)
    mark = "ok  " if c >= bar else "FAIL"
    print(f"{mark} cos={c:.6f}  {q['prefixed_input']}")
print(f"{len(queries)} queries, min cosine {worst:.6f}, bar {bar}")
sys.exit(0 if worst >= bar else 1)
PY
