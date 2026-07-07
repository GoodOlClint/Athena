#!/bin/bash
#
# soak-throughput-decay.sh — reproduce + diagnose the M60 sustained-load
# throughput decay (a downstream client's 540s timeouts). Self-measuring against the
# M60.1 /healthz fields (thermalState, lastDecodeTokensPerSec, mlxCacheBytes,
# physFootprintBytes) so NO sudo/powermetrics is required.
#
# What it does:
#   1. Starts the freshly-built binary in the foreground (`load`) with the
#      real model, prompt_cache + speculative as configured, on a test port.
#   2. Fires N back-to-back large structured chat completions (prefill-heavy
#      + long decode — the downstream-client shape). After each call it records
#      wall time, HTTP status, completion tokens, tok/s, and a /healthz sample.
#   3. Flags the first call whose wall time crosses DEGRADE_SECS.
#
# Discriminator (run after a soak degrades — these are the decisive legs):
#   MODE=cooldown : idle COOLDOWN_SECS with NO restart, then 1 call. Faster ⇒ THERMAL.
#   MODE=restart  : stop+restart the daemon (no cooldown gap), then 1 call. Faster only here ⇒ STATEFUL (M59/MLX pool).
# Controls:
#   PROMPT_CACHE=0  : rerun with the prefix cache off (isolates M59).
#   SPECULATIVE=0   : rerun greedy (isolates the MTP path; greedy uses the faster gemv kernel).
#
# Usage:
#   ./deploy/integration/soak-throughput-decay.sh            # full soak
#   MODE=cooldown ./deploy/integration/soak-throughput-decay.sh
#   PROMPT_CACHE=0 ./deploy/integration/soak-throughput-decay.sh
#
# Env (defaults in []):
#   MODEL [Qwen3.5-27B-4bit-mtp]  PORT [7458]  CALLS [12]
#   MAX_TOKENS [4096]  PROMPT_TOKENS [4000]  DOC_TOKENS [6000]
#   DEGRADE_SECS [300]  COOLDOWN_SECS [600]  PROMPT_CACHE [1]  SPECULATIVE [1]
#   BIN [.build/xcode/Build/Products/Release/athena]
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-.build/xcode/Build/Products/Release/athena}"
MODEL="${MODEL:-Qwen3.5-27B-4bit-mtp}"
PORT="${PORT:-7458}"
CALLS="${CALLS:-12}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4000}"
DOC_TOKENS="${DOC_TOKENS:-6000}"
DEGRADE_SECS="${DEGRADE_SECS:-300}"
COOLDOWN_SECS="${COOLDOWN_SECS:-600}"
PROMPT_CACHE="${PROMPT_CACHE:-1}"
SPECULATIVE="${SPECULATIVE:-1}"
MODE="${MODE:-soak}"
UNIQUE_DOCS="${UNIQUE_DOCS:-0}"
# Throughput-cliff flag: decode tok/s below this = a real slowdown (distinct
# from a long-output call, which inflates wall time at a healthy rate).
DEGRADE_TPS="${DEGRADE_TPS:-20}"

HOST=127.0.0.1
BASE="http://$HOST:$PORT"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/athena-soak-$STAMP"
mkdir -p "$OUT"
SRVLOG="$OUT/daemon.log"
CSV="$OUT/calls.csv"
echo "call,t_start,wall_s,http,completion_tokens,tok_s,thermal,healthz_decode_tps,mlx_cache_bytes,phys_footprint_bytes,prompt_cache_pool_bytes" > "$CSV"

echo "soak run -> $OUT"
echo "model=$MODEL calls=$CALLS max_tokens=$MAX_TOKENS prompt~${PROMPT_TOKENS}tok doc~${DOC_TOKENS}tok prompt_cache=$PROMPT_CACHE speculative=$SPECULATIVE unique_docs=$UNIQUE_DOCS mode=$MODE"

# --- build a big prompt (system + verbatim 'document') ----------------------
# ~4 chars/token; a long filler doc reproduces the prefill-heavy shape.
# nonce="" → bit-identical body every call (prefix cache HITS → prefill ~0,
# isolates DECODE). nonce="<n>" (UNIQUE_DOCS=1) → a per-call nonce at the FRONT
# of the system prompt + a unique prompt_cache_key, so common-prefix < 512 ⇒
# every call is a cold MISS that pays a full prefill (isolates PREFILL — the
# compute-bound path that throttling hits hardest, matching a downstream client's
# single-pass uncached documents).
gen_body () {
  local nonce="${1:-}"
  NONCE="$nonce" python3 - "$PROMPT_TOKENS" "$DOC_TOKENS" "$MAX_TOKENS" <<'PY'
import json, os, sys
ptok, dtok, maxtok = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
nonce = os.environ.get("NONCE", "")
lead = f"[case {nonce}] " if nonce else ""
sysprompt = lead + ("You are a meticulous legal-document event extractor. " * (ptok//8))[:ptok*4]
doc = ("On the fourteenth day the party of the first part did convey unto the "
       "party of the second part the following chattels and tenements. " * (dtok//16))[:dtok*4]
body = {
  "model": "MODEL_PLACEHOLDER",
  "messages": [
    {"role":"system","content": sysprompt},
    {"role":"user","content": "Extract every dated event as JSON.\n\nDOCUMENT:\n"+doc}
  ],
  "temperature": 0.1,
  "max_completion_tokens": maxtok,
  "prompt_cache_key": ("soak-doc-"+nonce) if nonce else "soak-doc-fixed",
  "response_format": {"type":"json_schema","json_schema":{"name":"events","schema":{
    "type":"object","properties":{"events":{"type":"array","items":{"type":"object",
    "properties":{"date":{"type":"string"},"summary":{"type":"string"}},
    "required":["date","summary"]}}},"required":["events"]}}}
}
print(json.dumps(body).replace("MODEL_PLACEHOLDER", "__MODEL__"))
PY
}
# Pre-build the fixed body (used when UNIQUE_DOCS != 1).
BODY="$(gen_body)"
BODY="${BODY/__MODEL__/$MODEL}"
echo "$BODY" > "$OUT/request.json"

# --- daemon lifecycle -------------------------------------------------------
DATADIR="$OUT/data"
start_daemon () {
  local spec_flag=()
  [ "$SPECULATIVE" = "1" ] && spec_flag=(--speculative)
  ATHENA_PROMPT_CACHE="$PROMPT_CACHE" \
    "$BIN" load --host "$HOST" --port "$PORT" --data-dir "$DATADIR" \
    --engine mlx --model "$MODEL" "${spec_flag[@]}" >>"$SRVLOG" 2>&1 &
  DPID=$!
  echo "daemon pid=$DPID — waiting for server to bind (watch $SRVLOG)..."
  # Athena loads the model LAZILY on the first request, so we wait only for
  # the server to bind; the cold model load is paid inside call #1 (its wall
  # time therefore includes ~10-20s of load — note it when reading results).
  local bound=0
  for i in $(seq 1 120); do
    if curl -s --max-time 2 "$BASE/healthz" >/dev/null 2>&1; then bound=1; break; fi
    sleep 1
  done
  [ "$bound" = 1 ] || { echo "WARN: server did not bind in 120s."; return 0; }
  echo "server up — warming llm module (POST /api/models/load, blocks until resident)..."
  local wc
  wc=$(curl -s -o "$OUT/warm.json" -w '%{http_code}' --max-time 600 \
       -X POST "$BASE/api/models/load" -H 'Content-Type: application/json' \
       -d '{"module":"llm"}' || echo 000)
  local st
  st=$(curl -s --max-time 3 "$BASE/healthz" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(next((m["state"] for m in d["modules"] if m["id"]=="llm"),"?"))' 2>/dev/null || echo "?")
  echo "warm http=$wc  llm_state=$st"
}
stop_daemon () { kill "${DPID:-0}" 2>/dev/null || true; wait "${DPID:-0}" 2>/dev/null || true; }

# --- GPU clock sampler (powermetrics, sudo) --------------------------------
# Logs GPU HW active frequency every 2s, epoch-timestamped, so MHz can be
# correlated against calls.csv (t_start) and the /healthz tok/s. Uses `sudo -n`
# (non-interactive): if creds aren't cached it skips gracefully — run
# `sudo -v` in your terminal just before launching to enable it. Die temp is
# NOT sampled here (no `smc` sampler on this macOS) — read it from mactop.
THERMAL_CSV="$OUT/gpu_clock.csv"
start_clock_sampler () {
  command -v powermetrics >/dev/null 2>&1 || { echo "clock: powermetrics absent — skipping MHz log"; return 0; }
  if ! sudo -n true 2>/dev/null; then
    echo "clock: no cached sudo — MHz log skipped (run 'sudo -v' first to enable; read mactop meanwhile)"
    return 0
  fi
  echo "epoch,gpu_mhz" > "$THERMAL_CSV"
  ( sudo -n powermetrics --samplers gpu_power -i 2000 2>/dev/null \
      | while IFS= read -r line; do
          case "$line" in
            *"GPU HW active frequency"*)
              mhz=$(printf '%s' "$line" | grep -oE '[0-9]+' | head -1)
              printf '%s,%s\n' "$(date +%s)" "$mhz" >> "$THERMAL_CSV" ;;
          esac
        done ) &
  CLKPID=$!
  echo "clock: GPU MHz sampler pid=$CLKPID -> $THERMAL_CSV"
}
stop_clock_sampler () {
  kill "${CLKPID:-0}" 2>/dev/null || true
  sudo -n pkill -f "powermetrics --samplers gpu_power" 2>/dev/null || true
}
trap 'stop_clock_sampler; stop_daemon' EXIT

# --- one call + /healthz sample --------------------------------------------
fire_call () {
  local n="$1"
  local t0 t1 wall http ctoks toks_s hz body
  # UNIQUE_DOCS=1 ⇒ a fresh per-call body that misses the prefix cache (cold
  # full prefill, the downstream-client single-pass shape); else the fixed body.
  if [ "$UNIQUE_DOCS" = "1" ]; then
    body="$(gen_body "$n")"; body="${body/__MODEL__/$MODEL}"
  else
    body="$BODY"
  fi
  t0=$(python3 -c 'import time;print(time.time())')
  http=$(curl -s -o "$OUT/resp-$n.json" -w '%{http_code}' --max-time 600 \
        -H 'Content-Type: application/json' \
        -X POST "$BASE/v1/chat/completions" -d "$body" || echo 000)
  t1=$(python3 -c 'import time;print(time.time())')
  wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")
  ctoks=$(python3 -c 'import json,sys;
try:
 d=json.load(open(sys.argv[1]));print(d.get("usage",{}).get("completion_tokens",0))
except Exception:print(0)' "$OUT/resp-$n.json" 2>/dev/null || echo 0)
  toks_s=$(python3 -c "print(f'{($ctoks/max($wall,0.001)):.1f}')")
  hz=$(curl -s --max-time 3 "$BASE/healthz" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(f'"'"'{d.get("thermalState","?")},{d.get("lastDecodeTokensPerSec",0):.1f},{d.get("mlxCacheBytes",0)},{d.get("physFootprintBytes",0)},{d.get("promptCachePoolBytes",0)}'"'"')' 2>/dev/null || echo '?,0,0,0,0')
  echo "$n,$t0,$wall,$http,$ctoks,$toks_s,$hz" >> "$CSV"
  printf 'call %2s  wall=%6ss  http=%s  ctoks=%-5s  tok/s=%-5s  thermal/mlxcache/phys/pool=%s\n' \
    "$n" "$wall" "$http" "$ctoks" "$toks_s" "$hz"
  # degradation flag — key on the DECODE RATE (healthz lastDecodeTokensPerSec),
  # NOT wall time. Wall conflates a long output with slow throughput: a call
  # that emits 4000 tokens at a healthy 36 tok/s takes ~130s yet hasn't
  # degraded at all. A real throughput cliff shows as decode tps falling.
  hz_tps=$(printf '%s' "$hz" | cut -d, -f2)
  if python3 -c "import sys;sys.exit(0 if float('$hz_tps')>0 and float('$hz_tps')<$DEGRADE_TPS else 1)"; then
    echo "  >> THROUGHPUT DEGRADED (decode ${hz_tps} tok/s < ${DEGRADE_TPS})"
  fi
  # Also note unusually long walls (output length OR slowdown — disambiguate
  # via the decode-tps column).
  if python3 -c "import sys;sys.exit(0 if $wall>=$DEGRADE_SECS else 1)"; then
    echo "  >> long wall ${wall}s (check decode tps column to tell output-length from slowdown)"
  fi
  if [ "$http" != "200" ]; then echo "  >> NON-200 ($http) — see $OUT/resp-$n.json"; fi
}

# --- modes ------------------------------------------------------------------
case "$MODE" in
  cooldown)
    start_daemon
    echo "discriminator: idle cooldown ${COOLDOWN_SECS}s WITHOUT restart, then 1 call"
    sleep "$COOLDOWN_SECS"
    fire_call 1
    ;;
  restart)
    start_daemon; stop_daemon
    echo "discriminator: restarted daemon (no cooldown gap), then 1 call"
    start_daemon
    fire_call 1
    ;;
  *)  # full soak
    start_daemon
    start_clock_sampler
    for n in $(seq 1 "$CALLS"); do fire_call "$n"; done
    ;;
esac

echo
echo "=== summary ($CSV) ==="
column -t -s, "$CSV"
if [ -s "${THERMAL_CSV:-/nonexistent}" ]; then
  echo
  echo "=== GPU clock (MHz) over the run ($THERMAL_CSV) ==="
  python3 -c 'import sys;v=[int(l.split(",")[1]) for l in open(sys.argv[1]).read().splitlines()[1:] if l.split(",")[1].isdigit()];print(f"samples={len(v)} min={min(v)} max={max(v)} first={v[0]} last={v[-1]}" if v else "no samples")' "$THERMAL_CSV"
fi
echo
echo "daemon log: $SRVLOG  (grep 'speculative summary' for server-side tok/s,"
echo "            'decode heartbeat' for per-5s phase/tps/mlx_cache,"
echo "            'prefix-cache HIT|MISS' for M59 reuse)"

# The EXIT trap SIGTERMs the daemon; `wait` then returns 143/144, which would
# become the script's exit status and read as "failed" despite a clean run.
# Force a 0 exit now that all work + the summary have printed.
trap - EXIT
stop_clock_sampler
stop_daemon
exit 0
