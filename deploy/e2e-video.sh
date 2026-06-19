#!/bin/bash
#
# e2e-video.sh — end-to-end check of POST /v1/video/transcriptions (ADR 022,
# M78.1). Synthesizes real video fixtures with ffmpeg, runs a dev-mode daemon
# against the real model store, and asserts:
#   - a video with a speech audio track  → 200 with a non-empty transcript
#   - a video with NO audio track         → 400 video_no_audio_track
#   - a video whose audio is < 0.1 s      → 400 audio_too_short
#   - a truncated/corrupt container       → 400 (invalid_audio)
#   - diarize=true on video               → 501 not_implemented
# and that the daemon SURVIVES every degenerate input (no crash) — the video
# analogue of the audio crash-hardening sweep.
#
# Requires: ffmpeg, `say`, and a built Release binary with whisper in the store.
# Usage: ./deploy/e2e-video.sh [path-to-athena-binary]
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-.build/xcode/Build/Products/Release/athena}"
PORT=7789
STORE="$HOME/.athena/models"
WORK="$(mktemp -d)"
DATA="$(mktemp -d)"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$WORK" "$DATA"' EXIT

command -v ffmpeg >/dev/null || { echo "SKIP: ffmpeg not found"; exit 0; }
[ -x "$BIN" ] || { echo "error: no binary at $BIN (build it first)"; exit 1; }
[ -d "$STORE/whisper-large-v3-turbo" ] || { echo "SKIP: no whisper model in $STORE"; exit 0; }

echo "== synthesizing fixtures =="
say -o "$WORK/speech.aiff" "Athena transcribes video end to end" 2>/dev/null
ffmpeg -loglevel error -y -f lavfi -i color=c=blue:s=320x240:d=4 \
  -i "$WORK/speech.aiff" -shortest -c:v libx264 -pix_fmt yuv420p \
  -c:a aac "$WORK/speech.mp4"
ffmpeg -loglevel error -y -f lavfi -i color=c=red:s=320x240:d=2 \
  -c:v libx264 -pix_fmt yuv420p "$WORK/noaudio.mp4"
ffmpeg -loglevel error -y -f lavfi -i color=c=green:s=320x240:d=1 \
  -f lavfi -i "sine=frequency=440:duration=0.03" -shortest \
  -c:v libx264 -pix_fmt yuv420p -c:a aac "$WORK/tiny.mp4"
head -c 3000 "$WORK/speech.mp4" > "$WORK/corrupt.mp4"

echo "== starting dev-mode daemon on :$PORT =="
"$BIN" load --port "$PORT" --data-dir "$DATA" --model-store "$STORE" \
  >"$WORK/daemon.log" 2>&1 &
DPID=$!
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done

URL="http://127.0.0.1:$PORT/v1/video/transcriptions"
fail=0
code() { curl -s -o "$WORK/out" -w "%{http_code}" -X POST "$URL" "$@"; }
ecode() { python3 -c "import sys,json;print(json.load(open('$WORK/out'))['error']['code'])" 2>/dev/null; }

echo "== 1) speech video → 200 + transcript =="
h=$(code -F "file=@$WORK/speech.mp4" -F "response_format=verbose_json")
txt=$(python3 -c "import json;print(json.load(open('$WORK/out')).get('text',''))" 2>/dev/null)
if [ "$h" = "200" ] && [ -n "${txt// /}" ]; then echo "  ok 200 text='$txt'"; else echo "  FAIL http=$h text='$txt'"; fail=1; fi

echo "== 2) no audio track → 400 video_no_audio_track =="
h=$(code -F "file=@$WORK/noaudio.mp4"); c=$(ecode)
if [ "$h" = "400" ] && [ "$c" = "video_no_audio_track" ]; then echo "  ok"; else echo "  FAIL http=$h code=$c"; fail=1; fi

echo "== 3) sub-0.1s audio → 400 audio_too_short =="
h=$(code -F "file=@$WORK/tiny.mp4"); c=$(ecode)
if [ "$h" = "400" ] && [ "$c" = "audio_too_short" ]; then echo "  ok"; else echo "  FAIL http=$h code=$c"; fail=1; fi

echo "== 4) corrupt container → 400 =="
h=$(code -F "file=@$WORK/corrupt.mp4"); c=$(ecode)
if [ "$h" = "400" ]; then echo "  ok (code=$c)"; else echo "  FAIL http=$h code=$c"; fail=1; fi

echo "== 5) diarize=true → 501 not_implemented =="
h=$(code -F "file=@$WORK/speech.mp4" -F "diarize=true"); c=$(ecode)
if [ "$h" = "501" ] && [ "$c" = "not_implemented" ]; then echo "  ok"; else echo "  FAIL http=$h code=$c"; fail=1; fi

echo "== daemon still alive after the degenerate sweep? =="
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then echo "  ok — no crash"; else echo "  FAIL — daemon died"; fail=1; fi

[ "$fail" = 0 ] && echo "e2e-video: PASS" || echo "e2e-video: FAIL"
exit $fail
