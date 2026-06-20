#!/bin/bash
#
# e2e-rbac.sh — adversarial end-to-end test of the M15.2 RBAC
# enforcement core. Builds nothing; runs the already-built `athena`
# binary as a stub daemon and exercises the security boundaries with
# curl + offline CLI:
#
#   1. permission gating       (member→admin route ⇒ 403)
#   2. scoped-token downgrade  (admin user + member-scoped token =
#                               member perms only)
#   3. cross-tenant queue      (non-owner ⇒ 404, not 403)
#   4. last-admin protection   (cannot delete/demote the only admin)
#   5. CLI escalation guards   (unknown role / unknown user refused)
#   6. fail-safe startup       (no creds + non-loopback ⇒ refuse)
#   7. auth-disabled loopback  (no creds + loopback ⇒ open)
#   8. kv_compression knob     (triattention/turboquant accepted at
#                               daemon start; unknown ⇒ fail-closed
#                               start refusal, not a silent fallback)
#   9. lifecycle UX (M22)       (`stop` halts a user daemon and surfaces
#                               sudo guidance for a system one; `init`
#                               aux-pull is idempotent; `install`
#                               synthesizes a default config)
#
# It touches ONLY an ephemeral temp data dir + a loopback port, never
# the real ~/.athena or the login Keychain (bearer tokens are passed
# inline, not stored). Ports are derived from Athena's own 7447: the
# "test realm" prefixes a 1 → 17447/17448.
#
# Usage:  deploy/e2e-rbac.sh [path/to/athena]
#
set -uo pipefail
cd "$(dirname "$0")/.."

ATHENA="${1:-}"
if [ -z "$ATHENA" ]; then
  # Pick the NEWEST built binary, not a fixed Debug-first order: a stale
  # Debug binary left over from an old build would otherwise shadow a fresh
  # Release build and run the suite against the wrong code (e.g. a 6-day-old
  # binary that predates a behavior change ⇒ a spurious mass-failure cascade).
  for c in \
    .build/xcode/Build/Products/Debug/athena \
    .build/xcode/Build/Products/Release/athena; do
    [ -x "$c" ] || continue
    if [ -z "$ATHENA" ] || [ "$c" -nt "$ATHENA" ]; then ATHENA="$c"; fi
  done
fi
[ -x "$ATHENA" ] || { echo "no athena binary (build first)"; exit 2; }
echo "using: $ATHENA"
# Absolute path for phases that `cd` into a temp dir (install dry-run).
ATHENA_ABS="$(cd "$(dirname "$ATHENA")" >/dev/null 2>&1 && pwd)/$(basename "$ATHENA")"

D="$(mktemp -d)"
EMPTY="$(mktemp -d)"
MSTORE="$(mktemp -d)"          # ephemeral model store (never ~/.athena)
PORT=17447
DPID=""
PASS=0
FAIL=0

# Seed one fake model so list/show/cp/rm have something to act on,
# plus a dangling symlink so `prune` (M16.3) has a clear victim.
mkdir -p "$MSTORE/fake-model"
printf '{"model_type":"test","hidden_size":8}' \
  > "$MSTORE/fake-model/config.json"
: > "$MSTORE/fake-model/model.safetensors"
ln -s /nonexistent/gone "$MSTORE/dead"

D2=""                          # phase-8 isolated RBAC-admin data dir
D3=""                          # phase-8.7 audit-retention data dir
D4=""                          # phase-25 queue-TTL data dir
D5=""                          # phase-25.1 queue-max-rows data dir
D6=""                          # phase-25.2 vector-TTL data dir
D7=""                          # phase-25.3 content opt-out data dir
D8=""                          # phase-25.3 content-retained control dir
D9=""                          # phase-26 encrypted-store data dir
D10=""                         # phase-26 plaintext→encrypted migration dir
cleanup() {
  [ -n "$DPID" ] && kill "$DPID" 2>/dev/null
  wait "$DPID" 2>/dev/null
  rm -rf "$D" "$EMPTY" "$MSTORE" ${D2:+"$D2"} ${D3:+"$D3"} \
    ${D4:+"$D4"} ${D5:+"$D5"} ${D6:+"$D6"} ${D7:+"$D7"} ${D8:+"$D8"} \
    ${D9:+"$D9"} ${D10:+"$D10"}
  unset ATHENA_STORE_KEY
}
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok   $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL $1"; }

# code EXPECTED METHOD PATH [BEARER] [BODY]
code() {
  local want="$1" method="$2" path="$3" bearer="${4:-}" body="${5:-}"
  local args=(-s -o /dev/null -w "%{http_code}" -X "$method"
              "http://127.0.0.1:$PORT$path")
  [ -n "$bearer" ] && args+=(-H "Authorization: Bearer $bearer")
  if [ -n "$body" ]; then
    args+=(-H "Content-Type: application/json" -d "$body")
  fi
  local got
  got="$(curl "${args[@]}")"
  # M43.2 — cold-load 503 is a transient signal that a module is
  # warming; retry briefly when the test expects something other than
  # 503 (mirrors what a Retry-After-aware client does). Tests that
  # explicitly want 503 (budget pressure) skip this entirely.
  if [ "$got" = "503" ] && [ "$want" != "503" ]; then
    for _ in $(seq 1 20); do
      sleep 0.1
      got="$(curl "${args[@]}")"
      [ "$got" != "503" ] && break
    done
  fi
  if [ "$got" = "$want" ]; then
    ok "$method $path ($bearer) → $got"
  else
    bad "$method $path ($bearer) → $got (want $want)"
  fi
}

start_daemon() { # DATADIR HOST [EXTRA-FLAGS...]
  local dd="$1" host="$2"; shift 2
  "$ATHENA" load --engine stub --host "$host" --port "$PORT" \
    --data-dir "$dd" --model-store "$MSTORE" "$@" \
    > "$D/daemon.log" 2>&1 &
  DPID=$!
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
      return 0
    fi
    if ! kill -0 "$DPID" 2>/dev/null; then return 1; fi
    sleep 0.5
  done
  return 1
}
stop_daemon() {
  [ -n "$DPID" ] && kill "$DPID" 2>/dev/null
  wait "$DPID" 2>/dev/null
  DPID=""
}

# M43.2 — preload all module slots via /api/models/load (the explicit-
# load endpoint kept blocking semantics on purpose). Lets body-parsing
# tests that fire a single inference call avoid the cold-load 503. The
# stub loads in microseconds; retries cover the brief loading-state
# window. ADMIN_TOK must be set; explicit-load needs `model.write`.
warm() {
  local tok="$1" mod rc
  for mod in llm textEmbedding transcription diarization speakerEmbedding; do
    for _ in $(seq 1 30); do
      rc="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $tok" \
        -H 'Content-Type: application/json' \
        -d "{\"module\":\"$mod\"}" \
        "http://127.0.0.1:$PORT/api/models/load")"
      [ "$rc" = "200" ] && break
      sleep 0.05
    done
  done
}

CHAT='{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}]}'

echo
echo "== phase 0: seed RBAC subjects (offline CLI) =="
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin \
  --role admin --data-dir "$D" >/dev/null \
  && ok "create admin user (role admin)" || bad "create admin user"
ATHENA_PASSWORD=alicepass1 "$ATHENA" auth user add alice \
  --role member --data-dir "$D" >/dev/null \
  && ok "create alice (member)" || bad "create alice"
ATHENA_PASSWORD=bobpass123 "$ATHENA" auth user add bob \
  --role member --data-dir "$D" >/dev/null \
  && ok "create bob (member)" || bad "create bob"
ATHENA_PASSWORD=ropass1234 "$ATHENA" auth user add ro \
  --role readonly --data-dir "$D" >/dev/null \
  && ok "create ro (readonly)" || bad "create ro"
# boss is a full admin USER but we mint a member-SCOPED token for it.
ATHENA_PASSWORD=bosspass123 "$ATHENA" auth user add boss \
  --role admin --data-dir "$D" >/dev/null \
  && ok "create boss (admin)" || bad "create boss"

tok() { # USER [--role R ...]
  "$ATHENA" auth token add --user "$1" "${@:2}" --data-dir "$D" \
    2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1
}
ADMIN_TOK="$(tok admin)"
ALICE_TOK="$(tok alice)"
BOB_TOK="$(tok bob)"
RO_TOK="$(tok ro)"
BOSS_SCOPED="$(tok boss --role member)"   # admin user, member scope
for v in ADMIN_TOK ALICE_TOK BOB_TOK RO_TOK BOSS_SCOPED; do
  [ -n "${!v}" ] && ok "minted $v" || bad "minted $v (empty!)"
done

echo
echo "== phase 1: CLI escalation / validation guards =="
"$ATHENA" auth role grant alice notarole --data-dir "$D" \
  >/dev/null 2>&1 \
  && bad "unknown role accepted" || ok "unknown role refused"
"$ATHENA" auth role grant ghost member --data-dir "$D" \
  >/dev/null 2>&1 \
  && bad "grant to nonexistent user accepted" \
  || ok "grant to nonexistent user refused"
# B5 (M66.2): `user add` must not silently overwrite an existing account.
ATHENA_PASSWORD=duppass12 "$ATHENA" auth user add dupuser --role member \
  --data-dir "$D" >/dev/null 2>&1 \
  && ok "B5: first user add succeeds" || bad "B5: first add failed"
ATHENA_PASSWORD=other123 "$ATHENA" auth user add dupuser --role member \
  --data-dir "$D" >/dev/null 2>&1 \
  && bad "B5: re-add overwrote existing user without --force" \
  || ok "B5: re-add refused without --force"
ATHENA_PASSWORD=other123 "$ATHENA" auth user add dupuser --role member \
  --force --data-dir "$D" >/dev/null 2>&1 \
  && ok "B5: re-add with --force succeeds" \
  || bad "B5: --force re-add failed"
# B2 (ADR 005): `--password` is REMOVED from argv. A valid ATHENA_PASSWORD
# is set, so the only reason this fails is the now-unknown --password flag.
B2OUT="$(ATHENA_PASSWORD=validpass12 "$ATHENA" auth user add b2user \
  --password secret123 --role member --data-dir "$D" 2>&1)"
if [ $? -ne 0 ] && echo "$B2OUT" | grep -qi "password"; then
  ok "B2: --password rejected (removed from argv)"
else
  bad "B2: --password still accepted ($B2OUT)"
fi
# NB1 (M66.3): a malformed --label must fail UP FRONT (before euid
# branching), not fall through to a root-owned daemon spawn. Validated
# offline (non-root) — an invalid label dies before any daemon is spawned.
NBOUT="$("$ATHENA" start --label 'bad label' --data-dir "$D" 2>&1)"
if [ $? -ne 0 ] && echo "$NBOUT" | grep -qi "invalid --label"; then
  ok "NB1: malformed --label rejected up front"
else
  bad "NB1: malformed --label not rejected ($NBOUT)"
fi
# M66.4 config-set validation (NB8/NB2/B15), offline against a temp config.
CFG="$D/cfgtest.toml"
printf 'listen_host = "127.0.0.1"\nlisten_port = 7447\nlog_dir = "/tmp/l"\n' \
  > "$CFG"
# --no-apply: edit the TOML only (skip the root-gated plist re-render), so
# the exit code reflects set-time VALIDATION, not the install state.
"$ATHENA" config set engine bogus --no-apply --config "$CFG" \
  >/dev/null 2>&1 \
  && bad "NB8: invalid engine accepted" \
  || ok "NB8: invalid engine rejected at set-time"
"$ATHENA" config set kv_compression bogus --no-apply --config "$CFG" \
  >/dev/null 2>&1 \
  && bad "NB8: invalid kv_compression accepted" \
  || ok "NB8: invalid kv_compression rejected at set-time"
"$ATHENA" config set engine stub --no-apply --config "$CFG" \
  >/dev/null 2>&1 \
  && ok "NB8: valid engine accepted" || bad "NB8: valid engine rejected"
# NB2: a value carrying a newline (config-line injection) or a quote is
# rejected before it can be written.
"$ATHENA" config set model "$(printf 'a\nauth_keys_file = "/x')" \
  --no-apply --config "$CFG" >/dev/null 2>&1 \
  && bad "NB2: newline-injection value accepted" \
  || ok "NB2: newline-injection value rejected"
"$ATHENA" config set model 'a"b' --no-apply --config "$CFG" \
  >/dev/null 2>&1 \
  && bad "NB2: quoted value accepted" || ok "NB2: quote in value rejected"
# B15: a NEW bare key on a config that already has a [section] must land
# BEFORE the section header (stay top-level), not inside the table.
printf 'listen_host = "127.0.0.1"\nlisten_port = 7447\nlog_dir = "/tmp/l"\n[prompt_cache]\nprompt_cache_enabled = true\n' \
  > "$CFG"
"$ATHENA" config set max_tokens 2048 --no-apply --config "$CFG" \
  >/dev/null 2>&1
if awk '/^\[prompt_cache\]/{seen=1} /^max_tokens/{if(seen)after=1} END{exit(after?1:0)}' "$CFG"; then
  ok "B15: bare key inserted before [section]"
else
  bad "B15: bare key landed inside [section]"
fi
# M71.3 — `athena show` surfaces a vision-capability line driven by the
# config-only `vision_config` signal (== servesVision at load). Offline,
# config-only: a VLM checkpoint reports "vision:   yes", a text model "no".
mkdir -p "$MSTORE/vlm-model"
printf '{"model_type":"gemma4","vision_config":{"hidden_size":1}}' \
  > "$MSTORE/vlm-model/config.json"
SHOW_VLM="$("$ATHENA" show vlm-model --model-store "$MSTORE" 2>&1)"
echo "$SHOW_VLM" | grep -Eq '^vision: +yes' \
  && ok "M71.3: show prints vision: yes for a VLM" \
  || bad "M71.3: show missing/wrong vision line for VLM ($SHOW_VLM)"
SHOW_TXT="$("$ATHENA" show fake-model --model-store "$MSTORE" 2>&1)"
echo "$SHOW_TXT" | grep -Eq '^vision: +no' \
  && ok "M71.3: show prints vision: no for a text model" \
  || bad "M71.3: show missing/wrong vision line for text model ($SHOW_TXT)"
rm -rf "$MSTORE/vlm-model"
# #11 / M43.5 — `athena allowlist` edits the on-disk store directly with
# `--data-dir` while NO daemon is running (operability for a wedged box).
ALLOWDD="$(mktemp -d)"
"$ATHENA" allowlist add --module llm --id m-foo --default \
  --data-dir "$ALLOWDD" >/dev/null 2>&1 \
  && ok "#11: offline allowlist add (daemon down)" \
  || bad "#11: offline allowlist add failed"
"$ATHENA" allowlist add --module llm --id m-bar --data-dir "$ALLOWDD" \
  >/dev/null 2>&1
ALOUT="$("$ATHENA" allowlist list --data-dir "$ALLOWDD" 2>&1)"
echo "$ALOUT" | grep -q 'm-foo' && echo "$ALOUT" | grep -q 'm-bar' \
  && ok "#11: offline allowlist list renders rows" \
  || bad "#11: offline allowlist list missing rows ($ALOUT)"
"$ATHENA" allowlist default --module llm --id m-bar --data-dir "$ALLOWDD" \
  >/dev/null 2>&1
"$ATHENA" allowlist list --module llm --data-dir "$ALLOWDD" 2>&1 \
  | grep -Eq '\*\s+m-bar' \
  && ok "#11: offline allowlist default re-points" \
  || bad "#11: offline allowlist default did not re-point"
"$ATHENA" allowlist rm --module llm --id m-foo --data-dir "$ALLOWDD" \
  >/dev/null 2>&1 \
  && ok "#11: offline allowlist rm" || bad "#11: offline allowlist rm failed"
"$ATHENA" allowlist add --module bogus --id x --data-dir "$ALLOWDD" \
  >/dev/null 2>&1 \
  && bad "#11: offline allowlist accepted bad module" \
  || ok "#11: offline allowlist rejects unknown module"
rm -rf "$ALLOWDD"

echo
echo "== phase 2: permission gating (auth enforced) =="
start_daemon "$D" 127.0.0.1 \
  --embedding-model athena-embedding-default \
  --embedding-model athena-embedding-alt \
  || { echo "daemon failed"; \
  cat "$D/daemon.log"; exit 1; }
grep -q "auth: enabled (RBAC" "$D/daemon.log" \
  && ok "daemon reports RBAC enabled" || bad "RBAC-enabled log line"
# M43.2 — pre-warm module slots so body-parsing tests don't trip on
# the new cold-load 503 (handled in phase 3.687 explicitly).
warm "$ADMIN_TOK"

code 401 POST /v1/chat/completions "" "$CHAT"          # no token
code 401 POST /v1/chat/completions "sk-athena-bogus" "$CHAT"
# v0.10.38 — auth-deny bodies must be valid JSON. The pre-fix
# hand-formatted string produced three consecutive double-quotes
# between `auth_error,` and `"code"`, breaking strict JSON parsers.
DENYBODY="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$CHAT")"
echo "$DENYBODY" | python3 -c 'import json,sys; json.loads(sys.stdin.read())' \
  >/dev/null 2>&1 \
  && ok "auth deny body parses as JSON" \
  || bad "auth deny body is malformed JSON ($DENYBODY)"
echo "$DENYBODY" | grep -q '"code":"unauthorized"' \
  && ok "auth deny body carries code:unauthorized" \
  || bad "auth deny body missing code field ($DENYBODY)"
# M43.4 #5 — auth-deny envelope carries an operator-facing `hint` so a
# 401/403 names a remedy (ATHENA_KEY, /ui/login, role grant) instead of
# bare "missing or invalid bearer token".
echo "$DENYBODY" | grep -q '"hint":' \
  && ok "auth deny body carries hint field (M43.4)" \
  || bad "auth deny body missing hint ($DENYBODY)"
echo "$DENYBODY" | grep -q "ATHENA_KEY" \
  && ok "auth hint mentions ATHENA_KEY remediation" \
  || bad "auth hint did not name the standard remediation"
# M45.6: hint also names `athena auth login` so operators know about
# the Keychain-cache path (not just env + flag).
echo "$DENYBODY" | grep -q "athena auth login" \
  && ok 'auth hint mentions "athena auth login" (Keychain cache)' \
  || bad "auth hint omits the Keychain-cache remediation"
FORBODY="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  "http://127.0.0.1:$PORT/v1/store/stats")"
echo "$FORBODY" | grep -q '"hint":' \
  && ok "forbidden envelope carries hint field" \
  || bad "forbidden envelope missing hint ($FORBODY)"
code 200 POST /v1/chat/completions "$ALICE_TOK" "$CHAT" # member: inference
code 200 GET  /metrics "$ADMIN_TOK"                     # admin: all
code 403 GET  /metrics "$ALICE_TOK"                     # member ∌ metricsRead
code 200 GET  /metrics "$RO_TOK"                        # readonly ∋ metricsRead
code 403 POST /v1/chat/completions "$RO_TOK" "$CHAT"    # readonly ∌ inference
code 200 GET  /v1/store/stats "$ADMIN_TOK"              # storeAdmin
code 403 GET  /v1/store/stats "$ALICE_TOK"              # member ∌ storeAdmin
code 303 GET  /ui "$ALICE_TOK"                          # ∌ daemonAdmin → login
code 200 GET  /healthz ""                               # always open

echo
echo "== phase 2.1: OpenAI finish_reason length on truncation (M31.2) =="
# A positive max_tokens truncates the stub stream ⇒ finish_reason
# "length"; an uncapped request ends naturally ⇒ "stop". Same signal on
# the sync body and the terminal SSE chunk.
CHATCAP='{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"max_tokens":2}'
FRL="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$CHATCAP" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$FRL" | grep -q '"finish_reason":"length"' \
  && ok "sync truncation ⇒ finish_reason length" \
  || bad "sync finish_reason not length ($FRL)"
FRS="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$CHAT" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$FRS" | grep -q '"finish_reason":"stop"' \
  && ok "sync natural end ⇒ finish_reason stop" \
  || bad "sync finish_reason not stop ($FRS)"
SSEL="$(curl -s -N -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":2}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$SSEL" | grep -q '"finish_reason":"length"' \
  && ok "SSE truncation ⇒ terminal finish_reason length" \
  || bad "SSE finish_reason not length ($SSEL)"
echo "$SSEL" | grep -q 'data: \[DONE\]' \
  && ok "SSE still terminates with [DONE]" \
  || bad "SSE missing [DONE] ($SSEL)"

echo
echo "== phase 2.2: OpenAI sampling params — stop/seed/top_p + 400s (M31.3) =="
# Unsupported under greedy/MTP/structured determinism ⇒ a clear 400
# (NOT a silent ignore). Member token clears auth so the handler runs.
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"n":2}'
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"logprobs":true}'
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"logit_bias":{"50256":-100}}'
# n:1 is the supported single-decode case ⇒ 200.
code 200 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"n":1}'

# C2 (ADR 013 §4): logprobs is HONORED on the deterministic path. With
# temperature:0 the request returns 200 carrying choices[].logprobs.content,
# each entry with token/logprob/top_logprobs (synthesized by the stub).
LP="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"temperature":0,"logprobs":true,"top_logprobs":2}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
{ echo "$LP" | grep -q '"logprobs"' && echo "$LP" | grep -q '"top_logprobs"' \
  && echo "$LP" | grep -q '"content"'; } \
  && ok "logprobs:true + temperature:0 ⇒ 200 with logprobs.content + top_logprobs" \
  || bad "logprobs not honored on deterministic path ($LP)"
# Streamed deterministic request emits a chunk carrying logprobs.
LPS="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"temperature":0,"logprobs":true,"stream":true}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$LPS" | grep -q '"logprobs"' \
  && ok "streamed logprobs ⇒ a chunk carries the logprobs object" \
  || bad "no logprobs in streamed chunks ($LPS)"
# top_logprobs out of range (0..20) ⇒ 400 invalid_top_logprobs.
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"temperature":0,"logprobs":true,"top_logprobs":21}'
# top_logprobs without logprobs:true ⇒ 400.
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"temperature":0,"top_logprobs":2}'
# logprobs on a sampling request (temperature>0, no schema) ⇒ 400 (no capture seam).
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"logprobs":true}'
# M46.4 — case-insensitive model resolution. The allowlist stores the
# canonical id `Qwen3.5-27B-4bit-mtp` (mixed case); requesting it
# fully lowercased must still resolve and serve. The truthful served-
# model field on the response must echo the CANONICAL stored id, not
# the request's spelling.
CI="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-27b-4bit-mtp","messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$CI" | grep -q '"finish_reason":"stop"' \
  && ok "lowercased model resolves ⇒ 200/stop" \
  || bad "lowercased model rejected ($CI)"
echo "$CI" | grep -q '"model":"Qwen3.5-27B-4bit-mtp"' \
  && ok "response echoes canonical id (not the request casing)" \
  || bad "response did not canonicalize served model ($CI)"
# top_p/seed are accepted (inert on the stub's model-less path) ⇒ 200.
TPS="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"top_p":0.9,"seed":42}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$TPS" | grep -q '"finish_reason":"stop"' \
  && ok "top_p+seed accepted ⇒ 200/stop" \
  || bad "top_p+seed request rejected ($TPS)"
# stop truncates the output at the first sequence ⇒ finish_reason stop and
# the text after the sequence is gone (string form).
STOPR="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"stop":"governed"}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$STOPR" | grep -q '"finish_reason":"stop"' \
  && ok "stop hit ⇒ finish_reason stop" \
  || bad "stop did not set finish_reason ($STOPR)"
echo "$STOPR" | grep -q 'governed' \
  && bad "stop did not truncate (text still has 'governed': $STOPR)" \
  || ok "stop truncated the body before the sequence"
# stop also accepts an array form, applied on the SSE stream.
SSES="$(curl -s -N -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"stream":true,"stop":["governed"]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$SSES" | grep -q '"finish_reason":"stop"' \
  && ok "SSE stop ⇒ terminal finish_reason stop" \
  || bad "SSE stop finish_reason missing ($SSES)"
echo "$SSES" | grep -q 'governed' \
  && bad "SSE stop did not truncate the stream ($SSES)" \
  || ok "SSE stop suppressed text from the sequence on"
# M46.3b — `chat_template_kwargs` is accepted on the request shape and
# does not break decoding. The stub engine has no tokenizer/chat
# template, so the actual `enable_thinking=false` behaviour is exercised
# end-to-end against the real MLX module on the manual host-bound tier;
# this assertion confirms the DTO decodes the field and the daemon
# returns a normal 200 (not a 400/500 from a JSON mismatch).
CTK="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false,"foo":"bar"}}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$CTK" | grep -q '"finish_reason":"stop"' \
  && ok "chat_template_kwargs accepted ⇒ 200/stop" \
  || bad "chat_template_kwargs request rejected ($CTK)"

echo
echo "== phase 2.3: M71.1 image content-parts — wire protocol + passive-oracle =="
# Vision wire protocol: OpenAI image_url content-parts decode; the passive-
# oracle rule rejects remote (http/https) image URLs with 400; a valid inline
# data: image is accepted by the DTO but has no model path yet (M71.1) ⇒ a
# clean 400 vision_not_supported (flipped per-model when the VLM path lands,
# M71.2). A plain-string content message is unaffected (still 200).
# 1x1 PNG, base64.
PNG_B64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
IMG_DATA="{\"model\":\"Qwen3.5-27B-4bit-mtp\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"describe\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$PNG_B64\"}}]}]}"
IMG_HTTP='{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}'
# valid inline image, no VLM path yet ⇒ 400 vision_not_supported
code 400 POST /v1/chat/completions "$ALICE_TOK" "$IMG_DATA"
VNS="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$IMG_DATA" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$VNS" | grep -q '"code":"vision_not_supported"' \
  && ok "inline data: image ⇒ 400 vision_not_supported" \
  || bad "inline image not vision_not_supported ($VNS)"
# remote image URL ⇒ 400 invalid_image (passive-oracle: no outbound fetch)
code 400 POST /v1/chat/completions "$ALICE_TOK" "$IMG_HTTP"
REM="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$IMG_HTTP" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$REM" | grep -q '"code":"invalid_image"' \
  && ok "remote http(s) image ⇒ 400 invalid_image (passive-oracle)" \
  || bad "remote image not invalid_image ($REM)"
# plain-string content path unchanged ⇒ still 200
code 200 POST /v1/chat/completions "$ALICE_TOK" "$CHAT"

echo
echo "== phase 2.5: WebUI session cookie + RBAC nav + CSRF (M18.1) =="
# /ui is SESSION-cookie authed (not bearer). admin (daemonAdmin) gets
# the control shell; a member is bounced to login; mutations require
# the per-session CSRF token ON TOP of the cookie. The positive POST
# uses an EMPTY body {} — it clears CSRF + the per-action daemonAdmin
# re-check and returns 200 while writing NOTHING (keeps this script's
# "never touches the real config" invariant).
UIJAR="$D/ui-admin.jar"; UIJAR_A="$D/ui-alice.jar"
B="http://127.0.0.1:$PORT"
LC="$(curl -s -o /dev/null -w '%{http_code}' -c "$UIJAR" \
  -d 'username=admin&password=adminpass1' "$B/ui/login")"
[ "$LC" = 303 ] && ok "admin /ui/login → 303" \
  || bad "admin /ui/login → $LC (want 303)"
grep -q athena_session "$UIJAR" \
  && ok "session cookie set" || bad "no session cookie"
DASH="$(curl -s -b "$UIJAR" "$B/ui")"
echo "$DASH" | grep -q 'class=topbar' \
  && ok "/ui renders RBAC shell nav" || bad "/ui missing shell nav"
echo "$DASH" | grep -q 'href="/ui/config"' \
  && ok "admin nav exposes config (daemonAdmin)" \
  || bad "admin nav missing config link"
CSRF="$(curl -s -b "$UIJAR" "$B/ui/config" \
  | sed -n 's/.*name="csrf" content="\([^"]*\)".*/\1/p' | head -1)"
[ -n "$CSRF" ] && ok "per-session CSRF token minted in page" \
  || bad "no CSRF token in config page"
NC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR" -X POST \
  -H 'Content-Type: application/json' -d '{}' "$B/ui/api/config")"
[ "$NC" = 403 ] && ok "config POST sans CSRF → 403" \
  || bad "config POST sans CSRF → $NC (want 403)"
WC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR" -X POST \
  -H 'Content-Type: application/json' -H 'X-CSRF-Token: forged' \
  -d '{}' "$B/ui/api/config")"
[ "$WC" = 403 ] && ok "config POST bad CSRF → 403" \
  || bad "config POST bad CSRF → $WC (want 403)"
GC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR" -X POST \
  -H 'Content-Type: application/json' -H "X-CSRF-Token: $CSRF" \
  -d '{}' "$B/ui/api/config")"
[ "$GC" = 200 ] && ok "config POST with valid CSRF → 200" \
  || bad "config POST with valid CSRF → $GC (want 200)"
curl -s -o /dev/null -c "$UIJAR_A" \
  -d 'username=alice&password=alicepass1' "$B/ui/login"
AC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" "$B/ui")"
[ "$AC" = 303 ] && ok "member /ui (cookie) → 303 (∌ daemonAdmin)" \
  || bad "member /ui (cookie) → $AC (want 303)"
AM="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" -X POST \
  -H 'Content-Type: application/json' -H 'X-CSRF-Token: x' \
  -d '{}' "$B/ui/api/config")"
[ "$AM" = 303 ] && ok "member config POST → 303 (gated pre-handler)" \
  || bad "member config POST → $AM (want 303)"

echo
echo "== phase 2.6: WebUI model console reuse + RBAC/CSRF (M18.2) =="
# /ui/api/models* re-checks the cookie user's model.read/write + CSRF
# then REUSES the M16 op layer (ModelStoreOps / enqueueModelOp).
MPAGE="$(curl -s -b "$UIJAR" "$B/ui/models")"
echo "$MPAGE" | grep -q 'href="/ui/models"' \
  && ok "models nav link present (model.read)" \
  || bad "models nav link missing"
CSRF="$(printf '%s' "$MPAGE" \
  | sed -n 's/.*name="csrf" content="\([^"]*\)".*/\1/p' | head -1)"
ML="$(curl -s -b "$UIJAR" "$B/ui/api/models")"
echo "$ML" | grep -q 'fake-model' \
  && ok "/ui/api/models lists the seeded model (reuse)" \
  || bad "/ui/api/models missing fake-model ($ML)"
uic() { # WANT METHOD PATH [CSRF] [BODY]
  local w="$1" m="$2" p="$3" cs="${4:-}" bd="${5:-}"
  local a=(-s -o /dev/null -w '%{http_code}' -b "$UIJAR"
           -X "$m" "$B$p")
  [ -n "$cs" ] && a+=(-H "X-CSRF-Token: $cs")
  [ -n "$bd" ] && a+=(-H 'Content-Type: application/json' -d "$bd")
  local g; g="$(curl "${a[@]}")"
  [ "$g" = "$w" ] && ok "$m $p → $g" \
    || bad "$m $p → $g (want $w)"
}
uic 200 GET  "/ui/api/models/default"
uic 200 GET  "/ui/api/models/show?name=fake-model"
uic 404 GET  "/ui/api/models/show?name=nope"
uic 403 POST "/ui/api/models/rm" "" '{"name":"fake-model"}' # no CSRF
uic 403 POST "/ui/api/models/rm" "bad" '{"name":"fake-model"}'
uic 404 POST "/ui/api/models/rm" "$CSRF" '{"name":"ghost"}'   # reached
uic 200 POST "/ui/api/models/copy" "$CSRF" \
  '{"src":"fake-model","dst":"ui-tmp","copy":false}'
uic 200 POST "/ui/api/models/rm" "$CSRF" '{"name":"ui-tmp"}'
# pull enqueues via the SAME M16 path → 202 {job_id}; poll the
# console job endpoint (nil-owner, model.read-gated).
JR="$(curl -s -b "$UIJAR" -X POST -H "X-CSRF-Token: $CSRF" \
  -H 'Content-Type: application/json' -d '{"id":"hf/none"}' \
  "$B/ui/api/models/pull")"
JID="$(printf '%s' "$JR" | sed -n \
  's/.*"job_id":"\([^"]*\)".*/\1/p')"
[ -n "$JID" ] && ok "pull enqueued via reuse (job $JID)" \
  || bad "pull did not return job_id ($JR)"
uic 200 GET "/ui/api/job?id=$JID"
uic 400 POST "/ui/api/models/pull" "$CSRF" '{"id":""}'  # validator
# member: model nav/page/mutation all gated pre-handler (∌ daemonAdmin)
MC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" \
  "$B/ui/models")"
[ "$MC" = 303 ] && ok "member /ui/models → 303 (∌ daemonAdmin)" \
  || bad "member /ui/models → $MC (want 303)"
RC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" \
  -X POST -H 'X-CSRF-Token: x' -H 'Content-Type: application/json' \
  -d '{"name":"fake-model"}' "$B/ui/api/models/rm")"
[ "$RC" = 303 ] && ok "member model rm → 303 (gated pre-handler)" \
  || bad "member model rm → $RC (want 303)"

echo
echo "== phase 2.7: WebUI daemon console + shared admin op (M18.3) =="
# /ui/api/admin/* re-checks the cookie user's daemon.admin + CSRF,
# then runs the SAME adminUnloadLLM/adminLoadLLM/adminStatus the
# public /api/admin/* routes now delegate to (one impl).
DP="$(curl -s -b "$UIJAR" "$B/ui/daemon")"
echo "$DP" | grep -q 'href="/ui/daemon"' \
  && ok "daemon nav link present (daemon.admin)" \
  || bad "daemon nav link missing"
SJ="$(curl -s -b "$UIJAR" "$B/ui/api/admin/status")"
echo "$SJ" | grep -q '"listen"' \
  && ok "/ui/api/admin/status posture (shared impl)" \
  || bad "/ui/api/admin/status missing posture ($SJ)"
uic 403 POST "/ui/api/admin/load"   ""        # no CSRF
uic 403 POST "/ui/api/admin/stop"   "bad"     # bad CSRF
uic 200 POST "/ui/api/admin/load"   "$CSRF"   # warm (stub loads)
uic 200 POST "/ui/api/admin/stop"   "$CSRF"   # unload
# the refactor must not regress the public bearer admin routes
code 200 GET  /api/admin/status "$ADMIN_TOK"
code 200 POST /api/admin/stop   "$ADMIN_TOK"
code 403 POST /api/admin/stop   "$ALICE_TOK"  # member ∌ daemonAdmin
# member: daemon nav/page/control all gated pre-handler
DC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" \
  "$B/ui/daemon")"
[ "$DC" = 303 ] && ok "member /ui/daemon → 303 (∌ daemonAdmin)" \
  || bad "member /ui/daemon → $DC (want 303)"
SC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" \
  -X POST -H 'X-CSRF-Token: x' "$B/ui/api/admin/stop")"
[ "$SC" = 303 ] && ok "member admin stop → 303 (gated pre-handler)" \
  || bad "member admin stop → $SC (want 303)"

echo
echo "== phase 2.8: WebUI RBAC admin reuse (cookie-aware) (M18.4) =="
# REGRESSION-CRITICAL: before M18.4 the M16.4 handlers derived the
# caller from the BEARER header, so a cookie admin got perms=[] and
# every canGrant failed (403). callerPermissions is now cookie-aware
# ⇒ the SAME handlers enforce canGrant / last-admin against the
# LOGGED-IN user.
UP="$(curl -s -b "$UIJAR" "$B/ui/users")"
echo "$UP" | grep -q 'href="/ui/users"' \
  && ok "users nav link present (users.read)" \
  || bad "users nav link missing"
curl -s -b "$UIJAR" "$B/ui/api/users" | grep -q '"admin"' \
  && ok "/ui/api/users lists accounts (reuse)" \
  || bad "/ui/api/users missing accounts"
curl -s -b "$UIJAR" "$B/ui/api/roles" | grep -q '"operator"' \
  && ok "/ui/api/roles catalog (shared impl)" \
  || bad "/ui/api/roles missing catalog"
uic 200 GET  "/ui/api/tokens"
uic 403 POST "/ui/api/users" ""    '{"username":"webu","password":"webpass12","role":"member"}'
uic 403 POST "/ui/api/users" "bad" '{"username":"webu","password":"webpass12","role":"member"}'
# the assertion that would have FAILED pre-M18.4 (cookie admin can
# canGrant member):
uic 200 POST "/ui/api/users" "$CSRF" \
  '{"username":"webu","password":"webpass12","role":"member"}'
uic 200 POST "/ui/api/users/role/grant"  "$CSRF" \
  '{"name":"webu","role":"operator"}'
uic 200 POST "/ui/api/users/role/revoke" "$CSRF" \
  '{"name":"webu","role":"operator"}'
uic 404 POST "/ui/api/users/delete" "$CSRF" '{"name":"ghost"}'
uic 200 POST "/ui/api/users/delete" "$CSRF" '{"name":"webu"}'
# mint a token for bob over the cookie (admin ⊇ member ⇒ allowed),
# shown once; then revoke it by its own hash prefix.
TKR="$(curl -s -b "$UIJAR" -X POST -H "X-CSRF-Token: $CSRF" \
  -H 'Content-Type: application/json' -d '{"user":"bob"}' \
  "$B/ui/api/tokens")"
echo "$TKR" | grep -q '"token":"sk-athena-' \
  && ok "cookie token mint returns secret ONCE" \
  || bad "cookie token mint missing secret ($TKR)"
HP="$(printf '%s' "$TKR" | sed -n \
  's/.*"hash_prefix":"\([0-9a-f]*\)".*/\1/p')"
uic 200 POST "/ui/api/tokens/delete" "$CSRF" \
  "{\"prefix\":\"$HP\"}"
# entry gate: only daemonAdmin reaches /ui* — a readonly user holds
# users.read but is still bounced (defense-in-depth design).
curl -s -o /dev/null -c "$D/ui-ro.jar" \
  -d 'username=ro&password=ropass1234' "$B/ui/login"
RR="$(curl -s -o /dev/null -w '%{http_code}' -b "$D/ui-ro.jar" \
  "$B/ui/users")"
[ "$RR" = 303 ] && ok "readonly /ui/users → 303 (∌ daemonAdmin gate)" \
  || bad "readonly /ui/users → $RR (want 303)"
UC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" -X POST \
  -H 'X-CSRF-Token: x' -H 'Content-Type: application/json' \
  -d '{"username":"x","password":"xxxxxxxx"}' "$B/ui/api/users")"
[ "$UC" = 303 ] && ok "member user-create → 303 (gated pre-handler)" \
  || bad "member user-create → $UC (want 303)"

echo
echo "== phase 2.85: WebUI allowlist console (M44.1) =="
# /ui/allowlist + /ui/api/allowlist* re-check the cookie user's
# model.read (list) / model.write (mutations) + CSRF, then DELEGATE to
# the SAME M42.2 handlers the bearer /api/models/allow routes use — so
# the live-refresh + audit tail can't drift between the two surfaces.
APAGE="$(curl -s -b "$UIJAR" "$B/ui/allowlist")"
echo "$APAGE" | grep -q 'href="/ui/allowlist"' \
  && ok "allowlist nav link present (model.read)" \
  || bad "allowlist nav link missing ($APAGE)"
echo "$APAGE" | grep -q 'name="write" content="1"' \
  && ok "allowlist page exposes WRITE=1 for admin (model.write)" \
  || bad "allowlist page missing WRITE=1 ($APAGE)"
ACSRF="$(printf '%s' "$APAGE" \
  | sed -n 's/.*name="csrf" content="\([^"]*\)".*/\1/p' | head -1)"
[ -n "$ACSRF" ] && ok "allowlist page mints CSRF token" \
  || bad "allowlist page missing CSRF ($APAGE)"
# list reuses handleAllowlistList — must include the stub's seeded
# entries (set by `athena load --llm-model fake-model` at boot).
AL="$(curl -s -b "$UIJAR" "$B/ui/api/allowlist")"
echo "$AL" | grep -q '"allowlist"' \
  && ok "/ui/api/allowlist returns allowlist[] (reuse)" \
  || bad "/ui/api/allowlist missing allowlist[] ($AL)"
# add + default + rm via the cookie path. The bearer /api side covers
# the audit + live-refresh tail (asserted elsewhere); this confirms the
# JSON-body wrappers reach the SAME handlers.
uic 403 POST "/ui/api/allowlist" "" \
  '{"module":"textEmbedding","id":"ui-test-emb"}'  # no CSRF
uic 403 POST "/ui/api/allowlist" "bad" \
  '{"module":"textEmbedding","id":"ui-test-emb"}'  # bad CSRF
uic 400 POST "/ui/api/allowlist" "$ACSRF" \
  '{"module":"NOPE","id":"x"}'                      # invalid_module
uic 400 POST "/ui/api/allowlist" "$ACSRF" \
  '{"module":"textEmbedding","id":""}'              # empty id
uic 200 POST "/ui/api/allowlist" "$ACSRF" \
  '{"module":"textEmbedding","id":"ui-test-emb"}'   # added
# the addition must be visible to the same /ui list reader
curl -s -b "$UIJAR" "$B/ui/api/allowlist" \
  | grep -q '"id":"ui-test-emb"' \
  && ok "added row visible in /ui/api/allowlist (live refresh)" \
  || bad "added row missing after /ui POST"
uic 200 POST "/ui/api/allowlist/default" "$ACSRF" \
  '{"module":"textEmbedding","id":"ui-test-emb"}'
uic 400 POST "/ui/api/allowlist/rm" "$ACSRF" \
  '{"module":"NOPE","id":"x"}'                      # invalid body
uic 404 POST "/ui/api/allowlist/rm" "$ACSRF" \
  '{"module":"textEmbedding","id":"never-was"}'     # not present
# rotate the default OFF this row first (current row IS default; rm of
# a default leaves the slot in the documented "next add wins" state,
# but here we just want a clean rm — restore the boot-seeded sibling
# as default before removing the test row).
uic 200 POST "/ui/api/allowlist/default" "$ACSRF" \
  '{"module":"textEmbedding","id":"athena-embedding-default"}'
uic 200 POST "/ui/api/allowlist/rm" "$ACSRF" \
  '{"module":"textEmbedding","id":"ui-test-emb"}'
# member: nav/page/mutation all gated pre-handler (∌ daemonAdmin)
AMC="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" \
  "$B/ui/allowlist")"
[ "$AMC" = 303 ] && ok "member /ui/allowlist → 303 (∌ daemonAdmin)" \
  || bad "member /ui/allowlist → $AMC (want 303)"
AMM="$(curl -s -o /dev/null -w '%{http_code}' -b "$UIJAR_A" -X POST \
  -H 'X-CSRF-Token: x' -H 'Content-Type: application/json' \
  -d '{"module":"llm","id":"x"}' "$B/ui/api/allowlist")"
[ "$AMM" = 303 ] && ok "member allowlist add → 303 (gated pre-handler)" \
  || bad "member allowlist add → $AMM (want 303)"

echo
echo "== phase 2.9: usage accounting — non-zero token counts (M27.1) =="
# The OpenAI `usage` object must report REAL token counts (was a
# hardcoded {0,0,0}). Under --engine stub the counts are synthesized
# from whitespace tokenization, so they're deterministic and non-zero.
# M43.2 — phase 2.7's /api/admin/stop unloaded the LLM; re-warm so the
# first body-parsed inference call doesn't trip on the cold-load 503.
warm "$ADMIN_TOK"
intfield() { # JSON FIELD  → echoes the integer value (or empty)
  printf '%s' "$1" | grep -o "\"$2\":[0-9]*" | grep -o '[0-9]*' | head -1
}
CHATBODY="$(curl -s -X POST \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$CHAT" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
CP="$(intfield "$CHATBODY" prompt_tokens)"
CC="$(intfield "$CHATBODY" completion_tokens)"
CT="$(intfield "$CHATBODY" total_tokens)"
{ [ -n "$CP" ] && [ "$CP" -gt 0 ]; } \
  && ok "chat usage.prompt_tokens = $CP (>0)" \
  || bad "chat usage.prompt_tokens not >0 ($CHATBODY)"
{ [ -n "$CC" ] && [ "$CC" -gt 0 ]; } \
  && ok "chat usage.completion_tokens = $CC (>0)" \
  || bad "chat usage.completion_tokens not >0 ($CHATBODY)"
{ [ -n "$CT" ] && [ "$CT" -eq $((CP + CC)) ]; } \
  && ok "chat usage.total_tokens = $CT (= prompt+completion)" \
  || bad "chat usage.total_tokens != prompt+completion ($CHATBODY)"

# M39: the `model` field now SELECTS among the configured embedding set
# instead of being an ignored false-friend echo. Omit it ⇒ the daemon's
# default model (here the stub's lone configured id). An id outside the
# set is a 400 `model_not_available` (asserted just below) — never a
# silent wrong-model fallback.
EMBBODY="$(curl -s -X POST \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello world","another sentence here"]}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
EP="$(intfield "$EMBBODY" prompt_tokens)"
ET="$(intfield "$EMBBODY" total_tokens)"
{ [ -n "$EP" ] && [ "$EP" -gt 0 ]; } \
  && ok "embeddings usage.prompt_tokens = $EP (>0)" \
  || bad "embeddings usage.prompt_tokens not >0 ($EMBBODY)"
{ [ -n "$ET" ] && [ "$ET" -eq "$EP" ]; } \
  && ok "embeddings usage.total_tokens = $ET (= prompt; no completion)" \
  || bad "embeddings usage.total_tokens != prompt_tokens ($EMBBODY)"
# M39: an unknown model id is refused (400 model_not_available) under the
# stub too — selection/allowlist parity is engine-independent.
ENB="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nope/not-loaded","input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
echo "$ENB" | grep -q 'model_not_available' \
  && ok "embeddings unknown model ⇒ model_not_available" \
  || bad "embeddings unknown model not refused ($ENB)"
code 400 POST /v1/embeddings "$ALICE_TOK" \
  '{"model":"nope/not-loaded","input":"hi"}'

# The revived global metrics counter must reflect that work (was dead).
# /metrics is content-negotiated (M37) — ask for JSON explicitly.
MET="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  -H "Accept: application/json" \
  "http://127.0.0.1:$PORT/metrics")"
MT="$(intfield "$MET" llmTokens)"
{ [ -n "$MT" ] && [ "$MT" -gt 0 ]; } \
  && ok "/metrics llmTokens = $MT (>0, addTokens revived)" \
  || bad "/metrics llmTokens not >0 ($MET)"

echo
echo "== phase 2.10: per-principal usage counters persist (M27.2) =="
# Token usage is metered into AthenaStore keyed by the caller's auth
# principal (u:<user> for a managed token). Read the DB directly to
# prove independent per-principal rows that survive in SQLite.
DB="$D/athena.sqlite"
usagecol() { # PRINCIPAL COLUMN → integer (0 when no row yet)
  sqlite3 "$DB" \
    "SELECT COALESCE((SELECT $2 FROM usage_counters \
       WHERE principal='$1'),0);"
}
A0="$(usagecol 'u:alice' requests)"
AP0="$(usagecol 'u:alice' prompt_tokens)"
B0="$(usagecol 'u:bob' requests)"
for _ in 1 2; do
  curl -s -o /dev/null -X POST -H "Authorization: Bearer $ALICE_TOK" \
    -H 'Content-Type: application/json' -d "$CHAT" \
    "http://127.0.0.1:$PORT/v1/chat/completions"
done
curl -s -o /dev/null -X POST -H "Authorization: Bearer $BOB_TOK" \
  -H 'Content-Type: application/json' -d "$CHAT" \
  "http://127.0.0.1:$PORT/v1/chat/completions"
A1="$(usagecol 'u:alice' requests)"
AP1="$(usagecol 'u:alice' prompt_tokens)"
B1="$(usagecol 'u:bob' requests)"
# Exactly +2 for alice (not +3) proves bob's request did NOT touch
# alice's row — i.e. counters are keyed per principal.
[ "$A1" -eq $((A0 + 2)) ] \
  && ok "alice requests +2 exactly ($A0 → $A1)" \
  || bad "alice requests not +2 ($A0 → $A1)"
[ "$AP1" -gt "$AP0" ] \
  && ok "alice prompt_tokens accumulated ($AP0 → $AP1)" \
  || bad "alice prompt_tokens did not grow ($AP0 → $AP1)"
[ "$B1" -eq $((B0 + 1)) ] \
  && ok "bob has an independent counter row (+1: $B0 → $B1)" \
  || bad "bob counter not +1 ($B0 → $B1)"
NROWS="$(sqlite3 "$DB" 'SELECT COUNT(*) FROM usage_counters;')"
[ "$NROWS" -ge 2 ] \
  && ok "usage_counters holds ≥2 principal rows ($NROWS)" \
  || bad "usage_counters has <2 rows ($NROWS)"
# D1 (ADR 007 #8): native /api/embed must meter like /v1/embeddings — the
# embed twin previously dropped usage entirely. Prove alice's prompt_tokens
# grow after a native /api/embed call (the embedding model is warm from 2.9).
AEP0="$(usagecol 'u:alice' prompt_tokens)"
curl -s -o /dev/null -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":["native embed metering check"]}' \
  "http://127.0.0.1:$PORT/api/embed"
AEP1="$(usagecol 'u:alice' prompt_tokens)"
[ "$AEP1" -gt "$AEP0" ] \
  && ok "native /api/embed meters alice prompt_tokens ($AEP0 → $AEP1)" \
  || bad "native /api/embed did not meter ($AEP0 → $AEP1)"

echo
echo "== phase 2.11: GET /api/usage owner-scoping + CLI (M27.3) =="
# A member sees only its OWN principal row; an admin sees all;
# readonly is refused (usage is billing-sensitive).
AU="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  "http://127.0.0.1:$PORT/api/usage")"
echo "$AU" | grep -q '"u:alice"' \
  && ok "member sees its own usage row" \
  || bad "member missing own usage row ($AU)"
echo "$AU" | grep -q '"u:bob"' \
  && bad "member LEAKED another principal's row ($AU)" \
  || ok "member does not see other principals"
ADU="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/usage")"
{ echo "$ADU" | grep -q '"u:alice"' \
    && echo "$ADU" | grep -q '"u:bob"'; } \
  && ok "admin sees all principals (alice + bob)" \
  || bad "admin usage missing principals ($ADU)"
code 403 GET /api/usage "$RO_TOK"     # readonly ∌ inference
code 401 GET /api/usage ""            # no token
# Local CLI overload: a loopback --host reads the store directly.
LU="$("$ATHENA" usage --data-dir "$D" 2>/dev/null)"
echo "$LU" | grep -q 'u:alice' \
  && ok "athena usage (local) renders the store" \
  || bad "athena usage local missing u:alice ($LU)"

echo
echo "== phase 2.12: streamed usage — stream_options.include_usage (M27.4) =="
SREQ='{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}'
SBODY="$(curl -s -N -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$SREQ" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$SBODY" | grep -q '"usage"' \
  && ok "include_usage emits a terminal usage chunk" \
  || bad "no usage chunk with include_usage ($SBODY)"
echo "$SBODY" | grep -Eq '"total_tokens":[1-9]' \
  && ok "streamed usage carries non-zero total_tokens" \
  || bad "streamed usage total_tokens not >0 ($SBODY)"
echo "$SBODY" | grep -q '\[DONE\]' \
  && ok "stream still terminates with [DONE]" \
  || bad "stream missing [DONE] ($SBODY)"
# Without opting in, a streamed response carries NO usage object.
NREQ='{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"hi"}],"stream":true}'
NBODY="$(curl -s -N -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$NREQ" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$NBODY" | grep -q '"usage"' \
  && bad "stream without opt-in leaked usage ($NBODY)" \
  || ok "no usage chunk when not requested"
# Streamed requests are metered too (was a gap: M27.1 metered sync only).
SB="$(usagecol 'u:alice' requests)"
curl -s -N -o /dev/null -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' -d "$NREQ" \
  "http://127.0.0.1:$PORT/v1/chat/completions"
SA="$(usagecol 'u:alice' requests)"
[ "$SA" -gt "$SB" ] \
  && ok "streamed request is metered ($SB → $SA)" \
  || bad "streamed request not metered ($SB → $SA)"

echo
echo "== phase 2.13: OpenAPI spec discovery /openapi.json (M32.1) =="
# Served unauthenticated (discovery), even while RBAC is enabled, with the
# running daemon's version stamped into info.version.
code 200 GET /openapi.json ""                            # public, no token
SPEC="$(curl -s "http://127.0.0.1:$PORT/openapi.json")"
# M45.5: bash substring match (no pipeline) — the prior `echo "$SPEC"
# | grep -q ...` pattern raced with SIGPIPE under `set -o pipefail`
# once the embedded spec grew past ~64KB (echo gets SIGPIPE the moment
# grep -q matches and exits; pipefail propagates 141, sinking the
# whole pipeline). Bash globbing is in-process — no race.
VER="$("$ATHENA" --version)"
[[ "$SPEC" == *'"openapi": "3.0.3"'* ]] \
  && ok "openapi 3.0.3 document" || bad "not an openapi 3.0.3 doc"
[[ "$SPEC" == *"\"version\": \"$VER\""* ]] \
  && ok "info.version matches binary ($VER)" \
  || bad "info.version != binary version $VER"
[[ "$SPEC" == *'"bearerAuth"'* ]] \
  && ok "bearerAuth security scheme present" || bad "no bearerAuth scheme"
[[ "$SPEC" == *'"/v1/chat/completions"'* ]] \
  && ok "documents /v1/chat/completions" || bad "missing chat path"
[[ "$SPEC" == *'"error"'* ]] \
  && ok "documents the error envelope" || bad "missing error shape"

echo
echo "== phase 2.14: OpenAPI drift-guard — every route documented (M32.3) =="
# Fetch the LIVE document, parse it, and assert every /v1, /api, and
# operational route registered in AthenaServer is present (path + method)
# — a new route cannot ship without a matching spec entry. Param segments
# (:id) are normalized to OpenAPI templates ({id}); the /ui browser
# console is intentionally out of the spec's scope and excluded.
SPECF="$D/openapi.json"
curl -s "http://127.0.0.1:$PORT/openapi.json" > "$SPECF"
if python3 - "$SPECF" Sources/athena/Server/AthenaServer.swift <<'PY'
import json, re, sys
spec = json.load(open(sys.argv[1]))               # fetch + parse
if spec.get("openapi") != "3.0.3":
    print("not an openapi 3.0.3 document"); sys.exit(1)
paths = spec.get("paths", {})
src = open(sys.argv[2]).read()
routes = re.findall(r'router\.(get|post|put|delete|patch)\("([^"]+)"', src)
operational = {"/healthz", "/metrics", "/openapi.json"}
HTTP = {"get", "post", "put", "delete", "patch"}
def in_scope(p):
    return p.startswith("/v1/") or p.startswith("/api/") or p in operational
def norm(p):
    return re.sub(r":([A-Za-z_]+)", r"{\1}", p)
live = {(m, norm(p)) for m, p in routes if in_scope(p)}
documented = {(m, path) for path, ops in paths.items() for m in ops if m in HTTP}
missing = sorted(m.upper() + " " + p for m, p in live - documented)   # route, no spec
stale   = sorted(m.upper() + " " + p for m, p in documented - live)   # spec, no route
if missing:
    print("UNDOCUMENTED (route has no spec entry): " + "; ".join(missing)); sys.exit(1)
if stale:
    print("STALE (spec entry has no route): " + "; ".join(stale)); sys.exit(1)
print("parsed live spec: %d routes ↔ %d documented paths, exact match" % (len(live), len(paths)))
PY
then ok "spec ↔ routes exact match (every /v1+/api+operational route documented, no stale entries)"
else bad "OpenAPI drift or parse failure (see line above)"
fi

echo
echo "== phase 3: scoped-token downgrade =="
# boss is an admin USER, but BOSS_SCOPED narrows to member.
code 403 GET  /metrics "$BOSS_SCOPED"                   # member perms only
code 403 GET  /v1/store/stats "$BOSS_SCOPED"
code 200 POST /v1/chat/completions "$BOSS_SCOPED" "$CHAT"  # inference ok

echo
echo "== phase 3.5: native /api inference + admin (M16.1) =="
code 200 POST /api/chat       "$ALICE_TOK" "$CHAT"  # member: inference
code 401 POST /api/chat       "" "$CHAT"            # no token
code 403 POST /api/chat       "$RO_TOK" "$CHAT"     # readonly ∌ infer
code 200 POST /api/embed      "$ALICE_TOK" '{"input":"hi"}'
code 403 POST /api/admin/stop "$ALICE_TOK"          # ∌ daemonAdmin
code 200 POST /api/admin/stop "$ADMIN_TOK"          # admin: daemonAdmin
# M43.2 — the admin/stop unloaded every module; re-warm before the
# body-parsed native-chat assertion below.
warm "$ADMIN_TOK"
# Ollama is GONE — deleted routes 404 (auth passes; no route).
code 404 GET  /api/tags       "$ALICE_TOK"
code 404 GET  /api/version    "$ALICE_TOK"
code 404 POST /api/generate   "$ALICE_TOK" "$CHAT"
code 404 POST /api/embeddings "$ALICE_TOK" '{"prompt":"hi"}'
code 404 POST /api/stop       "$ADMIN_TOK"          # renamed → /admin
# Native chat shape: has "content", NOT Ollama message/done_reason.
CB="$(curl -s -X POST "http://127.0.0.1:$PORT/api/chat" \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" -d "$CHAT")"
echo "$CB" | grep -q '"content"' \
  && ok "native chat reply carries content" \
  || bad "native chat missing content ($CB)"
if echo "$CB" | grep -q '"done_reason"\|"message"'; then
  bad "native chat still Ollama-shaped ($CB)"
else
  ok "native chat is not Ollama-shaped"
fi

echo
echo "== phase 3.6: model store /api/models (M16.2) =="
# read = model.read (admin, operator, readonly); mutate = model.write
code 200 GET    /api/models            "$ADMIN_TOK"
code 200 GET    /api/models            "$RO_TOK"   # readonly ∋ read
code 403 GET    /api/models            "$ALICE_TOK"  # member ∌ read
code 200 GET    /api/models/fake-model "$RO_TOK"
code 404 GET    /api/models/nope       "$RO_TOK"
code 400 GET    /api/models/bad~name   "$RO_TOK"   # name guard
code 200 GET    /api/models/default    "$RO_TOK"   # read default
code 403 DELETE /api/models/fake-model "$RO_TOK"   # ro ∌ model.write
code 403 PUT    /api/models/default    "$ALICE_TOK" '{"name":"x"}'
code 403 POST   /api/models/copy       "$ALICE_TOK" '{"src":"fake-model","dst":"c1"}'
# model.write mutations (admin) — symlink alias, force, delete
code 200 POST   /api/models/copy       "$ADMIN_TOK" '{"src":"fake-model","dst":"alias1"}'
code 200 GET    /api/models/alias1     "$ADMIN_TOK"
code 409 POST   /api/models/copy       "$ADMIN_TOK" '{"src":"fake-model","dst":"alias1"}'
code 200 POST   /api/models/copy       "$ADMIN_TOK" '{"src":"fake-model","dst":"alias1","force":true}'
code 404 POST   /api/models/copy       "$ADMIN_TOK" '{"src":"ghost","dst":"a2"}'
code 200 DELETE /api/models/alias1     "$ADMIN_TOK"
code 404 DELETE /api/models/alias1     "$ADMIN_TOK"
# TOML-injection guard on the default setter (rejected pre-write).
code 400 PUT    /api/models/default    "$ADMIN_TOK" '{"name":"x\"\nlisten_host=\"0.0.0.0"}'
# list payload shape: contains the seeded model name
ML="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/models")"
echo "$ML" | grep -q '"fake-model"' \
  && ok "model list carries seeded model" \
  || bad "model list missing fake-model ($ML)"

echo
echo "== phase 3.66: per-module model lifecycle /api/models/{load,unload,resident} (M41.1) =="
# Generalizes M39's embedding-only model selection across every module.
# GET resident = model.read; POST load/unload = model.write. RBAC mirrors
# the other native model-store routes.
code 401 GET  /api/models/resident ""              # no token
code 200 GET  /api/models/resident "$ADMIN_TOK"
code 200 GET  /api/models/resident "$RO_TOK"       # readonly ∋ model.read
code 403 GET  /api/models/resident "$ALICE_TOK"    # member ∌ model.read
code 403 POST /api/models/load     "$ALICE_TOK" '{"module":"textEmbedding"}'
code 403 POST /api/models/load     "$RO_TOK"    '{"module":"textEmbedding"}'
code 401 POST /api/models/load     ""           '{"module":"textEmbedding"}'
# Unknown module + unknown id are classified 400s, not 5xx.
code 400 POST /api/models/load     "$ADMIN_TOK" '{"module":"nope"}'
code 400 POST /api/models/load     "$ADMIN_TOK" '{"module":"textEmbedding","id":"nope/not-loaded"}'
# Resident snapshot has all 5 module slots with allowlist + default.
RES="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/models/resident")"
for m in llm textEmbedding transcription diarization speakerEmbedding; do
  echo "$RES" | grep -q "\"module\":\"$m\"" \
    && ok "resident snapshot carries module=$m slot" \
    || bad "resident missing module=$m ($RES)"
done
# Load the embedding slot to its default (omit id ⇒ default).
LDR="$(curl -s -X POST -H "Authorization: Bearer $ADMIN_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"module":"textEmbedding"}' \
  "http://127.0.0.1:$PORT/api/models/load")"
echo "$LDR" | grep -q '"status":"loaded"' \
  && ok "explicit load returned status=loaded ($LDR)" \
  || bad "explicit load missing status=loaded ($LDR)"
echo "$LDR" | grep -q '"module":"textEmbedding"' \
  && ok "explicit load echoes module" \
  || bad "explicit load missing module ($LDR)"
# After explicit load, /api/models/resident shows the resident id.
RES2="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/models/resident")"
echo "$RES2" \
  | python3 -c '
import json, sys
s=json.load(sys.stdin)
slot=[x for x in s["slots"] if x["module"]=="textEmbedding"][0]
r=slot.get("resident")
assert r is not None, "textEmbedding resident is null/absent"
assert r==slot["default"], "resident != default after default-load"
' && ok "textEmbedding resident reflects default load" \
   || bad "textEmbedding resident not reflected ($RES2)"
# Unload that module's slot.
ULR="$(curl -s -X POST -H "Authorization: Bearer $ADMIN_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"module":"textEmbedding"}' \
  "http://127.0.0.1:$PORT/api/models/unload")"
echo "$ULR" | grep -q '"status":"unloaded"' \
  && ok "explicit unload returned status=unloaded ($ULR)" \
  || bad "explicit unload missing status=unloaded ($ULR)"
# After unload, /api/models/resident shows resident=null.
RES3="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/models/resident")"
echo "$RES3" \
  | python3 -c '
import json, sys
s=json.load(sys.stdin)
slot=[x for x in s["slots"] if x["module"]=="textEmbedding"][0]
# absent key (Swift omits nil Codable Optional) and JSON null both count
# as "not resident" here.
assert slot.get("resident") is None, "textEmbedding still resident"
' && ok "textEmbedding slot freed after unload" \
   || bad "textEmbedding slot still resident ($RES3)"
# Unload all modules at once.
code 200 POST /api/models/unload "$ADMIN_TOK" '{}'
# M43.2 — phase 3.66 just emptied every slot. Re-warm so subsequent
# body-parsing inference tests don't trip on cold-load 503.
warm "$ADMIN_TOK"
# Audit trail captures model.load / model.unload mutations (M30 + M41).
AUD="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.load&limit=20")"
echo "$AUD" | grep -q 'model.load' \
  && ok "audit trail recorded model.load (M41 + M30)" \
  || bad "audit trail missing model.load ($AUD)"
AUD2="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.unload&limit=20")"
echo "$AUD2" | grep -q 'model.unload' \
  && ok "audit trail recorded model.unload (M41 + M30)" \
  || bad "audit trail missing model.unload ($AUD2)"

echo
echo "== phase 3.67: per-request LLM model selection (M41.2) =="
# body.model on /v1/chat/completions and /api/chat SELECTS among the
# operator-declared LLM allowlist (--llm-model, repeatable). Default
# daemon was started without --llm-model ⇒ a single-id allowlist
# ("Qwen3.5-27B-4bit-mtp"). The echo + resident snapshot must reflect
# the served model truthfully; an unknown id is a 400.
CHATM="$(curl -s -X POST \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d "$CHAT" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$CHATM" | grep -q '"model":"Qwen3.5-27B-4bit-mtp"' \
  && ok "chat response.model echoes resident id (default allowlist)" \
  || bad "chat response.model not resident id ($CHATM)"
# An LLM id outside the allowlist ⇒ 400 model_not_available.
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"nope/not-in-allowlist","messages":[{"role":"user","content":"hi"}]}'
LBAD="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nope/not-in-allowlist","messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$LBAD" | grep -q 'model_not_available' \
  && ok "chat unknown model ⇒ model_not_available" \
  || bad "chat unknown model not refused ($LBAD)"
# Native chat: same per-request gate.
ABAD="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nope/not-in-allowlist","messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/api/chat")"
echo "$ABAD" | grep -q 'model_not_available' \
  && ok "native chat unknown model ⇒ model_not_available" \
  || bad "native chat unknown model not refused ($ABAD)"

echo
echo "== phase 3.68: per-request audio model selection (M41.3) =="
# A non-empty fake file lets the request pass missing_file and reach
# the rebind gate (which validates `model` ∈ allowlist before any
# audio decode). Bytes are non-WAV but server-side rebind fires before
# the decoder ever runs.
DUMMY_AUDIO="$D/audio.dummy"
printf 'audio-fake-bytes' > "$DUMMY_AUDIO"
# /api/models/resident also lists transcription / diarization /
# speakerEmbedding allowlists (their `--whisper-model` /
# `--diarization-model` / `--speaker-embedding-model` declarations).
# An unknown `model` form-field on any audio endpoint ⇒ 400
# model_not_available (no on-request HF download).
RES4="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/models/resident")"
echo "$RES4" \
  | python3 -c '
import json, sys
s=json.load(sys.stdin)
have = {x["module"]: x for x in s["slots"]}
for m in ["transcription","diarization","speakerEmbedding"]:
    slot=have[m]
    assert slot["allowed"], f"{m} allowed empty"
    assert slot["default"], f"{m} default missing"
' && ok "resident snapshot exposes whisper/diarization/speaker allowlists" \
   || bad "audio allowlists missing in resident ($RES4)"
# Forging an audio request with a tiny fake-file part is enough to
# reach the rebind gate; the model_not_available 400 fires before any
# decode. Use plain multipart with a 4-byte file body.
T_BAD=$(curl -s -o /tmp/abad.json -w '%{http_code}' \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -F "file=@$DUMMY_AUDIO;filename=x.wav" \
  -F 'model=nope/not-in-allowlist' \
  "http://127.0.0.1:$PORT/v1/audio/transcriptions")
[ "$T_BAD" = "400" ] \
  && grep -q 'model_not_available' /tmp/abad.json \
  && ok "transcription unknown model ⇒ 400 model_not_available" \
  || bad "transcription unknown model not refused ($T_BAD: $(cat /tmp/abad.json))"
D_BAD=$(curl -s -o /tmp/dbad.json -w '%{http_code}' \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -F "file=@$DUMMY_AUDIO;filename=x.wav" \
  -F 'model=nope/not-in-allowlist' \
  "http://127.0.0.1:$PORT/v1/audio/diarizations")
[ "$D_BAD" = "400" ] \
  && grep -q 'model_not_available' /tmp/dbad.json \
  && ok "diarization unknown model ⇒ 400 model_not_available" \
  || bad "diarization unknown model not refused ($D_BAD: $(cat /tmp/dbad.json))"
# C1 (ADR 013 #3): response_format=diarized_json IMPLIES diarization (no
# `diarize` flag) and returns the verbose envelope, NOT the bare {text} default.
# No model= ⇒ default transcription model (skips the rebind gate). Under the stub
# the speaker stays unlabeled (zero-width turn) — label correctness is real-model
# RUNBOOK — but "recognized + ran the diarization tenant" is deterministic.
DJ=$(curl -s -o /tmp/dj.json -w '%{http_code}' \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -F "file=@$DUMMY_AUDIO;filename=x.wav" \
  -F 'response_format=diarized_json' \
  "http://127.0.0.1:$PORT/v1/audio/transcriptions")
{ [ "$DJ" = "200" ] && grep -q '"task"' /tmp/dj.json && grep -q '"segments"' /tmp/dj.json; } \
  && ok "diarized_json returns the verbose diarized envelope (not bare {text})" \
  || bad "diarized_json not wired ($DJ: $(cat /tmp/dj.json))"
# Video diarization is deferred (ADR 022): diarized_json on video ⇒ 501, like
# diarize=true. The 501 fires before decode, so the dummy file is fine.
VDJ=$(curl -s -o /tmp/vdj.json -w '%{http_code}' \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -F "file=@$DUMMY_AUDIO;filename=x.mp4" \
  -F 'response_format=diarized_json' \
  "http://127.0.0.1:$PORT/v1/video/transcriptions")
[ "$VDJ" = "501" ] \
  && ok "video diarized_json ⇒ 501 (video diarization not wired)" \
  || bad "video diarized_json not 501 ($VDJ: $(cat /tmp/vdj.json))"
S_BAD=$(curl -s -o /tmp/sbad.json -w '%{http_code}' \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -F "file=@$DUMMY_AUDIO;filename=x.wav" \
  -F 'model=nope/not-in-allowlist' \
  "http://127.0.0.1:$PORT/v1/audio/embeddings")
[ "$S_BAD" = "400" ] \
  && grep -q 'model_not_available' /tmp/sbad.json \
  && ok "speaker-embedding unknown model ⇒ 400 model_not_available" \
  || bad "speaker-embedding unknown model not refused ($S_BAD: $(cat /tmp/sbad.json))"

echo
echo "== phase 3.685: persistent allowlist via /api/models/allow (M42) =="
# Read = model.read; mutations = model.write. RBAC mirrors the rest
# of /api/models/*. Add/remove/default round-trip + the running
# module's allowlist refreshes in place (so the next inference
# validates against the new set without a restart).
code 401 GET    /api/models/allow ""
code 200 GET    /api/models/allow "$ADMIN_TOK"
code 200 GET    /api/models/allow "$RO_TOK"        # readonly ∋ read
code 403 POST   /api/models/allow "$ALICE_TOK" '{"module":"textEmbedding","id":"athena-embedding-third"}'
code 403 POST   /api/models/allow "$RO_TOK"    '{"module":"textEmbedding","id":"athena-embedding-third"}'
# Add a third id to the textEmbedding allowlist.
code 200 POST   /api/models/allow "$ADMIN_TOK" '{"module":"textEmbedding","id":"athena-embedding-third"}'
ALL="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/models/allow?module=textEmbedding")"
echo "$ALL" | grep -q '"id":"athena-embedding-third"' \
  && ok "allow add: row visible in /api/models/allow" \
  || bad "allow add: row missing ($ALL)"
# Live refresh: the next /v1/embeddings request with the new id rebinds
# successfully (would have been 400 model_not_available pre-mutation).
EOK="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"athena-embedding-third","input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
echo "$EOK" | grep -q '"model":"athena-embedding-third"' \
  && ok "live refresh: new id served without daemon restart" \
  || bad "live refresh: new id not served ($EOK)"
# Mark the new id as default; subsequent default-request must serve it.
code 200 PUT    /api/models/allow/default "$ADMIN_TOK" '{"module":"textEmbedding","id":"athena-embedding-third"}'
EDEF="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
echo "$EDEF" | grep -q '"model":"athena-embedding-third"' \
  && ok "allow default: nil-model request serves the new default" \
  || bad "allow default: default did not change ($EDEF)"
# Remove the new default; the resident slot is invalidated and the
# next nil-model request serves whichever row now sits at position 0.
code 200 DELETE "/api/models/allow?module=textEmbedding&id=athena-embedding-third" "$ADMIN_TOK"
EREM="$(curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
echo "$EREM" | grep -q '"model":"athena-embedding-third"' \
  && bad "allow rm: removed id still resident ($EREM)" \
  || ok "allow rm: resident slot rotated to a remaining row"
# A subsequent inference with the removed id is now a 400.
code 400 POST /v1/embeddings "$ALICE_TOK" \
  '{"model":"athena-embedding-third","input":"hi"}'
# Audit captures the three mutations.
AUDA="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.allow.add&limit=20")"
echo "$AUDA" | grep -q 'model.allow.add' \
  && ok "audit recorded model.allow.add" \
  || bad "audit missing model.allow.add ($AUDA)"
AUDD="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.allow.default&limit=20")"
echo "$AUDD" | grep -q 'model.allow.default' \
  && ok "audit recorded model.allow.default" \
  || bad "audit missing model.allow.default ($AUDD)"
AUDR="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.allow.rm&limit=20")"
echo "$AUDR" | grep -q 'model.allow.rm' \
  && ok "audit recorded model.allow.rm" \
  || bad "audit missing model.allow.rm ($AUDR)"

echo
echo "== phase 3.686: governor truth + /healthz signals (M43.1) =="
# Three things in one phase:
#   1. /healthz carries inflight, queueDepth, lastRequestAt (new fields).
#   2. After serving a request, lastRequestAt > 0.
#   3. Removing the resident id from the allowlist used to leave the
#      governor at state=loaded with stale residentBytes (the lying-
#      /healthz symptom). Post-M43.1 refreshAllowlist reconciles the
#      governor when the module drops its container, so state goes to
#      unloaded and residentBytes goes to 0.
# Force the textEmbedding slot loaded so the post-drop assertion is
# meaningful.
curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings" >/dev/null
HZ1="$(curl -s "http://127.0.0.1:$PORT/healthz")"
echo "$HZ1" | python3 -c '
import json, sys
h = json.loads(sys.stdin.read())
for k in ("inflight", "queueDepth", "lastRequestAt"):
    if k not in h:
        sys.exit("missing:" + k)
if h["lastRequestAt"] <= 0:
    sys.exit("lastRequestAt=0")
te = [m for m in h["modules"] if m["id"] == "textEmbedding"][0]
if te["state"] != "loaded" or te["residentBytes"] <= 0:
    sys.exit("textEmbedding not loaded:" + json.dumps(te))
# ADR 023 G3 — per-module measured-vs-estimate flag is surfaced.
if "measured" not in te:
    sys.exit("missing measured field:" + json.dumps(te))
' >/dev/null \
  && ok "/healthz adds inflight/queueDepth/lastRequestAt + loaded state" \
  || bad "/healthz missing M43.1 fields or textEmbedding not loaded ($HZ1)"
# ADR 023 G1: /healthz exposes the serve-path MLX cache bound so an operator
# can confirm the reclaimable buffer pool is bounded (mlxCacheBytes plateaus ≤ it).
echo "$HZ1" | python3 -c '
import json, sys
h = json.loads(sys.stdin.read())
if "mlxCacheLimitBytes" not in h:
    sys.exit("missing mlxCacheLimitBytes")
' >/dev/null \
  && ok "/healthz exposes mlxCacheLimitBytes (ADR 023 G1 cache bound)" \
  || bad "/healthz missing mlxCacheLimitBytes ($HZ1)"
# Add a second id and make it default so the next nil-model rebinds the
# resident slot onto it.
code 200 POST /api/models/allow "$ADMIN_TOK" \
  '{"module":"textEmbedding","id":"athena-embedding-m43"}'
code 200 PUT  /api/models/allow/default "$ADMIN_TOK" \
  '{"module":"textEmbedding","id":"athena-embedding-m43"}'
curl -s -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings" >/dev/null
# Remove the now-resident id. Pre-M43.1 the module nils its container
# directly and the governor never sees it; /healthz then lies. Post-fix
# refreshAllowlist calls governor.releaseSlot when residentBytes==0.
code 200 DELETE \
  "/api/models/allow?module=textEmbedding&id=athena-embedding-m43" \
  "$ADMIN_TOK"
HZ2="$(curl -s "http://127.0.0.1:$PORT/healthz")"
echo "$HZ2" | python3 -c '
import json, sys
h = json.loads(sys.stdin.read())
te = [m for m in h["modules"] if m["id"] == "textEmbedding"][0]
if te["state"] != "unloaded" or te["residentBytes"] != 0:
    sys.exit("textEmbedding lies post-drop:" + json.dumps(te))
' >/dev/null \
  && ok "governor reconciled on allowlist drop (state=unloaded, residentBytes=0)" \
  || bad "governor lies after allowlist drop ($HZ2)"
# M46.5 — the unload path that fired here was an allowlist drop,
# classified as `operator_unload`; the post-drop snapshot must
# expose that reason on the textEmbedding slot.
echo "$HZ2" | python3 -c '
import json, sys
h = json.loads(sys.stdin.read())
te = [m for m in h["modules"] if m["id"] == "textEmbedding"][0]
r = te.get("unloadedReason")
if r != "operator_unload":
    sys.exit("expected operator_unload, got " + repr(r))
' >/dev/null \
  && ok "unloadedReason=operator_unload exposed for allowlist drop" \
  || bad "unloadedReason missing/wrong after allowlist drop ($HZ2)"

echo
echo "== phase 3.687: cold-load blocks-until-ready → 200 (ADR 015) + allowlist warn + init =="
# Phase 3.686 ended with textEmbedding in state=unloaded after the
# allowlist drop. ADR 015 (v0.10.164) narrowed M43.2's 503: a request for a
# non-resident-but-ON-DISK model now BLOCKS until the (bounded, local) load
# finishes and returns 200 — peer-runner ergonomics for arbitrary OpenAI
# clients. 503 + Retry-After is now reserved for the download (`pulling`) and
# `cold_load_wait_secs` timeout cases, neither of which the stub reproduces.
CL_HEADERS="$(curl -s -D - -o /dev/null -X POST \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
echo "$CL_HEADERS" | grep -qi '^HTTP/.* 200' \
  && ok "cold-load blocks until ready then serves 200 (ADR 015)" \
  || bad "cold-load did not block-then-200 ($CL_HEADERS)"
CL_BODY="$(curl -s -X POST \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
# The second call may STILL be .loading (race with the detached task)
# OR already .loaded — poll up to ~2s for the warm state so the rest of
# the e2e doesn't trip on a cold slot. Stub loads in microseconds so
# this almost always resolves in one extra iteration.
for _ in $(seq 1 20); do
  RC="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $ALICE_TOK" \
    -H 'Content-Type: application/json' \
    -d '{"input":"hi"}' \
    "http://127.0.0.1:$PORT/v1/embeddings")"
  [ "$RC" = "200" ] && break
  sleep 0.1
done
[ "$RC" = "200" ] \
  && ok "cold-load completes detached and serves on retry" \
  || bad "cold-load never finished (last code $RC)"

# `athena allowlist add` warns on stderr when the LLM id isn't in the
# local model store. The store on D has no entries, so any add should
# trigger the warning. Best-effort: a 401/network error is silent (the
# add itself already succeeded).
WARN_OUT="$("$ATHENA" allowlist add --module llm --id ghost/not-in-store \
  --host 127.0.0.1 --port $PORT --key "$ADMIN_TOK" 2>&1 >/dev/null || true)"
echo "$WARN_OUT" | grep -q "not in the local model store" \
  && ok "allowlist add warns when llm id absent from /api/models" \
  || bad "allowlist add did not warn ($WARN_OUT)"
# Clean up the test row so subsequent phases see a stable LLM allowlist.
"$ATHENA" allowlist rm --module llm --id ghost/not-in-store \
  --host 127.0.0.1 --port $PORT --key "$ADMIN_TOK" >/dev/null 2>&1 || true

# `athena init --from-allowlist` against an EMPTY data-dir reports the
# no-rows message instead of pulling the compiled-in defaults. Validates
# the flag is wired + the DB-read path opens cleanly.
EMPTY2="$(mktemp -d)"
INIT_OUT="$("$ATHENA" init --from-allowlist --data-dir "$EMPTY2" 2>&1 || true)"
echo "$INIT_OUT" | grep -q "no allowlist rows" \
  && ok "init --from-allowlist reports no-rows on empty data-dir" \
  || bad "init --from-allowlist behavior unexpected ($INIT_OUT)"
rm -rf "$EMPTY2"

echo
echo "== phase 3.688: operator-legibility hints (M43.4) =="
# CLI surface for the M43.4 cluster: auth-deny `hint` rendered on
# stderr by the client `fail()` helper, the `athena restart` /
# `athena config set --no-apply` / `athena doctor`/CLI-only-flags
# survey, and the model_not_available remediation when the store is
# empty. None of these need the daemon mutated — we just probe.
ALLOW_FAIL="$("$ATHENA" allowlist list \
  --host 127.0.0.1 --port $PORT --key sk-athena-bogus 2>&1 || true)"
echo "$ALLOW_FAIL" | grep -q "^hint:" \
  && ok "client fail() renders hint: line on stderr from 401" \
  || bad "client did not render auth hint ($ALLOW_FAIL)"
# `athena restart` exists + help mentions bootout/bootstrap (so an
# operator searching `--help` for `kickstart -k` finds the right verb).
RESTART_HELP="$("$ATHENA" restart --help 2>&1 || true)"
echo "$RESTART_HELP" | grep -qi "Re-bootstrap" \
  && ok "athena restart command surface" \
  || bad "athena restart not wired ($RESTART_HELP)"
# `athena config set --no-apply` writes the TOML without trying to
# bootstrap a plist (which would need root + an installed plist). The
# subcommand's help carries the flag.
CSET_HELP="$("$ATHENA" config set --help 2>&1 || true)"
echo "$CSET_HELP" | grep -q -- "--no-apply" \
  && ok "athena config set carries --no-apply flag" \
  || bad "config set missing --no-apply ($CSET_HELP)"
# `athena doctor` lists CLI-only knobs so operators reading athena.toml
# know what's not surfaced there (--prompt-cache-cap-bytes, the per-
# module --*-model seeds).
DOC_OUT="$("$ATHENA" doctor --config deploy/athena.toml \
  --model-store "$MSTORE" 2>&1 || true)"
echo "$DOC_OUT" | grep -q "CLI-only knobs" \
  && ok "doctor reports the CLI-only flag set" \
  || bad "doctor missing CLI-only survey ($DOC_OUT)"
echo "$DOC_OUT" | grep -q -- "--prompt-cache-cap-bytes" \
  && ok "doctor names --prompt-cache-cap-bytes specifically" \
  || bad "doctor missing --prompt-cache-cap-bytes entry"

echo
echo "== phase 3.689: unified-log consolidation (M45.1) =="
# M45.1 dropped the opt-in syslog UDP shipper + the `syslog_remote`
# config knob; logs now live solely in the macOS unified log
# (foreground adds a stderr terminal handler). Verify:
# 1. `config set syslog_remote=…` is rejected as unknown key.
# 2. `config set log_level=…` still works (didn't accidentally drop
#    the surviving knob).
# 3. Launchd plist generation injects `--background` so the daemon
#    drops its stdout sink under launchctl.
TCFG="$(mktemp)"
cp deploy/athena.toml "$TCFG"
CSET_SYSLOG="$("$ATHENA" config set \
  --config "$TCFG" --no-apply \
  syslog_remote 'udp://10.0.0.5:514' 2>&1 || true)"
echo "$CSET_SYSLOG" | grep -q "unknown key 'syslog_remote'" \
  && ok "config set syslog_remote rejected (M45.1 removed knob)" \
  || bad "config set syslog_remote should be unknown ($CSET_SYSLOG)"
CSET_LL="$("$ATHENA" config set \
  --config "$TCFG" --no-apply \
  log_level 'notice' 2>&1 || true)"
if grep -q '^log_level = "notice"' "$TCFG"; then
  ok "config set log_level still functional (terminal-scoped)"
else
  bad "config set log_level didn't update the file ($CSET_LL)"
fi
# `athena load --help` should NOT advertise --background (it's
# launchd-internal; ArgumentParser visibility: .hidden).
LOAD_HELP="$("$ATHENA" load --help 2>&1 || true)"
echo "$LOAD_HELP" | grep -q -- "--background" \
  && bad "--background should be hidden from athena load --help" \
  || ok "--background hidden from athena load --help (launchd internal)"
# `athena load --help` should NOT mention --syslog-remote anymore.
echo "$LOAD_HELP" | grep -q -- "--syslog-remote" \
  && bad "--syslog-remote still appears in athena load --help" \
  || ok "--syslog-remote dropped from athena load --help"
# `athena start --help` should NOT mention --syslog-remote anymore.
START_HELP="$("$ATHENA" start --help 2>&1 || true)"
echo "$START_HELP" | grep -q -- "--syslog-remote" \
  && bad "--syslog-remote still appears in athena start --help" \
  || ok "--syslog-remote dropped from athena start --help"
rm -f "$TCFG"

echo
echo "== phase 3.6891: request-scoped log metadata (M45.3) =="
# M45.3 plumbs req/principal/function through swift-log's
# MetadataProvider via the LogScope TaskLocal set by AuthMiddleware.
# Verify by submitting a queue job and checking the daemon.log line
# carries req=<uuid> + principal=<admin> + function=<swift fn>.
M45_3_JID="$(curl -s -X POST \
  "http://127.0.0.1:$PORT/v1/queue/conversation" \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"m45.3"}]}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
# Give the worker a moment to log `queue job running` / done.
sleep 0.5
# The notice-level emission for THIS job id should carry M45.3
# metadata.
M45_3_LINE="$(grep -E "queue (submit|job running|job done).*id=$M45_3_JID" \
  "$D/daemon.log" | head -1)"
echo "$M45_3_LINE" | grep -q "req=" \
  && ok "M45.3 daemon log carries req= field for $M45_3_JID" \
  || bad "M45.3 daemon log missing req= ($M45_3_LINE)"
echo "$M45_3_LINE" | grep -q "principal=" \
  && ok "M45.3 daemon log carries principal= field" \
  || bad "M45.3 daemon log missing principal= ($M45_3_LINE)"
echo "$M45_3_LINE" | grep -q "function=" \
  && ok "M45.3 daemon log carries function= field" \
  || bad "M45.3 daemon log missing function= ($M45_3_LINE)"

echo
echo '== phase 3.6892: athena logs wraps log show/stream (M45.4) =='
# M45.4 reshaped `athena logs` as an operator-friendly wrapper over
# /usr/bin/log show / log stream. Verify the new flag surface
# (--follow, --since, --category, --debug) and the absence of the
# pre-M45 file-tail surface (--source, --lines).
LOGS_HELP="$("$ATHENA" logs --help 2>&1 || true)"
for flag in "--follow" "--since" "--category" "--debug"; do
  echo "$LOGS_HELP" | grep -q -- "$flag" \
    && ok "athena logs --help advertises $flag" \
    || bad "athena logs --help missing $flag"
done
# The pre-M45 file-tail flags should be gone (-n / --lines, --source).
echo "$LOGS_HELP" | grep -q -- "--source" \
  && bad "--source still appears in athena logs --help (pre-M45 surface)" \
  || ok "--source dropped from athena logs --help"
# `athena logs --offline --since 1m` is the offline escape hatch
# (direct /usr/bin/log shell-out, no daemon). Should exit 0.
"$ATHENA" logs --offline --since 1m >/dev/null 2>&1 \
  && ok "athena logs --offline --since 1m exits 0" \
  || bad "athena logs --offline --since 1m failed"

echo
echo '== phase 3.6893: /api/logs API surface (M45.5) =='
# M45.5: athena logs becomes a client of /api/logs (one-shot JSON)
# and /api/logs/stream (SSE), both gated daemon.admin so they work
# remotely + respect RBAC.
code 401 GET "/api/logs"                                  # no token
code 401 GET "/api/logs/stream"                           # no token
code 403 GET "/api/logs"           "$ALICE_TOK"           # member ∌ daemonAdmin
code 403 GET "/api/logs/stream"    "$ALICE_TOK"
code 200 GET "/api/logs?since=1m"  "$ADMIN_TOK"           # admin: 200
LOGS_JSON="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/logs?since=1m")"
echo "$LOGS_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if isinstance(d.get("logs"), list) else 1)' \
  && ok "/api/logs returns {logs: [...]}" \
  || bad "/api/logs response shape wrong ($LOGS_JSON)"
# `athena logs` defaults to API: with --host 127.0.0.1 + admin key,
# should produce some output (the daemon's been emitting notice logs
# for several phases by now).
LOGS_OUT="$("$ATHENA" logs --host 127.0.0.1 --port $PORT \
  --key "$ADMIN_TOK" --since 5m 2>&1 || true)"
[ -n "$LOGS_OUT" ] && ! echo "$LOGS_OUT" | grep -q '^error:' \
  && ok "athena logs (API mode) produced output" \
  || bad "athena logs (API mode) failed: $(echo "$LOGS_OUT" | head -2)"
# Non-admin invocation → API returns 403; client surfaces as JSON.
LOGS_DENY="$("$ATHENA" logs --host 127.0.0.1 --port $PORT \
  --key "$ALICE_TOK" --since 5m 2>&1 || true)"
echo "$LOGS_DENY" | grep -q -E 'forbidden|insufficient' \
  && ok "athena logs surfaces 403 for non-admin" \
  || bad "athena logs non-admin → unexpected: $(echo "$LOGS_DENY" | head -2)"
# SSE content-delivery is NOT asserted here — `log stream` has
# variable startup latency on macOS (200ms–1.5s observed), and the
# stream-vs-trigger ordering is timing-sensitive enough that any
# bounded window flakes. The 401/403/200 RBAC + content-type checks
# above cover endpoint plumbing; SSE content correctness is verified
# by direct manual testing rather than in the gate.
SSE_HEAD="$(curl -s -D - -o /dev/null --max-time 1 \
  -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/logs/stream" 2>&1 || true)"
echo "$SSE_HEAD" | grep -qi 'content-type:[[:space:]]*text/event-stream' \
  && ok "/api/logs/stream returns text/event-stream" \
  || bad "/api/logs/stream missing event-stream content-type"

echo
echo "== phase 3.69: inference-time rebind audited (M41.4) =="
# An /v1/embeddings request that names a NON-resident allowlist member
# rebinds the slot in place + audits the change (trigger=inference).
# A second request to the SAME model is a no-op rebind and emits NO
# new audit row (audit only on actual resident-id change).
RB_BEFORE="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.rebind&limit=100")"
RB_N_BEFORE="$(echo "$RB_BEFORE" | python3 -c '
import json, sys
print(len(json.load(sys.stdin).get("audit", [])))')"
# Trigger a rebind to the alt member.
curl -s -o /dev/null -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"athena-embedding-alt","input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings"
RB_MID="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.rebind&limit=100")"
RB_N_MID="$(echo "$RB_MID" | python3 -c '
import json, sys
print(len(json.load(sys.stdin).get("audit", [])))')"
[ "$RB_N_MID" -gt "$RB_N_BEFORE" ] \
  && ok "inference-time rebind emitted model.rebind audit row (+$((RB_N_MID-RB_N_BEFORE)))" \
  || bad "no model.rebind audit row after inference rebind ($RB_N_BEFORE → $RB_N_MID)"
echo "$RB_MID" | grep -q 'trigger=inference' \
  && ok "model.rebind detail includes trigger=inference" \
  || bad "model.rebind missing trigger=inference"
# Same request again ⇒ same resident, no new audit.
curl -s -o /dev/null -X POST -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"athena-embedding-alt","input":"hi"}' \
  "http://127.0.0.1:$PORT/v1/embeddings"
RB_AFTER="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/api/audit?action=model.rebind&limit=100")"
RB_N_AFTER="$(echo "$RB_AFTER" | python3 -c '
import json, sys
print(len(json.load(sys.stdin).get("audit", [])))')"
[ "$RB_N_AFTER" -eq "$RB_N_MID" ] \
  && ok "no-op rebind does not audit (same resident, $RB_N_AFTER rows)" \
  || bad "no-op rebind emitted an audit row ($RB_N_MID → $RB_N_AFTER)"

echo
echo "== phase 3.65: OpenAI model discovery /v1/models (M31.1) =="
# Read-only store projection in the OpenAI list/retrieve shape; gated
# model.read (same as the native /api/models reads), NOT the inference
# catch-all — so a member with inference but ∌ model.read is 403.
code 401 GET /v1/models             ""            # no token
code 200 GET /v1/models             "$ADMIN_TOK"
code 200 GET /v1/models             "$RO_TOK"     # readonly ∋ model.read
code 403 GET /v1/models             "$ALICE_TOK"  # member ∌ model.read
code 200 GET /v1/models/fake-model  "$RO_TOK"
code 404 GET /v1/models/nope        "$RO_TOK"
code 400 GET /v1/models/bad~name    "$RO_TOK"     # name guard
# list payload shape: OpenAI envelope + the seeded model + athena owner
OML="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/v1/models")"
echo "$OML" | grep -q '"object":"list"' \
  && ok "/v1/models is an OpenAI list" \
  || bad "/v1/models not a list ($OML)"
echo "$OML" | grep -q '"owned_by":"athena"' \
  && ok "/v1/models owned_by athena" \
  || bad "/v1/models missing owned_by athena ($OML)"
echo "$OML" | grep -q '"id":"fake-model"' \
  && ok "/v1/models carries the seeded model id" \
  || bad "/v1/models missing fake-model id ($OML)"
# retrieve payload shape: a single OpenAI model object
ORET="$(curl -s -H "Authorization: Bearer $RO_TOK" \
  "http://127.0.0.1:$PORT/v1/models/fake-model")"
echo "$ORET" | grep -q '"object":"model"' \
  && ok "/v1/models/:id is an OpenAI model object" \
  || bad "/v1/models/:id wrong object ($ORET)"

echo
echo "== phase 3.7: async model ops via queue (M16.3) =="
# perm: model.write only; AuthMiddleware blocks before any enqueue.
code 403 POST /api/models/pull   "$ALICE_TOK" '{"id":"x/y"}'
code 403 POST /api/models/prune  "$RO_TOK"
code 400 POST /api/models/pull   "$ADMIN_TOK" '{}'   # 'id' required
# Security: the public /v1/queue/:kind route must NOT accept model_*
# (a queue.submit-only caller would otherwise bypass model.write).
code 400 POST /v1/queue/model_pull   "$ALICE_TOK" '{"id":"x/y"}'
code 400 POST /v1/queue/model_prune  "$ALICE_TOK" '{}'
# prune round-trip (offline, deterministic): submit → poll → done.
PJID="$(curl -s -X POST "http://127.0.0.1:$PORT/api/models/prune" \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -H "Content-Type: application/json" -d '{}' \
  | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')"
[ -n "$PJID" ] && ok "prune enqueued ($PJID)" || bad "prune enqueue"
PRES="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
  "http://127.0.0.1:$PORT/v1/queue/$PJID?wait=15")"
echo "$PRES" | grep -q '"status":"done"' \
  && ok "prune job completed" || bad "prune job ($PRES)"
echo "$PRES" | grep -q '"dead"' \
  && ok "prune found the dangling symlink" \
  || bad "prune candidates missing 'dead' ($PRES)"
[ ! -e "$MSTORE/dead" ] \
  && ok "dangling symlink removed" || bad "dead symlink survived"
# pull dispatch + owner-scoped polling (no completion wait — network).
DJID="$(curl -s -X POST "http://127.0.0.1:$PORT/api/models/pull" \
  -H "Authorization: Bearer $ADMIN_TOK" \
  -H "Content-Type: application/json" \
  -d '{"id":"athena-e2e/does-not-exist"}' \
  | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')"
[ -n "$DJID" ] && ok "pull enqueued ($DJID)" || bad "pull enqueue"
code 200 GET "/v1/queue/$DJID" "$ADMIN_TOK"   # owner/admin sees it
code 404 GET "/v1/queue/$DJID" "$BOB_TOK"     # other tenant hidden

echo
echo "== phase 4: cross-tenant queue isolation =="
JID="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/queue/conversation" \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" -d "$CHAT" \
  | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
[ -n "$JID" ] && ok "alice submitted job $JID" || bad "alice submit"
code 200 GET "/v1/queue/$JID" "$ALICE_TOK"              # owner
code 404 GET "/v1/queue/$JID" "$BOB_TOK"                # other tenant
code 200 GET "/v1/queue/$JID" "$ADMIN_TOK"              # admin sees all

echo
echo "== phase 5: last-admin protection =="
# Seeded admins: {admin, boss}. Demote boss ⇒ `admin` is the sole
# admin, and the guard must then refuse to remove/demote it.
"$ATHENA" auth role revoke boss admin --data-dir "$D" >/dev/null 2>&1 \
  && ok "revoke boss admin (2→1 admins) allowed" \
  || bad "revoke boss admin unexpectedly refused"
"$ATHENA" auth user rm admin --data-dir "$D" >/dev/null 2>&1 \
  && bad "deleted the ONLY admin" \
  || ok "delete of sole admin refused"
"$ATHENA" auth role revoke admin admin --data-dir "$D" \
  >/dev/null 2>&1 \
  && bad "revoked admin from the ONLY admin" \
  || ok "revoke admin from sole admin refused"
# Re-grant boss admin → two admins again → removing one is permitted.
"$ATHENA" auth role grant boss admin --data-dir "$D" >/dev/null 2>&1 \
  && ok "re-grant boss admin allowed" \
  || bad "re-grant boss admin refused"
"$ATHENA" auth user rm admin --data-dir "$D" >/dev/null 2>&1 \
  && ok "delete admin allowed once a 2nd admin exists" \
  || bad "delete admin still refused with 2 admins"

stop_daemon

echo
echo "== phase 5.5: CLI seeds merge on every boot (M44.2) =="
# Pre-M44.2 the seed was first-boot-only: editing --*-model on the
# launchd plist and restarting did nothing because the table already
# had rows. Post-M44.2 each boot idempotently INSERTs the seeded ids,
# so the CLI flag means what it says; but it never removes or flips
# the default, so an operator's `allowlist rm` / `allowlist default`
# choices are preserved (modulo the documented trade — a seed flag
# the operator removed comes BACK if the flag is still set).
D5="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$D5" >/dev/null && ok "seed D5 admin" || bad "seed admin"
A5="$("$ATHENA" auth token add --user admin --data-dir "$D5" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
# First boot: seed two LLM models from CLI. The DB table for the llm
# module is empty ⇒ the first seed becomes the default.
start_daemon "$D5" 127.0.0.1 \
  --llm-model fake-model --llm-model alt-llm-id \
  || { echo "d5 failed"; cat "$D/daemon.log"; exit 1; }
# Row-extraction helper: print the {…} object for a given id (order-
# agnostic so this doesn't hinge on JSON key emission order).
row_for() { # JSON ID  → prints the matching {…} object, or empty
  printf '%s' "$1" | tr '{}' '\n\n' | grep "\"id\":\"$2\"" | head -1
}
AL5="$(curl -s -H "Authorization: Bearer $A5" \
  "http://127.0.0.1:$PORT/api/models/allow?module=llm")"
R="$(row_for "$AL5" fake-model)"
[ -n "$R" ] && echo "$R" | grep -q '"default":true' \
  && ok "first boot: fake-model becomes the seeded default" \
  || bad "first boot default missing ($AL5)"
[ -n "$(row_for "$AL5" alt-llm-id)" ] \
  && ok "first boot: alt-llm-id seeded too" \
  || bad "first boot alt-llm-id missing ($AL5)"
# Operator-only edits via /api: add a custom id (operator-x) and
# rotate the default to it; then REMOVE alt-llm-id. Neither write
# touches the boot-seed flags.
curl -s -X POST -H "Authorization: Bearer $A5" \
  -H 'Content-Type: application/json' \
  -d '{"module":"llm","id":"operator-x"}' \
  "http://127.0.0.1:$PORT/api/models/allow" >/dev/null
curl -s -X PUT -H "Authorization: Bearer $A5" \
  -H 'Content-Type: application/json' \
  -d '{"module":"llm","id":"operator-x"}' \
  "http://127.0.0.1:$PORT/api/models/allow/default" >/dev/null
curl -s -o /dev/null -X DELETE -H "Authorization: Bearer $A5" \
  "http://127.0.0.1:$PORT/api/models/allow?module=llm&id=alt-llm-id"
stop_daemon
# Second boot: SAME CLI flags. Behavior contract:
#   - alt-llm-id was rm'd by operator → COMES BACK (CLI flag still set
#     — the M44.2 trade the brief flagged).
#   - operator-x was added by operator → still there (CLI never removes).
#   - default is operator-x (CLI never re-flips the default once the
#     table has rows).
start_daemon "$D5" 127.0.0.1 \
  --llm-model fake-model --llm-model alt-llm-id \
  || { echo "d5 restart failed"; cat "$D/daemon.log"; exit 1; }
AL5b="$(curl -s -H "Authorization: Bearer $A5" \
  "http://127.0.0.1:$PORT/api/models/allow?module=llm")"
[ -n "$(row_for "$AL5b" alt-llm-id)" ] \
  && ok "merge: CLI re-adds operator-removed seed on restart" \
  || bad "alt-llm-id not re-added ($AL5b)"
[ -n "$(row_for "$AL5b" operator-x)" ] \
  && ok "preserve: operator-added id survives restart" \
  || bad "operator-x lost on restart ($AL5b)"
R="$(row_for "$AL5b" operator-x)"
echo "$R" | grep -q '"default":true' \
  && ok "preserve: operator-chosen default survives restart" \
  || bad "default reverted on restart ($AL5b)"
R="$(row_for "$AL5b" fake-model)"
echo "$R" | grep -q '"default":true' \
  && bad "default leaked back to the first seed (regression)" \
  || ok "first seed is NOT re-flipped to default on restart"
# The M43.4 divergence notice must NOT fire — there's no divergence
# anymore (the DB is a superset of the seed by construction).
grep -q "model_allowlist DB differs" "$D/daemon.log" \
  && bad "stale divergence-notice log line emitted" \
  || ok "divergence notice dropped (no longer applicable)"
stop_daemon

echo
echo "== phase 6: fail-safe — no creds + non-loopback ⇒ refuse =="
out="$("$ATHENA" load --engine stub --host 0.0.0.0 --port "$PORT" \
  --data-dir "$EMPTY" 2>&1)"
rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -qi "refus"; then
  ok "refused to start wide-open on 0.0.0.0 (rc=$rc)"
else
  bad "did NOT refuse (rc=$rc): $out"
fi

echo
echo "== phase 7: auth-disabled loopback stays open =="
if start_daemon "$EMPTY" 127.0.0.1; then
  grep -q "auth: DISABLED" "$D/daemon.log" \
    && ok "daemon reports auth DISABLED" || bad "auth-DISABLED log"
  code 200 POST /v1/chat/completions "" "$CHAT"   # open, no token
  stop_daemon
else
  bad "daemon failed to start open on loopback"
  cat "$D/daemon.log"
fi

echo
echo "== phase 8: RBAC admin over HTTP (M16.4) =="
# Isolated DB so create/delete/last-admin can't disturb phases 0–7.
D2="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$D2" >/dev/null && ok "seed D2 admin" || bad "seed admin"
ATHENA_PASSWORD=ropass1234 "$ATHENA" auth user add ro --role readonly \
  --data-dir "$D2" >/dev/null && ok "seed D2 ro" || bad "seed ro"
ATHENA_PASSWORD=mempass12 "$ATHENA" auth user add mem --role member \
  --data-dir "$D2" >/dev/null && ok "seed D2 mem" || bad "seed mem"
t2() {
  "$ATHENA" auth token add --user "$1" --data-dir "$D2" 2>/dev/null \
    | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1
}
A2="$(t2 admin)"; R2="$(t2 ro)"; M2="$(t2 mem)"
start_daemon "$D2" 127.0.0.1 || { echo "d2 failed"; \
  cat "$D/daemon.log"; exit 1; }
# Perm gating (AuthMiddleware): users.read / users.admin / tokens.admin
code 200 GET    /api/users  "$A2"
code 200 GET    /api/users  "$R2"     # readonly ∋ users.read
code 403 GET    /api/users  "$M2"     # member ∌ users.read
code 200 GET    /api/roles  "$R2"
code 403 GET    /api/roles  "$M2"
code 403 POST   /api/users  "$R2" '{"username":"x","password":"abcdefgh"}'
code 403 GET    /api/tokens "$R2"     # tokens.admin only
code 403 GET    /api/tokens "$M2"
code 200 GET    /api/tokens "$A2"
# /api/admin/status (M16.5) — daemon.admin only
code 200 GET    /api/admin/status "$A2"
code 403 GET    /api/admin/status "$R2"   # readonly ∌ daemon.admin
code 403 GET    /api/admin/status "$M2"
AS="$(curl -s -H "Authorization: Bearer $A2" \
  "http://127.0.0.1:$PORT/api/admin/status")"
echo "$AS" | grep -q '"auth_enabled":true' \
  && ok "admin status reports auth posture" \
  || bad "admin status shape ($AS)"
# Functional CRUD round-trip (admin) + fail-closed validation
code 200 POST   /api/users "$A2" '{"username":"e2e1","password":"pw123456","role":"member"}'
code 200 POST   /api/users/e2e1/roles/operator "$A2"
code 400 POST   /api/users/e2e1/roles/notarole "$A2"   # unknown role
code 404 POST   /api/users/ghost/roles/member  "$A2"   # unknown user
code 400 POST   /api/users "$A2" '{"username":"weak","password":"short"}'
code 400 POST   /api/users "$A2" '{"username":"bad","password":"pw123456","role":"nope"}'
code 200 DELETE /api/users/e2e1/roles/operator "$A2"
code 200 DELETE /api/users/e2e1 "$A2"
code 404 DELETE /api/users/e2e1 "$A2"
# Token mint → use → list → delete
CRESP="$(curl -s -X POST "http://127.0.0.1:$PORT/api/tokens" \
  -H "Authorization: Bearer $A2" -H 'Content-Type: application/json' \
  -d '{"user":"mem","label":"e2e"}')"
NT="$(echo "$CRESP" \
  | sed -n 's/.*"token":"\(sk-athena-[^"]*\)".*/\1/p')"
HP="$(echo "$CRESP" | grep -o '"hash_prefix":"[0-9a-f]*"' \
  | sed 's/.*:"//;s/"//')"
[ -n "$NT" ] && ok "minted token via /api/tokens" || bad "api mint"
# H12 (M66.2): the API exposes only a truncated hash prefix (12 hex), never
# the full 64-hex SHA-256 digest.
LT="$(curl -s -H "Authorization: Bearer $A2" \
  "http://127.0.0.1:$PORT/api/tokens" \
  | grep -o '"hash_prefix":"[0-9a-f]*"' | head -1 \
  | sed 's/.*:"//;s/"//')"
[ "${#LT}" = 12 ] \
  && ok "H12: /api/tokens hash_prefix truncated to 12 hex" \
  || bad "H12: hash_prefix not truncated (len ${#LT})"
code 200 POST /v1/chat/completions "$NT" "$CHAT"   # member inference
code 200 DELETE "/api/tokens/$HP" "$A2"            # delete the new one
code 404 DELETE /api/tokens/deadbeef9999 "$A2"
code 400 DELETE /api/tokens/abc "$A2"               # <6 hex
# Last-admin protection over HTTP (D2 has exactly one admin).
code 403 DELETE /api/users/admin "$A2"              # sole admin
code 403 DELETE /api/users/admin/roles/admin "$A2"  # revoke sole admin
# #12 / M43.4 — auth-deny envelopes must NAME a recovery `hint`, for BOTH
# the route-middleware 403 (insufficient permissions) and the in-handler
# RBAC guard (last-admin protection), matching the 401 hint asserted in
# phase 3.688. The CLI client renders error.hint as a `hint:` line.
H12MW="$(curl -s -X GET "http://127.0.0.1:$PORT/api/users" \
  -H "Authorization: Bearer $M2")"
echo "$H12MW" | grep -q '"hint"' \
  && ok "#12: middleware 403 (insufficient perms) carries error.hint" \
  || bad "#12: middleware 403 missing hint ($H12MW)"
H12GUARD="$(curl -s -X DELETE "http://127.0.0.1:$PORT/api/users/admin" \
  -H "Authorization: Bearer $A2")"
echo "$H12GUARD" | grep -q '"hint"' \
  && ok "#12: in-handler deny403 (last admin) carries error.hint" \
  || bad "#12: in-handler deny403 missing hint ($H12GUARD)"

echo
echo "== phase 8.5: audit trail — admin mutations recorded (M30.1) =="
# Every RBAC/admin mutation writes an append-only audit_log row at the
# SHARED handler chokepoint, so BOTH the bearer /api path (phase 8, D2)
# and the cookie /ui path (phase 2.8, D) are captured — each keyed by
# the acting principal. Read the DBs directly to prove the rows landed
# (the second sink, a unified-log `audit` line, is verified manually
# per the M10 unified-log approach — `log show` timing is too flaky for
# the stub gate).
adb2="$D2/athena.sqlite"
arow() { # DB ACTION TARGET RESULT  → count for principal u:admin
  sqlite3 "$1" "SELECT COUNT(*) FROM audit_log \
    WHERE action='$2' AND target='$3' AND result='$4' \
      AND principal='u:admin';"
}
[ "$(arow "$adb2" user.create e2e1 ok)" -ge 1 ] \
  && ok "audit: bearer user.create e2e1 ok" \
  || bad "audit: missing bearer user.create e2e1"
[ "$(arow "$adb2" role.grant e2e1:operator ok)" -ge 1 ] \
  && ok "audit: bearer role.grant e2e1:operator ok" \
  || bad "audit: missing bearer role.grant"
[ "$(arow "$adb2" role.revoke e2e1:operator ok)" -ge 1 ] \
  && ok "audit: bearer role.revoke e2e1:operator ok" \
  || bad "audit: missing bearer role.revoke"
[ "$(arow "$adb2" user.delete e2e1 ok)" -ge 1 ] \
  && ok "audit: bearer user.delete e2e1 ok" \
  || bad "audit: missing bearer user.delete"
[ "$(arow "$adb2" token.create mem ok)" -ge 1 ] \
  && ok "audit: bearer token.create mem ok" \
  || bad "audit: missing bearer token.create"
[ "$(arow "$adb2" token.delete "$HP" ok)" -ge 1 ] \
  && ok "audit: bearer token.delete ok" \
  || bad "audit: missing bearer token.delete"
# Denials are recorded too (security-relevant): last-admin protection.
[ "$(arow "$adb2" user.delete admin denied)" -ge 1 ] \
  && ok "audit: last-admin user.delete denied recorded" \
  || bad "audit: missing denied user.delete admin"
[ "$(arow "$adb2" role.revoke admin:admin denied)" -ge 1 ] \
  && ok "audit: last-admin role.revoke denied recorded" \
  || bad "audit: missing denied role.revoke admin"
# Plain validation 400s change nothing ⇒ NOT audited (no 'weak' row).
NW="$(sqlite3 "$adb2" "SELECT COUNT(*) FROM audit_log \
  WHERE target='weak';")"
[ "$NW" -eq 0 ] \
  && ok "audit: validation 400 not recorded (no 'weak' row)" \
  || bad "audit: validation failure leaked a row ($NW)"
# Cookie /ui path (phase 2.8 on D) records the LOGGED-IN principal.
adb1="$D/athena.sqlite"
[ "$(arow "$adb1" user.create webu ok)" -ge 1 ] \
  && ok "audit: cookie user.create webu ok (u:admin)" \
  || bad "audit: missing cookie user.create webu"
[ "$(arow "$adb1" token.create bob ok)" -ge 1 ] \
  && ok "audit: cookie token.create bob ok (u:admin)" \
  || bad "audit: missing cookie token.create bob"

echo
echo "== phase 8.6: audit read API + CLI (M30.2) =="
# GET /api/audit is admin-only (daemon.admin); readonly + member are
# refused. Filterable by action/principal/since/limit. Pull only.
code 200 GET /api/audit "$A2"
code 403 GET /api/audit "$R2"          # readonly ∌ daemon.admin
code 403 GET /api/audit "$M2"          # member ∌ daemon.admin
AJ="$(curl -s -H "Authorization: Bearer $A2" \
  "http://127.0.0.1:$PORT/api/audit")"
echo "$AJ" | grep -q '"action":"user.create"' \
  && ok "audit API returns recorded actions" \
  || bad "audit API missing user.create ($AJ)"
echo "$AJ" | grep -q '"result":"denied"' \
  && ok "audit API surfaces denied outcomes" \
  || bad "audit API missing denied rows"
AF="$(curl -s -H "Authorization: Bearer $A2" \
  "http://127.0.0.1:$PORT/api/audit?action=user.delete")"
if echo "$AF" | grep -q '"action":"user.delete"' \
   && ! echo "$AF" | grep -q '"action":"user.create"'; then
  ok "audit API action filter narrows results"
else
  bad "audit API action filter leaked other actions"
fi
AP="$(curl -s -H "Authorization: Bearer $A2" \
  "http://127.0.0.1:$PORT/api/audit?principal=u:admin&limit=5")"
echo "$AP" | grep -q '"principal":"u:admin"' \
  && ok "audit API principal filter + limit" \
  || bad "audit API principal filter empty ($AP)"
# Local CLI reads the store directly (loopback default host).
CLIO="$("$ATHENA" audit --data-dir "$D2" 2>/dev/null)"
echo "$CLIO" | grep -q "user.create" \
  && ok "athena audit (local) renders the trail" \
  || bad "athena audit (local) empty ($CLIO)"
CLIF="$("$ATHENA" audit --action user.delete --data-dir "$D2" \
  2>/dev/null)"
if echo "$CLIF" | grep -q "user.delete" \
   && ! echo "$CLIF" | grep -q "user.create"; then
  ok "athena audit --action filters locally"
else
  bad "athena audit --action did not filter ($CLIF)"
fi

stop_daemon

echo
echo "== phase 8.7: audit retention bound (M30.3) =="
# audit_retention_days prunes rows past the window as the trail grows.
# Backdate a row ~10 days old directly in a fresh store, start a daemon
# with --audit-retention-days 1, then trigger ONE mutation: the
# opportunistic prune drops the stale row and keeps the fresh one.
D3="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$D3" >/dev/null && ok "seed D3 admin" || bad "seed D3 admin"
A3="$("$ATHENA" auth token add --user admin --data-dir "$D3" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
adb3="$D3/athena.sqlite"
OLDTS=$(( $(date +%s) - 864000 ))   # now − 10 days
sqlite3 "$adb3" "INSERT INTO audit_log\
(ts,principal,action,target,result,detail) \
VALUES($OLDTS,'u:ancient','user.create','old','ok',NULL);"
PRE="$(sqlite3 "$adb3" \
  "SELECT COUNT(*) FROM audit_log WHERE target='old';")"
[ "$PRE" -eq 1 ] && ok "seeded a 10-day-old audit row" \
  || bad "could not seed old audit row ($PRE)"
start_daemon "$D3" 127.0.0.1 --audit-retention-days 1 \
  || { echo "d3 failed"; cat "$D/daemon.log"; exit 1; }
# One fresh mutation triggers the opportunistic prune.
code 200 POST /api/users "$A3" \
  '{"username":"fresh","password":"pw123456","role":"member"}'
GONE="$(sqlite3 "$adb3" \
  "SELECT COUNT(*) FROM audit_log WHERE target='old';")"
[ "$GONE" -eq 0 ] \
  && ok "10-day-old row pruned past the retention window" \
  || bad "stale audit row survived retention ($GONE)"
FRESH="$(sqlite3 "$adb3" "SELECT COUNT(*) FROM audit_log \
  WHERE target='fresh' AND result='ok';")"
[ "$FRESH" -ge 1 ] \
  && ok "fresh audit row retained within the window" \
  || bad "fresh audit row missing after prune ($FRESH)"
stop_daemon
rm -rf "$D3"; D3=""

echo
echo "== phase 9: remote model verbs via the CLIENT (M17.1) =="
# Prove the portable `athena` (RemoteModels over /api/models) surfaces
# the server's RBAC outcome as the client's EXIT STATUS. The portable
# client always takes the HTTP path (no local daemon off-Apple); the
# macOS overload reuses the SAME RemoteModels code when --host is
# off-box. Build it with:  (cd clients && swift build) — or set
# ATHENA_CLIENT. Skipped (not failed) when absent, so CI without the
# portable build still passes the curl phases.
CLIENT="${ATHENA_CLIENT:-clients/.build/debug/athena}"
clic() { # WANT(0|nz) DESC SUBCMD ARGS...
  local want="$1" desc="$2"
  shift 2
  "$CLIENT" "$@" --host 127.0.0.1 --port "$PORT" \
    >/dev/null 2>&1
  local got=$?
  if [ "$want" = 0 ]; then
    if [ "$got" -eq 0 ]; then ok "$desc (rc=0)"
    else bad "$desc (rc=${got}, want 0)"; fi
  else
    if [ "$got" -ne 0 ]; then ok "$desc (rc=${got} nonzero)"
    else bad "$desc (rc=0, want nonzero)"; fi
  fi
}
if [ ! -x "$CLIENT" ]; then
  echo "  skip (no portable client at $CLIENT — build: cd clients"\
       "&& swift build)"
else
  # Re-seed fake-model: phase 3.7's real prune legitimately removed it
  # (0-byte safetensors ⇒ "truncated" ⇒ broken). Phase 9 needs a model
  # present, independent of earlier phases.
  mkdir -p "$MSTORE/fake-model"
  printf '{"model_type":"test","hidden_size":8}' \
    > "$MSTORE/fake-model/config.json"
  : > "$MSTORE/fake-model/model.safetensors"
  if start_daemon "$D2" 127.0.0.1; then
  # read = model.read (admin, readonly); member lacks model.read => 403
  clic 0  "client list (admin -> model.read)"    list --key "$A2"
  clic 0  "client list (readonly -> model.read)" list --key "$R2"
  clic nz "client list (member !model.read=403)" list --key "$M2"
  clic 0  "client show fake-model (readonly)"  show fake-model --key "$R2"
  clic nz "client show ghost (404 surfaced)"   show ghost --key "$R2"
  clic 0  "client default get (readonly)"       default --key "$R2"
  clic nz "client default set (member !write)"  default zz --key "$M2"
  clic nz "client cp (member !model.write=403)" cp fake-model c9 --key "$M2"
  clic 0  "client cp (admin -> model.write)"    cp fake-model c9 --key "$A2"
  clic 0  "client rm c9 (admin)"                rm c9 --key "$A2"
  clic nz "client rm c9 again (404 surfaced)"   rm c9 --key "$A2"
  # The client must render the server's data, not merely exit 0.
  # Asserted BEFORE the prune submit so the async prune worker can't
  # race-delete the (deliberately broken) seed first.
  LO="$("$CLIENT" list --host 127.0.0.1 --port "$PORT" \
    --key "$A2" 2>/dev/null)"
  echo "$LO" | grep -q 'fake-model' \
    && ok "client list renders the seeded model" \
    || bad "client list missing fake-model ($LO)"
  clic nz "client prune (member !model.write)"  prune --key "$M2"
  clic 0  "client prune submit (admin -> job)"  prune --key "$A2"
  stop_daemon
  else
    bad "daemon failed to start for client phase"
    cat "$D/daemon.log"
  fi
fi

echo
echo "== phase 10: remote RBAC admin via the CLIENT (M17.2) =="
# Same portable client, the /api/users|tokens|roles surface. Asserts
# the server's users.read/users.admin/tokens.admin gating + canGrant
# + last-admin protection is faithfully surfaced as the client's
# EXIT STATUS (no client-side trust).
if [ ! -x "$CLIENT" ]; then
  echo "  skip (no portable client at $CLIENT)"
elif start_daemon "$D2" 127.0.0.1; then
  # read = users.read (admin, readonly); member lacks it => 403
  clic 0  "auth user list (admin)"        auth user list --key "$A2"
  clic 0  "auth user list (readonly)"     auth user list --key "$R2"
  clic nz "auth user list (member=403)"   auth user list --key "$M2"
  clic 0  "auth role list (readonly)"     auth role list --key "$R2"
  clic nz "auth role list (member=403)"   auth role list --key "$M2"
  # users.admin: create/delete/grant (admin only). ADR 005: the password
  # comes from ATHENA_PASSWORD (exported to the client), never argv.
  export ATHENA_PASSWORD=pw12345678
  clic nz "auth user add (readonly=403)"  auth user add e2c \
    --role member --key "$R2"
  clic 0  "auth user add (admin)"         auth user add e2c \
    --role member --key "$A2"
  unset ATHENA_PASSWORD
  clic 0  "auth role grant (admin)"  auth role grant e2c operator --key "$A2"
  clic nz "auth role grant bad role=400" auth role grant e2c nope --key "$A2"
  clic nz "auth role grant ghost=404" auth role grant gh member --key "$A2"
  clic 0  "auth role revoke (admin)" auth role revoke e2c operator --key "$A2"
  clic 0  "auth user rm e2c (admin)"      auth user rm e2c --key "$A2"
  clic nz "auth user rm e2c again=404"    auth user rm e2c --key "$A2"
  # tokens.admin only
  clic nz "auth list tokens (member=403)" auth list --key "$M2"
  clic nz "auth list tokens (readonly=403)" auth list --key "$R2"
  clic 0  "auth list tokens (admin)"      auth list --key "$A2"
  clic nz "auth token add (readonly=403)" auth token add --user mem --key "$R2"
  clic 0  "auth token add (admin)"        auth token add --user mem --key "$A2"
  clic nz "auth rm bogus prefix=404"      auth rm deadbeef9999 --key "$A2"
  clic nz "auth rm short prefix=400"      auth rm abc --key "$A2"
  # last-admin protection over HTTP, surfaced by the client.
  clic nz "auth user rm sole admin=403"   auth user rm admin --key "$A2"
  clic nz "auth role revoke sole admin=403" \
    auth role revoke admin admin --key "$A2"
  # The client must render server data (a positive token delete too).
  UO="$("$CLIENT" auth user list --host 127.0.0.1 --port "$PORT" \
    --key "$A2" 2>/dev/null)"
  echo "$UO" | grep -q 'admin' \
    && ok "client auth user list renders accounts" \
    || bad "client user list missing admin ($UO)"
  HP="$(curl -s -X POST "http://127.0.0.1:$PORT/api/tokens" \
    -H "Authorization: Bearer $A2" -H 'Content-Type: application/json' \
    -d '{"user":"mem","label":"p10"}' \
    | grep -o '"hash_prefix":"[0-9a-f]*"' | sed 's/.*:"//;s/"//')"
  if [ -n "$HP" ]; then
    clic 0 "auth rm real prefix (admin)" auth rm "$HP" --key "$A2"
  else
    bad "could not mint a token to delete"
  fi
  stop_daemon
else
  bad "daemon failed to start for RBAC client phase"
  cat "$D/daemon.log"
fi

echo
echo "== phase 11: client status verb (M17.3) =="
# `athena status` off-box / portable -> GET /api/admin/status,
# daemon.admin-gated. The client must surface admin=200(rc0) vs
# member/readonly=403(rc!=0), and render the posture fields.
if [ ! -x "$CLIENT" ]; then
  echo "  skip (no portable client at $CLIENT)"
elif start_daemon "$D2" 127.0.0.1; then
  clic 0  "client status (admin -> daemon.admin)" status --key "$A2"
  clic nz "client status (readonly=403)"          status --key "$R2"
  clic nz "client status (member=403)"            status --key "$M2"
  SO="$("$CLIENT" status --host 127.0.0.1 --port "$PORT" \
    --key "$A2" 2>/dev/null)"
  echo "$SO" | grep -q '"auth_enabled"' \
    && ok "client status renders the RBAC posture" \
    || bad "client status shape ($SO)"
  stop_daemon
else
  bad "daemon failed to start for status client phase"
  cat "$D/daemon.log"
fi

echo
echo "== phase 12: kv_compression knob — daemon-start contract (M20/M21) =="
# `kv_compression` is resolved ONCE at daemon start (Load.swift),
# independent of the engine — so the stub daemon exercises the real
# accept / fail-closed contract. TriAttention (M21) is token eviction;
# turboquant (M20) is a quant codec; both must be accepted. An unknown
# value must REFUSE to start (no silent fallback to none). Honors this
# script's invariants: --engine stub, loopback, ephemeral data dir.
kv_daemon() { # KVVALUE  -> rc 0 = came up healthy, 1 = process exited
  local kv="$1" dd rc=1
  dd="$(mktemp -d)"
  ATHENA_KV_COMPRESSION="$kv" "$ATHENA" load --engine stub \
    --host 127.0.0.1 --port "$PORT" --data-dir "$dd" \
    --model-store "$MSTORE" > "$D/kv-$kv.log" 2>&1 &
  DPID=$!
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
      rc=0; break
    fi
    if ! kill -0 "$DPID" 2>/dev/null; then rc=1; break; fi
    sleep 0.5
  done
  stop_daemon
  rm -rf "$dd"
  return $rc
}
if kv_daemon triattention; then
  ok "kv_compression=triattention accepted (daemon healthy)"
else
  bad "kv_compression=triattention rejected"; cat "$D/kv-triattention.log"
fi
if kv_daemon turboquant; then
  ok "kv_compression=turboquant accepted (daemon healthy)"
else
  bad "kv_compression=turboquant rejected"; cat "$D/kv-turboquant.log"
fi
if kv_daemon bogus; then
  bad "kv_compression=bogus started (should fail closed, no fallback)"
else
  ok "kv_compression=bogus fail-closed (daemon refused to start)"
fi
if grep -qi "unrecognized kv_compression" "$D/kv-bogus.log"; then
  ok "fail-closed surfaces a clear 'unrecognized kv_compression' error"
else
  bad "fail-closed error message missing/unclear"
  cat "$D/kv-bogus.log"
fi

echo
echo "== phase 13: athena stop — user daemon + system root guidance (M22.1) =="
# 13a: `athena start` brings up a user-context daemon (no install, no
# root) and `athena stop` halts it via the pidfile.
SD="$(mktemp -d)"; SP=17449
# `start` has no --model-store (stub engine loads no model); just bind.
"$ATHENA" start --engine stub --host 127.0.0.1 --port "$SP" \
  --data-dir "$SD" >/dev/null 2>&1
up=0
for _ in $(seq 1 40); do
  if curl -s -o /dev/null "http://127.0.0.1:$SP/healthz"; then up=1; break; fi
  sleep 0.5
done
[ "$up" = 1 ] && ok "athena start brought up a user daemon (no install)" \
  || bad "athena start failed to come up"
SO="$("$ATHENA" stop --data-dir "$SD" 2>&1)"; rc=$?
{ [ $rc -eq 0 ] && echo "$SO" | grep -qi "stopped"; } \
  && ok "athena stop halts the user daemon (pidfile path)" \
  || bad "athena stop rc=$rc: $SO"
sleep 0.5
curl -s -o /dev/null "http://127.0.0.1:$SP/healthz" \
  && bad "daemon still serving after stop" \
  || ok "endpoint down after stop"
rm -rf "$SD"
# 13b: invalid --label is rejected before reaching launchctl.
IL="$("$ATHENA" stop --data-dir "$(mktemp -d)" --label 'bad;rm -rf /' 2>&1)"
echo "$IL" | grep -qi "invalid --label" \
  && ok "stop rejects an injection-shaped --label" \
  || bad "stop did not reject bad label ($IL)"
# 13c: with no user pidfile, stop takes the system path. We are not
# root and cannot install a real /Library/LaunchDaemons plist in CI, so
# assert whichever branch this host is in — both must exit nonzero and
# touch nothing (skip entirely if running as root, where stop WOULD
# bootout a real daemon).
if [ "$(id -u)" -ne 0 ]; then
  SE="$(mktemp -d)"
  SYSOUT="$("$ATHENA" stop --data-dir "$SE" 2>&1)"; rc=$?
  [ $rc -ne 0 ] && ok "stop (no pidfile) exits nonzero" \
    || bad "stop should fail with nothing user-managed (rc=$rc)"
  if [ -f /Library/LaunchDaemons/me.goodolclint.athena.plist ] \
     || launchctl print system/me.goodolclint.athena >/dev/null 2>&1; then
    echo "$SYSOUT" | grep -qi "sudo athena stop" \
      && ok "system daemon present + not root → sudo guidance" \
      || bad "expected sudo guidance ($SYSOUT)"
  else
    echo "$SYSOUT" | grep -qi "no athena daemon to stop" \
      && ok "clean host → clear no-daemon message" \
      || bad "expected no-daemon message ($SYSOUT)"
  fi
  rm -rf "$SE"
else
  echo "  skip system-stop probe (running as root would bootout a real daemon)"
fi

echo
echo "== phase 14: athena init — aux-model pull (help + idempotent) (M22.2) =="
# `init` pulls only the aux modules (no LLM). We don't fetch multi-GB
# weights in CI; instead assert the help text and the idempotent skip
# path by pre-seeding all three aux model dirs so init does no network.
"$ATHENA" init --help 2>&1 \
  | grep -qi "auxiliary" \
  && ok "init --help documents the auxiliary pulls" \
  || bad "init --help missing description"
IS="$(mktemp -d)"
for n in bge-small-en-v1.5 whisper-large-v3-turbo \
         diar_streaming_sortformer_4spk-v2.1-fp16; do
  mkdir -p "$IS/$n"
  printf '{"model_type":"test"}' > "$IS/$n/config.json"
done
IO="$("$ATHENA" init --model-store "$IS" 2>&1)"; rc=$?
[ $rc -eq 0 ] && ok "init all-present exits 0 (idempotent, no network)" \
  || bad "init idempotent rc=$rc: $IO"
SKIPN="$(printf '%s\n' "$IO" | grep -ci "already present")"
[ "$SKIPN" -eq 3 ] && ok "init skipped all 3 pre-seeded aux models" \
  || bad "init skip count = $SKIPN (want 3): $IO"
echo "$IO" | grep -qi "LLM is NOT included" \
  && ok "init states the LLM is not auto-pulled" \
  || bad "init missing the no-LLM note ($IO)"
rm -rf "$IS"

echo
echo "== phase 15: athena install — default-config synthesis (M22.3) =="
# With NO config supplied AND no ./deploy/athena.toml in cwd, install
# must synthesize a documented default (not throw). Run --dry-run from
# an empty temp dir so the dev copy isn't found; an absolute binary
# path is needed since we cd away from the repo. --dry-run returns
# before the root/metallib checks, so this is safe as a normal user.
IT="$(mktemp -d)"
DR="$(cd "$IT" && "$ATHENA_ABS" install --dry-run 2>&1)"; rc=$?
[ $rc -eq 0 ] && ok "install --dry-run (no config) does not throw" \
  || bad "install --dry-run rc=$rc: $DR"
echo "$DR" | grep -qi "synthesized" \
  && ok "dry-run reports synthesized defaults" \
  || bad "dry-run did not mention synthesis ($DR)"
echo "$DR" | grep -q "listen_port = 7447" \
  && ok "synthesized config carries listen_port 7447" \
  || bad "synthesized config missing listen_port ($DR)"
echo "$DR" | grep -q "<key>Label</key>" \
  && ok "dry-run still renders the launchd plist" \
  || bad "dry-run plist missing"
# v0.10.39: the LaunchDaemon's WorkingDirectory MUST be the libexec
# dir so mlx-swift's Bundle.module resolver finds
# `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` via the
# cwd-adjacent candidate chain. cwd=data_dir leaves MLX printing
# "Failed to load the default metallib library not found …" on every
# init at launchd-spawned startup; foreground `athena load` works
# because the operator's shell cwd happens to be the build dir.
echo "$DR" | tr '\n' ' ' \
  | grep -qE '<key>WorkingDirectory</key>[[:space:]]*<string>[^<]*libexec/athena</string>' \
  && ok "launchd WorkingDirectory points at libexec (metallib reachable)" \
  || bad "launchd WorkingDirectory does not target libexec ($DR)"
# v0.10.40: launchd MUST exec `athena load …` directly. The pre-fix
# plist ran `athenad`, which `execv`-ed `athena` with `argv[0]="athena"`
# (bare); under launchd that argv[0]/binary-path discrepancy broke
# Swift's Bundle.main resolution, so mlx-c couldn't find its SwiftPM
# bundle and the daemon failed with "Failed to load the default
# metallib library not found …" on first MLX call. Foreground
# `athena load` was unaffected (shell preserved the full path as
# argv[0]). The plist's ProgramArguments[0] now points at the athena
# binary, with "load" as the first arg.
echo "$DR" | tr '\n' ' ' \
  | grep -qE '<key>ProgramArguments</key>[[:space:]]*<array>[[:space:]]*<string>[^<]*/athena</string>[[:space:]]*<string>load</string>' \
  && ok "launchd exec target = athena (not athenad) + first arg = load" \
  || bad "launchd ProgramArguments not '<athena> load …' ($DR)"
echo "$DR" | tr '\n' ' ' \
  | grep -q 'athenad</string>' \
  && bad "launchd plist still references athenad (regressed)" \
  || ok "launchd plist no longer references athenad"
rm -rf "$IT"
# Explicit --config to a missing path still errors (unchanged behavior).
EC="$("$ATHENA_ABS" install --config /no/such/athena.toml --dry-run 2>&1)"
rc=$?
[ $rc -ne 0 ] \
  && ok "explicit missing --config still errors (unchanged)" \
  || bad "explicit missing --config did not error ($EC)"

echo
echo "== phase 16: in-daemon TLS — HTTPS + fail-closed (M28.1) =="
# TLS is resolved at daemon start (AthenaServer.serverBuilder), engine-
# independent — so the stub daemon exercises the real contract:
#   • both cert+key ⇒ serves HTTPS (handshake + 200 on /healthz);
#   • plaintext HTTP to the TLS port fails (no silent downgrade);
#   • exactly one of cert/key ⇒ refuse to start (fail-closed).
# Honors this script's invariants: --engine stub, loopback, ephemeral
# data dir, port derived from 7447 (17451 here).
TPORT=17451
TLSDIR="$(mktemp -d)"
CERT="$TLSDIR/cert.pem"; KEY="$TLSDIR/key.pem"
if ! command -v openssl >/dev/null 2>&1; then
  echo "  skip TLS phase (openssl not available)"
elif ! openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$KEY" -out "$CERT" -days 1 -subj "/CN=athena-e2e" \
        -addext "subjectAltName=IP:127.0.0.1" \
        >/dev/null 2>&1; then
  echo "  skip TLS phase (self-signed cert generation failed)"
else
  chmod 600 "$KEY"
  # 16a: both keys ⇒ HTTPS comes up; verify against the self-signed CA.
  "$ATHENA" load --engine stub --host 127.0.0.1 --port "$TPORT" \
    --data-dir "$(mktemp -d)" --model-store "$MSTORE" \
    --tls-cert "$CERT" --tls-key "$KEY" \
    > "$D/tls.log" 2>&1 &
  DPID=$!
  up=0
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null --cacert "$CERT" \
         "https://127.0.0.1:$TPORT/healthz"; then up=1; break; fi
    if ! kill -0 "$DPID" 2>/dev/null; then break; fi
    sleep 0.5
  done
  [ "$up" = 1 ] \
    && ok "HTTPS /healthz 200 (cert verified against self-signed CA)" \
    || { bad "TLS daemon did not serve HTTPS"; cat "$D/tls.log"; }
  # 16b: plaintext HTTP to a TLS port must fail (handshake mismatch),
  # never silently downgrade.
  if [ "$up" = 1 ]; then
    curl -s -o /dev/null "http://127.0.0.1:$TPORT/healthz" \
      && bad "plaintext HTTP accepted on the TLS port" \
      || ok "plaintext HTTP to the TLS port is refused (no downgrade)"
  fi
  stop_daemon
  # 16c: fail-closed — cert without key refuses to start, with a clear
  # message; the daemon must NOT come up serving plaintext.
  "$ATHENA" load --engine stub --host 127.0.0.1 --port "$TPORT" \
    --data-dir "$(mktemp -d)" --model-store "$MSTORE" \
    --tls-cert "$CERT" \
    > "$D/tls-incomplete.log" 2>&1 &
  DPID=$!
  half_up=0
  for _ in $(seq 1 16); do
    if curl -s -o /dev/null "http://127.0.0.1:$TPORT/healthz" \
       || curl -s -o /dev/null -k \
            "https://127.0.0.1:$TPORT/healthz"; then half_up=1; break; fi
    if ! kill -0 "$DPID" 2>/dev/null; then break; fi
    sleep 0.5
  done
  stop_daemon
  [ "$half_up" = 0 ] \
    && ok "tls_cert without tls_key fail-closed (daemon refused to start)" \
    || bad "half-configured TLS started anyway (should fail closed)"
  grep -qi "TLS misconfigured" "$D/tls-incomplete.log" \
    && ok "fail-closed surfaces a clear 'TLS misconfigured' error" \
    || { bad "fail-closed message missing"; cat "$D/tls-incomplete.log"; }
fi
rm -rf "$TLSDIR"

echo
echo "== phase 17: athena doctor — TLS posture (M28.2) =="
# `doctor` reads the config and predicts the daemon's TLS behavior
# WITHOUT starting it: both keys + a valid cert ⇒ "TLS: enabled";
# exactly one key ⇒ predicts the fail-closed start refusal; an
# unreadable cert ⇒ FAIL. doctor's overall exit is governed by the
# CI temp env's other checks (model store etc.), so we grep its
# output for the TLS line rather than trust the exit code.
DOCDIR="$(mktemp -d)"
DCERT="$DOCDIR/cert.pem"; DKEY="$DOCDIR/key.pem"
if ! command -v openssl >/dev/null 2>&1; then
  echo "  skip doctor-TLS phase (openssl not available)"
elif ! openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$DKEY" -out "$DCERT" -days 30 -subj "/CN=athena-doctor" \
        >/dev/null 2>&1; then
  echo "  skip doctor-TLS phase (cert generation failed)"
else
  chmod 600 "$DKEY"
  mkcfg() { # CFGPATH  extra-lines…
    local p="$1"; shift
    { echo 'listen_host = "127.0.0.1"'
      echo 'listen_port = 7447'
      echo 'log_dir = "/usr/local/var/log/athena"'
      for l in "$@"; do echo "$l"; done
    } > "$p"
  }
  # 17a: both keys + valid cert ⇒ "TLS: enabled".
  CFG="$DOCDIR/ok.toml"
  mkcfg "$CFG" "tls_cert = \"$DCERT\"" "tls_key = \"$DKEY\""
  DOUT="$("$ATHENA" doctor --config "$CFG" --model-store "$MSTORE" 2>&1)"
  echo "$DOUT" | grep -qi "TLS: enabled" \
    && ok "doctor reports TLS enabled for a valid cert/key pair" \
    || { bad "doctor missing 'TLS: enabled'"; echo "$DOUT" | grep -i tls; }
  # 17b: only tls_cert ⇒ predicts the fail-closed start refusal.
  CFG2="$DOCDIR/half.toml"
  mkcfg "$CFG2" "tls_cert = \"$DCERT\""
  DOUT2="$("$ATHENA" doctor --config "$CFG2" --model-store "$MSTORE" 2>&1)"
  echo "$DOUT2" | grep -qi "only one of tls_cert/tls_key" \
    && ok "doctor flags half-configured TLS as a startup refusal" \
    || { bad "doctor missed half-config TLS"; echo "$DOUT2" | grep -i tls; }
  # 17c: unreadable cert path ⇒ FAIL line.
  CFG3="$DOCDIR/missing.toml"
  mkcfg "$CFG3" "tls_cert = \"$DOCDIR/nope.pem\"" \
                "tls_key = \"$DKEY\""
  DOUT3="$("$ATHENA" doctor --config "$CFG3" --model-store "$MSTORE" 2>&1)"
  echo "$DOUT3" | grep -qi "TLS: cert not readable" \
    && ok "doctor FAILs on an unreadable TLS cert path" \
    || { bad "doctor missed unreadable cert"; echo "$DOUT3" | grep -i tls; }
fi
rm -rf "$DOCDIR"

echo
echo "== phase 18: reverse-proxy guide — contract guard (M28.3) =="
# The blessed proxy recipe must stay in sync with the daemon's real
# constraints. This guards docs/reverse-proxy.md from drift: if the
# loopback port, the 25 MB audio body cap, the streaming-buffering note,
# the Authorization passthrough, or the "seed credentials behind a
# loopback bind" warning ever change, this phase flags the stale doc.
RPG="docs/reverse-proxy.md"
if [ ! -f "$RPG" ]; then
  bad "reverse-proxy guide missing at $RPG"
else
  ok "reverse-proxy guide present ($RPG)"
  grep -q "127.0.0.1:7447" "$RPG" \
    && ok "guide pins the loopback daemon target 127.0.0.1:7447" \
    || bad "guide missing 127.0.0.1:7447 loopback target"
  # 100 MiB audio cap (ADR 017, default max_audio_upload_bytes) — present in
  # both the nginx and Caddy snippets.
  grep -qi "client_max_body_size 100m" "$RPG" \
    && grep -qi "max_size 100MB" "$RPG" \
    && ok "guide sets a 100 MiB body cap (matches audio upload limit)" \
    || bad "guide missing the 100 MiB body-size directive"
  grep -qi "proxy_buffering" "$RPG" \
    && ok "guide disables proxy buffering (SSE/long-poll streaming)" \
    || bad "guide missing the streaming/buffering note"
  grep -q "Authorization" "$RPG" \
    && ok "guide preserves the Authorization header" \
    || bad "guide missing Authorization passthrough note"
  grep -q "athena auth" "$RPG" \
    && ok "guide tells operators to seed auth creds behind loopback" \
    || bad "guide missing the seed-credentials warning"
fi

echo
echo "== phase 19: per-principal rate limiting — 429 + Retry-After (M29.1) =="
# Rate limiting is resolved at daemon start (--rate-limit), keyed by the
# authenticated principal. With auth enabled (the $D realm has users), a
# rate of 1 req/s and a burst of 1 means a principal's FIRST request is
# admitted (200) and the immediate SECOND is throttled (429 +
# Retry-After), while a DIFFERENT principal (its own bucket) and the
# exempt /healthz are unaffected. Same invariants: --engine stub,
# loopback, ephemeral data dir, port 17447.
stop_daemon
start_daemon "$D" 127.0.0.1 --rate-limit 1 --rate-burst 1 --preload \
  --llm-model athena-stub \
  || { echo "rate-limited daemon failed"; cat "$D/daemon.log"; exit 1; }
grep -q "auth: enabled (RBAC" "$D/daemon.log" \
  && ok "rate-limited daemon still enforces auth" \
  || bad "rate-limited daemon auth-enabled log line"
# alice's first request consumes her one token (200)…
code 200 POST /v1/chat/completions "$ALICE_TOK" "$CHAT"
# …the immediate second is throttled. One -i curl captures the status
# line, the Retry-After header, and the error body together (so all three
# assertions read the SAME response, with no extra request to refill).
RL="$(curl -s -i -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" -d "$CHAT" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$RL" | head -1 | grep -q " 429" \
  && ok "second back-to-back request from one principal → 429" \
  || bad "expected 429 on alice's burst ($(echo "$RL" | head -1))"
echo "$RL" | grep -qi "^Retry-After: *[0-9]" \
  && ok "429 carries a numeric Retry-After header" \
  || bad "429 missing/non-numeric Retry-After"
echo "$RL" | grep -q '"type":"rate_limit_error"' \
  && ok "429 body is the standard {error:…} rate_limit_error" \
  || bad "429 body shape unexpected"
# A different principal has its own bucket — bob is NOT throttled by
# alice's exhausted one.
code 200 POST /v1/chat/completions "$BOB_TOK" "$CHAT"
# /healthz is exempt from rate limiting (launchd / monitoring probes).
code 200 GET /healthz "$ALICE_TOK"
stop_daemon

echo
echo "== phase 20: concurrency caps — global + per-principal 429 (M29.2) =="
# Concurrency caps bound IN-FLIGHT request COUNT (orthogonal to the rate
# limit, which bounds requests over time). The stub chat handler streams
# 10 chunks at 15 ms each (~150 ms), reliably holding a slot while a
# second request probes the cap. Two daemon configs isolate the two
# dimensions. --engine stub, loopback, ephemeral data dir.
probe() { # WANT-CODE BEARER  → captures $PROBE (status+headers+body)
  PROBE="$(curl -s -i -H "Authorization: Bearer $2" \
    -H "Content-Type: application/json" -d "$CHAT" \
    "http://127.0.0.1:$PORT/v1/chat/completions")"
  echo "$PROBE" | grep -E "^HTTP/" | tail -1 | grep -q " $1"
}
hold() { # BEARER  → backgrounds a ~150 ms in-flight chat, sets $HPID
  curl -s -o /dev/null -H "Authorization: Bearer $1" \
    -H "Content-Type: application/json" -d "$CHAT" \
    "http://127.0.0.1:$PORT/v1/chat/completions" &
  HPID=$!
  sleep 0.05   # let the holder acquire its slot before we probe
}

# 20a: per-principal cap = 1 (global unlimited). alice's in-flight chat
# holds her one slot, so her concurrent SECOND request is rejected —
# while a DIFFERENT principal (bob) is unaffected.
stop_daemon
start_daemon "$D" 127.0.0.1 --max-concurrency-per-principal 1 \
  --preload --llm-model athena-stub \
  || { echo "concurrency daemon failed"; cat "$D/daemon.log"; exit 1; }
hold "$ALICE_TOK"
probe 429 "$ALICE_TOK" \
  && ok "per-principal cap: alice's concurrent 2nd request → 429" \
  || bad "per-principal cap: expected 429 ($(echo "$PROBE" | head -1))"
echo "$PROBE" | grep -qi "^Retry-After: *[0-9]" \
  && ok "concurrency 429 carries a Retry-After header" \
  || bad "concurrency 429 missing Retry-After"
echo "$PROBE" | grep -q '"code":"concurrency_limit"' \
  && ok "concurrency 429 body uses code concurrency_limit" \
  || bad "concurrency 429 body shape unexpected"
probe 200 "$BOB_TOK" \
  && ok "per-principal cap is per-caller: bob admitted (200)" \
  || bad "bob should be admitted ($(echo "$PROBE" | head -1))"
wait "$HPID" 2>/dev/null
probe 200 "$ALICE_TOK" \
  && ok "slot released after the in-flight request completes (alice 200)" \
  || bad "alice not re-admitted after release ($(echo "$PROBE" | head -1))"
stop_daemon

# 20b: global cap = 1 (per-principal unlimited). One in-flight request
# fills the whole daemon, so a DIFFERENT principal is rejected — proving
# the cap is global, not per-caller.
start_daemon "$D" 127.0.0.1 --max-concurrency 1 --preload \
  --llm-model athena-stub \
  || { echo "global-concurrency daemon failed"; cat "$D/daemon.log"; \
       exit 1; }
hold "$ALICE_TOK"
probe 429 "$BOB_TOK" \
  && ok "global cap: a different principal is rejected while 1 in flight" \
  || bad "global cap: expected 429 for bob ($(echo "$PROBE" | head -1))"
wait "$HPID" 2>/dev/null
stop_daemon

echo
echo "== phase 21: athena doctor — abuse-protection posture (M29.3) =="
# doctor reads the config and PREDICTS the daemon's throttle behavior
# without starting it. It derives data_dir from the config (no
# --data-dir flag), so each test config pins data_dir explicitly to
# control which credentials doctor sees. We grep its output (its overall
# exit is governed by the CI temp env's other checks).
DOC2="$(mktemp -d)"
dcfg() { # CFGPATH HOST DATADIR extra-lines…
  local p="$1" host="$2" dd="$3"; shift 3
  { echo "listen_host = \"$host\""
    echo 'listen_port = 7447'
    echo 'log_dir = "/usr/local/var/log/athena"'
    echo "data_dir = \"$dd\""
    for l in "$@"; do echo "$l"; done
  } > "$p"
}
# 21a: limits set (+ authed data dir) ⇒ doctor reports both.
C1="$DOC2/limits.toml"
dcfg "$C1" 127.0.0.1 "$D" "rate_limit = 10" "rate_burst = 20" \
  "max_concurrency = 8" "max_concurrency_per_principal = 2"
O1="$("$ATHENA" doctor --config "$C1" --model-store "$MSTORE" 2>&1)"
echo "$O1" | grep -qi "rate limiting:.*per principal" \
  && ok "doctor reports configured rate limiting" \
  || { bad "doctor missing rate-limiting line"; echo "$O1" | grep -i rate; }
echo "$O1" | grep -qi "concurrency caps:.*global 8.*per-principal 2" \
  && ok "doctor reports configured concurrency caps" \
  || { bad "doctor missing concurrency-caps line"
       echo "$O1" | grep -i concurrency; }
# 21b: limits set but NO auth creds (empty data dir, loopback) ⇒ warns
# the limits are inert (throttling only applies to authed callers).
ED="$(mktemp -d)"
C1b="$DOC2/limits-noauth.toml"
dcfg "$C1b" 127.0.0.1 "$ED" "rate_limit = 10"
O2="$("$ATHENA" doctor --config "$C1b" --model-store "$MSTORE" 2>&1)"
echo "$O2" | grep -qi "configured but auth is disabled" \
  && ok "doctor warns limits are inert when auth is off" \
  || { bad "doctor missed the auth-off-limits warning"
       echo "$O2" | grep -i auth; }
rm -rf "$ED"
# 21c: NO limits + authed ($D) + non-loopback ⇒ flood warning.
C2="$DOC2/none.toml"
dcfg "$C2" 0.0.0.0 "$D"
O3="$("$ATHENA" doctor --config "$C2" --model-store "$MSTORE" 2>&1)"
echo "$O3" | grep -qi "no rate_limit or max_concurrency on non-loopback" \
  && ok "doctor warns an authed non-loopback bind has no throttle" \
  || { bad "doctor missed the no-throttle warning"
       echo "$O3" | grep -i "abuse\|throttle\|flood"; }
rm -rf "$DOC2"

echo
echo "== phase 22: per-request inference timeout — 504 + truncate (M33.1) =="
# The deadline (--request-timeout-secs) bounds a single generation by
# wall-clock, independent of auth. The stub's canned stream is paced by
# ATHENA_STUB_DELAY_MS (a dev/e2e-only knob): 10 chunks × 200 ms ≈ 2 s, so
# a 1 s timeout fires mid-decode. Same invariants: --engine stub, loopback,
# ephemeral data dir, the authed $D realm (timeout is orthogonal to RBAC).
stop_daemon
export ATHENA_STUB_DELAY_MS=200
start_daemon "$D" 127.0.0.1 --request-timeout-secs 1 --preload \
  --llm-model athena-stub \
  || { echo "timeout daemon failed"; cat "$D/daemon.log"; exit 1; }
# 22a: a sync chat that overruns the 1 s deadline → classified 504.
TO="$(curl -s -i -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" -d "$CHAT" \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$TO" | head -1 | grep -q " 504" \
  && ok "sync generation past the deadline → 504" \
  || bad "expected 504 ($(echo "$TO" | head -1))"
echo "$TO" | grep -q '"code":"inference_timeout"' \
  && ok "504 body uses code inference_timeout" \
  || bad "504 body shape unexpected ($(echo "$TO" | tail -1))"
# 22b: a streamed chat can't be a 504 (200 + headers already sent), so it
# truncates and still closes the wire cleanly with data: [DONE].
SR="$(curl -s -N -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-27B-4bit-mtp","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$SR" | grep -q "data: \[DONE\]" \
  && ok "streamed generation truncates but closes with [DONE]" \
  || bad "streamed timeout did not close cleanly"
# 22b2 (M46.3a): the SAME slow generation under the 1 s daemon deadline,
# but with a per-request `timeout: 30` override on the request body, must
# NOT 504 — the override widens the deadline for THIS call only.
RELAX="$(curl -s -i -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" \
  -d '{"model":"athena-stub","timeout":30,"messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$RELAX" | head -1 | grep -q " 200" \
  && ok "per-request timeout=30 widens past daemon's 1 s cap → 200" \
  || bad "per-request timeout override didn't take effect ($(echo "$RELAX" | head -1))"
# 22b3 (M46.3a): timeout=0 (or negative) disables the deadline for this
# call, same outcome as the relaxed-override case above. Belt-and-braces
# guard so the "0 ⇒ disable" branch stays alive across refactors.
OFF="$(curl -s -i -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" \
  -d '{"model":"athena-stub","timeout":0,"messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$OFF" | head -1 | grep -q " 200" \
  && ok "per-request timeout=0 disables deadline → 200" \
  || bad "per-request timeout=0 didn't disable deadline ($(echo "$OFF" | head -1))"
stop_daemon
# 22c: the SAME slow generation under a generous timeout is NOT killed —
# proves the deadline doesn't false-fire on legitimate work.
start_daemon "$D" 127.0.0.1 --request-timeout-secs 30 --preload \
  --llm-model athena-stub \
  || { echo "generous-timeout daemon failed"; cat "$D/daemon.log"; exit 1; }
code 200 POST /v1/chat/completions "$ALICE_TOK" "$CHAT"
# 22c2 (M46.3a): per-request `timeout: 1` override TIGHTENS the deadline
# below the daemon's generous 30 s cap — the same body that just passed
# at 200 must now 504 with the per-call cap.
TIGHT="$(curl -s -i -H "Authorization: Bearer $ALICE_TOK" \
  -H "Content-Type: application/json" \
  -d '{"model":"athena-stub","timeout":1,"messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$TIGHT" | head -1 | grep -q " 504" \
  && ok "per-request timeout=1 tightens below daemon's 30 s cap → 504" \
  || bad "per-request timeout override didn't tighten ($(echo "$TIGHT" | head -1))"
echo "$TIGHT" | grep -q '"code":"inference_timeout"' \
  && ok "per-call 504 body uses code inference_timeout" \
  || bad "504 body code unexpected ($(echo "$TIGHT" | tail -1))"
stop_daemon
unset ATHENA_STUB_DELAY_MS

echo
echo "== phase 23: graceful shutdown — drain in-flight on SIGTERM (M33.2) =="
# The queue worker is a managed Service and the HTTP server drains on
# graceful shutdown, so a SIGTERM mid-request lets the in-flight request
# COMPLETE (200, full body) instead of dropping the connection, and the
# daemon then exits on its own. The stub paces at 300 ms/chunk (~3 s) so
# the request is reliably mid-generation when the signal lands.
stop_daemon
export ATHENA_STUB_DELAY_MS=300
start_daemon "$D" 127.0.0.1 --preload --llm-model athena-stub \
  || { echo "graceful-shutdown daemon failed"; cat "$D/daemon.log"; exit 1; }
GDPID="$DPID"
INFLIGHT="$(mktemp)"
# Fire a long in-flight chat; capture its final HTTP status to a file.
( curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $ALICE_TOK" \
    -H "Content-Type: application/json" -d "$CHAT" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$INFLIGHT" ) &
CPID=$!
sleep 0.6                      # let it reach mid-generation
kill -TERM "$GDPID"            # graceful shutdown while a request is in flight
wait "$CPID" 2>/dev/null       # the request must finish, not be dropped
[ "$(cat "$INFLIGHT")" = "200" ] \
  && ok "in-flight request drained to completion (200) across SIGTERM" \
  || bad "in-flight request not drained (got '$(cat "$INFLIGHT")')"
# The daemon should exit on its own after draining (graceful, not SIGKILL).
EXITED=0
for _ in $(seq 1 20); do
  if ! kill -0 "$GDPID" 2>/dev/null; then EXITED=1; break; fi
  sleep 0.5
done
[ "$EXITED" = "1" ] \
  && ok "daemon exited on its own after draining (no SIGKILL needed)" \
  || bad "daemon still alive after graceful shutdown window"
rm -f "$INFLIGHT"
DPID=""                        # already gone; don't let cleanup re-kill
unset ATHENA_STUB_DELAY_MS

echo
echo "== phase 24: preload — warm every module with a configured default (M33.3 + M46.2) =="
# M46.2 — `--preload` warms every module class whose persisted
# allowlist has an `is_default=1` row, not only `.llm`. The daemon
# serves immediately (HTTP surface up — start_daemon's healthz wait
# proves that). Without `--preload`, loading is lazy (no preload log
# lines). --engine stub, loopback, ephemeral data dir. Seed the LLM
# allowlist via the --llm-model first-boot merge so there is at least
# one configured-default module for the preload pass to warm.
stop_daemon
# M46.2 — fresh data dir for phase 24 so prior phases' /api/models/allow
# CRUD doesn't leave textEmbedding's allowlist in a state that confuses
# the warm-line assertions. Daemon log path stays $D/daemon.log per the
# start_daemon helper, so the existing greps work unchanged.
D24="$(mktemp -d)"
start_daemon "$D24" 127.0.0.1 --preload --llm-model athena-stub \
  || { echo "preload daemon failed"; cat "$D/daemon.log"; exit 1; }
ok "daemon serves immediately (healthz up) with --preload"
WARM=0
for _ in $(seq 1 20); do
  if grep -q "preload: llm warm" "$D/daemon.log"; then WARM=1; break; fi
  sleep 0.25
done
[ "$WARM" = "1" ] \
  && ok "llm warmed at startup (preload log line present)" \
  || { bad "no 'preload: llm warm' line"; grep -i preload "$D/daemon.log"; }
# M46.2 — also asserts the new "warming N modules" summary line so
# changes to the multi-module shape are surfaced if it ever drops back
# to the LLM-only path.
grep -qE "preload: warming [0-9]+ module" "$D/daemon.log" \
  && ok "preload summary line names module count" \
  || bad "no 'preload: warming N module(s)' summary line"
# M46.2 — assert preload-all warms EVERY registered module class.
# Load.swift's @Option arrays carry a non-empty fallback default per
# module class, which resolveAllowlist merges into the DB on every
# boot, so an empty allowlist never survives one start; warming
# everything is the new contract. Assert BEFORE the next start_daemon
# (which would truncate $D/daemon.log and lose these lines).
for mod in textEmbedding transcription diarization speakerEmbedding; do
  WARM_M=0
  for _ in $(seq 1 20); do
    if grep -q "preload: $mod warm" "$D/daemon.log"; then
      WARM_M=1; break
    fi
    sleep 0.25
  done
  [ "$WARM_M" = "1" ] \
    && ok "preload warmed $mod alongside llm" \
    || { bad "no 'preload: $mod warm' line"; \
         grep -i preload "$D/daemon.log" | head -10; }
done
code 200 GET /healthz ""
stop_daemon
rm -rf "$D24"
# Opt-in: a default start does NOT preload (lazy load, no preload lines).
start_daemon "$D" 127.0.0.1 \
  || { echo "lazy daemon failed"; cat "$D/daemon.log"; exit 1; }
grep -q "preload:" "$D/daemon.log" \
  && bad "lazy default unexpectedly preloaded" \
  || ok "default start is lazy (no preload without the flag)"
stop_daemon

echo
echo "== phase 25: queue-result retention sweeper (M34.1) =="
# A submitted job's request+result persist so a client can poll after the
# fact; left unbounded those inference-output blobs accumulate forever.
# queue_result_ttl_secs prunes TERMINAL results older than the window,
# swept on the worker's idle path. Fresh data dir, auth off (loopback,
# no creds) so submission needs no token. Deterministic via a backdated
# `updated` (mirrors phase 8.7's audit-retention proof).
stop_daemon
D4="$(mktemp -d)"
qdb4="$D4/athena.sqlite"
start_daemon "$D4" 127.0.0.1 --queue-result-ttl-secs 1 \
  || { echo "queue-ttl daemon failed"; cat "$D/daemon.log"; exit 1; }
J1="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/queue/conversation" \
  -H "Content-Type: application/json" -d "$CHAT" \
  | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
[ -n "$J1" ] && ok "submitted job 1 ($J1)" || bad "submit job 1"
R1="$(curl -s "http://127.0.0.1:$PORT/v1/queue/$J1?wait=15")"
echo "$R1" | grep -q '"status":"done"' \
  && ok "job 1 completed (result persisted)" || bad "job 1 done ($R1)"
# Backdate its completion ~10 days so the 1 s TTL must reap it.
OLDQ=$(( $(date +%s) - 864000 ))
sqlite3 "$qdb4" "UPDATE jobs SET updated=$OLDQ WHERE id='$J1';"
# A second submit wakes the worker; after it drains, the idle sweep fires.
J2="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/queue/conversation" \
  -H "Content-Type: application/json" -d "$CHAT" \
  | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/queue/$J2?wait=15"
GONEQ=0
for _ in $(seq 1 20); do
  N="$(sqlite3 "$qdb4" "SELECT COUNT(*) FROM jobs WHERE id='$J1';")"
  [ "$N" = "0" ] && { GONEQ=1; break; }
  sleep 0.25
done
[ "$GONEQ" = "1" ] \
  && ok "stale terminal result pruned past the TTL window" \
  || bad "stale queue result survived TTL"
KEPTQ="$(sqlite3 "$qdb4" "SELECT COUNT(*) FROM jobs WHERE id='$J2';")"
[ "$KEPTQ" = "1" ] \
  && ok "fresh result retained within the window" \
  || bad "fresh result missing after sweep ($KEPTQ)"
stop_daemon
rm -rf "$D4"; D4=""

echo
echo "== phase 25.1: queue_max_rows caps total rows (M34.1) =="
# An independent bound: cap total job rows, trimming the oldest terminal
# results first. Submit three jobs; once they finish and the worker idles,
# the cap reaps down to 2.
D5="$(mktemp -d)"
qdb5="$D5/athena.sqlite"
start_daemon "$D5" 127.0.0.1 --queue-max-rows 2 \
  || { echo "queue-max daemon failed"; cat "$D/daemon.log"; exit 1; }
for _ in 1 2 3; do
  JX="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/queue/conversation" \
    -H "Content-Type: application/json" -d "$CHAT" \
    | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
  curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/queue/$JX?wait=15"
done
CAPQ=0
for _ in $(seq 1 20); do
  N="$(sqlite3 "$qdb5" "SELECT COUNT(*) FROM jobs;")"
  [ "$N" = "2" ] && { CAPQ=1; break; }
  sleep 0.25
done
[ "$CAPQ" = "1" ] \
  && ok "queue trimmed to the row cap (2)" \
  || bad "queue not capped (rows=$(sqlite3 "$qdb5" 'SELECT COUNT(*) FROM jobs;'))"
stop_daemon
rm -rf "$D5"; D5=""

echo
echo "== phase 25.2: vector-store TTL (prune-on-write, M34.2) =="
# vector_ttl_secs prunes vectors written longer ago than the window; the
# sweep runs opportunistically on each upsert. Backdate one vector's
# write time, then upsert another to trigger the prune. Auth off.
D6="$(mktemp -d)"
vdb6="$D6/athena.sqlite"
start_daemon "$D6" 127.0.0.1 --vector-ttl-secs 1 \
  || { echo "vector-ttl daemon failed"; cat "$D/daemon.log"; exit 1; }
curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"v1","vector":[1,2,3]}'
V1="$(sqlite3 "$vdb6" "SELECT COUNT(*) FROM vectors WHERE id='v1';")"
[ "$V1" = "1" ] && ok "vector v1 upserted" || bad "v1 upsert ($V1)"
# Backdate v1's write time ~10 days so the 1 s TTL must reap it.
OLDV=$(( $(date +%s) - 864000 ))
sqlite3 "$vdb6" "UPDATE vectors SET created=$OLDV WHERE id='v1';"
# A second upsert triggers the opportunistic prune.
curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"v2","vector":[4,5,6]}'
GONEV="$(sqlite3 "$vdb6" "SELECT COUNT(*) FROM vectors WHERE id='v1';")"
[ "$GONEV" = "0" ] \
  && ok "stale vector pruned past the TTL window" \
  || bad "stale vector survived TTL ($GONEV)"
KEPTV="$(sqlite3 "$vdb6" "SELECT COUNT(*) FROM vectors WHERE id='v2';")"
[ "$KEPTV" = "1" ] \
  && ok "fresh vector retained within the window" \
  || bad "fresh vector missing after sweep ($KEPTV)"
stop_daemon
rm -rf "$D6"; D6=""

echo
echo "== phase 25.2b: vector owner-scoping cross-tenant (H5/ADR 006) =="
# Two operator tenants (vectors.read+write, non-admin) + an admin, auth
# enabled. Each tenant's vectors are private; an admin sees across owners.
DV="$(mktemp -d)"
ATHENA_PASSWORD=opapass123 "$ATHENA" auth user add opa --role operator \
  --data-dir "$DV" >/dev/null 2>&1
ATHENA_PASSWORD=opbpass123 "$ATHENA" auth user add opb --role operator \
  --data-dir "$DV" >/dev/null 2>&1
ATHENA_PASSWORD=admpass123 "$ATHENA" auth user add vadm --role admin \
  --data-dir "$DV" >/dev/null 2>&1
vtok() { "$ATHENA" auth token add --user "$1" --data-dir "$DV" 2>/dev/null \
  | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1; }
TOKA="$(vtok opa)"; TOKB="$(vtok opb)"; TOKADM="$(vtok vadm)"
start_daemon "$DV" 127.0.0.1 \
  || { echo "vector-owner daemon failed"; cat "$DV/daemon.log"; exit 1; }
BV="http://127.0.0.1:$PORT"
vpost() { curl -s -o /dev/null -w '%{http_code}' -X POST "$BV/v1/vectors" \
  -H "Authorization: Bearer $1" -H 'Content-Type: application/json' \
  -d "$2"; }
[ "$(vpost "$TOKA" '{"id":"oa","vector":[1,0,0]}')" = 200 ] \
  && ok "opa upserts its vector" || bad "opa upsert failed"
[ "$(vpost "$TOKB" '{"id":"ob","vector":[0,1,0]}')" = 200 ] \
  && ok "opb upserts its vector" || bad "opb upsert failed"
# Query isolation: opa's search sees only opa's vector, never opb's.
QA="$(curl -s -X POST "$BV/v1/vectors/query" \
  -H "Authorization: Bearer $TOKA" -H 'Content-Type: application/json' \
  -d '{"vector":[1,0,0],"k":10}')"
echo "$QA" | grep -q '"id":"oa"' && ! echo "$QA" | grep -q '"id":"ob"' \
  && ok "opa query returns only its own vectors" \
  || bad "opa query leaked cross-tenant ($QA)"
# Stats are owner-scoped; admin sees across owners.
SA="$(curl -s "$BV/v1/vectors/stats" -H "Authorization: Bearer $TOKA")"
echo "$SA" | grep -q '"count":1' \
  && ok "opa stats count=1 (own only)" || bad "opa stats not scoped ($SA)"
SADM="$(curl -s "$BV/v1/vectors/stats" -H "Authorization: Bearer $TOKADM")"
echo "$SADM" | grep -q '"count":2' \
  && ok "admin stats count=2 (all owners)" || bad "admin stats ($SADM)"
# opb cannot delete or overwrite opa's vector.
code 404 DELETE /v1/vectors/oa "$TOKB"          # cross-tenant delete hidden
[ "$(vpost "$TOKB" '{"id":"oa","vector":[9,9,9]}')" = 409 ] \
  && ok "cross-tenant overwrite → 409 owner_conflict" \
  || bad "cross-tenant overwrite not blocked"
# opa still owns the original; admin can delete it.
code 200 DELETE /v1/vectors/oa "$TOKADM"
code 404 DELETE /v1/vectors/oa "$TOKA"          # already gone
stop_daemon
rm -rf "$DV"; DV=""

echo
echo "== phase 25.3: content opt-out — drop prompt on completion (M34.2) =="
# With --drop-request-content the queue wipes a job's request (prompt)
# blob the moment it finishes; the result the client polls for stays.
# Default (no flag) retains the prompt. Auth off.
D7="$(mktemp -d)"
cdb7="$D7/athena.sqlite"
start_daemon "$D7" 127.0.0.1 --drop-request-content \
  || { echo "content-optout daemon failed"; cat "$D/daemon.log"; exit 1; }
JC="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/queue/conversation" \
  -H "Content-Type: application/json" -d "$CHAT" \
  | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/queue/$JC?wait=15"
REQLEN="$(sqlite3 "$cdb7" "SELECT length(request) FROM jobs WHERE id='$JC';")"
RESLEN="$(sqlite3 "$cdb7" "SELECT length(result) FROM jobs WHERE id='$JC';")"
[ "$REQLEN" = "0" ] \
  && ok "prompt blob wiped on completion (request length 0)" \
  || bad "prompt blob survived with opt-out (len=$REQLEN)"
[ "${RESLEN:-0}" -gt 0 ] \
  && ok "result still retained for the client to poll (len=$RESLEN)" \
  || bad "result missing after content opt-out (len=$RESLEN)"
stop_daemon
rm -rf "$D7"; D7=""
# Control: default daemon (no flag) keeps the prompt.
D8="$(mktemp -d)"
cdb8="$D8/athena.sqlite"
start_daemon "$D8" 127.0.0.1 \
  || { echo "content-control daemon failed"; cat "$D/daemon.log"; exit 1; }
JK="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/queue/conversation" \
  -H "Content-Type: application/json" -d "$CHAT" \
  | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/queue/$JK?wait=15"
KREQ="$(sqlite3 "$cdb8" "SELECT length(request) FROM jobs WHERE id='$JK';")"
[ "${KREQ:-0}" -gt 0 ] \
  && ok "default retains the prompt (no opt-out ⇒ request kept)" \
  || bad "default unexpectedly dropped the prompt (len=$KREQ)"
stop_daemon
rm -rf "$D8"; D8=""

echo
echo "== phase 26: at-rest encryption + migration (M34.3b) =="
# encrypt_store opens (and migrates) the SQLite store under SQLCipher.
# Deterministic key via ATHENA_STORE_KEY env (no Keychain in automated
# runs). The on-disk file becomes ciphertext; the same key still serves.
stop_daemon
export ATHENA_STORE_KEY="e2e-store-key-0123456789abcdef0123"

# 26a: a store written by the keyed CLI is encrypted on disk, the daemon
# (same key) serves it, and a plaintext sqlite3 cannot read it.
D9="$(mktemp -d)"; edb="$D9/athena.sqlite"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$D9" >/dev/null \
  && ok "seed admin into keyed store" || bad "seed keyed admin"
EMAGIC="$(head -c 15 "$edb" | tr -d '\0')"
[ "$EMAGIC" != "SQLite format 3" ] \
  && ok "keyed store is ciphertext on disk (no SQLite magic)" \
  || bad "keyed store header is plaintext ($EMAGIC)"
if sqlite3 "$edb" "SELECT count(*) FROM auth_users;" >/dev/null 2>&1; then
  bad "plaintext sqlite3 read the encrypted store"
else
  ok "plaintext sqlite3 cannot read the encrypted store"
fi
A9="$("$ATHENA" auth token add --user admin --data-dir "$D9" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
start_daemon "$D9" 127.0.0.1 --encrypt-store \
  || { echo "encrypted daemon failed"; cat "$D/daemon.log"; exit 1; }
code 200 GET /api/users "$A9"   # daemon decrypts + serves with the key
stop_daemon
rm -rf "$D9"; D9=""

# 26b: a plaintext store is migrated to encrypted on the first encrypted
# start, and pre-migration data + credentials survive.
unset ATHENA_STORE_KEY
D10="$(mktemp -d)"; mdb="$D10/athena.sqlite"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$D10" >/dev/null \
  && ok "seed admin into plaintext store" || bad "seed plaintext admin"
PMAGIC="$(head -c 15 "$mdb" | tr -d '\0')"
[ "$PMAGIC" = "SQLite format 3" ] \
  && ok "store starts plaintext" || bad "store not plaintext ($PMAGIC)"
A10="$("$ATHENA" auth token add --user admin --data-dir "$D10" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
export ATHENA_STORE_KEY="e2e-store-key-0123456789abcdef0123"
start_daemon "$D10" 127.0.0.1 --encrypt-store \
  || { echo "migration daemon failed"; cat "$D/daemon.log"; exit 1; }
grep -q "migrating plaintext store to encrypted" "$D/daemon.log" \
  && ok "daemon migrated the plaintext store on first encrypted start" \
  || { bad "no migration log line"; grep -i encrypt "$D/daemon.log"; }
MMAGIC="$(head -c 15 "$mdb" | tr -d '\0')"
[ "$MMAGIC" != "SQLite format 3" ] \
  && ok "store is ciphertext after migration" \
  || bad "store still plaintext after migration ($MMAGIC)"
code 200 GET /api/users "$A10"  # pre-migration token still valid post-encrypt
stop_daemon
rm -rf "$D10"; D10=""
unset ATHENA_STORE_KEY

echo
echo "== phase 27: athena doctor — data-at-rest posture (M34.4) =="
# doctor (check #13) reports store encryption / FileVault fallback +
# retention bounds. It derives data_dir from the config, so each test
# config pins data_dir to control which store it inspects.
DOC3="$(mktemp -d)"
dcfg27() { # CFGPATH DATADIR extra-lines…
  local p="$1" dd="$2"; shift 2
  { echo 'listen_host = "127.0.0.1"'
    echo 'listen_port = 7447'
    echo 'log_dir = "/usr/local/var/log/athena"'
    echo "data_dir = \"$dd\""
    for l in "$@"; do echo "$l"; done
  } > "$p"
}
# 27a: encrypt_store=true over an ENCRYPTED store ⇒ "store encrypted".
export ATHENA_STORE_KEY="e2e-store-key-0123456789abcdef0123"
EDD="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$EDD" >/dev/null
CE="$DOC3/enc.toml"; dcfg27 "$CE" "$EDD" "encrypt_store = true"
OE="$("$ATHENA" doctor --config "$CE" --model-store "$MSTORE" 2>&1)"
echo "$OE" | grep -qi "at-rest: store encrypted" \
  && ok "doctor reports the store encrypted at rest" \
  || { bad "doctor missing encrypted-at-rest line"; echo "$OE" | grep -i at-rest; }
echo "$OE" | grep -qi "key from ATHENA_STORE_KEY env" \
  && ok "doctor reports the encryption key source" \
  || { bad "doctor missing key-source line"; echo "$OE" | grep -i key; }
rm -rf "$EDD"; unset ATHENA_STORE_KEY
# 27b: plaintext store, no encrypt_store ⇒ at-rest warns/relies on FileVault.
PDD="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$PDD" >/dev/null
CP="$DOC3/plain.toml"; dcfg27 "$CP" "$PDD"
OP="$("$ATHENA" doctor --config "$CP" --model-store "$MSTORE" 2>&1)"
echo "$OP" | grep -qi "at-rest: store is plaintext" \
  && ok "doctor reports a plaintext store (FileVault fallback)" \
  || { bad "doctor missing plaintext at-rest line"; echo "$OP" | grep -i at-rest; }
rm -rf "$PDD"
# 27c: retention knobs are surfaced.
CR="$DOC3/ret.toml"
dcfg27 "$CR" "$D" "queue_result_ttl_secs = 604800" "vector_ttl_secs = 2592000"
OR="$("$ATHENA" doctor --config "$CR" --model-store "$MSTORE" 2>&1)"
echo "$OR" | grep -qi "retention:.*queue TTL 604800s" \
  && ok "doctor reports configured retention bounds" \
  || { bad "doctor missing retention line"; echo "$OR" | grep -i retention; }
rm -rf "$DOC3"

echo
echo "== phase 28: athena doctor — audit-log posture (M35) =="
# doctor (check #14) reports how many RBAC/admin records the audit_log
# holds and whether a day-based retention prunes them. It derives
# data_dir from the config; $D already carries audit rows written in
# phase 8.5 (cookie u:admin mutations). Line shape is asserted (not an
# exact count) so the phase stays deterministic regardless of how
# StoreKey.resolve() sources a key in the test env (mirrors phase 21/27).
DOC4="$(mktemp -d)"
# 28a: no audit_retention_days ⇒ unbounded (audit trail kept for compliance).
CA="$DOC4/audit-unbounded.toml"; dcfg27 "$CA" "$D"
OA="$("$ATHENA" doctor --config "$CA" --model-store "$MSTORE" 2>&1)"
echo "$OA" | grep -qiE "audit: [0-9]+ record\(s\); retention unbounded" \
  && ok "doctor reports audit records + unbounded retention" \
  || { bad "doctor missing audit-unbounded line"; echo "$OA" | grep -i audit; }
# 28b: audit_retention_days set ⇒ doctor surfaces the prune window.
CB="$DOC4/audit-ret.toml"; dcfg27 "$CB" "$D" "audit_retention_days = 30"
OB="$("$ATHENA" doctor --config "$CB" --model-store "$MSTORE" 2>&1)"
echo "$OB" | grep -qi "audit:.*retention 30 day(s)" \
  && ok "doctor reports configured audit retention window" \
  || { bad "doctor missing audit-retention line"; echo "$OB" | grep -i audit; }
rm -rf "$DOC4"

echo
echo "== phase 29: bearer-token expiry + global age cap (M36.1) =="
# Tokens carry an optional per-token TTL (--ttl) and the daemon can
# impose a global token_max_age_days cap (relative to mint time). Both
# are enforced at validation: an expired / over-cap token resolves to nil
# → 401, exactly like an unknown one (no "expired" oracle). Deterministic
# via sqlite3 backdating (no sleeps), mirroring phase 8.7's audit backdate.
DE="$(mktemp -d)"
ATHENA_PASSWORD=expupass12 "$ATHENA" auth user add expu --role member \
  --data-dir "$DE" >/dev/null \
  && ok "expiry: seed expu" || bad "expiry: seed expu"
ATHENA_PASSWORD=goodpass12 "$ATHENA" auth user add gooduser --role member \
  --data-dir "$DE" >/dev/null
ATHENA_PASSWORD=capupass123 "$ATHENA" auth user add capu --role member \
  --data-dir "$DE" >/dev/null
# Parser: a malformed --ttl is rejected offline (nothing minted).
"$ATHENA" auth token add --user expu --ttl bogus --data-dir "$DE" \
  >/dev/null 2>&1 \
  && bad "expiry: malformed --ttl accepted" \
  || ok "expiry: malformed --ttl rejected"
mintt() { # USER [extra flags…]
  "$ATHENA" auth token add --user "$1" "${@:2}" --data-dir "$DE" \
    2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1
}
EXP_TOK="$(mintt expu --ttl 1h)"   # valid --ttl; expires backdated below
GOOD_TOK="$(mintt gooduser)"       # no ttl, minted now → valid
CAP_TOK="$(mintt capu)"            # no ttl; created backdated 10d below
{ [ -n "$EXP_TOK" ] && [ -n "$GOOD_TOK" ] && [ -n "$CAP_TOK" ]; } \
  && ok "expiry: minted 3 test tokens (--ttl 1h accepted)" \
  || bad "expiry: a test token was empty"
# Backdate expu's per-token expiry into the past + capu's MINT time 10d ago.
sqlite3 "$DE/athena.sqlite" \
  "UPDATE auth_tokens SET expires=strftime('%s','now')-100 \
   WHERE username='expu';"
sqlite3 "$DE/athena.sqlite" \
  "UPDATE auth_tokens SET created=strftime('%s','now')-864000 \
   WHERE username='capu';"
# Daemon WITH a 1-day cap: ttl-expired → 401, over-cap → 401, fresh → 200.
start_daemon "$DE" 127.0.0.1 --token-max-age-days 1 \
  || { echo "expiry daemon failed"; cat "$D/daemon.log"; exit 1; }
code 401 POST /v1/chat/completions "$EXP_TOK" "$CHAT"
code 401 POST /v1/chat/completions "$CAP_TOK" "$CHAT"
code 200 POST /v1/chat/completions "$GOOD_TOK" "$CHAT"
stop_daemon
# Cap is OPT-IN: without it the 10-day-old token (no per-token TTL) is
# accepted again, while the per-token-expired one is STILL rejected.
start_daemon "$DE" 127.0.0.1 \
  || { echo "expiry daemon (no cap) failed"; cat "$D/daemon.log"; exit 1; }
code 200 POST /v1/chat/completions "$CAP_TOK" "$CHAT"
code 401 POST /v1/chat/completions "$EXP_TOK" "$CHAT"
stop_daemon
rm -rf "$DE"

echo
echo "== phase 30: token rotate + list-expiry + doctor posture (M36.2) =="
# auth list surfaces each token's expiry; `auth token rotate` (and
# POST /api/tokens/{prefix}/rotate) revoke+reissue so the old secret dies
# and a fresh one carries the same owner/scope; doctor #15 reports token
# posture. Hash prefixes are derived from the printed key via shasum
# (the store hashes the full sk-athena-… string, like AuthConfig.sha).
DR="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$DR" >/dev/null
ATHENA_PASSWORD=rotupass12 "$ATHENA" auth user add rotu --role member \
  --data-dir "$DR" >/dev/null
ADMTOK="$("$ATHENA" auth token add --user admin --data-dir "$DR" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
ROTU_TOK="$("$ATHENA" auth token add --user rotu --ttl 1h \
  --data-dir "$DR" 2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' \
  | head -1)"
# auth list shows an expiry column: rotu (TTL) → "exp …", admin → "no-expiry".
LL="$("$ATHENA" auth list --data-dir "$DR" 2>&1)"
echo "$LL" | grep -q "exp " \
  && ok "auth list shows a TTL token (exp …)" \
  || { bad "auth list missing exp column"; echo "$LL"; }
echo "$LL" | grep -q "no-expiry" \
  && ok "auth list marks never-expiring tokens" \
  || bad "auth list missing no-expiry"
# Local CLI rotate: revoke + reissue rotu's token.
RPFX="$(printf '%s' "$ROTU_TOK" | shasum -a 256 | cut -c1-16)"
ROTU_NEW="$("$ATHENA" auth token rotate "$RPFX" --ttl 2h \
  --data-dir "$DR" 2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' \
  | head -1)"
{ [ -n "$ROTU_NEW" ] && [ "$ROTU_NEW" != "$ROTU_TOK" ]; } \
  && ok "CLI rotate reissued a new secret" \
  || bad "CLI rotate produced no/identical secret"
start_daemon "$DR" 127.0.0.1 \
  || { echo "rotate daemon failed"; cat "$D/daemon.log"; exit 1; }
code 200 POST /v1/chat/completions "$ROTU_NEW" "$CHAT"   # rotated-in works
code 401 POST /v1/chat/completions "$ROTU_TOK" "$CHAT"   # old revoked
# /api/tokens carries the expires field (rotu's TTL token).
AL="$(curl -s -H "Authorization: Bearer $ADMTOK" \
  "http://127.0.0.1:$PORT/api/tokens")"
echo "$AL" | grep -q '"expires"' \
  && ok "/api/tokens carries the expires field" \
  || { bad "/api/tokens missing expires"; echo "$AL"; }
# Remote /api rotate (admin) revokes the prior secret too.
NPFX="$(printf '%s' "$ROTU_NEW" | shasum -a 256 | cut -c1-16)"
RR="$(curl -s -X POST -H "Authorization: Bearer $ADMTOK" \
  -H "Content-Type: application/json" -d '{}' \
  "http://127.0.0.1:$PORT/api/tokens/$NPFX/rotate")"
ROTU_NEW2="$(echo "$RR" | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
[ -n "$ROTU_NEW2" ] \
  && ok "/api rotate returns a fresh secret" \
  || { bad "/api rotate returned no secret"; echo "$RR"; }
code 200 POST /v1/chat/completions "$ROTU_NEW2" "$CHAT"  # newest works
code 401 POST /v1/chat/completions "$ROTU_NEW" "$CHAT"   # prior revoked
code 400 POST /api/tokens/abc/rotate "$ADMTOK"           # short prefix
code 403 POST /api/tokens/$NPFX/rotate "$ROTU_NEW2"      # member ∌ tokens.admin
stop_daemon
# doctor #15 token-expiry posture.
DTC="$DR/doc.toml"; dcfg27 "$DTC" "$DR"
DO="$("$ATHENA" doctor --config "$DTC" --model-store "$MSTORE" 2>&1)"
echo "$DO" | grep -qiE "tokens: [0-9]+ managed" \
  && ok "doctor reports token-expiry posture (#15)" \
  || { bad "doctor missing token posture"; echo "$DO" | grep -i token; }
rm -rf "$DR"

echo
echo "== phase 31: /metrics Prometheus exposition (content-negotiated, M37) =="
# /metrics now serves Prometheus text 0.0.4 by default (the scrape
# target) and the JSON snapshot only when Accept: application/json is
# sent. Auth/role gating (metrics.read) is unchanged. Self-contained
# realm (own admin/member) so it's independent of earlier phases' state.
DM="$(mktemp -d)"
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin --role admin \
  --data-dir "$DM" >/dev/null
ATHENA_PASSWORD=mupass12345 "$ATHENA" auth user add mu --role member \
  --data-dir "$DM" >/dev/null
AM="$("$ATHENA" auth token add --user admin --data-dir "$DM" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
MM="$("$ATHENA" auth token add --user mu --data-dir "$DM" \
  2>/dev/null | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1)"
start_daemon "$DM" 127.0.0.1 \
  || { echo "metrics daemon failed"; cat "$D/daemon.log"; exit 1; }
# Generate a little counted work so the counters are non-zero.
code 200 POST /v1/chat/completions "$MM" "$CHAT"
# Default scrape (no Accept; curl sends */*) → Prometheus text 0.0.4.
PM="$(curl -s -i -H "Authorization: Bearer $AM" \
  "http://127.0.0.1:$PORT/metrics")"
echo "$PM" | grep -qi "^content-type: text/plain; version=0.0.4" \
  && ok "/metrics default content-type is Prometheus 0.0.4" \
  || { bad "/metrics default content-type wrong"
       echo "$PM" | grep -i "content-type"; }
echo "$PM" | grep -q "# TYPE athena_requests_total counter" \
  && ok "/metrics emits Prometheus HELP/TYPE + athena_requests_total" \
  || { bad "/metrics missing prometheus exposition"; }
echo "$PM" | grep -q 'athena_request_latency_ms{quantile="0.95"}' \
  && ok "/metrics emits the latency summary quantiles" \
  || bad "/metrics missing latency summary"
echo "$PM" | grep -q "athena_requests_by_kind_total{kind=\"chat\"}" \
  && ok "/metrics emits per-kind counters" \
  || bad "/metrics missing per-kind counter"
# H14 (M66.1) — the audit-write-failure health signal is exposed (0 on a
# healthy run; its presence is what makes a trail gap observable).
echo "$PM" | grep -q "# TYPE athena_audit_write_failures_total counter" \
  && ok "/metrics exposes audit_write_failures counter (H14)" \
  || bad "/metrics missing athena_audit_write_failures_total"
# Accept: application/json still returns the JSON snapshot (negotiation).
JM="$(curl -s -H "Authorization: Bearer $AM" \
  -H "Accept: application/json" \
  "http://127.0.0.1:$PORT/metrics")"
echo "$JM" | grep -q '"totalRequests"' \
  && ok "/metrics honors Accept: application/json (JSON snapshot)" \
  || { bad "/metrics JSON negotiation failed"; echo "$JM"; }
# Role gating is unchanged: member ∌ metrics.read ⇒ 403 either way.
code 403 GET /metrics "$MM"
stop_daemon
rm -rf "$DM"

echo
echo "== phase 32: install version/upgrade guard (M38) =="
# `athena install --dry-run` (no root) classifies this build against the
# version marker a prior install leaves in <prefix>/etc/athena. Fresh
# when absent; upgrade/downgrade vs a seeded marker. Detection only.
IP="$(mktemp -d)"
F="$("$ATHENA" install --dry-run --prefix "$IP" \
  --config deploy/athena.toml 2>&1)"
echo "$F" | grep -qi "fresh install" \
  && ok "install dry-run: fresh when no version marker" \
  || { bad "install missing fresh-install line"; echo "$F" | grep -i version; }
mkdir -p "$IP/etc/athena"
echo "0.0.1" > "$IP/etc/athena/installed-version"
U="$("$ATHENA" install --dry-run --prefix "$IP" \
  --config deploy/athena.toml 2>&1)"
echo "$U" | grep -qi "upgrading 0.0.1" \
  && ok "install dry-run: upgrade detected vs older marker" \
  || { bad "install missing upgrade line"; echo "$U" | grep -i version; }
echo "99.0.0" > "$IP/etc/athena/installed-version"
DN="$("$ATHENA" install --dry-run --prefix "$IP" \
  --config deploy/athena.toml 2>&1)"
echo "$DN" | grep -q "DOWNGRADING 99.0.0" \
  && ok "install dry-run: downgrade flagged vs newer marker" \
  || { bad "install missing downgrade line"; echo "$DN" | grep -i version; }
rm -rf "$IP"

echo
echo "== phase 33: login brute-force throttle (M65.6 / audit A3) =="
# POST /ui/login is exempt from the global per-principal limiter (it runs
# before any principal exists), so a remote client could brute-force the
# password unbounded. The per-IP login bucket (burst 5) now returns 429 +
# Retry-After once a peer fires a rapid run of attempts. Own auth-enabled
# daemon ($D already has seeded users) ⇒ a full bucket; bad creds so no
# session is minted; fired with no delay so the 0.2/s refill is negligible.
stop_daemon
start_daemon "$D" 127.0.0.1 \
  && ok "A3: login-throttle daemon up" || bad "A3: daemon failed"
BA3="http://127.0.0.1:$PORT"
L429=0; LRA=""
for _ in $(seq 1 12); do
  read -r LSTATUS LRETRY < <(curl -s -o /dev/null \
    -w '%{http_code} %header{retry-after}' \
    -d 'username=admin&password=wrongwrong' "$BA3/ui/login")
  if [ "$LSTATUS" = 429 ]; then
    L429=$((L429+1)); [ -n "$LRETRY" ] && LRA="$LRETRY"
  fi
done
[ "$L429" -ge 1 ] \
  && ok "rapid /ui/login burst throttled ($L429/12 → 429)" \
  || bad "no /ui/login attempt throttled (got 0 429s in 12)"
[ -n "$LRA" ] \
  && ok "login 429 carries Retry-After ($LRA s)" \
  || bad "login 429 missing Retry-After header"
stop_daemon

echo
echo "════════════════════════════════════════"
echo "  PASS=$PASS  FAIL=$FAIL"
echo "════════════════════════════════════════"
[ "$FAIL" -eq 0 ] || exit 1
