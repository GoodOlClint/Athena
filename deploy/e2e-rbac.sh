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
  for c in \
    .build/xcode/Build/Products/Debug/athena \
    .build/xcode/Build/Products/Release/athena; do
    [ -x "$c" ] && ATHENA="$c" && break
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
cleanup() {
  [ -n "$DPID" ] && kill "$DPID" 2>/dev/null
  wait "$DPID" 2>/dev/null
  rm -rf "$D" "$EMPTY" "$MSTORE" ${D2:+"$D2"} ${D3:+"$D3"}
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

CHAT='{"model":"x","messages":[{"role":"user","content":"hi"}]}'

echo
echo "== phase 0: seed RBAC subjects (offline CLI) =="
"$ATHENA" auth user add admin --password adminpass1 \
  --role admin --data-dir "$D" >/dev/null \
  && ok "create admin user (role admin)" || bad "create admin user"
"$ATHENA" auth user add alice --password alicepass1 \
  --role member --data-dir "$D" >/dev/null \
  && ok "create alice (member)" || bad "create alice"
"$ATHENA" auth user add bob --password bobpass123 \
  --role member --data-dir "$D" >/dev/null \
  && ok "create bob (member)" || bad "create bob"
"$ATHENA" auth user add ro --password ropass1234 \
  --role readonly --data-dir "$D" >/dev/null \
  && ok "create ro (readonly)" || bad "create ro"
# boss is a full admin USER but we mint a member-SCOPED token for it.
"$ATHENA" auth user add boss --password bosspass123 \
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

echo
echo "== phase 2: permission gating (auth enforced) =="
start_daemon "$D" 127.0.0.1 || { echo "daemon failed"; \
  cat "$D/daemon.log"; exit 1; }
grep -q "auth: enabled (RBAC" "$D/daemon.log" \
  && ok "daemon reports RBAC enabled" || bad "RBAC-enabled log line"

code 401 POST /v1/chat/completions "" "$CHAT"          # no token
code 401 POST /v1/chat/completions "sk-athena-bogus" "$CHAT"
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
CHATCAP='{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":2}'
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
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":2}' \
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
  '{"model":"x","messages":[{"role":"user","content":"hi"}],"n":2}'
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"x","messages":[{"role":"user","content":"hi"}],"logprobs":true}'
code 400 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"x","messages":[{"role":"user","content":"hi"}],"logit_bias":{"50256":-100}}'
# n:1 is the supported single-decode case ⇒ 200.
code 200 POST /v1/chat/completions "$ALICE_TOK" \
  '{"model":"x","messages":[{"role":"user","content":"hi"}],"n":1}'
# top_p/seed are accepted (inert on the stub's model-less path) ⇒ 200.
TPS="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"top_p":0.9,"seed":42}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$TPS" | grep -q '"finish_reason":"stop"' \
  && ok "top_p+seed accepted ⇒ 200/stop" \
  || bad "top_p+seed request rejected ($TPS)"
# stop truncates the output at the first sequence ⇒ finish_reason stop and
# the text after the sequence is gone (string form).
STOPR="$(curl -s -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stop":"governed"}' \
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
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true,"stop":["governed"]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$SSES" | grep -q '"finish_reason":"stop"' \
  && ok "SSE stop ⇒ terminal finish_reason stop" \
  || bad "SSE stop finish_reason missing ($SSES)"
echo "$SSES" | grep -q 'governed' \
  && bad "SSE stop did not truncate the stream ($SSES)" \
  || ok "SSE stop suppressed text from the sequence on"

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
echo "== phase 2.9: usage accounting — non-zero token counts (M27.1) =="
# The OpenAI `usage` object must report REAL token counts (was a
# hardcoded {0,0,0}). Under --engine stub the counts are synthesized
# from whitespace tokenization, so they're deterministic and non-zero.
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

EMBBODY="$(curl -s -X POST \
  -H "Authorization: Bearer $ALICE_TOK" \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","input":["hello world","another sentence here"]}' \
  "http://127.0.0.1:$PORT/v1/embeddings")"
EP="$(intfield "$EMBBODY" prompt_tokens)"
ET="$(intfield "$EMBBODY" total_tokens)"
{ [ -n "$EP" ] && [ "$EP" -gt 0 ]; } \
  && ok "embeddings usage.prompt_tokens = $EP (>0)" \
  || bad "embeddings usage.prompt_tokens not >0 ($EMBBODY)"
{ [ -n "$ET" ] && [ "$ET" -eq "$EP" ]; } \
  && ok "embeddings usage.total_tokens = $ET (= prompt; no completion)" \
  || bad "embeddings usage.total_tokens != prompt_tokens ($EMBBODY)"

# The revived global metrics counter must reflect that work (was dead).
MET="$(curl -s -H "Authorization: Bearer $ADMIN_TOK" \
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
SREQ='{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}'
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
NREQ='{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true}'
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
echo "$SPEC" | grep -q '"openapi": "3.0.3"' \
  && ok "openapi 3.0.3 document" || bad "not an openapi 3.0.3 doc"
VER="$("$ATHENA" --version)"
echo "$SPEC" | grep -q "\"version\": \"$VER\"" \
  && ok "info.version matches binary ($VER)" \
  || bad "info.version != binary version $VER"
echo "$SPEC" | grep -q '"bearerAuth"' \
  && ok "bearerAuth security scheme present" || bad "no bearerAuth scheme"
echo "$SPEC" | grep -q '"/v1/chat/completions"' \
  && ok "documents /v1/chat/completions" || bad "missing chat path"
echo "$SPEC" | grep -q '"error"' \
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
"$ATHENA" auth user add admin --password adminpass1 --role admin \
  --data-dir "$D2" >/dev/null && ok "seed D2 admin" || bad "seed admin"
"$ATHENA" auth user add ro --password ropass1234 --role readonly \
  --data-dir "$D2" >/dev/null && ok "seed D2 ro" || bad "seed ro"
"$ATHENA" auth user add mem --password mempass12 --role member \
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
code 200 POST /v1/chat/completions "$NT" "$CHAT"   # member inference
code 200 DELETE "/api/tokens/$HP" "$A2"            # delete the new one
code 404 DELETE /api/tokens/deadbeef9999 "$A2"
code 400 DELETE /api/tokens/abc "$A2"               # <6 hex
# Last-admin protection over HTTP (D2 has exactly one admin).
code 403 DELETE /api/users/admin "$A2"              # sole admin
code 403 DELETE /api/users/admin/roles/admin "$A2"  # revoke sole admin

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
"$ATHENA" auth user add admin --password adminpass1 --role admin \
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
  # users.admin: create/delete/grant (admin only)
  clic nz "auth user add (readonly=403)"  auth user add e2c \
    --password pw12345678 --role member --key "$R2"
  clic 0  "auth user add (admin)"         auth user add e2c \
    --password pw12345678 --role member --key "$A2"
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
  # 25 MB audio cap — present in both the nginx and Caddy snippets.
  grep -qi "client_max_body_size 25m" "$RPG" \
    && grep -qi "max_size 25MB" "$RPG" \
    && ok "guide sets a 25 MB body cap (matches audio upload limit)" \
    || bad "guide missing the 25 MB body-size directive"
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
start_daemon "$D" 127.0.0.1 --rate-limit 1 --rate-burst 1 \
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
start_daemon "$D" 127.0.0.1 --max-concurrency 1 \
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
start_daemon "$D" 127.0.0.1 --request-timeout-secs 1 \
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
  -d '{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
echo "$SR" | grep -q "data: \[DONE\]" \
  && ok "streamed generation truncates but closes with [DONE]" \
  || bad "streamed timeout did not close cleanly"
stop_daemon
# 22c: the SAME slow generation under a generous timeout is NOT killed —
# proves the deadline doesn't false-fire on legitimate work.
start_daemon "$D" 127.0.0.1 --request-timeout-secs 30 \
  || { echo "generous-timeout daemon failed"; cat "$D/daemon.log"; exit 1; }
code 200 POST /v1/chat/completions "$ALICE_TOK" "$CHAT"
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
start_daemon "$D" 127.0.0.1 \
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
echo "════════════════════════════════════════"
echo "  PASS=$PASS  FAIL=$FAIL"
echo "════════════════════════════════════════"
[ "$FAIL" -eq 0 ] || exit 1
