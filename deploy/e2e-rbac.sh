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
cleanup() {
  [ -n "$DPID" ] && kill "$DPID" 2>/dev/null
  wait "$DPID" 2>/dev/null
  rm -rf "$D" "$EMPTY" "$MSTORE" ${D2:+"$D2"}
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

start_daemon() { # DATADIR HOST
  "$ATHENA" load --engine stub --host "$2" --port "$PORT" \
    --data-dir "$1" --model-store "$MSTORE" \
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
stop_daemon

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
echo "════════════════════════════════════════"
echo "  PASS=$PASS  FAIL=$FAIL"
echo "════════════════════════════════════════"
[ "$FAIL" -eq 0 ] || exit 1
