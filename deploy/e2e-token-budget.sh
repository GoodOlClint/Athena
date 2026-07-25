#!/bin/bash
#
# e2e-token-budget.sh — end-to-end DoD for ADR 041 (per-principal token budgets).
# Auth ON, `--engine stub` (the budget algebra is engine-independent, and the
# stub meters real synthesized token counts), one day-window budget on ONE user:
#
#   1) alice's first /v1/chat/completions succeeds and carries
#      x-athena-tokens-{limit,remaining,reset}, remaining < limit.
#   2) alice's next request → 429 with code "quota_exceeded", type
#      "insufficient_quota", and a Retry-After that lands before tomorrow's
#      local midnight.
#   3) GET /api/usage STILL succeeds for exhausted alice (enforcement scope:
#      an over-budget principal must be able to diagnose why).
#   4) POST /v1/chat/completions/count_tokens is NOT quota-refused for
#      exhausted alice (ADR 042 — it exists to keep a client under budget).
#   5) bob (no override, no global default) is unaffected AND carries NO
#      x-athena-tokens-* headers — absent means "no cap", never "zero".
#   6) Rewinding alice's stored period_start by one period lets her request
#      succeed again with the period counters reset, while LIFETIME total_tokens
#      is unchanged (the roll destroys no history).
#
# Pre-change: no headers, no 429, no period columns.
# Usage: ./deploy/e2e-token-budget.sh [binary]
set -uo pipefail
cd "$(dirname "$0")/.."

ATHENA="${1:-.build/xcode/Build/Products/Release/athena}"
PORT=7793
D="$(mktemp -d)"
MSTORE="$(mktemp -d)"
trap 'kill ${DPID:-0} 2>/dev/null; wait ${DPID:-0} 2>/dev/null; rm -rf "$D" "$MSTORE"' EXIT

[ -x "$ATHENA" ] || { echo "error: no binary at $ATHENA (build it first)"; exit 1; }

pass=0
fail=0
ok() { echo "  ok   $1"; pass=$((pass + 1)); }
bad() { echo "  FAIL $1"; fail=$((fail + 1)); }

echo "== phase 0: seed users + tokens (offline CLI) =="
ATHENA_PASSWORD=adminpass1 "$ATHENA" auth user add admin \
  --role admin --data-dir "$D" >/dev/null \
  && ok "create admin" || bad "create admin"
ATHENA_PASSWORD=alicepass1 "$ATHENA" auth user add alice \
  --role member --data-dir "$D" >/dev/null \
  && ok "create alice" || bad "create alice"
ATHENA_PASSWORD=bobpass1234 "$ATHENA" auth user add bob \
  --role member --data-dir "$D" >/dev/null \
  && ok "create bob" || bad "create bob"

tok() { "$ATHENA" auth token add --user "$1" --data-dir "$D" 2>/dev/null \
  | grep -o 'sk-athena-[A-Za-z0-9_-]*' | head -1; }
ADMIN_TOK="$(tok admin)"
ALICE_TOK="$(tok alice)"
BOB_TOK="$(tok bob)"
for v in ADMIN_TOK ALICE_TOK BOB_TOK; do
  [ -n "${!v}" ] && ok "minted $v" || bad "minted $v (empty)"
done

# A per-USER override with NO global default: proves the override path and that
# an un-overridden user stays unlimited.
"$ATHENA" auth user budget alice 5 --data-dir "$D" >/dev/null \
  && ok "set alice budget = 5 tokens/period" || bad "set alice budget"
"$ATHENA" auth user budget bob --clear --data-dir "$D" >/dev/null \
  && ok "bob has no override (inherits: unlimited)" || bad "clear bob budget"

echo
echo "== phase 1: start auth-on daemon (stub engine, day window) =="
"$ATHENA" load --engine stub --host 127.0.0.1 --port "$PORT" \
  --data-dir "$D" --model-store "$MSTORE" \
  --token-budget-window day >"$D/daemon.log" 2>&1 &
DPID=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 \
  && ok "daemon up on :$PORT" || bad "daemon did not come up (see $D/daemon.log)"

B="http://127.0.0.1:$PORT"
# The stub engine's model id (same as deploy/e2e-rbac.sh uses).
CHAT='{"model":"Qwen3.5-27B-4bit-mtp","messages":[{"role":"user","content":"count these tokens please"}],"max_tokens":32}'

# Warm the stub LLM slot so a cold-load 503 can't masquerade as a quota verdict.
for _ in $(seq 1 30); do
  rc="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $ADMIN_TOK" -H 'Content-Type: application/json' \
    -d '{"module":"llm"}' "$B/api/models/load")"
  [ "$rc" = "200" ] && break
  sleep 0.2
done

post_chat() { # TOKEN OUTFILE HEADERFILE
  curl -s -o "$2" -D "$3" -w '%{http_code}' -X POST "$B/v1/chat/completions" \
    -H "Authorization: Bearer $1" -H 'Content-Type: application/json' \
    -d "$CHAT"
}
hdr() { # HEADERFILE NAME (lowercase)
  # Header names are case-insensitive on the wire and BSD awk has no IGNORECASE,
  # so fold the whole dump to lower case before matching.
  tr -d '\r' < "$1" | tr 'A-Z' 'a-z' \
    | awk -v n="$2" '$1==n":" {print $2}' | tail -1
}

echo
echo "== phase 2: first request succeeds + advisory headers =="
code=$(post_chat "$ALICE_TOK" "$D/a1.json" "$D/a1.hdr")
LIMIT="$(hdr "$D/a1.hdr" x-athena-tokens-limit)"
REMAIN="$(hdr "$D/a1.hdr" x-athena-tokens-remaining)"
RESET="$(hdr "$D/a1.hdr" x-athena-tokens-reset)"
[ "$code" = "200" ] && ok "alice request 1 → 200" \
  || bad "alice request 1 → $code (body: $(cat "$D/a1.json"))"
[ "$LIMIT" = "5" ] && ok "x-athena-tokens-limit = 5" \
  || bad "x-athena-tokens-limit = '$LIMIT'"
if [ -n "$REMAIN" ] && [ "$REMAIN" -lt 5 ]; then
  ok "x-athena-tokens-remaining = $REMAIN (< limit — the request was charged)"
else
  bad "x-athena-tokens-remaining = '$REMAIN' (want < 5)"
fi
# (the dump was lower-cased for matching, so compare case-insensitively)
case "$RESET" in
  *t*z) ok "x-athena-tokens-reset = $RESET" ;;
  *) bad "x-athena-tokens-reset = '$RESET' (want a UTC ISO8601 instant)" ;;
esac

echo
echo "== phase 3: the next request is refused 429 quota_exceeded =="
code=$(post_chat "$ALICE_TOK" "$D/a2.json" "$D/a2.hdr")
RA="$(hdr "$D/a2.hdr" retry-after)"
QCODE="$(python3 -c "
import json
try: print(json.load(open('$D/a2.json'))['error'].get('code',''))
except Exception: print('')" 2>/dev/null)"
QTYPE="$(python3 -c "
import json
try: print(json.load(open('$D/a2.json'))['error'].get('type',''))
except Exception: print('')" 2>/dev/null)"
[ "$code" = "429" ] && ok "alice request 2 → 429" \
  || bad "alice request 2 → $code (want 429; body: $(cat "$D/a2.json"))"
[ "$QCODE" = "quota_exceeded" ] && ok "code = quota_exceeded" \
  || bad "code = '$QCODE'"
[ "$QTYPE" = "insufficient_quota" ] && ok "type = insufficient_quota" \
  || bad "type = '$QTYPE'"
# Retry-After must land before tomorrow's LOCAL midnight (day window).
TILL_MIDNIGHT="$(python3 -c "
import datetime
now = datetime.datetime.now()
tomorrow = (now + datetime.timedelta(days=1)).replace(
    hour=0, minute=0, second=0, microsecond=0)
print(int((tomorrow - now).total_seconds()) + 2)")"
if [ -n "$RA" ] && [ "$RA" -ge 1 ] && [ "$RA" -le "$TILL_MIDNIGHT" ]; then
  ok "Retry-After = ${RA}s (≤ ${TILL_MIDNIGHT}s to local midnight)"
else
  bad "Retry-After = '$RA' (want 1..$TILL_MIDNIGHT)"
fi

echo
echo "== phase 4: enforcement scope — the exhausted principal can diagnose =="
code=$(curl -s -o "$D/usage.json" -w '%{http_code}' \
  -H "Authorization: Bearer $ALICE_TOK" "$B/api/usage")
[ "$code" = "200" ] && ok "GET /api/usage → 200 while exhausted" \
  || bad "GET /api/usage → $code while exhausted"
LIFETIME="$(python3 -c "
import json
u=json.load(open('$D/usage.json'))['usage']
print(sum(r['total_tokens'] for r in u))" 2>/dev/null)"
[ -n "$LIFETIME" ] && [ "$LIFETIME" -gt 0 ] \
  && ok "usage reports lifetime total_tokens = $LIFETIME" \
  || bad "usage lifetime total_tokens = '$LIFETIME'"

# ADR 042 — counting must NOT be quota-refused (it exists to stay under budget).
# The stub has no tokenizer, so 501 is the expected engine answer; what matters
# is that it is not a 429.
code=$(curl -s -o "$D/count.json" -w '%{http_code}' \
  -X POST "$B/v1/chat/completions/count_tokens" \
  -H "Authorization: Bearer $ALICE_TOK" -H 'Content-Type: application/json' \
  -d "$CHAT")
[ "$code" != "429" ] \
  && ok "count_tokens not quota-refused while exhausted (→ $code)" \
  || bad "count_tokens → 429 (must be excluded from enforcement)"

echo
echo "== phase 5: an un-budgeted principal is untouched, and gets NO headers =="
code=$(post_chat "$BOB_TOK" "$D/b1.json" "$D/b1.hdr")
BLIMIT="$(hdr "$D/b1.hdr" x-athena-tokens-limit)"
BREMAIN="$(hdr "$D/b1.hdr" x-athena-tokens-remaining)"
[ "$code" = "200" ] && ok "bob request → 200 (unaffected by alice's budget)" \
  || bad "bob request → $code"
[ -z "$BLIMIT" ] && [ -z "$BREMAIN" ] \
  && ok "no x-athena-tokens-* headers when unlimited (absent ≠ zero)" \
  || bad "unlimited bob got limit='$BLIMIT' remaining='$BREMAIN'"

echo
echo "== phase 6: a period roll resets the period, preserves lifetime =="
# Rewind alice's stored period_start by one full day — the same thing the clock
# does at midnight, without waiting for it.
BEFORE_LIFETIME="$LIFETIME"
sqlite3 "$D/athena.sqlite" \
  "UPDATE usage_counters SET period_start = period_start - 86400
   WHERE principal = 'u:alice';" \
  && ok "rewound alice's period_start by one day" || bad "sqlite rewind failed"
code=$(post_chat "$ALICE_TOK" "$D/a3.json" "$D/a3.hdr")
[ "$code" = "200" ] && ok "alice request 3 → 200 after the roll" \
  || bad "alice request 3 → $code after the roll (body: $(cat "$D/a3.json"))"
PERIOD_AFTER="$(sqlite3 "$D/athena.sqlite" \
  "SELECT period_prompt_tokens + period_completion_tokens
   FROM usage_counters WHERE principal='u:alice';")"
LIFE_AFTER="$(sqlite3 "$D/athena.sqlite" \
  "SELECT prompt_tokens + completion_tokens
   FROM usage_counters WHERE principal='u:alice';")"
PERIOD_PRE="$(sqlite3 "$D/athena.sqlite" \
  "SELECT requests FROM usage_counters WHERE principal='u:alice';")"
if [ -n "$PERIOD_AFTER" ] && [ -n "$LIFE_AFTER" ] \
  && [ "$PERIOD_AFTER" -lt "$LIFE_AFTER" ]; then
  ok "period tokens reset to $PERIOD_AFTER while lifetime grew to $LIFE_AFTER"
else
  bad "period=$PERIOD_AFTER lifetime=$LIFE_AFTER (want period < lifetime)"
fi
[ -n "$BEFORE_LIFETIME" ] && [ "$LIFE_AFTER" -ge "$BEFORE_LIFETIME" ] \
  && ok "lifetime accounting never went backwards ($BEFORE_LIFETIME → $LIFE_AFTER)" \
  || bad "lifetime went backwards: $BEFORE_LIFETIME → $LIFE_AFTER"
[ -n "$PERIOD_PRE" ] && [ "$PERIOD_PRE" -ge 2 ] \
  && ok "lifetime request count preserved across the roll ($PERIOD_PRE)" \
  || bad "lifetime requests = '$PERIOD_PRE' (want ≥ 2)"

echo
echo "════════════════════════════════════════"
echo "  PASS=$pass  FAIL=$fail"
echo "════════════════════════════════════════"
[ "$fail" = "0" ] || exit 1
