# 041 — Per-principal token budgets (rolling window)

**Status:** Accepted — **IMPLEMENTED 2026-07-25** (A1 `222601b5`, A2 `626886ed`, A3 `5e98a6da`, A4 `1b51bc46`). Brownfield-gated; operator-approved 2026-07-25 (window semantics, budget source, response shape all confirmed by interview). DoD: `deploy/e2e-token-budget.sh` 31/0; usage: `docs/token-budgets.md`.
**Date:** 2026-07-25
**Milestone:** TBD (operability / product surface)
**Executes:** ADR 007 #9 (token-budget quotas).
**Amends:** ADR 007 — #8 (native `/api` metering) is **closed by obsolescence**, see below.

## Context

ADR 007 brought two backlog items into the program: **#8** meter the native `/api/*` surface, and **#9** per-principal token budgets (which needed #8's accounting first). #8 was recorded as partially implemented with a standing caveat that `handleNativeEmbed` still dropped `meter()`. #9 was never scheduled.

**#8 is now obsolete rather than outstanding.** ADR 013 made `/v1` the single inference surface and `/api/*` the control plane; ADR 025 S2 removed the queue, ADR 031 removed `/api/chat`, and the `/api/embed` alias went with it (`Sources/athena/Server/AthenaServer.swift:239` records the removal). There is no native inference route left, so there is no `/api/*` token traffic to meter — the whole control plane consumes zero model tokens. The ADR 007 caveat names a function that no longer exists. NA8 (single auth resolution threaded into metering) shipped in v0.10.140 and remains in force at `AthenaServer.bearerPrincipal` / `usagePrincipal`. #8 therefore closes with no code change; this ADR records that so the item stops reading as debt.

**#9 is the live work, and the accounting it depends on already exists.** `meter()` (`AthenaServer.swift`) is the single chokepoint every token-bearing route funnels through — `/v1/chat/completions` (streaming and blocking), `/v1/embeddings`, and the ADR 036 Anthropic `/v1/messages` adapter. It bumps the Prometheus counter and upserts `usage_counters` (principal, requests, prompt_tokens, completion_tokens, updated). Those counters are **lifetime cumulative**: there is no period, so nothing in the schema can express "50M tokens this month".

Three existing constraints shape the design rather than being re-litigated:

- **ADR 025 (stateless loopback).** Auth-off loopback creates no `athena.sqlite`; usage lives in an in-memory store and dies with the process. `RateLimitMiddleware` already bypasses entirely when auth is off, for the same reason: with no principals there is nothing to key on. Quotas inherit this — **auth-on only**, inert in loopback dev mode. A quota is an accounting control over identified callers, not a safety net for an unauthenticated dev box.
- **ADR 029/038 (serialized inference).** Quota rejection is admission control on *accounting*, distinct from the governor's memory 503 and from the M29 concurrency 429. Three orthogonal reasons to refuse, three distinct codes.
- **ADR 009 (MLX-free decision seams).** The budget algebra and window arithmetic are pure functions in `AthenaServerKit`, unit-pinned under `swift test`; only the store I/O and middleware wiring sit outside.

## Decision

### 1. Rolling period, additive columns

The budget applies to tokens consumed in the **current period**, which resets when the period rolls. `usage_counters` gains three additive columns (`try? ALTER TABLE`, the established idiom at `AthenaStore.swift:257`):

```
period_start REAL NOT NULL DEFAULT 0
period_prompt_tokens INTEGER NOT NULL DEFAULT 0
period_completion_tokens INTEGER NOT NULL DEFAULT 0
```

The existing `prompt_tokens` / `completion_tokens` / `requests` columns keep their meaning — **lifetime** totals — so `athena usage` and `GET /api/usage` do not change behavior and no historical data is destroyed by a period roll. Rejected: resetting the existing columns (silently redefines a shipped surface and loses lifetime accounting).

The roll is **lazy and pure**: `QuotaWindow.periodStart(containing: now, window:)` maps a timestamp to the start of its period (`day` = local midnight, `month` = first of the local month). On any read or write, a stored `period_start` older than the current period means the period counters read as zero and are overwritten on the next `addUsage`. No timer, no background job, no cron — the arithmetic is the state machine. Window is a config key with `day`/`month` values; an hour-granular or sliding window is not built (see Rejected alternatives).

### 2. Budget = global config default, overridable per user

- `token_budget` (TOML, per period; unset or `0` ⇒ unlimited) is the default applied to every principal, following the 5-touchpoint config pattern.
- `token_budget_window` (TOML, `day` | `month`; default `month`).
- `auth_users.token_budget INTEGER` (nullable, additive ALTER) overrides the global default for one user. NULL ⇒ inherit the config default. Set via the existing user-admin control-plane route and the `athena` user CLI (`AuthCmd.swift`), audited as `user.budget` like every other admin mutation.

Per-token-**principal** granularity is deliberately not offered: budgets key on the same `u:<user>` / `t:<hash>` principal the rate limiter and usage metering already use, so a user's tokens share one budget. That is the existing identity model, not a new one.

Under ADR 037's `PUT /api/config` deny-list, `token_budget` and `token_budget_window` are **not** deny-listed — they are operability knobs, not a path to daemon takeover, so they are settable through the config API like `rate_limit`.

### 3. Enforcement: refuse at admission on token-bearing routes only

A `QuotaMiddleware` — same shape and slot as `RateLimitMiddleware` (auth-gated, principal-keyed, 429) — refuses a request when the principal's period usage has already **reached or passed** its budget:

```
429 {"error":{"message":"token budget exhausted; resets <ISO8601>",
              "type":"insufficient_quota","code":"quota_exceeded"}}
```

with `Retry-After` set to the seconds until the period rolls.

Enforcement scope is the **token-bearing routes only** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/messages`) — deliberately *not* the `RateLimitMiddleware.throttled()` set. A principal that is over budget must keep being able to read `GET /api/usage` and its own state to see why it was refused; locking the control plane behind a token budget would make the failure undiagnosable from the client side. Control-plane routes consume no tokens, so including them would add no protection.

**Honesty boundary (binding):** a request's token cost is unknown before it runs, so enforcement is *pre-request against accrued usage*. A principal with 1 token of budget left is **admitted** and may overshoot by that one request's size — bounded above by the effective prompt ceiling plus `max_completion_tokens`, not unbounded, but real. Mid-stream termination on budget exhaustion is explicitly not built: it would truncate a response the client is already committed to and cannot be made exact anyway. The budget is a *cap with a bounded overshoot*, and the docs say so in those words.

### 4. Advisory headers on every metered response

Normal (non-refused) responses on the enforced routes carry, read after `meter()` has run:

```
x-athena-tokens-limit: 50000000
x-athena-tokens-remaining: 12873440
x-athena-tokens-reset: 2026-08-01T00:00:00Z
```

Omitted entirely when the principal has no budget (unlimited) — an absent header means "no cap", never "zero remaining". This is the cheap half of the feature: a client self-throttles from headers it reads on traffic it is already making, instead of polling `/api/usage` or discovering the wall by hitting it.

### 5. `/api/usage` reports the budget

`UsageEntry` gains `budget`, `period_tokens`, and `period_reset` (all omitted-when-nil, so existing consumers are byte-unchanged). `athena usage` renders them. Owner-scoping is unchanged: a member sees its own row, an admin sees all.

## Rejected alternatives

- **Lifetime cumulative budget** (no window). Zero schema change, but a budget that never resets is a one-shot cap requiring an admin reset to clear — operationally worse than the column it saves.
- **Sliding window over a time-bucketed table.** Most precise; needs a new table, an aggregation query on every request, and a pruning job. Rejected as unearned precision for an appliance-scale deployment: nobody is arbitraging the boundary of their own budget.
- **Headers only, no enforcement.** Zero risk that a quota bug refuses valid inference, and zero enforcement — it is not #9.
- **Extending `RateLimit`'s token bucket to tokens.** The unit is wrong (requests/sec vs tokens/period), the state must be durable across restarts (the limiter's is not), and the refusal codes must stay distinct so an operator can tell throttling from exhaustion.
- **Mid-stream cancellation when the budget is crossed.** Truncates a committed response, still cannot be exact (the crossing is only known after the token is generated), and creates a partial-response class of client bug for a bounded-overshoot problem.
- **Metering the control plane** (ADR 007 #8 as literally written). Nothing to meter; see Context.

## Consequences

- Three additive columns on `usage_counters`, one on `auth_users`, all `try? ALTER TABLE` — an old DB opens and works with budgets unset (unlimited), which is the pre-change behavior.
- Two new config keys, one new error code (`quota_exceeded` / `insufficient_quota`), three new response headers, three new omitted-when-nil `/api/usage` fields, one new audited admin mutation. `OpenAPISpec.swift` is updated in the same edit per the CLAUDE.md canonical-pipeline rule.
- Quotas are **inert in loopback dev mode** (ADR 025). Any operator expecting a budget to bound their own dev box will be surprised; `athena doctor` gains a note, and the docs lead with it.
- The overshoot bound above is a permanent property of the design, not a defect to fix later.
- ADR 007 is closed by this ADR plus its own #8 amendment: metering is complete for every surface that has tokens, and quotas exist.
