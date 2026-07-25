# Token budgets

Per-principal token budgets over a rolling period ([ADR 041](decisions/041-per-principal-token-budgets.md)). A budget caps the tokens a caller may spend on the inference surface between period rolls; over budget, the caller gets a `429 quota_exceeded` until the period rolls.

**Quotas are inert in loopback dev mode.** With auth off there is no principal to charge and no store to charge it against ([ADR 025](decisions/025-collapse-persistent-data-tenants.md)), exactly like the rate limiter. Setting `token_budget` on an auth-off dev box buys you nothing; `athena doctor` says so out loud. Seed credentials first.

## Configure

```toml
token_budget = 50000000        # tokens per period per principal; 0 or absent = unlimited
token_budget_window = "month"  # "day" | "month" (default "month")
```

Boundaries are **local** — midnight, or the first instant of the month, on the daemon's clock. An unrecognized window fails loudly (config parse, `athena config set`, and daemon start all refuse it) rather than quietly falling back.

Both keys are settable at runtime through the config API and the CLI:

```sh
athena config set token_budget 50000000
athena config set token_budget_window day
```

## Per-user overrides

The global value is the default. Any user can carry its own, stored in the auth DB:

```sh
athena auth user budget alice 5000000     # 5M tokens per period for alice
athena auth user budget alice 0           # unlimited for alice, even with a global default
athena auth user budget alice --clear     # back to inheriting the default
```

The same verb works off-box (`--host`), where it drives `PUT /api/users/{name}/budget` (`users.admin`, audited as `user.budget`). `GET /api/users` reports each user's `token_budget`, omitted when they inherit.

Budgets key on the **principal** the daemon already meters and rate-limits (`u:<user>`, or `t:<hash>` for a bootstrap key), so all of a user's tokens share one budget. A bootstrap-key principal has no user row and can only inherit the global default.

## What a client sees

Every non-refused response on a budgeted route carries advisory headers:

```
x-athena-tokens-limit: 50000000
x-athena-tokens-remaining: 12873440
x-athena-tokens-reset: 2026-08-01T00:00:00Z
```

They are **omitted entirely** when the principal has no budget — an absent header means "no cap", never "zero remaining". `remaining` is read after the request was metered, so it already accounts for the call that returned it.

Exhausted:

```json
429 {"error":{"message":"token budget exhausted; resets 2026-08-01T00:00:00Z",
              "type":"insufficient_quota","code":"quota_exceeded"}}
```

with `Retry-After` set to the seconds until the roll. This is a different refusal from the rate limiter's 429 (`rate_limited`, request rate) and the governor's 503 (memory) — three orthogonal reasons, three codes.

## What is enforced

Only the routes that spend model tokens:

| Route | Enforced |
|---|---|
| `POST /v1/chat/completions` | yes |
| `POST /v1/embeddings` | yes |
| `POST /v1/messages` (Anthropic dialect) | yes |
| `POST /v1/chat/completions/count_tokens` | **no** — it exists so a client can stay *under* budget ([ADR 042](decisions/042-context-window-discovery-and-token-counting.md)) |
| `GET /api/usage` and the rest of `/api/*` | **no** — an exhausted principal must still be able to see why it was refused |
| `/v1/audio/*`, `/v1/video/*` | not yet — they consume no LLM tokens today |

## Reading the state

```sh
athena usage
```

```
PRINCIPAL                       REQS      PROMPT       COMPL       TOTAL      PERIOD      BUDGET
u:alice                           42       18310        4120       22430       22430     5000000
u:bob                              7        1200         300        1500           -   unlimited
period resets 2026-08-01T00:00:00Z
```

`PROMPT`/`COMPL`/`TOTAL` are **lifetime** totals and are never reset by a period roll; `PERIOD` is spend in the current period only. `GET /api/usage` carries the same as `budget` / `period_tokens` / `period_reset`, omitted when no budget applies.

## Honesty boundary

**A budget is a cap with a bounded overshoot, not a hard ceiling.** A request's token cost is unknown before it runs, so enforcement is pre-request against *accrued* usage: a principal with one token left is admitted, and that request may push it over. The overshoot is bounded by the effective prompt ceiling plus the request's `max_completion_tokens` — real, but not unbounded. Mid-stream termination on crossing the budget is deliberately not built: it would truncate a response the client is already committed to, and it could not be made exact anyway (the crossing is only knowable after the token is generated).

The period reset is lazy arithmetic, not a job: a stored period older than the current one reads as zero and is overwritten on the next request. Nothing runs at midnight, and nothing needs to.
