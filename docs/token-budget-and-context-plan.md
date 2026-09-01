# Change plan — token budgets (ADR 041) + context discovery (ADR 042)

**Status:** operator-approved. **BOTH TRACKS SHIPPED 2026-07-25.** Track B: B1 `76134365`, B2 `eae8724f`, B3 `56b13ccb`. Track A: A1 `222601b5`, A2 `626886ed`, A3 `5e98a6da`, A4 `1b51bc46`. Plain commits, no version bump per ADR 040 S8.
**Date:** 2026-07-25
**ADRs:** [041 — per-principal token budgets](decisions/041-per-principal-token-budgets.md), [042 — context-window discovery + exact token counting](decisions/042-context-window-discovery-and-token-counting.md), [007 amended](decisions/007-api-metering-and-quotas.md) (#8 closed by obsolescence).

## Scope

Two independent tracks. Track B is what the consuming project actually needs and has no dependency on Track A, so it ships first.

**Track B — context discovery (ADR 042, amended):** publish the two limits a client needs (`context_length`, effective `max_prompt_tokens`) on `GET /v1/models{,/id}`, and add `POST /v1/chat/completions/count_tokens` `[native]` so a client can measure a prompt exactly before sending it. A downstream consumer's requirement spec (2026-07-25) pins the semantics; its post-turn-usage ask (`stream_options.include_usage`) **already ships** — verified at `AthenaServer+SSE.swift:388`, no work.

**Track A — token budgets (ADR 041):** per-principal rolling-period token budget, enforced with a 429 on the token-bearing routes, advertised via advisory headers on every metered response, administered through config + a per-user override.

Out of scope, recorded so it is not silently assumed: mid-stream budget cancellation; sliding-window accounting; per-token (as opposed to per-principal) budgets; metering the control plane (ADR 007 #8, closed); quota enforcement in loopback dev mode (structurally impossible per ADR 025 — there is no store and no principal); the Anthropic-dialect `POST /v1/messages/count_tokens` (deferred, ADR 042 §4(a) — same core plus the existing decoder, lands when a Messages-dialect consumer needs it); `max_output_tokens` on the model object (refused — no such cap exists, ADR 042 §4(d)); token ids or per-message breakdowns from the count route; image-bearing count requests (400, ADR 042 §4(b)).

## AGENTS.md rules this change touches

Surfaced per the gate, resolved in-plan rather than deferred:

| Rule | How this change honors it |
|---|---|
| All HTTP routes live in `OpenAPISpec.swift`, updated in the same edit | Every slice that adds a route or field edits the spec in the same commit; `deploy/e2e-rbac.sh`'s path-count drift guard is updated with it. |
| All errors use the `{"error":{message,type,code}}` envelope | `quota_exceeded` / `insufficient_quota` uses the envelope verbatim, mirroring `RateLimit.tooMany`. |
| `/v1` compatibility rule — non-OpenAI surfaces must be tagged in the introducing ADR, the spec description, and the AGENTS.md endpoint list | ADR 042 §2 marks the two model fields as native extensions on an `[oai]` route; `POST /v1/chat/completions/count_tokens` is tagged `[native]`. All three edits land with the change — the introducing ADR with B3, where the route lands, a commit after B2's two model fields. |
| No parallel implementation of a canonical pipeline | `count_tokens` reuses `container.prepare` (the request path's own tokenizer + template) rather than re-implementing counting; quotas reuse the `meter()` chokepoint and the `RateLimitMiddleware` shape. |
| ADR 009 — decision logic MLX-free and unit-pinned | `QuotaWindow` / `QuotaDecision` in `AthenaServerKit`; `ModelConfigInfo.maxPositionEmbeddings` and the effective-ceiling resolution in `AthenaCore`. All pinned by `./deploy/test.sh`. |
| ADR 040 S8 — versions are release events | Slices land as plain commits, **no `appVersion` bump, no per-slice tag**. A version bump happens only in an operator-approved release commit. |
| Passive oracle | No outbound calls added. |

## Slices

Each is one commit, test-pinned, landing direct to `main` per house workflow.

### B1 — context window in `ModelConfigInfo`
Add `maxPositionEmbeddings: Int?` to `Sources/AthenaCore/ModelConfigInfo.swift`, read top-level-then-`text_config` via the existing `int(_:)` accessor. Extend the existing `ModelConfigInfo` unit tests: top-level present, nested-only present, absent ⇒ nil, non-object config ⇒ nil.

### B2 — publish both limits on `/v1/models`
`OpenAIModel` gains `context_length: Int?` + `max_prompt_tokens: Int?` (both `omitted-when-nil`, so an existing consumer sees byte-identical JSON for a model whose config lacks the field). `openAIModel(_:_:)` resolves them per model: window from `ModelConfigInfo.read(modelDirectory:)`, ceiling from the configured `max_prompt_tokens` else `GovernorMemory.defaultPromptTokenCeiling(maxBufferBytes:)`, absent under the explicit `0` opt-out. `handleOpenAIModelRetrieve` already has the entry's directory; `handleOpenAIModelsList` reads config per entry (bounded by store size — the list already stat-walks). Spec + CLAUDE.md endpoint-list edits in the same commit, both fields marked native extensions on an `[oai]` route. Pure resolution (`effectiveCeiling(configured:derived:)`) unit-pinned.

### B3 — `POST /v1/chat/completions/count_tokens` `[native]`
Route on the OpenAI dialect reusing the existing `ChatCompletionRequest` decoder (`model`/`messages`/`tools`/`response_format`; generation params ignored, not rejected — a client reusing its outbound payload verbatim is the point). `InferenceModule` (LLM) gains `countPromptTokens(turns:tools:)` that calls the same `container.prepare` the request path uses and returns `lmInput.text.tokens.shape.last` — **shape, never `asArray`**, so nothing evaluates. Response `{"prompt_tokens": N}`. Obeys ADR 015 cold-load (can block on a load); **does not** acquire the ADR 029 gate (ADR 042 §4(b), with the tripwire recorded there); requires `inference`; **not** metered, **not** quota-enforced. Image content parts ⇒ 400 cause-naming refusal. Spec + `[native]` tag + CLAUDE.md endpoint-list entry in the same commit. Stub-module test asserts the route contract and the image refusal; exactness is proven by the DoD below, not a unit test.

### A1 — period accounting in the store
`usage_counters` gains `period_start` / `period_prompt_tokens` / `period_completion_tokens` via `try? ALTER TABLE`; `auth_users` gains nullable `token_budget`. `addUsage` bumps both lifetime and period counters, resetting the period trio when the stored `period_start` predates the current period. `UsageRow` carries the new fields. `QuotaWindow.periodStart(containing:window:)` + `nextRoll(after:window:)` are pure and unit-pinned (day/month boundaries, DST, month-end, epoch-zero legacy rows). Store tests: fresh row, same-period accumulate, period roll resets period-only and preserves lifetime, old DB opens with defaults.

### A2 — config + per-user budget admin
`token_budget` / `token_budget_window` through the 5 config touchpoints (`AthenaConfig` field + TOML parse + `DefaultConfig` + `ConfigEditor` get/set + `Load` flag plumbing); not deny-listed under ADR 037. Per-user override on the existing user-admin control-plane route + `athena` user CLI, audited as `user.budget`. Invalid window value fails config parse loudly rather than defaulting.

### A3 — enforcement + advisory headers
`QuotaMiddleware` in `AthenaServerKit` mirroring `RateLimitMiddleware` (auth-gated, principal-keyed): refuse with 429 `quota_exceeded` + `Retry-After` when accrued period usage ≥ resolved budget; otherwise attach `x-athena-tokens-{limit,remaining,reset}` to the response after the handler has metered. Scoped to the token-bearing routes only (chat completions, embeddings, messages) — explicitly not `RateLimitMiddleware.throttled()`, so an exhausted principal can still read `/api/usage`. `QuotaDecision.evaluate(used:budget:)` pure and unit-pinned including the unlimited (nil/0) and headers-omitted cases.

### A4 — surface the budget
`/api/usage`'s `UsageEntry` + `athena usage` gain `budget` / `period_tokens` / `period_reset` (omitted-when-nil). `athena doctor` notes that quotas are inert in loopback mode. Spec + `docs/` usage page in the same commit.

## Track B outcome (2026-07-25)

~~All three slices landed with their spec + CLAUDE.md edits in the same commit.~~ — **corrected #164:** B2 (`eae8724f`) and B3 (`56b13ccb`) each landed with both edits; B1 (`76134365`) owed neither — it adds no route and no surface field, only `ModelConfigInfo.maxPositionEmbeddings` and its tests, and touched no other file. Gates: `./deploy/test.sh` 748/0 (35 skipped), `./deploy/e2e-rbac.sh` 496/0 including the OpenAPI drift guard (which derives routes from source, so the new route needed no count bump), `./deploy/build.sh Release` green, and the new `deploy/e2e-count-tokens.sh` green against two real models.

The DoD model (`gemma-4-26b-a4b-it-8bit`) lives on the Studio appliance, not this workstation's store, so the heavy DoD ran on `Llama-3.2-3B-Instruct-8bit` (fast) and `gemma-4-31b-it-4bit` (same family, vision wrapper — which also exercises B1's nested-`text_config` read on a real checkpoint). Measured:

| Assertion | Llama-3.2-3B-Instruct-8bit | gemma-4-31b-it-4bit |
|---|---|---|
| `context_length` / `max_prompt_tokens` present | 131072 / 26008 | 262144 / 26008 |
| count == `usage.prompt_tokens`, same body with a tool | 279 == 279 | 146 == 146 |
| image parts refused | 400 `image_count_unsupported` | 400 `image_count_unsupported` |
| idle count latency | 38 ms | 38 ms |
| count issued mid-decode | 7.4 s | 34.8 s |

Two findings worth carrying forward:

**The ceiling is the real budget, and it is ~10× below the advertised window.** Both models report `max_prompt_tokens = 26008` — the ADR 030 device-derived floor on this box — against checkpoints advertising 131k and 262k. A client budgeting on `context_length` alone would over-plan by an order of magnitude. This was the risk listed below; it is now measured, and setting `max_prompt_tokens` deliberately is the open operator decision.

**Counting is cheap but not concurrent with generation.** ADR 042 §4(b) claimed a count would never queue behind another caller's decode. The no-gate/no-eval half holds (38 ms idle); the concurrency half does not — `ModelContainer.prepare` needs the substrate's `SerialAccessContainer` mutex, which a generation holds for its whole decode, so a mid-decode count waits it out. ADR 042 §4(b) is amended with the measurement, every client-facing surface says so, and the e2e script reports the latency rather than asserting a bound. Fixing it is an upstream `mlx-swift-lm` change; re-implementing tokenization outside `container.prepare` is explicitly ruled out (it would forfeit the exactness the route exists for).

## Track A outcome (2026-07-25)

Gates: `./deploy/test.sh` 787/0 (35 skipped), `./deploy/e2e-rbac.sh` 496/0, `./deploy/build.sh Release` green, and the new `deploy/e2e-token-budget.sh` **31/0** against a real auth-on daemon (`--engine stub` — the budget algebra is engine-independent and the stub meters real token counts, so the DoD needs no resident model).

Every Track A DoD assertion passed as specified, with one deliberate substitution: the budget under test is a **per-user override of 5 tokens with no global default**, rather than a global 100. It exercises strictly more (the override path *and* that an un-overridden user stays unlimited), and 5 tokens is what makes a single stub request exhaust it.

| DoD assertion | Result |
|---|---|
| First request succeeds, `x-athena-tokens-remaining` below the limit | 200, `limit=5 remaining=0` |
| Next request 429 `quota_exceeded`, `Retry-After` before local midnight | 429, `insufficient_quota`, `Retry-After=36725s` (≤ 36726s to midnight) |
| `GET /api/usage` still succeeds while exhausted, `period_tokens ≥ budget` | 200, `budget=5 period_tokens=14` |
| A second user with no override and no global budget is unaffected | 200, and **no** `x-athena-tokens-*` headers or usage budget fields |
| Rewinding `period_start` one period restores service; period resets, lifetime preserved | 200; period 14, lifetime 28, requests 2 |
| (added) `count_tokens` is not quota-refused while exhausted | 501 from the stub engine — not 429 |

Two implementation notes worth carrying:

**`QuotaWindow` landed in `AthenaCore`, not `AthenaServerKit`** as the ADR wrote. Both the config editor (`AthenaDeploy`) and the store's caller need the case list, and `AthenaDeploy` cannot depend on `AthenaServerKit`. This is the house pattern for enum-ish config values (`Engine`, `KVCompression`, `AdmissionMode`); `QuotaDecision` and the middleware stayed server-side.

**`putUser` was `INSERT OR REPLACE`**, which deletes the row — a password change would have silently wiped the new `token_budget` column. Fixed to an upsert of the credential columns, test-pinned. Any future additive `auth_users` column would have hit the same trap.

## Test bar

- `./deploy/test.sh` green, including new unit tests for every pure seam named above (`QuotaWindow`, `QuotaDecision`, `effectiveCeiling`, `ModelConfigInfo.maxPositionEmbeddings`) and the store's period-roll behavior.
- `./deploy/e2e-rbac.sh` green with its path-count guard updated for the one new route.
- Release build via `./deploy/build.sh Release` (Metal shaders need xcodebuild).
- New `deploy/e2e-token-budget.sh` (auth-on) and `deploy/e2e-count-tokens.sh` (loopback) carrying the DoDs below.

## Definition of Done (discriminating — must fail before, pass after)

**Track B.** Against a resident `gemma-4-26b-a4b-it-8bit`: `GET /v1/models/{id}` returns both `context_length` and `max_prompt_tokens` (pre-change: neither key exists). Then, for a fixed multi-turn body carrying a system prompt **and at least one tool definition**, `POST /v1/chat/completions/count_tokens` returns `prompt_tokens` **exactly equal** to the `usage.prompt_tokens` returned by `POST /v1/chat/completions` for the identical body (pre-change: the route 404s). Equality — not proximity — is the assertion; it is the whole value of routing both through `container.prepare`, and the tool definition is what proves tool-schema tokens are counted (the consumer's stated make-or-break requirement). Third assertion, same script: the count call while a long decode is in flight returns **without waiting for it** (proves no gate acquisition, ADR 042 §4(b)).

**Track A.** With auth on, `token_budget_window = "day"` and a user budget of 100 tokens: the first `/v1/chat/completions` succeeds and its `x-athena-tokens-remaining` header is below 100; a subsequent request returns 429 with `code:"quota_exceeded"` and a `Retry-After` that lands before tomorrow's local midnight; `GET /api/usage` for that principal still succeeds while exhausted (proving enforcement scope) and reports `period_tokens ≥ budget`; a second user with no override and no global budget is unaffected. Pre-change: no header, no 429, no fields. Additionally, rewinding the stored `period_start` by one period and re-requesting succeeds again with a reset `period_tokens` while lifetime `total_tokens` is unchanged (proving the roll preserves lifetime accounting).

## Sequencing

B1 → B2 → B3 unblocks the consuming project and is independently shippable. A1 → A2 → A3 → A4 follows. Track A's DoD requires an auth-on daemon (Mac Studio appliance per the manual integration-test tier); Track B's runs on loopback.

## Risks

- **B2 reads `config.json` per entry on `GET /v1/models`.** Bounded by store size and already-stat-walked; if a large store makes the list call slow, cache per (path, mtime) — not built pre-emptively.
- **B3 can cold-load a model** to reach its tokenizer. Inherent to exactness (ADR 042); documented on the route, bounded by `cold_load_wait_secs`.
- **The derived prefill ceiling may be the real budget, and it is low.** `sqrt(maxBufferSize / 128)` lands near ~17k tokens on a 48 GB device and ~28k on 128 GB, against checkpoints advertising 128k+. B2 makes this visible for the first time, which is the point — but a consumer that budgets on `context_length` alone will over-plan by an order of magnitude. The operator should decide `max_prompt_tokens` deliberately rather than inherit an OOM-safety floor as a context policy. Not a blocker for B2/B3; it is the first thing to look at once the number is published.
- **A3's overshoot** is bounded but real (ADR 041 §3 honesty boundary). Documented, not fixed.
- **Quotas are inert in loopback** (ADR 025). Doctor note + docs lead with it, or an operator will assume protection they do not have.
