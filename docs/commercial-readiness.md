# Athena commercial-readiness roadmap — M27+ scoping & fresh-chat kickoffs

Review-only planning artifact (drafted 2026-05-22, at v0.9.92). It scopes the
must-have commercial-host gaps from the readiness review into milestones and
gives each a **self-contained kickoff prompt** to seed a fresh Claude Code chat,
following the established M15–M18 pattern: *read memory first → AskUserQuestion
design-fork → sub-sliced build-e2e-ship*.

**Milestone numbers (M27–M34) are proposed, not locked.** The user drives
versioning and maturity framing — do not assume the next tag number; ask. These
are NOT version tags.

---

## Sequencing

Trust/security first, then accounting, then API-completeness/DX, then
reliability/retention (per the review). Dependency notes inline.

| # | Milestone | Sev | Depends on |
|---|---|---|---|
| M27 | Usage metering & token accounting | must | — (feeds M29 quotas, M30 principal reuse) |
| M28 | TLS / transport security | must | — |
| M29 | Rate limiting, quotas & concurrency caps | must | M27 for *token-budget* quotas only |
| M30 | Audit log | must | — |
| M31 | OpenAI surface completeness (`/v1/models` + …) | must | — |
| M32 | OpenAPI spec + developer docs | must | M31 (spec should reflect final routes) |
| M33 | Reliability hardening (timeout/drain/preload) | should | — (also lands the `--version` fix) |
| M34 | Data retention & at-rest posture | should | M27 (metering rows share the store) |

A defensible **v1.0-candidate cut line** sits after M32 (security + metering +
discoverable/documented API); M33–M34 are the stability follow-on. *(User's call
on the actual version.)*

---

## SHARED PREAMBLE — paste at the top of every kickoff below

```
START BY READING (memory persists across chats):
- ~/.internal/projects/-Users-goodolclint-Source-Athena/memory/MEMORY.md, then
  reference_commercial-readiness.md and the project entries it links. Know what
  already exists before building.

THESIS / LOCKED CONSTRAINTS (do NOT break; do NOT relitigate):
- Passive-oracle: the daemon initiates NO outbound connections except (a)
  model-weight fetch from HF and (b) the opt-in remote-syslog carve-out. No
  billing/result webhooks — anything a client needs is pull / long-poll / SSE.
- Single native macOS/MLX daemon (Apple-only). The `athena` CLI client is
  cross-platform (clients/ standalone SwiftPM package); the daemon (`athenad`)
  is not. Any AthenaClient change ⇒ re-prove the standalone package on Linux
  (docker swift:6.1).
- Config = TOML. Any new key follows the config-surface 5-touchpoint pattern
  (AthenaConfig field+parse, LaunchdPlist forward, ConfigEditor knownKeys+value
  switch, deploy/athena.toml documented, AthenaDeployTests). Precedence for any
  env override is env > TOML > built-in default.
- Mythology-derived identifiers (ports/sentinels/codenames) with an explainable
  etymology — never borrow another tool's value.
- Standing no-TLS caveat applies until M28 ships: bearer/cookie are plaintext.

RELEASE DISCIPLINE:
- Ship-small: each slice = one conventional commit + one annotated semantic tag
  pushed direct to origin/main (NO PRs, NO lightweight tags; tag body =
  milestone description, no vX.Y.Z prefix). Pre-commit order Tests → Security →
  Quality → Refactor.
- Conservative versioning: do NOT declare 1.0 or "production-ready"; ASK the
  user before choosing the next tag number.
- Bump the version string in BOTH Sources/athena/Athena.swift AND
  clients/Sources/athena/Athena.swift each slice. (NOTE: these are currently
  stale at 0.9.81 vs tag v0.9.92 — fix as part of your first slice.)
- Commit/tag/PR text: frame by the Athena-internal capability/correctness
  reason. NEVER name the the consuming application consumer repo in git history.

BUILD / TEST:
- MLX needs full Xcode: build via ./deploy/build.sh Release (xcodebuild,
  -derivedDataPath .build/xcode), NOT bare `swift build`. SwiftPM schemes have
  NO test action — pure tests run via `swift test`; real-model validation uses
  the Release binary + curl (not --engine stub, which stubs aux modules).
- Extend deploy/e2e-rbac.sh with a new phase (curl, stub engine, loopback) for
  every slice; that's the per-PR gate. Real-model/2-node checks go in
  deploy/integration/RUNBOOK.md (manual pre-release tier, not CI).

WORKFLOW: open with AskUserQuestion on the design fork(s) below (present the
alternatives — the user engages and may overrule — but lead toward the
recommendation). Then sub-slice and ship.
```

---

## M27 — Usage metering & token accounting

**Goal.** Real `prompt/completion/total_tokens` in every `usage` field, plus
per-principal persisted token+request counters retrievable locally.

**Why must-have.** `usage` is hardcoded to zeros (AthenaServer.swift:578-579 chat,
:630-631 embeddings); `AthenaMetrics.addTokens()` (Metrics.swift:37) is never
called so `llmTokens` is always 0. No metering ⇒ no billing, no quotas, no
capacity planning — the single biggest commercial gap.

**Design fork (ask first).** *How far on metering exposure?*
- **A (recommend):** real token counts in `usage` + revive metrics token counter
  + a NEW per-principal counter table in AthenaStore + `GET /api/usage` (pull,
  owner-scoped like the queue: self by default, admin sees all) + an `athena
  usage` CLI verb (overload local/remote per the cli-verb-overload rule).
- **B (minimal):** fix the `usage` field + global metrics only; defer
  per-principal.
- **C (full ledger):** A plus per-model and time-bucketed (hourly/daily) rollups
  for billing-grade reporting.

**Proposed slices.**
- M27.1 — Thread true token counts out of the generate path into `usage`
  (chat + embeddings) and call `metrics.addTokens`. (Tokenizer/iterator already
  counts; surface prompt vs completion.)
- M27.2 — Per-principal counters in AthenaStore (new table keyed by the
  `AuthSubject.principal` `u:`/`t:` id); increment in the handler/queue paths.
- M27.3 — `GET /api/usage` (owner-scoped, admin-all) + `athena usage`.
- M27.4 — Streaming `usage` via `stream_options.include_usage` (final chunk) + e2e.

**Files.** AthenaServer.swift (chat/embeddings handlers, streamSSE:2531),
Metrics.swift:37, OpenAIDTO.swift `Usage`, Sources/AthenaStore (new table +
accessors), Auth.swift (principal). **Thesis:** pull/stored only — no outbound.

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M27 — usage metering & token accounting. At v0.9.92 the OpenAI `usage` object is
hardcoded {0,0,0} (AthenaServer.swift:578-579 and :630-631) and
AthenaMetrics.addTokens (Metrics.swift:37) is dead — Athena reports no token
usage anywhere, so billing/quotas are impossible.

Build real per-request token accounting AND per-principal persisted metering,
retrievable LOCALLY (passive-oracle: pull only, never an outbound billing
callback). Start with AskUserQuestion on the exposure fork: (A, recommended)
real counts in `usage` + revive the metrics token counter + a new per-principal
counter table in AthenaStore + `GET /api/usage` (owner-scoped like the M12.6
queue: self by default, admin sees all) + an `athena usage` CLI verb overloading
local/remote per DaemonOptions.isRemote; (B) just fix `usage` + global metrics;
(C) full per-model + time-bucketed ledger. Lead toward A.

Then sub-slice: (1) thread true prompt/completion token counts out of the
generate path into `usage` for /v1/chat/completions + /v1/embeddings and call
metrics.addTokens; (2) per-principal counters in AthenaStore keyed by
AuthSubject.principal; (3) GET /api/usage + athena usage; (4)
stream_options.include_usage. Each slice: xcodebuild Release + a new
deploy/e2e-rbac.sh phase (assert non-zero usage and owner-scoped /api/usage) +
annotated tag direct-to-main. Ask me before choosing tag numbers.
```

---

## M28 — TLS / transport security

**Goal.** A first-class HTTPS story so bearer tokens/cookies aren't plaintext on
a non-loopback bind — resolving the standing no-TLS caveat.

**Why must-have.** No in-daemon TLS; deploy/athena.toml has no `tls_*` key. Today
the only safe non-loopback path is an undocumented reverse proxy.

**Design fork (ask first).** *Termination model?*
- **A:** in-daemon TLS only (Hummingbird/NIOSSL behind `tls_cert`/`tls_key`).
- **B:** blessed reverse-proxy only (document caddy/nginx; no daemon code).
- **C (recommend):** both — optional in-daemon TLS via config keys AND a
  documented reverse-proxy guide, so an appliance works standalone but slots
  behind existing TLS infra too.

**Proposed slices.**
- M28.1 — Hummingbird TLS via swift-nio-ssl behind `tls_cert`/`tls_key` TOML
  keys (5-touchpoint); interplay with the fail-safe auth gate (TLS doesn't change
  the no-auth-on-non-loopback refusal).
- M28.2 — `athena doctor` TLS check (cert readable, not expired, key perms 600)
  + cert/key path validation.
- M28.3 — Reverse-proxy guide in docs/ + e2e (curl https against a self-signed
  cert on the stub daemon).

**Files.** Package.swift (swift-nio-ssl), AthenaServer.swift `Application(...)`
config, deploy/athena.toml, Doctor.swift, Load.swift. **Thesis:** inbound only —
no conflict. **Naming:** no new sentinels; if a default TLS port is wanted, derive
it (mythology rule). **Dep note:** adds swift-nio-ssl; keep daemon-only (don't
pull it into the portable client graph).

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M28 — TLS / transport security. At v0.9.92 there is NO in-daemon TLS (athena.toml
has no tls key); bearer tokens and the WebUI session cookie are plaintext over
HTTP — the standing no-TLS caveat. A commercial host needs HTTPS.

Give Athena a first-class TLS story. Start with AskUserQuestion on the
termination fork: (A) in-daemon TLS only; (B) blessed reverse-proxy only,
documented; (C, recommended) BOTH — optional in-daemon TLS via tls_cert/tls_key
TOML keys plus a documented reverse-proxy guide. Lead toward C.

Then sub-slice: (1) wire Hummingbird/swift-nio-ssl TLS behind tls_cert/tls_key
config keys (config-surface 5-touchpoint; keep swift-nio-ssl out of the portable
client graph); preserve the fail-safe gate (no-auth + non-loopback still
refuses). (2) athena doctor TLS posture check (cert exists/readable/unexpired,
key perms). (3) reverse-proxy guide in docs/ + an e2e that curls https against a
self-signed cert on the stub daemon. Inbound-only ⇒ passive-oracle intact. Each
slice = xcodebuild Release + e2e phase + annotated tag. Ask before tag numbers.
```

---

## M29 — Rate limiting, quotas & concurrency caps

**Goal.** Protect the box from abuse: per-principal rate limits + concurrency
caps, composing with the existing governor 503 backpressure.

**Why must-have.** No throttle/quota anywhere. Mitigations today: request-size
caps (4 MB chat/embed, 25 MB audio), governor Metal-OOM→503, serial queue worker.
Nothing stops one key from flooding the sync path.

**Design fork (ask first).** *Scope + enforcement?*
- **A (recommend):** per-principal token-bucket rate limit + a global + per-
  principal concurrency cap; reject with `429` + `Retry-After`. Quota *budgets*
  (daily token caps) deferred until M27 metering exists.
- **B:** global rate limit only.
- **C:** A plus token-budget quotas now (requires M27 first).

**Proposed slices.**
- M29.1 — `RateLimitMiddleware` (after auth, before metrics) keyed by
  `AuthSubject.principal`; token-bucket; 429 + Retry-After + OpenAI error body.
- M29.2 — Concurrency cap (per-principal + global semaphore) composing with the
  governor 503 (distinguish "busy, retry" from "over budget").
- M29.3 — Config keys (rate, burst, concurrency) 5-touchpoint + e2e (hammer a
  key → 429; verify Retry-After).

**Files.** New middleware in Sources/athena/Server, AthenaServer.swift router
order, Auth.swift principal, deploy/athena.toml. **Thesis:** inbound gating — OK.
**Dep note:** loopback/auth-off single-tenant mode should bypass limits (dev).

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M29 — rate limiting, quotas & concurrency caps. At v0.9.92 there is no per-key or
per-user rate limiting, quota, or concurrency cap; the only protections are
request-size caps and the governor's Metal-OOM→503. One key can flood the sync
path.

Add abuse protection that composes with the governor. Start with AskUserQuestion
on the scope fork: (A, recommended) per-principal token-bucket rate limit + a
global + per-principal concurrency cap, rejecting with 429 + Retry-After, with
token-BUDGET quotas deferred to after M27 metering; (B) global rate limit only;
(C) include token-budget quotas now (needs M27). Lead toward A.

Then sub-slice: (1) a RateLimitMiddleware inserted after AuthMiddleware, keyed by
AuthSubject.principal, token-bucket, 429 + Retry-After using the standard
{error:{message,type,code}} body; auth-off loopback bypasses (dev). (2)
per-principal + global concurrency semaphore, distinct from the governor 503. (3)
config keys (rate/burst/concurrency) via the 5-touchpoint pattern + an
e2e-rbac.sh phase that trips a 429 and checks Retry-After. Inbound-only.
Each slice = xcodebuild Release + e2e + annotated tag. Ask before tag numbers.
```

---

## M30 — Audit log

**Goal.** An append-only record of who did what when for RBAC + admin mutations
— security/compliance evidence.

**Why must-have.** No audit anywhere. RBAC mutations only emit ordinary app logs;
there's no queryable record of user/role/token/model/daemon changes.

**Design fork (ask first).** *Sink?*
- **A:** dedicated SQLite audit table (queryable, retention-bounded) +
  `GET /api/audit` pull.
- **B:** structured unified-log lines only (rides M10 + the opt-in remote-syslog
  carve-out).
- **C (recommend):** both — authoritative pull-queryable table AND a log line
  (so it can also ride the sanctioned syslog egress).

**Audit points.** handleUserCreate (2141), handleUserDelete (2210),
handleRoleGrant (2233), handleRoleRevoke (2273), handleTokenCreate (2312),
handleTokenDelete (2393); admin ops: handleModelRemove (1762),
handleDefaultModelSet (1839), daemon stop/load. Record principal / action /
target / result / timestamp. Both the Bearer `/api/*` and cookie `/ui/*` paths.

**Proposed slices.**
- M30.1 — Audit table + writer; emit at every mutation chokepoint (reuse the
  shared op layer so /api and /ui both record).
- M30.2 — `GET /api/audit` (admin-only; filter by principal/action/time) +
  `athena audit` CLI.
- M30.3 — Retention bound + e2e (mutate → assert an audit row with the right
  actor).

**Files.** Sources/AthenaStore (table+accessors), AthenaServer.swift mutation
handlers, AuthCmd/RemoteAuth (CLI). **Thesis:** local store + existing syslog
carve-out — OK.

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M30 — audit log. At v0.9.92 there is no audit trail: RBAC and admin mutations
(user/role/token create+delete, model rm, default-model set, daemon stop/load)
leave only ordinary app-log lines, nothing queryable. A commercial/compliance
buyer needs who-did-what-when.

Build an append-only audit log. Start with AskUserQuestion on the sink fork: (A)
a dedicated SQLite audit table + GET /api/audit; (B) structured unified-log lines
only (riding M10 + opt-in remote syslog); (C, recommended) BOTH. Lead toward C.

Then sub-slice: (1) audit table + writer in AthenaStore, emitted at the mutation
chokepoints — handleUserCreate/Delete, handleRoleGrant/Revoke,
handleTokenCreate/Delete, handleModelRemove, handleDefaultModelSet, daemon
stop/load — recording principal/action/target/result/timestamp for BOTH the
Bearer /api/* and cookie /ui/* callers (record in the shared op layer so neither
path is missed). (2) GET /api/audit (admin-only, filterable) + athena audit CLI.
(3) retention bound + an e2e phase asserting a mutation produces the right audit
row. Local store ⇒ passive-oracle intact. Each slice = xcodebuild Release + e2e +
annotated tag. Ask before tag numbers.
```

---

## M31 — OpenAI surface completeness

**Goal.** Close the OpenAI-compat gaps that break drop-in SDK/LiteLLM use:
`/v1/models`, `finish_reason:"length"`, and the safe sampling params.

**Why must-have.** No `GET /v1/models` (only native `/api/models` at :335) — many
clients probe it for discovery/validation. `finish_reason` is only
"stop"/"tool_calls" (never "length"), and only `max_tokens`/`temperature` are
honored (OpenAIDTO.swift:62).

**Design fork (ask first).** *How far on sampling params, given the greedy/MTP/
structured-output determinism thesis?*
- **A (recommend):** `/v1/models`+`:id`, `finish_reason:"length"`, and the safe
  params (`stop`, `seed`, `top_p`); explicitly reject `n>1`/`logprobs`/`logit_bias`
  with a clear 400.
- **B (minimal):** `/v1/models` + `finish_reason:"length"` only.
- **C:** also attempt `n>1`/`logprobs` (needs a real design — conflicts with
  greedy/MTP/structured; likely a separate milestone).

**Proposed slices.**
- M31.1 — `GET /v1/models` + `GET /v1/models/:id` (OpenAI list/retrieve shape)
  mapping onto ModelStoreOps; gate `model.read`.
- M31.2 — `finish_reason:"length"` when generation hits max_tokens (sync + stream).
- M31.3 — Plumb `stop`/`seed`/`top_p` into the generate path where the substrate
  supports them; reject unsupported params with a clear, documented 400; e2e.

**Files.** OpenAIDTO.swift (request fields, a `ModelObject`), AthenaServer.swift
(new /v1/models routes + chat handler), AthenaLLM ModelStoreOps. **Thesis:**
neutral.

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M31 — OpenAI surface completeness. At v0.9.92 there is no GET /v1/models (only
native /api/models at AthenaServer.swift:335), finish_reason is only
"stop"/"tool_calls" (never "length"), and only max_tokens+temperature are honored
(OpenAIDTO.swift:62). These break drop-in OpenAI SDK / LiteLLM clients.

Close the OpenAI-compat gaps. Start with AskUserQuestion on the sampling-params
fork given Athena's greedy/MTP/structured-output determinism: (A, recommended)
add /v1/models(+:id), finish_reason:"length", and the safe params stop/seed/
top_p, while rejecting n>1/logprobs/logit_bias with a clear 400; (B) /v1/models +
finish_reason:"length" only; (C) attempt n>1/logprobs (separate design — they
fight greedy/MTP/structured). Lead toward A.

Then sub-slice: (1) GET /v1/models + GET /v1/models/:id in the OpenAI list/
retrieve shape, mapping onto ModelStoreOps, gated model.read; (2)
finish_reason:"length" on max_tokens truncation (sync + SSE); (3) plumb
stop/seed/top_p into generate where the substrate supports them and 400 the
unsupported params. Each slice = xcodebuild Release + e2e phase + annotated tag.
Ask before tag numbers.
```

---

## M32 — OpenAPI spec + developer docs

**Goal.** A machine-readable API spec and human onboarding docs — neither exists.

**Why must-have.** No OpenAPI/Swagger anywhere; no README/CLAUDE.md/quickstart
in-repo (only docs/the consuming application-integration.md + this file). Integrators have no
reference.

**Design fork (ask first).** *Spec authoring?*
- **A (recommend):** hand-authored static OpenAPI 3 doc served at `/openapi.json`
  + a README + docs/quickstart. Zero new deps (matches the zero-deps ethos).
- **B:** code-generated spec (needs a framework/macro — heavier dep).
- **C:** spec file only, not served.

**Proposed slices.**
- M32.1 — Hand-authored openapi.json covering all `/v1/*` + `/api/*` routes, the
  bearer scheme, and the error shape; serve at `/openapi.json`.
- M32.2 — README + docs/quickstart (install → seed admin → first request → WebUI
  → TLS pointer).
- M32.3 — Drift guard: a test asserting every `router.add` path appears in the
  spec; e2e (fetch + parse /openapi.json).

**Files.** New static resource + a route in AthenaServer.swift, README.md,
docs/. Sequence AFTER M31 so the spec reflects the final routes. **Thesis:**
neutral (static spec, no dep).

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M32 — OpenAPI spec + developer docs. At v0.9.92 there is no OpenAPI/Swagger spec
and no README/quickstart in-repo (only docs/the consuming application-integration.md and
docs/commercial-readiness.md). Integrators have no reference. Best done after M31
so the spec reflects the final OpenAI routes.

Give Athena a machine-readable spec + onboarding docs. Start with AskUserQuestion
on the authoring fork: (A, recommended) a hand-authored static OpenAPI 3 doc
served at /openapi.json + README + docs/quickstart, zero new deps; (B)
code-generated (heavier dep); (C) spec file only, unserved. Lead toward A
(matches the zero-deps ethos).

Then sub-slice: (1) hand-author openapi.json for all /v1/* and /api/* routes +
the bearer security scheme + the {error:{message,type,code}} shape, served at
/openapi.json; (2) README + docs/quickstart (install → seed admin → first curl →
WebUI → TLS); (3) a drift-guard test asserting every router.add path is in the
spec + an e2e that fetches and parses /openapi.json. Each slice = xcodebuild
Release + e2e + annotated tag. Ask before tag numbers.
```

---

## M33 — Reliability hardening

**Goal.** Per-request inference timeout + explicit graceful in-flight draining +
optional model preload-on-start. (Also lands the stale `--version` fix.)

**Why should-have.** No per-request inference deadline (only the queue long-poll
deadline at AthenaServer.swift:1284) — a runaway generation is bounded only by
max_tokens. Graceful shutdown relies on Hummingbird `runService()` + the `stop`
5 s SIGTERM→SIGKILL window (DaemonLifecycle.swift:180-189) with no explicit
in-flight drain. Models load lazily on first request (no boot warmup).

**Design fork (ask first).** *Bundle or split?*
- **A (recommend):** all three (timeout + explicit drain + preload) as one
  reliability milestone.
- **B:** per-request timeout only.
- **C:** graceful drain only.

**Proposed slices.**
- M33.1 — Per-request inference timeout (config `request_timeout_secs`; wrap
  generate in a deadline; classified 504/AthenaError). **Also fix the version
  string** (0.9.81 → current) in both Athena.swift files here.
- M33.2 — Explicit graceful shutdown: track in-flight requests + the queue worker
  via ServiceLifecycle so SIGTERM drains before exit; verify within the stop 5 s
  window.
- M33.3 — Optional preload-on-start (config `preload_model`: warm the default at
  boot vs lazy) + e2e.

**Files.** AthenaServer.swift (chat handler, run() runService), DaemonLifecycle.swift,
Load.swift, deploy/athena.toml, both Athena.swift version strings. **Thesis:**
neutral.

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M33 — reliability hardening. At v0.9.92 there is no per-request inference timeout
(only the queue long-poll deadline at AthenaServer.swift:1284; a runaway
generation is bounded only by max_tokens), no explicit in-flight draining on
shutdown (just Hummingbird runService + stop's 5s SIGTERM→SIGKILL at
DaemonLifecycle.swift:180-189), and models load lazily with no boot warmup. Also,
`athena --version` is stale (0.9.81 vs tag v0.9.92) in both
Sources/athena/Athena.swift:19 and clients/Sources/athena/Athena.swift:17 —
breaking the deploy-install-hygiene `--version` pre-install guard.

Harden runtime reliability. Start with AskUserQuestion on bundling: (A,
recommended) timeout + explicit graceful drain + optional preload as one
milestone; (B) timeout only; (C) drain only. Lead toward A.

Then sub-slice: (1) per-request inference timeout (config request_timeout_secs;
deadline-wrap the generate path; classified 504) AND fix the version strings; (2)
explicit graceful shutdown — track in-flight requests + the queue worker via
ServiceLifecycle so SIGTERM drains within the stop window; (3) optional
preload-on-start config + e2e. Each slice = xcodebuild Release + e2e + annotated
tag. Ask before tag numbers.
```

---

## M34 — Data retention & at-rest posture

**Goal.** Bounded retention for stored inference artifacts + a documented
at-rest-encryption posture.

**Why should-have.** Queue results persist forever until manual `queue rm`
(RequestQueue.swift has no TTL) — and result blobs hold inference outputs. Vectors
and the store grow unbounded. The SQLite store is plaintext on disk (passwords
PBKDF2 + tokens SHA-256 + Keychain secrets are fine; vector/queue blobs are not),
relying on FileVault.

**Design fork (ask first).** *Retention only, or also encrypt-at-rest?*
- **A (recommend):** retention/TTL sweeper for queue results + vectors (config
  keys) + content-logging opt-out + a documented FileVault-reliance posture +
  doctor check. Flag SQLCipher as a follow-on if a buyer requires app-level
  encryption.
- **B:** retention only.
- **C:** A plus SQLCipher at-rest encryption now (new dep, key-management design).

**Proposed slices.**
- M34.1 — Queue-result TTL + retention sweeper (config `queue_result_ttl`, max
  rows); runs in the worker idle path.
- M34.2 — Vector/store retention + content-logging opt-out controls.
- M34.3 — At-rest posture doc + `athena doctor` FileVault check + e2e.

**Files.** RequestQueue.swift, Sources/AthenaStore, Doctor.swift,
deploy/athena.toml, docs/. **Thesis:** local — OK.

**Kickoff prompt** (prepend the SHARED PREAMBLE):
```
M34 — data retention & at-rest posture. At v0.9.92 queue results persist forever
until manual `queue rm` (RequestQueue.swift has no TTL) and the blobs hold
inference outputs; vectors/store grow unbounded; the SQLite store is plaintext on
disk (passwords PBKDF2 + tokens SHA-256 + Keychain secrets are fine, but
vector/queue blobs rely on FileVault).

Add bounded retention + a clear at-rest posture. Start with AskUserQuestion on
the encryption fork: (A, recommended) retention/TTL sweeper for queue results +
vectors (config keys) + content-logging opt-out + documented FileVault reliance +
a doctor check, with SQLCipher flagged as a follow-on; (B) retention only; (C)
add SQLCipher app-level encryption now (new dep + key management). Lead toward A.

Then sub-slice: (1) queue-result TTL + retention sweeper (config
queue_result_ttl + max rows) in the worker idle path; (2) vector/store retention
+ content-logging opt-out controls; (3) at-rest posture doc + athena doctor
FileVault check + e2e. Local store ⇒ passive-oracle intact. Each slice =
xcodebuild Release + e2e + annotated tag. Ask before tag numbers.
```

---

## Appendix — quick fix found during scoping

`athena --version` reports **0.9.81** (Sources/athena/Athena.swift:19,
clients/Sources/athena/Athena.swift:17) but the latest tag is **v0.9.92** — M24–M26
bumped tags without bumping the in-source string, despite the per-slice
discipline. This breaks the deploy-install-hygiene `<binary> --version`
pre-install guard against the stale-binary trap. Either fix immediately as a
standalone patch or fold into M33.1 (noted there).
