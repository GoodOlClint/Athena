# Athena

Single native macOS/MLX daemon providing LLM chat, text embeddings, audio transcription/diarization/speaker-embeddings, a built-in vector database, and an async job queue — all governed by one Metal memory budget. Passive oracle: answers inbound requests only.

## Architecture

- Swift package (`Package.swift`, `Sources/`) targeting macOS on Apple Silicon. Build requires full Xcode (Command-Line Tools alone cannot compile the MLX Metal shaders).
- HTTP daemon on `127.0.0.1:7447` by default. Bearer-token RBAC; auth disabled in loopback dev mode.
- Modules: `AthenaLLM`, `AthenaEmbedding`, `AthenaTranscription` (each in `Sources/`).
- Cross-platform client CLI lives under `clients/` (Swift package, builds on Linux/Windows).
- Rust shim in `rust-shim/` for components not yet Swift-native.

## Canonical pipelines

- **All HTTP routes are defined in `Sources/athena/Server/OpenAPISpec.swift`.** Do not add or modify a route without updating that spec in the same edit. The daemon serves it verbatim at `GET /openapi.json`.
- **All errors return the envelope `{"error":{"message","type","code"}}`.** Never invent ad-hoc error shapes.
- **Outbound network is forbidden except model-weight fetches from Hugging Face** (and the opt-in remote-syslog sink). The "passive oracle" rule is binding — no result webhooks, no billing callbacks, no telemetry pings.
- **Native DTOs for the `/api/*` dialect live in `Sources/athena/Server/NativeAPIDTO.swift`.** Do not duplicate `/v1/*` OpenAI shapes there.

## Public surface

Athena is a passive oracle. Consumers (the consuming application, etc.) interact only via HTTP.

**Self-describing**: `GET /openapi.json` returns the full OpenAPI 3.0.3 spec for both dialects. Always reachable, no auth required. For static reading from another repo, load `Sources/athena/Server/OpenAPISpec.swift`.

**Two HTTP dialects**:

| Surface | Shape | Use for |
|---|---|---|
| `/v1/*` | OpenAI-compatible | Drop-in for OpenAI SDKs |
| `/api/*` | Athena native (control) | Daemon **control plane** — model-store, RBAC, allowlist, lifecycle, audit, usage, logs, cache. **NOT inference** — `/api/chat`+`/api/embed` are deprecated (ADR 013); new inference features go to `/v1` only. |

**Stable `/v1/*` endpoints** (canonical list in `OpenAPISpec.swift`):

- `POST /v1/chat/completions` — streaming SSE, tool calls, JSON-schema structured output, `stop`/`seed`/`top_p`
- `POST /v1/embeddings`
- `POST /v1/audio/transcriptions` — word timestamps, SRT/VTT
- `POST /v1/audio/diarizations`
- `POST /v1/audio/embeddings` (speaker embeddings)
- `POST /v1/vectors`, `POST /v1/vectors/query`, `/v1/vectors/{id}`, `/v1/vectors/stats`
- `/v1/store/export`, `/v1/store/stats`
- `/v1/queue`, `/v1/queue/{arg}`, `/v1/queue/{arg}/events` (SSE)
- `GET /v1/models`, `GET /v1/models/{id}`

**Native `/api/*`**: model-store and RBAC admin. Surface defined in `OpenAPISpec.swift` and `NativeAPIDTO.swift`.

**Auth**: `Authorization: Bearer <token>`. Each route requires a single RBAC permission. Loopback dev mode (no seeded users) opens every route.

**Diagnostics**: macOS unified log, subsystem `athena`. Off-box log shipping is operator-side — see `docs/logging.md`, `docs/logging-shipping.md`.

## Dependencies (consumed by this repo)

- Hugging Face — model weight fetches only. No other outbound dependencies.

## ADRs

Read `docs/decisions/` before architectural changes — particularly anything that would touch the passive-oracle rule, the OpenAPI spec, or the Metal memory governor.

- [`001-dflash-speculative-decoding.md`](docs/decisions/001-dflash-speculative-decoding.md) — DFlash lossless speculative decoding for non-MTP targets (Gemma4-first), default-off; vendored from bstnxbt/dflash-mlx (Apache-2.0). M63.
- [`002-gemma4-moe-arch.md`](docs/decisions/002-gemma4-moe-arch.md) — Gemma4 MoE architecture support (26B-A4B; 128-expert), additive substrate delta reusing `SwitchGLU`, dense path byte-unchanged; unblocks ADR 001's M63.5 (DFlash on the MoE target). M64.
- [`003-rust-shim-panic-strategy.md`](docs/decisions/003-rust-shim-panic-strategy.md) — rust-shim FFI panic strategy: per-entry `catch_unwind` (crate stays `panic="unwind"`, NOT `abort`) so a hostile schema degrades to an error, not a daemon abort. Shipped M65.1 v0.10.117 (audit G1).
- [`004-nonloopback-tls-posture.md`](docs/decisions/004-nonloopback-tls-posture.md) — non-loopback auth-on plaintext is **warn-only** (loud startup warning + doctor finding), not fail-closed; no `--insecure` gate, client https deferred; login limiter (A3) keys on TCP peer IP, no XFF trust. M65 (audit A2/K1/K8/A3).
- [`005-remove-secrets-from-argv.md`](docs/decisions/005-remove-secrets-from-argv.md) — hard-remove `--password` from argv; prompt / `--password-stdin` / env only (breaking CLI change). M66 (audit B2/K7/K13).
- [`006-vector-store-owner-scoping.md`](docs/decisions/006-vector-store-owner-scoping.md) — add a nullable `owner` column to the vector store; query/get/delete owner-filtered, admin sees all, legacy NULL rows admin-only (additive migration). M66 (audit H5).
- [`007-api-metering-and-quotas.md`](docs/decisions/007-api-metering-and-quotas.md) — bring native `/api` metering (#8) + token-budget quotas (#9) into the program as their own milestone; NA8 single-resolution folds in. M69+ (audit NA8).
- [`008-testable-server-seam.md`](docs/decisions/008-testable-server-seam.md) — extract the daemon's pure HTTP-server primitives (auth/ratelimit/concurrency/session/multipart/metrics/logging) into a new MLX-free `AthenaServerKit` library so they're unit-testable under `swift test`; chosen over `@testable import` of the `@main` executable (which would co-link the whole MLX/Metal graph). Pure refactor. Implemented M70.1 v0.10.144 (audit NA2/NB4).
- [`009-stub-decode-ci-tier.md`](docs/decisions/009-stub-decode-ci-tier.md) — **Accepted** (M70.2/.3, from v0.10.151): a stub-model CI tier that is a **control-flow / decision-algebra tier, not a numeric tier** — MLX can't evaluate *any* `MLXArray` under `swift test` (no metallib), so extract each decode path's MLX-free decision logic into pure-Swift static seams pinned by `./deploy/test.sh`; `MLXArray` numerics stay in `AthenaLLM` byte-unchanged. Resolves the L-cluster "CI blindness" (L1–L11, NC4/NC5/NC6).
- [`010-vision-input-vlm-chat.md`](docs/decisions/010-vision-input-vlm-chat.md) — **Accepted** (M71.1 v0.10.159 + M71.2 v0.10.160 shipped): add image input to chat by wiring the daemon to the substrate's already-implemented `MLXVLM.Gemma4` vision path (not a new encoder port); OpenAI `image_url` content-parts; base64/`data:` only (no outbound image fetch — passive-oracle preserved, 400 on `http(s)`); audio-in-chat deferred (substrate strips the audio tower, overlaps `/v1/audio/*`). Resolves issue #3. Plan: `docs/m71-vision-input-plan.md`.
- [`011-unified-memory-governor-as-thesis.md`](docs/decisions/011-unified-memory-governor-as-thesis.md) — **Accepted** (positioning): the **unified Metal memory governor is Athena's reason to exist**; audio/embeddings/vision/vectors/queue are *tenants* proving it across modalities; chat-GUI/model-discovery/download-UX are undifferentiated *tax* (maintain, don't out-polish LM Studio — a client, not a competitor). Coexist; **never compose at the inference layer** (two uncoordinated allocators on one Metal pool defeats the governor). Next milestone = **governor accounting truthfulness** (heartbeat RSS undercounts GPU). GPU-compute-extension governance model (in-process tenants vs. cooperative reservation) left **open**. Retire-tripwire: if audio + multi-tenant + near-ceiling concurrency lapse, retire for LM Studio + embeddings sidecar.
- [`012-vision-aware-convert.md`](docs/decisions/012-vision-aware-convert.md) — **Accepted** (M72.1 shipped v0.10.161): make `athena convert` vision-aware so `athena convert google/gemma-4-26B-A4B` produces a load/serve-compatible vision model. Route the convert load through `VLMModelFactory` (keep the `vision_tower`, not strip it); quantize the language model only (closure-form `quantize`, MoE mixed 4/8-bit) and leave the vision tower **full-precision** (no `.scales` ⇒ the `.scales`-driven loader skips it — matches `mlx-community/gemma-4-26b-a4b-it-4bit`); emit the matching per-layer quant config from one shared quant-rule. Serve side already validated (M71.2). Plan: `docs/m72-vision-convert-plan.md`. M72.
- [`013-v1-inference-surface-api-control-only.md`](docs/decisions/013-v1-inference-surface-api-control-only.md) — **Accepted** (staged rollout): **`/v1` is the single inference surface; `/api/*` is the control plane** (model-store/RBAC/lifecycle/audit/usage/logs/cache). Deprecate native inference `/api/chat`+`/api/embed` (verified: `/api/embed` 0 callers, `/api/chat` only `athena run` → migrate to `/v1`); removal is breaking, gated on external-consumer confirmation. Also: audio division-of-labor (`/v1/audio/*`=analysis, chat parts=reasoning), add `diarized_json` response_format (#4a), honor `logprobs` on the greedy path (stop 400-ing), reclassify client/precondition faults from catch-all **500→cause-naming 4xx** (issue #4 no-chat-template + cluster #6: audio-upload faults, 4× rebind catch-alls, queue), keep per-request `enable_thinking` + add `reasoning_effort` alias on demand, and a **don't-build list** for the OpenAI platform tail (responses/batches/images/moderations/files/fine_tuning/assistants/legacy completions). Keeps `n>1`/`logit_bias`→400. Removal gate cleared (the platform N/A; the consuming application self-migrates).
- [`014-speaker-identity-client-side.md`](docs/decisions/014-speaker-identity-client-side.md) — **Accepted** (decision; no daemon change): cross-file **speaker identity stays client-side** — the daemon is not extended (no enrollment store, no `/api/speakers`, no `/v1/audio/identifications`). It fails the ADR-011 governor test (cosine matching over 256-d WeSpeaker vectors is CPU bookkeeping — no Metal budget, no tenant to multiplex), has no cross-vendor standard to honor (ADR 013), and at ≤10 voices a local-JSON voiceprint store + brute-force cosine beats the built-in vector DB (single-dimension: 2560-d text vs 256-d speaker — can't coexist). Delivered as a self-contained **handoff prompt** (`docs/speaker-identification-agent-prompt.md` + `docs/speaker-identification-plan.md`). Tripwire: re-open if the roster grows past ~hundreds.

## Build / run / test

```sh
./deploy/build.sh Release           # xcodebuild → .build/xcode/.../Release/athena
./deploy/test.sh                    # unit tests
./deploy/e2e-rbac.sh                # RBAC end-to-end
athena load                         # run daemon in foreground on loopback (no auth)
curl http://127.0.0.1:7447/healthz  # liveness
curl http://127.0.0.1:7447/openapi.json  # self-describing surface
```

For production install (boot-time launchd, TLS, bearer auth, WebUI), see `docs/quickstart.md`.

## graphify

This project has a knowledge graph at `graphify-out/` with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
