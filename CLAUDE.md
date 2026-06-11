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
| `/api/*` | Athena native (minimal) | Native clients, model-store + RBAC admin |

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
