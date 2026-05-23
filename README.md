# Athena

The inference appliance of Project the platform: **one** native macOS/MLX
daemon that hosts LLM chat, text embeddings, audio
transcription/diarization/speaker-embeddings, a built-in vector database,
and an async job queue — all sharing a single Metal memory governor so
the box never oversubscribes its unified memory.

Athena is a **passive oracle**. The daemon answers inbound requests only;
it never initiates outbound connections except to fetch model weights
from Hugging Face and an opt-in remote-syslog sink. There are no result
or billing webhooks — anything a client needs is delivered by pull,
long-poll, or Server-Sent Events.

## Two HTTP dialects

| Surface | Shape | Use it for |
| --- | --- | --- |
| `/v1/*` | OpenAI-compatible | Drop-in for OpenAI SDKs and existing tooling |
| `/api/*` | Athena's own minimal dialect | Native clients, model-store + RBAC admin |

Every endpoint is described in a machine-readable **OpenAPI 3.0.3**
document the daemon serves at **`GET /openapi.json`** (always open, no
auth — the appliance describes itself). Load it into Swagger UI, Postman,
or an SDK generator.

All errors share one envelope:

```json
{ "error": { "message": "...", "type": "...", "code": "..." } }
```

## Capabilities

- **Chat** — `POST /v1/chat/completions` (streaming SSE, tool calls,
  JSON-schema structured output, `stop`/`seed`/`top_p`).
- **Embeddings** — `POST /v1/embeddings`.
- **Audio** — transcription (`/v1/audio/transcriptions`, with word
  timestamps + SRT/VTT), diarization, and speaker embeddings.
- **Vector DB** — `POST /v1/vectors` + `/v1/vectors/query`, governed
  cosine search.
- **Async queue** — submit a job, then poll or stream its status
  (`/v1/queue`).
- **RBAC** — token → user → roles, managed over `/api/*` or the WebUI.
- **Usage + audit** — per-principal token metering and an append-only
  admin audit trail (both pull-only).

## Requirements

- macOS on Apple Silicon (the daemon is Apple/MLX-only).
- A **full Xcode** install to build — MLX Metal shaders cannot be built
  by the Command-Line Tools alone.

The `athena` **client** CLI is cross-platform (Linux/Windows build the
same command from a standalone package); only the daemon is Apple-bound.

## Build

```sh
./deploy/build.sh Release        # xcodebuild → .build/xcode/.../Release/athena
```

## Get started

```sh
# Dev: run in the foreground on loopback (auth off until you seed users).
athena load

# In another shell — the appliance describes itself, no token needed:
curl http://127.0.0.1:7447/openapi.json
curl http://127.0.0.1:7447/healthz
```

For a production install (boot-time launchd daemon, auto-seeded admin,
TLS), bearer-token auth, the WebUI, and your first chat request, see the
**[Quickstart](docs/quickstart.md)**.

## Configuration

All daemon settings live in a flat TOML file (`deploy/athena.toml` is the
documented template) and are editable in place with `athena config set
<key> <value>`. Every key has a sane built-in default; any environment
override takes precedence over TOML, which takes precedence over the
default.

## Documentation

- **[Quickstart](docs/quickstart.md)** — install → seed admin → first
  curl → WebUI → TLS.
- **[Reverse proxy](docs/reverse-proxy.md)** — front Athena with
  nginx/Caddy for managed TLS.
- **`GET /openapi.json`** — the live, machine-readable API reference.
- **[deploy/athena.toml](deploy/athena.toml)** — every config key, with
  defaults and notes.
- CLI help — `athena --help`, `athena <command> --help`.
