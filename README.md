# Athena

A self-hosted inference appliance: **one** native macOS/MLX
daemon that hosts LLM chat, text embeddings, and audio/video
transcription/diarization/speaker-embeddings — all sharing a single Metal
memory governor so the box never oversubscribes its unified memory.

Athena is a **passive oracle**. The daemon answers inbound requests only;
it never initiates outbound connections except to fetch model weights
from Hugging Face. There are no result or billing webhooks — anything
a client needs is delivered by pull, long-poll, or Server-Sent Events.
Diagnostic logs land in the macOS unified log (subsystem `athena`);
off-box shipping is an operator concern — see [docs/logging.md](docs/logging.md)
and [docs/logging-shipping.md](docs/logging-shipping.md).

## How Athena is built

Athena is developed primarily by autonomous AI agents (Anthropic's
Claude) working under human direction. A human operator sets the goals,
reviews every change, and gates all architecturally significant
decisions — but most of the design exploration, implementation, and
documentation is authored by agents across many sessions.

Two things in this repository reflect that:

- **The decision trail.** Every significant architectural choice is
  recorded as an ADR under [docs/decisions/](docs/decisions/), and
  changes land in small, test-pinned slices. The ADRs are how a human
  governs an agent-authored codebase — load-bearing, not ceremonial.
- **The commit timeline.** Commits land whenever agent sessions run,
  not on one person's schedule — expect a distribution spanning
  business hours and a long overnight tail. `Co-Authored-By` trailers
  mark agent-authored commits.

This disclosure keeps the history honest: the timestamps and the volume
are a property of how the software is made, not an anomaly.

## Three API surfaces, one engine

Athena serves multiple API dialects as thin protocol adapters over a
single inference engine — pick whichever your tooling already speaks:

| Surface | Shape | Use it for |
| --- | --- | --- |
| `/v1/*` | OpenAI-compatible | Drop-in for OpenAI SDKs and existing tooling |
| `POST /v1/messages` | Anthropic-compatible | Drop-in for Anthropic SDKs and Claude Code ([docs/claude-code.md](docs/claude-code.md)) |
| `/api/*` | Athena native | Daemon **control plane**: model store + lifecycle, RBAC, config, audit/usage/logs — not inference |

A few `/v1/*` routes are Athena-native extensions with no OpenAI
equivalent (video transcription, diarization, speaker embeddings) —
**[docs/api-surface.md](docs/api-surface.md)** lists exactly which
endpoints are drop-in compatible and which are extensions.

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
  JSON-schema structured output, vision/image input,
  `stop`/`seed`/`top_p`) and the Anthropic-compatible
  `POST /v1/messages` (streaming, tools, system prompts — verified
  end-to-end with real Claude Code).
- **Embeddings** — `POST /v1/embeddings`.
- **Audio** — transcription (`/v1/audio/transcriptions`, with word
  timestamps + SRT/VTT), diarization, and speaker embeddings.
- **Video** — `POST /v1/video/transcriptions` demuxes a video's audio
  track and transcribes it.
- **Model lifecycle** — `POST /api/models/{pull,convert,prune}` run
  synchronously and stream Server-Sent Events progress (ADR 025).
- **RBAC** — token → user → roles, managed over `/api/*` or the WebUI.
- **Usage + audit** — per-principal token metering and an append-only
  admin audit trail (both pull-only).

## Requirements

- macOS on Apple Silicon (the daemon is Apple/MLX-only).
- A **full Xcode** install to build, **Xcode 26.5 or newer (Swift 6.3)** —
  MLX Metal shaders cannot be built by the Command-Line Tools alone, and
  `mlx-swift` 0.31.5+ ships a Swift 6.3 manifest, so an older Xcode fails
  at dependency *resolution* (an error naming mlx-swift's tools version),
  before any build starts. (26.5 is the verified floor; 26.3's Swift
  6.2.4 fails, 26.4 is untested.)

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
- **[API surface](docs/api-surface.md)** — every endpoint, tagged
  OpenAI-compatible / Anthropic-compatible / Athena-native.
- **[Reverse proxy](docs/reverse-proxy.md)** — front Athena with
  nginx/Caddy for managed TLS.
- **`GET /openapi.json`** — the live, machine-readable API reference.
- **[deploy/athena.toml](deploy/athena.toml)** — every config key, with
  defaults and notes.
- CLI help — `athena --help`, `athena <command> --help`.
