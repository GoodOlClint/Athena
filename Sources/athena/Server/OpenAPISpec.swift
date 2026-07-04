import Foundation

/// Hand-authored OpenAPI 3.0.3 description of Athena's HTTP surface
/// (M32.1). This is the single source of truth for the document served
/// at `GET /openapi.json`. It is embedded as a string literal — the same
/// zero-dependency approach the WebUI takes with its HTML — so there is
/// no resource bundle to locate at runtime and no code-generation step.
///
/// Scope: the integrator-facing surface only — the OpenAI-compatible
/// `/v1/*` routes, the Athena-native `/api/*` dialect, and the three
/// operational endpoints (`/healthz`, `/metrics`, `/openapi.json`). The
/// browser console under `/ui/*` is an internal control surface, not a
/// programmatic contract, and is deliberately excluded.
///
/// `info.version` is interpolated from the daemon version so the spec can
/// never report a version other than the build that serves it. A
/// drift-guard (deploy/e2e-rbac.sh) asserts every `/v1`+`/api` route
/// registered in AthenaServer is present here.
enum OpenAPISpec {
    static func json(version: String) -> String {
        // Raw literal: JSON backslash escapes pass through verbatim; only
        // `\#(version)` interpolates. No `"""#` sequence appears in the body.
        #"""
        {
          "openapi": "3.0.3",
          "info": {
            "title": "Athena",
            "summary": "Project the platform inference appliance (passive oracle).",
            "description": "Athena is a single native macOS/MLX daemon that hosts LLM chat, embeddings, and audio/video transcription/diarization behind one Metal memory governor. It is a **passive oracle**: the daemon answers inbound requests only and never initiates outbound connections (except fetching model weights and an opt-in remote-syslog sink). Anything a client needs is delivered by pull / long-poll / SSE — there are no result or billing webhooks.\n\nTwo HTTP dialects are served:\n- `/v1/*` — OpenAI-compatible (drop-in for OpenAI SDKs).\n- `/api/*` — Athena's own minimal native dialect.\n\nAll errors share the envelope `{\"error\":{\"message\",\"type\",\"code\"}}`. Authentication is a bearer token (`Authorization: Bearer <token>`); each route requires a single RBAC permission. When auth is disabled (loopback, no seeded users) every route is open.",
            "version": "\#(version)",
            "license": { "name": "Proprietary" }
          },
          "servers": [
            { "url": "/", "description": "This Athena daemon (loopback default 127.0.0.1:7447, or behind a TLS reverse proxy)." }
          ],
          "security": [ { "bearerAuth": [] } ],
          "tags": [
            { "name": "Operational", "description": "Health, metrics, and this spec. Unauthenticated discovery." },
            { "name": "Chat", "description": "OpenAI-compatible chat completions." },
            { "name": "Models", "description": "OpenAI-compatible model discovery." },
            { "name": "Embeddings", "description": "OpenAI-compatible text embeddings." },
            { "name": "Audio", "description": "Transcription, diarization, and speaker embeddings (multipart)." },
            { "name": "Native", "description": "Athena-native /api chat + embed." },
            { "name": "Admin", "description": "Daemon administration." },
            { "name": "Model store", "description": "Native model-store read + lifecycle ops." },
            { "name": "RBAC", "description": "Users, roles, and API tokens." },
            { "name": "Usage", "description": "Per-principal token metering (pull-only)." },
            { "name": "Audit", "description": "Append-only RBAC/admin audit trail." }
          ],
          "paths": {
            "/healthz": {
              "get": {
                "tags": ["Operational"],
                "summary": "Liveness + memory-governor snapshot + live signals.",
                "description": "Always open (no auth). Returns the governor's current reservations and budget, plus `inflight` (live request count) and `lastRequestAt` (unix-epoch seconds; 0 ⇒ none since boot) for at-a-glance diagnosis of a hung daemon (M43.1). M60.1 adds `thermalState` (`nominal`/`fair`/`serious`/`critical` from macOS thermal pressure — the throttle indicator to back off on), `lastDecodeTokensPerSec` (most recent decode throughput), and `mlxActiveBytes`/`mlxCacheBytes` (MLX allocator counters). M60.2 adds `powerAssertionHeld` (whether the daemon holds a PreventUserIdleSystemSleep assertion; `false` means an unattended Mac can sleep and suspend inference). M60.3 adds `gpuClockMHz` and `gpuActiveResidency` (sudoless in-process GPU telemetry via IOReport; `null` when unavailable). ADR 023 adds `mlxCacheLimitBytes` (G1 — the serve-path MLX buffer-cache bound; `mlxCacheBytes` should plateau at/under it) and `admissionMode` (G2 — `footprint` means admission/`freeBytes` are metered against the real Metal footprint `budget − max(committed, reserved)`; `estimate` is the reservation-only revert switch).",
                "security": [],
                "responses": {
                  "200": { "description": "Governor snapshot + live signals.", "content": { "application/json": { "schema": { "type": "object" } } } }
                }
              }
            },
            "/metrics": {
              "get": {
                "tags": ["Operational"],
                "summary": "Metrics (Prometheus text by default; JSON via Accept).",
                "description": "Global, process-lifetime counters (reset on restart). Content-negotiated: returns Prometheus text-exposition format 0.0.4 by default (the scrape target), or the JSON snapshot when the request sends `Accept: application/json`. Requires the `metrics.read` permission.",
                "responses": {
                  "200": { "description": "Metrics.", "content": { "text/plain": { "schema": { "type": "string" } }, "application/json": { "schema": { "type": "object" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/openapi.json": {
              "get": {
                "tags": ["Operational"],
                "summary": "This OpenAPI document.",
                "description": "Always open (no auth) so the appliance can describe itself.",
                "security": [],
                "responses": {
                  "200": { "description": "The OpenAPI 3.0.3 document.", "content": { "application/json": { "schema": { "type": "object" } } } }
                }
              }
            },
            "/v1/chat/completions": {
              "post": {
                "tags": ["Chat"],
                "summary": "Create a chat completion.",
                "description": "OpenAI-compatible. Set `stream:true` for an SSE stream of `chat.completion.chunk` events terminated by `data: [DONE]`; with `stream_options.include_usage:true` a final usage-only chunk precedes it. `response_format`/`tools` route into structured output. `top_p`/`seed` are honored only on the sampling path and are inert under greedy/MTP/structured decoding. `logprobs`/`top_logprobs` (0–20) are honored on the deterministic decode path (temperature 0 or structured) and return `choices[].logprobs.content`; a sampling request (temperature>0, no schema) with `logprobs` ⇒ 400. `n>1` and non-empty `logit_bias` are rejected 400. Requires `inference`.",
                "requestBody": {
                  "required": true,
                  "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ChatCompletionRequest" } } }
                },
                "responses": {
                  "200": {
                    "description": "A chat completion, or (when streaming) an SSE event stream.",
                    "content": {
                      "application/json": { "schema": { "$ref": "#/components/schemas/ChatCompletionResponse" } },
                      "text/event-stream": { "schema": { "type": "string", "description": "SSE: `data: {chat.completion.chunk}` lines, then `data: [DONE]`." } }
                    }
                  },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "413": { "$ref": "#/components/responses/PayloadTooLarge" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/messages": {
              "post": {
                "tags": ["Chat"],
                "summary": "Anthropic Messages API (chat over the same engine).",
                "description": "**Athena-native dialect, Anthropic-compatible (NOT OpenAI).** The Anthropic Messages shape over the SAME inference engine as `/v1/chat/completions` (ADR 036): one engine, multiple protocol adapters. Lets an Anthropic-API harness (e.g. Claude Code, `ANTHROPIC_BASE_URL`) point at Athena directly. Supports text + `system` + `tool_use`/`tool_result` + `tools` (mapped to the same model menu) + `tool_choice` (`auto`/`any`/`tool`) + `stop_sequences`, both non-streaming and streaming (`stream:true` ⇒ the Anthropic SSE event sequence: message_start → content_block_start/delta/stop → message_delta → message_stop). `image`/`document` content blocks are refused with a cause-naming 400 (`unsupported_content_block`). Unknown top-level fields — `cache_control` (prompt-caching) and extended `thinking` — are **silently ignored** (OpenAI-adapter convention), not rejected, so a Claude Code client that sends them still works. Auth: `Authorization: Bearer` (Claude Code authenticates via bearer; there is no separate `x-api-key` handling). Requires `inference`.",
                "requestBody": {
                  "required": true,
                  "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["model", "max_tokens", "messages"],
                    "properties": {
                      "model": { "type": "string", "description": "A resident/store LLM id (same resolution as /v1/chat/completions)." },
                      "max_tokens": { "type": "integer", "description": "Required by the Anthropic shape; the completion token cap." },
                      "system": { "description": "System prompt — a string or an array of text blocks." },
                      "messages": { "type": "array", "items": { "type": "object" }, "description": "Anthropic messages (role + string|block content; tool_use/tool_result blocks)." },
                      "tools": { "type": "array", "items": { "type": "object" }, "description": "Anthropic tools ({name, description, input_schema})." },
                      "tool_choice": { "type": "object", "description": "{type: auto|any|tool|none, name?}." },
                      "stop_sequences": { "type": "array", "items": { "type": "string" } },
                      "temperature": { "type": "number" },
                      "top_p": { "type": "number" },
                      "stream": { "type": "boolean", "description": "When true, emit the Anthropic SSE event stream (text/event-stream)." },
                      "timeout": { "type": "number", "description": "Athena extension (tolerated field, ADR 036 section 7 rung 2 - not part of the Anthropic wire shape): per-request deadline override in seconds, same semantics as the /v1/chat/completions `timeout`. Omitted => the daemon default." }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Anthropic message response ({id, type:message, role:assistant, content[], stop_reason, usage})." },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/models": {
              "get": {
                "tags": ["Models"],
                "summary": "List available models.",
                "description": "OpenAI-compatible list shape. Requires `model.read`.",
                "responses": {
                  "200": { "description": "Model list.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIModelList" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/models/{id}": {
              "get": {
                "tags": ["Models"],
                "summary": "Retrieve a model.",
                "description": "OpenAI-compatible retrieve shape. Requires `model.read`.",
                "parameters": [ { "name": "id", "in": "path", "required": true, "schema": { "type": "string" }, "description": "Model id (store entry name)." } ],
                "responses": {
                  "200": { "description": "The model.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIModel" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "404": { "$ref": "#/components/responses/NotFound" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/embeddings": {
              "post": {
                "tags": ["Embeddings"],
                "summary": "Create embeddings.",
                "description": "OpenAI-compatible. `input` is a string or array of strings. Requires `inference`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/EmbeddingRequest" } } } },
                "responses": {
                  "200": { "description": "Embeddings.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/EmbeddingResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "413": { "$ref": "#/components/responses/PayloadTooLarge" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/audio/transcriptions": {
              "post": {
                "tags": ["Audio"],
                "summary": "Transcribe audio.",
                "description": "OpenAI-compatible multipart upload. `response_format` of `json` (default), `text`, `srt`, `vtt`, or `verbose_json`; `timestamp_granularities[]=word` adds word timings; `diarize=true` adds per-segment speaker ids in verbose_json. `diarized_json` is an **Athena-native** alias (ADR 013 #3) that *implies* diarization (no `diarize` flag needed) and returns the verbose envelope with every segment speaker-labeled — speaker is Athena's integer id, not OpenAI's string label. The engine is chosen by the resident model's class (ADR 020): Whisper (default) or Parakeet-TDT (multilingual; word/segment timestamps from TDT durations). Max upload size = `max_audio_upload_bytes` (default 100 MiB) over the raw multipart body; over it ⇒ 413 payload_too_large. Requires `inference`.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary", "description": "Audio file." },
                      "model": { "type": "string", "description": "Selects among the store's transcription models (ADR 026) — a Whisper or Parakeet-TDT id (ADR 020); the resident model's class picks the engine. Omit ⇒ default (Whisper); unknown id ⇒ 400 model_not_available; an unsupported ASR arch ⇒ 400 unsupported_transcription_arch." },
                      "language": { "type": "string" },
                      "response_format": { "type": "string", "enum": ["json", "text", "srt", "vtt", "verbose_json", "diarized_json"] },
                      "diarize": { "type": "boolean" }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Transcription.", "content": {
                    "application/json": { "schema": { "$ref": "#/components/schemas/VerboseTranscriptionResponse" } },
                    "text/plain": { "schema": { "type": "string" } }
                  } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "413": { "$ref": "#/components/responses/PayloadTooLarge" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/video/transcriptions": {
              "post": {
                "tags": ["Video"],
                "summary": "Transcribe a video's audio track.",
                "description": "**Athena-native, NOT OpenAI-compatible** (OpenAI has no video API). Multipart upload of a video container (mp4/mov/…); Athena demuxes the audio track and transcribes it via the same Whisper/Parakeet tenant as `/v1/audio/transcriptions`, returning the identical response shapes (`json` default, `text`, `srt`, `vtt`, `verbose_json`; `timestamp_granularities[]=word`). `model` selects the resident transcription model. A video with no audio track ⇒ 400 video_no_audio_track; sub-0.1 s or undecodable audio ⇒ 400 (shared decode floor). diarization on video is not yet wired ⇒ 501 not_implemented for both `diarize=true` and `response_format=diarized_json` (transcribe, then POST the extracted audio to /v1/audio/diarizations). Max upload size = `max_video_upload_bytes` (default 1 GiB); over it ⇒ 413 payload_too_large. Requires `inference`.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary", "description": "Video file (any container AVFoundation reads)." },
                      "model": { "type": "string", "description": "Selects among the store's transcription models — a Whisper or Parakeet-TDT id (ADR 020). Omit ⇒ default." },
                      "language": { "type": "string" },
                      "response_format": { "type": "string", "enum": ["json", "text", "srt", "vtt", "verbose_json"] }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Transcription.", "content": {
                    "application/json": { "schema": { "$ref": "#/components/schemas/VerboseTranscriptionResponse" } },
                    "text/plain": { "schema": { "type": "string" } }
                  } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "413": { "$ref": "#/components/responses/PayloadTooLarge" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/audio/diarizations": {
              "post": {
                "tags": ["Audio"],
                "summary": "Diarize audio (who spoke when).",
                "description": "**Athena-native, NOT OpenAI-compatible** (OpenAI has no diarization API). Multipart upload. Returns speaker-labelled time spans. `method` selects the engine (ADR 018): `sortformer` (default, end-to-end, ≤4 speakers), `cluster` (embedding+clustering, >4, no overlap), `pyannote` (learned segmentation + embedding + global clustering — arbitrary speakers, overlap-aware, file-stable ids; emits overlapping segments). Speaker ids are global across the whole file. Max upload size = `max_audio_upload_bytes` (default 100 MiB); over it ⇒ 413 payload_too_large. Requires `inference`.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary" },
                      "method": { "type": "string", "enum": ["sortformer", "cluster", "pyannote"], "description": "Diarization engine. Omit ⇒ sortformer. A method that mismatches the resident model's backend ⇒ 400 invalid_method." },
                      "model": { "type": "string", "description": "Selects among the store's diarization models (ADR 026). For method=pyannote this must be a pyannote-segmentation model. Omit ⇒ default; unknown id ⇒ 400 model_not_available." },
                      "num_speakers": { "type": "integer", "description": "Exact speaker count (cluster/pyannote)." },
                      "min_speakers": { "type": "integer", "description": "Floor on auto speaker count (cluster/pyannote)." },
                      "max_speakers": { "type": "integer", "description": "Cap on auto speaker count (cluster/pyannote)." },
                      "threshold": { "type": "number", "description": "Cosine-distance merge threshold for auto speaker count (cluster/pyannote; default 0.75)." },
                      "min_cluster_seconds": { "type": "number", "description": "pyannote auto mode only: minimum total airtime for a cluster to count as a speaker; smaller clusters are reassigned to the nearest speaker (default 6). Ignored when num_speakers is set." }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Diarization.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DiarizationResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "413": { "$ref": "#/components/responses/PayloadTooLarge" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/audio/embeddings": {
              "post": {
                "tags": ["Audio"],
                "summary": "Speaker (voice) embeddings.",
                "description": "**Athena-native, NOT OpenAI-compatible** (OpenAI has no speaker-embedding API). Multipart upload. Returns an embedding per requested segment. Max upload size = `max_audio_upload_bytes` (default 100 MiB); over it ⇒ 413 payload_too_large. Requires `inference`.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary" },
                      "model": { "type": "string", "description": "Selects among the store's speaker-embedding models (ADR 026). Omit ⇒ default; unknown id ⇒ 400 model_not_available. The response `model` is the model actually served." },
                      "segments": { "type": "string", "description": "JSON array of {start,end} segment specs (seconds)." }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Speaker embeddings.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SpeakerEmbeddingResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "413": { "$ref": "#/components/responses/PayloadTooLarge" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/api/admin/stop": {
              "post": {
                "tags": ["Admin"],
                "summary": "Unload the model (daemon keeps running).",
                "description": "Requires `daemon.admin`.",
                "responses": {
                  "200": { "description": "Unloaded.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AthenaStopResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/admin/status": {
              "get": {
                "tags": ["Admin"],
                "summary": "Daemon + RBAC posture.",
                "description": "Requires `daemon.admin`.",
                "responses": {
                  "200": { "description": "Status.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AdminStatusResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/usage": {
              "get": {
                "tags": ["Usage"],
                "summary": "Per-principal token usage (pull-only).",
                "description": "A member sees its own counters; an admin sees all principals. Requires `inference`.",
                "responses": {
                  "200": { "description": "Usage report.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UsageReportResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/audit": {
              "get": {
                "tags": ["Audit"],
                "summary": "Append-only RBAC/admin audit trail.",
                "description": "Admin-only oversight view, most-recent-first. Requires `daemon.admin`.",
                "parameters": [
                  { "name": "principal", "in": "query", "required": false, "schema": { "type": "string" }, "description": "Filter by acting principal." },
                  { "name": "action", "in": "query", "required": false, "schema": { "type": "string" }, "description": "Filter by action." },
                  { "name": "result", "in": "query", "required": false, "schema": { "type": "string", "enum": ["ok", "denied"] } },
                  { "name": "limit", "in": "query", "required": false, "schema": { "type": "integer" } }
                ],
                "responses": {
                  "200": { "description": "Audit entries.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AuditReportResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/logs": {
              "get": {
                "tags": ["Logs"],
                "summary": "Daemon unified-log entries (one-shot).",
                "description": "Admin-only projection of `subsystem == \"athena\"` from the macOS unified log. Returns the NEWEST `limit` entries of the window, in `log show`'s native oldest-first order (the daemon drains the whole window and keeps the tail — see `truncated`). Mirrors `/usr/bin/log show --style ndjson` filtered to the daemon's subsystem. Requires `daemon.admin`.",
                "parameters": [
                  { "name": "since", "in": "query", "required": false, "schema": { "type": "string", "default": "1h" }, "description": "How far back to look, e.g. `5m`, `1h`, `1d` (passed to `log show --last`)." },
                  { "name": "category", "in": "query", "required": false, "schema": { "type": "array", "items": { "type": "string" } }, "style": "form", "explode": true, "description": "Category filter, repeatable. e.g. `daemon`, `audit`, `model.llm`." },
                  { "name": "debug", "in": "query", "required": false, "schema": { "type": "boolean" }, "description": "Include memory-only info/debug entries (requires `log config --mode \"level:debug\"`)." },
                  { "name": "limit", "in": "query", "required": false, "schema": { "type": "integer", "default": 200, "maximum": 5000 } }
                ],
                "responses": {
                  "200": { "description": "Log entries.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/LogsReportResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/logs/stream": {
              "get": {
                "tags": ["Logs"],
                "summary": "Daemon unified-log entries (SSE follow).",
                "description": "Admin-only SSE wrapper over `log stream --style ndjson` filtered to `subsystem == \"athena\"`. Each event is `data: {<LogEntry JSON>}\\n\\n`. Capped at ~10 min per connection. Requires `daemon.admin`.",
                "parameters": [
                  { "name": "category", "in": "query", "required": false, "schema": { "type": "array", "items": { "type": "string" } }, "style": "form", "explode": true },
                  { "name": "debug", "in": "query", "required": false, "schema": { "type": "boolean" } }
                ],
                "responses": {
                  "200": { "description": "SSE stream of LogEntry events.", "content": { "text/event-stream": { "schema": { "type": "string" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/cache/prompt": {
              "get": {
                "tags": ["Prompt cache"],
                "summary": "Cross-request prompt-prefix KV pool stats.",
                "description": "Admin-only. Reports whether the pool is enabled, its live entry/byte occupancy and caps, and cumulative hit/miss/eviction counters. Requires `daemon.admin`.",
                "responses": {
                  "200": { "description": "Pool stats.", "content": { "application/json": { "schema": { "type": "object", "properties": { "enabled": { "type": "boolean" }, "entries": { "type": "integer" }, "bytes": { "type": "integer" }, "hits": { "type": "integer" }, "misses": { "type": "integer" }, "evictions": { "type": "integer" }, "max_entries": { "type": "integer" }, "max_bytes": { "type": "integer" } } } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "delete": {
                "tags": ["Prompt cache"],
                "summary": "Flush the prompt-prefix KV pool.",
                "description": "Admin-only. Drops every cached prefix not currently held by an in-flight generation (those are freed when their request completes). Audited. Requires `daemon.admin`.",
                "responses": {
                  "200": { "description": "Flush result.", "content": { "application/json": { "schema": { "type": "object", "properties": { "flushed": { "type": "integer" }, "entries": { "type": "integer" }, "bytes": { "type": "integer" } } } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/config": {
              "get": {
                "tags": ["Daemon config"],
                "summary": "Current daemon configuration (ADR 037).",
                "description": "Admin-only projection of the TOML the daemon reads. `values` carries every known scalar (empty when unset); `readonly_keys` are the deny-listed keys `PUT` refuses (auth/TLS/encryption/data-dir/debugger — edit the file + sudo-restart). Requires `daemon.admin`.",
                "responses": {
                  "200": { "description": "Config projection.", "content": { "application/json": { "schema": { "type": "object", "properties": { "path": { "type": "string" }, "keys": { "type": "array", "items": { "type": "string" } }, "values": { "type": "object", "additionalProperties": { "type": "string" } }, "readonly_keys": { "type": "array", "items": { "type": "string" } } } } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "put": {
                "tags": ["Daemon config"],
                "summary": "Set one config scalar (ADR 037; sudoless).",
                "description": "Admin-only. Sets one `{key,value}` scalar in place via the hardened config editor, then takes effect after a restart (see POST /api/admin/restart). A deny-listed key (`auth_keys_file`/`tls_cert`/`tls_key`/`encrypt_store`/`data_dir`/`deny_debugger_attach`) returns 400 `config_key_readonly` — config takeover would be daemon takeover. Audited. Requires `daemon.admin`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object", "required": ["key", "value"], "properties": { "key": { "type": "string" }, "value": { "type": "string" } } } } } },
                "responses": {
                  "200": { "description": "Saved.", "content": { "application/json": { "schema": { "type": "object", "properties": { "ok": { "type": "boolean" }, "key": { "type": "string" }, "value": { "type": "string" }, "note": { "type": "string" } } } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/admin/restart": {
              "post": {
                "tags": ["Daemon config"],
                "summary": "Restart the daemon (ADR 037; sudoless).",
                "description": "Admin-only. Drains in-flight inference (ADR 029 execution gate) then `exit(0)`s; launchd `KeepAlive` relaunches (~10s throttle). Responds 200 before exiting. Lets `athena restart` run without sudo. Audited. Requires `daemon.admin`.",
                "responses": {
                  "200": { "description": "Restart acknowledged.", "content": { "application/json": { "schema": { "type": "object", "properties": { "restarting": { "type": "boolean" }, "note": { "type": "string" } } } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models": {
              "get": {
                "tags": ["Model store"],
                "summary": "List installed models.",
                "description": "Requires `model.read`.",
                "responses": {
                  "200": { "description": "Models.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelListResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/default": {
              "get": {
                "tags": ["Model store"],
                "summary": "Get the default model.",
                "description": "Requires `model.read`.",
                "responses": {
                  "200": { "description": "Default model.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DefaultModelResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "put": {
                "tags": ["Model store"],
                "summary": "Set the default model.",
                "description": "Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SetDefaultModelRequest" } } } },
                "responses": {
                  "200": { "description": "Default set.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DefaultModelResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/copy": {
              "post": {
                "tags": ["Model store"],
                "summary": "Copy/alias a model.",
                "description": "Symlink alias by default; `copy:true` deep-copies. Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelCopyRequest" } } } },
                "responses": {
                  "200": { "description": "Copy result.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelCopyResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/pull": {
              "post": {
                "tags": ["Model store"],
                "summary": "Pull a model from Hugging Face (synchronous, SSE progress).",
                "description": "Athena-native (ADR 025 S2): runs synchronously and streams Server-Sent Events directly — no async queue, no job id. Frames (all additive; unknown `event`s are ignored by older clients): aggregate `{\"event\":\"progress\",\"fraction\":F,\"bytes\":B,\"total\":T}`, per-shard `{\"event\":\"file\",\"name\",\"index\",\"count\",\"bytes\",\"total\",\"done\"}` (throttled ~500ms/1% per file), then a terminal `{\"event\":\"done\",\"result\":{…}}` or `{\"event\":\"error\",\"error\":{message,type,code}}`, ending with `[DONE]`. Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelPullRequest" } } } },
                "responses": {
                  "200": { "description": "SSE stream of progress + terminal event.", "content": { "text/event-stream": { "schema": { "type": "string" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/convert": {
              "post": {
                "tags": ["Model store"],
                "summary": "Convert/quantize a model (synchronous, SSE progress).",
                "description": "Athena-native (ADR 025 S2): synchronous, streams SSE progress + a terminal done/error event (see POST /api/models/pull). Beyond download `progress`/`file` frames it emits `{\"event\":\"phase\",\"phase\":\"download|load|quantize|write\"}` and, during the minutes-long quantize materialize, `{\"event\":\"quantize\",\"index\":i,\"count\":N}` (audit §3) so the op is never silent. Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelConvertRequest" } } } },
                "responses": {
                  "200": { "description": "SSE stream of progress + terminal event.", "content": { "text/event-stream": { "schema": { "type": "string" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/prune": {
              "post": {
                "tags": ["Model store"],
                "summary": "Prune unreferenced model blobs (synchronous, SSE progress).",
                "description": "Athena-native (ADR 025 S2): synchronous, streams a terminal done/error SSE event (see POST /api/models/pull); `dry_run:true` lists candidates only. Requires `model.write`.",
                "requestBody": { "required": false, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelPruneRequest" } } } },
                "responses": {
                  "200": { "description": "SSE stream of the terminal event.", "content": { "text/event-stream": { "schema": { "type": "string" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/resident": {
              "get": {
                "tags": ["Model store"],
                "summary": "Every module's resident model slot.",
                "description": "Reports, for each module class (llm, textEmbedding, transcription, diarization, speakerEmbedding), the store-derived selectable set, the configured default, and the id currently resident in the slot (nil ⇒ unloaded). M41.1. Requires `model.read`.",
                "responses": {
                  "200": { "description": "Resident slots.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelResidentResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/load": {
              "post": {
                "tags": ["Model store"],
                "summary": "Rebind a module's slot to a model id (M41.1).",
                "description": "Loads `body.module`'s slot (under the governor) and rebinds it to `body.id` (omit ⇒ the module's default). An id outside the module's store models is a 400 (`model_not_available`) — never a silent fallback or on-request download. Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelLoadRequest" } } } },
                "responses": {
                  "200": { "description": "Loaded.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelLoadResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/api/models/unload": {
              "post": {
                "tags": ["Model store"],
                "summary": "Release a module's slot (M41.1).",
                "description": "Unloads `body.module`'s slot and returns its bytes to the governor; absent / `\"all\"` ⇒ every module. The daemon keeps running; the next inference lazily reloads the module's default. Requires `model.write`.",
                "requestBody": { "required": false, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelUnloadRequest" } } } },
                "responses": {
                  "200": { "description": "Unloaded.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelUnloadResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/{name}": {
              "get": {
                "tags": ["Model store"],
                "summary": "Model detail (with config.json).",
                "description": "Requires `model.read`.",
                "parameters": [ { "name": "name", "in": "path", "required": true, "schema": { "type": "string" } } ],
                "responses": {
                  "200": { "description": "Model detail.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelDetailResponse" } } } },
                  "404": { "$ref": "#/components/responses/NotFound" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "delete": {
                "tags": ["Model store"],
                "summary": "Remove a model.",
                "description": "Requires `model.write`.",
                "parameters": [ { "name": "name", "in": "path", "required": true, "schema": { "type": "string" } } ],
                "responses": {
                  "200": { "description": "Removal result.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelRemovedResponse" } } } },
                  "404": { "$ref": "#/components/responses/NotFound" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/users": {
              "get": {
                "tags": ["RBAC"],
                "summary": "List users.",
                "description": "Requires `users.read`.",
                "responses": {
                  "200": { "description": "Users.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UserListResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "post": {
                "tags": ["RBAC"],
                "summary": "Create a user.",
                "description": "Requires `users.admin` and the ability to grant the requested role.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateUserRequest" } } } },
                "responses": {
                  "200": { "description": "Created.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UserSummary" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/users/{name}": {
              "delete": {
                "tags": ["RBAC"],
                "summary": "Delete a user.",
                "description": "Requires `users.admin`. The last admin cannot be removed.",
                "parameters": [ { "name": "name", "in": "path", "required": true, "schema": { "type": "string" } } ],
                "responses": {
                  "200": { "description": "Removal result.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UserRemovedResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/users/{name}/roles/{role}": {
              "post": {
                "tags": ["RBAC"],
                "summary": "Grant a role to a user.",
                "description": "Requires `users.admin` and the ability to grant `role`.",
                "parameters": [
                  { "name": "name", "in": "path", "required": true, "schema": { "type": "string" } },
                  { "name": "role", "in": "path", "required": true, "schema": { "type": "string" } }
                ],
                "responses": {
                  "200": { "description": "Granted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Ok" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "delete": {
                "tags": ["RBAC"],
                "summary": "Revoke a role from a user.",
                "description": "Requires `users.admin`.",
                "parameters": [
                  { "name": "name", "in": "path", "required": true, "schema": { "type": "string" } },
                  { "name": "role", "in": "path", "required": true, "schema": { "type": "string" } }
                ],
                "responses": {
                  "200": { "description": "Revoked.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Ok" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/roles": {
              "get": {
                "tags": ["RBAC"],
                "summary": "List roles and their permissions.",
                "description": "Read-only RBAC catalog. Requires `users.read`.",
                "responses": {
                  "200": { "description": "Roles.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/RolesResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/tokens": {
              "get": {
                "tags": ["RBAC"],
                "summary": "List API tokens (metadata only).",
                "description": "Hash prefixes only — secrets are never returned. Requires `tokens.admin`.",
                "responses": {
                  "200": { "description": "Tokens.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/TokenListResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "post": {
                "tags": ["RBAC"],
                "summary": "Mint an API token.",
                "description": "The secret is returned ONCE and never persisted. Requires `tokens.admin`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateTokenRequest" } } } },
                "responses": {
                  "200": { "description": "Minted token.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateTokenResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/tokens/{prefix}": {
              "delete": {
                "tags": ["RBAC"],
                "summary": "Revoke API token(s) by hash prefix.",
                "description": "Requires `tokens.admin`.",
                "parameters": [ { "name": "prefix", "in": "path", "required": true, "schema": { "type": "string" } } ],
                "responses": {
                  "200": { "description": "Revocation count.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/TokensRemovedResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/tokens/{prefix}/rotate": {
              "post": {
                "tags": ["RBAC"],
                "summary": "Rotate (revoke + reissue) an API token by hash prefix.",
                "description": "Matches exactly one token; its owner/scope/label carry to a fresh secret returned ONCE, and the old hash is revoked. `ttl_secs` sets the new lifetime (absent means no expiry). Requires `tokens.admin`.",
                "parameters": [ { "name": "prefix", "in": "path", "required": true, "schema": { "type": "string" } } ],
                "requestBody": { "required": false, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/RotateTokenRequest" } } } },
                "responses": {
                  "200": { "description": "Reissued token (secret shown once).", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateTokenResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "404": { "description": "No token matched the prefix." },
                  "409": { "description": "Prefix matched more than one token." }
                }
              }
            }
          },
          "components": {
            "securitySchemes": {
              "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "API token: `Authorization: Bearer <token>`. Mint one with `POST /api/tokens` (or `athena auth token`). When auth is disabled (loopback, no seeded users) all routes are open."
              }
            },
            "responses": {
              "BadRequest": { "description": "Invalid request.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
              "Unauthorized": { "description": "Missing or invalid bearer token.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
              "Forbidden": { "description": "Insufficient permissions.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
              "NotFound": { "description": "Not found.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
              "Overloaded": { "description": "Backpressure — the memory governor or a rate/concurrency cap declined the request, or (code `module_loading`) a model is still being made resident. A request for an on-disk model now BLOCKS until ready (up to `cold_load_wait_secs`) and serves 200; a `module_loading` 503 here means the load exceeded that budget or an operator weight download is in progress. Retry after the indicated delay.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
              "PayloadTooLarge": { "description": "Request body exceeds the configured upload limit (code `payload_too_large`). Audio uploads (`/v1/audio/*`) are bounded by `max_audio_upload_bytes` (default 100 MiB); JSON bodies by `max_request_body_bytes` (default 4 MiB). Pre-check with `Content-Length` to avoid the rejected transfer.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } }
            },
            "schemas": {
              "Error": {
                "type": "object",
                "required": ["error"],
                "properties": {
                  "error": {
                    "type": "object",
                    "required": ["message", "type", "code"],
                    "properties": {
                      "message": { "type": "string" },
                      "type": { "type": "string" },
                      "code": { "type": "string" }
                    }
                  }
                }
              },
              "Usage": {
                "type": "object",
                "properties": {
                  "prompt_tokens": { "type": "integer" },
                  "completion_tokens": { "type": "integer" },
                  "total_tokens": { "type": "integer" }
                }
              },
              "ChatMessage": {
                "type": "object",
                "required": ["role"],
                "properties": {
                  "role": { "type": "string", "enum": ["system", "user", "assistant", "tool"] },
                  "content": { "description": "Either a plain string, or an OpenAI content-parts array for vision input (M71). Image parts accept ONLY inline data: URLs (base64) — http(s) image URLs are rejected with 400 (passive-oracle: the daemon performs no outbound image fetch). Text parts are concatenated.", "nullable": true, "oneOf": [ { "type": "string" }, { "type": "array", "items": { "$ref": "#/components/schemas/ChatContentPart" } } ] },
                  "reasoning_content": { "type": "string", "description": "Chain-of-thought extracted from a channel-delimited model's output (e.g. gemma-4 <|channel>thought…<channel|>), surfaced separately so content is clean. Omitted when the model emits no reasoning channel. ADR 035." },
                  "tool_calls": { "type": "array", "items": { "type": "object" } }
                }
              },
              "ChatContentPart": {
                "type": "object",
                "required": ["type"],
                "properties": {
                  "type": { "type": "string", "enum": ["text", "image_url"] },
                  "text": { "type": "string" },
                  "image_url": { "type": "object", "required": ["url"], "properties": { "url": { "type": "string", "description": "Inline data: URL (base64), e.g. data:image/png;base64,…. http(s) URLs are rejected with 400 (passive-oracle)." }, "detail": { "type": "string" } } }
                }
              },
              "ChatCompletionRequest": {
                "type": "object",
                "required": ["messages"],
                "properties": {
                  "model": { "type": "string", "description": "Selects among the LLM models the daemon was loaded with (--llm-model, repeatable; first = default). Omit ⇒ the resident default. An id outside the configured set is rejected with 400 model_not_available — never a silent fallback or on-request download. The response `model` reports the model actually served. M41.2." },
                  "messages": { "type": "array", "items": { "$ref": "#/components/schemas/ChatMessage" } },
                  "stream": { "type": "boolean" },
                  "stream_options": { "type": "object", "properties": { "include_usage": { "type": "boolean" } } },
                  "response_format": { "type": "object", "description": "OpenAI response_format (text | json_object | json_schema)." },
                  "tools": { "type": "array", "items": { "type": "object" } },
                  "tool_choice": { "description": "OpenAI tool_choice. 'auto' (default when tools are present) ⇒ the model decides text-vs-tool each turn — NOT forced. 'required' ⇒ forces some tool call (grammar-constrained). {type:function,function:{name}} ⇒ forces that one. 'none' ⇒ no tool call (menu withheld). Auto tool-call detection is substrate-arch-gated (Gemma 4 supported). ADR 034." },
                  "max_tokens": { "type": "integer", "description": "Output-token cap (deprecated alias of max_completion_tokens; max_completion_tokens wins if both are sent)." },
                  "max_completion_tokens": { "type": "integer", "description": "Output-token cap (OpenAI's current field). Absent ⇒ the daemon default." },
                  "temperature": { "type": "number" },
                  "top_p": { "type": "number", "description": "Honored only on the sampling path; inert under greedy/MTP/structured decoding." },
                  "seed": { "type": "integer", "description": "Honored only on the sampling path." },
                  "stop": { "description": "String or array of strings; truncates output at the first match.", "oneOf": [ { "type": "string" }, { "type": "array", "items": { "type": "string" } } ] },
                  "n": { "type": "integer", "description": "Rejected with 400 when > 1." },
                  "logprobs": { "description": "Honored on the deterministic decode path (temperature 0 or structured); returns choices[].logprobs.content. A sampling request (temperature>0, no schema) with logprobs ⇒ 400." },
                  "top_logprobs": { "type": "integer", "description": "0–20 top alternatives per token; requires logprobs:true. Out of range ⇒ 400." },
                  "logit_bias": { "type": "object", "description": "Rejected with 400 when non-empty." },
                  "speculative": { "type": "boolean", "description": "Athena extension. Per-request speculative-decoding override for an MTP-capable model: true opts into the bit-identical-greedy MTP loop at temperature=0 or the Leviathan/Chen sampling loop (distributionally identical to non-speculative sampling at the same temp/top_p/seed) at temperature>0. false forces the standard path. omit ⇒ daemon's --speculative default. MTP engages on a Qwen3.5 fused-head checkpoint, or (ADR 032) on a Gemma 4 target with a paired gemma4_assistant drafter loaded (pull it with `athena pull <target> --with-drafter`); inert (single-token) on any model without a drafter." }
                }
              },
              "ChatChoice": {
                "type": "object",
                "properties": {
                  "index": { "type": "integer" },
                  "message": { "$ref": "#/components/schemas/ChatMessage" },
                  "finish_reason": { "type": "string", "enum": ["stop", "length", "tool_calls"] }
                }
              },
              "ChatCompletionResponse": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "object": { "type": "string" },
                  "created": { "type": "integer" },
                  "model": { "type": "string" },
                  "choices": { "type": "array", "items": { "$ref": "#/components/schemas/ChatChoice" } },
                  "usage": { "$ref": "#/components/schemas/Usage" }
                }
              },
              "OpenAIModel": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "object": { "type": "string" },
                  "created": { "type": "integer" },
                  "owned_by": { "type": "string" }
                }
              },
              "OpenAIModelList": {
                "type": "object",
                "properties": {
                  "object": { "type": "string" },
                  "data": { "type": "array", "items": { "$ref": "#/components/schemas/OpenAIModel" } }
                }
              },
              "EmbeddingRequest": {
                "type": "object",
                "required": ["input"],
                "properties": {
                  "model": { "type": "string", "description": "Selects among the embedding models the daemon was loaded with (--embedding-model, repeatable; first = default). Omit ⇒ the default. An id outside the configured set is rejected with 400 model_not_available — never a silent fallback or on-request download. The response `model` reports the model actually served." },
                  "input": { "oneOf": [ { "type": "string" }, { "type": "array", "items": { "type": "string" } } ] },
                  "encoding_format": { "type": "string" }
                }
              },
              "EmbeddingResponse": {
                "type": "object",
                "properties": {
                  "object": { "type": "string" },
                  "data": { "type": "array", "items": { "type": "object", "properties": { "object": { "type": "string" }, "index": { "type": "integer" }, "embedding": { "type": "array", "items": { "type": "number" } } } } },
                  "model": { "type": "string" },
                  "usage": { "$ref": "#/components/schemas/Usage" }
                }
              },
              "VerboseTranscriptionResponse": {
                "type": "object",
                "properties": {
                  "task": { "type": "string" },
                  "language": { "type": "string" },
                  "duration": { "type": "number" },
                  "text": { "type": "string" },
                  "segments": { "type": "array", "items": { "type": "object" } },
                  "words": { "type": "array", "items": { "type": "object" } }
                }
              },
              "DiarizationResponse": {
                "type": "object",
                "properties": {
                  "num_speakers": { "type": "integer" },
                  "segments": { "type": "array", "items": { "type": "object", "properties": { "start": { "type": "number" }, "end": { "type": "number" }, "speaker": { "type": "integer" } } } }
                }
              },
              "SpeakerEmbeddingResponse": {
                "type": "object",
                "properties": {
                  "object": { "type": "string" },
                  "data": { "type": "array", "items": { "type": "object" } },
                  "model": { "type": "string" },
                  "dimension": { "type": "integer" }
                }
              },
              "AthenaStopResponse": { "type": "object", "properties": { "status": { "type": "string" }, "model": { "type": "string" } } },
              "AdminStatusResponse": {
                "type": "object",
                "properties": { "model": { "type": "string" }, "listen": { "type": "string" }, "auth_enabled": { "type": "boolean" }, "users": { "type": "integer" }, "tokens": { "type": "integer" }, "admins": { "type": "integer" } }
              },
              "UsageEntry": {
                "type": "object",
                "properties": { "principal": { "type": "string" }, "requests": { "type": "integer" }, "prompt_tokens": { "type": "integer" }, "completion_tokens": { "type": "integer" }, "total_tokens": { "type": "integer" }, "updated": { "type": "number" } }
              },
              "UsageReportResponse": { "type": "object", "properties": { "usage": { "type": "array", "items": { "$ref": "#/components/schemas/UsageEntry" } } } },
              "AuditEntry": {
                "type": "object",
                "properties": { "id": { "type": "integer" }, "ts": { "type": "number" }, "principal": { "type": "string" }, "action": { "type": "string" }, "target": { "type": "string", "nullable": true }, "result": { "type": "string" }, "detail": { "type": "string", "nullable": true } }
              },
              "AuditReportResponse": { "type": "object", "properties": { "audit": { "type": "array", "items": { "$ref": "#/components/schemas/AuditEntry" } } } },
              "LogEntry": {
                "type": "object",
                "properties": { "ts": { "type": "string", "description": "ISO 8601 timestamp (unified-log native format)." }, "level": { "type": "string", "description": "OSLogType name: debug | info | default | error | fault." }, "category": { "type": "string", "description": "swift-log label minus the `athena.` prefix (e.g. `daemon`, `audit`, `model.llm`)." }, "message": { "type": "string", "description": "Daemon-emitted message body; includes `req=`/`principal=`/`function=` fields when the line was emitted inside a request task hierarchy (M45.3)." } }
              },
              "LogsReportResponse": { "type": "object", "properties": { "logs": { "type": "array", "items": { "$ref": "#/components/schemas/LogEntry" } }, "truncated": { "type": "boolean", "description": "Present and true only when older entries were dropped (window held more than `limit`, or the read deadline was hit). Omitted when nothing was dropped." } } },
              "ModelEntry": { "type": "object", "properties": { "name": { "type": "string" }, "bytes": { "type": "integer" }, "modified": { "type": "string" }, "modality": { "type": "string", "description": "Modality classified by ModelSupport (ADR 021): llm | vision | embedding | transcription | diarization | speaker | draft | unsupported.", "enum": ["llm", "vision", "embedding", "transcription", "diarization", "speaker", "draft", "unsupported"] }, "engine": { "type": "string", "description": "Sub-engine for asr/diarization: whisper | parakeet | sortformer | pyannote. Omitted otherwise." }, "loadability": { "type": "string", "description": "Packaging verdict: loadable | unknown | unsupported.", "enum": ["loadable", "unknown", "unsupported"] }, "draft": { "type": "boolean", "description": "Present and true for an MTP speculative drafter (never independently servable)." }, "fused_mtp": { "type": "boolean", "description": "Present and true for a servable LLM carrying fused MTP weights (weight-index probe, audit §4)." } } },
              "ModelListResponse": { "type": "object", "properties": { "models": { "type": "array", "items": { "$ref": "#/components/schemas/ModelEntry" } } } },
              "ModelDetailResponse": { "type": "object", "properties": { "name": { "type": "string" }, "path": { "type": "string" }, "bytes": { "type": "integer" }, "config": {} } },
              "ModelRemovedResponse": { "type": "object", "properties": { "name": { "type": "string" }, "removed": { "type": "boolean" } } },
              "DefaultModelResponse": { "type": "object", "properties": { "model": { "type": "string" }, "source": { "type": "string", "enum": ["config", "builtin"] } } },
              "SetDefaultModelRequest": { "type": "object", "required": ["name"], "properties": { "name": { "type": "string" } } },
              "ModelCopyRequest": { "type": "object", "required": ["src", "dst"], "properties": { "src": { "type": "string" }, "dst": { "type": "string" }, "copy": { "type": "boolean" }, "force": { "type": "boolean" } } },
              "ModelCopyResponse": { "type": "object", "properties": { "src": { "type": "string" }, "dst": { "type": "string" }, "path": { "type": "string" }, "aliased": { "type": "boolean" } } },
              "ModelPullRequest": { "type": "object", "required": ["id"], "properties": { "id": { "type": "string" }, "revision": { "type": "string" } } },
              "ModelConvertRequest": { "type": "object", "required": ["id"], "properties": { "id": { "type": "string" }, "revision": { "type": "string" }, "bits": { "type": "integer" }, "group_size": { "type": "integer" }, "name": { "type": "string" } } },
              "ModelPruneRequest": { "type": "object", "properties": { "dry_run": { "type": "boolean" } } },
              "ModelSlot": { "type": "object", "properties": { "module": { "type": "string", "enum": ["llm", "textEmbedding", "transcription", "diarization", "speakerEmbedding"] }, "allowed": { "type": "array", "items": { "type": "string" } }, "default": { "type": "string" }, "resident": { "type": "string", "nullable": true } } },
              "ModelResidentResponse": { "type": "object", "properties": { "slots": { "type": "array", "items": { "$ref": "#/components/schemas/ModelSlot" } } } },
              "ModelLoadRequest": { "type": "object", "required": ["module"], "properties": { "module": { "type": "string", "enum": ["llm", "textEmbedding", "transcription", "diarization", "speakerEmbedding"] }, "id": { "type": "string", "description": "Model id within the module's store models. Omit ⇒ the module's configured default." } } },
              "ModelLoadResponse": { "type": "object", "properties": { "module": { "type": "string" }, "id": { "type": "string", "description": "Id actually loaded into the slot." }, "status": { "type": "string", "enum": ["loaded"] } } },
              "ModelUnloadRequest": { "type": "object", "properties": { "module": { "type": "string", "description": "Module class to unload, or absent / `\"all\"` for every module." } } },
              "ModelUnloadResponse": { "type": "object", "properties": { "modules": { "type": "array", "items": { "type": "string" } }, "status": { "type": "string", "enum": ["unloaded"] } } },
              "UserSummary": { "type": "object", "properties": { "username": { "type": "string" }, "roles": { "type": "array", "items": { "type": "string" } } } },
              "UserListResponse": { "type": "object", "properties": { "users": { "type": "array", "items": { "$ref": "#/components/schemas/UserSummary" } } } },
              "CreateUserRequest": { "type": "object", "required": ["username", "password"], "properties": { "username": { "type": "string" }, "password": { "type": "string" }, "role": { "type": "string" } } },
              "UserRemovedResponse": { "type": "object", "properties": { "username": { "type": "string" }, "removed": { "type": "boolean" } } },
              "RoleCatalogEntry": { "type": "object", "properties": { "role": { "type": "string" }, "permissions": { "type": "array", "items": { "type": "string" } } } },
              "RolesResponse": { "type": "object", "properties": { "roles": { "type": "array", "items": { "$ref": "#/components/schemas/RoleCatalogEntry" } } } },
              "TokenSummary": { "type": "object", "properties": { "username": { "type": "string" }, "scope": { "type": "array", "items": { "type": "string" }, "nullable": true }, "hash_prefix": { "type": "string" }, "label": { "type": "string", "nullable": true }, "expires": { "type": "number", "nullable": true, "description": "Per-token expiry epoch seconds; null means never expires." } } },
              "TokenListResponse": { "type": "object", "properties": { "tokens": { "type": "array", "items": { "$ref": "#/components/schemas/TokenSummary" } } } },
              "CreateTokenRequest": { "type": "object", "required": ["user"], "properties": { "user": { "type": "string" }, "role": { "type": "array", "items": { "type": "string" } }, "label": { "type": "string" }, "ttl_secs": { "type": "integer", "description": "Per-token lifetime in seconds; absent/non-positive means never expires (subject to the daemon token_max_age_days cap)." } } },
              "RotateTokenRequest": { "type": "object", "properties": { "ttl_secs": { "type": "integer", "description": "New token lifetime in seconds; absent/non-positive means never expires. The rotated token's old TTL is not carried over." } } },
              "CreateTokenResponse": { "type": "object", "properties": { "user": { "type": "string" }, "scope": { "type": "array", "items": { "type": "string" }, "nullable": true }, "token": { "type": "string" }, "hash_prefix": { "type": "string" } } },
              "TokensRemovedResponse": { "type": "object", "properties": { "removed": { "type": "integer" } } },
              "Ok": { "type": "object", "properties": { "ok": { "type": "boolean" } } }
            }
          }
        }
        """#
    }
}
