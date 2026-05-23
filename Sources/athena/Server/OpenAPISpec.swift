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
            "description": "Athena is a single native macOS/MLX daemon that hosts LLM chat, embeddings, audio transcription/diarization, a built-in vector DB, and an async job queue behind one Metal memory governor. It is a **passive oracle**: the daemon answers inbound requests only and never initiates outbound connections (except fetching model weights and an opt-in remote-syslog sink). Anything a client needs is delivered by pull / long-poll / SSE — there are no result or billing webhooks.\n\nTwo HTTP dialects are served:\n- `/v1/*` — OpenAI-compatible (drop-in for OpenAI SDKs).\n- `/api/*` — Athena's own minimal native dialect.\n\nAll errors share the envelope `{\"error\":{\"message\",\"type\",\"code\"}}`. Authentication is a bearer token (`Authorization: Bearer <token>`); each route requires a single RBAC permission. When auth is disabled (loopback, no seeded users) every route is open.",
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
            { "name": "Vectors", "description": "Built-in vector database." },
            { "name": "Store", "description": "Shared-store admin (export/stats)." },
            { "name": "Queue", "description": "Async job queue (submit / poll / SSE / cancel)." },
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
                "summary": "Liveness + memory-governor snapshot.",
                "description": "Always open (no auth). Returns the governor's current reservations and budget.",
                "security": [],
                "responses": {
                  "200": { "description": "Governor snapshot.", "content": { "application/json": { "schema": { "type": "object" } } } }
                }
              }
            },
            "/metrics": {
              "get": {
                "tags": ["Operational"],
                "summary": "In-memory metrics snapshot (JSON).",
                "description": "Global, process-lifetime counters (reset on restart). Requires the `metrics.read` permission.",
                "responses": {
                  "200": { "description": "Metrics snapshot.", "content": { "application/json": { "schema": { "type": "object" } } } },
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
                "description": "OpenAI-compatible. Set `stream:true` for an SSE stream of `chat.completion.chunk` events terminated by `data: [DONE]`; with `stream_options.include_usage:true` a final usage-only chunk precedes it. `response_format`/`tools` route into structured output. `top_p`/`seed` are honored only on the sampling path and are inert under greedy/MTP/structured decoding. `n>1`, `logprobs`, `top_logprobs`, and non-empty `logit_bias` are rejected 400.",
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
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/models": {
              "get": {
                "tags": ["Models"],
                "summary": "List available models.",
                "description": "OpenAI list shape. Requires `model.read`.",
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
                "description": "OpenAI retrieve shape. Requires `model.read`.",
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
                "description": "OpenAI-compatible. `input` is a string or array of strings.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/EmbeddingRequest" } } } },
                "responses": {
                  "200": { "description": "Embeddings.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/EmbeddingResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/audio/transcriptions": {
              "post": {
                "tags": ["Audio"],
                "summary": "Transcribe audio.",
                "description": "OpenAI-compatible multipart upload. `response_format` of `json` (default), `text`, `srt`, `vtt`, or `verbose_json`; `timestamp_granularities[]=word` adds word timings; `diarize=true` adds per-segment speaker ids in verbose_json.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary", "description": "Audio file." },
                      "model": { "type": "string" },
                      "language": { "type": "string" },
                      "response_format": { "type": "string", "enum": ["json", "text", "srt", "vtt", "verbose_json"] },
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
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/audio/diarizations": {
              "post": {
                "tags": ["Audio"],
                "summary": "Diarize audio (who spoke when).",
                "description": "Multipart upload. Returns speaker-labelled time spans.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary" },
                      "model": { "type": "string" },
                      "num_speakers": { "type": "integer" }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Diarization.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DiarizationResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/audio/embeddings": {
              "post": {
                "tags": ["Audio"],
                "summary": "Speaker (voice) embeddings.",
                "description": "Multipart upload. Returns an embedding per requested segment.",
                "requestBody": {
                  "required": true,
                  "content": { "multipart/form-data": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                      "file": { "type": "string", "format": "binary" },
                      "model": { "type": "string" },
                      "segments": { "type": "string", "description": "JSON array of {start,end} segment specs (seconds)." }
                    }
                  } } }
                },
                "responses": {
                  "200": { "description": "Speaker embeddings.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SpeakerEmbeddingResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/v1/vectors": {
              "post": {
                "tags": ["Vectors"],
                "summary": "Upsert a vector.",
                "description": "Provide `vector` directly, or `text` to embed server-side. Requires `vectors.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorUpsertRequest" } } } },
                "responses": {
                  "200": { "description": "Upserted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorIdResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/vectors/{id}": {
              "delete": {
                "tags": ["Vectors"],
                "summary": "Delete a vector by id.",
                "description": "Requires `vectors.write`.",
                "parameters": [ { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } } ],
                "responses": {
                  "200": { "description": "Delete result.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorIdResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/vectors/query": {
              "post": {
                "tags": ["Vectors"],
                "summary": "Nearest-neighbour query.",
                "description": "Provide `vector` or `text`; `k` results (default server-side). Requires `vectors.read`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorQueryRequest" } } } },
                "responses": {
                  "200": { "description": "Matches.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorQueryResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/vectors/stats": {
              "get": {
                "tags": ["Vectors"],
                "summary": "Vector DB statistics.",
                "description": "Requires `vectors.read`.",
                "responses": {
                  "200": { "description": "Stats.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorStatsResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/store/export": {
              "post": {
                "tags": ["Store"],
                "summary": "Export the shared store to a file.",
                "description": "Requires `store.admin`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/StoreExportRequest" } } } },
                "responses": {
                  "200": { "description": "Export result.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/StoreExportResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/store/stats": {
              "get": {
                "tags": ["Store"],
                "summary": "Shared-store statistics.",
                "description": "Requires `store.admin`.",
                "responses": {
                  "200": { "description": "Stats.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/StoreStatsResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/queue/{arg}": {
              "post": {
                "tags": ["Queue"],
                "summary": "Submit an async job.",
                "description": "`{arg}` is the job kind (e.g. `chat`, `embed`). Returns a job id to poll. Requires `queue.submit`. Jobs are owner-scoped.",
                "parameters": [ { "name": "arg", "in": "path", "required": true, "schema": { "type": "string" }, "description": "On POST: the job kind. On GET/DELETE: the job id." } ],
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object", "description": "The same body the synchronous endpoint of this kind accepts." } } } },
                "responses": {
                  "202": { "description": "Job accepted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/QueueSubmitResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "get": {
                "tags": ["Queue"],
                "summary": "Get job status/result.",
                "description": "`{arg}` is the job id. Owner-scoped. Requires `queue.submit`.",
                "parameters": [ { "name": "arg", "in": "path", "required": true, "schema": { "type": "string" }, "description": "Job id." } ],
                "responses": {
                  "200": { "description": "Job status.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/QueueStatusResponse" } } } },
                  "404": { "$ref": "#/components/responses/NotFound" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              },
              "delete": {
                "tags": ["Queue"],
                "summary": "Cancel/remove a job.",
                "description": "`{arg}` is the job id. Owner-scoped. Requires `queue.submit`.",
                "parameters": [ { "name": "arg", "in": "path", "required": true, "schema": { "type": "string" }, "description": "Job id." } ],
                "responses": {
                  "200": { "description": "Removal result.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/QueueRemoveResponse" } } } },
                  "404": { "$ref": "#/components/responses/NotFound" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/queue": {
              "get": {
                "tags": ["Queue"],
                "summary": "List the caller's jobs.",
                "description": "Owner-scoped. Requires `queue.submit`.",
                "responses": {
                  "200": { "description": "Jobs.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/QueueListResponse" } } } },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/v1/queue/{arg}/events": {
              "get": {
                "tags": ["Queue"],
                "summary": "Stream a job's status transitions (SSE).",
                "description": "`{arg}` is the job id. An inbound long-lived SSE stream until the job reaches a terminal state — the passive-oracle alternative to a webhook. Requires `queue.submit`.",
                "parameters": [ { "name": "arg", "in": "path", "required": true, "schema": { "type": "string" }, "description": "Job id." } ],
                "responses": {
                  "200": { "description": "SSE stream of status objects.", "content": { "text/event-stream": { "schema": { "type": "string" } } } },
                  "404": { "$ref": "#/components/responses/NotFound" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/chat": {
              "post": {
                "tags": ["Native"],
                "summary": "Native chat (Athena dialect).",
                "description": "Minimal non-OpenAI shape. `stream:true` returns NDJSON content chunks ending in `{content:\"\",done:true}`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AthenaChatRequest" } } } },
                "responses": {
                  "200": { "description": "Chat reply (object or NDJSON stream).", "content": {
                    "application/json": { "schema": { "$ref": "#/components/schemas/AthenaChatResponse" } },
                    "application/x-ndjson": { "schema": { "type": "string" } }
                  } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
                  "503": { "$ref": "#/components/responses/Overloaded" }
                }
              }
            },
            "/api/embed": {
              "post": {
                "tags": ["Native"],
                "summary": "Native embeddings (Athena dialect).",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AthenaEmbedRequest" } } } },
                "responses": {
                  "200": { "description": "Embeddings.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AthenaEmbedResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" },
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
                "summary": "Pull a model from Hugging Face (async).",
                "description": "Dispatched to the queue; poll the returned job id. Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelPullRequest" } } } },
                "responses": {
                  "202": { "description": "Job accepted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelJobResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/convert": {
              "post": {
                "tags": ["Model store"],
                "summary": "Convert/quantize a model (async).",
                "description": "Dispatched to the queue. Requires `model.write`.",
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelConvertRequest" } } } },
                "responses": {
                  "202": { "description": "Job accepted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelJobResponse" } } } },
                  "400": { "$ref": "#/components/responses/BadRequest" },
                  "401": { "$ref": "#/components/responses/Unauthorized" },
                  "403": { "$ref": "#/components/responses/Forbidden" }
                }
              }
            },
            "/api/models/prune": {
              "post": {
                "tags": ["Model store"],
                "summary": "Prune unreferenced model blobs (async).",
                "description": "Dispatched to the queue; `dry_run:true` lists candidates only. Requires `model.write`.",
                "requestBody": { "required": false, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelPruneRequest" } } } },
                "responses": {
                  "202": { "description": "Job accepted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelJobResponse" } } } },
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
              "Overloaded": { "description": "Backpressure — the memory governor or a rate/concurrency cap declined the request. Retry after the indicated delay.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } }
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
                  "content": { "type": "string", "nullable": true },
                  "tool_calls": { "type": "array", "items": { "type": "object" } }
                }
              },
              "ChatCompletionRequest": {
                "type": "object",
                "required": ["messages"],
                "properties": {
                  "model": { "type": "string" },
                  "messages": { "type": "array", "items": { "$ref": "#/components/schemas/ChatMessage" } },
                  "stream": { "type": "boolean" },
                  "stream_options": { "type": "object", "properties": { "include_usage": { "type": "boolean" } } },
                  "response_format": { "type": "object", "description": "OpenAI response_format (text | json_object | json_schema)." },
                  "tools": { "type": "array", "items": { "type": "object" } },
                  "tool_choice": {},
                  "max_tokens": { "type": "integer" },
                  "temperature": { "type": "number" },
                  "top_p": { "type": "number", "description": "Honored only on the sampling path; inert under greedy/MTP/structured decoding." },
                  "seed": { "type": "integer", "description": "Honored only on the sampling path." },
                  "stop": { "description": "String or array of strings; truncates output at the first match.", "oneOf": [ { "type": "string" }, { "type": "array", "items": { "type": "string" } } ] },
                  "n": { "type": "integer", "description": "Rejected with 400 when > 1." },
                  "logprobs": { "description": "Rejected with 400 when requested." },
                  "top_logprobs": { "type": "integer", "description": "Rejected with 400 when present." },
                  "logit_bias": { "type": "object", "description": "Rejected with 400 when non-empty." }
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
                  "model": { "type": "string" },
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
              "VectorUpsertRequest": {
                "type": "object",
                "required": ["id"],
                "properties": {
                  "id": { "type": "string" },
                  "vector": { "type": "array", "items": { "type": "number" } },
                  "text": { "type": "string" },
                  "metadata": {}
                }
              },
              "VectorIdResponse": { "type": "object", "properties": { "id": { "type": "string" } } },
              "VectorQueryRequest": {
                "type": "object",
                "properties": {
                  "vector": { "type": "array", "items": { "type": "number" } },
                  "text": { "type": "string" },
                  "k": { "type": "integer" }
                }
              },
              "VectorQueryResponse": {
                "type": "object",
                "properties": { "matches": { "type": "array", "items": { "type": "object", "properties": { "id": { "type": "string" }, "score": { "type": "number" }, "metadata": {} } } } }
              },
              "VectorStatsResponse": {
                "type": "object",
                "properties": { "count": { "type": "integer" }, "dim": { "type": "integer" }, "bytes": { "type": "integer" }, "cap_bytes": { "type": "integer" } }
              },
              "StoreExportRequest": { "type": "object", "required": ["path"], "properties": { "path": { "type": "string" } } },
              "StoreExportResponse": { "type": "object", "properties": { "path": { "type": "string" }, "bytes": { "type": "integer" } } },
              "StoreStatsResponse": { "type": "object", "properties": { "vectors": { "type": "integer" }, "jobs": { "type": "integer" }, "bytes": { "type": "integer" }, "path": { "type": "string" } } },
              "QueueSubmitResponse": { "type": "object", "properties": { "id": { "type": "string" }, "status": { "type": "string" } } },
              "QueueStatusResponse": {
                "type": "object",
                "properties": { "id": { "type": "string" }, "kind": { "type": "string" }, "status": { "type": "string", "enum": ["queued", "running", "done", "error", "canceled"] }, "result": {}, "error": { "type": "string", "nullable": true } }
              },
              "QueueListResponse": {
                "type": "object",
                "properties": { "jobs": { "type": "array", "items": { "type": "object", "properties": { "id": { "type": "string" }, "kind": { "type": "string" }, "status": { "type": "string" }, "created": { "type": "number" }, "updated": { "type": "number" } } } } }
              },
              "QueueRemoveResponse": { "type": "object", "properties": { "id": { "type": "string" }, "removed": { "type": "boolean" } } },
              "AthenaChatRequest": {
                "type": "object",
                "required": ["messages"],
                "properties": {
                  "model": { "type": "string" },
                  "messages": { "type": "array", "items": { "type": "object", "properties": { "role": { "type": "string" }, "content": { "type": "string" } } } },
                  "stream": { "type": "boolean" },
                  "max_tokens": { "type": "integer" },
                  "temperature": { "type": "number" }
                }
              },
              "AthenaChatResponse": { "type": "object", "properties": { "model": { "type": "string" }, "content": { "type": "string" }, "done": { "type": "boolean" } } },
              "AthenaEmbedRequest": {
                "type": "object",
                "required": ["input"],
                "properties": { "model": { "type": "string" }, "input": { "oneOf": [ { "type": "string" }, { "type": "array", "items": { "type": "string" } } ] } }
              },
              "AthenaEmbedResponse": { "type": "object", "properties": { "model": { "type": "string" }, "embeddings": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } } } },
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
              "ModelEntry": { "type": "object", "properties": { "name": { "type": "string" }, "bytes": { "type": "integer" }, "modified": { "type": "string" } } },
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
              "ModelJobResponse": { "type": "object", "properties": { "job_id": { "type": "string" }, "status": { "type": "string" } } },
              "UserSummary": { "type": "object", "properties": { "username": { "type": "string" }, "roles": { "type": "array", "items": { "type": "string" } } } },
              "UserListResponse": { "type": "object", "properties": { "users": { "type": "array", "items": { "$ref": "#/components/schemas/UserSummary" } } } },
              "CreateUserRequest": { "type": "object", "required": ["username", "password"], "properties": { "username": { "type": "string" }, "password": { "type": "string" }, "role": { "type": "string" } } },
              "UserRemovedResponse": { "type": "object", "properties": { "username": { "type": "string" }, "removed": { "type": "boolean" } } },
              "RoleCatalogEntry": { "type": "object", "properties": { "role": { "type": "string" }, "permissions": { "type": "array", "items": { "type": "string" } } } },
              "RolesResponse": { "type": "object", "properties": { "roles": { "type": "array", "items": { "$ref": "#/components/schemas/RoleCatalogEntry" } } } },
              "TokenSummary": { "type": "object", "properties": { "username": { "type": "string" }, "scope": { "type": "array", "items": { "type": "string" }, "nullable": true }, "hash_prefix": { "type": "string" }, "label": { "type": "string", "nullable": true } } },
              "TokenListResponse": { "type": "object", "properties": { "tokens": { "type": "array", "items": { "$ref": "#/components/schemas/TokenSummary" } } } },
              "CreateTokenRequest": { "type": "object", "required": ["user"], "properties": { "user": { "type": "string" }, "role": { "type": "array", "items": { "type": "string" } }, "label": { "type": "string" }, "ttl_secs": { "type": "integer", "description": "Per-token lifetime in seconds; absent/non-positive means never expires (subject to the daemon token_max_age_days cap)." } } },
              "CreateTokenResponse": { "type": "object", "properties": { "user": { "type": "string" }, "scope": { "type": "array", "items": { "type": "string" }, "nullable": true }, "token": { "type": "string" }, "hash_prefix": { "type": "string" } } },
              "TokensRemovedResponse": { "type": "object", "properties": { "removed": { "type": "integer" } } },
              "Ok": { "type": "object", "properties": { "ok": { "type": "boolean" } } }
            }
          }
        }
        """#
    }
}
