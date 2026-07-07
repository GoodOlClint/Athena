# API surface — compatibility map

Athena serves multiple API dialects as protocol adapters over one inference engine. This page tells you, per endpoint, whether it is a drop-in for a vendor SDK or an Athena-native extension. The canonical, machine-readable reference is always **`GET /openapi.json`** (served live by the daemon, no auth) — if this page and the spec ever disagree, the spec wins.

Tags:

- **[oai]** — OpenAI-compatible drop-in: same route, request, and response shapes as the OpenAI API; point an OpenAI SDK at Athena's base URL and it works.
- **[anthropic]** — Anthropic-compatible drop-in: the Anthropic Messages API shape.
- **[native]** — Athena extension: no vendor equivalent. Stable, but only Athena serves it.

## Inference — `/v1/*`

| Endpoint | Tag | Notes |
| --- | --- | --- |
| `POST /v1/chat/completions` | [oai] | Streaming SSE, tool calls (incl. `tool_choice: "auto"`), JSON-schema structured output, vision/image input (base64 / `data:` URLs only — the daemon never fetches remote images), `stop`/`seed`/`top_p`, `logprobs`. |
| `POST /v1/messages` | [anthropic] | Anthropic Messages API: streaming, system prompts, tool use, `x-api-key` accepted as a bearer alias. Verified end-to-end with real Claude Code — see [claude-code.md](claude-code.md). |
| `POST /v1/embeddings` | [oai] | Text embeddings. |
| `POST /v1/audio/transcriptions` | [oai] | Whisper + Parakeet backends (selected by `model=`), word timestamps, `srt`/`vtt`/`verbose_json` response formats. See [transcription.md](transcription.md). |
| `POST /v1/video/transcriptions` | [native] | Demuxes the video's audio track in memory, then transcribes it; OpenAI audio-transcription response shape for convenience, but the route itself has no vendor equivalent. See [video.md](video.md). |
| `POST /v1/audio/diarizations` | [native] | Who-spoke-when. Multiple backends via `method=` (`sortformer` default ≤4 speakers, `pyannote`, `cluster`); `num_speakers` / `min_speakers` / `max_speakers` hints; overlap-aware segments. See [diarization.md](diarization.md). |
| `POST /v1/audio/embeddings` | [native] | Speaker embeddings (256-d voice vectors) for client-side speaker identification. |
| `GET /v1/models`, `GET /v1/models/{id}` | [oai] | Lists models present in the local store. |

### Request-level extensions on compatible routes

These are Athena extensions accepted *inside* otherwise-compatible requests. Vendor SDKs pass them through as extra fields; other servers will reject or ignore them.

- `chat_template_kwargs` (chat) — opaque kwargs passed to the model's chat template; the canonical use is `{"enable_thinking": false}` to suppress the reasoning channel on thinking-capable models. When a model does emit reasoning, it is surfaced as `reasoning_content` on the message/delta rather than leaking into `content`.
- `timeout` (chat) — per-request generation timeout in seconds, within the daemon cap.
- `speculative` (chat) — opt-in speculative decoding on models with an MTP head or a paired drafter; lossless (target-verified), speed is model-dependent.
- `response_format: "diarized_json"` (audio transcriptions) — transcript merged with diarization turns in one call.

## Control plane — `/api/*`

Athena-native daemon administration. **Not inference** — inference lives exclusively under `/v1/*`.

| Area | Endpoints |
| --- | --- |
| Model store + lifecycle | `GET /api/models`, `GET /api/models/{name}`, `POST /api/models/{pull,convert,prune}` (synchronous, SSE progress), `POST /api/models/{load,unload,copy,default}`, `GET /api/models/resident` |
| RBAC | `/api/users*`, `/api/roles`, `/api/tokens*` (create/rotate/revoke) |
| Observability | `GET /api/audit`, `GET /api/usage`, `GET /api/logs`, `GET /api/logs/stream` |
| Daemon | `GET /api/admin/status`, `POST /api/admin/{restart,stop}`, `GET`/`PUT /api/config` (security-critical keys are TOML-plus-sudo only), `GET /api/cache/prompt` (prompt-cache stats) |

Unauthenticated basics: `GET /healthz` (liveness + governor/memory stats) and `GET /openapi.json`.

## Compatibility promises

- Every route requires a single RBAC permission (`Authorization: Bearer <token>`); loopback dev mode with no seeded users opens every route.
- All errors, on every surface, use one envelope: `{"error":{"message","type","code"}}`.
- Unsupported OpenAI parameters are rejected with a clear `400` (e.g. `n > 1`, `logit_bias`) rather than silently ignored.
- OpenAI platform-tail endpoints (assistants, batches, files, fine-tuning, moderations) are deliberately not implemented.
