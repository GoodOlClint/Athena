# 013 — `/v1` is the inference surface; `/api` is control-only; refuse the OpenAI platform tail

**Status:** Accepted — decisions + staged rollout (implementation gated per slice below)
**Date:** 2026-06-17
**Milestone:** proposed (operator numbers/tags); follows the API "keep vs toss" review.
**Supersedes/relates:** refines the "Public surface" section of `CLAUDE.md`; pairs with the
ADR 010 audio-division-of-labor correction.

> **Amended (ADR 025, M80):** three native `/v1/*` data-plane surfaces were
> **removed** as breaking changes — `/v1/vectors*` + `/v1/store/export`+`/v1/store/stats`
> (v0.10.201) and `/v1/queue*` (v0.10.203). The "single inference surface" principle is
> unchanged; the daemon now persists no request content, and long-running model
> lifecycle ops stream Server-Sent Events synchronously on `POST /api/models/{pull,
> convert,prune}` (no async queue, no job id). The CLAUDE.md "Stable `/v1/*`" list is
> updated to match.

## Context

A "keep vs toss" review mapped Athena's full HTTP surface (48 paths / 67 methods) against
OpenAI's. Findings:

- **There is a de-facto standard, but it moves and it's single-vendor.** OpenAI publishes a
  machine-readable spec (`github.com/openai/openai-openapi`, "PRs will not be merged"). The
  *interop-stable subset the whole local ecosystem froze around* is `chat/completions` +
  `embeddings` + `models`; everything past it (Responses, Batches, Assistants, fine-tuning,
  images, moderations, files) is OpenAI **platform** surface that no local server implements.
- **Athena's `/v1` core is faithful** on that frozen subset (SSE + `[DONE]`,
  `stream_options.include_usage`, `max_completion_tokens` precedence,
  `prompt_tokens_details.cached_tokens`, `tool_choice`, `json_schema`, `finish_reason`,
  `GET /v1/models{/id}`).
- **The native `/api/*` dialect mixes two unlike things:** (a) **inference** endpoints
  `/api/chat` + `/api/embed` that *duplicate* `/v1` with a divergent streaming format
  (NDJSON vs SSE), and (b) a **control** plane (model-store, RBAC, allowlist, lifecycle,
  audit, usage, logs, cache) with **no OpenAI equivalent**.
- **Consumer reality (verified in-repo):** `/api/embed` has **zero callers**. `/api/chat`'s
  only in-repo caller is the `athena run` CLI ([`clients/Sources/AthenaClient/Run.swift:57`])
  sending `{model, messages, stream:false}` and reading `content` — trivially representable
  on `/v1/chat/completions`. External consumers speak `/v1` (the embeddings call-sites hit
  `/v1/embeddings`; consumers are OpenAI-compatible).
- This is the API-surface application of ADR 011: the OpenAI core is **tax you must pay**
  (the interop standard) and should be paid faithfully but minimally; the control plane is
  the **moat surface**; the duplicate native inference dialect is **redundant tax**; and the
  OpenAI platform tail is **tax you must refuse**.

## Decision

### 1. `/v1` is the single canonical inference surface; `/api` is the control plane.

Deprecate the native **inference** dialect (`/api/chat`, `/api/embed`) and keep `/api/*`
strictly for daemon **control** (model-store, RBAC, allowlist, lifecycle, audit, usage,
logs, cache — things OpenAI has no shape for). One clean story: *OpenAI dialect for
inference, native dialect for governing the daemon.* Amends the `CLAUDE.md` "Public surface"
table (`/api/*` is **control**, not "native inference").

Staged, non-breaking rollout (each its own commit + tag, appVersion bump in the
slice commit):

- **(a) `/api/embed` → deprecate immediately.** No callers. Mark `deprecated: true` in
  `OpenAPISpec.swift`; remove on a later minor.
- **(b) migrate `athena run` to `/v1/chat/completions`.** One-function change in `Run.swift`
  (swap URL; read `choices[0].message.content`; keep non-stream). After this, `/api/chat`
  has no in-repo caller.
- **(c) `/api/chat` → deprecate.** Mark `deprecated: true`; keep serving through the sunset.
- **(d) removal (breaking).** The external-consumer gate is **cleared** (2026-06-17):
  the platform does not consume Athena yet, and the consuming application — the one external consumer that
  does — will handle its own migration to `/v1`. So removal is no longer blocked on a
  consumer audit; it lands as its own ratified, breaking slice once the deprecation period
  has given the consuming application time to migrate, with a version bump the operator chooses.

The **control** plane stays and keeps growing — it is the governor/multi-tenant moat surface.

### 2. Audio division of labor (write it down once).

`/v1/audio/*` = audio **analysis** (transcribe / diarize / speaker-embed via dedicated
Whisper / Sortformer / WeSpeaker — best-in-class, canonical *forever*). Chat content-parts
(`image_url`, and future `input_audio`) = **reasoning** over media via the VLM. Complementary,
not overlapping — exactly as OpenAI ships both Whisper and `gpt-4o-audio`. (Pairs with the
ADR 010 deferral-rationale correction made the same day.)

### 3. Converge diarization output to OpenAI's `diarized_json` (actionable).

OpenAI shipped diarization as `response_format: diarized_json` on the transcription endpoint
(model `gpt-4o-transcribe-diarize`). Add `diarized_json` as a `response_format` on
`POST /v1/audio/transcriptions`, emitting speaker-labeled segments in OpenAI's shape — the
long-standing backlog item **#4a**, now with a published shape to match. Athena's existing
`diarize=true` + segment `speaker` field stays; this adds the standard alias. The standalone
`/v1/audio/diarizations` endpoint remains as an Athena extension (diarization-without-
transcription, which OpenAI has no equivalent for); consider an `rttm` `response_format` on
it later for pyannote-ecosystem interop (NIST RTTM is the only neutral cross-tool artifact).

### 4. Emit `top_logprobs` on the deterministic path instead of 400 (actionable).

The greedy/structured decode path already computes the logits, so `logprobs`/`top_logprobs`
can be **honored**, not rejected. Stop 400-ing them in `unsupportedParameter()`; emit the
OpenAI `logprobs` response object. A cheap OpenAI-compat win for eval/confidence consumers.

### 5. Client/precondition faults return a cause-naming 4xx, not a catch-all 5xx (actionable; issue #4 + cluster).

**Principle:** a fault caused by the caller's request or by the requested model's shape (a
precondition the caller can fix) returns a **4xx** with a `code`/`message` that names the
real cause; only genuine daemon faults return **5xx**.

Issue #4 is one instance, not the whole problem. An audit found a **cluster** rooted in a
catch-all: `AthenaError.classify()` maps any *unrecognized* error to `500
module_load_failed`, and several client-caused conditions aren't modeled as error cases, so
they fall through to that 500 (and to misleading text — "module failed to load" when the
module loaded fine). Confirmed siblings, highest operator-impact first:

- **Audio upload faults** (`AthenaTranscription/AudioDecode.swift`) — malformed/corrupt
  audio, over-length (~4h decode cap), and unsupported codec all surface as `500
  module_load_failed`. Should be `400 invalid_audio` / `413`-or-`400 audio_too_long` /
  `400 audio_format_unsupported`. (Higher real-world impact than #4 — a bad MP3 upload is
  common.)
- **No chat template (issue #4)** — `missingChatTemplate` → `500 module_load_failed`;
  should be `400 no_chat_template` / `invalid_request_error`, message pointing at an
  instruct/`-it` checkpoint. Model loads & serves fine for capable callers — only the
  classification changes.
- **Per-module rebind catch-alls (4×: embed/transcribe/diarize/speaker)** — generic `500
  internal_error` swallows the real cause; route through the classification seam instead.
- **Queue submit** — generic `500 queue_error` hides the underlying cause (e.g. a queued
  chat hitting `model_not_available`); pass the real error through.

Fix shape: model these as proper `AthenaError` cases (or map them before the catch-all) so
the seam emits the right 4xx; keep the substrate's detail string in the server log, not the
client body. Tracked as error-legibility issue #6 (seeded by #4); each fix is a
small, test-pinned slice with a regression test asserting the status+code.

### 6. Refuse the OpenAI platform tail (don't-build list, so it isn't re-litigated as "gaps").

**Will not implement** (not local-inference features; no peer local server implements them):
`/v1/images/generations`, `/v1/moderations`, `/v1/fine_tuning`, `/v1/assistants`,
`/v1/files`, `/v1/batches`, `/v1/responses` (stateful — conflicts with the stateless
passive-oracle; **watch** only, reassess if it displaces Chat Completions as the interop
standard), and legacy `/v1/completions` (OpenAI-deprecated). `/v1/audio/speech` (TTS) is a
genuine *modality* gap, not platform tail — out of scope here, build only if a consumer
wants audio output (its own gate; no TTS model today).

### 7. Reasoning/thinking control: keep `enable_thinking`; add `reasoning_effort` only on demand.

Per-request thinking control **already ships** via `chat_template_kwargs:
{"enable_thinking": false}` (M46.3b) — the de-facto ecosystem mechanism for template-driven
thinking (Qwen3-class). OpenAI's *idiomatic* knob is `reasoning_effort`
(`minimal|low|medium|high`), which Athena currently **silently ignores** (it isn't in
`unsupportedParameter()`; unknown fields just decode away). Decision: **document the existing
`enable_thinking` knob now** (cheap, in OpenAPI + quickstart), and **recognize
`reasoning_effort` as an alias only when a consumer actually drives it** — mapping
`minimal`/(non-standard) `none` → `enable_thinking=false`, and `low/medium/high` → template
default (the models have an on/off toggle, not a graded dial, so a finer mapping would be
fiction). Recorded here so it isn't re-litigated; not a build-now.

### Kept, deliberate divergences (recorded, not changing)

- `n > 1` and non-empty `logit_bias` → **400** (incompatible with the deterministic
  greedy/structured path). Unchanged.
- `top_p`/`seed` honored on the sampling path, **inert** under greedy/structured. Document as
  inert rather than error.
- Error envelope `{"error":{"message","type","code"}}` — unchanged, matches OpenAI.

## Consequences

- **One inference surface.** New chat/embedding features (vision already; audio if it ever
  lands; `diarized_json`; `logprobs`) target `/v1` only — no double-maintenance. This is why
  M71 vision and the deferred M72 audio were/are `/v1`-only.
- **Control plane unaffected and affirmed** as the moat surface.
- **`athena run` gains a tiny coupling to `/v1`** — acceptable; it's the OpenAI-standard path.
- **Removal is breaking** but the external-consumer gate is **cleared** (the platform N/A;
  the consuming application self-migrates) — it lands after the deprecation period gives the consuming application
  time to move. Deprecation (steps a–c) is non-breaking and safe to land now.
- **Drift-guard** (spec↔routes) must stay green across every `deprecated:`/schema edit.
- **`diarized_json` + `logprobs` + the issue-#4 400 reclassification** are additive/behavioural
  `/v1` slices, each test-pinned (stub-tier e2e + real-model RUNBOOK where a model is needed).
  The #4 fix is the cheapest (pure error-classification; a converted base checkpoint reproduces
  it).
