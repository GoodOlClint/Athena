# ADR 013 rollout — change plan (slice order, test bar, sequencing)

**Decision record:** `docs/decisions/013-v1-inference-surface-api-control-only.md`.
**Tracking:** error-legibility cluster = issue #6 (seed #4); native-inference deprecation +
`diarized_json`/`logprobs` = ADR 013 decisions.
**Milestone number operator-driven** (M72 is taken by vision-convert; this plan is
number-neutral — operator assigns tags).

## Principle

Three independent workstreams, ordered low-risk / high-legibility first. Each slice is a
stacked, test-pinned commit honoring the pre-commit pipeline (Tests → Security → Quality →
Refactor): `./deploy/build.sh Release` (xcodebuild — MLX needs full Xcode) → unit test +
a `deploy/e2e-rbac.sh` phase → **appVersion bump in the slice commit** → annotated tag
direct-to-main → `graphify update .`. Every fix lands with a regression test; the
spec↔routes drift-guard stays green across any `OpenAPISpec.swift` change.

## Status (verified against code 2026-06-19, v0.10.187)

Backlog audit (`docs/backlog-hitlist.md`) reconciled this plan against `Sources/`:

| Slice | State | Evidence |
|---|---|---|
| A1 no-chat-template → 400 | ✅ shipped | `AthenaError.swift:135` `.noChatTemplate → 400`, code `no_chat_template`; `isMissingChatTemplate` maps the substrate error |
| A2 audio-upload faults → 4xx | ✅ shipped | `AthenaError` `audioTooLong`/`invalidAudio`/`audioFormatUnsupported → 400`, wired in `AudioDecode.swift` |
| A3 catch-all sweep (4× rebind + queue) | ✅ **shipped** (verified 2026-06-19) | all 4 rebind handlers route their generic `catch` through `Self.classified()` (embed `:1694`, transcription `:1788`, diarization `:2148`, speaker `:2251`); both queue-submit sites classify `AthenaError` + 500 only on a genuine fault (`:3189`,`:4596`); `classify()` unit-pinned in `AthenaErrorTests`. The "partial" verdict was conservative — no code change needed |
| B1 deprecate `/api/embed` | ✅ shipped | `OpenAPISpec.swift` `deprecated:true` |
| B2 migrate `athena run` → `/v1` | ✅ shipped | `clients/.../Run.swift:59,101-108` |
| B3 deprecate `/api/chat` | ✅ shipped | `OpenAPISpec.swift` `deprecated:true` |
| C1 `diarized_json` response_format | ❌ **open** | transcription switch (`AthenaServer.swift:1827,2042`) handles srt/vtt/verbose_json only; 0 hits for `diarized_json` |
| C2 honor `logprobs`/`top_logprobs` | ❌ **open** | `OpenAIDTO.swift:242-248` still 400s both; spec still documents the rejection |
| D1 native `/api/embed` metering (ADR 007 #8) | ✅ **shipped** (v0.10.188) | `handleNativeEmbed` now calls `meter()` mirroring `/v1/embeddings`; e2e asserts alice prompt_tokens grow after `/api/embed` |

**Track 2 (this batch) = the open remainder: A3-finish, D, C1, C2.** Recommended order below
keeps the plan's low-risk/high-legibility-first principle, but **C1 is the one silently
degrading a live consumer today** (a `diarized_json` request falls through to plain `{text}`,
0 speakers) — pull it forward if external impact outranks risk-ordering.

1. **A3-finish** — per-handler audit of the 4× rebind catch-alls (embed/transcribe/diarize/
   speaker) + queue-submit; route any remaining generic `500 internal_error`/`500 queue_error`
   through `AthenaError.classify()`. Cheapest, no contract change. *(Also closes hitlist #9.)*
2. **D — native `/api/embed` metering** — compute `TokenUsage` from the embed batch and call
   `meter(principal:usage:)`, mirroring `handleNativeChat`. One handler. *(Closes hitlist #6;
   ADR 007 #8 — note `/api/embed` is itself deprecated, so this is metering-completeness, not
   new surface.)*
3. **C1 — `diarized_json`** — add the enum value + a switch case at both transcription sites
   reusing the existing `diarize=true` segment+speaker path; serialize OpenAI's published
   shape. Mark native-flavored in the op `description` (reusing an OpenAI response shape over
   a route whose semantics are Athena's does **not** make it a drop-in — `/v1` compatibility
   rule). Real-model RUNBOOK validation. *(Closes hitlist #4.)*
4. **C2 — honor `logprobs`/`top_logprobs`** — remove both from `unsupportedParameter()`;
   capture the top-k logits the greedy/structured path already computes and emit the OpenAI
   `logprobs` response object; update the spec text (drift-guard). Keep `n>1`/`logit_bias` →
   400. *(Closes hitlist #5; reverses M31's reject-behavior per ADR 013 §4.)*

## Slice order (original; A/B shipped, C open — see Status above)

### A. Error legibility (issue #6) — cheapest, immediate operator-legibility win
- **A1 — no-chat-template → `400 no_chat_template`** (issue #4). Model the
  `missingChatTemplate` condition as a client error (4xx) instead of falling through
  `AthenaError.classify()` → `500 module_load_failed`. Regression test asserts
  `status=400`, `code=no_chat_template`, `type=invalid_request_error`. **Cheapest; first.**
- **A2 — audio-upload faults → 4xx** (highest real-world impact). Malformed/corrupt →
  `400 invalid_audio`; over-length (~4h cap) → `413`/`400 audio_too_long`; unsupported
  codec → `400 audio_format_unsupported`. Model `AudioDecode` errors as `AthenaError`
  cases. Regression tests per condition.
- **A3 — catch-all sweep.** The 4× per-module rebind handlers (embed/transcribe/diarize/
  speaker) and queue-submit route their `500 internal_error`/`500 queue_error` through the
  classification seam so the real cause surfaces. Keep substrate detail in the server log,
  not the client body.

### B. Native-inference deprecation (non-breaking) — ADR 013 decision #1 (a)–(c)
- **B1 — deprecate `/api/embed`.** `deprecated: true` in `OpenAPISpec.swift` (0 callers).
- **B2 — migrate `athena run` → `/v1/chat/completions`.** One-function change in
  `clients/Sources/AthenaClient/Run.swift` (swap URL; read `choices[0].message.content`;
  keep non-stream). Re-prove the client builds on Linux. After this `/api/chat` has no
  in-repo caller.
- **B3 — deprecate `/api/chat`.** `deprecated: true`; keep serving through the sunset.

### C. Additive `/v1` OpenAI-compat — ADR 013 decisions #3, #4
- **C1 — `diarized_json` response_format** on `POST /v1/audio/transcriptions` (OpenAI's
  shape; the `#4a` item). Existing `diarize=true`/segment `speaker` stays. Real-model
  RUNBOOK validation.
- **C2 — `logprobs`/`top_logprobs`** honored on the greedy/structured path (already
  computes logits); stop 400-ing in `unsupportedParameter()`; emit the OpenAI `logprobs`
  response object.

### D. Native `/api` metering completion — ADR 007 #8 (added by the 2026-06-19 audit)
- **D1 — `/api/embed` per-principal metering.** `handleNativeChat` already meters
  (`generateMetered` + `meter()`, v0.10.140); the embed twin was missed. Compute the embed
  `TokenUsage` and call `meter(principal:usage:)` so `/api/embed` traffic reaches
  `usage_counters`. Regression test asserts the counter increments after an `/api/embed`
  call. (`/api/embed` is deprecated → low urgency, but a one-handler completeness fix and a
  prerequisite for any future token-budget quota, ADR 007 #9.)

### Deferred / out of this batch
- **Removal of `/api/chat` + `/api/embed`** (breaking) — gate cleared (the platform N/A;
  the consuming application self-migrates), but lands as its own ratified slice **after** the deprecation
  period gives the consuming application time to move. Operator picks the version bump.
- **`reasoning_effort` alias** — recorded (ADR 013 #7), built only when a consumer drives
  it; document the existing `enable_thinking` knob in the meantime.

## Test bar (every slice)

- Unit/regression test asserting the new `status`+`code` (error slices) or response field
  (additive slices).
- `deploy/e2e-rbac.sh` phase (stub engine, loopback, curl) where the path is stub-reachable;
  real-model RUNBOOK tier where a model is required (C1, and any chat-template repro).
- Drift-guard green across `deprecated:`/schema edits.
- Text/vision chat regression unchanged.

## Approval gate

This plan + ADR 013 (+ ADR 007 for slice D) are the approval artifacts. Slices A1/A2 and all
of B shipped (v0.10.x). **Track 2 execution resumes at A3-finish → D1 → C1 → C2** (see Status
above; no new ADR required — all four are already ratified). Breaking removal stays gated as
noted. Awaiting operator approval of this Track 2 ordering before implementation.
