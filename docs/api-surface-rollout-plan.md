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

## Slice order

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

This plan + ADR 013 are the approval artifacts. Execution begins at **A1** (cheapest,
lowest-risk). Breaking removal stays gated as noted.
