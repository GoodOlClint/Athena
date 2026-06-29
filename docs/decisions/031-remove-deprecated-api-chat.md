# 031 — Remove the deprecated `/api/chat` native inference surface

**Status:** Accepted — **IMPLEMENTED**. Removed the `/api/chat` route +
`handleNativeChat` + `textChunks` + the `AthenaChatRequest`/`Response`/`Chunk`/
`Message` + `AthenaUsage` DTOs + the `OpenAPISpec.swift` path & schemas (spec now
36 paths, validated) + the e2e-rbac.sh assertions (now asserts `/api/chat` → 404).
`/api/embed` retained on the shared `governedEmbed` path. External-consumer gate:
operator confirmed via session approval (ADR 013 indicated none; `athena run` is
already on `/v1`).
**Date:** 2026-06-26
**Milestone:** TBD (surface cleanup)
**Executes:** ADR 013 (the `/api/chat` + `/api/embed` inference deprecation).

## Context

ADR 013 established **`/v1` is the single inference surface; `/api/*` is the
control plane**, and deprecated the native inference routes `/api/chat` and
`/api/embed`. Removal was declared **breaking** and gated on external-consumer
confirmation ("verified: `/api/embed` 0 callers, `/api/chat` only `athena run` →
migrate to `/v1`"; "the consuming application self-migrates; the platform N/A").

The 2026-06-26 code review (over-engineering lens) confirmed the current state:

- `/api/embed` is **already de-duplicated** — it shares `governedEmbed` with the
  `/v1` path. Nothing to delete there beyond the route alias if desired.
- `/api/chat` is a **full parallel inference implementation**: the route
  registration + `handleNativeChat` (streaming NDJSON + non-streaming +
  metering, ~89 LOC) + the `textChunks` helper (~13 LOC) + the DTOs
  `AthenaChatRequest` / `AthenaChatResponse` / `AthenaChatChunk` in
  `NativeAPIDTO.swift`. It duplicates `/v1/chat/completions` — a **defect** under
  the CLAUDE.md "no parallel implementation of a canonical pipeline" rule.
- **Zero live callers** of `/api/chat` in `Sources/` or `clients/` (`athena run`
  already targets `/v1`). The only references are ADR text and stale
  `clients/.build/` artifacts.

## Decision

Remove the deprecated native chat inference surface:

- Delete the `/api/chat` route registration, `handleNativeChat`, `textChunks`,
  and the `AthenaChatRequest` / `AthenaChatResponse` / `AthenaChatChunk` DTOs.
- Remove the `/api/chat` path from `OpenAPISpec.swift` (the SSOT) **in the same
  edit**, and update the e2e drift-guard expectations.
- `/v1/chat/completions` remains the sole chat surface. `/api/*` stays
  control-plane only (model-store, RBAC, lifecycle, audit, usage, logs, cache) —
  reinforcing the ADR 013/025/026 thesis.
- `/api/embed`: retain or remove the route alias per the same confirmation; no
  inference code to delete (shared path). **Recommend** removing the alias too
  for surface consistency, but it is lower-priority (not a duplicate impl).

## Consequences

- **Breaking** for any external HTTP consumer of `/api/chat`. This is the gate:
  the change does not land until the operator confirms no external consumer
  depends on it (ADR 013 indicated none; this ADR records the explicit
  go/no-go).
- Deletes ~120 LOC of duplicate inference + 3 DTOs; removes a second metering /
  streaming path that must otherwise be kept in lockstep with
  `/v1/chat/completions`.
- One fewer surface to document and drift-guard.

## Open questions (resolve in review)

- **External-consumer confirmation (the gate):** confirm no consumer calls
  `/api/chat`. If any does, they migrate to `/v1/chat/completions` first.
- **`/api/embed` alias:** remove for consistency, or keep as a thin shared-path
  alias? (No correctness impact either way.)
- **Deprecation window:** straight removal vs a release of `410 Gone` /
  `Deprecation` header first. Given zero known callers, lean straight removal.
