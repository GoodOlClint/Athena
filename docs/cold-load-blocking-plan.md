# Block-until-ready cold-load — change plan

**Status:** Implemented — slices 1–5 landed in the working tree; logic-tier
tests green (477/0, incl. drift-guard). Remaining: real-model RUNBOOK
(end-to-end load timing + SSE keep-alives) on a Metal host, `appVersion` bump,
and the per-slice commits + annotated tags (operator's release step).
**Date:** 2026-06-17
**Decision of record:** ADR 015 — block-until-ready for on-disk cold-loads; 503 only on
timeout, download, or failure.
**Milestone:** operator-assigned (numbers + tags). appVersion bumped *in* each slice commit.

This is the approval gate. Goal: a request for a non-resident-but-on-disk model **waits for
the local load (bounded), then serves `200`** — matching Ollama/LM Studio — instead of an
immediate `503 module_loading`. Downloads, timeouts, and load failures keep returning 503/the
real error.

## Current behavior (verified)

`governedLLM` → `selectColdLoadModel` (bind requested model; 400 if not local) →
`governor.beginLoadIfNeeded(id)` → on `.loading`, return `coldLoadResponse`
(`503 module_loading`, `Retry-After: 5`) and run the load detached. Same shape for embeddings
and audio. Blocking primitive `governor.ensureLoaded` (in-flight coalescing) already exists,
used by preload / queue worker / `/api/models/load`.
([AthenaServer.swift:3544](../Sources/athena/Server/AthenaServer.swift#L3544),
[MemoryGovernor.swift](../Sources/AthenaCore/MemoryGovernor.swift))

## Target behavior

| State at request time | Today | After |
|---|---|---|
| Loaded | serve `200` | serve `200` (unchanged) |
| On-disk, unloaded/loading | `503 module_loading` now | **wait ≤ `cold_load_wait_secs` → serve `200`**; on timeout → `503` |
| Operator pull in progress (`pulling`) | `503 module_loading` | `503 module_loading` (unchanged) |
| Not on disk | `400 model_not_available` | `400 model_not_available` (unchanged) |
| Load failed / over budget | real `4xx`/`503` (M62/NE2) | real `4xx`/`503` immediately (unchanged) |

Wait budget is **separate** from the inference deadline: `cold_load_wait_secs` covers the
load; `request_timeout_secs` / per-request `timeout` covers generation, applied after load.
`cold_load_wait_secs = 0` ⇒ legacy immediate-503 (revert switch).

## Config

- New key `cold_load_wait_secs` (default **120**), TOML `[server]` (per the TOML-not-JSON
  house default) + CLI flag mirror where the sibling timeouts live. Doctor surfaces it.
- Validation: non-negative int; 0 = disabled. Documented in the config reference + quickstart.

## Slices (each: commit + annotated tag → main; appVersion bump in the slice commit; test-pinned)

### Slice 1 — Governor bounded-wait primitive (AthenaCore, MLX-free, unit-tested)
Add `governor.awaitLoad(id, within: Duration) -> LoadOutcome` where `LoadOutcome ∈
{ .loaded, .stillLoading (timeout), .pulling, .failed(Error) }`:
- `.loaded` → return immediately.
- `pulling.contains(id)` → `.pulling` immediately (no wait — downloads stay 503).
- cached `lastLoadError` and admission (`makeRoom`) failure → `.failed` immediately.
- else ensure single-flight load started (reuse `beginLoadIfNeeded`'s spawn / `inFlight[id]`),
  then `await` the in-flight Task with a timeout race → `.loaded` / `.failed` / `.stillLoading`.
- Honor task cancellation (client disconnect): stop waiting; leave the detached load to
  complete so the next request finds it resident.
**Test (ADR 009, `swift test`):** the decision algebra — loaded/pulling/failed/timeout/cancel
→ correct `LoadOutcome` — with a fake module whose load is a controllable continuation. No MLX.

### Slice 2 — Non-streaming gate routes through the wait
`governedLLM` (and the embeddings + audio gates) call `awaitLoad(id, within:
coldLoadWaitSecs)` instead of `beginLoadIfNeeded`:
- `.loaded` → proceed (then the existing rebind/servesVision/preflight checks run as today,
  clean HTTP status).
- `.pulling` / `.stillLoading` → `503 module_loading` + `Retry-After` (existing
  `coldLoadResponse`).
- `.failed` → existing classification (real 4xx/503).
- `coldLoadWaitSecs == 0` → behaves exactly like today (call sites unchanged in effect).
**Test:** stub-tier pins the gate's outcome→response mapping; real-model RUNBOOK: a curl to a
cold model returns `200` after the load (no 503), and a model whose load exceeds a tiny
`cold_load_wait_secs` returns `503`.

### Slice 3 — Streaming heartbeat during load (the invasive one)
For `stream:true` chat, when `awaitLoad` would wait (state is local unloaded/loading):
- Keep **model-independent** validations pre-stream (clean 4xx): body decode,
  `unsupportedParameter`, image-part *decode* (`chatTurns`), `structuredRequestError`.
- Probe load state pre-stream: `.loaded` → current path (no heartbeat); `.pulling`/`.failed`
  → clean `503`/error before any bytes.
- Otherwise start the SSE `200` response and run a producer that: emits `: loading` comment
  frames every ~10s while `await`ing the load; on `.loaded` runs the **model-dependent**
  checks (`servesVision`, `preflightPromptCache`) — failures become an in-stream OpenAI-style
  `error` event + `[DONE]` — then begins `generateMetered` token frames; on `.stillLoading`
  emits a terminal error event advising retry; honors consumer-cancel (reuse
  `onConsumerCancel`/`HeartbeatCounter`).
**Test:** stub-tier pins the producer state machine (loading→heartbeat, loaded→validate→tokens,
timeout→error-event); real-model RUNBOOK: SSE shows `: loading` keep-alives then tokens; image
→ text-only model yields an in-stream error event (not a dropped connection).

### Slice 4 — Docs + spec descriptions
- `OpenAPISpec.swift`: update the inference endpoints' 503 response *description* (now
  "timeout / download in progress / unavailable," not "first request"). No path/schema change;
  drift-guard stays green.
- `docs/reverse-proxy.md`: non-streaming cold-loads need `proxy_read_timeout ≥
  cold_load_wait_secs`; streaming is covered by heartbeats.
- Config reference + quickstart: document `cold_load_wait_secs` (incl. `0` = legacy 503).

### Slice 5 — ADR cross-link + changelog
Land ADR 015 reference in `CLAUDE.md` ADR list; note the behavior change in release notes.

## Open design points (carried into review)

1. **Concurrency slot during wait.** v1 keeps the load-wait inside the existing concurrency
   gate (a burst on a cold model coalesces to one load but each waiter holds a slot; overflow
   → `429`). Moving the wait *outside* the inference cap (so loads don't consume inference
   slots) is a follow-up if bursts bite. Confirm v1 is acceptable.
2. **Per-request override.** Decision was a global default (not per-request). A future
   `X-Athena-Cold-Load-Wait` header / body field can layer on without breaking callers — out
   of scope unless you want it now.
3. **DFlash drafter download — RESOLVED, no action.** The drafter `.download(`
   ([MLXLLMModule.swift:538](../Sources/AthenaLLM/MLXLLMModule.swift#L538)) is reached only
   via `ensureDFlashDraft`, whose sole call site is inside `generateMetered`
   ([:943](../Sources/AthenaLLM/MLXLLMModule.swift#L943)) — the **generation** path, never
   `load()`/`loadModel()`. So it is outside the `cold_load_wait_secs` budget; the request-path
   load (`loadModel`) is purely local-disk. The drafter fetch is lazy/first-decode and already
   bounded by the inference deadline (pre-existing behavior, untouched by this change). It is
   also default-off and VLM-skipped.
4. **Default `cold_load_wait_secs = 120`** — sane for big local models on this hardware?
   Adjust to taste; `0` always reverts.

## Out of scope

No route/schema additions, no change to the operator-pull/download path, no governor eviction
policy change, no per-modality wait tuning (one global budget for v1).
