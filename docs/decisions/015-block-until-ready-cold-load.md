# 015 — Block-until-ready for on-disk cold-loads; 503 only on timeout, download, or failure

**Status:** Accepted — decision; staged implementation (operator-assigned milestone/tags)
**Date:** 2026-06-17
**Supersedes/refines:** the M43.2 "never block the request thread on cold-load" rationale —
narrows it to the *download* case, which was its real target.
**Relates:** ADR 011 (memory governor), ADR 013 (`/v1` inference surface), M62 (cold-load
model selection), M33 (preload-on-start), ADR 009 (stub-tier CI for MLX-free decision logic).

## Context

A request for a model that isn't resident returns **`503 module_loading` + `Retry-After: 5`**
immediately; the load runs detached and the client must retry-poll. Path:
`governedLLM` → `selectColdLoadModel` → `governor.beginLoadIfNeeded` → `.loading` ⇒
`coldLoadResponse` ([AthenaServer.swift:3544-3577](../../Sources/athena/Server/AthenaServer.swift#L3544),
[:5288](../../Sources/athena/Server/AthenaServer.swift#L5288)). Most peer runners
(Ollama, LM Studio, llama.cpp server) instead **block the first request until the model is
loaded, then return the result** — better drop-in ergonomics for arbitrary OpenAI clients.

The 503 was a deliberate M43.2 choice, comment verbatim:

> `// M43.2: never block the request thread on a multi-GB cold-load. .loading ⇒`
> `// 503+Retry-After so the client paces its retries instead of hitting its own timeout.`

**The fact that reopens it:** M43.2's fear is a *multi-GB HF download*. But the **request
path never downloads.** `MLXLLMModule` "loads a model from a local directory — no download,
no HF hub round-trip" ([MLXLLMModule.swift:51-52](../../Sources/AthenaLLM/MLXLLMModule.swift#L51),
:91); `selectColdLoadModel` validates against the local store and returns **400
`model_not_available`** if weights aren't present. The only download path is the governor's
`pulling` flag — the operator-initiated background pull ([MemoryGovernor.swift:167](../../Sources/AthenaCore/MemoryGovernor.swift#L167)).
So a request-path cold-load is a **bounded local-disk load** (seconds to a minute or two),
which is exactly what the peer runners block on. M43.2's rationale applies to *downloads*,
not to local loads — so blocking the local load does not re-create the problem it fixed.

Async clarification: "block the request" here means an `await` on the load Task, not a thread
block. The governor is an `actor`; a suspended request coroutine does not stop it from
processing concurrent eviction/reconciliation. The two real costs are (a) a waiting request
**holds its concurrency slot** for the load's duration, and (b) a silent multi-minute wait
with no bytes flushed can trip a **reverse-proxy idle timeout**.

The existing blocking primitive `governor.ensureLoaded` (with in-flight coalescing so
concurrent callers share one load) already serves preload, the queue worker, and
`/api/models/load`. This decision routes the request path to a **bounded** form of it.

## Decision

**On the request path, wait for a non-resident-but-on-disk model to load, then serve — up to
a bounded budget; fall back to today's 503 only on timeout, download-in-progress, or load
failure.** This is the new default behavior, configurable off.

1. **Block-until-ready is the default.** A request for an on-disk model that is `unloaded`/
   `loading` awaits the (single-flight) load and then proceeds to inference, returning `200`.
   No client change; matches peer runners for every OpenAI-compatible client.

2. **A separate `cold_load_wait_secs` budget bounds the wait (default 120s).** It is distinct
   from the inference `request_timeout_secs` / per-request `timeout`, which applies *after*
   the load, to generation only. If the load exceeds `cold_load_wait_secs`, fall back to
   today's **`503 module_loading` + `Retry-After`**. Setting `cold_load_wait_secs = 0`
   restores the legacy immediate-503 behavior (escape hatch / revert switch).

3. **Downloads still 503.** An in-progress operator pull (`pulling`) is genuinely
   minutes/GB; it returns `503 module_loading` immediately and is **not** waited on — blocking
   on it is exactly what M43.2 forbids. (The request path itself never *initiates* a download;
   a missing model is still `400 model_not_available`.)

4. **Streaming requests emit SSE keep-alives during the load.** For a streamed chat hitting a
   local cold-load, the daemon starts the SSE response and emits `: loading` comment frames on
   a timer until the model is ready, then streams tokens — so reverse proxies don't idle-time-
   out and the client sees liveness. Non-streaming requests (and embeddings/audio) block
   silently up to the budget.

5. **Real load faults surface immediately, never as a wait.** A failed load
   (`lastLoadError`), an over-budget admission (`memory_budget_exceeded`), or a missing model
   (`model_not_available`) returns its real status at once — the bounded wait only covers a
   genuinely in-progress local load.

## Consequences

- **Streamed model-dependent validation moves in-band (deliberate, OpenAI-consistent).**
  Emitting `: loading` bytes commits the streamed response to `200`, so the two checks that
  need a *loaded* model — `servesVision` (image → text-only model) and `preflightPromptCache`
  (over-cap prompt) — can no longer be a pre-stream HTTP 4xx on the **streaming** path. They
  run after load inside the producer and surface as an in-stream OpenAI-style error event +
  `[DONE]`. This mirrors OpenAI's own streaming (post-start errors are stream events, not
  status codes). **Model-independent** checks (`unsupportedParameter`, image-part *decode*,
  `structuredRequestError`) stay pre-stream as clean 4xx. The **non-streaming** path is
  unchanged: it still waits, then runs all checks as normal HTTP status codes.
- **A waiting request holds its concurrency slot** for the load's duration. A thundering herd
  on one cold model coalesces to a single load, but each waiter holds a slot; bursts past the
  global cap get `429` backpressure (honest, with `Retry-After`). For v1 the load-wait stays
  inside the existing concurrency gate; moving it outside the gate is a noted follow-up if it
  bites.
- **Reverse-proxy guidance changes.** Streaming is covered by heartbeats; **non-streaming**
  cold-loads need `proxy_read_timeout` (nginx) / equivalent ≥ `cold_load_wait_secs`. Update
  `docs/reverse-proxy.md`.
- **`503 module_loading` becomes rare**, not gone — it now means "timeout / download in
  progress / unavailable," not "first request." The `OpenAPISpec.swift` response descriptions
  for inference endpoints are updated to say so (no route/schema change; drift-guard stays
  green).
- **Governor thesis intact (ADR 011).** Blocking is `await`, not a thread/actor block;
  concurrent eviction and reconciliation still run. Passive-oracle intact (inbound only).
- **Testable under `swift test` (ADR 009).** The bounded-wait decision algebra lives in the
  MLX-free `AthenaCore` governor (loaded/loading/pulling/failed/timeout → wait/serve/503/error)
  and is unit-pinned; end-to-end load timing is a real-model RUNBOOK item.
- **Revert is one config line** (`cold_load_wait_secs = 0`), de-risking the default flip.

## Plan

See [`docs/cold-load-blocking-plan.md`](../cold-load-blocking-plan.md) — staged, test-pinned
slices, appVersion bumped in each slice commit.
