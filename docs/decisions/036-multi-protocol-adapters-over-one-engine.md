# 036 — Multiple protocol adapters over one inference engine (Anthropic Messages; one stream, two terminal ops)

**Status:** Accepted — **S1 (seam) SHIPPED v0.10.233; S2 Anthropic `/v1/messages` SHIPPED v0.10.234 (non-stream) + v0.10.235 (streaming)**. `e2e-anthropic-messages.sh` passes 5/5 on gemma-4-26b-a4b-it-8bit (non-stream + streaming, text + tool_use), reusing `prepareChat`/`collectMetered`/the engine with zero duplication; OpenAI regression-free (e2e-rbac 495/0, e2e-tool-choice-auto). Remaining: the `x-api-key` alias (S3). NOTE: a tool `description` field breaks forced-tool generation on this model for BOTH dialects (pre-existing engine/template quirk, not adapter-specific).
**Date:** 2026-06-30
**Milestone:** TBD (protocol-adapter surface)
**Amends:** ADR 013 (single inference *surface* → single inference *engine*, multiple protocols), ADR 031 (clarifies "no parallel *implementation*" — adapters comply), and the `CLAUDE.md` `/v1` compatibility tagging rule (adds a third notion: third-party-protocol-compatible).
**Relates:** ADR 011 (governor thesis — adapters never compose at inference), ADR 015 (cold-load surfacing), ADR 029 (inference execution gate), ADR 034/035 (tool-call + reasoning surfacing the adapters must preserve).

## Context

A bake-off used Claude Code as the harness against a local LLM. Claude Code speaks the
**Anthropic Messages API** (`POST /v1/messages`, `x-api-key` + `anthropic-version` headers);
Athena exposes only the OpenAI dialect (`/v1/chat/completions`), so the harness could not point
at Athena and an **Ollama shim had to be rolled onto the system** to bridge it. The driver is
concrete and recurring: a customer's harness may speak a protocol Athena doesn't, and today the
only fix is a third-party bridge process.

Two structural findings from mapping the serve path:

1. **The engine is already native; OpenAI is a translation layer at the edge.** The inference
   engine speaks Athena's own `ChatTurn` / `GenChunk` types (`AthenaLLM`); the OpenAI shape lives
   in `OpenAIDTO.swift` (`toolSpecs()`, `effectiveSchema()`, `stopSequences()`, …) plus
   `chatTurns(from:)`. There is no "OpenAI all the way down" — adding a second dialect is adding a
   second edge translation, not a second engine.

2. **"Streaming vs non-streaming chat" is not two generation paths — it is one producer with two
   hand-maintained consumers that keep drifting.** Both branches of `handleChatCompletions` consume
   the same `llm.generateMetered(...) → AsyncStream<GenChunk>`. The blocking branch drains it
   (`collectMetered`); the streaming branch forwards it (`streamSSE`). Because the two consumers
   re-implement orchestration independently, they have drifted **at least three times**, each a
   case where the streaming consumer lacked what the blocking one had:
   - **cancel counter** (A8/E3/E13) — a client disconnect kept decoding to `maxTokens` because the
     streamed path didn't bind a cancel counter;
   - **tool calls** (v0.10.230, `c971855`) — `tool_calls` worked blocking but not streaming until the
     fix added emission to the streaming consumer;
   - and the orchestration generally (admission, cold-load, metering, finish) is duplicated and must
     be kept in lockstep by hand.

**Conflict with prior decisions (surfaced, resolved here).** ADR 013 declared "`/v1` is the single
inference *surface*" and ADR 031 deleted `/api/chat` as a "parallel implementation." A naive
Anthropic handler that copied `handleChatCompletions`' ~600 lines of orchestration **would** violate
ADR 031 — it would be a second inference implementation. The reconciliation is that ADR 031's rule is
about duplicated **implementation**, not about a second **protocol**: an adapter that maps a dialect
into the *same* engine + the *same* orchestration is a translation layer, not a pipeline. ADR 013's
principle is therefore restated, not reversed: **one inference engine, surfaced in multiple
protocols.** Extracting the shared orchestration seam is what makes the Anthropic adapter *legal*
under ADR 031 — it is load-bearing, not gold-plating.

**Namespace reality.** OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) do not collide —
with each other or with Athena's control plane. Ollama is the only known dialect that collides: it
squats the generic `/api/*` prefix (`/api/chat`, `/api/generate`, `/api/tags`), which fights Athena's
control plane. To be a drop-in for Claude Code, the Anthropic route **must** be exactly `/v1/messages`
(the harness appends that path to its base URL), so there is no namespace *choice* for Anthropic — the
policy question only bites if/when a colliding dialect (Ollama) is ever added.

## Decision

### 1. One inference engine, multiple concurrent protocol adapters.

A **protocol adapter** is **decode + encode only** — it maps a dialect's request into the native
engine request and the engine's `GenChunk` stream into the dialect's response. It owns **no**
orchestration. All adapters are **compiled-in and live simultaneously**; a client selects a dialect by
hitting the matching route. This is not a plugin runtime (no dynamic loading — the protocol set is
small and ours) and not a swappable mode (see Rejected).

### 2. Extract the protocol-agnostic orchestration seam.

Pull the engine orchestration out of `handleChatCompletions` into a shared unit consuming a native
request and yielding the `GenChunk` stream: admission (`governedLLM`), cold-load deferral (ADR 015),
vision gate, `preflightPromptCache`, cancel/metering, principal scoping. The OpenAI handler becomes
*decode-OpenAI → seam → encode-OpenAI*. A native request value type (`NativeChatRequest`) is born here,
now legitimately shaped by **two** real dialects (OpenAI + Anthropic) rather than guessed from one.

### 3. One stream, two terminal ops (collapse the streaming/blocking drift).

The seam exposes a single `AsyncStream<GenChunk>`. Two **generic** terminal operations consume it,
parameterized by a per-protocol encoder:

- **`forward`** — incremental (SSE / Anthropic event stream);
- **`drain`** — accumulate to completion, encode once (blocking).

Orchestration that previously lived in each consumer (metering, cancel, finish, **tool-call
detection**) lives **once**. Tool-call *detection* (the `GenChunk.toolCall` event **and** the
Guide-forced text-is-a-tool-call reparse) happens in the shared path; only the *surfacing* differs by
transport (blocking = one complete `tool_calls` array; streaming = `delta.tool_calls` deltas). A future
drift cannot silently drop a feature from one mode because there is no second copy to forget.

**Transport-intrinsic differences are kept, not collapsed:** cold-load keep-alives (`: loading`) exist
only on the streaming surface (ADR 015), and a mid-generation `GenChunk.error` becomes an HTTP error
status when blocking but an in-band event when already-streaming. The seam unifies the *logic*; the
encoder decides *surfacing*. Blocking and streaming should stop **duplicating** orchestration — they
should not become byte-identical.

### 4. Anthropic Messages adapter (`POST /v1/messages`), scoped to what Claude Code drives.

First cut: text content, `system`, `tool_use`/`tool_result`, streaming events
(`message_start` → `content_block_{start,delta,stop}` → `message_delta` → `message_stop`, `ping`),
`stop_reason` + `usage.{input,output}_tokens`. **400** on the rest (image/document content blocks —
the vision path exists and can map them when a consumer needs it; prompt-caching `cache_control`;
extended-thinking blocks). Accept **`x-api-key`** as a bearer-token alias on the Anthropic routes (and
tolerate `anthropic-version`); `Authorization: Bearer` still works, so authed Anthropic harnesses are
real drop-ins.

### 5. Namespace policy: idiomatic path where free, mount-prefix for colliders.

A dialect mounts at its idiomatic path when that path is free (Anthropic `/v1/messages`). A dialect
whose paths collide with Athena's surface (Ollama `/api/*`) mounts under an explicit **prefix**
(e.g. `/ollama/...`), and the client points its base URL there. Every major harness takes a
configurable base URL (`ANTHROPIC_BASE_URL`, `OLLAMA_HOST`, OpenAI `base_url`), so a static prefix
always works and **all dialects + the control plane stay live at once**. No colliding dialect exists
today, so no prefix machinery is built now.

### 6. Compatibility tagging: a third notion.

The `CLAUDE.md` `/v1` rule's `[oai]` / `[native]` tags gain a third: **third-party-protocol-compatible**
(tag `[anthropic]`, generalizable to `[compat:<vendor>]`). `/v1/messages` is Anthropic-compatible — it
is neither an OpenAI drop-in nor an Athena-native extension. The Stable `/v1/*` list adds
`POST /v1/messages [anthropic]`.

### 7. Where native capabilities and extensions live (the three-rung rule).

Athena has features no vendor dialect models ("weird things"). Placement is decided by *what kind*
of thing it is, not by which dialect a caller happens to use:

1. **A knob on an existing inference call** (e.g. `enable_thinking`, `speculative`,
   `reasoning_effort`) → a **tolerated extension field** on the dialect request, ignored by vanilla
   clients. No new surface.
2. **A whole capability with no vendor shape** (diarization, speaker embeddings, video) → a
   **`[native]` endpoint on `/v1`** (the inference plane; the tag marks it non-portable). This is the
   existing pattern — it is not "re-introduced," it never left. (ADR 025 removed `/v1/{vectors,store,
   queue}` as unused *data-persistence* tenants, not as a ban on native inference endpoints.)
3. **Daemon control/management** → `/api/*`.

Multi-protocol changes rung 1's *home*: a native knob is a field on **`NativeChatRequest`** (the
engine boundary), and **each adapter maps its own dialect's extension syntax onto it** — so the
capability is defined once at the engine and each dialect decides how (or whether) to surface it
(an OpenAI client sets `enable_thinking` via `chat_template_kwargs`; an Anthropic client sets it
Anthropic-idiomatically; both land on the same `NativeChatRequest` field). A capability that fits no
dialect's request shape becomes its own rung-2 endpoint.

**No framework.** This rule places each weird thing *when it actually arrives* — there is no
speculative native namespace, no extensions registry, no generic knob plumbing built ahead of a named
feature (YAGNI). The plane axis (Decision: `/v1` inference vs `/api` control) and the three rungs are
the whole policy.

## Rejected alternatives

- **Swappable API-layer plugin (the original instinct).** A daemon mode flipped between dialects is
  strictly worse than concurrent adapters: (a) it **disables the control plane** whenever a dialect
  (Ollama) squats `/api/*`; (b) it serves **one protocol at a time**, so two harnesses on different
  dialects can't share a daemon — wrong for a multi-tenant oracle; (c) it adds mutable global state +
  a reload to flip. The collision it's meant to solve is solved for free by a static mount prefix,
  with all surfaces live. No harness *requires* swapping, because all take a configurable base URL.
- **Dynamic plugin runtime.** The protocol set (OpenAI, Anthropic, maybe Ollama) is small and owned;
  dynamic loading / third-party registration is ceremony for a fixed, in-tree set. Adapters are
  compiled-in `decode`/`encode` pairs.
- **Off-the-shelf OpenAI↔Anthropic proxy (LiteLLM et al.).** An extra process and dependency in front
  of a single-binary passive oracle — breaks the deployment story and the passive-oracle posture. It
  is the fastest *interim* unblock (minutes) if a bake-off needs Anthropic before this lands, but not
  the durable home.
- **Naive duplicated Anthropic handler (copy `handleChatCompletions`).** Re-introduces the exact
  parallel-implementation defect ADR 031 removed, and a fourth+ copy of the orchestration that has
  already drifted three times. Rejected in favor of the shared seam (Decision 2/3).
- **Native HTTP inference surface (revive `/api/chat`-style native dialect).** ADR 031 deleted this; a
  *native* protocol has no consumer (every harness speaks a vendor dialect). The native shape lives as
  the internal `NativeChatRequest`/`GenChunk` engine boundary, not as an HTTP route.

## Consequences

- **Anthropic harnesses (Claude Code) point straight at Athena**, deleting the Ollama shim. The next
  dialect is one `decode`/`encode` pair over the proven seam.
- **The streaming/blocking drift class is closed.** Cancel/metering/finish/tool-call **detection** live
  once; the v0.10.230 (streaming tool_calls) and A8/E3/E13 (streaming cancel) bug shapes become
  structurally impossible. Anthropic cannot reintroduce "tools work blocking but not streaming."
- **Slice 1 touches `handleChatCompletions`** — the most load-bearing endpoint. Pinned by a
  **byte-identical** DoD: OpenAI streaming + non-streaming responses unchanged before/after, existing
  `deploy/e2e-rbac.sh`, the chat e2e, **and `deploy/e2e-tool-choice-auto.sh` (ADR 034, both modes)**
  stay green.
- **ADR 013 restated, ADR 031 clarified, `CLAUDE.md` amended** in the same edit window as the enacting
  code: single inference *engine* (not surface); "no parallel *implementation*" (adapters comply);
  `/v1` tag set gains `[anthropic]` and the Stable list gains `/v1/messages`.
- **OpenAPI SSOT + drift-guard:** `/v1/messages` is added to `OpenAPISpec.swift` in the same edit; the
  spec↔routes drift-guard stays green.
- **Honesty boundary:** Anthropic coverage is the Claude-Code-driven subset; unsupported content blocks
  / features return a cause-naming Anthropic-shaped 4xx, never a silent drop. Tool-call detection is
  substrate-arch-gated exactly as ADR 034 (gemma4 wired); an arch with no inferred `toolCallFormat`
  under-detects on the auto path regardless of dialect.

## Implementation slices (each its own commit + tag, appVersion bump in the slice)

- **S1 — Seam + drain/forward collapse (OpenAI, byte-unchanged).** Extract the orchestration seam;
  express consumption as `stream → {drain | forward}(encoder)`; route OpenAI through it; delete the
  `collectMetered`/`streamSSE` duplication. DoD: OpenAI stream + non-stream byte-identical;
  `e2e-rbac.sh` + chat e2e + `e2e-tool-choice-auto.sh` green. No new route.
- **S2 — Anthropic encoder + `POST /v1/messages`.** `AnthropicDTO.swift` decode → `NativeChatRequest`;
  one `AnthropicEncoder` used by both `drain` and `forward` (non-stream + streaming together, since
  `forward` is shared). Route in `OpenAPISpec.swift`, tagged `[anthropic]`. DoD:
  `deploy/e2e-anthropic-messages.sh` — non-stream + streaming valid, tool round-trip.
- **S3 — `x-api-key` auth alias (authed mode).** Anthropic routes accept `x-api-key` as a bearer alias;
  tolerate `anthropic-version`. DoD: **real Claude Code pointed at `http://127.0.0.1:7447` completes a
  tool-using turn end-to-end** (fails today, no `/v1/messages`; passes after); authed-mode bad-key →
  Anthropic-shaped 401.
- **Deferred:** image/document content blocks, prompt-caching, extended-thinking, Ollama + the
  mount-prefix machinery (no colliding dialect yet).

Decode/encode/auth-alias decision logic is MLX-free and unit-pinned (ADR 008/009). Plan:
`docs/anthropic-adapter-plan.md` (on approval).
