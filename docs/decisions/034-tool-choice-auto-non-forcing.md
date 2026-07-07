# 034 — `tool_choice: auto` must not force a tool call

**Status:** Accepted — **IMPLEMENTED** v0.10.231 (gate-approved 2026-06-30).
Un-forcing + tool-history fidelity shipped together; no revert knob; trust the
substrate `.gemma4` detection. Local e2e (`deploy/e2e-tool-choice-auto.sh` on
Qwen3.5-27B-4bit) PASSES checks 1/2/4 (auto answers in text, multi-turn no
re-call, plain chat untouched) + 3 (required still forces). Gemma-4 free
tool-DETECTION (a model freely calling under auto via the substrate `.toolCall`
event) is the studio DoD — run the e2e against `gemma-4-26b-a4b-it-8bit` there.
**Date:** 2026-06-30
**Milestone:** TBD (tool-calling correctness)
**Relates to:** ADR 013 (`/v1` is the single inference surface; OpenAI
compatibility rule), ADR 010 (vision-input chat reused the substrate path
rather than forking), ADR 018/020 (multi-backend = compose, don't fork).
**Motivated by:** a live tool-call spiral on the studio appliance
(`gemma-4-26b-a4b-it-8bit`), 2026-06-30 — see
`docs/tool-call-streaming-handoff.md` and `docs/tool-choice-auto-plan.md`.

## Context

`POST /v1/chat/completions` is OpenAI-compatible **[oai]**. OpenAI's
`tool_choice` defaults to `"auto"` when `tools` are present: the model decides
each turn whether to emit a tool call **or** a normal text answer. `"required"`
forces *some* tool call; `{type:function,function:{name}}` forces a specific
one; `"none"` forbids tool calls.

Athena collapses **`auto`/absent/`required`** into one behavior. In
[`OpenAIDTO.selectedTools()`](../../Sources/athena/Server/OpenAIDTO.swift) the
guard is: `"none"` ⇒ no tools; a named function ⇒ that one; **everything else ⇒
all declared tools**. `effectiveSchema()` then returns `isToolCall: true` and a
tool-call JSON schema, which engages the structured-output **Guide** (guided
greedy, `MLXLLMModule.runSpeculative`). The Guide masks every decode position to
the tool-call grammar, so the model is **incapable of emitting free text** — it
*must* produce a tool-call object on every turn.

Consequence (the field incident): the client (a passive consumer) sends a tool
result back as the next turn and re-requests; Athena, still forcing, emits
*another* identical tool call; the client runs the tool again; repeat forever.
The model never "realizes it can answer" because it is structurally forbidden
from answering. v0.10.230 (streaming `tool_calls` deltas) did not cause this —
it merely let the client *recognize* the tool call and start the round-trip,
exposing the latent forcing.

Two further facts established during triage:

1. **The substrate already supports free-generation tool detection for Gemma 4.**
   `container.generate(input:parameters:)` builds a `TextToolTokenLoopHandler`
   with `format: configuration.toolCallFormat ?? .json`; the VLM load path sets
   `toolCallFormat = ToolCallFormat.infer(modelType)` → `.gemma4` for
   `gemma4*`. The stream emits `Generation.toolCall(ToolCall)` when the model
   freely calls a tool. **Athena drops it** — the consumer in
   `MLXLLMModule.generateMetered`'s substrate branch has `default: break`.

2. **Tool-turn history is lossy.** `ChatTurn` and `MLXLLMModule.chatMessages`
   carry only `role` + `content` (+ images). An assistant tool-call turn arrives
   with empty content and its `tool_calls` dropped; a `tool` result keeps its
   content but loses `tool_call_id`. The substrate `Chat.Message` likewise has
   only role/content/images. So even un-forced, the model sees an empty
   assistant turn followed by an orphaned result and may re-search.

## Decision

Make `tool_choice: auto` (and absent) behave per OpenAI: the model decides.

1. **`auto`/absent ⇒ do not force the Guide.** `selectedTools()` returns the
   tools **only** for `required` and a named function (forcing cases). For
   `auto`/absent, return nil from the *schema-forcing* selector so
   `effectiveSchema()` yields no tool-call schema — but the tool **menu**
   (`toolSpecs()`) is still passed to the chat template so the model knows the
   tools exist and how to call them. `required`/named keep forcing the Guide
   exactly as today; `none` unchanged.

2. **Route un-forced tool requests through the substrate free-generation path**
   (already the no-schema path) and **wire the dropped `.toolCall` event.** Add
   a `GenChunk.toolCall(name:argumentsJSON:)` case; map the substrate
   `Generation.toolCall(ToolCall)` to it (serialize `function.arguments`
   `[String:JSONValue]` with sorted keys, matching `parseToolCall`'s output).
   **Trust the substrate's `.gemma4` detection only** — no free-text JSON
   fallback (the format is confirmed set; a fallback adds ambiguity about when
   free text "is" a tool call).

3. **Serialize the `.toolCall` GenChunk as OpenAI `tool_calls` in both paths.**
   Non-streaming: `GenCollected` gains a captured tool call; the response
   surfaces `tool_calls` + `finish_reason:"tool_calls"`. Streaming: `pumpTokens`
   emits the `delta.tool_calls` chunk + terminal `finish_reason:"tool_calls"`
   on a `.toolCall` event — reusing the v0.10.230 delta shape. The existing
   `isToolCall` (Guide-forced) path is unchanged; the new `.toolCall` event is
   an additional trigger for the same serialization.

4. **Carry tool-turn history faithfully.** Extend `ChatTurn` (and the
   DTO→ChatTurn mapping + `chatMessages`) so an assistant turn carries its
   `tool_calls` and a `tool` turn carries its `tool_call_id`, mapped onto the
   substrate `Chat.Message` tool fields so the Gemma 4 template renders a
   coherent call→result history.

**No revert knob.** Per operator decision — clean OpenAI semantics, no config
surface for the old always-force behavior. (The `required` path remains for
callers that genuinely want forced tool use.)

## Rejected alternatives

- **Relax forcing only when the last turn is a tool result.** Hacky, not OpenAI
  semantics; turn-1 "how are you" would still be wrongly forced into a search.
- **Free-text JSON fallback alongside substrate detection.** More code; creates
  ambiguity (is a JSON-looking answer a tool call?). Substrate `.gemma4`
  detection is confirmed wired — rely on it.
- **A revert knob.** Considered (house default-on-with-knob pattern); operator
  chose clean semantics. The `required` value already serves anyone wanting
  forced calls.

## Consequences

- `/v1/chat/completions` becomes genuinely multi-turn tool-capable and matches
  OpenAI `tool_choice` semantics — the **[oai]** label stays honest.
- Tool calls now arrive on **two** internal triggers (Guide-forced for
  `required`/named; substrate `.toolCall` for `auto`), both funneling to one
  serialization. The forced path is byte-unchanged.
- Tool detection for `auto` depends on the substrate's per-arch
  `toolCallFormat`. Gemma 4 is wired (`.gemma4`); an arch with no inferred
  format falls back to `.json` and may under-detect. Honesty boundary: `auto`
  tool detection is **substrate-arch-gated**, not universal — documented, not
  promised for every model.
- `ChatTurn` gains fields; the change touches the DTO→module seam but adds no
  new pipeline (composes the existing substrate path, per ADR 010/018/020).
- Decision logic (the `tool_choice` → force/no-force/menu resolution and the
  `ToolCall` → OpenAI mapping) is MLX-free and unit-pinned (ADR 008/009).

## Open questions (resolve in/after review)

- **Consumer impact:** any consumer relying on `auto`-always-calls? (ADR 013
  indicated the downstream client self-migrates; confirm none depends on the old forcing.)
- **Multi-tool-call turns:** OpenAI can emit several `tool_calls` in one turn.
  Does the substrate `.gemma4` handler emit multiple `.toolCall` events per
  turn? If so, the serializers must aggregate (array index 0,1,…). Scope check
  during S2.
- **`finish_reason` when un-forced tool + text interleave:** confirm the
  substrate emits `.toolCall` cleanly without leading `.chunk` text for Gemma 4
  (if it prepends prose, decide whether that prose is dropped or surfaced).
