# Plan — `tool_choice: auto` non-forcing (ADR 034)

**Goal:** `POST /v1/chat/completions` with `tools` and `tool_choice` absent/`auto`
lets the model answer with text OR call a tool, each turn — killing the tool
spiral. `required`/named keep forcing; `none` unchanged. Both halves (un-forcing
+ history fidelity) ship together (operator decision).

**Definition of Done (discriminating, end-to-end):** against a resident
`gemma-4-26b-a4b-it-8bit`, the two-turn exchange

1. user: "Hello there, how are you?" + one tool `searchWeb` (no `tool_choice`)
2. (if the model calls) tool result appended → re-request

terminates in a **normal text answer** with no repeated identical tool call.
Fails before the change (infinite forced `searchWeb`), passes after. Captured as
an e2e script (`deploy/e2e-tool-choice-auto.sh`, run against the studio since it
needs a live model — MLX numerics can't run under `swift test`, ADR 009).

## Slices (small, test-pinned commits; each bumps appVersion)

### S1 — `tool_choice` resolution: stop forcing on `auto`
- `OpenAIDTO.swift`: split the selector. `forcedTools()` (named/`required` only)
  drives `effectiveSchema()`'s tool-call schema; `auto`/absent ⇒ no forcing
  schema. `toolSpecs()` (the template menu) still returns all tools for
  auto/required/named, nil for none/empty. Keep `structuredRequestError()` happy.
- **Pin (MLX-free, ADR 009):** unit test the matrix — `{none, auto, absent,
  required, named-hit, named-miss} × {0,1,N tools}` → expected
  `(forcesGuide, menuAdvertised)`. New `ToolChoiceResolutionTests`.
- Behavior after S1 alone: `auto` no longer forces → falls to the substrate
  free-gen path → tool call is currently **dropped** (S2 fixes). So S1 is not
  independently shippable; it lands stacked with S2.

### S2 — wire the substrate `.toolCall` event → `GenChunk`
- `GenChunk.swift`: add `case toolCall(name: String, argumentsJSON: String)`.
- `MLXLLMModule.swift`: in the substrate-stream consumer, replace `default:
  break` with a `case .toolCall(let tc):` that maps `tc.function.name` +
  `JSONSerialization(tc.function.argumentsObject, .sortedKeys)` → `.toolCall`.
  Reuse the same arg-serialization as `parseToolCall` (extract a shared
  `AthenaServerKit` helper `toolArgumentsJSON(_:)` so both agree).
- **Scope check (ADR 034 open Q):** does the `.gemma4` handler emit multiple
  `.toolCall` per turn? If yes, carry an array; if one-per-turn, keep scalar +
  `log` a note if a second arrives. Confirm before finalizing the GenChunk shape.
- **Pin:** unit-test the `ToolCall → (name, argsJSON)` mapping incl. nested/empty
  args and key-order stability (mirror `ToolCallParseTests`).

### S3 — serialize `.toolCall` in both response paths
- Non-streaming: `GenCollected` gains `toolCall: (name,args)?`; `collectMetered`
  captures it on `.toolCall`; `chatCompletionResponse`/`chatChoice` surface
  `tool_calls` + `finish_reason:"tool_calls"` when present (independent of the
  Guide-forced `isToolCall`). Both triggers → one builder.
- Streaming: `pumpTokens` handles `.toolCall` directly — emit the v0.10.230
  `delta.tool_calls` chunk + set `finishOverride = "tool_calls"`. The
  Guide-forced `isToolCall` buffer path stays; the `.toolCall` event is an
  additional emit trigger.
- **Pin:** the existing `parseToolCall`/delta-shape tests cover the wire; add a
  pump-level assertion only if a pure seam exists (else rely on the e2e DoD).

### S4 — faithful tool-turn history
- `ChatTurn.swift`: add `toolCalls: [(id,name,argumentsJSON)]` (assistant) and
  `toolCallID: String?` (tool). DTO→ChatTurn mapping populates them from the
  OpenAI request (`message.tool_calls`, `message.tool_call_id`).
- `chatMessages`: map onto the substrate `Chat.Message` tool fields so the Gemma
  4 template renders assistant-call + tool-result coherently. Verify the
  substrate `Chat.Message` exposes tool_call/tool_call_id fields; if not, render
  via `additionalContext`/content per the template's expectation.
- **Pin:** unit-test the DTO→ChatTurn→Chat.Message mapping for a 3-turn
  user→assistant(tool_call)→tool sequence (MLX-free).

### S5 — OpenAPI spec + docs + e2e
- `OpenAPISpec.swift`: document `tool_choice` semantics on
  `/v1/chat/completions` (auto=model-decides, required/named=forced, none).
  **[oai]** stays. Note the substrate-arch-gated auto-detection honesty boundary.
- `docs/tool-calling.md` (new or fold into existing): usage + the `auto` vs
  `required` distinction + the arch-gating caveat.
- `deploy/e2e-tool-choice-auto.sh`: the DoD script (studio-run).

## Test bar
- Every slice lands with its MLX-free unit pin (ADR 008/009); `./deploy/test.sh`
  green.
- Heavy/numeric correctness = the e2e DoD on the studio (no metallib under
  `swift test`).
- Regression guards retained: `stream:false`+forced tools (v0.10.230 shape),
  plain chat both stream modes, `none` (no tool call).

## Rollout
- Stacked commits S1→S5, direct-to-main per slice with annotated tag (house
  workflow). S1 stacks with S2 (not independently shippable). Build is
  xcodebuild Release; deploy to studio; run the e2e DoD there.
- No revert knob (operator decision). `required` remains the forced-call escape.

## Risks / open
- **Multi-`.toolCall` per turn** (S2 scope check) — could reshape GenChunk.
- **Leading prose before a Gemma 4 `.toolCall`** — decide drop vs surface (ADR
  034 open Q); confirm empirically on the studio during S2/S3.
- **Consumer impact** — confirm nothing depends on `auto`-always-calls.
- **Substrate `Chat.Message` tool-field availability** (S4) — may need a
  template-specific rendering if the fields aren't first-class.
