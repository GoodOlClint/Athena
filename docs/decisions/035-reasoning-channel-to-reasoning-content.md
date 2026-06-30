# 035 — Route channel-delimited reasoning to `reasoning_content`

**Status:** Accepted (gate-approved 2026-06-30) — implementing.
**Date:** 2026-06-30
**Milestone:** TBD (tool-calling / reasoning correctness)
**Relates to:** ADR 034 (tool_choice:auto un-forcing surfaced this), ADR 013
(`/v1` OpenAI-compat surface).

## Context

`gemma-4-26b-a4b-it` (and its family) emit a **channel-delimited** output format
as **literal text** (the markers are NOT registered special tokens):

```
<|channel>thought
…reasoning…
<channel|>…final answer…
```

(tokenizer_config: `soc_token: "<|channel>"`, `eoc_token: "<channel|>"`, plus an
`x-regex` splitting output into `thinking` / `tool_calls` / `content`.) The chat
template **forces the thought channel on whenever `tools` are present**
(`{% if … or tools %}`), so it cannot be suppressed via `enable_thinking`.

The substrate's `.gemma4` tool parser (`ToolCallProcessor`) strips **tool-call**
markers (`<|tool_call>…<tool_call|>`) but has **no concept of the thought
channel**, so `<|channel>thought…<channel|>` falls straight through into the
`content` stream — the observed token leak.

It was invisible until now because ADR 034's old forced path masked all output
to pure tool-call JSON (no channel tokens possible). Free generation (what
`auto` now uses, and what plain chat already used) surfaces the model's native
format. This is therefore a **pre-existing reasoning-format gap**, exposed — not
an ADR 034 regression.

## Decision

Parse the channel-delimited reasoning out of the content stream and surface it
as OpenAI **`reasoning_content`** (operator-approved over dropping it — preserves
chain-of-thought for clients that want it; keeps `content` clean for those that
don't).

1. **MLX-free reasoning filter** (`AthenaServerKit`, unit-pinned per ADR 008/009):
   - `splitReasoningChannel(_:)` — one-shot split of a complete generation into
     `(content, reasoning)` for the non-streaming path.
   - `ReasoningChannelFilter` — a stateful streaming splitter (mirrors
     `StopStreamFilter`'s partial-marker holdback) that routes incremental text
     to content vs reasoning across chunk boundaries.
   - Strips `<|channel>thought…<channel|>` blocks; the leading channel-name
     header (`thought\n`) is dropped from the reasoning text.

2. **DTOs:** `ChatMessage` gains `reasoning_content: String?` (non-streaming),
   `ChatDelta` gains `reasoning_content: String?` (streaming). Both omitted from
   JSON when nil — plain responses are byte-unchanged.

3. **Wiring:** non-streaming splits the collected text before building the
   choice; the streaming pump feeds content `.text` through the filter, emitting
   `delta.content` and `delta.reasoning_content` separately. The Guide-forced
   tool path (masked JSON, no channels) and the substrate `.toolCall` path are
   unaffected; the filter applies only to free-generation content text.

**Universal, not model-gated.** The filter keys on the literal
`<|channel>`/`<channel|>` strings, which do not occur in normal content, so it is
a no-op for models that don't emit them — no per-model branching. (Upgrade path:
read `soc_token`/`eoc_token` from tokenizer_config if a future model uses
different channel delimiters.)

## Rejected alternatives

- **Drop the reasoning entirely.** Simpler, but discards chain-of-thought a
  client may want. Operator chose `reasoning_content`.
- **Suppress via `enable_thinking=false`.** Blocked: the template forces thinking
  on when `tools` are present.
- **Fix in the substrate `.gemma4` parser.** The "right" layer, but a broad
  change in the vendored fork; the Athena-side filter is localized, testable, and
  composes with the existing `StopStreamFilter`.

## Consequences

- `/v1/chat/completions` surfaces `reasoning_content` (OpenAI-aligned) for
  channel-format models; `content` is clean.
- One more stateful streaming filter to maintain (holdback for partial markers).
- Honesty boundary: reasoning extraction is **format-specific** (the
  `<|channel>` delimiters). Models with other reasoning conventions are
  unaffected (no-op) and handled by their own mechanisms (e.g. Qwen
  `enable_thinking`).
- Decision logic MLX-free + unit-pinned (ADR 008/009).
