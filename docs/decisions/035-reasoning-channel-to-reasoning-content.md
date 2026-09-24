# 035 — Route channel-delimited reasoning to `reasoning_content`

**Status:** Accepted (gate-approved 2026-06-30) — **IMPLEMENTED v0.10.232** (`deabfe69`); **amended 2026-09-23 (#198)** for Qwen3.5's `<think>…</think>`, which required model-gating the filter — see "Amendment" below. This line read "implementing" from the moment the code landed; corrected #193 against the tree. Live: `ReasoningChannelFilter` + `splitReasoningChannel` (Gemma), `QwenThinkFilter` + `splitQwenThink` + `qwenExpectsThinking` (Qwen3.5) — all in `Sources/AthenaServerKit/ReasoningChannel.swift`.
**Date:** 2026-06-30
**Milestone:** TBD (tool-calling / reasoning correctness)
**Relates to:** ADR 034 (tool_choice:auto un-forcing surfaced this), ADR 013 (`/v1` OpenAI-compat surface).

## Context

`gemma-4-26b-a4b-it` (and its family) emit a **channel-delimited** output format as **literal text** (the markers are NOT registered special tokens):

```
<|channel>thought
…reasoning…
<channel|>…final answer…
```

(tokenizer_config: `soc_token: "<|channel>"`, `eoc_token: "<channel|>"`, plus an `x-regex` splitting output into `thinking` / `tool_calls` / `content`.) The chat template **forces the thought channel on whenever `tools` are present** (`{% if … or tools %}`), so it cannot be suppressed via `enable_thinking`.

The substrate's `.gemma4` tool parser (`ToolCallProcessor`) strips **tool-call** markers (`<|tool_call>…<tool_call|>`) but has **no concept of the thought channel**, so `<|channel>thought…<channel|>` falls straight through into the `content` stream — the observed token leak.

It was invisible until now because ADR 034's old forced path masked all output to pure tool-call JSON (no channel tokens possible). Free generation (what `auto` now uses, and what plain chat already used) surfaces the model's native format. This is therefore a **pre-existing reasoning-format gap**, exposed — not an ADR 034 regression.

## Decision

Parse the channel-delimited reasoning out of the content stream and surface it as OpenAI **`reasoning_content`** (operator-approved over dropping it — preserves chain-of-thought for clients that want it; keeps `content` clean for those that don't).

1. **MLX-free reasoning filter** (`AthenaServerKit`, unit-pinned per ADR 008/009):
   - `splitReasoningChannel(_:)` — one-shot split of a complete generation into `(content, reasoning)` for the non-streaming path.
   - `ReasoningChannelFilter` — a stateful streaming splitter (mirrors `StopStreamFilter`'s partial-marker holdback) that routes incremental text to content vs reasoning across chunk boundaries.
   - Strips `<|channel>thought…<channel|>` blocks; the leading channel-name header (`thought\n`) is dropped from the reasoning text.

2. **DTOs:** `ChatMessage` gains `reasoning_content: String?` (non-streaming), `ChatDelta` gains `reasoning_content: String?` (streaming). Both omitted from JSON when nil — plain responses are byte-unchanged.

3. **Wiring:** non-streaming splits the collected text before building the choice; the streaming pump feeds content `.text` through the filter, emitting `delta.content` and `delta.reasoning_content` separately. The Guide-forced tool path (masked JSON, no channels) and the substrate `.toolCall` path are unaffected; the filter applies only to free-generation content text.

**Universal, not model-gated.** The filter keys on the literal `<|channel>`/`<channel|>` strings, which do not occur in normal content, so it is a no-op for models that don't emit them — no per-model branching. (Upgrade path: read `soc_token`/`eoc_token` from tokenizer_config if a future model uses different channel delimiters.)

## Rejected alternatives

- **Drop the reasoning entirely.** Simpler, but discards chain-of-thought a client may want. Operator chose `reasoning_content`.
- **Suppress via `enable_thinking=false`.** Blocked: the template forces thinking on when `tools` are present.
- **Fix in the substrate `.gemma4` parser.** The "right" layer, but a broad change in the vendored fork; the Athena-side filter is localized, testable, and composes with the existing `StopStreamFilter`.

## Consequences

- `/v1/chat/completions` surfaces `reasoning_content` (OpenAI-aligned) for channel-format models; `content` is clean.
- One more stateful streaming filter to maintain (holdback for partial markers).
- Honesty boundary: reasoning extraction is **format-specific** (the `<|channel>` delimiters). Models with other reasoning conventions are unaffected (no-op) and handled by their own mechanisms (e.g. Qwen `enable_thinking`).
- Decision logic MLX-free + unit-pinned (ADR 008/009).

## Amendment (2026-09-23, #198) — Qwen3.5's `<think>…</think>` needed model-gating

**Observed.** `Qwen3.5-27B-4bit`, `temperature: 0`, no `chat_template_kwargs` (thinking on by default): `content` began `Thinking Process:\n\n1. **Analyze the Request:**…`, `reasoning_content` was `null` — the same leak this ADR fixed for Gemma, on a different model. Root cause: Qwen3.5's own chat template pre-inserts the OPENING `<think>\n` into the **prompt** whenever thinking is active (`{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- else %}<think>\n{%- endif %}` at the generation-prompt tail, confirmed against the shipped `tokenizer_config.json`'s `chat_template`). The model's own completion therefore carries **only the close marker** — there is no start delimiter in the output to key a marker-only filter on.

**Why this breaks "Universal, not model-gated."** Gemma's filter is safe as a no-op because `<|channel>`/`<channel|>` never occur in ordinary content, and the state machine only ever enters `.reasoning` after it SEES the start marker — an ordinary response from any model that never emits the marker is untouched. Qwen3.5's close-tag-only form has no equivalent: a filter that defaults to "everything is reasoning until `</think>`" would misclassify the entirety of any Qwen response that has no thinking at all (`enable_thinking: false`, or any other model, since `</think>` is not otherwise Qwen-specific in the way `<|channel>` is Gemma-specific-by-construction). There is no text-only rule that stays universal here.

**Decision.** `QwenThinkFilter` (`Sources/AthenaServerKit/ReasoningChannel.swift`) is model-gated: `qwenExpectsThinking(modelName:chatTemplateKwargs:)` resolves — from the resolved model id (a `qwen3.5` substring match on the canonical store id; MLX-free, no `ModelSupport`/config read, since `AthenaServerKit` has no `AthenaLLM` dependency to classify architecture properly) and the request's `enable_thinking` (mirroring the template's own default of `true`) — whether the prompt already opened the block, BEFORE generation starts. The filter's initial state is `.reasoning` when it did (streams reasoning tokens live, no buffering), `.content` otherwise (a defensive literal `<think>` match still handles a paired form if a differently-configured template ever emits the open tag itself). Chained after the Gemma filter in all three call sites (`/v1/chat/completions` non-stream + stream, `/v1/messages` non-stream + stream) so a model that emits neither format is fully unaffected.

**Rejected: buffer-until-decided (stay universal).** A filter that holds ALL output until it either sees `</think>` or reaches end-of-generation, deciding the split only then, needs no model gate — but it would defer the first `content` byte of every plain (non-thinking) response from every model until generation ends, since nothing can rule out a future `</think>` without knowing the model/request. Destroys live streaming UX universally to handle one model's ambiguous format. Rejected.

**Upgrade path (tokenizer_config).** ADR 035's original upgrade path — read `soc_token`/`eoc_token` from `tokenizer_config` — doesn't apply to Qwen3.5: its `tokenizer_config.json` carries no such field (checked directly against the `mlx-community/Qwen3.5-27B-4bit` snapshot); the `<think>`/`</think>` tags are literals baked into the Jinja chat template, not exposed tokenizer metadata. Hard-coded, same as Gemma's markers.
