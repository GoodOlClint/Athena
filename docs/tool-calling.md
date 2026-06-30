# Tool calling (`/v1/chat/completions`)

Athena implements OpenAI-style function/tool calling on `POST
/v1/chat/completions`. Declare tools with `tools`; control whether the model
calls one with `tool_choice` (ADR 034).

## `tool_choice` semantics

| `tool_choice` | Behavior |
|---|---|
| absent / `"auto"` | **The model decides** each turn — a normal text answer **or** a tool call. The default when `tools` are present. Not forced. |
| `"required"` | Forces *some* tool call (grammar-constrained to a valid call). |
| `{"type":"function","function":{"name":"X"}}` | Forces a call to tool `X`. A name matching no declared tool falls through to `auto`. |
| `"none"` | No tool call — the tool menu is withheld, the model answers in text. |

This makes multi-turn tool use work: send the assistant's tool call back as an
`assistant` message with `tool_calls`, append the result as a `role:"tool"`
message with `tool_call_id`, and re-request. Under `auto` the model sees the
call→result history and answers in text instead of re-calling — no spiral.

## Responses

- A tool call surfaces as `choices[0].message.tool_calls[]` (non-streaming) or
  `choices[0].delta.tool_calls[]` (streaming), with `finish_reason:"tool_calls"`.
- `arguments` is a JSON-encoded string (OpenAI shape), with stable key order.

## Honesty boundary

Under `auto`, a freely-chosen tool call is detected by the substrate's native,
**per-architecture** tool-call parser. Gemma 4 is supported (`gemma4`
format). An architecture with no recognized tool-call format may not detect a
free-form call; use `tool_choice:"required"` (grammar-forced) for guaranteed
extraction on such models.
