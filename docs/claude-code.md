# Pointing Claude Code at Athena

Claude Code speaks the Anthropic Messages API. As of ADR 036, Athena serves it
natively at `POST /v1/messages` — **no proxy** (no LiteLLM / claude-code-router).

## Connect

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:7447   # your Athena host:port
export ANTHROPIC_AUTH_TOKEN=athena                # sent as `Authorization: Bearer`
export ANTHROPIC_API_KEY=""                       # keep empty (don't send x-api-key)
claude --model gemma-4-26b-a4b-it-8bit            # any resident/store LLM id
```

Claude Code appends `/v1/messages` to `ANTHROPIC_BASE_URL`. `ANTHROPIC_AUTH_TOKEN`
is sent as a bearer token, which Athena's existing RBAC accepts (in loopback dev
mode auth is off and the token is ignored). Verified end-to-end on
`gemma-4-26b-a4b-it-8bit`.

## Run the daemon with a prompt cap that fits Claude Code

Claude Code's system prompt + tool definitions are large (~26k tokens). The
default per-input ceiling (ADR 030, conservatively derived from the Metal buffer)
can reject that with `400 input_too_long`. Raise it for the serving daemon:

```sh
athena load --llm-model gemma-4-26b-a4b-it-8bit --max-prompt-tokens 40000
```

(The default is a safe-prefill estimate; the substrate's attention handles more,
so a higher cap is the expected operator setting when serving agentic harnesses.)

## Notes / limits

- **Tool schemas:** Athena renders Claude Code's full tool set. Two engine fixes
  landed for this (v0.10.236): integer schema constraints (`minimum`, `maxLength`,
  …) and a typeless "any"-type property (`Workflow.args`) used to crash the
  chat-template renderer for *both* the OpenAI and Anthropic dialects.
- **`POST /v1/messages/count_tokens`** is not implemented (404). Claude Code uses
  it for context-window estimates and degrades gracefully; add it if estimates
  matter.
- **Streaming** is supported (Claude Code streams by default). `image`/`document`
  content blocks, prompt-caching, and extended-thinking blocks are not yet
  supported (clean 400).
- **`x-api-key`:** not required — Claude Code authenticates via
  `ANTHROPIC_AUTH_TOKEN` (bearer). A dedicated `x-api-key` alias is only needed
  for a client that sends *only* that header.
