# 042 — Context-window discovery + exact token counting

**Status:** Accepted — **IMPLEMENTED 2026-07-25** (B1 `76134365`, B2 `eae8724f`, B3 `56b13ccb`, Anthropic parity route same day). Brownfield-gated; operator-approved 2026-07-25 (surface shape confirmed by interview). DoD: `deploy/e2e-count-tokens.sh`.
**Date:** 2026-07-25
**Amended 2026-07-25 (same day)** on a downstream consumer requirement spec: the pre-flight count route moves to the **OpenAI dialect** as the primary surface (the deferral tripwire below fired — an OpenAI-dialect consumer asked), counting is specified as **no-eval / no-gate**, the model field is renamed `context_length` to match ecosystem convention, and `max_output_tokens` is refused as unpublishable. See §4.
**Milestone:** TBD (client-enablement surface)
**Relates to:** ADR 013 (`/v1` = inference surface), ADR 036 (`[anthropic]` protocol adapter), ADR 030 (prefill ceiling), ADR 015 (block-until-ready cold load), ADR 021 (config-only preflight).

## Context

A consumer needs to track **how much context room is left** in a conversation so it can compact or trim before the next turn. Today Athena makes that impossible to compute without guessing, and the gap is in two distinct places.

**The denominator is not published.** Nothing on the inference surface reports the model's context window. `GET /v1/models` / `GET /v1/models/{id}` return `{id, object, created, owned_by}` and nothing else (`AthenaServer+Admin.swift:976`). `ModelConfigInfo` — the MLX-free `config.json` reader already used for the governor's KV sizing, structured-output vocab, and `athena show` — parses layers, heads, head_dim, hidden_size, vocab_size, and vision presence, but **not** `max_position_embeddings`. Separately, Athena imposes its own **prefill ceiling** (ADR 030): the operator's `max_prompt_tokens` or, when unset, the device-derived `GovernorMemory.defaultPromptTokenCeiling`. That ceiling is frequently *lower* than the checkpoint's advertised window and is not published anywhere either — a client discovers it only by tripping the 400 `input_too_long`, whose message does at least name the cap. So the real limit is the **min of two numbers, neither of which is discoverable** from the inference surface.

(The control plane can already read the raw `config.json`: `GET /api/models/{name}` returns it verbatim. That is a control-plane call requiring `model.read`, returning an unresolved blob, on a surface ADR 013 reserves for daemon control — not a substitute for the inference client knowing its own budget.)

**The numerator can only be measured after the fact.** Every response already carries exact input accounting — `usage.prompt_tokens` (+ `prompt_tokens_details.cached_tokens`) on the OpenAI dialect, `usage.input_tokens` on the Anthropic one. That answers "what did I just spend", not "will the next turn fit". A client cannot count tokens locally without shipping the model's tokenizer *and* replicating its chat template, and any character-based estimate is wrong in exactly the case that matters (near the boundary). Anthropic's own protocol solves this with `POST /v1/messages/count_tokens`, which Athena does not implement — and since ADR 036 already ships the Messages adapter, its absence is a hole in a dialect Athena claims to speak.

## Decision

### 1. `ModelConfigInfo` learns the context window

Add `maxPositionEmbeddings: Int?`, read by the existing top-level-then-`text_config` accessor (multimodal wrappers put the transformer dims in the nested object, and that is where a VLM's text context length lives). Optional like every other field: a config that omits it yields `nil` and callers omit the field rather than substituting a guess. One field, one parse line, pinned by the existing `ModelConfigInfo` unit tests.

### 2. `/v1/models` publishes both numbers, tagged native

`GET /v1/models/{id}` and each element of `GET /v1/models` gain two **omitted-when-nil** fields:

```json
{ "id": "gemma-4-26b-a4b-it-8bit", "object": "model", "created": 1751000000, "owned_by": "athena",
  "context_length": 131072,
  "max_prompt_tokens": 28373 }
```

- `context_length` — what the checkpoint advertises (`max_position_embeddings`). Absent when the config omits it. (Named `context_window` in this ADR's first draft; renamed per §4(c).)
- `max_prompt_tokens` — the **effective** prefill ceiling this daemon will enforce for that model right now: the operator's `max_prompt_tokens` when set, else the ADR 030 device-derived default. Absent only under the explicit unbounded opt-out (`max_prompt_tokens = 0`).

The client computes `remaining = min(context_length, max_prompt_tokens) − input_tokens`. Athena publishes both inputs rather than a single pre-mined "remaining" number, because the two limits fail differently — exceeding the window is a model-capability error, exceeding the ceiling is a memory guard — and a client that logs which one bound it can act differently (switch model vs. compact history).

**`/v1` compatibility rule (CLAUDE.md, binding).** `GET /v1/models` is an `[oai]` drop-in and stays one: both fields are **Athena-native extensions on an OpenAI-compatible route**, and are marked as such in the `OpenAPISpec.swift` operation description in the same edit, plus in the CLAUDE.md stable-endpoint list. OpenAI's model object has no context-length field, so a strict OpenAI client ignores two unknown keys; nothing about the route's existing semantics changes. This ADR is the introducing record required by that rule.

### 3. A pre-flight count route — exact, not estimated

_(Dialect and gate behavior amended by §4 — the primary route is `POST /v1/chat/completions/count_tokens` `[native]` returning `{"prompt_tokens": N}`, and counting takes no execution gate. The reasoning below stands unchanged; only the surface and the gate claim moved.)_

The count is produced by **the request path's own tokenization**, not a parallel implementation: the adapter decodes to native `ChatTurn`s exactly as `/v1/messages` does, and the LLM module renders and tokenizes through the same `container.prepare(input: UserInput(...))` call whose `lmInput.text.tokens` count already feeds `usage.prompt_tokens` and the ADR 030 ceiling check. Same chat template, same tokenizer, same special tokens, same tool serialization — so a preflight count and the subsequent request's `input_tokens` agree by construction rather than by maintenance. Nothing generates; no prefill runs.

Consequences of routing through the real path, accepted deliberately:

- **It needs the model's tokenizer, which lives in the loaded model context.** So the route obeys ADR 015: resident ⇒ immediate, on-disk-not-resident ⇒ blocks until loaded within `cold_load_wait_secs`, `pulling` ⇒ 503. Counting tokens can therefore trigger a cold load. Rejected the alternative of loading the tokenizer standalone via swift-transformers: it is cheaper, but it re-implements template application outside the path that serves requests, and the moment those two disagree the preflight is worse than useless.
- **`prepare` performs a small MLX evaluation** (building and reading back the token array), so the route acquires the ADR 029 inference-execution gate for that span like every other Metal-touching op. The span is milliseconds and does not decode, but the rule has no exceptions.
- **Not metered, not quota-enforced.** It consumes no model output and exists specifically so a client can stay *under* its ADR 041 budget; refusing the preflight to an over-budget principal would be self-defeating. It requires the `inference` permission and is rate-limited like any other work route.

## 4. Amendment (2026-07-25) — consumer requirements reconciled

A downstream `/v1/chat/completions` consumer building conversation compaction supplied a requirement spec. It confirms the semantics above and forces four changes. Its own priority ranking was window-size ≥ post-turn usage > pre-flight count.

**(a) The pre-flight count belongs on the OpenAI dialect, not the Anthropic one.** §3's deferral was explicitly conditioned on "until an OpenAI-dialect consumer actually asks" — one has. The primary route becomes:

```
POST /v1/chat/completions/count_tokens        [native]
body:     a chat.completions request minus generation params (model, messages, tools, response_format)
response: { "prompt_tokens": 14231 }
```

Path and field names are chosen so the number is directly comparable to the `usage.prompt_tokens` the same body will report after inference — the consumer calibrates one against the other. `/v1/tokenize` is rejected as the name: llama.cpp's `/tokenize` returns token *ids*, and this route deliberately does not (nobody asked for ids, and returning them would leak a per-message breakdown we do not want to commit to). Tagged `[native]` per the `/v1` compatibility rule — it is an Athena extension under `/v1` with no OpenAI equivalent.

The Anthropic `POST /v1/messages/count_tokens` from §3 was **retained but deferred** — same counting core, a decoder that already exists, ~one thin handler. Building it ahead of a Messages-dialect consumer would have been speculative.

**Deferral LIFTED 2026-07-25 (operator decision), SHIPPED same day.** The reason is not that a Messages consumer appeared: it is that **dialect parity is itself a product property**. Athena claims to speak two dialects (ADR 036), and a capability present in one but absent in the other is a defect from a consumer's seat — a client choosing a dialect should not thereby lose the ability to budget its context. That argument does not depend on a named consumer, so the "wait for one" tripwire no longer applies. Cost was the predicted thin handler: decode with the existing `lower(principal:)`, the shared `prepareChat` seam, the same `countPromptTokens` core, and Anthropic's own field name (`input_tokens`, matching its `usage.input_tokens` so a client compares the two directly). Both routes return the same number for the same conversation **by construction** — one core, one tokenizer, one template — which is what makes parity a guarantee rather than a promise to maintain. Pinned by `deploy/e2e-count-tokens.sh` (cross-dialect equality + equality with each dialect's own post-inference usage field).

**(b) Counting performs no eval and takes no inference gate.** Verified in the pinned substrate: for text-only input, *both* the text-LLM processor (`LLMModelFactory.swift:492`) and the Gemma 4 VLM processor (`Gemma4.swift:2697`) implement `prepare` as `messageGenerator.generate(from:)` → `tokenizer.applyChatTemplate(messages:tools:additionalContext:)` → wrap in an `MLXArray`; image-token expansion is guarded by `!input.images.isEmpty`. So the count is `prepare(input:).text.tokens.shape.last` — read from the **shape**, never `asArray`, so **nothing is evaluated**: no kernel launch, no prefill, no forward pass. The route therefore does **not** acquire the ADR 029 execution gate, which reverses §3's provisional statement.

This is a deliberate, narrow exception to "every Metal-touching op acquires the gate," and its justification is that no *execution* occurs — the only Metal interaction is a bounded host-side buffer allocation (4 bytes/token; ~56 KB for a 14k-token prompt). Taking the gate would make a call the consumer intends to issue **every turn** queue behind other users' decodes on a shared appliance, which is precisely the latency the requirement forbids ("MUST be cheap"). **Tripwire:** if a substrate change makes `MLXArray` construction lazy-but-gated, or if `shape` ever forces an eval, this route acquires the gate and the cheapness claim is retracted.

**Amendment, 2026-07-25 (measured at implementation — the concurrency half of the cheapness claim is RETRACTED).** The no-eval / no-gate part holds: an idle count costs **~34 ms** (`Llama-3.2-3B-Instruct-8bit`, Release), and the route acquires no ADR 029 gate. But a count is **not concurrent with a decode**, for a reason the ADR did not anticipate: `ModelContainer.prepare` reaches the processor through `SerialAccessContainer.read`, an explicit async mutex the substrate holds **for the entire generation** (`ModelContainer.swift`, `Utilities/SerialAccessContainer.swift`) — exclusive by design, precisely so no caller can touch model state mid-decode. Measured: a count issued 3 s into an 11.3 s decode returned in **8.3 s** (it waited the decode out); counts before and after that decode were 34–37 ms. The wait tracks the decode, not the count: the same probe against `gemma-4-31b-it-4bit`, whose 900-token decode runs far longer, waited **34.8 s** while its idle count stayed at 38 ms. So the honest statement is: *counting is cheap, and does not enter the inference gate's queue, but on a busy single-tenant appliance it is delayed by the generation in flight.* Every client-facing surface says so (the `OpenAPISpec` route description, the handler and module doc comments, and `deploy/e2e-count-tokens.sh`, which reports the latency instead of asserting a bound it cannot hold).

This does not change the decision — the count is still exact, still gate-free, still the right shape — and the delay is bounded by one decode rather than by a queue. It is not worth fixing speculatively: the fix is substrate-side (hold the processor/tokenizer outside the serial container, or give `ModelContainer` a read path that doesn't need the generation mutex), and it should land upstream if a consumer reports the wait as a problem. **Tripwire (revised):** if a consumer needs counting concurrent with generation, that is an upstream `mlx-swift-lm` change, not a serve-path one — do not work around it by re-implementing tokenization outside `container.prepare`, which would forfeit the exactness that is this route's entire justification.

Image-bearing count requests are refused with a cause-naming 400 rather than silently under-counting: the image path adds `imageSeqLength` tokens per image and requires real preprocessing, which is neither cheap nor needed by any consumer that has asked. Text + tools is the supported surface.

**(c) `context_window` → `context_length`.** OpenRouter and the OpenAI-compatible server ecosystem use `context_length`; the consumer's spec proposed it. Since the field is a native extension either way, matching the ecosystem name is free and reduces the mapping every client must write. `max_prompt_tokens` keeps its name — it is literally the config key it reports.

**(d) `max_output_tokens` is refused, not deferred.** The consumer asked for a generation cap "if generation is separately capped". It is not: `AthenaConfig.maxTokens` is a **default** (1024) that any request overrides freely, not a ceiling. Publishing a default under a name that reads as a cap would be a lie in exactly the direction that breaks a budgeting client. The real output bound is whatever the client passes as `max_tokens`/`max_completion_tokens`, drawn from the same window as the prompt.

**(e) Post-turn usage already ships — no work.** `stream_options.include_usage:true` has been honored since M27.4: the SSE stream emits a final `chat.completion.chunk` with `choices: []` and a full `usage` object (`AthenaServer+SSE.swift:388`), plus `prompt_tokens_details.cached_tokens`. The consumer's item C is satisfied today by sending `stream_options`.

**Consequence worth stating plainly:** the consumer assumed `context_length` alone is "the real configured window". It is not — it is what the checkpoint declares, and Athena's ADR 030 prefill ceiling is frequently **an order of magnitude lower** (the derived default is `sqrt(maxBufferSize / 128)`, i.e. roughly 17k tokens on a 48 GB device, against a Gemma 4 checkpoint advertising 128k+). A client that budgets against `context_length` alone on a default-configured daemon will plan for a window it cannot use and hit 400 `input_too_long` long before its threshold fires. This is why §2 publishes **both** numbers and why the effective budget is their **min** — and it is a strong argument for the operator setting `max_prompt_tokens` deliberately rather than inheriting a heuristic floor calibrated for OOM safety, not for context capacity.

## Rejected alternatives

- **`POST /v1/tokenize` as the route name** (llama.cpp analogue). Implies returning token ids, which this route does not. See §4(a).
- **Anthropic-dialect-only counting** (the pre-amendment §3 position). Reversed: the consumer that asked speaks the OpenAI dialect. The Anthropic route is deferred, not deleted.
- **Publishing `max_output_tokens`.** No such cap exists; see §4(d).
- **Acquiring the inference gate for counting.** Correct-by-default but wrong here: nothing executes, and gating would serialize an every-turn call behind other users' decodes. See §4(b) for the tripwire that would reverse this.
- **A single server-computed `tokens_remaining`.** Requires the daemon to hold a notion of conversation state it deliberately does not have (Athena is stateless per request), and collapses two differently-actionable limits into one opaque number.
- **Character/word heuristics** (`chars / 4`) to avoid a round trip. Wrong precisely at the boundary the client is trying to respect.
- **Exposing the window only through the control plane** (status quo: `GET /api/models/{name}`'s raw config). Wrong surface (ADR 013), wrong permission, and unresolved — it reports what the checkpoint claims, never the ceiling Athena will actually enforce.
- **Deriving the window from the tokenizer or weights** instead of `config.json`. `max_position_embeddings` is the field every MLX/HF loader honors; anything else is inference about inference.

## Consequences

- One new optional field in `ModelConfigInfo`, two new omitted-when-nil fields on the OpenAI model object, one new route. `OpenAPISpec.swift` (SSOT) and the CLAUDE.md stable-endpoint list are updated in the same edit; the new route is tagged `[anthropic]`, the two fields explicitly marked native extensions.
- `context_window` is only as honest as the checkpoint: a config that omits `max_position_embeddings`, or advertises a window the weights cannot really sustain, yields a missing or optimistic field. Athena reports what the checkpoint declares and does not attempt to validate it — the same honesty boundary ADR 021 draws for config-only preflight (it confirms declared packaging, never numeric behavior).
- `count_tokens` can cold-load a model, which is surprising for a call that "just counts". Documented on the route.
- Athena's Anthropic dialect gets closer to complete, which is a maintenance commitment: a client that finds `count_tokens` will expect its count to match a subsequent request's `input_tokens`, and the shared-path decision above is what keeps that true.
