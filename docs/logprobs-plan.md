# C2 — honor `logprobs`/`top_logprobs` (ADR 013 §4) — implementation plan

**Status:** Shipped (v0.10.190). Uniform across all archs via the ArgMax pick /
`LogitProcessor` capture seams — **no substrate fork needed** (the key finding below
held). Deterministic-path only (temp>0 ⇒ 400, ADR 013 §4); non-streaming + streaming;
`LogprobMath` numerics unit-pinned (650/0), shape e2e-validated via the stub (573/0). Real
logprob *values* on a live Qwen + non-Qwen model remain a RUNBOOK check.
**Decision of record:** ADR 013 §4 ("Emit `top_logprobs` on the deterministic path
instead of 400"). **Scope (operator-chosen):** uniform support across **all** decode
paths/architectures, not just the Qwen family.
**Milestone:** operator-assigned tag (continues the Track 2 batch in
`docs/api-surface-rollout-plan.md`).

## Problem

`POST /v1/chat/completions` with `logprobs:true` (and optional `top_logprobs:N`)
currently returns **400 unsupported_parameter** (`OpenAIDTO.unsupportedParameter()`
`:242-248`). ADR 013 §4 ratified the reversal: the decode path already computes the
logits, so honor the request and emit the OpenAI `logprobs` response object. The code
still does the opposite — a direct contradiction of a ratified decision.

## Key finding (de-risks the "extend the substrate" choice)

The substrate decode path does **not** need a fork. `GuidedSubstrate` already drives the
substrate `TokenIterator` with a `LogitProcessor`
([GuidedSubstrate.swift:27-48](../Sources/AthenaLLM/GuidedSubstrate.swift#L27)) whose
`process(logits:)` sees the per-step logits and `didSample(token:)` sees the chosen
token. The unguided substrate stream (`beginGeneration`) can be given the same kind of
capturing processor. So uniform coverage is achievable through the **existing public
seam** — no change to vendored `mlx-swift-lm`, no growth of the un-pushed fork (release
blocker #10). This satisfies the chosen scope (all architectures) without its stated
downside.

## Decode paths to cover (the dispatch)

`MLXLLMModule.generateMetered` ([:649](../Sources/AthenaLLM/MLXLLMModule.swift#L649))
splits into two families:

| Family | Paths | Logits site | Capture seam |
|---|---|---|---|
| `runSpeculative` (generates fully, returns text+usage) | MTP greedy (`SpeculativeGeneration`), MTP sampling, dense `GuidedGreedy`, `GuidedSubstrate` | `GuidedDecoder.pick(slice)` / `guidedArgmax` / `GuidedLogitProcessor.process` | shared `pick` seam + the substrate `LogitProcessor` |
| `beginGeneration` (substrate token stream) | unguided non-MTP arches | inside the substrate `TokenIterator` | a capturing `LogitProcessor` injected here too |
| stub | `StubLLMModule` | n/a | synthesize fake logprobs for CI |

## Architecture

**Capture type (AthenaLLM, MLX-free, `Sendable`):**
```
public struct TokenLogprob: Sendable {
    public let token: String        // tokenizer.decode([id]) — the piece for this id
    public let logprob: Float        // log_softmax(logits)[id]
    public let bytes: [Int]          // UTF-8 of `token`
    public let top: [TopLogprob]     // top_logprobs alternatives (may be empty)
}
public struct TopLogprob: Sendable { let token: String; let logprob: Float; let bytes: [Int] }
```

**Compute seam (MLX-free decision logic, unit-pinned per ADR 009):** the *selection* of
top-K indices and the assembly from a `[Float]` logit row + chosen id is pure Swift; only
`log_softmax` over the `MLXArray` is MLX. Extract `LogprobMath.fromLogitRow(_ row: [Float],
chosen: Int, topK: Int) -> (logprob: Float, top: [(Int, Float)])` as a pure function
(testable), and keep the `MLXArray → [Float]` (a `log_softmax(...).asArray`) at the call
site. Token-id → string/bytes uses the module's tokenizer (decode happens in AthenaLLM so
the server never needs the tokenizer).

**Threading:**
- `generateMetered` gains `logprobs: Bool, topLogprobs: Int` params (default false/0).
- A capture config flows to each path. `runSpeculative`'s result struct gains
  `logprobs: [TokenLogprob]?`. The substrate `GuidedLogitProcessor` + a new capturing
  processor for `beginGeneration` accumulate `[TokenLogprob]`.
- New `GenChunk.logprobs([TokenLogprob])` terminal-ish variant
  ([GenChunk.swift:54](../Sources/AthenaLLM/GenChunk.swift#L54)): the `runSpeculative`
  family (full-text) emits one `.logprobs(list)` after `.text`; the streaming substrate
  family emits `.logprobs([oneToken])` interleaved with each `.text` piece.

**Response (server, `OpenAIDTO.swift` + `AthenaServer.swift`):**
- New DTOs: `ChatCompletionTokenLogprob {token, logprob, bytes, top_logprobs:[…]}`,
  `ChatLogprobs {content:[…]}`. Add optional `logprobs: ChatLogprobs?` to `ChatChoice`
  (:338) and `ChatChunkChoice` (:383).
- Non-streaming (`chatCompletionResponse`/`chatChoice` :1232-1283): consume the
  accumulated `.logprobs` → `choices[0].logprobs.content`.
- Streaming (`pumpTokens`/`emitDelta` :5750-5873): attach the per-delta token logprobs to
  `ChatChunkChoice.logprobs`.

**Validation (`unsupportedParameter()` → split):** stop returning `"logprobs"`/
`"top_logprobs"`. Add OpenAI's rules: `top_logprobs` ∈ 0…20 else 400
`invalid_top_logprobs`; `top_logprobs` present ⇒ requires `logprobs:true` else 400. Keep
`n>1` and non-empty `logit_bias` → 400. This split is MLX-free → unit-pinned.

## Slices (each: Release build → unit + e2e → appVersion bump → tag)

- **C2.1 — DTOs + validation + `LogprobMath` (pure).** Response DTOs, `ChatChoice`/
  `ChatChunkChoice` fields, the `unsupportedParameter` split, and the pure
  `LogprobMath.fromLogitRow` with unit tests. **Inert until capture lands** — so C2.1 does
  NOT flip the 400 yet (keeps the honest 400 until C2.3 can actually return logprobs).
- **C2.2 — capture in AthenaLLM.** `TokenLogprob`, `GenChunk.logprobs`, thread
  `logprobs`/`topLogprobs` through `generateMetered` → all paths (shared `pick` seam,
  substrate `LogitProcessor`, `beginGeneration` capturing processor), tokenizer decode.
  Stub synthesizes fake logprobs.
- **C2.3 — non-streaming response** + flip the 400 (now honest). e2e via stub asserts
  `choices[0].logprobs.content[]` shape.
- **C2.4 — streaming response.** Per-delta `logprobs`; e2e asserts a streamed chunk carries
  `logprobs`.

## Test bar

- **Unit (CI):** `LogprobMath.fromLogitRow` (top-K selection, chosen-token logprob,
  ordering, K=0, K>vocab clamp); the `unsupportedParameter` split (top_logprobs range,
  requires-logprobs, n>1/logit_bias still 400). MLX numerics gated (ADR 009).
- **e2e (stub):** `logprobs:true` ⇒ 200 with `choices[0].logprobs.content[]` of length =
  completion tokens, each with `token`/`logprob`/`top_logprobs`; `top_logprobs:21` ⇒ 400;
  `top_logprobs` without `logprobs` ⇒ 400; `n:2` still 400; streamed `logprobs:true` ⇒ a
  chunk carries `logprobs`.
- **Real-model RUNBOOK:** logprob *values* sane (≤ 0; chosen ≥ each non-chosen alt's
  logprob under greedy) on a real Qwen model AND a non-Qwen substrate model (the uniform-
  coverage claim).

## Honesty boundary

CI proves shape + the decision algebra; the numeric correctness of the captured logprobs
(real `log_softmax`) is MLX-gated and validated in the RUNBOOK tier, like every other MLX
numeric path (ADR 009).
