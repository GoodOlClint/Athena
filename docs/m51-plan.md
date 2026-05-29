# M51 — Symmetric Guide-mask in SpeculativeSamplingGenerate (M47.3)

Planning doc captured 2026-05-28. Implements the small C-1 item
from the deferred-items review — the M47.3 symmetric fix originally
deferred in M47's "out of scope" list.

## Status

Not started. The original deferral is documented in
[m47-plan.md](m47-plan.md):

> ## Out of scope (deferred follow-ups)
>
> The `samplingEligible` precondition relaxation that would let
> sampling-speculative engage under a Guide. Currently the Guide
> collapses sampling distributions to a single valid token (no
> real sampling), so this needs a thoughtful design pass before
> shipping.

This plan is NOT the precondition relaxation (that's M48-style
behavior widening — out of scope for M51). It IS the symmetric
mask fix in `SpeculativeSamplingGenerate.swift` mirroring what
M47.2 did in `SpeculativeGeneration.swift`, so that **when the
precondition does eventually relax**, the path is already correct.

## Trigger

M47.2 fixed three callsites in `SpeculativeGeneration.swift` where
the MTP draft was picked via unmasked argmax while the verify was
Guide-masked. Pre-fix, structured speculative under tight schemas
ran at ~0% acceptance. The fix replaced `argmaxLast(...)` with
`decoder.pick(... [0..., -1, 0...])` at the three draft callsites.

`SpeculativeSamplingGenerate.swift` has the structurally identical
shape: a draft is picked, a verify is picked, the verify-pick path
masks via the Guide and the draft-pick path doesn't. Today this is
inert because [MLXLLMModule.swift:489-490](
../Sources/AthenaLLM/MLXLLMModule.swift#L489-L490) gates sampling-
speculative to `schemaJSON == nil`. So the draft mismatch can't
fire — no live callsites engage.

But "inert today" is a brittle promise. When someone (us, a
contributor, or a future M-number) decides to relax the
precondition — most plausibly because M48.3 made the symmetric
case for greedy speculative clearly correct, and the analogous
relaxation for sampling-speculative comes up — the foot-gun fires
silently. A 30-line symmetric fix today prevents that.

## Goal

Make `SpeculativeSamplingGenerate.swift`'s draft picks Guide-aware
at the same three positions M47.2 fixed in
`SpeculativeGeneration.swift`. Behavior change is exactly zero
today (precondition blocks the engagement). Behavior change is
"correct by construction" if/when the precondition is later
relaxed.

## Implementation surface

### Sources/AthenaLLM/SpeculativeSamplingGenerate.swift

Three callsites mirror `SpeculativeGeneration.swift` (line numbers
verify before implementation — they may have drifted slightly with
M48-M49 work):

- Initial draft after prefill
- Post-accept next draft (the bonus-token branch)
- Post-reject next draft

Each currently picks via raw MLX sampling primitive. The fix routes
the draft through a `decoder.pick(...)`-equivalent — but because
the sampling path expects a *sampled* token (not argmax), the
analog is a Guide-masked sampling step:

```swift
// Current shape (rough):
let draft = sampleFromLogits(mtpForward(...), temperature, topP, seed)

// Symmetric shape:
let draft = sampleFromMaskedLogits(
    mtpForward(...), guide: guide, temperature, topP, seed)
```

The masked sampling primitive applies `guide.allowedMask(...)` to
the logits before sampling. The `guide` reference is the same one
the verify uses; M47.2's `decoder.pick` non-state-advancing
contract carries over (the Guide is only advanced via
`commit`, not via `pick` or `sample`).

If the existing sampling path doesn't have a `sampleFromMaskedLogits`
helper, add one — equivalent to `SpeculativeGeneration.guidedArgmax`
but for sampling instead of argmax. Could share the mask
materialization step.

### Pre-flight (matches M47's open question discipline)

Before writing code, verify:

1. The sampling path's RNG state IS request-local (no leaking
   across requests). It is — `var rng = SamplingRNG(seed: seed)`
   inside `SpeculativeSampling.generate`.
2. The Guide is non-state-advancing on a sampled token, same as
   on a picked token. It is — only `decoder.commit(_:)` advances
   the Guide state; `decoder.pick(_:)` / a hypothetical
   `decoder.sample(_:)` would not.
3. The `samplingEligible` precondition (currently
   `schemaJSON == nil`) is checked at the dispatch layer in
   `MLXLLMModule.runSpeculative`, not inside
   `SpeculativeSamplingGenerate.generate`. It is. So adding
   Guide-mask code to the inner function doesn't break the outer
   gate — the outer gate would have to be relaxed separately for
   the mask to engage.

## Tests

### M47.1 parity gate extended

The `StructuredSpeculativeParityTests` from M47.1 currently
exercises greedy speculative. Extend with a sampling-speculative
parity case — but because `samplingEligible` is gated to no-schema,
the test needs to drive the path directly:

```swift
// Heavy, gated on ATHENA_RUN_MODEL_TESTS=1.
// Construct a Guide + drive SpeculativeSampling.generate directly
// with a non-nil guide param. Bypasses MLXLLMModule's eligibility
// gate to exercise the masked-sampling code path even though it
// can't engage via the public API today.
let speculativeSampled = SpeculativeSampling.generate(
    model: model, promptTokens: prompt,
    maxTokens: 64, eosTokenId: eos,
    temperature: 0.7, topP: 0.95, seed: 42, guide: guide)

let greedyBaseline = GuidedGreedy.generate(
    model: model, promptTokens: prompt,
    maxTokens: 64, eosTokenId: eos, guide: guide)

// Under a Guide, sampling at temp > 0 with the mask collapses to
// the masked argmax sequence (single valid token per position).
// The two sequences must match byte-for-byte.
XCTAssertEqual(speculativeSampled, greedyBaseline)
```

This requires `SpeculativeSampling.generate` to accept a `guide:
StructuredGuide?` parameter (currently it doesn't — the path was
written as no-schema-only). That signature change is part of M51.

### CI-safe unit

The existing `StructuredGuideTests` already pin the
shared-index/independent-walker contract. M51 adds a test that the
Guide's `allowedMask(...)` produces a Boolean mask that, when
applied to logits, leaves the masked argmax bit-identical to the
unmasked argmax of the same logits with non-allowed positions
clamped to `-inf`. (This is the contract `SpeculativeGeneration.
guidedArgmax` relies on; M51 reuses that contract for sampling.)
Already implicit in M47 tests — extend if needed.

## Risks

| Risk | Mitigation |
|---|---|
| Behavior change on the existing temp>0 no-schema sampling path | The fix changes ONLY the schema-present-and-precondition-relaxed path. The current `schemaJSON == nil` gate stays put; existing no-schema sampling-speculative is untouched. |
| RNG drift across the masked-sampling code change | `SamplingRNG(seed:)` state is per-call; the mask change doesn't touch the RNG path. New test asserts deterministic output for fixed (seed, prompt). |
| The signature change (`guide: StructuredGuide?` on `SpeculativeSampling.generate`) breaks callers | Single caller is `MLXLLMModule.runSpeculative` — adjust both in the same commit. |
| Future precondition relaxation lands without re-verifying M51 | That relaxation must extend the M47.1-style parity test, which already covers the contract. |

## Sequencing

Single ship. ~30-50 lines of code, three callsite mirrors, one new
helper if needed (`sampleFromMaskedLogits`), one extended parity
test, one signature change.

## Definition of done for M51

- [ ] Three draft callsites in `SpeculativeSamplingGenerate.swift`
      route through a Guide-masked sampling helper when the guide
      is non-nil.
- [ ] `SpeculativeSampling.generate` accepts `guide:
      StructuredGuide?` and threads it through. Single caller
      updated to pass nil (preserving current behavior under the
      `schemaJSON == nil` gate).
- [ ] Extended `StructuredSpeculativeParityTests` asserts the
      sampling-speculative + Guide path produces byte-identical
      output to `GuidedGreedy` at the same prompt/schema (the
      precondition-relaxed contract).
- [ ] `MEMORY.md` updated. Closes the M47.3 "open" line.

## Out of scope (NOT in M51)

- Relaxing the `samplingEligible` precondition. That's an M48-style
  behavior widening and has the same template (the temp gate is
  inert under a Guide), but it's a separate decision that needs
  the consuming application's input on whether they'd benefit. Hold for a future
  M-number if the data calls for it.
- Sharing more code between `SpeculativeGeneration` and
  `SpeculativeSampling`. They diverge at the per-token primitive
  (argmax vs sample); a shared abstraction is achievable but not
  M51's scope.
- A `decoder.sample(_:)` member on `GuidedDecoder`. The mask
  materialization step can be reused (`guidedArgmax`'s buffer
  reuse pattern), but the sampling RNG state lives outside the
  decoder. Out of scope for M51.
