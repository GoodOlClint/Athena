# M47 — Guide-aware MTP drafts (working title)

Planning doc captured 2026-05-27 from the the consuming application sample-analysis
findings against v0.10.61 (and verified post-M46 against v0.10.66).
Not started. Save this doc as the resume point for a NEW chat session
when ready to ship.

## Status

In progress. M46 fully shipped (10 tags, v0.10.57 → v0.10.66).

| Slice | Tag | Date |
|---|---|---|
| M47.1 — bit-identical-greedy regression gate | v0.10.67 | 2026-05-27 |

- ✅ **M47.1** — v0.10.67 — `StructuredSpeculativeParityTests`
  (gated on `ATHENA_RUN_MODEL_TESTS=1`) drives a real MLX MTP model
  through both branches of `runSpeculative` under a tight enum
  schema (`speculative: true` then `false`), asserts the decoded
  strings match byte-for-byte. Pre-flight read of
  `GuidedDecoder.pick(_:)` confirmed it is non-state-advancing
  (only `commit(_:)` advances the Guide; `mutating` is solely for
  the reused `maskBuf`), so the M47.2 fix shape (use
  `decoder.pick` for the draft at the same Guide state the verify
  will use) is sound.

## Trigger

the consuming application E2E sampling (30s windows on the busy worker thread,
v0.10.61 unquantized 27B + 4B embedder, then re-verified on 8-bit):

| Sampled site (% of worker on-CPU) | GUIDE | PLAIN |
|---|---|---|
| Scheduler::wait_for_one (GPU cond_wait) | 59.5% | 61.4% |
| gpu::eval (CPU-side dispatch) | **0.8%** | **4.5%** |
| QuantizedMatmul::eval_gpu | **0.2%** | **1.0%** |
| runSpeculative active | 61% | 100% |

GUIDE drives **5× fewer GPU dispatches per wall-clock second** than
PLAIN despite spending the same fraction in cond_wait on the GPU.
With a 1.6 tok/s observed wall rate and a theoretical 23 tok/s
bandwidth floor on M5 Max (614 GB/s / 27 GB weight-pass = 44 ms ⇒
≈23 tok/s), the gap is ~12% of peak — and the symbol-table evidence
is consistent with structured speculative thrashing on draft
rejection, not memory pressure or Guide-side CPU overhead.

## Root cause — pinpointed

[Sources/AthenaLLM/SpeculativeGeneration.swift:138-143](../Sources/AthenaLLM/SpeculativeGeneration.swift#L138-L143)
spells it out in a comment:

```swift
// MTP drafts are NOT masked; a guide-invalid draft simply fails
// the masked verify and is rejected.
var draft = argmaxLast(
    model.mtpForward(
        hidden: hidden, nextTokenIds: tokenArray(prev),
        mtpCache: mtpCache) ?? logits)
```

The MTP head picks `argmax` of the **full** vocabulary distribution.
The verify at line 156 uses `decoder.pick(...)` which IS Guide-masked.
For tight JSON schemas (most positions have ≤10 allowed tokens out of
~150k), the MTP's argmax almost never matches the Guide-masked
verify → ~always reject → wasted MTP forward + KV trim + Mamba
rollback every iteration. Speculation actively costs MORE than it
saves under the Guide.

PLAIN doesn't have this problem because backbone and MTP pick from
the same unmasked distribution; they often agree.

## Goal

Recover speculative decoding's speedup under the Guide. Expected
wall-time impact: 2-3× on structured extraction-shape decode (drops
the consuming application's ~6 min extraction to ~2-3 min on 8-bit 27B).

## The fix (high level)

Mask the MTP draft to the **same Guide-allowed set** the verify will
use. The draft and verify are at the same token position (t+1 in the
verify-pass), so they share the same Guide state. With both picks
constrained to the valid set, draft/verify agreement rises from
~always-reject (under tight schemas) to the natural backbone/MTP
agreement rate within the valid set — which is high (~70-90% in
practice for unconstrained drafts; should be similar within a small
constrained set).

## Hard correctness constraint

Athena's M2-era bit-identical-greedy contract: speculative greedy
MUST produce the same token sequence as non-speculative greedy of
the same model + prompt + Guide state. **Masking the draft doesn't
change the contract**: the verify gate (`commit(draft)` only runs
when `draft == verifyPred`, AND `verifyPred` is the Guide-masked
argmax of the backbone) already enforces this. The draft mask only
affects *which token is proposed for verification*; what gets
committed is unchanged.

The bit-identical assertion needs a dedicated test (see Tests
section below) that runs the SAME prompt + schema with and without
speculative and asserts the token sequences match exactly.

## Implementation surface

### Sources/AthenaLLM/SpeculativeGeneration.swift

Three callsites to mask:

- **Line 140-143** — initial draft (before the while loop):

```swift
var draft = argmaxLast(model.mtpForward(...) ?? logits)
```
→ replace `argmaxLast` with a Guide-aware variant that takes the
GuidedDecoder's current state.

- **Line 167-170** — post-accept next draft (after `commit(bonus)`):
```swift
draft = argmaxLast(
    model.mtpForward(
        hidden: hiddenD, nextTokenIds: tokenArray(bonus),
        mtpCache: mtpCache) ?? logits2)
```
→ same fix; Guide state has advanced via commit(draft) +
commit(bonus).

- **Line 178-181** — post-reject next draft (after
  `commit(verifyPred)`):
```swift
draft = argmaxLast(
    model.mtpForward(
        hidden: hiddenC, nextTokenIds: tokenArray(verifyPred),
        mtpCache: mtpCache) ?? logits2)
```
→ same fix; Guide state has advanced via commit(verifyPred).

The existing `guidedArgmax` helper at lines 28-42 ALREADY exists and
takes the Guide. But it uses the Guide's `allowedMask` directly,
which is right BEFORE the commits above. To use the SAME Guide state
the verify will use, we need to either:

A. Use `GuidedDecoder.pick(_:)` (the same call the verify uses) for
   the draft, which encapsulates the Guide state lookup and the
   masking step. Cleanest.

B. Call `guidedArgmax(...)` directly with the current Guide state.
   Lower-level but explicit.

Recommend (A) — keeps the draft and verify on the same Guide-aware
pick path, so any future change to the masking behavior (e.g. the
deferred-thinking opener-alias machinery in `GuidedDecoder`)
applies to both consistently.

### Sources/AthenaLLM/SpeculativeSamplingGenerate.swift

Currently blocked from running with a Guide by precondition
(`samplingEligible` requires `schemaJSON == nil`). So the equivalent
draft fix here is for symmetry only — there are no live callsites
that hit Guide+sampling-speculative. Either:

A. Apply the symmetric fix anyway (no live behavior change, but the
   code stays consistent if/when the precondition is relaxed).

B. Skip and leave a TODO comment referencing M47.

Recommend (B) for slice scope; (A) is mechanical follow-up work.

### GuidedDecoder.pick semantics

Pre-flight: read `Sources/AthenaLLM/Generation/GuidedDecoder.swift`
(or wherever the type lives — verify path) to confirm:

- `decoder.pick(_:)` returns the Guide-masked argmax at the current
  decoder state (NOT after advancing the state).
- Calling `decoder.pick(_:)` multiple times at the same state
  returns the same token deterministically.
- The Guide state ONLY advances when `decoder.commit(_:)` is
  called.

If these invariants hold, the draft-mask fix is a drop-in. If any
break (e.g. `decoder.pick` is itself stateful), the fix needs a
different shape — possibly a parallel "peek without commit" entry
point on `GuidedDecoder`.

## Tests

### bit-identical-greedy contract (HARD requirement)

New test: same prompt + same JSON schema, run twice with the same
model:
1. `speculative: true` → emits a token sequence S₁
2. `speculative: false` → emits a token sequence S₂

Assert `S₁ == S₂` exactly. This MUST pass before merge — the M2-era
contract is load-bearing for every existing structured workflow.

Tests live in `Tests/AthenaCoreTests/`. The existing
`MLXLLMModuleTests.swift` is the natural home (it already has
greedy/speculative parity tests for unstructured prompts; this
extends to structured).

### acceptance-rate test (perf observability)

Add an opt-in counter inside `SpeculativeGeneration.generate` that
tracks accepted-draft vs total-iteration counts. Surface via a
TaskLocal (like `DecodeProgress.counter` in M46.8) so the test can
read it. Assert acceptance rate ≥ ~50% under a realistic JSON schema
(today it's effectively 0%).

### e2e wall-time before/after

`deploy/integration/e2e-extraction-perf.sh` (new) — runs a
canned extraction-shape request through `/v1/chat/completions` with
`response_format` + `speculative: true` on a real MLX engine, asserts
the wall time falls within a budget. Manual host-bound tier (NOT
CI), per the integration-test-topology memory. Baseline budget set
from a v0.10.66 measurement so future regressions are caught.

## Risks

| Risk | Mitigation |
|---|---|
| Bit-identical-greedy contract regression | New dedicated test asserting greedy-with-speculative == greedy-without-speculative under structured output. Block-on-fail. |
| `GuidedDecoder.pick` is stateful | Pre-flight read of the type before coding (see "GuidedDecoder.pick semantics" above). Adjust approach if needed. |
| MTP head outputs are differently-scaled vs backbone | The mask is additive `-inf` on disallowed tokens; argmax doesn't care about scale. Verified by inspection. |
| Other unmasked draft sites I missed | Grep `argmaxLast` across the file before claiming done. There are no other expected sites today, but worth a final sweep. |
| Acceptance rate stays low for other reasons | Some schemas may legitimately have low MTP/backbone agreement (e.g. very rare token sequences). The acceptance-rate test should set a realistic floor (~30-50%) rather than insist on a high number. |

## Sequencing within M47

1. **M47.1** — read `GuidedDecoder.pick` semantics, write the bit-
   identical-greedy test against current behavior (should pass on
   current code). Sets the regression gate.
2. **M47.2** — implement the draft mask at all 3 callsites in
   `SpeculativeGeneration.swift`. Acceptance-rate counter + test.
   Bit-identical-greedy test still passes.
3. **M47.3** (optional) — symmetric fix in
   `SpeculativeSamplingGenerate.swift`. Even though it's
   precondition-blocked today, keep the code shape consistent.
4. **M47.4** — manual host-bound e2e wall-time measurement on a real
   MLX engine via `deploy/integration/e2e-extraction-perf.sh`.
   Compare to pre-M47 baseline.

## Definition of done for M47

- [ ] Bit-identical-greedy assertion holds across structured
      speculative vs non-speculative.
- [ ] Acceptance-rate ≥ 30% under a realistic JSON schema (the
      "Guide effectively zero-accept" failure mode is gone).
- [ ] e2e-rbac.sh stays green (no stub-engine regressions; the
      stub doesn't exercise speculative, so this is just
      no-breakage).
- [ ] Manual host-bound run on 8-bit 27B + JSON-schema extraction
      shows 2-3× wall-time improvement vs v0.10.66.
- [ ] `MEMORY.md` updated with a `project_m47-guide-aware-drafts.md`
      entry.

## Out of scope (deferred follow-ups)

- A real `generationTokensPerSec` rolling window in `/healthz`
  (the M46.5 deferral). M47 makes this more meaningful since
  acceptance-rate-driven throughput becomes the primary perf
  signal; revisit after M47.
- The `samplingEligible` precondition relaxation that would let
  sampling-speculative engage under a Guide. Currently the Guide
  collapses sampling distributions to a single valid token (no
  real sampling), so this needs a thoughtful design pass before
  shipping.
- A `--no-speculative` daemon default for structured workloads.
  Operator can already pass `speculative: false` per-request
  (M46.3a); a daemon flag would be cosmetic after M47 lands.

## Open question (pre-flight for the new session)

**Where exactly does `GuidedDecoder.pick(_:)` advance state?** —
RESOLVED (M47.1, 2026-05-27): `pick` is non-state-advancing. Only
`commit(_:)` calls `guide.advance(...)` / `advanceOpenerTolerant(...)`;
`pick` reads the current `(guide, enforcing)` state and returns the
Guide-masked argmax (or plain argmax when not enforcing). The
`mutating` keyword on `pick` is solely because it reuses the
`maskBuf` byte buffer between calls — no Guide-state mutation. So
calling `decoder.pick(...)` for the draft at the same logical
position the verify will use is safe and produces the same token
deterministically. The M47.2 fix shape (replace `argmaxLast(...)`
with `decoder.pick(... [0..., -1, 0...])`) is the drop-in.
